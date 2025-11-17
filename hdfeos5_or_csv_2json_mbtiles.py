#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Alfredo Terrero, 2016                            #
############################################################


import os
import re
import sys
import argparse
import pickle
import json
import time
from datetime import date
import math
import geocoder
import numpy as np
from pathlib import Path

from mintpy.objects import HDFEOS
from mintpy.mask import mask_matrix
from mintpy.utils import utils as ut
import h5py
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing import Pool
from multiprocessing import Value

import pandas as pd
from datetime import datetime
from shapely.geometry import MultiPoint

chunk_num = Value("i", 0)
# ex: python Converter_unavco.py Alos_SM_73_2980_2990_20070107_20110420.h5

# This script takes a UNAVCO format timeseries h5 file, converts to mbtiles,
# and sends to database which allows website to make queries and display data
# ---------------------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------------------
# returns a dictionary of datasets that are stored in memory to speed up h5 read process
def get_date(date_string):
    year = int(date_string[0:4])
    month = int(date_string[4:6])
    day = int(date_string[6:8])
    return date(year, month, day)


# ---------------------------------------------------------------------------------------
# takes a date and calculates the number of days elapsed in the year of that date
# returns year + (days_elapsed / 365), a decimal representation of the date necessary
# for calculating linear regression of displacement vs time
def get_decimal_date(d):
    start = date(d.year, 1, 1)
    return abs(d-start).days / 365.0 + d.year

def region_name_from_project_name(project_name):
    track_index = project_name.find('T')

    return project_name[:track_index]

needed_attributes = {
    "prf", "first_date", "mission", "WIDTH", "X_STEP", "processing_software",
    "wavelength", "processing_type", "beam_swath", "Y_FIRST", "look_direction",
    "flight_direction", "last_frame", "post_processing_method", "min_baseline_perp",
    "unwrap_method", "relative_orbit", "beam_mode", "LENGTH", "max_baseline_perp",
    "X_FIRST", "atmos_correct_method", "last_date", "first_frame", "frame", "Y_STEP", "history",
    "scene_footprint", "data_footprint", "downloadUnavcoUrl", "referencePdfUrl", "areaName", "referenceText",
    "REF_LAT", "REF_LON", "CENTER_LINE_UTC", "insarmaps_download_flag", "mintpy.subset.lalo"
}

def serialize_dictionary(dictionary, fileName):
    """Serialize a dictionary to a pickle file."""

    with open(fileName, "wb") as file:
        pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)
    return

def get_attribute_or_remove_from_needed(needed_attributes, attributes, attribute_name):
    val = None

    try:
        val = attributes[attribute_name]
    except:
        needed_attributes.remove(attribute_name)

    return val

def generate_worker_args(decimal_dates, timeseries_datasets, dates, json_path, folder_name, chunk_size, lats, lons, num_columns, num_rows, quality_params=None):
    """
    Build argument tuples for worker processes that create JSON chunks.

    Each tuple contains:
    (
        decimal_dates,
        timeseries_datasets,
        dates,
        json_path,
        folder_name,
        (start_index, end_index),  # inclusive indices in flattened grid
        num_columns,
        num_rows,
        lats,
        lons,
        quality_params,
    )
    """
    num_points = num_columns * num_rows

    worker_args = []
    start = 0
    end = 0
    idx = 0
    for i in range(num_points // chunk_size):
        start = idx * chunk_size
        end = (idx + 1) * chunk_size
        if end > num_points:
            end = num_points
        args = [decimal_dates, timeseries_datasets, dates, json_path, folder_name, (start, end - 1), num_columns, num_rows, lats, lons, quality_params]
        worker_args.append(tuple(args))
        idx += 1

    if num_points % chunk_size != 0:
        start = end
        end = num_points
        args = [decimal_dates, timeseries_datasets, dates, json_path, folder_name, (start, end - 1), num_columns, num_rows, lats, lons, quality_params]
        worker_args.append(tuple(args))

    return worker_args

def create_json(decimal_dates, timeseries_datasets, dates, json_path, folder_name, work_idxs, num_columns, num_rows, lats=None, lons=None, quality_params=None):
    """
    Create GeoJSON point features for a subset of the grid defined by work_idxs
    and write them to a chunk_<N>.json file.
    """
    
    global chunk_num

    # List to store GeoJSON Feature objects for this chunk
    # create a siu_man array to store json point objects
    siu_man = []
    # Decimal dates as a numpy array (x in regression)
    x_arr = np.asarray(decimal_dates, dtype=float)

    start_idx, end_idx = work_idxs
    point_num = start_idx

    # Use the first date slice to determine which points exist in this chunk
    first_slice = timeseries_datasets[dates[0]]

    for (row, col), first_value in np.ndenumerate(first_slice):
        cur_iter_point_num = row * num_columns + col

        if cur_iter_point_num < start_idx:
            continue
        if cur_iter_point_num > end_idx:
            break

        displacement0 = float(first_value)

        # Skip points where the first date value is NaN
        if math.isnan(displacement0):
            continue

        longitude = float(lons[row][col])
        latitude = float(lats[row][col])

        # Collect displacement values across all dates for this point
        displacement_values = []
        for datei in dates:
            val = timeseries_datasets[datei][row][col]
            if not math.isnan(val):
                displacement_values.append(float(val))
            else:
                # Use None to produce JSON null for missing values
                displacement_values.append(None)

        # Build x/y arrays only where y is present
        mask = np.array([v is not None for v in displacement_values], dtype=bool)
        x_used = x_arr[mask]
        y = np.asarray(displacement_values, dtype=float)[mask]

        # Linear regression: y = m*x + c, only if we have at least 2 valid points
        if y.size >= 2:
            A = np.vstack([x_used, np.ones(x_used.size)]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        else:
            m = float("nan")

        # Base properties for all inputs
        safe_properties = {"d": displacement_values, "m": m, "p": point_num}

        # Add quality parameters at this location, if provided
        if quality_params:
            for key, arr in quality_params.items():
                val = arr[row][col]

                # For point_ID: include only if present (no None) and finite; cast to int
                if key == "point_ID":
                    if (
                        val is None
                        or (
                            isinstance(val, float)
                            and (math.isnan(val) or not math.isfinite(val))
                        )
                    ):
                        continue
                    safe_properties[key] = int(val)
                    continue

                # Existing behavior for other quality fields
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    safe_properties[key] = None
                else:
                    safe_properties[key] = val

        # Build GeoJSON feature
        data = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
            "properties": safe_properties,
        }

        siu_man.append(data)
        point_num += 1

    # Write out chunk if any features were created
    if len(siu_man) > 0:
        with chunk_num.get_lock():
            chunk_num_val = chunk_num.value
            chunk_num.value += 1

        make_json_file(chunk_num_val, siu_man, dates, json_path, folder_name)

# ---------------------------------------------------------------------------------------
# convert h5 file to json and upload it. folder_name == unavco_name
def convert_data(attributes, decimal_dates, timeseries_datasets, dates, json_path, folder_name, lats=None, lons=None, quality_params=None, num_workers=1):

    project_name = attributes["PROJECT_NAME"]
    region = region_name_from_project_name(project_name)

    # Get attributes for calculating latitude and longitude (legacy; midpoint now uses lats/lons)
    x_step = y_step = x_first = y_first = 0.0
    if high_res_mode(attributes):
        # In high-res mode, these keys are not meaningful; remove them from needed_attributes
        for key in ("X_STEP", "Y_STEP", "X_FIRST", "Y_FIRST"):
            if key in needed_attributes:
                needed_attributes.remove(key)
    else:
        x_step = float(attributes["X_STEP"])
        y_step = float(attributes["Y_STEP"])
        x_first = float(attributes["X_FIRST"])
        y_first = float(attributes["Y_FIRST"])

    num_columns = int(attributes["WIDTH"])
    num_rows = int(attributes["LENGTH"])
    print(f"columns: {num_columns}")
    print(f"rows: {num_rows}")

    # If lats/lons are not provided (HDFEOS5), compute them from attributes
    if lats is None and lons is None:
        lats, lons = ut.get_lat_lon(attributes, dimension=1)

    # ----------------------------------------------------------------------
    # Create JSON chunks in parallel
    # ----------------------------------------------------------------------
    CHUNK_SIZE = 20000

    worker_args = generate_worker_args(decimal_dates, timeseries_datasets, dates, json_path, folder_name, CHUNK_SIZE, lats, lons, num_columns, num_rows, quality_params)

    process_pool = Pool(num_workers)
    process_pool.starmap(create_json, worker_args)
    process_pool.close()

    # dictionary to contain metadata needed by db to be written to a file
    # and then be read by json_mbtiles2insarmaps.py
    # calculate mid lat and long of dataset - then use google python lib to get country
    # technically don't need the else since we always use lats and lons arrays now
    # ----------------------------------------------------------------------
    # Build metadata for Insarmaps
    # ----------------------------------------------------------------------
    insarmapsMetadata = {}

    # Midpoint: use mean of lat/lon arrays (works for both HDFEOS5 and CSV)
    mid_lat = float(np.nanmean(lats))
    mid_long = float(np.nanmean(lons))

    # Reverse geocode to get country name
    country = "None"
    try:
        g = geocoder.google([mid_lat, mid_long], method="reverse", timeout=60.0)
        country = str(g.country_long)
    except Exception:
        sys.stderr.write("timeout reverse geocoding country name")

    area = folder_name

    # Postgres-style date arrays: {d1,d2,...}
    string_dates_sql = "{" + ",".join(str(k) for k in dates) + "}"
    decimal_dates_sql = "{" + ",".join(str(d) for d in decimal_dates) + "}"

    # Add keys and values to area table.
    attribute_keys = []
    attribute_values = []
    max_digit = max([len(key) for key in list(needed_attributes)] + [0])

    for k, v in attributes.items():
        if k in needed_attributes:
            # Print in aligned format
            print(f"{k:<{max_digit}}     {v}")
            attribute_keys.append(k)
            attribute_values.append(v)

    # Force 'elevation' to show up in popup
    attribute_keys.append("dem")
    attribute_values.append("1")

    # Populate metadata dictionary
    insarmapsMetadata["area"] = area
    insarmapsMetadata["project_name"] = project_name
    insarmapsMetadata["mid_long"] = mid_long
    insarmapsMetadata["mid_lat"] = mid_lat
    insarmapsMetadata["country"] = country
    insarmapsMetadata["region"] = region
    insarmapsMetadata["chunk_num"] = 1
    insarmapsMetadata["attribute_keys"] = attribute_keys
    insarmapsMetadata["attribute_values"] = attribute_values
    insarmapsMetadata["string_dates_sql"] = string_dates_sql
    insarmapsMetadata["decimal_dates_sql"] = decimal_dates_sql
    insarmapsMetadata["attributes"] = attributes
    insarmapsMetadata["needed_attributes"] = needed_attributes

    metadataFilePath = json_path + "/metadata.pickle"
    serialize_dictionary(insarmapsMetadata, metadataFilePath)
    return

# ---------------------------------------------------------------------------------------
# create a json file out of siu man array
# then put json file into directory named after the h5 file
def make_json_file(chunk_num, points, dates, json_path, folder_name):

    chunk = "chunk_" + str(chunk_num) + ".json"
    json_file = open(json_path + "/" + chunk, "w")
    json_features = [json.dumps(feature) for feature in points]
    string_json = '\n'.join(json_features)
    json_file.write("%s" % string_json)
    json_file.close()

    print("converted chunk " + str(chunk_num))
    return chunk

def high_res_mode(attributes):
    high_res = False # default
    try:
        x_step = attributes["X_STEP"]
        y_step = attributes["Y_STEP"]
    except:
        high_res = True # one or both not there, so we are high res

    return high_res

# ---------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser for hdfeos5_or_csv_2json_mbtiles.py.
    """
    example_text = """\
This program will create temporary JSON chunk files which, when concatenated
together, comprise the whole dataset. Tippecanoe is used to convert these
JSON chunks into an MBTiles file.

Examples:
  hdfeos5_or_csv_2json_mbtiles.py mintpy/S1_IW1_128_0596_0597_20160605_XXXXXXXX_S00887_S00783_W091208_W091105.he5 mintpy/JSON --num-workers 3
  hdfeos5_or_csv_2json_mbtiles.py sarvey_test.csv ./JSON
  hdfeos5_or_csv_2json_mbtiles.py NOAA_SNT_A_VERT_10_50m.csv ./JSON_NOAA
    """

    parser = argparse.ArgumentParser(
        description="Convert an HDFEOS5 (.he5) or CSV file for ingestion into Insarmaps.",
        epilog=example_text,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    parser.add_argument(
        "--num-workers",
        help="Number of simultaneous processes to run for JSON creation.",
        required=False,
        default=1,
        type=int,
    )

    required = parser.add_argument_group("required arguments")

    required.add_argument(
        "file",
        help="(unavco) Input file to ingest (.he5 or .csv).",
    )

    required.add_argument(
        "outputDir",
        help="Directory to place JSON chunk files and MBTiles output.",
    )

    return parser

def add_dummy_attribute(attributes: dict, is_sarvey_format: bool) -> None:
    """
    Fill in missing attributes with hard-coded defaults to satisfy Insarmaps
    expectations, mainly for CSV imports.

    Notes
    -----
    - This is a temporary compatibility layer until needed_attributes is reduced
      or fully driven by real metadata.
    - For SARvey-style CSV (is_sarvey_format=True), we only set processing-related
      defaults. For other formats, we also set REF_LAT / REF_LON, data_footprint,
      first_date, and last_date if they are missing.
    """

    # Processing / acquisition defaults
    attributes.setdefault("atmos_correct_method", None)
    attributes.setdefault("beam_mode", "IW")
    attributes.setdefault("beam_swath", 1)
    attributes.setdefault("post_processing_method", "MintPy")
    attributes.setdefault("prf", 1717.128973878037)
    attributes.setdefault("processing_software", "isce")
    attributes.setdefault(
                        "scene_footprint",
                        "POLYGON((-90.79583946164999 -0.687890034792316,"
                                "-90.86911230465793 -1.0359825079903804,"
                                "-91.62407871076888 -0.8729106902243329,"
                                "-91.55064943686261 -0.5251520401739668,"
                                "-90.79583946164999 -0.687890034792316))",
                        )
    attributes.setdefault("wavelength", 0.05546576)
    attributes.setdefault("first_frame", 556)
    attributes.setdefault("last_frame", 557)

    # For non-SARvey formats, provide hard-coded spatial/time defaults if they are missing.
    if not is_sarvey_format:
        attributes.setdefault("REF_LAT", -0.83355445)
        attributes.setdefault("REF_LON", -91.12596)
        attributes.setdefault(
                            "data_footprint",
                            "POLYGON((-91.19760131835938 -0.7949774265289307,"
                                    "-91.11847686767578 -0.7949774265289307,"
                                    "-91.11847686767578 -0.8754903078079224,"
                                    "-91.19760131835938 -0.8754903078079224,"
                                    "-91.19760131835938 -0.7949774265289307))",
                            )
        attributes.setdefault("first_date", "2016-06-05")
        attributes.setdefault("last_date", "2016-08-28")

def add_data_footprint_attribute(attributes: dict, lats: np.ndarray, lons: np.ndarray) -> None:
    """
    Add/update 'data_footprint' (and 'scene_footprint') in attributes based
    on the min/max of the latitude/longitude arrays.

    Parameters
    ----------
    attributes : dict
        Attribute dictionary to update in-place.
    lats : 2D np.ndarray
        Latitude grid.
    lons : 2D np.ndarray
        Longitude grid.
    """

    min_lat = float(np.min(lats))
    max_lat = float(np.max(lats))
    min_lon = float(np.min(lons))
    max_lon = float(np.max(lons))

    # Counter-clockwise polygon around bounding box, starting and ending at lower-right corner.
    polygon = (
        f"POLYGON(({max_lon} {min_lat},"
        f"{min_lon} {min_lat},"
        f"{min_lon} {max_lat},"
        f"{max_lon} {max_lat},"
        f"{max_lon} {min_lat}))"
    )
    print("data_footprint: ", polygon)

    attributes['data_footprint'] = polygon
    attributes["scene_footprint"] = polygon

def read_from_hdfeos5_file(file_name):
    # read data from hdfeos5 file
    should_mask = True

    path_name_and_extension = os.path.basename(file_name).split(".")
    path_name = path_name_and_extension[0]

    # use h5py to open specified group(s) in the h5 file
    # then read datasets from h5 file into memory for faster reading of data
    he_obj = HDFEOS(file_name)
    he_obj.open(print_msg=False)
    displacement_3d_matrix = he_obj.read(datasetName='displacement')
    mask = he_obj.read(datasetName='mask')
    if should_mask:
        print("Masking displacement")
        displacement_3d_matrix = mask_matrix(displacement_3d_matrix, mask)
    del mask

    print("Creating shared memory for multiple processes")
    shm = shared_memory.SharedMemory(create=True, size=displacement_3d_matrix.nbytes)
    shared_displacement_3d_matrix = np.ndarray(displacement_3d_matrix.shape, dtype=displacement_3d_matrix.dtype, buffer=shm.buf)
    shared_displacement_3d_matrix[:] = displacement_3d_matrix[:]
    del displacement_3d_matrix
    displacement_3d_matrix = shared_displacement_3d_matrix

    dates = he_obj.dateList
    attributes = dict(he_obj.metadata)

    decimal_dates = []

    # read datasets in the group into a dictionary of 2d arrays and intialize decimal dates
    timeseries_datasets = {}
    num_date = len(dates)
    for i in range(num_date):
        timeseries_datasets[dates[i]] = np.squeeze(displacement_3d_matrix[i, :, :])
        d = get_date(dates[i])
        decimal = get_decimal_date(d)
        decimal_dates.append(decimal)
    del displacement_3d_matrix

    path_list = path_name.split("/")
    folder_name = path_name.split("/")[len(path_list)-1]

    # read lat and long. MintPy doesn't seem to support this yet, so we use the raw h5 file object
    f = h5py.File(he_obj.file, "r")
    lats = np.array(f["HDFEOS"]["GRIDS"]["timeseries"]["geometry"]["latitude"])
    lons = np.array(f["HDFEOS"]["GRIDS"]["timeseries"]["geometry"]["longitude"])

    attributes["collection"] = "hdfeos5"
    return attributes, decimal_dates, timeseries_datasets, dates, folder_name, lats, lons, shm

def add_calculated_attributes(attributes: dict) -> None:
    """
    Calculate and add attributes that can be inferred from CSV data.

    Uses:
    - LAT_ARRAY, LON_ARRAY: to compute REF_LAT and REF_LON (mean positions).
    - DATE_COLUMNS: to compute first_date, last_date, and history.

    These temporary keys are removed from the attributes dict.
    """

    # calculate attributes from lat/lon and date columns (csv):
    # REF_LAT, REF_LON, first_date, last_date, history

    required_keys = ("LAT_ARRAY", "LON_ARRAY", "DATE_COLUMNS")
    if all(key in attributes for key in required_keys):
        lats = attributes.pop("LAT_ARRAY")
        lons = attributes.pop("LON_ARRAY")
        date_columns = attributes.pop("DATE_COLUMNS")

        attributes["REF_LAT"] = float(np.nanmean(lats))
        attributes["REF_LON"] = float(np.nanmean(lons))

        sorted_dates = sorted(date_columns)
        attributes["first_date"] = sorted_dates[0]
        attributes["last_date"] = sorted_dates[-1]
        attributes["history"] = datetime.now().strftime("%Y-%m-%d")

def enrich_attributes_from_slcstack(attributes: dict, csv_path) -> dict:
    """
    Enrich attributes (mission, platform, beam_mode, look_direction, flight_direction,
    relative_orbit) using metadata from slcStack.h5 and/or the standardized CSV filename.

    Parameters
    ----------
    attributes : dict
        Attribute dictionary to update in-place.
    csv_path : str or Path
        Path to the CSV file (used to locate ../inputs/slcStack.h5 and to
        parse mission/orbit info from the filename).

    Returns
    -------
    dict
        The updated attributes dictionary.
    """

    try:
        csv_path = Path(csv_path).resolve()
        slc_path = csv_path.parent.parent / "inputs" / "slcStack.h5"

        # Primary source: metadata from slcStack.h5 (if available)
        if slc_path.exists():
            with h5py.File(slc_path, "r") as f:

                def _get(attr_name, default=None):
                    val = f.attrs.get(attr_name, default)
                    if isinstance(val, (bytes, bytearray)):
                        val = val.decode("utf-8", "ignore")
                    return val

                # Mission / platform
                mission = _get("mission") or _get("MISSION")
                platform = _get("platform") or _get("PLATFORM")
                mission_value = mission or platform
                if mission_value:
                    attributes["mission"] = mission_value
                    # Keep explicit platform too if available
                    if platform:
                        attributes["platform"] = platform

                # Beam mode
                bm = _get("beam_mode")
                if bm:
                    attributes["beam_mode"] = bm

                # Look / flight direction
                look = _get("look_direction")
                if look:
                    attributes["look_direction"] = look
                fd = _get("flight_direction")
                if fd:
                    attributes["flight_direction"] = fd

                # Relative orbit: accept several spellings
                rel_keys = ["relative_orbit", "orbit", "relativeOrbit", "relativeOrbitNumber"]
                rel_val = None
                for rk in rel_keys:
                    v = _get(rk)
                    if v is not None:
                        rel_val = v
                        break
                if rel_val is not None:
                    try:
                        attributes["relative_orbit"] = int(rel_val)
                    except Exception:
                        # leave as-is if non-integer
                        attributes["relative_orbit"] = rel_val
        else:
            print(f"[INFO] slcStack.h5 not found at: {slc_path}")

        # ------------------------------------------------------------------
        # Fallback 2: infer from standardized CSV filename if still missing or only generic defaults are set.
        # ------------------------------------------------------------------
        stem = Path(csv_path).stem.upper()
        m = re.match(r"^(S1|TSX|ALOS|ERS|ENVISAT)_(\d{3})_", stem)
        if m:
            file_mission = m.group(1)  # e.g., "TSX"
            file_rel = int(m.group(2))

            # If mission was never set or is still the generic default, override it
            if not attributes.get("mission") or attributes.get("mission", "").upper() == "S1":
                # Keep the existing title-case behavior (e.g., "TSX" -> "Tsx")
                attributes["mission"] = file_mission.title()

            # Fill relative orbit if missing
            attributes.setdefault("relative_orbit", file_rel)

            # Sensible default for TSX
            if attributes.get("mission", "").upper() == "TSX":
                attributes.setdefault("beam_mode", "SM")

    except Exception as e:
        print(f"[WARN] Could not read slcStack.h5 metadata: {e}")

    return attributes

def read_from_csv_file(file_name):
    """
    Read a SARvey/NOAA-style CSV and convert it into the structures expected by
    hdfeos5_or_csv_2json_mbtiles.py.

    Returns
    -------
    attributes : dict
    decimal_dates : list[float]
    timeseries_datasets : dict[str, np.ndarray]  # date -> (rows, cols)
    dates : list[str]
    folder_name : str
    lats_grid : np.ndarray  # (rows, cols)
    lons_grid : np.ndarray  # (rows, cols)
    shm : None              # kept for symmetry with HDFEOS5 reader
    quality_grids : dict[str, np.ndarray]  # name -> (rows, cols)
    """
    
    # read data from csv file, done by Emirhan
    # the shared memory shm is confusing. it may also works without but be careful about returning or not returning shm.

    df = pd.read_csv(file_name)

    # Normalize headers (trim spaces)
    df.columns = [c.strip() for c in df.columns]

    # ----------------------------------------------------------------------
    # Detect point_id (any case) and normalize to a new column 'point_ID'
    # ----------------------------------------------------------------------
    pid_col = next((c for c in df.columns if c.lower() == "point_id"), None)
    if pid_col is not None:
        df["point_ID"] = pd.to_numeric(df[pid_col], errors="coerce")  # float with NaN is fine
        print(f"[INFO] Detected point_ID column: {pid_col} (non-null={df['point_ID'].notna().sum()})")

    # ----------------------------------------------------------------------
    # Dynamically detect latitude/longitude columns
    # ----------------------------------------------------------------------
    lat_candidates = ["Y_geocorr", "Latitude", "Y", "ycoord"]
    lon_candidates = ["X_geocorr", "Longitude", "X", "xcoord"]

    lat_col = next((col for col in lat_candidates if col in df.columns), None)
    lon_col = next((col for col in lon_candidates if col in df.columns), None)

    print(f"Using columns: lat = {lat_col}, lon = {lon_col}")

    if lat_col is None or lon_col is None:
        raise ValueError(
            "Could not find latitude/longitude columns in the CSV. Supported names: "
            "'Latitude', 'Y', 'ycoord' and 'Longitude', 'X', 'xcoord'."
        )

    lats = df[lat_col].values
    lons = df[lon_col].values


    # ----------------------------------------------------------------------
    # Extract time-series data
    # ----------------------------------------------------------------------
    sarvey_time_cols = [col for col in df.columns if col.startswith("D") and col[1:].isdigit()]
    is_sarvey_format = bool(sarvey_time_cols)

    if is_sarvey_format:
        time_cols = sorted(sarvey_time_cols)
        timeseries_data = df[time_cols].values / 1000.0  # SARvey: mm -> m
        dates = [col[1:] for col in time_cols]           # remove "D" prefix
    else:
        time_cols = [col for col in df.columns if col.isdigit()]
        timeseries_data = df[time_cols].values / 1000.0  # e.g. NOAA-TRE: mm -> m
        dates = time_cols

    # ----------------------------------------------------------------------
    # Build 3D time-series array (time, y, x)
    # ----------------------------------------------------------------------
    num_points = len(df)
    num_dates = len(time_cols)

    # Reshape to a nearly square grid, padding with NaN
    num_rows = int(np.sqrt(num_points))
    num_cols = int(np.ceil(num_points / num_rows))

    padded = np.full((num_cols * num_rows, num_dates), np.nan)
    padded[:num_points, :] = timeseries_data
    reshaped = padded.reshape((num_rows, num_cols, num_dates)).transpose(2, 0, 1)

    # Decimal dates and per-date 2D arrays
    decimal_dates = [get_decimal_date(get_date(d)) for d in dates]
    timeseries_datasets = {d: reshaped[i, :, :] for i, d in enumerate(dates)}

    # ----------------------------------------------------------------------
    # Create initial attributes
    # ----------------------------------------------------------------------
    attributes = {}
    attributes["PROJECT_NAME"] = "CSV_IMPORT"
    attributes["WIDTH"] = str(num_cols)
    attributes["LENGTH"] = str(num_rows)
    attributes["LAT_ARRAY"] = lats
    attributes["LON_ARRAY"] = lons
    attributes["DATE_COLUMNS"] = dates
    attributes["processing_type"] = "LOS_TIMESERIES"
    attributes.setdefault("look_direction", "R")
    attributes["collection"] = "sarvey"

    # Automatically set data_type based on filename or vertical velocity columns
    filename_upper = Path(file_name).stem.upper()
    vert_col_candidates = ["VEL_V", "V_STDEV_V"]
    has_vertical_columns = any(col in df.columns for col in vert_col_candidates)

    if "VERT" in filename_upper or has_vertical_columns:
        attributes["data_type"] = "Vertical Displacement"
    else:
        attributes["data_type"] = "LOS Displacement"

    print(f"[INFO] Set data_type: {attributes['data_type']}")

    # replaced hard-coded lines with setdefault
    attributes.setdefault("PLATFORM", "S1")
    attributes.setdefault("MISSION", "S1")

    # Enrich using slcStack.h5 and filename conventions
    attributes = enrich_attributes_from_slcstack(attributes, file_name)

    # Normalize mission/platform for InsarMaps (expects lower-case)
    # (current behavior: just fill from MISSION/PLATFORM if mission/platform missing)
    if attributes.get("mission") in (None, "None", "", "null"):
        if attributes.get("MISSION"):
            attributes["mission"] = attributes["MISSION"]

    if attributes.get("platform") in (None, "None", "", "null"):
        if attributes.get("PLATFORM"):
            attributes["platform"] = attributes["PLATFORM"]

    # Make relative_orbit safe (turn into int if possible)
    if "relative_orbit" in attributes:
        try:
            attributes["relative_orbit"] = int(attributes["relative_orbit"])
        except Exception:
            # leave as-is if it cannot be converted
            pass

    # Calculate inferred attributes and geometry-related attributes
    add_calculated_attributes(attributes)
    add_data_footprint_attribute(attributes, lats, lons)
    add_dummy_attribute(attributes, is_sarvey_format)  # Remove once needed_attributes reduced    

    # ----------------------------------------------------------------------
    # Build lat/lon grids (2D) matching timeseries layout
    # ----------------------------------------------------------------------
    padded_lats = np.full(num_cols * num_rows, np.nan)
    padded_lats[:num_points] = lats
    lats_grid = padded_lats.reshape((num_rows, num_cols))

    padded_lons = np.full(num_cols * num_rows, np.nan)
    padded_lons[:num_points] = lons
    lons_grid = padded_lons.reshape((num_rows, num_cols))

    # ----------------------------------------------------------------------
    # Quality fields (DEM error, coherence, omega, st_consist, point_ID, ...)
    # ----------------------------------------------------------------------
    quality_fields = {}

    if "dem_error" in df.columns and "dem" in df.columns:
        quality_fields["dem_error"] = df["dem_error"].values
        quality_fields["elevation"] = df["dem_error"].values - df["dem"].values
    if "coherence" in df.columns:
        quality_fields["coherence"] = df["coherence"].values
    if "omega" in df.columns:
        quality_fields["omega"] = df["omega"].values
    if "st_consist" in df.columns:
        quality_fields["st_consist"] = df["st_consist"].values

    # Add point_ID if present (no default / None creation)
    if "point_ID" in df.columns:
        # keeping as 1-D vector; NaNs allowed
        quality_fields["point_ID"] = df["point_ID"].to_numpy(dtype=float)

    # Build 2D grids for all quality fields, padded like the time series
    quality_grids = {key: np.full(num_rows * num_cols, np.nan) for key in quality_fields}
    for key, values in quality_fields.items():
        quality_grids[key][:num_points] = values
        quality_grids[key] = quality_grids[key].reshape((num_rows, num_cols))

    # ----------------------------------------------------------------------
    # Final outputs
    # ----------------------------------------------------------------------
    folder_name = os.path.basename(file_name).split(".")[0]
    shm = None  # CSV path does not use shared memory

    print(" Check: read_from_csv_file output")
    print(" - timeseries_datasets keys:", list(timeseries_datasets.keys())[:3], "...")
    print(" - sample slice shape:", timeseries_datasets[dates[0]].shape)
    print(" - lat_grid shape:", lats_grid.shape)
    print(" - lon_grid shape:", lons_grid.shape)
    print(" - Number of dates:", len(dates))
    print(" - Attributes:", attributes)

    return (attributes, decimal_dates, timeseries_datasets, dates, folder_name, lats_grid, lons_grid, shm, quality_grids)

# ---------------------------------------------------------------------------------------
# START OF EXECUTABLE
# ---------------------------------------------------------------------------------------
def main():
    """Entry point for hdfeos5_or_csv_2json_mbtiles.py."""
    parser = build_parser()
    args = parser.parse_args()
    file_name = args.file
    output_folder = args.outputDir

    # Create path for output (or note if it already exists)
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        print(output_folder + " already exists")

    file_path = Path(file_name)

    # start clock to track how long conversion process takes
    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Read input: either HDFEOS5 (.he5) or CSV
    # ------------------------------------------------------------------
    suffix = file_path.suffix.lower()
    if suffix == ".he5":
        (
            attributes,
            decimal_dates,
            timeseries_datasets,
            dates,
            folder_name,
            lats,
            lons,
            shm,
        ) = read_from_hdfeos5_file(file_name)
        quality_params = None

    elif suffix == ".csv":
        (
            attributes,
            decimal_dates,
            timeseries_datasets,
            dates,
            folder_name,
            lats,
            lons,
            shm,
            quality_params,
        ) = read_from_csv_file(file_name)
    else:
        raise ValueError(
            f"The file '{file_path}' must have .he5 or .csv as extension."
        )

    # ------------------------------------------------------------------
    # Convert datasets to JSON chunks and write metadata
    # ------------------------------------------------------------------
    convert_data(
        attributes,
        decimal_dates,
        timeseries_datasets,
        dates,
        output_folder,
        folder_name,
        lats,
        lons,
        quality_params,
        args.num_workers,
    )

    # Clean up in-memory arrays
    del lats
    del lons
    if shm is not None:
        shm.close()
        shm.unlink()

    # ------------------------------------------------------------------
    # Run tippecanoe to generate MBTiles
    # ------------------------------------------------------------------
    os.chdir(os.path.abspath(output_folder))

    if high_res_mode(attributes):
        cmd = (
            "tippecanoe *.json -P -l chunk_1 -x d -pf -pk -o "
            + folder_name
            + ".mbtiles 2> tippecanoe_stderr.log"
        )
    else:
        cmd = (
            "tippecanoe *.json -P -l chunk_1 -x d -pf -pk "
            "-Bg -d9 -D12 -g12 -r0 -o "
            + folder_name
            + ".mbtiles 2> tippecanoe_stderr.log"
        )

    print("Now running tippecanoe with command %s" % cmd)
    os.system(cmd)

    # ------------------------------------------------------------------
    # Check how long it took to read data and create JSON / MBTiles
    # ------------------------------------------------------------------
    end_time = time.perf_counter()
    print("time elapsed: " + str(end_time - start_time))
    return


# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
