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
from datetime import datetime
import math
import geocoder
import numpy as np
from pathlib import Path

from mintpy.objects import HDFEOS
from mintpy.mask import mask_matrix
from mintpy.utils import utils as ut
import h5py
from multiprocessing import shared_memory
from multiprocessing import Pool
from multiprocessing import Value

import pandas as pd

chunk_num = Value("i", 0)

# Lat/lon column candidates (CSV). Keep in sync with minsar/insarmaps_utils/insarmaps_csv_geo.py
# Matching is case-insensitive (EGMS uses latitude/longitude).
LAT_CANDIDATES = ["Y_geocorr", "Latitude", "latitude", "Y", "ycoord"]
LON_CANDIDATES = ["X_geocorr", "Longitude", "longitude", "X", "xcoord"]

CHUNK_SIZE_DEFAULT = 20000

# ---------------------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------------------
def get_date(date_string):
    year = int(date_string[0:4])
    month = int(date_string[4:6])
    day = int(date_string[6:8])
    return date(year, month, day)


def get_decimal_date(d):
    start = date(d.year, 1, 1)
    return abs(d - start).days / 365.0 + d.year


def region_name_from_project_name(project_name):
    track_index = project_name.find("T")
    return project_name[:track_index]


needed_attributes = {
    "prf", "first_date", "mission", "WIDTH", "X_STEP", "processing_software",
    "wavelength", "processing_type", "beam_swath", "Y_FIRST", "look_direction",
    "flight_direction", "last_frame", "post_processing_method", "min_baseline_perp",
    "unwrap_method", "relative_orbit", "beam_mode", "LENGTH", "max_baseline_perp",
    "X_FIRST", "atmos_correct_method", "last_date", "first_frame", "frame", "Y_STEP", "history",
    "scene_footprint", "data_footprint", "downloadUnavcoUrl", "referencePdfUrl", "areaName", "referenceText",
    "REF_LAT", "REF_LON", "CENTER_LINE_UTC", "insarmaps_download_flag", "mintpy.subset.lalo",
}


def serialize_dictionary(dictionary, fileName):
    """Serialize a dictionary to a pickle file."""
    with open(fileName, "wb") as file:
        pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)


def flatten_to_1d(arr):
    """Return a 1D float array view/copy; None stays None."""
    if arr is None:
        return None
    return np.asarray(arr).reshape(-1)


def ensure_point_arrays(timeseries_datasets, lats, lons, quality_params=None):
    """
    Ensure timeseries, lat/lon, and quality arrays are 1D length n_points.
    HDFEOS readers may still produce 2D slices; flatten row-major (C order).
    """
    lats_1d = flatten_to_1d(lats)
    lons_1d = flatten_to_1d(lons)
    ts_1d = {d: flatten_to_1d(arr) for d, arr in timeseries_datasets.items()}
    quality_1d = None
    if quality_params:
        quality_1d = {k: flatten_to_1d(v) for k, v in quality_params.items()}
    return ts_1d, lats_1d, lons_1d, quality_1d


def generate_point_worker_args(
    decimal_dates,
    timeseries_datasets,
    dates,
    json_path,
    folder_name,
    chunk_size,
    lats,
    lons,
    quality_params=None,
):
    """
    Build argument tuples for worker processes that create JSON chunks.

    Each tuple:
      (decimal_dates, timeseries_datasets, dates, json_path, folder_name,
       (start_index, end_index), lats, lons, quality_params)
    Indices are inclusive over the 1D point arrays.
    """
    num_points = int(np.asarray(lats).size)
    worker_args = []
    start = 0
    while start < num_points:
        end = min(start + chunk_size, num_points) - 1
        worker_args.append(
            (
                decimal_dates,
                timeseries_datasets,
                dates,
                json_path,
                folder_name,
                (start, end),
                lats,
                lons,
                quality_params,
            )
        )
        start = end + 1
    return worker_args


def create_json(
    decimal_dates,
    timeseries_datasets,
    dates,
    json_path,
    folder_name,
    work_idxs,
    lats=None,
    lons=None,
    quality_params=None,
):
    """
    Create GeoJSON Point features for points in [start_idx, end_idx] (inclusive)
    and write them to a chunk_<N>.json file.

    timeseries_datasets[date], lats, lons, and quality arrays must be 1D.
    Point id property ``p`` is the 1D index (CSV row or row-major flat HDFEOS index).
    """
    global chunk_num

    siu_man = []
    x_arr = np.asarray(decimal_dates, dtype=float)
    start_idx, end_idx = work_idxs
    first_slice = timeseries_datasets[dates[0]]

    for i in range(start_idx, end_idx + 1):
        displacement0 = float(first_slice[i])
        if math.isnan(displacement0):
            continue

        longitude = float(lons[i])
        latitude = float(lats[i])

        displacement_values = []
        for datei in dates:
            val = timeseries_datasets[datei][i]
            if not math.isnan(val):
                displacement_values.append(float(val))
            else:
                displacement_values.append(None)

        mask = np.array([v is not None for v in displacement_values], dtype=bool)
        x_used = x_arr[mask]
        y = np.asarray(displacement_values, dtype=float)[mask]

        if y.size >= 2:
            A = np.vstack([x_used, np.ones(x_used.size)]).T
            m, _c = np.linalg.lstsq(A, y, rcond=None)[0]
        else:
            m = float("nan")

        safe_properties = {"d": displacement_values, "m": m, "p": i}

        if quality_params:
            for key, arr in quality_params.items():
                val = arr[i]

                if key == "point_ID":
                    if val is None or (
                        isinstance(val, float) and (math.isnan(val) or not math.isfinite(val))
                    ):
                        continue
                    safe_properties[key] = int(val)
                    continue

                if val is None or (isinstance(val, float) and math.isnan(val)):
                    safe_properties[key] = None
                else:
                    safe_properties[key] = val

        data = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
            "properties": safe_properties,
        }
        siu_man.append(data)

    if len(siu_man) > 0:
        with chunk_num.get_lock():
            chunk_num_val = chunk_num.value
            chunk_num.value += 1
        make_json_file(chunk_num_val, siu_man, dates, json_path, folder_name)


def convert_data(
    attributes,
    decimal_dates,
    timeseries_datasets,
    dates,
    json_path,
    folder_name,
    lats=None,
    lons=None,
    quality_params=None,
    num_workers=1,
    chunk_size=CHUNK_SIZE_DEFAULT,
):
    """Convert point (or flattened grid) timeseries to JSON chunks + metadata.pickle."""
    project_name = attributes["PROJECT_NAME"]
    region = region_name_from_project_name(project_name)

    # Per-run copy so removals do not mutate the module-level set
    attrs_needed = set(needed_attributes)

    if high_res_mode(attributes):
        # No regular geographic step (radar-coded / CSV point cloud)
        for key in ("X_STEP", "Y_STEP", "X_FIRST", "Y_FIRST", "WIDTH", "LENGTH"):
            attrs_needed.discard(key)

    if lats is None and lons is None:
        lats, lons = ut.get_lat_lon(attributes, dimension=1)

    timeseries_datasets, lats, lons, quality_params = ensure_point_arrays(
        timeseries_datasets, lats, lons, quality_params
    )
    num_points = int(lats.size)
    n_dates = len(dates)
    worker_args = generate_point_worker_args(
        decimal_dates,
        timeseries_datasets,
        dates,
        json_path,
        folder_name,
        chunk_size,
        lats,
        lons,
        quality_params,
    )
    num_chunks = len(worker_args)
    print(
        f"[INFO] JSON plan: {num_points} points × {n_dates} dates → "
        f"{num_chunks} chunk file(s) (chunk_size={chunk_size}, num_workers={num_workers})"
    )

    with chunk_num.get_lock():
        chunk_num.value = 0

    process_pool = Pool(num_workers)
    process_pool.starmap(create_json, worker_args)
    process_pool.close()

    insarmapsMetadata = {}
    mid_lat = float(np.nanmean(lats))
    mid_long = float(np.nanmean(lons))

    country = "None"
    try:
        g = geocoder.google([mid_lat, mid_long], method="reverse", timeout=60.0)
        country = str(g.country_long)
    except Exception:
        sys.stderr.write("timeout reverse geocoding country name")

    area = folder_name
    string_dates_sql = "{" + ",".join(str(k) for k in dates) + "}"
    decimal_dates_sql = "{" + ",".join(str(d) for d in decimal_dates) + "}"

    attribute_keys = []
    attribute_values = []
    max_digit = max([len(key) for key in list(attrs_needed)] + [0])

    for k, v in attributes.items():
        if k in attrs_needed:
            print(f"{k:<{max_digit}}     {v}")
            attribute_keys.append(k)
            attribute_values.append(v)

    # Force 'elevation' to show up in popup
    attribute_keys.append("dem")
    attribute_values.append("1")

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
    insarmapsMetadata["needed_attributes"] = attrs_needed

    metadataFilePath = json_path + "/metadata.pickle"
    serialize_dictionary(insarmapsMetadata, metadataFilePath)


def make_json_file(chunk_num_val, points, dates, json_path, folder_name):
    chunk = "chunk_" + str(chunk_num_val) + ".json"
    json_file = open(json_path + "/" + chunk, "w")
    json_features = [json.dumps(feature) for feature in points]
    string_json = "\n".join(json_features)
    json_file.write("%s" % string_json)
    json_file.close()
    print("converted chunk " + str(chunk_num_val))
    return chunk


def high_res_mode(attributes):
    """True when geographic X_STEP/Y_STEP are absent (radar-coded or point CSV)."""
    try:
        _x_step = attributes["X_STEP"]
        _y_step = attributes["Y_STEP"]
    except Exception:
        return True
    return False


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for hdfeos5_or_csv_2json_mbtiles.py."""
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
    required.add_argument("file", help="(unavco) Input file to ingest (.he5 or .csv).")
    required.add_argument("outputDir", help="Directory to place JSON chunk files and MBTiles output.")
    return parser


def add_dummy_attribute(attributes: dict, is_sarvey_format: bool) -> None:
    """
    Fill in missing attributes with hard-coded defaults to satisfy Insarmaps
    expectations, mainly for CSV imports.
    """
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
    """Add/update data_footprint and scene_footprint from lat/lon extents."""
    min_lat = float(np.nanmin(lats))
    max_lat = float(np.nanmax(lats))
    min_lon = float(np.nanmin(lons))
    max_lon = float(np.nanmax(lons))

    polygon = (
        f"POLYGON(({max_lon} {min_lat},"
        f"{min_lon} {min_lat},"
        f"{min_lon} {max_lat},"
        f"{max_lon} {max_lat},"
        f"{max_lon} {min_lat}))"
    )
    print("data_footprint: ", polygon)
    attributes["data_footprint"] = polygon
    attributes["scene_footprint"] = polygon


def read_from_hdfeos5_file(file_name):
    """Read HDFEOS5 and return 1D point arrays (row-major flatten of the 2D grid)."""
    should_mask = True

    path_name_and_extension = os.path.basename(file_name).split(".")
    path_name = path_name_and_extension[0]

    he_obj = HDFEOS(file_name)
    he_obj.open(print_msg=False)
    displacement_3d_matrix = he_obj.read(datasetName="displacement")
    mask = he_obj.read(datasetName="mask")
    if should_mask:
        print("Masking displacement")
        displacement_3d_matrix = mask_matrix(displacement_3d_matrix, mask)
    del mask

    print("Creating shared memory for multiple processes")
    shm = shared_memory.SharedMemory(create=True, size=displacement_3d_matrix.nbytes)
    shared_displacement_3d_matrix = np.ndarray(
        displacement_3d_matrix.shape, dtype=displacement_3d_matrix.dtype, buffer=shm.buf
    )
    shared_displacement_3d_matrix[:] = displacement_3d_matrix[:]
    del displacement_3d_matrix
    displacement_3d_matrix = shared_displacement_3d_matrix

    dates = he_obj.dateList
    attributes = dict(he_obj.metadata)

    decimal_dates = []
    timeseries_datasets = {}
    num_date = len(dates)
    for i in range(num_date):
        # Flatten row-major so p = row * WIDTH + col
        timeseries_datasets[dates[i]] = np.squeeze(displacement_3d_matrix[i, :, :]).reshape(-1)
        d = get_date(dates[i])
        decimal_dates.append(get_decimal_date(d))
    del displacement_3d_matrix

    path_list = path_name.split("/")
    folder_name = path_name.split("/")[len(path_list) - 1]

    f = h5py.File(he_obj.file, "r")
    lats = np.array(f["HDFEOS"]["GRIDS"]["timeseries"]["geometry"]["latitude"]).reshape(-1)
    lons = np.array(f["HDFEOS"]["GRIDS"]["timeseries"]["geometry"]["longitude"]).reshape(-1)

    attributes["collection"] = "hdfeos5"
    return attributes, decimal_dates, timeseries_datasets, dates, folder_name, lats, lons, shm


def add_calculated_attributes(attributes: dict) -> None:
    """Infer REF_LAT/REF_LON/first_date/last_date/history from temporary CSV keys."""
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


def enrich_attributes_from_csv_filename(attributes: dict, csv_path) -> dict:
    """
    Enrich mission / relative_orbit / beam_mode from a standardized CSV stem
    like ``S1_044_...`` or ``TSX_135_...`` when still missing.
    """
    stem = Path(csv_path).stem.upper()
    m = re.match(r"^(S1|TSX|ALOS|ERS|ENVISAT)_(\d{3})_", stem)
    if not m:
        return attributes

    file_mission = m.group(1)
    file_rel = int(m.group(2))

    if not attributes.get("mission") or attributes.get("mission", "").upper() == "S1":
        attributes["mission"] = file_mission.title()

    attributes.setdefault("relative_orbit", file_rel)

    if attributes.get("mission", "").upper() == "TSX":
        attributes.setdefault("beam_mode", "SM")

    return attributes


def detect_lat_lon_columns(columns):
    """Return (lat_col, lon_col) from header names using LAT/LON_CANDIDATES (case-insensitive)."""
    lower_map = {str(c).lower(): c for c in columns}
    lat_col = next((lower_map[c.lower()] for c in LAT_CANDIDATES if c.lower() in lower_map), None)
    lon_col = next((lower_map[c.lower()] for c in LON_CANDIDATES if c.lower() in lower_map), None)
    return lat_col, lon_col


def read_from_csv_file(file_name):
    """
    Read a SARvey/NOAA-style CSV into 1D point arrays (no fake spatial grid).

    Returns
    -------
    attributes, decimal_dates, timeseries_datasets, dates, folder_name,
    lats, lons, shm (None), quality_params
    """
    df = pd.read_csv(file_name)
    df.columns = [c.strip() for c in df.columns]

    pid_col = next((c for c in df.columns if c.lower() == "point_id"), None)
    if pid_col is not None:
        df["point_ID"] = pd.to_numeric(df[pid_col], errors="coerce")
        print(f"[INFO] Detected point_ID column: {pid_col} (non-null={df['point_ID'].notna().sum()})")

    lat_col, lon_col = detect_lat_lon_columns(df.columns)
    print(f"Using columns: lat = {lat_col}, lon = {lon_col}")

    if lat_col is None or lon_col is None:
        raise ValueError(
            "Could not find latitude/longitude columns in the CSV. Supported names: "
            f"{LAT_CANDIDATES} and {LON_CANDIDATES}."
        )

    lats = df[lat_col].to_numpy(dtype=float)
    lons = df[lon_col].to_numpy(dtype=float)

    sarvey_time_cols = [col for col in df.columns if col.startswith("D") and col[1:].isdigit()]
    is_sarvey_format = bool(sarvey_time_cols)

    if is_sarvey_format:
        time_cols = sorted(sarvey_time_cols)
        timeseries_data = df[time_cols].to_numpy(dtype=float) / 1000.0  # mm -> m
        dates = [col[1:] for col in time_cols]
    else:
        time_cols = [col for col in df.columns if col.isdigit()]
        timeseries_data = df[time_cols].to_numpy(dtype=float) / 1000.0
        dates = time_cols

    num_points = len(df)
    decimal_dates = [get_decimal_date(get_date(d)) for d in dates]
    # timeseries_data shape (n_points, n_dates) -> one 1D array per date
    timeseries_datasets = {d: timeseries_data[:, i] for i, d in enumerate(dates)}

    attributes = {}
    attributes["PROJECT_NAME"] = "CSV_IMPORT"
    attributes["LAT_ARRAY"] = lats
    attributes["LON_ARRAY"] = lons
    attributes["DATE_COLUMNS"] = dates
    attributes["processing_type"] = "LOS_TIMESERIES"
    attributes.setdefault("look_direction", "R")
    attributes["collection"] = "sarvey"

    filename_upper = Path(file_name).stem.upper()
    vert_col_candidates = ["VEL_V", "V_STDEV_V"]
    has_vertical_columns = any(col in df.columns for col in vert_col_candidates)

    if "VERT" in filename_upper or has_vertical_columns:
        attributes["data_type"] = "Vertical Displacement"
    else:
        attributes["data_type"] = "LOS Displacement"

    print(f"[INFO] Set data_type: {attributes['data_type']}")
    print(f"[INFO] CSV points: {num_points} (1D; no fake grid)")

    attributes.setdefault("PLATFORM", "S1")
    attributes.setdefault("MISSION", "S1")
    attributes = enrich_attributes_from_csv_filename(attributes, file_name)

    if attributes.get("mission") in (None, "None", "", "null"):
        if attributes.get("MISSION"):
            attributes["mission"] = attributes["MISSION"]

    if attributes.get("platform") in (None, "None", "", "null"):
        if attributes.get("PLATFORM"):
            attributes["platform"] = attributes["PLATFORM"]

    if "relative_orbit" in attributes:
        try:
            attributes["relative_orbit"] = int(attributes["relative_orbit"])
        except Exception:
            pass

    add_calculated_attributes(attributes)
    add_data_footprint_attribute(attributes, lats, lons)
    add_dummy_attribute(attributes, is_sarvey_format)

    quality_params = {}
    if "dem_error" in df.columns and "dem" in df.columns:
        quality_params["dem_error"] = df["dem_error"].to_numpy(dtype=float)
        quality_params["elevation"] = (
            df["dem_error"].to_numpy(dtype=float) - df["dem"].to_numpy(dtype=float)
        )
    elif "height_ortho" in df.columns:
        # EGMS / similar: orthometric height as elevation popup field
        quality_params["elevation"] = df["height_ortho"].to_numpy(dtype=float)
    if "coherence" in df.columns:
        quality_params["coherence"] = df["coherence"].to_numpy(dtype=float)
    elif "temporal_coherence" in df.columns:
        quality_params["coherence"] = df["temporal_coherence"].to_numpy(dtype=float)
    if "omega" in df.columns:
        quality_params["omega"] = df["omega"].to_numpy(dtype=float)
    if "st_consist" in df.columns:
        quality_params["st_consist"] = df["st_consist"].to_numpy(dtype=float)
    if "point_ID" in df.columns:
        quality_params["point_ID"] = df["point_ID"].to_numpy(dtype=float)
    elif "pid" in df.columns:
        quality_params["point_ID"] = pd.to_numeric(df["pid"], errors="coerce").to_numpy(dtype=float)

    folder_name = os.path.basename(file_name).split(".")[0]
    shm = None

    print(" Check: read_from_csv_file output")
    print(" - timeseries_datasets keys:", list(timeseries_datasets.keys())[:3], "...")
    print(" - sample slice shape:", timeseries_datasets[dates[0]].shape)
    print(" - lats shape:", lats.shape)
    print(" - lons shape:", lons.shape)
    print(" - Number of dates:", len(dates))
    print(" - Attributes:", attributes)

    return (
        attributes,
        decimal_dates,
        timeseries_datasets,
        dates,
        folder_name,
        lats,
        lons,
        shm,
        quality_params if quality_params else None,
    )


def main():
    """Entry point for hdfeos5_or_csv_2json_mbtiles.py."""
    parser = build_parser()
    args = parser.parse_args()
    file_name = args.file
    output_folder = args.outputDir

    try:
        os.mkdir(output_folder)
    except FileExistsError:
        print(output_folder + " already exists")

    file_path = Path(file_name)
    start_time = time.perf_counter()

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
        raise ValueError(f"The file '{file_path}' must have .he5 or .csv as extension.")

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

    del lats
    del lons
    if shm is not None:
        shm.close()
        shm.unlink()

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

    end_time = time.perf_counter()
    print("time elapsed: " + str(end_time - start_time))


if __name__ == "__main__":
    main()
