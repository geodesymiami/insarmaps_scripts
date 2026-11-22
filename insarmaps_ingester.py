#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Alfredo Terrero, 2016                            #
############################################################
"""Step 1 of an Insarmaps ingest: an HDF-EOS5 file (.he5) or a CSV to the dataset's points, as arrays
(points_bundle.py), and metadata.pickle. Step 2, json_mbtiles2insarmaps.py, loads them into the database and builds
the dataset's recolor store, from which the map draws it.

This script replaces hdfeos5_2json_mbtiles.py, which minsar ran for an HDF-EOS5 file, and
hdfeos5_or_csv_2json_mbtiles.py, which it ran for a CSV. Each input gets what its script wrote. The CSV script also
took an .he5 and wrote other point numbers and metadata for it; --he5-like-csv writes that (minsar doesn't use it, and
no dataset on insarmaps came from it).
"""
import argparse
import os
import pickle
import re
import sys
import time
from datetime import date
from datetime import datetime
from pathlib import Path

import geocoder
import h5py
import numpy as np
import pandas as pd
from mintpy.mask import mask_matrix
from mintpy.objects import HDFEOS

import points_bundle

# Lat/lon column candidates (CSV). Keep in sync with minsar/insarmaps_utils/insarmaps_csv_geo.py
# Matching is case-insensitive (EGMS uses latitude/longitude).
LAT_CANDIDATES = ["Y_geocorr", "Latitude", "latitude", "Y", "ycoord"]
LON_CANDIDATES = ["X_geocorr", "Longitude", "longitude", "X", "xcoord"]

# The attributes that go into the dataset's area row (attribute_keys/attribute_values) and its extra_attributes
NEEDED_ATTRIBUTES = frozenset({
    "prf", "first_date", "mission", "WIDTH", "X_STEP", "processing_software",
    "wavelength", "processing_type", "beam_swath", "Y_FIRST", "look_direction",
    "flight_direction", "last_frame", "post_processing_method", "min_baseline_perp",
    "unwrap_method", "relative_orbit", "beam_mode", "LENGTH", "max_baseline_perp",
    "X_FIRST", "atmos_correct_method", "last_date", "first_frame", "frame", "Y_STEP", "history",
    "scene_footprint", "data_footprint", "downloadUnavcoUrl", "referencePdfUrl", "areaName", "referenceText",
    "REF_LAT", "REF_LON", "CENTER_LINE_UTC", "insarmaps_download_flag", "mintpy.subset.lalo",
})
# hdfeos5_2json_mbtiles.py's list lacked a comma between these two, which made them one name that no attribute has,
# so no HDF-EOS5 dataset got them (MintPy files carry unwrap_method = Unknown). An .he5 still leaves them out.
NOT_FROM_HDFEOS5 = frozenset({"min_baseline_perp", "unwrap_method"})
GRID_ATTRIBUTES = ("X_STEP", "Y_STEP", "X_FIRST", "Y_FIRST")


# ---------------------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------------------
def get_date(date_string):
    year = int(date_string[0:4])
    month = int(date_string[4:6])
    day = int(date_string[6:8])
    return date(year, month, day)


def get_decimal_date(d):
    """year + days elapsed in the year / 365: the date as the velocity fit's time"""
    start = date(d.year, 1, 1)
    return abs(d - start).days / 365.0 + d.year


def region_name_from_project_name(project_name):
    track_index = project_name.find("T")
    return project_name[:track_index]


def dataset_name(file_name):
    """The dataset's name (the area's unavco_name): the file name up to its first dot."""
    return os.path.basename(file_name).split(".")[0]


def high_res_mode(attributes):
    """True when geographic X_STEP/Y_STEP are absent (radar-coded or point CSV)."""
    return "X_STEP" not in attributes or "Y_STEP" not in attributes


def write_metadata(json_path, area, attributes, dates, decimal_dates, mid_lat, mid_long, needed, country_default,
                   extra_keys=()):
    """Write metadata.pickle, from which json_mbtiles2insarmaps.py makes the dataset's area row and adds the
    attributes named in needed to extra_attributes. attribute_keys/attribute_values: the attributes named in needed, in
    the order of attributes, then extra_keys ((key, value) pairs). country is country_default when the lookup fails."""
    project_name = attributes["PROJECT_NAME"]

    country = country_default
    try:
        g = geocoder.google([mid_lat, mid_long], method="reverse", timeout=60.0)
        country = str(g.country_long)
    except Exception:
        sys.stderr.write("timeout reverse geocoding country name")

    attribute_keys = []
    attribute_values = []
    max_digit = max([len(key) for key in needed] + [0])
    for k, v in attributes.items():
        if k in needed:
            print(f"{k:<{max_digit}}     {v}")
            attribute_keys.append(k)
            attribute_values.append(v)
    for k, v in extra_keys:
        attribute_keys.append(k)
        attribute_values.append(v)

    insarmapsMetadata = {
        "area": area,
        "project_name": project_name,
        "mid_long": mid_long,
        "mid_lat": mid_lat,
        "country": country,
        "region": region_name_from_project_name(project_name),
        "chunk_num": 1,
        "attribute_keys": attribute_keys,
        "attribute_values": attribute_values,
        # for some reason pgsql only takes {} not []
        "string_dates_sql": "{" + ",".join(str(k) for k in dates) + "}",
        "decimal_dates_sql": "{" + ",".join(str(d) for d in decimal_dates) + "}",
        "attributes": attributes,
        "needed_attributes": needed,
    }
    with open(os.path.join(json_path, "metadata.pickle"), "wb") as file:
        pickle.dump(insarmapsMetadata, file, protocol=pickle.HIGHEST_PROTOCOL)


def write_metadata_like_csv(json_path, area, attributes, dates, decimal_dates, lats, lons):
    """metadata.pickle as hdfeos5_or_csv_2json_mbtiles.py wrote it, for a CSV and for an HDF-EOS5 file with
    --he5-like-csv: the needed attributes, less WIDTH, LENGTH and the grid's when there is no geographic grid
    (high_res_mode, always so for a CSV); the mean position of all the rows or cells (lats, lons); "dem" last, which
    makes 'elevation' show up in the popup; "None" when the country lookup fails."""
    needed = set(NEEDED_ATTRIBUTES)
    if high_res_mode(attributes):
        needed.difference_update(GRID_ATTRIBUTES + ("WIDTH", "LENGTH"))
    write_metadata(json_path, area, attributes, dates, decimal_dates, float(np.nanmean(lats)),
                   float(np.nanmean(lons)), needed, country_default="None", extra_keys=[("dem", "1")])


# ---------------------------------------------------------------------------------------
# HDF-EOS5 (.he5)
# ---------------------------------------------------------------------------------------
def read_from_hdfeos5_file(file_name):
    """Read HDFEOS5: the attributes, the dates, the masked displacement (dates x rows x cols, NaN where masked) and the
    2D lat/lon grids."""
    he_obj = HDFEOS(file_name)
    he_obj.open(print_msg=False)
    displacement = he_obj.read(datasetName="displacement")
    mask = he_obj.read(datasetName="mask")
    print("Masking displacement")
    displacement = mask_matrix(displacement, mask)
    del mask

    # MintPy doesn't read the lat/lon grids, so they come from the raw h5 file
    with h5py.File(he_obj.file, "r") as f:
        lats = np.array(f["HDFEOS"]["GRIDS"]["timeseries"]["geometry"]["latitude"])
        lons = np.array(f["HDFEOS"]["GRIDS"]["timeseries"]["geometry"]["longitude"])
    return dict(he_obj.metadata), he_obj.dateList, displacement, lats, lons


def convert_hdfeos5(file_name, output_folder, like_csv=False):
    """Writes an HDF-EOS5 file's points and metadata.pickle as hdfeos5_2json_mbtiles.py did or, with like_csv, as
    hdfeos5_or_csv_2json_mbtiles.py did: each cell numbered by its flat index, and the metadata a CSV gets."""
    attributes, dates, displacement, lats, lons = read_from_hdfeos5_file(file_name)
    decimal_dates = [get_decimal_date(get_date(d)) for d in dates]

    # every cell with values at two dates or more, its velocity fitted over the dates it has, numbered from the first
    # cell of each block of 20,000 cells or, like_csv, by its flat index (row * WIDTH + col)
    num_points = points_bundle.write_grid_points(output_folder, displacement, lats, lons, decimal_dates,
                                                 numbering="flat" if like_csv else "blocks")
    print("points: %d of %d cells, %d dates" % (num_points, lats.size, len(dates)))
    del displacement

    if like_csv:
        attributes["collection"] = "hdfeos5"
        write_metadata_like_csv(output_folder, dataset_name(file_name), attributes, dates, decimal_dates,
                                lats.reshape(-1), lons.reshape(-1))
        return

    num_columns = int(attributes["WIDTH"])
    num_rows = int(attributes["LENGTH"])
    print("columns: %d" % num_columns)
    print("rows: %d" % num_rows)
    needed = set(NEEDED_ATTRIBUTES - NOT_FROM_HDFEOS5)
    if high_res_mode(attributes):
        # radar-coded: no regular geographic grid; the middle cell's position
        needed.difference_update(GRID_ATTRIBUTES)
        num_rows, num_columns = lats.shape
        mid_long = float(lons[num_rows // 2][num_columns // 2])
        mid_lat = float(lats[num_rows // 2][num_columns // 2])
    else:
        # geocoded: the middle of the grid
        x_step = float(attributes["X_STEP"])
        y_step = float(attributes["Y_STEP"])
        x_first = float(attributes["X_FIRST"])
        y_first = float(attributes["Y_FIRST"])
        mid_long = x_first + ((num_columns / 2) * x_step)
        mid_lat = y_first + ((num_rows / 2) * y_step)

    write_metadata(output_folder, dataset_name(file_name), attributes, dates, decimal_dates, mid_lat, mid_long,
                   needed, country_default=None)


# ---------------------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------------------
def integer_or_double(values):
    """The point table's type for a column of numbers held as floats (NaN where missing): integer when every value
    there is a whole number that fits int32, as ogr2ogr typed the JSON this script used to write, else double
    precision."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0 or (np.all(v == np.round(v)) and v.min() >= -2**31 and v.max() < 2**31):
        return "integer"
    return "double precision"


def write_csv_points(json_path, timeseries_data, lats, lons, decimal_dates, quality_params=None):
    """Writes the points of a CSV (points_bundle.py): every row with values at two dates or more, numbered by its row
    (p = row index), with its velocity fitted over the dates it has; quality_params {name: values per row} become
    the table's attribute columns, in this order, with NULL where a value is missing. Returns P."""
    timeseries_data = np.asarray(timeseries_data, dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(timeseries_data).sum(axis=1) >= 2)
    if len(rows) == 0:
        raise ValueError("no row of the CSV has values at two dates or more")
    lon, lat = np.asarray(lons, dtype=np.float64)[rows], np.asarray(lats, dtype=np.float64)[rows]
    bad = ~(np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) < 90))
    if bad.any():
        raise ValueError("%d rows with values have no position (latitude/longitude missing, or |latitude| >= 90), "
                         "the first at CSV row %d (counting the first data row as 0)" % (bad.sum(), rows[bad][0]))
    d = timeseries_data[rows]
    m = points_bundle.fit_velocities(decimal_dates, d)
    attributes = {name: np.asarray(v, dtype=np.float64)[rows] for name, v in (quality_params or {}).items()}
    # the JSON this script wrote left point_ID out of a row without one, so a CSV with no numeric point_ID at all
    # (EGMS's pid is text) got no point_ID column: the same here, rather than a column of NULLs
    if "point_ID" in attributes and np.isnan(attributes["point_ID"]).all():
        del attributes["point_ID"]
    types = {"point_ID": integer_or_double(attributes["point_ID"])} if "point_ID" in attributes else {}
    return points_bundle.write_points(json_path, rows, lon, lat, m, d, attributes, types)


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
    attributes, decimal_dates, timeseries_data (rows x dates, m), dates, folder_name,
    lats, lons, quality_params
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

    folder_name = dataset_name(file_name)

    print(" Check: read_from_csv_file output")
    print(" - dates:", dates[:3], "...")
    print(" - values shape (rows, dates):", timeseries_data.shape)
    print(" - lats shape:", lats.shape)
    print(" - lons shape:", lons.shape)
    print(" - Number of dates:", len(dates))
    print(" - Attributes:", attributes)

    return (
        attributes,
        decimal_dates,
        timeseries_data,
        dates,
        folder_name,
        lats,
        lons,
        quality_params if quality_params else None,
    )


def convert_csv(file_name, output_folder):
    (
        attributes,
        decimal_dates,
        timeseries_data,
        dates,
        folder_name,
        lats,
        lons,
        quality_params,
    ) = read_from_csv_file(file_name)
    num_points = write_csv_points(output_folder, timeseries_data, lats, lons, decimal_dates, quality_params)
    print("points: %d of %d rows, %d dates" % (num_points, lats.size, len(dates)))
    write_metadata_like_csv(output_folder, folder_name, attributes, dates, decimal_dates, lats, lons)


# ---------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for insarmaps_ingester.py."""
    example_text = """\
This program writes the dataset's points as arrays (points.json, p.npy, lon.npy,
lat.npy, m.npy, d.npy and attr_NAME.npy for a CSV's quality columns) and
metadata.pickle. json_mbtiles2insarmaps.py uploads them and builds the dataset's
recolor store, from which the map draws it. A file whose name ends in .csv is read
as a CSV, any other as HDF-EOS5.

Examples:
  insarmaps_ingester.py mintpy/S1_IW1_128_0596_0597_20160605_XXXXXXXX_S00887_S00783_W091208_W091105.he5 mintpy/JSON
  insarmaps_ingester.py --he5-like-csv mintpy/S1_IW1_128_0596_0597_20160605_XXXXXXXX.he5 mintpy/JSON
  insarmaps_ingester.py sarvey_test.csv ./JSON
  insarmaps_ingester.py NOAA_SNT_A_VERT_10_50m.csv ./JSON_NOAA
    """

    parser = argparse.ArgumentParser(
        description="Convert an HDFEOS5 (.he5) or CSV file for ingestion into Insarmaps.",
        epilog=example_text,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--num-workers",
        help="Not needed any more: the conversion runs in one process. Accepted for the callers that still pass it.",
        required=False,
        default=1,
        type=int,
    )
    parser.add_argument(
        "--he5-like-csv",
        action="store_true",
        help="For an HDF-EOS5 file, write what hdfeos5_or_csv_2json_mbtiles.py wrote for one, as for a CSV:\n"
             "p is the cell's flat index (row * WIDTH + col), the dataset's position the mean of its cells',\n"
             "and the area's attributes a CSV's (all needed ones, WIDTH and LENGTH only with a geographic grid,\n"
             "and dem). Without it, an HDF-EOS5 file gets what hdfeos5_2json_mbtiles.py wrote, as minsar\n"
             "uses. No effect on a CSV.",
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument("file", help="(unavco) Input file to ingest (.he5 or .csv).")
    required.add_argument("outputDir", help="Directory to place the points files and metadata.pickle.")
    return parser


# ---------------------------------------------------------------------------------------
# START OF EXECUTABLE
# ---------------------------------------------------------------------------------------
def main():
    """Entry point for insarmaps_ingester.py."""
    args = build_parser().parse_args()
    start_time = time.perf_counter()

    if os.path.isdir(args.outputDir):
        print(args.outputDir + " already exists")
    os.makedirs(args.outputDir, exist_ok=True)

    # as minsar's ingest_insarmaps.bash decides: a .csv is a CSV, any other file HDF-EOS5
    if Path(args.file).suffix.lower() == ".csv":
        convert_csv(args.file, args.outputDir)
    else:
        convert_hdfeos5(args.file, args.outputDir, like_csv=args.he5_like_csv)

    print("time elapsed: " + str(time.perf_counter() - start_time))


if __name__ == "__main__":
    main()
