#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Alfredo Terrero, 2016                            #
############################################################


import os
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
#  FA 4/2025 suggestions:
# required_attributes_in_data = {
#     "mission",  
#     "beam_mode",  
#     "flight_direction", 
#     "relative_orbit",  
#     "processing_method",            # {MiaplPy, MintPy, Sarvey, TRE}
# }
# required_attributes_inferred = {    
#     "data_footprint",               # infer if not given
#     "data_type",                    # Default: LOS_TIMESERIES
#     "look_direction",               # Default: R  (L for mission=NISAR)
#     "start_date",                   # Always infer
#     "end_date",                     # Always infer
#     "history",                      # Always infer (processing day's date, i.e. today)
# }
# optional_attributes_in_data = {
#     "REF_LAT", 
#     "REF_LON", 
#     "areaName",                     # to be used for search
#     "beamSwath"                  
# }

# Renamed_attributes:
# "data_type"   (formerly "processing_type") 
# "processing_method" (formerly "post_processing_method")

def serialize_dictionary(dictionary, fileName):
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
    global chunk_num
    # create a siu_man array to store json point objects
    siu_man = []
    displacement_values = []
    displacements = '{'
    # np array of decimal dates, x parameter in linear regression equation
    x = decimal_dates
    A = np.vstack([x, np.ones(len(x))]).T
    y = []
    point_num = work_idxs[0]
    # iterate through h5 file timeseries
    for (row, col), value in np.ndenumerate(timeseries_datasets[dates[0]]):
        cur_iter_point_num = row * num_columns + col
        if cur_iter_point_num < work_idxs[0]:
            continue
        elif cur_iter_point_num > work_idxs[1]:
            break

        longitude = float(lons[row][col])
        latitude = float(lats[row][col])
        displacement = float(value) 
        # if value is not equal to naN, create a new json point object and append to siu_man array
        if not math.isnan(displacement):
            # get displacement values for all the dates into array for json and string for pgsql
            for datei in dates:
                displacement = timeseries_datasets[datei][row][col]
                displacements += (str(displacement) + ",")
                displacement_values.append(float(displacement))
            displacements = displacements[:len(displacements) - 1] + '}'

            # np array of displacement values, y parameter in linear regression equation
            y = displacement_values

            # y = mx + c -> we want m = slope of the linear regression line 
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]

            # Base properties for all inputs
            data = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
                "properties": {"d": displacement_values, "m": m, "p": point_num}
            }

            # Add quality parameters at this location
            if quality_params:   
                for key in quality_params.keys():
                    data["properties"][key] = quality_params[key][row][col]

            siu_man.append(data)

            # clear displacement array for json and the other string for dictionary, for next point
            displacement_values = []
            displacements = '{'
            point_num += 1

    if len(siu_man) > 0:
        chunk_num_val = -1
        with chunk_num.get_lock():
            chunk_num_val = chunk_num.value
            chunk_num.value += 1
  
        make_json_file(chunk_num_val, siu_man, dates, json_path, folder_name)

        siu_man = []

# ---------------------------------------------------------------------------------------
# convert h5 file to json and upload it. folder_name == unavco_name
def convert_data(attributes, decimal_dates, timeseries_datasets, dates, json_path, folder_name, lats=None, lons=None, quality_params=None, num_workers=1):

    project_name = attributes["PROJECT_NAME"]
    region = region_name_from_project_name(project_name)
    # get the attributes for calculating latitude and longitude
    x_step, y_step, x_first, y_first = 0, 0, 0, 0
    if high_res_mode(attributes):
        needed_attributes.remove("X_STEP")
        needed_attributes.remove("Y_STEP")
        needed_attributes.remove("X_FIRST")
        needed_attributes.remove("Y_FIRST")
    else:
        x_step = float(attributes["X_STEP"])
        y_step = float(attributes["Y_STEP"])
        x_first = float(attributes["X_FIRST"])
        y_first = float(attributes["Y_FIRST"])

    num_columns = int(attributes["WIDTH"])
    num_rows = int(attributes["LENGTH"])
    print("columns: %d" % num_columns)
    print("rows: %d" % num_rows)
    if lats is None and lons is None:
        lats, lons = ut.get_lat_lon(attributes, dimension=1)

    CHUNK_SIZE = 20000
    process_pool = Pool(num_workers)
    process_pool.starmap(create_json, generate_worker_args(decimal_dates, timeseries_datasets, dates, json_path, folder_name, CHUNK_SIZE, lats, lons, num_columns, num_rows, quality_params))
    process_pool.close()

    # dictionary to contain metadata needed by db to be written to a file
    # and then be read by json_mbtiles2insarmaps.py
    insarmapsMetadata = {}
    # calculate mid lat and long of dataset - then use google python lib to get country
    # technically don't need the else since we always use lats and lons arrays now
    
    #below part removed for now ,csv
    #if high_res_mode(attributes):
    #    num_rows, num_columns = lats.shape
    #    mid_long = float(lons[num_rows // 2][num_columns // 2])
    #    mid_lat = float(lats[num_rows // 2][num_columns // 2])
    #else:
    #    mid_long = x_first + ((num_columns/2) * x_step)
    #    mid_lat = y_first + ((num_rows/2) * y_step)
    mid_lat = float(np.nanmean(lats))
    mid_long = float(np.nanmean(lons))

    country = "None"
    try:
        g = geocoder.google([mid_lat,mid_long], method='reverse', timeout=60.0)
        country = str(g.country_long)
    except Exception:
        sys.stderr.write("timeout reverse geocoding country name")

    area = folder_name

    # for some reason pgsql only takes {} not [] - format date arrays and attributes to be inserted to pgsql
    string_dates_sql = '{'
    for k in dates:
        string_dates_sql += (str(k) + ",")
    string_dates_sql = string_dates_sql[:len(string_dates_sql) - 1] + '}'

    decimal_dates_sql = '{'
    for d in decimal_dates:
        decimal_dates_sql += (str(d) + ",")
    decimal_dates_sql = decimal_dates_sql[:len(decimal_dates_sql) - 1] + '}'
    # add keys and values to area table. TODO: this will be removed eventually
    # and all attributes will be put in extra_attributes table
    attribute_keys = '{'
    attribute_values = '{'
    max_digit = max([len(key) for key in list(needed_attributes)] + [0])
    for k in attributes:
        v = attributes[k]
        if k in needed_attributes:
            print('{k:<{w}}     {v}'.format(k=k, w=max_digit, v=v))
            attribute_keys += (str(k) + ",")
            attribute_values += (str(v) + ',')
    # force 'elevation' to show up in popup
    attribute_keys += "dem,"
    attribute_values += "1,"

    attribute_keys = attribute_keys[:len(attribute_keys)-1] + '}'
    attribute_values = attribute_values[:len(attribute_values)-1] + '}'

    # write out metadata to json file
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
def build_parser():
    example_text = """\
This program will create temporary json chunk files which, when concatenated together, comprise the whole dataset. Tippecanoe is used for concatenating these chunk files into the mbtiles file.

Examples:
  hdfeos5_or_csv_2json_mbtiles.py mintpy/S1_IW1_128_0596_0597_20160605_XXXXXXXX_S00887_S00783_W091208_W091105.he5 mintpy/JSON --num-workers 3
  hdfeos5_or_csv_2json_mbtiles.py sarvey_test.csv ./JSON
  hdfeos5_or_csv_2json_mbtiles.py NOAA_SNT_A_VERT_10_50m.csv ./JSON_NOAA
  """
    parser = argparse.ArgumentParser(description="Convert a hdfeos5 or CSV file for ingestion into insarmaps.", epilog=example_text, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--num-workers", help="Number of simultaneous processes to run for ingest.", required=False, default=1, type=int)
    required = parser.add_argument_group("required arguments")
    required.add_argument("file", help="unavco file to ingest")
    required.add_argument("outputDir", help="directory to place json files and mbtiles file")

    return parser

def add_dummy_attribute(attributes, is_sarvey_format):
    # add needed attributes to attributes dictionary
    # FA 4/2025: parameters to be added manually: flight_direction, mission,relative_orbit,look_direction
    attributes['atmos_correct_method'] = None
    attributes['beam_mode'] = 'IW'
    attributes['beam_swath'] = 1
    attributes['post_processing_method'] = 'MintPy'
    attributes['prf'] = 1717.128973878037
    attributes['processing_software'] = 'isce'
    attributes['scene_footprint'] = 'POLYGON((-90.79583946164999 -0.687890034792316,-90.86911230465793 -1.0359825079903804,-91.62407871076888 -0.8729106902243329,-91.55064943686261 -0.5251520401739668,-90.79583946164999 -0.687890034792316))'
    attributes['wavelength'] = 0.05546576
    attributes['first_frame'] = 556
    attributes['last_frame'] = 557

    # FA 4/2025: These attributes can also be calcuated from the data
    if not is_sarvey_format:
        attributes['REF_LAT'] = -0.83355445
        attributes['REF_LON'] = -91.12596
        attributes['data_footprint'] = 'POLYGON((-91.19760131835938 -0.7949774265289307,-91.11847686767578 -0.7949774265289307,-91.11847686767578 -0.8754903078079224,-91.19760131835938 -0.8754903078079224,-91.19760131835938 -0.7949774265289307))'
        attributes['first_date'] = '2016-06-05'
        attributes['last_date'] = '2016-08-28'

def add_data_footprint_attribute(attributes, lats, lons):
    """
    Adds a 'data_footprint' attribute to the attributes dictionary based on min/max of lat/lon arrays.

    Parameters:
        attributes (dict): Dictionary to update.
        lats (2D array): Latitude values.
        lons (2D array): Longitude values.
    """
    min_lat = float(np.min(lats))
    max_lat = float(np.max(lats))
    min_lon = float(np.min(lons))
    max_lon = float(np.max(lons))

    # Counter-clockwise starting from lower-right
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

def add_calculated_attributes(attributes):
    # calculate attributes from lat/lon and date columns (csv):
    # REF_LAT, REF_LON, first_date, last_date, history

    if all(key in attributes for key in ["LAT_ARRAY", "LON_ARRAY", "DATE_COLUMNS"]):
        lats = attributes.pop("LAT_ARRAY")
        lons = attributes.pop("LON_ARRAY")
        date_columns = attributes.pop("DATE_COLUMNS")

        attributes["REF_LAT"] = float(np.nanmean(lats))
        attributes["REF_LON"] = float(np.nanmean(lons))

        sorted_dates = sorted(date_columns)
        attributes["first_date"] = sorted_dates[0]
        attributes["last_date"] = sorted_dates[-1]
        attributes["history"] = datetime.now().strftime("%Y-%m-%d")
            
def read_from_csv_file(file_name):
    # read data from csv file to be done by Emirhan
    # the shared memory shm is confusing. it may also works without but be careful about returning or not returning shm.

    df = pd.read_csv(file_name)

    # dynamically detect latitude/longitude columns
    lat_candidates = ["Latitude", "Y", "ycoord"]
    lon_candidates = ["Longitude", "X", "xcoord"]

    lat_col = next((col for col in lat_candidates if col in df.columns), None)
    lon_col = next((col for col in lon_candidates if col in df.columns), None)

    if lat_col is None or lon_col is None:
        raise ValueError("Could not find latitude/longitude columns in the CSV. Supported names: 'Latitude', 'Y', 'ycoord' and 'Longitude', 'X', 'xcoord'.")

    lats = df[lat_col].values
    lons = df[lon_col].values

    # extract time-series data
    sarvey_time_cols = [col for col in df.columns if col.startswith("D") and col[1:].isdigit()]
    is_sarvey_format = bool(sarvey_time_cols)

    if is_sarvey_format:
        time_cols = sarvey_time_cols
        time_cols.sort()
        timeseries_data = df[time_cols].values / 1000  # SARvey
        dates = [col[1:] for col in time_cols]  # remove "D" prefix
    else:
        time_cols = [col for col in df.columns if col.isdigit()]
        timeseries_data = df[time_cols].values / 1000  # e.g. NOAA-TRE
        dates = time_cols

    # 3D time-series array (time, y, x)
    num_points = len(df)
    num_dates = len(time_cols)
    
    # reshape to 3D
    num_rows = int(np.sqrt(num_points))
    num_cols = int(np.ceil(num_points / num_rows))
    padded = np.full((num_cols * num_rows, num_dates), np.nan)
    padded[:num_points, :] = timeseries_data
    reshaped = padded.reshape((num_rows, num_cols, num_dates)).transpose(2, 0, 1)

    # decimal dates
    decimal_dates = [get_decimal_date(get_date(d)) for d in dates]
    timeseries_datasets = {d: reshaped[i, :, :] for i, d in enumerate(dates)}

    # create initial attributes
    attributes = {}
    attributes["PROJECT_NAME"] = "CSV_IMPORT"
    attributes["WIDTH"] = str(num_cols)
    attributes["LENGTH"] = str(num_rows)
    attributes["LAT_ARRAY"] = lats
    attributes["LON_ARRAY"] = lons
    attributes["DATE_COLUMNS"] = dates
    attributes["processing_type"] = "LOS_TIMESERIES"
    attributes["look_direction"] = "R"
    attributes["collection"] = "sarvey"

    # FA 4/2025: attribute to be included in *.csv file as sarvey2csv.py --mission S1 --flight-direction D --relative-orbit 128
    attributes["mission"] = "S1"
    attributes["flight_direction"] = "D"
    attributes["relative_orbit"] = 128

    add_calculated_attributes(attributes)   # FA 4/2025: need to make work for NOAA-TRE
    add_data_footprint_attribute(attributes, lats, lons)
    add_dummy_attribute(attributes, is_sarvey_format)  # Remove once needed_attributes have been reduced. FA 4/2025: this shoulkd not depend on sarvey or NOAA
    
    padded_lats = np.full(num_cols * num_rows, np.nan)
    padded_lats [:num_points] = lats
    lats_grid = padded_lats.reshape((num_rows, num_cols))

    padded_lons = np.full(num_cols * num_rows, np.nan)
    padded_lons[:num_points] = lons
    lons_grid = padded_lons.reshape((num_rows, num_cols))

    # point quality parameters
    dem_error = df['dem_error'].values
    dem = df['dem'].values
    coherence = df['coherence'].values
    omega = df['omega'].values
    st_consist = df['st_consist'].values

    quality_fields = {
        'dem_error': dem_error,
        'elevation': dem + dem_error,
        'coherence': coherence,
        'omega': omega,
        'st_consist': st_consist
    }

    quality_grids = {
        key: np.full(num_rows * num_cols, np.nan) for key in quality_fields
    }

    for key, values in quality_fields.items():
        quality_grids[key][:num_points] = values
        quality_grids[key] = quality_grids[key].reshape((num_rows, num_cols))

    # attributes["X_STEP"] = float(np.abs(lons_grid[0, 1] - lons_grid[0, 0]))
    # attributes["Y_STEP"] = float(np.abs(lats_grid[1, 0] - lats_grid[0, 0]))
    # attributes["X_FIRST"] = float(lons_grid[0, 0])
    # attributes["Y_FIRST"] = float(lats_grid[0, 0])

    folder_name = os.path.basename(file_name).split(".")[0]

    #shm none
    shm = None

    print(" Check: read_from_csv_file output")
    print(" - timeseries_datasets keys:", list(timeseries_datasets.keys())[:3], "...")
    print(" - sample slice shape:", timeseries_datasets[dates[0]].shape)
    print(" - lat_grid shape:", lats_grid.shape)
    print(" - lon_grid shape:", lons_grid.shape)
    print(" - Number of dates:", len(dates))
    print(" - Attributes:", attributes)

    return attributes, decimal_dates, timeseries_datasets, dates, folder_name, lats_grid, lons_grid, shm, quality_grids
# ---------------------------------------------------------------------------------------
# START OF EXECUTABLE
# ---------------------------------------------------------------------------------------
def main():
    parser = build_parser()
    parseArgs = parser.parse_args()
    file_name = parseArgs.file
    output_folder = parseArgs.outputDir

    try: # create path for output
        os.mkdir(output_folder)
    except:
        print(output_folder + " already exists")

    file_path = Path(file_name)

    # start clock to track how long conversion process takes
    start_time = time.perf_counter()
    
    if file_path.suffix.lower() == ".he5":
        attributes, decimal_dates, timeseries_datasets, dates, folder_name, lats, lons, shm = read_from_hdfeos5_file(file_name)
        quality_params = None
        # FA 4/2025: Initialize missing quality_params as zero arrays did not work. Problem in: [json.dumps(feature) for feature in points]
        # shape = lats.shape
        # quality_params = {
        #     'dem_error': np.zeros(shape, dtype=np.float32),
        #     'elevation': np.zeros(shape, dtype=np.float32),
        #     'coherence': np.zeros(shape, dtype=np.float32),
        #     'omega': np.zeros(shape, dtype=np.float32),
        #     'st_consist': np.zeros(shape, dtype=np.float32)
        # }
    elif file_path.suffix.lower() == ".csv":
        attributes, decimal_dates, timeseries_datasets, dates, folder_name, lats, lons, shm, quality_params = read_from_csv_file(file_name)
    else:
        raise FileNotFoundError(f"The file '{file_path}' does not exist or has not .he5 or .csv as extension.")   

    # read and convert the datasets, then write them into json files and insert into database
    convert_data(attributes, decimal_dates, timeseries_datasets, dates, output_folder, folder_name, lats, lons, quality_params, parseArgs.num_workers)
    del lats
    del lons
    if shm is not None:
        shm.close()
        shm.unlink()

    # run tippecanoe command to get mbtiles file
    os.chdir(os.path.abspath(output_folder))
    cmd = None
    if high_res_mode(attributes):
        cmd = "tippecanoe *.json -P -l chunk_1 -x d -pf -pk -o " + folder_name + ".mbtiles 2> tippecanoe_stderr.log"
    else:
        cmd = "tippecanoe *.json -P -l chunk_1 -x d -pf -pk -Bg -d9 -D12 -g12 -r0 -o " + folder_name + ".mbtiles 2> tippecanoe_stderr.log"

    print("Now running tippecanoe with command %s" % cmd)
    os.system(cmd)
    # ---------------------------------------------------------------------------------------
    # check how long it took to read h5 file data and create json files
    end_time =  time.perf_counter()
    print(("time elapsed: " + str(end_time - start_time)))
    return

# ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
