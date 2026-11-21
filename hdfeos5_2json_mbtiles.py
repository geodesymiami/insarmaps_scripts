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
import time
from datetime import date
import geocoder
import numpy as np

from mintpy.objects import HDFEOS
from mintpy.mask import mask_matrix
from mintpy.utils import utils as ut
import h5py

import points_bundle

# ex: hdfeos5_2json_mbtiles.py Alos_SM_73_2980_2990_20070107_20110420.he5 JSON

# This script takes a UNAVCO format timeseries h5 file and writes the dataset's points as arrays (points_bundle.py)
# and metadata.pickle, which json_mbtiles2insarmaps.py loads into the database and turns into the dataset's recolor
# store. The name stays because minsar calls the script by it.
# ---------------------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------------------
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
    "flight_direction", "last_frame", "post_processing_method", "min_baseline_perp"
    "unwrap_method", "relative_orbit", "beam_mode", "LENGTH", "max_baseline_perp",
    "X_FIRST", "atmos_correct_method", "last_date", "first_frame", "frame", "Y_STEP", "history",
    "scene_footprint", "data_footprint", "downloadUnavcoUrl", "referencePdfUrl", "areaName", "referenceText",
    "REF_LAT", "REF_LON", "CENTER_LINE_UTC", "insarmaps_download_flag", "mintpy.subset.lalo"
}

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

# ---------------------------------------------------------------------------------------
# write metadata.pickle, from which json_mbtiles2insarmaps.py makes the dataset's area row. folder_name == unavco_name
def write_metadata(attributes, decimal_dates, dates, json_path, folder_name, lats=None, lons=None):

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

    # dictionary to contain metadata needed by db to be written to a file
    # and then be read by json_mbtiles2insarmaps.py
    insarmapsMetadata = {}
    # calculate mid lat and long of dataset - then use google python lib to get country
    # technically don't need the else since we always use lats and lons arrays now
    if high_res_mode(attributes):
        num_rows, num_columns = lats.shape
        mid_long = float(lons[num_rows // 2][num_columns // 2])
        mid_lat = float(lats[num_rows // 2][num_columns // 2])
    else:
        mid_long = x_first + ((num_columns/2) * x_step)
        mid_lat = y_first + ((num_rows/2) * y_step)

    country = None
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
    attribute_keys = []
    attribute_values = []
    max_digit = max([len(key) for key in list(needed_attributes)] + [0])
    for k in attributes:
        v = attributes[k]
        if k in needed_attributes:
            print('{k:<{w}}     {v}'.format(k=k, w=max_digit, v=v))
            attribute_keys.append(k)
            attribute_values.append(v)

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
    parser = argparse.ArgumentParser(
        description='Convert a Unavco format H5 file for ingestion into insarmaps.',
        epilog="This program writes the dataset's points as arrays (points.json, p.npy, lon.npy, lat.npy, m.npy, "
               "d.npy) and metadata.pickle. json_mbtiles2insarmaps.py loads them into the database and builds the "
               "dataset's recolor store, from which the map draws it.")
    parser.add_argument("--num-workers", help="Not needed any more: the conversion runs in one process. Accepted for "
                        "the callers that still pass it.", required=False, default=1, type=int)
    required = parser.add_argument_group("required arguments")
    required.add_argument("file", help="unavco file to ingest")
    required.add_argument("outputDir", help="directory to place the points files and metadata.pickle")

    return parser

# ---------------------------------------------------------------------------------------
# START OF EXECUTABLE
# ---------------------------------------------------------------------------------------
def main():
    parser = build_parser()
    parseArgs = parser.parse_args()
    file_name = parseArgs.file
    output_folder = parseArgs.outputDir
    should_mask = True

    path_name_and_extension = os.path.basename(file_name).split(".")
    path_name = path_name_and_extension[0]
    # ---------------------------------------------------------------------------------------
    # start clock to track how long conversion process takes
    start_time = time.perf_counter()

    # read the displacement and the mask, and mask the displacement (masked cells become NaN)
    he_obj = HDFEOS(file_name)
    he_obj.open(print_msg=False)
    displacement_3d_matrix = he_obj.read(datasetName='displacement')
    mask = he_obj.read(datasetName='mask')
    if should_mask:
        print("Masking displacement")
        displacement_3d_matrix = mask_matrix(displacement_3d_matrix, mask)
    del mask

    dates = he_obj.dateList
    attributes = dict(he_obj.metadata)
    decimal_dates = [get_decimal_date(get_date(d)) for d in dates]

    path_list = path_name.split("/")
    folder_name = path_name.split("/")[len(path_list)-1]

    try: # create path for output
        os.mkdir(output_folder)
    except:
        print(output_folder + " already exists")

    # read lat and long. MintPy doesn't seem to support this yet, so we use the raw
    # h5 file object
    with h5py.File(he_obj.file, "r") as f:
        lats = np.array(f["HDFEOS"]["GRIDS"]["timeseries"]["geometry"]["latitude"])
        lons = np.array(f["HDFEOS"]["GRIDS"]["timeseries"]["geometry"]["longitude"])

    # the points: every cell with values at two dates or more, its velocity fitted over the dates it has
    num_points = points_bundle.write_grid_points(output_folder, displacement_3d_matrix, lats, lons, decimal_dates)
    print("points: %d of %d cells, %d dates" % (num_points, lats.size, len(dates)))
    del displacement_3d_matrix
    write_metadata(attributes, decimal_dates, dates, output_folder, folder_name, lats, lons)
    del lats
    del lons

    # ---------------------------------------------------------------------------------------
    # check how long it took to read h5 file data and write the points
    end_time =  time.perf_counter()
    print(("time elapsed: " + str(end_time - start_time)))
    return

# ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
