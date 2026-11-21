#! /usr/bin/env python3
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
import numpy

from add_attribute_insarmaps import (
    InsarDatabaseController,
    InsarDatasetController,
)
from queue import Queue
from threading import Thread

import recolor_store_ingest as rsi


dbUsername = "INSERT"
dbPassword = "INSERT"
dbHost = "INSERT"

def fail(message):
    sys.stderr.write("Error: " + message + "\n")
    sys.exit(1)

def read_metadata(json_path):
    with open(json_path + "/metadata.pickle", "rb") as file:
        return pickle.load(file)

# use pickle file to get unavco_name of dataset
def get_unavco_name(json_path):
    return read_metadata(json_path)["area"]

def upload_insarmaps_metadata(fileName):
    insarmapsMetadata = None

    with open(fileName, "rb") as file:
        insarmapsMetadata = pickle.load(file)

    area = insarmapsMetadata["area"]
    project_name = insarmapsMetadata["project_name"]
    mid_long = insarmapsMetadata["mid_long"]
    mid_lat = insarmapsMetadata["mid_lat"]
    country = insarmapsMetadata["country"]
    region = insarmapsMetadata["region"]
    chunk_num = insarmapsMetadata["chunk_num"]
    attribute_keys = insarmapsMetadata["attribute_keys"]
    attribute_values = insarmapsMetadata["attribute_values"]
    string_dates_sql = insarmapsMetadata["string_dates_sql"]
    decimal_dates_sql = insarmapsMetadata["decimal_dates_sql"]
    attributes = insarmapsMetadata["attributes"]
    needed_attributes = insarmapsMetadata["needed_attributes"]

    attributesController = InsarDatabaseController(dbUsername, dbPassword, dbHost, 'pgis')
    attributesController.connect()
    attributesController.create_area_table_if_not_exists()
    attributesController.insert_dataset_into_area_table(area, project_name,
    mid_long, mid_lat, country, region, chunk_num, attribute_keys,
    attribute_values, string_dates_sql, decimal_dates_sql)

    # put in attributes into standalone attributes table
    for k in attributes:
        v = attributes[k]
        # convert numpy.int64 objects to native python types otherwise psycopg2 can't upload to db
        # needed because we use pickle.HIGHEST_PROTOCOL to serialize with pickle now
        if isinstance(v, numpy.int64):
            v = v.item()

        if k in needed_attributes:
            attributesController.add_attribute(area, k, v)
        elif k == "plotAttributes":
            attributesController.add_plot_attribute(area, k, v)

    attributesController.close()

def worker(queue):
    # """Process files from the queue."""
    for args in iter(queue.get, None):
        try:
            command = args[0]
            file = args[1]

            res = os.system(command)

            if res != 0:
                msg = "Error inserting into the database."
                msg += " This is most often due to running out of Memory (RAM)"
                msg += ", or incorrect database credentials... quitting"
                sys.stderr.write(msg)
                os._exit(-1)

            print("Inserted " + file + " to db")

        except Exception as e: # catch exceptions to avoid exiting the
                               # thread prematurely
           print('%r failed: %s' % (args, e,))

def upload_json(folder_path, num_workers=1, store_root=rsi.DEFAULT_ROOT, store_tools=rsi.DEFAULT_TOOLS,
                min_free_gb=rsi.DEFAULT_MIN_FREE_GB):
    """Loads the dataset of folder_path (metadata.pickle, chunk_*.json) into the database, replacing a dataset of
    the same name, then builds and checks its recolor store in store_root. Exits with code 1, saying why, when the
    table doesn't hold every feature (the partly loaded dataset is removed) or the store can't be built or fails its
    check (the dataset stays, without a store)."""
    global dbUsername, dbPassword, dbHost
    t_start = time.time()
    timing = {}
    metadata = read_metadata(folder_path)
    area_name = metadata["area"]
    chunk_files = rsi.json_chunk_files(folder_path)
    if not chunk_files:
        fail("no chunk_*.json files in " + folder_path)
    features = rsi.count_features(folder_path, chunk_files)
    num_dates = rsi.dates_in_metadata(metadata)
    num_extra = rsi.extra_properties(folder_path, chunk_files)
    print("%s: %d features, %d dates, in %d chunk files" % (area_name, features, num_dates, len(chunk_files)))
    # everything the recolor store needs is checked before the database is touched
    try:
        rsi.check_root(store_root, store_tools)
        rsi.check_space(store_root, rsi.estimate_store_bytes(features, num_dates, num_extra), min_free_gb)
    except rsi.StoreError as e:
        fail(str(e))

    attributesController = InsarDatabaseController(dbUsername, dbPassword, dbHost, 'pgis')
    attributesController.connect()
    try:
        rsi.check_root_belongs(store_root, rsi.area_ids(attributesController.cursor))
    except rsi.StoreError as e:
        fail(str(e))
    print("Clearing old dataset, if it is there")
    old_id = attributesController.get_dataset_id(area_name)
    try:
        attributesController.remove_dataset(area_name)
    except Exception as e:
        print(str(e))
        old_id = -1

    attributesController.close()
    # its store too: the new upload gets a new area id
    if old_id != -1:
        try:
            if rsi.remove_store(store_root, old_id, area_name):
                print("Removed the recolor store of the previous upload (area id %d)" % old_id)
        except (rsi.StoreError, OSError) as e:
            sys.stderr.write("Warning: the recolor store of the previous upload (area id %d) stays: %s\n" % (old_id, e))

    # uploading metadata for area. this creates entry into area table.
    # we need this entry to get the db to generate an id for area which
    # we use to name the corresponding table for the dataset
    upload_insarmaps_metadata(folder_path + "/metadata.pickle")
    # create index
    print("Creating index on " + area_name)
    attributesController.connect()
    area_id = str(attributesController.get_dataset_id(area_name))
    attributesController.close()
    print("Area id: " + area_id)
    firstJsonFile = True
    t0 = time.time()

    q = Queue()
    threads = [Thread(target=worker, args=(q,)) for _ in range(num_workers)]
    for t in threads:
        t.daemon = True # threads die if the program dies
        t.start()

    for file in chunk_files:
        # insert json file to pgsql using ogr2ogr
        command = 'ogr2ogr -append -f "PostgreSQL" PG:"dbname=pgis ' + \
                  ' host=' + dbHost + ' user=' + dbUsername + ' password=' + dbPassword + \
                  '" --config PG_USE_COPY YES -nln ' + area_id + ' ' + folder_path + '/' + file
        # only provide layer creation options if this is the first file
        if firstJsonFile:
            command = 'ogr2ogr -lco LAUNDER=NO -append -f "PostgreSQL" PG:"dbname=pgis ' + \
                      ' host=' + dbHost + ' user=' + dbUsername + ' password=' + dbPassword + \
                      '" --config PG_USE_COPY YES -nln ' + area_id + ' ' + folder_path + '/' + file
            firstJsonFile = False
            # upload first one with layer creation options prior
            # to spawning processes to upload the rest
            res = os.system(command)
            print("Inserted " + file + " to db")

            if res != 0:
                msg = "Error inserting into the database."
                msg += " This is most often due to running out of Memory (RAM)"
                msg += ", or incorrect database credentials... quitting"
                sys.stderr.write(msg)
                os._exit(-1)
        else:
            q.put_nowait([command, file])

    for _ in threads: q.put_nowait(None) # signal no more files
    for t in threads: t.join() # wait for completion
    timing["load"] = time.time() - t0

    # a load that silently lost points fails the ingest; the m extents come from the same pass over the table
    t0 = time.time()
    attributesController.connect()
    rows, m_min, m_max = rsi.rows_and_m_extents(attributesController.con, area_id)
    timing["rows_and_m_extents"] = time.time() - t0
    if rows != features:
        try:
            attributesController.remove_dataset(area_name)
            what = "removed the partly loaded dataset"
        except Exception as e:
            what = "removing the partly loaded dataset failed too: %s" % e
        attributesController.close()
        fail("the table got %d rows of the %d features in the chunk files; %s" % (rows, features, what))
    print("Loaded %d points; m_min %s, m_max %s" % (rows, m_min, m_max))
    print("Indexing table")
    t0 = time.time()
    attributesController.index_table_on(area_id, "p", None)
    timing["p_index"] = time.time() - t0
    print("Clustering table")
    t0 = time.time()
    attributesController.cluster_table_using(area_id, area_id + "_p_idx")
    timing["cluster"] = time.time() - t0
    attributesController.close()

    print("Building the recolor store in " + store_root)
    try:
        store = rsi.build_and_check_store(int(area_id), area_name, rows, num_dates, num_extra,
                                          rsi.dsn(dbHost, dbUsername), dbPassword, store_root, store_tools,
                                          min_free_gb)
    except rsi.StoreError as e:
        fail("%s. The dataset is loaded (area id %s) but the map can't show it without its store: after fixing the "
             "cause, build it with tools/recolor_store/migrate.py --only %s of the insarmaps repository, or upload "
             "the dataset again." % (e, area_id, area_id))
    timing["store"] = store["build_s"]
    timing["check"] = store["check_s"]
    print("Recolor store %s/%s: %d points x %d dates, %.1f MB, checked against the table (largest velocity "
          "difference %.2g cm/yr)" % (store_root, area_id, store["points"], store["dates"], store["bytes"] / 1e6,
                                      store["max_abs_diff_cm_per_yr"] or 0.0))
    print("Seconds: " + ", ".join("%s %.1f" % (k, v) for k, v in timing.items()) +
          ", total %.1f" % (time.time() - t_start))

def remove_dataset_and_store(dbController, unavco_name, store_root):
    """Removes a dataset from the database (its point table and its rows in area, extra_attributes and
    plot_attributes), then its recolor store. dbController: a connected InsarDatabaseController."""
    rsi.check_root_belongs(store_root, rsi.area_ids(dbController.cursor))
    area_id = dbController.get_dataset_id(unavco_name)
    dbController.remove_dataset(unavco_name)
    if area_id != -1 and rsi.remove_store(store_root, area_id, unavco_name):
        print("Removed its recolor store (area id %d)" % area_id)

def build_parser():
    dbHost = "insarmaps.rsmas.miami.edu"
    parser = argparse.ArgumentParser(description='Upload a dataset (the metadata.pickle and chunk_*.json that '
        'hdfeos5_2json_mbtiles.py or hdfeos5_or_csv_2json_mbtiles.py wrote) into the insarmaps database and build '
        'its recolor store, from which the map draws it; or remove or list datasets.')
    parser.add_argument("--num-workers", help="Number of simultaneous processes to run for ingest.", required=False, default=1, type=int)
    parser.add_argument("--json_folder", help="folder containing json to upload.", required=False)
    parser.add_argument("json_folder_positional", help="folder containing json to upload.", nargs="?")
    parser.add_argument("-U", "--server_user", required=False, 
        help="username for the insarmaps server (the machine where the tileserver and http server reside)")
    parser.add_argument("--remove",
        help="UNAVCO name of dataset to remove from insarmaps website", required=False)
    parser.add_argument("--list",
        help="List datasets currently on the insarmaps website", required=False, action="store_true")
    parser.add_argument("-P", "--server_password", required=False,
        help="password for the insarmaps server (the machine where the tileserver and http server reside)")
    parser.add_argument("--mbtiles_file", help="mbtiles file to upload", required=False)
    parser.add_argument("mbtiles_file_positional",
        help="mbtiles file to upload, as a positional argument", nargs="?")
    parser.add_argument("--recolor-store-root", default=rsi.DEFAULT_ROOT,
        help="folder of the datasets' recolor stores, where the insarmaps app reads them (default: "
             "$INSARMAPS_RECOLOR_STORE_DIR, else /data/insarmaps2/recolor_store; now %(default)s)")
    parser.add_argument("--recolor-tools", default=rsi.DEFAULT_TOOLS,
        help="the insarmaps repository's tools/recolor_store (default: beside this repository, %(default)s)")
    parser.add_argument("--recolor-store-min-free-gb", type=float, default=rsi.DEFAULT_MIN_FREE_GB,
        help="free space to keep on the store folder's file system (default %(default)s)")

    required = parser.add_argument_group("required arguments")
    required.add_argument("-u", "--user", help="username for the insarmaps database", required=True)
    required.add_argument("-p", "--password", help="password for the insarmaps database", required=True)
    required.add_argument("--host", default=dbHost, help="postgres DB URL for insarmaps database", required=True)

    return parser

def main():
    global dbUsername, dbPassword, dbHost
    sys.stdout.reconfigure(line_buffering=True)   # in the ingest's logs, messages stay in order with the errors
    parser = build_parser()
    parseArgs = parser.parse_args()
    dbUsername = parseArgs.user
    dbPassword = parseArgs.password
    dbHost = parseArgs.host

    store_args = (parseArgs.recolor_store_root, parseArgs.recolor_tools, parseArgs.recolor_store_min_free_gb)
    if parseArgs.json_folder:
        print("Uploading json chunks...")
        upload_json(parseArgs.json_folder, parseArgs.num_workers, *store_args)
    elif parseArgs.json_folder_positional:
        print("Uploading json chunks....")
        upload_json(parseArgs.json_folder_positional, parseArgs.num_workers, *store_args)

    dbController = InsarDatasetController(dbUsername,
                                         dbPassword,
                                         dbHost,
                                         'pgis',
                                         parseArgs.server_user,
                                         parseArgs.server_password)

    if parseArgs.mbtiles_file or parseArgs.mbtiles_file_positional:
        if not parseArgs.server_user or not parseArgs.server_password:
            sys.stderr.write("Error: credentials for the insarmaps server not provided")
        elif parseArgs.mbtiles_file:
            print("Uploading mbtiles...")
            if os.path.isfile(parseArgs.mbtiles_file):
                dbController.upload_mbtiles(parseArgs.mbtiles_file)
            else:
                print(parseArgs.mbtiles_file + " doesn't exist")
        else:
            print("Uploading mbtiles....")
            if os.path.isfile(parseArgs.mbtiles_file_positional):
                dbController.upload_mbtiles(parseArgs.mbtiles_file_positional)
            else:
                print(parseArgs.mbtiles_file_positional + " doesn't exist")

    if parseArgs.remove:
        if not parseArgs.server_user or not parseArgs.server_password:
            sys.stderr.write("Error: credentials for the insarmaps server not provided")
        else:
            print("Trying to remove %s" % (parseArgs.remove))
            dbController.connect()
            try:
                remove_dataset_and_store(dbController, parseArgs.remove, parseArgs.recolor_store_root)
                dbController.remove_mbtiles(parseArgs.remove + ".mbtiles")
                print("Successfully removed %s" % (parseArgs.remove))
            except Exception as e:
                print(str(e))

            dbController.close()


    if parseArgs.list:
        dbController.connect()
        dbController.list_dataset_names()
        dbController.close()

if __name__ == '__main__':
    main()
