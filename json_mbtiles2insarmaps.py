#! /usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Alfredo Terrero, 2016                            #
############################################################


import io
import os
import sys
import argparse
import pickle
import struct
import time
import numpy

from add_attribute_insarmaps import (
    InsarDatabaseController,
    InsarDatasetController,
)

import points_bundle
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

# ---------------------------------------------------------------------------------------------- the point table

COPY_BYTES = 64 << 20      # binary COPY data per COPY statement, about

def copy_rows(p, lon, lat, m, d, attributes, types):
    """The points as binary COPY data (PostgreSQL's binary format) for the columns wkb_geometry, d, m, p and then the
    attribute columns, in p order: yields one complete COPY body (header, rows, trailer) per block of rows.
    wkb_geometry is an EWKB point with SRID 4326; d a float8 array with one element per date, NaN where missing. An
    attribute value that is missing (NaN; for an integer column any value that isn't finite) goes in as NULL, as the
    JSON null of an older step 1 did; nothing else is NULL."""
    P, N = d.shape
    names = list(attributes)
    base = [('nf', '>i2'), ('lg', '>i4'), ('bo', 'u1'), ('gt', '<u4'), ('srid', '<u4'), ('x', '<f8'), ('y', '<f8'),
            ('ld', '>i4'), ('ndim', '>i4'), ('flags', '>i4'), ('oid', '>u4'), ('n', '>i4'), ('lb', '>i4'),
            ('e', [('len', '>i4'), ('v', '>f8')], (N,)), ('lm', '>i4'), ('m', '>f8'), ('lp', '>i4'), ('p', '>i4')]

    def row_type(present):
        fields = list(base)
        for k, name in enumerate(names):
            fields.append(('la%d' % k, '>i4'))
            if present[k]:
                fields.append(('a%d' % k, '>i4' if types[name] == 'integer' else '>f8'))
        return numpy.dtype(fields)

    def encode(s0, s1, present):
        r = numpy.zeros(s1 - s0, dtype=row_type(present))
        r['nf'] = 4 + len(names)
        r['lg'] = 25                        # the EWKB point: byte order, type, SRID, x, y
        r['bo'] = 1                         # little endian
        r['gt'] = 1 | 0x20000000            # Point, with an SRID
        r['srid'] = 4326
        r['x'] = lon[s0:s1]
        r['y'] = lat[s0:s1]
        r['ld'] = 20 + 12 * N               # the array: 5 int32 of header, then N x (int32 length, float8)
        r['ndim'] = 1
        r['flags'] = 0                      # no NULL elements
        r['oid'] = 701                      # float8
        r['n'] = N
        r['lb'] = 1
        r['e']['len'] = 8
        r['e']['v'] = d[s0:s1]
        r['lm'] = 8
        r['m'] = m[s0:s1]
        r['lp'] = 4
        r['p'] = p[s0:s1]
        for k, name in enumerate(names):
            if present[k]:
                r['la%d' % k] = 4 if types[name] == 'integer' else 8
                r['a%d' % k] = attributes[name][s0:s1]
            else:
                r['la%d' % k] = -1              # NULL
        return r.tobytes()

    header = b'PGCOPY\n\xff\r\n\x00' + struct.pack('>ii', 0, 0)
    trailer = struct.pack('>h', -1)
    step = max(1, COPY_BYTES // row_type([True] * len(names)).itemsize)
    for c0 in range(0, P, step):
        c1 = min(c0 + step, P)
        if names:
            missing = numpy.column_stack([
                ~numpy.isfinite(attributes[nm][c0:c1]) if types[nm] == 'integer' else numpy.isnan(attributes[nm][c0:c1])
                for nm in names])
        else:
            missing = numpy.zeros((c1 - c0, 0), dtype=bool)
        if not missing.any():
            yield header + encode(c0, c1, [True] * len(names)) + trailer
            continue
        # rows with NULLs are shorter: encode each run of rows that have the same columns missing, in order
        patterns, pattern = numpy.unique(missing, axis=0, return_inverse=True)
        pattern = pattern.reshape(-1)
        starts = numpy.concatenate(([0], numpy.flatnonzero(numpy.diff(pattern)) + 1, [c1 - c0]))
        parts = [encode(c0 + s0, c0 + s1, ~patterns[pattern[s0]]) for s0, s1 in zip(starts[:-1], starts[1:])]
        yield header + b''.join(parts) + trailer

def load_table(con, area_id, p, lon, lat, m, d, attributes, types):
    """Creates the dataset's point table as ogr2ogr made it (ogc_fid, d, m, p, the attribute columns, wkb_geometry;
    the primary key and the spatial index) and loads the points, in p order, with binary COPY, in one transaction.
    The table's name is the area id, and the attribute names passed points_bundle's check (letters, digits, _)."""
    table = '"%s"' % int(area_id)
    extra = "".join(', "%s" %s' % (name, types[name]) for name in attributes)
    columns = "wkb_geometry, d, m, p" + "".join(', "%s"' % name for name in attributes)
    cur = con.cursor()
    cur.execute('CREATE TABLE %s (ogc_fid serial PRIMARY KEY, d double precision[], m double precision, '  # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
                'p integer%s, wkb_geometry geometry(Point,4326))' % (table, extra))
    for body in copy_rows(p, lon, lat, m, d, attributes, types):
        cur.copy_expert('COPY %s (%s) FROM STDIN (FORMAT binary)' % (table, columns), io.BytesIO(body))
    cur.execute('CREATE INDEX "%s_wkb_geometry_geom_idx" ON %s USING gist (wkb_geometry)' % (int(area_id), table))  # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
    con.commit()

def mark_clustered(con, area_id):
    """The rows went in in p order: marking the p index as the table's clustering index is what CLUSTER would do,
    without rewriting the table."""
    cur = con.cursor()
    cur.execute('ALTER TABLE "%d" CLUSTER ON "%d_p_idx"' % (int(area_id), int(area_id)))  # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
    con.commit()

def upload_points(folder_path, store_root=rsi.DEFAULT_ROOT, store_tools=rsi.DEFAULT_TOOLS,
                  min_free_gb=rsi.DEFAULT_MIN_FREE_GB, drop_rate=None, base_zoom=None):
    """Loads the dataset of folder_path (metadata.pickle and the points step 1 wrote, points_bundle.py) into the
    database, replacing a dataset of the same name, then builds and checks its recolor store in store_root. Exits
    with code 1, saying why, when the folder holds no points (the JSON chunk files of an older step 1 are not read any
    more), when the table doesn't hold every point (the partly loaded dataset is removed), or when the store can't be
    built or fails its check (the dataset stays, without a store). drop_rate and base_zoom, when given, replace the
    dataset's settings of the store's drop rule (insarmaps docs/recolor-store.md section 5)."""
    global dbUsername, dbPassword, dbHost
    t_start = time.time()
    timing = {}
    if not points_bundle.has_points(folder_path):
        if rsi.json_chunk_files(folder_path):
            fail("%s holds the JSON chunk files of an older step 1, which the upload doesn't read any more. On the "
                 "machine that ran step 1, update both minsar and insarmaps_scripts (git pull in $MINSAR_HOME and in "
                 "$MINSAR_HOME/tools/insarmaps_scripts; a minsar from before 2026-11-21 still expects an .mbtiles "
                 "file), then run step 1 again" % folder_path)
        fail("no points in %s (points.json): run step 1 first" % folder_path)
    metadata = read_metadata(folder_path)
    area_name = metadata["area"]
    try:
        manifest, p, lon, lat, m, d, attributes = points_bundle.read_points(folder_path)
    except (points_bundle.PointsError, OSError, ValueError, KeyError) as e:
        fail("the points in %s can't be read: %s" % (folder_path, e))
    num_dates = rsi.dates_in_metadata(metadata)
    if manifest["dates"] != num_dates:
        fail("the points in %s have %d dates, its metadata.pickle %d" % (folder_path, manifest["dates"], num_dates))
    features = manifest["points"]
    num_extra = len(attributes)
    print("%s: %d points, %d dates%s" % (area_name, features, num_dates,
                                         (", attributes " + ", ".join(attributes)) if attributes else ""))
    # everything the recolor store needs is checked before anything changes
    try:
        rsi.check_root(store_root, store_tools)
    except rsi.StoreError as e:
        fail(str(e))

    attributesController = InsarDatabaseController(dbUsername, dbPassword, dbHost, 'pgis')
    attributesController.connect()
    old_id = attributesController.get_dataset_id(area_name)
    try:
        rsi.check_root_belongs(store_root, rsi.area_ids(attributesController.cursor))
        rsi.check_root_identity(store_root, rsi.database_identity(attributesController.cursor))
        attributesController.con.rollback()
        # the store of the dataset this upload replaces is deleted first: its space counts as free
        old_store = rsi.store_bytes(os.path.join(store_root, str(old_id))) if old_id != -1 else 0
        rsi.check_space(store_root, rsi.estimate_store_bytes(features, num_dates, num_extra) - old_store,
                        min_free_gb)
    except rsi.StoreError as e:
        fail(str(e))
    if old_id == -1:
        print("No dataset %s yet" % area_name)
    else:
        print("Removing the dataset's previous upload (area id %d)" % old_id)
        try:
            attributesController.remove_dataset(area_name)
        except Exception as e:
            fail("removing the previous upload of %s (area id %d) failed, so nothing was uploaded: %s"
                 % (area_name, old_id, e))
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
    attributesController.connect()
    area_id = str(attributesController.get_dataset_id(area_name))
    print("Area id: " + area_id)

    t0 = time.time()
    try:
        load_table(attributesController.con, area_id, p, lon, lat, m, d, attributes, manifest["attribute_types"])
    except Exception as e:
        attributesController.con.rollback()
        try:
            attributesController.remove_dataset(area_name)
            what = "removed the dataset"
        except Exception as e2:
            what = "removing the dataset failed too: %s" % e2
        attributesController.close()
        fail("loading the points into table %s failed: %s; %s" % (area_id, e, what))
    timing["load"] = time.time() - t0

    # a load that silently lost points fails the ingest; the m extents come from the same pass over the table
    t0 = time.time()
    rows, m_min, m_max = rsi.rows_and_m_extents(attributesController.con, area_id)
    timing["rows_and_m_extents"] = time.time() - t0
    if rows != features:
        try:
            attributesController.remove_dataset(area_name)
            what = "removed the partly loaded dataset"
        except Exception as e:
            what = "removing the partly loaded dataset failed too: %s" % e
        attributesController.close()
        fail("the table got %d rows of the %d points; %s" % (rows, features, what))
    print("Loaded %d points; m_min %s, m_max %s" % (rows, m_min, m_max))
    print("Indexing table")
    t0 = time.time()
    attributesController.index_table_on(area_id, "p", None)
    mark_clustered(attributesController.con, area_id)
    timing["p_index"] = time.time() - t0
    attributesController.close()

    print("Building the recolor store in " + store_root)
    try:
        store = rsi.build_and_check_store(int(area_id), area_name, rows, num_dates, num_extra, folder_path,
                                          rsi.dsn(dbHost, dbUsername), dbPassword, store_root, store_tools,
                                          min_free_gb, drop_rate=drop_rate, base_zoom=base_zoom)
    except rsi.StoreError as e:
        fail("%s. The dataset is loaded (area id %s) but the map can't show it without its store: after fixing the "
             "cause, upload the dataset again, or build its store with tools/recolor_store/migrate.py --only %s of "
             "the insarmaps repository." % (e, area_id, area_id))
    timing["store"] = store["build_s"]
    timing["check"] = store["check_s"]
    levels = store.get("levels_source") or {}
    print("Recolor store %s/%s: %d points x %d dates, %.1f MB, drop rate %s and base zoom %s, checked against the "
          "table (largest velocity difference %.2g cm/yr)" % (store_root, area_id, store["points"], store["dates"],
                                                              store["bytes"] / 1e6, levels.get("rate"),
                                                              levels.get("base_zoom"),
                                                              store["max_abs_diff_cm_per_yr"] or 0.0))
    print("Seconds: " + ", ".join("%s %.1f" % (k, v) for k, v in timing.items()) +
          ", total %.1f" % (time.time() - t_start))

def remove_dataset_and_store(dbController, unavco_name, store_root):
    """Removes a dataset from the database (its point table and its rows in area, extra_attributes and
    plot_attributes), then its recolor store. dbController: a connected InsarDatabaseController."""
    rsi.check_root_belongs(store_root, rsi.area_ids(dbController.cursor))
    if os.path.isdir(store_root):
        rsi.check_root_identity(store_root, rsi.database_identity(dbController.cursor))
    area_id = dbController.get_dataset_id(unavco_name)
    dbController.remove_dataset(unavco_name)
    if area_id != -1 and rsi.remove_store(store_root, area_id, unavco_name):
        print("Removed its recolor store (area id %d)" % area_id)

def drop_rate_value(text):
    value = float(text)
    if not value >= 1:
        raise argparse.ArgumentTypeError("must be 1 or more (1 drops no point): %r" % text)
    return value

def base_zoom_value(text):
    value = int(text)
    if not 0 <= value <= 14:
        raise argparse.ArgumentTypeError("must be 0..14: %r" % text)
    return value

def non_negative(text):
    value = float(text)
    if not value >= 0:
        raise argparse.ArgumentTypeError("must be a number of GB, 0 or more: %r" % text)
    return value

def build_parser():
    dbHost = "insarmaps.rsmas.miami.edu"
    parser = argparse.ArgumentParser(description='Upload a dataset (the metadata.pickle and points that '
        'insarmaps_ingester.py wrote) into the insarmaps database and build its recolor store, from which the map '
        'draws it; or remove or list datasets.')
    parser.add_argument("--num-workers", help="Not needed any more: the points load in one stream. Accepted for the "
        "callers that still pass it.", required=False, default=1, type=int)
    parser.add_argument("--json_folder", help="the folder step 1 wrote (metadata.pickle, points.json and its .npy "
        "files)", required=False)
    parser.add_argument("json_folder_positional", help="the folder step 1 wrote, as a positional argument", nargs="?")
    parser.add_argument("-U", "--server_user", required=False, 
        help="username for the insarmaps server (the machine where the tileserver and http server reside)")
    parser.add_argument("--remove",
        help="UNAVCO name of dataset to remove from insarmaps website", required=False)
    parser.add_argument("--list",
        help="List datasets currently on the insarmaps website", required=False, action="store_true")
    parser.add_argument("-P", "--server_password", required=False,
        help="password for the insarmaps server (the machine where the tileserver and http server reside)")
    parser.add_argument("--recolor-store-root", default=rsi.DEFAULT_ROOT,
        help="folder of the datasets' recolor stores, where the insarmaps app reads them (default: "
             "$INSARMAPS_RECOLOR_STORE_DIR, else /data/insarmaps2/recolor_store; now %(default)s)")
    parser.add_argument("--recolor-tools", default=rsi.DEFAULT_TOOLS,
        help="the insarmaps repository's tools/recolor_store (default: beside this repository, %(default)s)")
    parser.add_argument("--recolor-store-min-free-gb", type=non_negative, default=rsi.DEFAULT_MIN_FREE_GB,
        help="free space to keep on the store folder's file system (default %(default)s)")
    parser.add_argument("--drop-rate", type=drop_rate_value,
        help="how fast points drop out below the base zoom, as tippecanoe's -r (default 2.5; 1 drops none)")
    parser.add_argument("--base-zoom", type=base_zoom_value,
        help="the zoom from which the map shows every point, as tippecanoe's -B (default 12 for a geocoded "
             "dataset, else 14)")

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
    if parseArgs.json_folder or parseArgs.json_folder_positional:
        print("Uploading points...")
        upload_points(parseArgs.json_folder or parseArgs.json_folder_positional, *store_args,
                      drop_rate=parseArgs.drop_rate, base_zoom=parseArgs.base_zoom)

    dbController = InsarDatasetController(dbUsername,
                                         dbPassword,
                                         dbHost,
                                         'pgis',
                                         parseArgs.server_user,
                                         parseArgs.server_password)

    removal_failed = False
    if parseArgs.remove:
        if not parseArgs.server_user or not parseArgs.server_password:
            sys.stderr.write("Error: credentials for the insarmaps server not provided\n")
            removal_failed = True
        else:
            print("Trying to remove %s" % (parseArgs.remove))
            dbController.connect()
            try:
                remove_dataset_and_store(dbController, parseArgs.remove, parseArgs.recolor_store_root)
                dbController.remove_mbtiles(parseArgs.remove + ".mbtiles")
                print("Successfully removed %s" % (parseArgs.remove))
            except Exception as e:
                sys.stderr.write("Error: %s\n" % e)
                removal_failed = True

            dbController.close()


    if parseArgs.list:
        dbController.connect()
        dbController.list_dataset_names()
        dbController.close()

    if removal_failed:
        sys.exit(1)

if __name__ == '__main__':
    main()
