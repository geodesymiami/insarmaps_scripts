#!/usr/bin/env python3
"""A dataset's points as arrays: what step 1 hands to step 2, in place of JSON chunk files.

Step 1 (hdfeos5_2json_mbtiles.py, hdfeos5_or_csv_2json_mbtiles.py) writes the folder. Step 2
(json_mbtiles2insarmaps.py) loads the point table from it with binary COPY, and the insarmaps tool
tools/recolor_store/convert_from_points.py builds the dataset's recolor store from it. Beside metadata.pickle:

    points.json       {"format": "insarmaps-points", "version": 1, "points": P, "dates": N,
                       "d_dtype": "float32" or "float64", "attributes": [NAME, ...],
                       "attribute_types": {NAME: "integer" or "double precision"}}
    p.npy             int32 (P,): the point numbers, strictly increasing (the table's p)
    lon.npy, lat.npy  float64 (P,): the position, in degrees (EPSG:4326)
    m.npy             float64 (P,): the velocity, m/yr: the least-squares slope over the dates with a value
    d.npy             float32 or float64 (P, N): the displacement, m, one value per date; NaN where missing
    attr_NAME.npy     float64 (P,): optional per-point columns (a CSV's quality columns), one file each, in the
                      order of "attributes"; "attribute_types" gives each one's column type in the point table
                      (integer for a column of whole numbers that fit int32, as ogr2ogr typed them)

points.json is written last: a folder without it is incomplete.

Missing values (decided 2026-11-21): a grid cell or CSV row becomes a point when it has values at two dates or
more; its m is fitted over the dates it has; a missing value stays NaN in d.
"""
import json
import os
import re

import numpy as np

FORMAT = "insarmaps-points"
VERSION = 1
MANIFEST = "points.json"
BLOCK = 20000          # gridded inputs number their cells in blocks of 20,000, as step 1 always has
FIT_CHUNK = 50000      # points per velocity fit and per copy into d.npy
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")   # attribute names: the recolor store's rule
ARRAYS = ("p", "lon", "lat", "m", "d")


class PointsError(Exception):
    pass


def fit_velocities(t, d):
    """For each row of d (P, N), the least-squares slope against t (N,) over the row's finite values; NaN for a row
    with fewer than two. Rows without a gap are solved together by np.linalg.lstsq. Step 1's JSON solved each point
    on its own; that differs in the last bit only (at most 1.7e-17 m/yr on 77,842 points of a 212-date dataset), and
    solving point by point would cost about 30 us a point."""
    t = np.asarray(t, dtype=np.float64)
    a = np.vstack([t, np.ones(len(t))]).T
    m = np.full(d.shape[0], np.nan)
    for c0 in range(0, d.shape[0], FIT_CHUNK):
        rows = np.asarray(d[c0:c0 + FIT_CHUNK], dtype=np.float64)
        finite = np.isfinite(rows)
        full = finite.all(axis=1)
        out = m[c0:c0 + FIT_CHUNK]                  # a view: writes go to m
        if full.any():
            out[full] = np.linalg.lstsq(a, rows[full].T, rcond=None)[0][0]
        for i in np.flatnonzero(~full):
            f = finite[i]
            if f.sum() >= 2:
                out[i] = np.linalg.lstsq(a[f], rows[i, f], rcond=None)[0][0]
    return m


def grid_points(disp, block=BLOCK, numbering="blocks"):
    """The cells of a grid (disp: N dates x rows x cols, NaN where masked or missing) that become points: their flat
    indices (row * cols + col, increasing) and their p. A cell is a point when it has values at two dates or more.
    numbering "blocks" (hdfeos5_2json_mbtiles.py): p counts the points from the first cell of each block of `block`
    cells; "flat" (hdfeos5_or_csv_2json_mbtiles.py): p is the flat index. Either way it is the number that script's
    JSON gave the point, for every dataset without gaps."""
    n_dates = disp.shape[0]
    values = np.zeros(disp.shape[1] * disp.shape[2], dtype=np.int32)
    flat = disp.reshape(n_dates, -1)
    for k in range(n_dates):                         # one date at a time: no N x cells boolean array
        values += np.isfinite(flat[k])
    idx = np.flatnonzero(values >= 2)
    if numbering == "flat":
        p = idx
    elif numbering == "blocks":
        blocks = idx // block
        rank = np.arange(len(idx)) - np.searchsorted(blocks, blocks, side="left")
        p = blocks * block + rank
    else:
        raise ValueError("numbering must be 'blocks' or 'flat', not %r" % numbering)
    if len(p) and p.max() > np.iinfo(np.int32).max:
        raise PointsError("point numbers beyond int32: %d cells" % (disp.shape[1] * disp.shape[2]))
    return idx, p.astype(np.int32)


def write_grid_points(folder, disp, lats, lons, decimal_dates, numbering="blocks"):
    """Writes the points of a gridded dataset to folder: disp (N, rows, cols) as read from the file, with NaN where
    masked or missing; lats, lons (rows, cols). d keeps disp's dtype (float32 for HDF-EOS5). Returns P."""
    idx, p = grid_points(disp, numbering=numbering)
    if len(idx) == 0:
        raise PointsError("no cell has values at two dates or more")
    flat = disp.reshape(disp.shape[0], -1)
    lon = np.asarray(lons, dtype=np.float64).reshape(-1)[idx]
    lat = np.asarray(lats, dtype=np.float64).reshape(-1)[idx]
    os.makedirs(folder, exist_ok=True)
    _clear(folder)
    d = np.lib.format.open_memmap(os.path.join(folder, "d.npy"), mode="w+", dtype=disp.dtype,
                                  shape=(len(idx), disp.shape[0]))
    for c0 in range(0, len(idx), FIT_CHUNK):
        d[c0:c0 + FIT_CHUNK] = flat[:, idx[c0:c0 + FIT_CHUNK]].T
    m = fit_velocities(decimal_dates, d)
    d.flush()
    _finish(folder, p, lon, lat, m, d, {})
    del d
    return len(p)


def write_points(folder, p, lon, lat, m, d, attributes=None, attribute_types=None):
    """Writes points held in memory (the CSV path): p (P,) integers, lon, lat, m (P,), d (P, N) float32 or float64,
    attributes {NAME: (P,)} in the table's column order, NaN where missing, or None; attribute_types {NAME:
    "integer" or "double precision"} for those whose type isn't column_type()'s."""
    os.makedirs(folder, exist_ok=True)
    _clear(folder)
    d = np.asarray(d)
    if d.dtype not in (np.float32, np.float64):
        d = d.astype(np.float64)
    np.save(os.path.join(folder, "d.npy"), d)
    _finish(folder, np.asarray(p), np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64),
            np.asarray(m, dtype=np.float64), d, attributes or {}, attribute_types or {})
    return len(p)


def _clear(folder):
    """Removes an earlier step 1's output from folder: points files and JSON chunks (metadata.pickle is rewritten)."""
    for name in os.listdir(folder):
        if (name == MANIFEST or (name.endswith(".npy") and (name[:-4] in ARRAYS or name.startswith("attr_")))
                or (name.startswith("chunk_") and name.endswith(".json"))):
            os.remove(os.path.join(folder, name))


def _finish(folder, p, lon, lat, m, d, attributes, attribute_types=None):
    p = np.asarray(p)
    if p.dtype.kind not in "iu" or (len(p) and (p.min() < 0 or p.max() > np.iinfo(np.int32).max)):
        raise PointsError("p must be non-negative integers that fit int32")
    p = p.astype(np.int32)
    attributes = {name: np.asarray(values) for name, values in attributes.items()}
    manifest = {"format": FORMAT, "version": VERSION, "points": int(len(p)), "dates": int(d.shape[1]),
                "d_dtype": str(d.dtype), "attributes": list(attributes),
                "attribute_types": {name: (attribute_types or {}).get(name) or column_type(values)
                                    for name, values in attributes.items()}}
    check(manifest, p, lon, lat, m, d, attributes)
    for name, values in (("p", p), ("lon", lon), ("lat", lat), ("m", m)):
        np.save(os.path.join(folder, name + ".npy"), values)
    for name, values in attributes.items():
        np.save(os.path.join(folder, "attr_%s.npy" % name), np.asarray(values, dtype=np.float64))
    tmp = os.path.join(folder, MANIFEST + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, os.path.join(folder, MANIFEST))


def column_type(values):
    """The point table's type for an attribute column: integer when every value is a whole number that fits int32
    (as ogr2ogr typed a JSON property that held only integers), else double precision."""
    v = np.asarray(values)
    if v.dtype.kind in "iu" and (len(v) == 0 or (v.min() >= -2**31 and v.max() < 2**31)):
        return "integer"
    return "double precision"


def check(manifest, p, lon, lat, m, d, attributes):
    """Raises PointsError unless the arrays are a valid set of points for manifest."""
    P, N = manifest["points"], manifest["dates"]
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise PointsError("not %s version %d: %r" % (FORMAT, VERSION, manifest))
    if P < 1 or N < 1:
        raise PointsError("no points or no dates")
    for name, a, dtype, shape in (("p", p, np.int32, (P,)), ("lon", lon, np.float64, (P,)),
                                  ("lat", lat, np.float64, (P,)), ("m", m, np.float64, (P,)),
                                  ("d", d, np.dtype(manifest["d_dtype"]), (P, N))):
        if a.dtype != dtype or a.shape != shape:
            raise PointsError("%s is %s %s, expected %s %s" % (name, a.dtype, a.shape, np.dtype(dtype), shape))
    if manifest["d_dtype"] not in ("float32", "float64"):
        raise PointsError("d must be float32 or float64")
    if P > 1 and not np.all(np.diff(p) > 0):
        raise PointsError("p is not strictly increasing")
    if not (np.all(np.isfinite(lon)) and np.all(np.isfinite(lat)) and np.all(np.abs(lat) < 90)):
        raise PointsError("a position is not finite or has |lat| >= 90")
    if list(attributes) != list(manifest["attributes"]):
        raise PointsError("attributes %s, manifest says %s" % (list(attributes), manifest["attributes"]))
    if set(manifest.get("attribute_types", {})) != set(attributes) or not all(
            t in ("integer", "double precision") for t in manifest.get("attribute_types", {}).values()):
        raise PointsError("attribute_types %r don't match the attributes" % manifest.get("attribute_types"))
    for name, a in attributes.items():
        if not NAME.match(name):
            raise PointsError("attribute name %r: letters, digits and _, not starting with a digit" % name)
        if np.asarray(a).shape != (P,):
            raise PointsError("attribute %s has shape %s, expected (%d,)" % (name, np.asarray(a).shape, P))
        if manifest["attribute_types"][name] == "integer":
            v = np.asarray(a, dtype=np.float64)
            v = v[np.isfinite(v)]
            if len(v) and (np.any(v != np.round(v)) or v.min() < -2**31 or v.max() >= 2**31):
                raise PointsError("attribute %s is an integer column but holds a value that isn't an int32" % name)


def has_points(folder):
    return os.path.isfile(os.path.join(folder, MANIFEST))


def read_points(folder, mmap=True):
    """The points of folder: (manifest, p, lon, lat, m, d, attributes), checked. Memory-mapped unless mmap=False."""
    path = os.path.join(folder, MANIFEST)
    if not os.path.isfile(path):
        raise PointsError("no %s in %s" % (MANIFEST, folder))
    with open(path) as f:
        manifest = json.load(f)
    mode = "r" if mmap else None
    arrays = {}
    for name in ARRAYS:
        arrays[name] = np.load(os.path.join(folder, name + ".npy"), mmap_mode=mode, allow_pickle=False)
    attributes = {name: np.load(os.path.join(folder, "attr_%s.npy" % name), mmap_mode=mode, allow_pickle=False)
                  for name in manifest.get("attributes", [])}
    check(manifest, arrays["p"], arrays["lon"], arrays["lat"], arrays["m"], arrays["d"], attributes)
    return manifest, arrays["p"], arrays["lon"], arrays["lat"], arrays["m"], arrays["d"], attributes
