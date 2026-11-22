# insarmaps_scripts

Collection of scripts to upload and manipulate data on https://insarmaps.miami.edu/

## Installation
### Docker
If you have Docker installed, simply run `./init_docker.sh && ./run_docker.sh` which will create a docker instance where the scripts can be run from.

### Manual install
The below install instructions have been tested on ubuntu 22.04. Installing on Windows/MacOS/other linux distributions will involve modifying the below commands to the equivalent ones on the target system.

1. Install gdal: ```sudo apt-get install gdal-bin```
2. Install Mintpy: https://github.com/insarlab/MintPy/blob/main/docs/installation.md
3. Run `conda install -file environment.yml` to install non-python code (gdal, libpq)
4. Run `pip install -r requirements.txt` to install remaining requirements

## Converting a dataset: insarmaps_ingester.py

`insarmaps_ingester.py FILE DIR` writes the dataset in FILE, a CSV when its name ends in `.csv` and an HDF-EOS5 file
(`.he5`) otherwise, to DIR as its points, numpy arrays (`points_bundle.py`: `points.json`, `p.npy`, `lon.npy`,
`lat.npy`, `m.npy`, `d.npy`, and `attr_NAME.npy` for a CSV's quality columns; insarmaps `docs/recolor-store.md`
section 12), and `metadata.pickle`. It reads the input once and writes no JSON and no `.mbtiles`; files of an earlier
run in DIR are replaced. `--num-workers` is accepted and not needed.

It replaces `hdfeos5_2json_mbtiles.py`, which minsar ran for an HDF-EOS5 file, and `hdfeos5_or_csv_2json_mbtiles.py`,
which it ran for a CSV, and writes the same points and metadata for each. A minsar that still calls those names stops
at step 1 with "command not found": update minsar too. With `--he5-like-csv`, an HDF-EOS5 file gets what
`hdfeos5_or_csv_2json_mbtiles.py` wrote for one, as for a CSV: `p` is the cell's flat index (row * WIDTH + col), the
dataset's position the mean of its cells', and the attributes a CSV's (`dem` among them, WIDTH and LENGTH only with a
geographic grid). minsar doesn't use it.

- A grid cell (HDF-EOS5) or CSV row becomes a point when it has values at two dates or more; a missing value is NaN
  in `d`, and the point's velocity `m` is the least-squares slope over the dates it has.
- `p`: an HDF-EOS5 file's points are counted from the first cell of each block of 20,000 cells (with
  `--he5-like-csv`, `p` is the flat index); a CSV's `p` is the row (0 for the first data row). These are the numbers
  step 1's JSON gave the points.
- `d` holds the input's values exactly: float32 from an HDF-EOS5 file, float64 from a CSV (mm converted to m).
- A CSV's quality columns keep their order; `point_ID` is an integer column when its values are whole numbers that fit
  int32; a missing quality value is NaN, and NULL in the point table.
- A point without a finite position stops step 1 with a message saying which row.

## Uploading a dataset: json_mbtiles2insarmaps.py

`json_mbtiles2insarmaps.py --json_folder DIR -u DBUSER -p DBPASS --host HOST` uploads the dataset that
`insarmaps_ingester.py` wrote to DIR (`metadata.pickle` and the points). It refuses a DIR that holds only the
`chunk_*.json` of an older step 1: update both minsar and insarmaps_scripts where step 1 ran (a minsar from before
2026-11-21 still expects an .mbtiles file, and fails after the upload), and run it again.
It runs on the insarmaps server itself, where the database is (minsar's `ingest_insarmaps.bash` step 2 does this over
ssh, with `--host 127.0.0.1`), and connects to port 5432 unless `PGPORT` names another.

1. It removes a dataset of the same name, with its recolor store (if that removal fails, nothing is uploaded), then
   adds the dataset's row to `area`, creates its point table as ogr2ogr used to (`ogc_fid`, `d`, `m`, `p`, the quality
   columns, `wkb_geometry`, and the spatial index), and loads the points into it with binary COPY, in `p` order. A
   failed load removes the dataset.
2. It checks that the table got every point. If not, it removes the partly loaded dataset and fails.
3. It writes the dataset's `m_min` and `m_max` to `extra_attributes` (insarmaps `POST /preLoad` answers from them),
   then indexes the table on `p` and marks that index as the table's clustering index (the rows went in in `p`
   order, so no rewrite is needed).
4. It builds the dataset's recolor store (insarmaps `docs/recolor-store.md`), from which the map draws the dataset,
   from the same points (`tools/recolor_store/convert_from_points.py`), and checks it against the table. Points
   drop out at low zooms as with tippecanoe's `-r 2.5` and base zoom 14, or base zoom 12 for a geocoded dataset;
   `--drop-rate` and `--base-zoom` choose others. If the store can't be built or fails its check, the store is
   removed and the upload fails; the dataset stays in the database, but the map can't show it until its store is
   built (an upload of the dataset again, or insarmaps `tools/recolor_store/migrate.py --only AREA_ID`).

It fails before anything changes when DIR has no points, the store folder is missing or not writable, the
insarmaps tools are not there, or the store wouldn't leave `--recolor-store-min-free-gb` (20) free (the store of a
dataset it replaces counts as free). It also refuses another database's store folder (for example the live one, while
uploading to a copy of the database): the folder records its database in `.insarmaps-database` (the first upload or
removal writes it: the cluster's system identifier and the database's name), and a folder without that record is
refused when most of its stores belong to datasets the database has no point table for.

The store needs:

- the insarmaps repository checked out beside this one, as minsar does (`$MINSAR_HOME/tools/insarmaps`): its
  `tools/recolor_store` builds and checks the store (`--recolor-tools` names another copy);
- the store folder, where the insarmaps app reads the stores: `$INSARMAPS_RECOLOR_STORE_DIR`, else
  `/data/insarmaps2/recolor_store` (`--recolor-store-root` names another), writable by the user who uploads;
- numpy and psycopg2 in the Python that runs the script.

`json_mbtiles2insarmaps.py --remove NAME -u DBUSER -p DBPASS --host HOST -U WEBUSER -P WEBPASS` removes a dataset: its
table and rows, its recolor store and, for a dataset uploaded before the stores, its .mbtiles; it exits with code 1
when that fails.

## Tests

From the repository's root, in the Python environment the scripts run in: `python3 -m unittest discover -s tests`.
They need no database and no network.
