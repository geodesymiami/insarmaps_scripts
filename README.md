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

## Converting a dataset: hdfeos5_2json_mbtiles.py, hdfeos5_or_csv_2json_mbtiles.py

`hdfeos5_2json_mbtiles.py FILE.he5 DIR` and `hdfeos5_or_csv_2json_mbtiles.py FILE.he5|FILE.csv DIR` write the dataset to
DIR as `chunk_*.json` (one GeoJSON Feature per point and line) and `metadata.pickle`. They no longer make a
`.mbtiles`: the map draws a dataset only from its recolor store, which the upload builds. (The names stay, since minsar
calls the scripts by them.)

## Uploading a dataset: json_mbtiles2insarmaps.py

`json_mbtiles2insarmaps.py --json_folder DIR -u DBUSER -p DBPASS --host HOST` uploads the dataset that
`hdfeos5_2json_mbtiles.py` or `hdfeos5_or_csv_2json_mbtiles.py` wrote to DIR (`metadata.pickle` and `chunk_*.json`).
It runs on the insarmaps server itself, where the database is (minsar's `ingest_insarmaps.bash` step 2 does this over
ssh, with `--host 127.0.0.1`), and connects to port 5432 unless `PGPORT` names another.

1. It removes a dataset of the same name, with its recolor store (if that removal fails, nothing is uploaded), then
   adds the dataset's row to `area` and loads the chunk files into its point table with ogr2ogr.
2. It checks that the table got every feature of the chunk files. If not, it removes the partly loaded dataset and
   fails.
3. It writes the dataset's `m_min` and `m_max` to `extra_attributes` (insarmaps `POST /preLoad` answers from them),
   then indexes and clusters the table on `p`.
4. It builds the dataset's recolor store (insarmaps `docs/recolor-store.md`), from which the map draws the dataset,
   and checks it against the table. If the store can't be built or fails its check, the store is removed and the
   upload fails; the dataset stays in the database, but the map can't show it until its store is built (insarmaps
   `tools/recolor_store/migrate.py --only AREA_ID`, or an upload of the dataset again).

It fails before anything changes when there are no chunk files, the store folder is missing or not writable, the
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
