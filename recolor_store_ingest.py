"""Recolor stores at ingest, for json_mbtiles2insarmaps.py (insarmaps repository: docs/recolor-store.md).

The insarmaps map draws a dataset only from its recolor store, a set of files per dataset in a folder named by its
area id. After json_mbtiles2insarmaps.py has loaded a dataset's point table, this module

- checks that the table holds every feature of the chunk files,
- writes the extra_attributes m_min and m_max, which POST /preLoad answers from,
- builds the store from the table and checks it against the table, with the insarmaps repository's tools
  (tools/recolor_store/convert_from_postgres.py and check_against_postgres.py), each run as its own process by the
  Python running the ingest, which has the numpy and psycopg2 they need;

and when a dataset is removed or ingested again (a re-ingest gets a new area id), it removes the old store.

Standard library only.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

# where the app reads the stores: run_docker.sh mounts INSARMAPS_RECOLOR_STORE_DIR (default: recolor_store beside
# postgres_dir, /data/insarmaps2/recolor_store on insarmaps2) read-only at /recolor_store
DEFAULT_ROOT = os.environ.get('INSARMAPS_RECOLOR_STORE_DIR') or '/data/insarmaps2/recolor_store'
# in a minsar install the insarmaps repository is checked out beside this one ($MINSAR_HOME/tools/insarmaps)
DEFAULT_TOOLS = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                                              'insarmaps', 'tools', 'recolor_store'))
DEFAULT_MIN_FREE_GB = 20.0      # free space to keep on the store root's file system, as migrate.py keeps
BLOCK = 16                      # the store's block of dates (docs/recolor-store.md section 2)
HEX = re.compile(r'^[0-9a-f]+$')
# in the store root: which database the stores belong to (its cluster's system identifier and its name), written
# by the first upload or removal; the app and migrate.py ignore it (they read only <root>/<area_id>/)
MARKER = '.insarmaps-database'
CONNECT_TIMEOUT_S = 30

# One pass over the point table: its row count, and m_min/m_max as PR #183 of insarmaps defines them: Postgres's own
# min(m) and max(m) (NULLs ignored, NaN above every number, so one NaN m makes m_max NaN) as their shortest
# round-trip text, both written in one transaction, neither when the table has no m at all.
ROWS_AND_M_EXTENTS = """
WITH e AS (SELECT count(*) AS n, min(m) AS lo, max(m) AS hi FROM "{id}"),
old AS (DELETE FROM extra_attributes WHERE area_id = {id} AND attributekey IN ('m_min', 'm_max')),
ins AS (INSERT INTO extra_attributes (area_id, attributekey, attributevalue)
        SELECT {id}, kv.k, kv.v FROM e CROSS JOIN LATERAL (VALUES ('m_min', e.lo::text), ('m_max', e.hi::text)) kv(k, v)
        WHERE e.lo IS NOT NULL)
SELECT e.n, e.lo::text, e.hi::text FROM e
"""


class StoreError(Exception):
    pass


# ---------------------------------------------------------------------------------------------- the chunk files

def json_chunk_files(folder):
    """The files json_mbtiles2insarmaps.py loads, sorted: those whose second dot-separated part is json
    (chunk_<k>.json)."""
    out = []
    for name in sorted(os.listdir(folder)):
        parts = name.split('.')
        if len(parts) > 1 and parts[1] == 'json' and os.path.isfile(os.path.join(folder, name)):
            out.append(name)
    return out


def count_features(folder, files):
    """The features in the chunk files: the ingest scripts write one GeoJSON Feature per line."""
    n = 0
    for name in files:
        with open(os.path.join(folder, name), 'rb') as f:
            for line in f:
                if line.strip():
                    n += 1
    return n


def extra_properties(folder, files):
    """How many properties besides d, m and p the first feature has (the CSV quality columns): each becomes a
    column of the table and a file of the store."""
    for name in files:
        with open(os.path.join(folder, name), 'rb') as f:
            for line in f:
                if line.strip():
                    props = json.loads(line).get('properties') or {}
                    return len([k for k in props if k not in ('d', 'm', 'p')])
    return 0


def dates_in_metadata(metadata):
    """The number of dates of a metadata.pickle dict: its decimal_dates_sql, a Postgres array literal '{...}'."""
    v = metadata['decimal_dates_sql']
    if isinstance(v, str):
        inner = v.strip().strip('{}').strip()
        return len(inner.split(',')) if inner else 0
    return len(v)


# ---------------------------------------------------------------------------------------------- the store root

def store_ids(root):
    """Area ids that have a store directory in root (none when root doesn't exist)."""
    if not os.path.isdir(root):
        return []
    return sorted(int(d) for d in os.listdir(root) if d.isdigit() and os.path.isdir(os.path.join(root, d)))


def read_meta(store_dir):
    """A store's meta.json, None without the directory, {} when meta.json can't be read."""
    if not os.path.isdir(store_dir):
        return None
    try:
        with open(os.path.join(store_dir, 'meta.json')) as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


def check_root(root, tools):
    """The store root exists and can be written; the insarmaps tools are there."""
    if not os.path.isdir(root):
        raise StoreError(f'the recolor store root {root} does not exist (--recolor-store-root)')
    if not os.access(root, os.W_OK | os.X_OK):
        raise StoreError(f'the recolor store root {root} is not writable by this user')
    for name in ('convert_from_postgres.py', 'check_against_postgres.py'):
        if not os.path.isfile(os.path.join(tools, name)):
            raise StoreError(f'{os.path.join(tools, name)} not found: --recolor-tools names the insarmaps '
                             f'repository\'s tools/recolor_store')


def check_root_belongs(root, area_ids, min_stores=20):
    """Refuses a store root whose stores mostly belong to datasets the database doesn't have (area_ids: those with a
    point table): then it is another database's root (for example the live root while the ingest runs against a copy
    of the database), and deleting or replacing a store there would break that database's map."""
    stores = store_ids(root)
    unknown = [i for i in stores if i not in area_ids]
    if len(stores) >= min_stores and 2 * len(unknown) > len(stores):
        raise StoreError(f'{len(unknown)} of the {len(stores)} stores in {root} belong to datasets this database '
                         f'does not have: it is the store root of another database. Give this database\'s root '
                         f'with --recolor-store-root')


def check_root_identity(root, identity):
    """The store root records the database its stores belong to (MARKER): refuses a root that records another one,
    and records this one in a root that has no record yet. Run after check_root_belongs, which also catches the
    root of another database that has no record yet."""
    path = os.path.join(root, MARKER)
    try:
        with open(path) as f:
            recorded = json.load(f).get('database')
    except FileNotFoundError:
        recorded = None
    except (OSError, ValueError, AttributeError) as e:
        raise StoreError(f'{path} can not be read: {e}')
    if recorded is None:
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            json.dump({'database': identity, 'recorded': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       'by': 'insarmaps_scripts json_mbtiles2insarmaps.py'}, f)
            f.write('\n')
        os.replace(tmp, path)
    elif recorded != identity:
        raise StoreError(f'{root} holds the stores of database {recorded}, not of this one ({identity}). Give this '
                         f'database\'s store root with --recolor-store-root')


def store_bytes(store_dir):
    """The size of a store's files (0 without the directory)."""
    if not os.path.isdir(store_dir):
        return 0
    return sum(e.stat().st_size for e in os.scandir(store_dir) if e.is_file())


def estimate_store_bytes(points, dates, attributes=0):
    """An upper bound of a store's size (docs/recolor-store.md section 2; the bound migrate.py uses)."""
    nb = -(-dates // BLOCK) + 1
    return int(points * (24 + 4 * dates + 18 * nb + 8 * attributes))


def check_space(root, need_bytes, min_free_gb):
    free = shutil.disk_usage(root).free
    if free - need_bytes < min_free_gb * 2 ** 30:
        raise StoreError(f'not enough space for the recolor store: {free / 1e9:.1f} GB free in {root}, the store '
                         f'needs up to {need_bytes / 1e9:.1f} GB and {min_free_gb:g} GB must stay free '
                         f'(--recolor-store-min-free-gb)')


def remove_store(root, area_id, unavco_name=None):
    """Removes <root>/<area_id> as docs/recolor-store.md section 8 says: renamed to .<id>.<build>.old first, so that
    the app never opens a half-deleted store, then deleted. With unavco_name, refuses a store whose meta.json names
    another dataset. Returns whether there was a store."""
    area_id = int(area_id)
    src = os.path.join(root, str(area_id))
    meta = read_meta(src)
    if meta is None:
        return False
    named = (meta.get('dataset') or {}).get('unavco_name')
    if unavco_name is not None and named is not None and named != unavco_name:
        raise StoreError(f'{src} holds the store of {named!r}, not of {unavco_name!r}: not removing it')
    build = meta.get('build')
    if not isinstance(build, str) or not HEX.match(build):
        build = '0' * 32            # a name migrate.py's cleanup of interrupted removals recognizes
    dst = os.path.join(root, f'.{area_id}.{build}.old')
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.rename(src, dst)
    shutil.rmtree(dst)
    return True


# ---------------------------------------------------------------------------------------------- the database

def database_identity(cursor):
    """The database's identity for MARKER: its cluster's system identifier (another cluster, such as a copy of the
    database restored into a test server, has another) and its name."""
    cursor.execute('SELECT system_identifier::text, current_database() FROM pg_control_system()')
    sysid, name = cursor.fetchone()
    return f'{sysid}/{name}'


def area_ids(cursor):
    """The area ids that have a point table: the datasets a store can belong to. (Not every area row: a copy of the
    database may hold all the area rows but only a few point tables.)"""
    cursor.execute("SELECT a.id FROM area a JOIN pg_class c ON c.relname = a.id::text AND c.relkind = 'r' "
                   "AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = current_schema())")
    return {int(r[0]) for r in cursor.fetchall() if r[0] is not None}


def rows_and_m_extents(con, area_id):
    """Counts the rows of the dataset's point table and writes its m_min and m_max (ROWS_AND_M_EXTENTS) in one
    transaction; returns (rows, m_min, m_max), the extents as written (None when the table has no m)."""
    cur = con.cursor()
    cur.execute('SET LOCAL extra_float_digits = 1')     # shortest round-trip text, whatever the server's default
    # the only value in this SQL is int(area_id), which can't carry SQL; a table name can't be a query parameter
    cur.execute(ROWS_AND_M_EXTENTS.format(id=int(area_id)))  # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
    n, lo, hi = cur.fetchone()
    con.commit()
    return int(n), lo, hi


def dsn(host, user, dbname='pgis'):
    """A libpq connection string without the password (the tools get it in PGPASSWORD, not on their command line)
    and without a port, so that the tools connect where the ingest does (PGPORT, else 5432)."""
    def q(v):
        return "'" + str(v).replace('\\', '\\\\').replace("'", "\\'") + "'"
    return f'host={q(host)} dbname={q(dbname)} user={q(user)} connect_timeout={CONNECT_TIMEOUT_S}'


# ---------------------------------------------------------------------------------------------- building

def run_tool(script, args, password, python=None):
    """Runs one of the tools; returns (exit code, its last stdout line parsed as JSON or None, stderr)."""
    env = dict(os.environ)
    if password is not None:            # also an empty one: never a PGPASSWORD inherited from the environment
        env['PGPASSWORD'] = password
    # an argument list without a shell: the tool's path from --recolor-tools, then an area id, paths and a libpq
    # connection string without the password
    p = subprocess.run([python or sys.executable, script] + list(args), capture_output=True, text=True, env=env)  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    report = None
    lines = [x for x in p.stdout.splitlines() if x.strip()]
    if lines:
        try:
            report = json.loads(lines[-1])
        except ValueError:
            report = None
    return p.returncode, report, p.stderr.strip()


def build_and_check_store(area_id, unavco_name, points, dates, attributes, conninfo, password, root=DEFAULT_ROOT,
                          tools=DEFAULT_TOOLS, min_free_gb=DEFAULT_MIN_FREE_GB, python=None):
    """Builds <root>/<area_id> from the dataset's point table (convert_from_postgres.py, levels from tippecanoe's
    drop rule), then checks it against the table (check_against_postgres.py). Raises StoreError when it can't be
    built, or when it fails its check, after removing it. Returns what was built and how long it took."""
    area_id = int(area_id)
    check_root(root, tools)
    store = os.path.join(root, str(area_id))
    meta = read_meta(store)
    if meta:
        ds = meta.get('dataset') or {}
        if ds.get('unavco_name') != unavco_name or ds.get('area_id') != area_id:
            raise StoreError(f'{store} already holds the store of {ds.get("unavco_name")!r} (area id '
                             f'{ds.get("area_id")}): not replacing it. Is {root} the store root of this database?')
    check_space(root, estimate_store_bytes(points, dates, attributes), min_free_gb)
    t0 = time.time()
    code, out, err = run_tool(os.path.join(tools, 'convert_from_postgres.py'),
                              [str(area_id), '--root', root, '--no-mbtiles', '--replace', '--dsn', conninfo],
                              password, python)
    built_s = time.time() - t0
    if code != 0 or not out:
        # a store the converter published before it failed has not been checked: not left for the app to serve
        try:
            remove_store(root, area_id, unavco_name)
        except (StoreError, OSError):
            pass
        raise StoreError(f'building the recolor store failed (exit code {code}): {err[-3000:]}')
    t1 = time.time()
    code, rep, err = run_tool(os.path.join(tools, 'check_against_postgres.py'), [store, '--dsn', conninfo],
                              password, python)
    check_s = time.time() - t1
    if code != 0 or not rep or rep.get('ok') is not True:
        removed = remove_store(root, area_id)
        why = '; '.join((rep or {}).get('failures') or []) or err[-3000:] or f'exit code {code}'
        raise StoreError(f'the recolor store failed its check against the table ({why}); '
                         f'{"removed it" if removed else "no store left"}')
    return {'area_id': area_id, 'build': out.get('build'), 'points': out.get('points'), 'dates': out.get('dates'),
            'bytes': out.get('bytes'), 'missing_values': (out.get('missing') or {}).get('values'),
            'attributes': out.get('attributes'), 'max_abs_diff_cm_per_yr': rep.get('max_abs_diff_cm_per_yr'),
            'build_s': round(built_s, 1), 'check_s': round(check_s, 1)}
