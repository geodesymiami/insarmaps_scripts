"""Tests of recolor_store_ingest.py: no database, no network. Run from the repository's root:
python3 -m unittest discover -s tests"""
import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recolor_store_ingest as rsi  # noqa: E402

BUILD = 'ab' * 16


def make_store(root, area_id, name='DS', build=BUILD, meta=True):
    d = os.path.join(root, str(area_id))
    os.mkdir(d)
    with open(os.path.join(d, 'ydm.f32'), 'wb') as f:
        f.write(b'\0' * 64)
    if meta is True:
        with open(os.path.join(d, 'meta.json'), 'w') as f:
            json.dump({'build': build, 'dataset': {'area_id': area_id, 'unavco_name': name}}, f)
    elif meta:
        with open(os.path.join(d, 'meta.json'), 'w') as f:
            f.write(meta)
    return d


class TempDirs(unittest.TestCase):
    def tempdir(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        return d


class ChunkFiles(TempDirs):
    def setUp(self):
        self.dir = self.tempdir()

    def write(self, name, text):
        with open(os.path.join(self.dir, name), 'w') as f:
            f.write(text)

    def test_the_loaders_selection_sorted(self):
        for name in ('chunk_1.json', 'chunk_0.json', 'chunk_10.json', 'metadata.pickle', 'DS.mbtiles', 'log',
                     'tippecanoe_stderr.log'):
            self.write(name, '')
        os.mkdir(os.path.join(self.dir, 'folder.json'))
        self.assertEqual(rsi.json_chunk_files(self.dir), ['chunk_0.json', 'chunk_1.json', 'chunk_10.json'])

    def test_dates_of_the_metadata(self):
        self.assertEqual(rsi.dates_in_metadata({'decimal_dates_sql': '{2016.4,2016.5,2016.6}'}), 3)
        self.assertEqual(rsi.dates_in_metadata({'decimal_dates_sql': [2016.4, 2016.5]}), 2)
        self.assertEqual(rsi.dates_in_metadata({'decimal_dates_sql': '{}'}), 0)


class Root(TempDirs):
    def setUp(self):
        self.root = self.tempdir()

    def test_remove_store(self):
        make_store(self.root, 5)
        self.assertTrue(rsi.remove_store(self.root, 5, 'DS'))
        self.assertEqual(os.listdir(self.root), [])
        self.assertFalse(rsi.remove_store(self.root, 5, 'DS'))

    def test_renamed_before_it_is_deleted(self):
        make_store(self.root, 5)
        make_store(self.root, 6, meta=False)
        deleted = []
        with mock.patch.object(rsi.shutil, 'rmtree', side_effect=deleted.append):
            rsi.remove_store(self.root, 5)
            rsi.remove_store(self.root, 6)
        self.assertEqual([os.path.basename(p) for p in deleted], [f'.5.{BUILD}.old', '.6.' + '0' * 32 + '.old'])
        self.assertEqual(sorted(os.listdir(self.root)), sorted([f'.5.{BUILD}.old', '.6.' + '0' * 32 + '.old']))

    def test_an_old_leftover_of_the_same_name_is_replaced(self):
        os.makedirs(os.path.join(self.root, f'.5.{BUILD}.old', 'x'))
        make_store(self.root, 5)
        self.assertTrue(rsi.remove_store(self.root, 5))
        self.assertEqual(os.listdir(self.root), [])

    def test_refuses_the_store_of_another_dataset(self):
        make_store(self.root, 5, name='OTHER')
        with self.assertRaises(rsi.StoreError):
            rsi.remove_store(self.root, 5, 'DS')
        self.assertTrue(os.path.isdir(os.path.join(self.root, '5')))

    def test_a_store_without_readable_meta_is_removed(self):
        make_store(self.root, 5, meta='{not json')
        self.assertTrue(rsi.remove_store(self.root, 5, 'DS'))
        self.assertEqual(os.listdir(self.root), [])

    def test_check_root(self):
        tools = self.tempdir()
        with self.assertRaises(rsi.StoreError):
            rsi.check_root(os.path.join(self.root, 'missing'), tools)
        with self.assertRaises(rsi.StoreError):
            rsi.check_root(self.root, tools)                 # no tools there
        for name in ('convert_from_points.py', 'check_against_postgres.py'):
            open(os.path.join(tools, name), 'w').close()
        rsi.check_root(self.root, tools)

    @unittest.skipIf(hasattr(os, 'geteuid') and os.geteuid() == 0, 'root can write anywhere')
    def test_check_root_not_writable(self):
        tools = self.tempdir()
        for name in ('convert_from_points.py', 'check_against_postgres.py'):
            open(os.path.join(tools, name), 'w').close()
        os.chmod(self.root, 0o555)
        self.addCleanup(os.chmod, self.root, 0o755)
        with self.assertRaises(rsi.StoreError):
            rsi.check_root(self.root, tools)

    def test_the_root_of_another_database_is_refused(self):
        for i in range(100, 130):
            make_store(self.root, i)
        rsi.check_root_belongs(self.root, set(range(100, 130)) | {1, 2})
        rsi.check_root_belongs(self.root, set(range(100, 116)))        # 14 of 30 unknown: not most of them
        with self.assertRaises(rsi.StoreError):
            rsi.check_root_belongs(self.root, {100, 101, 657})          # a copy of the database with few datasets
        small = self.tempdir()
        for i in range(10):
            make_store(small, i)
        rsi.check_root_belongs(small, set())                            # too few stores to tell
        rsi.check_root_belongs(os.path.join(self.root, 'missing'), set())

    def test_the_root_records_its_database(self):
        rsi.check_root_identity(self.root, '7419/pgis')                 # first use: recorded
        with open(os.path.join(self.root, rsi.MARKER)) as f:
            self.assertEqual(json.load(f)['database'], '7419/pgis')
        rsi.check_root_identity(self.root, '7419/pgis')
        with self.assertRaisesRegex(rsi.StoreError, '7419/pgis'):
            rsi.check_root_identity(self.root, '1234/pgis')             # a copy of the database elsewhere
        self.assertEqual(rsi.store_ids(self.root), [])                  # the record is not a store
        self.assertEqual([n for n in os.listdir(self.root) if n.endswith('.tmp')], [])
        with open(os.path.join(self.root, rsi.MARKER), 'w') as f:
            f.write('{broken')
        with self.assertRaises(rsi.StoreError):
            rsi.check_root_identity(self.root, '7419/pgis')

    def test_store_bytes(self):
        make_store(self.root, 5)
        d = os.path.join(self.root, '5')
        self.assertEqual(rsi.store_bytes(d), sum(os.path.getsize(os.path.join(d, n)) for n in os.listdir(d)))
        self.assertEqual(rsi.store_bytes(os.path.join(self.root, '6')), 0)

    def test_estimate_and_space(self):
        self.assertEqual(rsi.estimate_store_bytes(100, 20, 2), 100 * (24 + 4 * 20 + 18 * 3 + 8 * 2))
        rsi.check_space(self.root, 0, 0)
        with self.assertRaises(rsi.StoreError):
            rsi.check_space(self.root, 0, 1e9)


class FakeCursor:
    def __init__(self, rows=None, one=None):
        self.sql, self.rows, self.one = [], rows or [], one

    def execute(self, sql, params=None):
        self.sql.append(sql)

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.one


class Database(unittest.TestCase):
    def test_rows_and_m_extents_in_one_transaction(self):
        cur = FakeCursor(one=(7, '-1.5', 'NaN'))
        con = mock.Mock()
        con.cursor.return_value = cur
        self.assertEqual(rsi.rows_and_m_extents(con, '42'), (7, '-1.5', 'NaN'))
        cur.one = (0, None, None)                                       # a table without m: no extents
        self.assertEqual(rsi.rows_and_m_extents(con, '42'), (0, None, None))
        cur.sql = cur.sql[:2]
        self.assertEqual(cur.sql[0], 'SET LOCAL extra_float_digits = 1')
        self.assertIn('FROM "42"', cur.sql[1])
        self.assertIn('area_id = 42', cur.sql[1])
        self.assertIn("('m_min', e.lo::text), ('m_max', e.hi::text)", cur.sql[1])
        self.assertIn('WHERE e.lo IS NOT NULL', cur.sql[1])
        self.assertEqual(con.commit.call_count, 2)

    def test_database_identity(self):
        cur = FakeCursor(one=('7419123', 'pgis'))
        self.assertEqual(rsi.database_identity(cur), '7419123/pgis')
        self.assertIn('pg_control_system()', cur.sql[0])

    def test_area_ids_are_those_with_a_point_table(self):
        cur = FakeCursor(rows=[(1,), (None,), (3,)])
        self.assertEqual(rsi.area_ids(cur), {1, 3})
        self.assertIn('JOIN pg_class c ON c.relname = a.id::text', cur.sql[0])

    def test_dsn_has_no_password_or_port(self):
        self.assertEqual(rsi.dsn('127.0.0.1', 'insaradmin'),
                         "host='127.0.0.1' dbname='pgis' user='insaradmin' connect_timeout=30")
        self.assertEqual(rsi.dsn('h', "o'b\\x"), "host='h' dbname='pgis' user='o\\'b\\\\x' connect_timeout=30")


STUB = '''
import json, os, sys
args = sys.argv[1:]
with open(os.environ['STUB_RECORD'], 'a') as f:
    f.write(json.dumps({'tool': TOOL, 'argv': args, 'pgpassword': os.environ.get('PGPASSWORD')}) + '\\n')
mode = os.environ.get('STUB_' + TOOL.upper(), 'ok')
if TOOL == 'convert':
    if mode == 'fail':
        sys.stderr.write('convert: boom\\n')
        sys.exit(1)
    if mode == 'publish_then_fail':
        area_id = int(args[0])
        d = os.path.join(args[args.index('--root') + 1], str(area_id))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'meta.json'), 'w') as f:
            json.dump({'build': 'ef' * 16, 'dataset': {'area_id': area_id, 'unavco_name': os.environ['STUB_NAME']}}, f)
        sys.stderr.write('convert: died after publishing\\n')
        sys.exit(1)
    area_id = int(args[0])
    d = os.path.join(args[args.index('--root') + 1], str(area_id))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'meta.json'), 'w') as f:
        json.dump({'build': 'cd' * 16, 'dataset': {'area_id': area_id, 'unavco_name': os.environ['STUB_NAME']}}, f)
    print('some output')
    print(json.dumps({'build': 'cd' * 16, 'points': 3, 'dates': 2, 'bytes': 1234, 'missing': {'values': 0},
                      'attributes': []}))
else:
    ok = mode == 'ok'
    print(json.dumps({'ok': ok, 'failures': [] if ok else ['p sets differ: table 3 points, store 2'],
                      'max_abs_diff_cm_per_yr': 1e-9}))
    sys.exit(0 if ok else 1)
'''


class Build(TempDirs):
    def setUp(self):
        self.root = self.tempdir()
        self.tools = self.tempdir()
        for tool, name in (('convert', 'convert_from_points.py'), ('check', 'check_against_postgres.py')):
            with open(os.path.join(self.tools, name), 'w') as f:
                f.write(f'TOOL = {tool!r}\n' + textwrap.dedent(STUB))
        self.record = os.path.join(self.tempdir(), 'calls.jsonl')
        env = {'STUB_RECORD': self.record, 'STUB_NAME': 'DS'}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.conninfo = rsi.dsn('127.0.0.1', 'insaradmin')
        self.points = self.tempdir()                   # the folder step 1 wrote

    def build(self, min_free_gb=0.0):
        return rsi.build_and_check_store(5, 'DS', 3, 2, 0, self.points, self.conninfo, 'secret-pw', self.root,
                                         self.tools, min_free_gb)

    def calls(self):
        if not os.path.exists(self.record):
            return []
        with open(self.record) as f:
            return [json.loads(line) for line in f]

    def test_builds_then_checks(self):
        out = self.build()
        self.assertEqual((out['area_id'], out['build'], out['points'], out['dates'], out['bytes']),
                         (5, 'cd' * 16, 3, 2, 1234))
        convert, check = self.calls()
        self.assertEqual(convert['argv'], ['5', '--root', self.root, '--points', self.points, '--replace', '--dsn',
                                           self.conninfo])
        self.assertEqual(check['argv'], [os.path.join(self.root, '5'), '--dsn', self.conninfo])
        for call in (convert, check):                 # the password goes in the environment, not the command line
            self.assertEqual(call['pgpassword'], 'secret-pw')
            self.assertNotIn('secret-pw', ' '.join(call['argv']))
        self.assertTrue(os.path.isfile(os.path.join(self.root, '5', 'meta.json')))

    def test_a_failed_build_raises(self):
        with mock.patch.dict(os.environ, {'STUB_CONVERT': 'fail'}):
            with self.assertRaisesRegex(rsi.StoreError, 'boom'):
                self.build()
        self.assertEqual([c['tool'] for c in self.calls()], ['convert'])

    def test_a_store_published_by_a_failed_build_is_removed(self):
        with mock.patch.dict(os.environ, {'STUB_CONVERT': 'publish_then_fail'}):
            with self.assertRaisesRegex(rsi.StoreError, 'died after publishing'):
                self.build()
        self.assertEqual(os.listdir(self.root), [])

    def test_an_empty_password_overrides_an_inherited_one(self):
        with mock.patch.dict(os.environ, {'PGPASSWORD': 'inherited'}):
            rsi.build_and_check_store(5, 'DS', 3, 2, 0, self.points, self.conninfo, '', self.root, self.tools, 0.0)
        self.assertEqual([c['pgpassword'] for c in self.calls()], ['', ''])

    def test_a_store_that_fails_its_check_is_removed(self):
        with mock.patch.dict(os.environ, {'STUB_CHECK': 'fail'}):
            with self.assertRaisesRegex(rsi.StoreError, 'p sets differ'):
                self.build()
        self.assertEqual(os.listdir(self.root), [])

    def test_the_store_of_another_dataset_is_not_replaced(self):
        make_store(self.root, 5, name='OTHER')
        with self.assertRaisesRegex(rsi.StoreError, 'OTHER'):
            self.build()
        self.assertEqual(self.calls(), [])
        with open(os.path.join(self.root, '5', 'meta.json')) as f:
            self.assertEqual(json.load(f)['dataset']['unavco_name'], 'OTHER')

    def test_the_same_datasets_store_is_replaced(self):
        make_store(self.root, 5, name='DS')            # e.g. built by migrate.py meanwhile
        self.assertEqual(self.build()['build'], 'cd' * 16)

    def test_not_enough_space(self):
        with self.assertRaisesRegex(rsi.StoreError, 'not enough space'):
            self.build(min_free_gb=1e9)
        self.assertEqual(self.calls(), [])


if __name__ == '__main__':
    unittest.main()
