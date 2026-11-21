"""Tests of json_mbtiles2insarmaps.py: upload_points' order of work, with the database mocked, and the binary COPY data
it loads the table with, decoded byte by byte: no database, no network. Needs the ingest's Python environment
(psycopg2, pycurl, requests, mintpy, numpy), which add_attribute_insarmaps.py imports. Run from the repository's root:
python3 -m unittest discover -s tests"""
import io
import json
import os
import pickle
import shutil
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# the packages of the ingest's environment that add_attribute_insarmaps.py imports; any other import error fails
OPTIONAL = {'psycopg2', 'pycurl', 'requests', 'mintpy', 'numpy'}
try:
    import numpy as np
    import json_mbtiles2insarmaps as j
    import points_bundle as pb
except ImportError as e:
    if (e.name or '').split('.')[0] not in OPTIONAL:
        raise
    j = None
    IMPORT_ERROR = e


def make_store(root, area_id, name):
    d = os.path.join(root, str(area_id))
    os.mkdir(d)
    with open(os.path.join(d, 'meta.json'), 'w') as f:
        json.dump({'build': 'ab' * 16, 'dataset': {'area_id': area_id, 'unavco_name': name}}, f)


def write_metadata(folder, dates=2):
    meta = {'area': 'DS', 'project_name': 'P', 'mid_long': -91.1, 'mid_lat': -0.8, 'country': 'None',
            'region': 'R', 'chunk_num': 1, 'attribute_keys': ['mission'], 'attribute_values': ['S1'],
            'string_dates_sql': '{' + ','.join('201606%02d' % (5 + k) for k in range(dates)) + '}',
            'decimal_dates_sql': '{' + ','.join('%.3f' % (2016.427 + 0.01 * k) for k in range(dates)) + '}',
            'attributes': {'mission': 'S1'}, 'needed_attributes': {'mission'}}
    with open(os.path.join(folder, 'metadata.pickle'), 'wb') as f:
        pickle.dump(meta, f)


def decode(body):
    """The rows of a binary COPY body: each a list of field bytes, None for NULL."""
    assert body[:11] == b'PGCOPY\n\xff\r\n\x00'
    off, rows = 19, []
    while True:
        nf, = struct.unpack_from('>h', body, off)
        off += 2
        if nf == -1:
            assert off == len(body)
            return rows
        fields = []
        for _ in range(nf):
            n, = struct.unpack_from('>i', body, off)
            off += 4
            fields.append(None if n == -1 else body[off:off + n])
            off += max(n, 0)
        rows.append(fields)


def point_of(ewkb):
    bo, gt, srid, x, y = struct.unpack('<BIIdd', ewkb)
    assert (bo, gt, srid) == (1, 0x20000001, 4326)
    return x, y


def array_of(b):
    ndim, flags, oid, n, lb = struct.unpack_from('>iiIii', b, 0)
    assert (ndim, flags, oid, lb) == (1, 0, 701, 1)
    values = []
    for k in range(n):
        ln, v = struct.unpack_from('>id', b, 20 + 12 * k)
        assert ln == 8
        values.append(v)
    return values


@unittest.skipIf(j is None, 'needs the ingest environment (%s)' % (None if j else IMPORT_ERROR))
class CopyRows(unittest.TestCase):
    def rows(self, p, lon, lat, m, d, attributes=None, types=None, copy_bytes=None):
        with mock.patch.object(j, 'COPY_BYTES', copy_bytes or j.COPY_BYTES):
            bodies = list(j.copy_rows(p, lon, lat, m, d, attributes or {}, types or {}))
        return bodies, [r for b in bodies for r in decode(b)]

    def test_every_value_as_written(self):
        p = np.array([3, 7, 20000, 20001], dtype=np.int32)
        lon = np.array([-91.1, -91.2, -91.3, 1e-300])
        lat = np.array([-0.8, 0.0, 89.5, -0.1])
        d = np.array([[0.0, 0.001, np.nan], [1.5e-3, -2e-3, 3e-3], [0, 0, 0], [1, 2, 3]], dtype=np.float32)
        m = np.array([0.0011, np.nan, 0.0, 1.0])
        bodies, rows = self.rows(p, lon, lat, m, d)
        self.assertEqual(len(bodies), 1)
        self.assertEqual(len(rows), 4)
        for k, row in enumerate(rows):
            self.assertEqual(len(row), 4)
            self.assertEqual(point_of(row[0]), (lon[k], lat[k]))
            got = array_of(row[1])
            self.assertEqual(np.array(got).tobytes(), d[k].astype(np.float64).tobytes())   # NaN too, bit for bit
            self.assertEqual(struct.pack('>d', m[k]), row[2])
            self.assertEqual(struct.unpack('>i', row[3])[0], p[k])

    def test_attributes_with_nulls_in_p_order_over_several_blocks(self):
        P = 50
        p = np.arange(P, dtype=np.int32) * 2
        lon, lat = np.linspace(-80.2, -80.1, P), np.linspace(25.7, 25.8, P)
        d = np.ones((P, 3))
        m = np.zeros(P)
        coherence = np.linspace(0.5, 0.99, P)
        coherence[[4, 5, 17]] = np.nan
        point_id = np.arange(P, dtype=np.float64) + 1000
        point_id[[5, 30]] = np.nan
        attributes = {'coherence': coherence, 'point_ID': point_id}
        types = {'coherence': 'double precision', 'point_ID': 'integer'}
        bodies, rows = self.rows(p, lon, lat, m, d, attributes, types, copy_bytes=1000)
        self.assertGreater(len(bodies), 3)
        self.assertEqual([struct.unpack('>i', r[3])[0] for r in rows], list(p))         # p order kept
        for k, r in enumerate(rows):
            self.assertEqual(len(r), 6)
            if np.isnan(coherence[k]):
                self.assertIsNone(r[4])
            else:
                self.assertEqual(struct.unpack('>d', r[4])[0], coherence[k])
            if np.isnan(point_id[k]):
                self.assertIsNone(r[5])
            else:
                self.assertEqual(struct.unpack('>i', r[5])[0], int(point_id[k]))


@unittest.skipIf(j is None, 'needs the ingest environment (%s)' % (None if j else IMPORT_ERROR))
class Upload(unittest.TestCase):
    def setUp(self):
        self.dir, self.root, self.tools = (tempfile.mkdtemp() for _ in range(3))
        for d in (self.dir, self.root, self.tools):
            self.addCleanup(shutil.rmtree, d, True)
        for name in ('convert_from_points.py', 'check_against_postgres.py'):
            open(os.path.join(self.tools, name), 'w').close()
        write_metadata(self.dir)
        pb.write_points(self.dir, [0, 1, 2], [-91.1, -91.2, -91.3], [-0.8, -0.8, -0.8], [0.001, 0.002, 0.003],
                        np.zeros((3, 2), dtype=np.float32))
        self.ctrl = mock.MagicMock(name='controller')
        self.ctrl.get_dataset_id.side_effect = [-1, 8]           # no old dataset; the new one gets id 8
        self.patch('InsarDatabaseController', return_value=self.ctrl)
        self.load = self.patch('load_table')
        self.clustered = self.patch('mark_clustered')
        self.rows = self.patch('rows_and_m_extents', target=j.rsi, return_value=(3, '0.001', '0.003'))
        self.area_ids = self.patch('area_ids', target=j.rsi, return_value={1, 2})
        self.identity = self.patch('database_identity', target=j.rsi, return_value='7419/pgis')
        self.build = self.patch('build_and_check_store', target=j.rsi, return_value={
            'points': 3, 'dates': 2, 'bytes': 1234, 'max_abs_diff_cm_per_yr': 1e-9, 'build_s': 0.1, 'check_s': 0.1,
            'levels_source': {'kind': 'tippecanoe-droprate', 'rate': 2.5, 'base_zoom': 14}})
        self.j_globals = mock.patch.multiple(j, dbUsername='insaradmin', dbPassword='pw', dbHost='127.0.0.1')
        self.j_globals.start()
        self.addCleanup(self.j_globals.stop)

    def patch(self, name, target=None, **kw):
        p = mock.patch.object(target if target is not None else j, name.split('.')[-1], **kw)
        m = p.start()
        self.addCleanup(p.stop)
        return m

    def upload(self, **kw):
        out, err = io.StringIO(), io.StringIO()
        code = None
        with redirect_stdout(out), redirect_stderr(err):
            try:
                j.upload_points(self.dir, self.root, self.tools, 0.0, **kw)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_load_check_index_store(self):
        code, out, err = self.upload()
        self.assertIsNone(code, err)
        (con, area_id, p, lon, lat, m, d, attributes, types), _ = self.load.call_args
        self.assertEqual((con, area_id, list(p), attributes, types), (self.ctrl.con, '8', [0, 1, 2], {}, {}))
        self.rows.assert_called_once_with(self.ctrl.con, '8')
        self.ctrl.index_table_on.assert_called_once_with('8', 'p', None)
        self.clustered.assert_called_once_with(self.ctrl.con, '8')
        self.build.assert_called_once_with(
            8, 'DS', 3, 2, 0, self.dir, "host='127.0.0.1' dbname='pgis' user='insaradmin' connect_timeout=30", 'pw',
            self.root, self.tools, 0.0)
        self.assertIn('drop rate 2.5 and base zoom 14', out)
        self.assertIn('Seconds:', out)
        self.ctrl.remove_dataset.assert_not_called()                     # there was none
        with open(os.path.join(self.root, j.rsi.MARKER)) as f:
            self.assertEqual(json.load(f)['database'], '7419/pgis')       # the folder now names its database

    def test_a_new_upload_removes_the_old_store(self):
        self.ctrl.get_dataset_id.side_effect = [7, 8]
        make_store(self.root, 7, 'DS')
        code, out, err = self.upload()
        self.assertIsNone(code, err)
        self.assertFalse(os.path.exists(os.path.join(self.root, '7')))
        names = [c[0] for c in self.ctrl.mock_calls]
        self.assertLess(names.index('remove_dataset'), names.index('insert_dataset_into_area_table'))
        self.assertEqual(self.build.call_args.args[0], 8)

    def test_lost_points_remove_the_dataset_and_fail(self):
        self.rows.return_value = (2, '0.001', '0.001')
        code, out, err = self.upload()
        self.assertEqual(code, 1)
        self.assertIn('2 rows of the 3 points', err)
        self.assertEqual([c.args for c in self.ctrl.remove_dataset.call_args_list], [('DS',)])
        self.ctrl.index_table_on.assert_not_called()
        self.build.assert_not_called()

    def test_a_failed_load_removes_the_dataset_and_fails(self):
        self.load.side_effect = Exception('COPY failed: disk full')
        code, out, err = self.upload()
        self.assertEqual(code, 1)
        self.assertIn('COPY failed: disk full', err)
        self.ctrl.con.rollback.assert_called()
        self.assertEqual([c.args for c in self.ctrl.remove_dataset.call_args_list], [('DS',)])
        self.build.assert_not_called()

    def test_a_store_failure_fails_and_keeps_the_dataset(self):
        self.build.side_effect = j.rsi.StoreError('the recolor store failed its check against the table (x)')
        code, out, err = self.upload()
        self.assertEqual(code, 1)
        self.assertIn('migrate.py --only 8', err)
        self.ctrl.remove_dataset.assert_not_called()                     # the loaded dataset stays

    def test_json_from_an_older_step_1_is_refused_before_the_database(self):
        for name in os.listdir(self.dir):
            if name != 'metadata.pickle':
                os.remove(os.path.join(self.dir, name))
        with open(os.path.join(self.dir, 'chunk_0.json'), 'w') as f:
            f.write('{}')
        code, out, err = self.upload()
        self.assertEqual(code, 1)
        self.assertIn('JSON chunk files of an older step 1', err)
        self.assertIn('git pull', err)
        j.InsarDatabaseController.assert_not_called()

    def test_no_points_fails_before_the_database(self):
        os.remove(os.path.join(self.dir, 'points.json'))
        code, out, err = self.upload()
        self.assertEqual(code, 1)
        self.assertIn('run step 1 first', err)
        j.InsarDatabaseController.assert_not_called()

    def test_points_of_other_dates_than_the_metadata_fail(self):
        write_metadata(self.dir, dates=3)
        code, out, err = self.upload()
        self.assertEqual(code, 1)
        self.assertIn('2 dates, its metadata.pickle 3', err)
        j.InsarDatabaseController.assert_not_called()

    def test_a_missing_root_fails_before_the_database(self):
        self.root = os.path.join(self.root, 'missing')
        code, out, err = self.upload()
        self.assertEqual(code, 1)
        self.assertIn('does not exist', err)
        j.InsarDatabaseController.assert_not_called()

    def test_the_root_of_another_database_fails_before_any_removal(self):
        for i in range(100, 130):
            make_store(self.root, i, 'X%d' % i)
        code, out, err = self.upload()
        self.assertEqual(code, 1)
        self.assertIn('store root of another database', err)
        self.ctrl.remove_dataset.assert_not_called()
        self.assertEqual(len(os.listdir(self.root)), 30)

    def test_a_failed_replacement_stops_before_uploading(self):
        self.ctrl.get_dataset_id.side_effect = [7, 8]
        self.ctrl.remove_dataset.side_effect = Exception('connection lost')
        make_store(self.root, 7, 'DS')
        code, out, err = self.upload()
        self.assertEqual(code, 1)
        self.assertIn('connection lost', err)
        self.ctrl.insert_dataset_into_area_table.assert_not_called()
        self.load.assert_not_called()
        self.assertTrue(os.path.isdir(os.path.join(self.root, '7')))     # the old dataset's store stays with it

    def test_the_replaced_store_counts_as_free_space(self):
        self.ctrl.get_dataset_id.side_effect = [7, 8]
        make_store(self.root, 7, 'DS')
        with open(os.path.join(self.root, '7', 'ydm.f32'), 'wb') as f:
            f.write(b'\0' * 5000)
        old = j.rsi.store_bytes(os.path.join(self.root, '7'))
        with mock.patch.object(j.rsi, 'check_space') as space:
            code, out, err = self.upload()
        self.assertIsNone(code, err)
        self.assertEqual(space.call_args.args[1], j.rsi.estimate_store_bytes(3, 2, 0) - old)

    def test_a_folder_of_another_database_fails_before_any_removal(self):
        self.ctrl.get_dataset_id.side_effect = [7, 8]
        make_store(self.root, 7, 'DS')
        with open(os.path.join(self.root, j.rsi.MARKER), 'w') as f:
            json.dump({'database': '1234/pgis'}, f)
        code, out, err = self.upload()
        self.assertEqual(code, 1)
        self.assertIn('1234/pgis', err)
        self.ctrl.remove_dataset.assert_not_called()
        self.assertTrue(os.path.isdir(os.path.join(self.root, '7')))

    def test_remove_dataset_and_store(self):
        ctrl = mock.MagicMock()
        ctrl.get_dataset_id.return_value = 7
        make_store(self.root, 7, 'DS')
        with redirect_stdout(io.StringIO()):
            j.remove_dataset_and_store(ctrl, 'DS', self.root)
        ctrl.remove_dataset.assert_called_once_with('DS')
        self.assertEqual(os.listdir(self.root), [j.rsi.MARKER])


if __name__ == '__main__':
    unittest.main()
