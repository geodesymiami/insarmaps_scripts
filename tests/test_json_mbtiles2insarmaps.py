"""Tests of json_mbtiles2insarmaps.upload_json's order of work, with the database and ogr2ogr mocked: no database, no
network. Needs the ingest's Python environment (psycopg2, pycurl, requests, mintpy), which add_attribute_insarmaps.py
imports. Run from the repository's root: python3 -m unittest discover -s tests"""
import io
import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import json_mbtiles2insarmaps as j
except ImportError as e:     # not the ingest's environment
    j = None
    IMPORT_ERROR = e


def feature(p):
    return json.dumps({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [-91.1, -0.8]},
                       'properties': {'d': [0.0, 0.001], 'm': 0.001, 'p': p}})


def make_store(root, area_id, name):
    d = os.path.join(root, str(area_id))
    os.mkdir(d)
    with open(os.path.join(d, 'meta.json'), 'w') as f:
        json.dump({'build': 'ab' * 16, 'dataset': {'area_id': area_id, 'unavco_name': name}}, f)


@unittest.skipIf(j is None, 'needs the ingest environment (%s)' % (None if j else IMPORT_ERROR))
class Upload(unittest.TestCase):
    def setUp(self):
        self.dir, self.root, self.tools = (tempfile.mkdtemp() for _ in range(3))
        for d in (self.dir, self.root, self.tools):
            self.addCleanup(shutil.rmtree, d, True)
        for name in ('convert_from_postgres.py', 'check_against_postgres.py'):
            open(os.path.join(self.tools, name), 'w').close()
        meta = {'area': 'DS', 'project_name': 'P', 'mid_long': -91.1, 'mid_lat': -0.8, 'country': 'None',
                'region': 'R', 'chunk_num': 1, 'attribute_keys': ['mission'], 'attribute_values': ['S1'],
                'string_dates_sql': '{20160605,20160629}', 'decimal_dates_sql': '{2016.427,2016.493}',
                'attributes': {'mission': 'S1'}, 'needed_attributes': {'mission'}}
        with open(os.path.join(self.dir, 'metadata.pickle'), 'wb') as f:
            pickle.dump(meta, f)
        with open(os.path.join(self.dir, 'chunk_0.json'), 'w') as f:
            f.write('\n'.join([feature(0), feature(1)]))
        with open(os.path.join(self.dir, 'chunk_1.json'), 'w') as f:
            f.write(feature(2))
        self.ctrl = mock.MagicMock(name='controller')
        self.ctrl.get_dataset_id.side_effect = [-1, 8]           # no old dataset; the new one gets id 8
        self.ctrl.remove_dataset.side_effect = [Exception('Dataset DS is not on insarmaps'), None]
        self.patch('InsarDatabaseController', return_value=self.ctrl)
        self.system = self.patch('os.system', target=j.os, return_value=0)
        self.rows = self.patch('rows_and_m_extents', target=j.rsi, return_value=(3, '0.001', '0.001'))
        self.area_ids = self.patch('area_ids', target=j.rsi, return_value={1, 2})
        self.build = self.patch('build_and_check_store', target=j.rsi, return_value={
            'points': 3, 'dates': 2, 'bytes': 1234, 'max_abs_diff_cm_per_yr': 1e-9, 'build_s': 0.1, 'check_s': 0.1})
        self.j_globals = mock.patch.multiple(j, dbUsername='insaradmin', dbPassword='pw', dbHost='127.0.0.1')
        self.j_globals.start()
        self.addCleanup(self.j_globals.stop)

    def patch(self, name, target=None, **kw):
        p = mock.patch.object(target if target is not None else j, name.split('.')[-1], **kw)
        m = p.start()
        self.addCleanup(p.stop)
        return m

    def upload(self):
        out, err = io.StringIO(), io.StringIO()
        code = None
        with redirect_stdout(out), redirect_stderr(err):
            try:
                j.upload_json(self.dir, 2, self.root, self.tools, 0.0)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_load_check_index_store(self):
        code, out, err = self.upload()
        self.assertIsNone(code, err)
        commands = [c.args[0] for c in self.system.call_args_list]
        self.assertEqual(len(commands), 2)
        self.assertIn('-lco LAUNDER=NO', commands[0])
        self.assertIn('chunk_0.json', commands[0])
        self.assertIn('-nln 8 ', commands[0])
        self.rows.assert_called_once_with(self.ctrl.con, '8')
        self.ctrl.index_table_on.assert_called_once_with('8', 'p', None)
        self.ctrl.cluster_table_using.assert_called_once_with('8', '8_p_idx')
        self.build.assert_called_once_with(8, 'DS', 3, 2, 0, "host='127.0.0.1' dbname='pgis' user='insaradmin'",
                                           'pw', self.root, self.tools, 0.0)
        self.assertIn('Seconds:', out)

    def test_a_new_upload_removes_the_old_store(self):
        self.ctrl.get_dataset_id.side_effect = [7, 8]
        self.ctrl.remove_dataset.side_effect = None
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
        self.assertIn('2 rows of the 3 features', err)
        self.assertEqual([c.args for c in self.ctrl.remove_dataset.call_args_list], [('DS',), ('DS',)])
        self.ctrl.index_table_on.assert_not_called()
        self.build.assert_not_called()

    def test_a_store_failure_fails_and_keeps_the_dataset(self):
        self.build.side_effect = j.rsi.StoreError('the recolor store failed its check against the table (x)')
        code, out, err = self.upload()
        self.assertEqual(code, 1)
        self.assertIn('migrate.py --only 8', err)
        self.assertEqual(self.ctrl.remove_dataset.call_count, 1)        # only the clearing of an old upload

    def test_no_chunks_fails_before_the_database(self):
        for name in ('chunk_0.json', 'chunk_1.json'):
            os.remove(os.path.join(self.dir, name))
        code, out, err = self.upload()
        self.assertEqual(code, 1)
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

    def test_remove_dataset_and_store(self):
        ctrl = mock.MagicMock()
        ctrl.get_dataset_id.return_value = 7
        make_store(self.root, 7, 'DS')
        with redirect_stdout(io.StringIO()):
            j.remove_dataset_and_store(ctrl, 'DS', self.root)
        ctrl.remove_dataset.assert_called_once_with('DS')
        self.assertEqual(os.listdir(self.root), [])


if __name__ == '__main__':
    unittest.main()
