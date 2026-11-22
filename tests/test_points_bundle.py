"""Tests of points_bundle.py, the points step 1 hands to step 2: numbering, the rule for missing values, the velocity
fit, a grid's per-cell attributes, and the files (no database). Needs numpy. Run from the repository's root:
python3 -m unittest discover -s tests"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import numpy as np
    import points_bundle as pb
except ImportError as e:
    if (e.name or '').split('.')[0] != 'numpy':
        raise
    pb = None
    IMPORT_ERROR = e

T = [2016.0, 2016.1, 2016.25, 2016.5]


def grid(rows=3, cols=5, seed=2):
    """A displacement grid (4 dates x rows x cols), float32, like a masked HDF-EOS5 file."""
    rng = np.random.default_rng(seed)
    disp = (np.outer(T, rng.normal(0, 0.01, rows * cols)) + rng.normal(0, 1e-3, (4, rows * cols)))
    return disp.astype(np.float32).reshape(4, rows, cols)


@unittest.skipIf(pb is None, 'needs numpy (%s)' % (None if pb else IMPORT_ERROR))
class Points(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_a_cell_with_two_values_is_a_point(self):
        disp = grid()
        disp[:, 0, 1] = np.nan                      # masked
        disp[0, 0, 2] = np.nan                      # first date missing: still a point
        disp[:3, 1, 0] = np.nan                     # one value: not a point
        idx, p = pb.grid_points(disp, numbering='flat')
        self.assertEqual(list(idx), [0, 2, 3, 4] + list(range(6, 15)))
        self.assertEqual(list(p), list(idx))

    def test_block_numbering_counts_from_each_blocks_first_cell(self):
        disp = grid(rows=2, cols=5)
        disp[:, 0, 0] = np.nan
        disp[:, 1, 1] = np.nan
        idx, p = pb.grid_points(disp, block=4)      # cells 0-3, 4-7, 8-9
        self.assertEqual(list(idx), [1, 2, 3, 4, 5, 7, 8, 9])
        self.assertEqual(list(p), [0, 1, 2, 4, 5, 6, 8, 9])
        with self.assertRaises(ValueError):
            pb.grid_points(disp, numbering='rows')

    def test_velocity_is_fitted_over_the_dates_a_point_has(self):
        d = np.array([[0.0, 0.001, 0.002, 0.004],
                      [np.nan, 0.001, 0.002, 0.004],
                      [0.0, np.nan, np.nan, 0.004],
                      [np.nan, np.nan, np.nan, 0.004]])
        m = pb.fit_velocities(T, d)
        for k in range(3):
            f = np.isfinite(d[k])
            self.assertAlmostEqual(m[k], np.polyfit(np.array(T)[f], d[k][f], 1)[0], places=12)
        self.assertTrue(np.isnan(m[3]))

    def test_grid_points_round_trip(self):
        disp = grid()
        disp[:, 2, 4] = np.nan
        disp[3, 0, 0] = np.nan
        lats, lons = np.meshgrid(np.linspace(-0.8, -0.9, 3), np.linspace(-91.1, -91.2, 5), indexing='ij')
        P = pb.write_grid_points(self.dir, disp, lats.astype(np.float32), lons.astype(np.float32), T)
        self.assertEqual(P, 14)
        manifest, p, lon, lat, m, d, attrs = pb.read_points(self.dir)
        self.assertEqual((manifest['points'], manifest['dates'], manifest['d_dtype']), (14, 4, 'float32'))
        self.assertEqual(manifest['attributes'], [])
        idx = np.array([k for k in range(15) if k != 14])
        np.testing.assert_array_equal(d, disp.reshape(4, -1)[:, idx].T)                   # the file's values
        np.testing.assert_array_equal(lon, lons.astype(np.float32).astype(np.float64).reshape(-1)[idx])
        self.assertTrue(np.isnan(d[0, 3]) and np.isfinite(m[0]))                           # the gap, and its fit

    def test_grid_attributes_are_the_points_cells_values(self):
        disp = grid()
        disp[:, 0, 1] = np.nan                      # masked: its coherence is no point's
        lats, lons = np.meshgrid(np.linspace(-0.8, -0.9, 3), np.linspace(-91.1, -91.2, 5), indexing='ij')
        coherence = np.linspace(0.5, 0.99, 15, dtype=np.float32).reshape(3, 5)
        coherence[1, 3] = np.nan                    # a point without a value
        height = np.arange(15, dtype=np.float32).reshape(3, 5) * 10
        idx = np.array([k for k in range(15) if k != 1])
        for numbering, p_expected in (('blocks', list(range(14))), ('flat', list(idx))):
            with self.subTest(numbering):
                P = pb.write_grid_points(self.dir, disp, lats, lons, T, numbering=numbering,
                                         attributes={'coherence': coherence, 'height': height})
                manifest, p, lon, lat, m, d, attrs = pb.read_points(self.dir)
                self.assertEqual((P, list(p)), (14, p_expected))
                self.assertEqual(manifest['attributes'], ['coherence', 'height'])          # in the order given
                self.assertEqual(manifest['attribute_types'],
                                 {'coherence': 'double precision', 'height': 'double precision'})
                self.assertEqual(pb.read_manifest(self.dir), manifest)
                np.testing.assert_array_equal(attrs['coherence'],
                                              coherence.reshape(-1)[idx].astype(np.float64))   # NaN at (1, 3)
                self.assertTrue(np.isnan(attrs['coherence'][list(idx).index(8)]))
                np.testing.assert_array_equal(attrs['height'], height.reshape(-1)[idx])

    def test_a_grid_attribute_without_a_value_at_any_point_is_left_out(self):
        disp = grid()
        disp[:, 0, 1] = np.nan
        lats, lons = np.meshgrid(np.linspace(-0.8, -0.9, 3), np.linspace(-91.1, -91.2, 5), indexing='ij')
        decomposition = np.full((3, 5), np.nan, dtype=np.float32)       # a horizontal or vertical decomposition's
        only_masked = np.full((3, 5), np.nan, dtype=np.float32)
        only_masked[0, 1] = 0.3                                         # a value only where no point is
        pb.write_grid_points(self.dir, disp, lats, lons, T, attributes={'a': decomposition, 'b': only_masked,
                                                                        'c': np.ones((3, 5))})
        manifest, p, lon, lat, m, d, attrs = pb.read_points(self.dir)
        self.assertEqual(manifest['attributes'], ['c'])
        self.assertEqual(sorted(n for n in os.listdir(self.dir) if n.startswith('attr_')), ['attr_c.npy'])
        with self.assertRaisesRegex(pb.PointsError, 'attribute a is'):
            pb.write_grid_points(self.dir, disp, lats, lons, T, attributes={'a': np.ones((5, 3))})

    def test_points_with_attributes_round_trip(self):
        point_id = np.array([7.0, np.nan, 9.0])
        coherence = np.array([0.9, 0.8, np.nan])
        pb.write_points(self.dir, [0, 4, 5], [-80.1, -80.2, -80.3], [25.7, 25.8, 25.9], [0.1, 0.2, np.nan],
                        np.zeros((3, 2)), {'point_ID': point_id, 'coherence': coherence}, {'point_ID': 'integer'})
        manifest, p, lon, lat, m, d, attrs = pb.read_points(self.dir)
        self.assertEqual(manifest['attributes'], ['point_ID', 'coherence'])                # the CSV's order
        self.assertEqual(manifest['attribute_types'], {'point_ID': 'integer', 'coherence': 'double precision'})
        self.assertEqual(manifest['d_dtype'], 'float64')
        np.testing.assert_array_equal(attrs['point_ID'], point_id)
        self.assertEqual(list(attrs), ['point_ID', 'coherence'])

    def test_refusals(self):
        good = dict(p=[0, 1], lon=[1.0, 2.0], lat=[3.0, 4.0], m=[0.0, 0.0], d=np.zeros((2, 2)))
        for what, change, message in (
                ('p out of order', {'p': [1, 0]}, 'strictly increasing'),
                ('negative p', {'p': [-1, 0]}, 'non-negative'),
                ('no position', {'lon': [1.0, np.nan]}, 'not finite'),
                ('a pole', {'lat': [3.0, 90.0]}, 'lat'),
                ('m too short', {'m': [0.0]}, 'm is'),
                ('a bad attribute name', {'attributes': {'9x': [1.0, 2.0]}}, 'attribute name'),
                ('a fraction in an integer column', {'attributes': {'point_ID': [1.5, 2.0]},
                                                     'attribute_types': {'point_ID': 'integer'}}, 'int32')):
            with self.subTest(what):
                args = dict(good, **change)
                with self.assertRaisesRegex(pb.PointsError, message):
                    pb.write_points(self.dir, **args)
                self.assertFalse(pb.has_points(self.dir))             # nothing that step 2 would read

    def test_an_earlier_run_is_cleared_but_not_other_files(self):
        for name in ('chunk_0.json', 'chunk_12.json', 'p.npy', 'attr_old.npy', 'points.json', 'metadata.pickle',
                     'notes.txt', 'DS.mbtiles'):
            with open(os.path.join(self.dir, name), 'w') as f:
                f.write('x')
        pb.write_points(self.dir, [0, 1], [1.0, 2.0], [3.0, 4.0], [0.0, 0.0], np.zeros((2, 2)))
        self.assertEqual(sorted(os.listdir(self.dir)), ['DS.mbtiles', 'd.npy', 'lat.npy', 'lon.npy', 'm.npy',
                                                         'metadata.pickle', 'notes.txt', 'p.npy', 'points.json'])
        with open(os.path.join(self.dir, 'points.json')) as f:
            self.assertEqual(json.load(f)['format'], 'insarmaps-points')


if __name__ == '__main__':
    unittest.main()
