"""Step 1 end to end on tiny inputs: hdfeos5_2json_mbtiles.py and hdfeos5_or_csv_2json_mbtiles.py write the points
(points_bundle.py) and metadata.pickle, and nothing else: no JSON, no .mbtiles, no tippecanoe. No network (the country
lookup is stubbed).
Needs the ingest's Python environment (mintpy, h5py, pandas, geocoder). Run from the repository's root:
python3 -m unittest discover -s tests"""
import math
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPTIONAL = {'mintpy', 'h5py', 'pandas', 'geocoder', 'numpy'}
try:
    import numpy as np
    import h5py
    import hdfeos5_2json_mbtiles as he5_script
    import hdfeos5_or_csv_2json_mbtiles as csv_script
    import points_bundle as pb
except ImportError as e:
    if (e.name or '').split('.')[0] not in OPTIONAL:
        raise
    he5_script = csv_script = None
    IMPORT_ERROR = e

DATES = ['20200101', '20200113', '20200125']
T = np.array([2020.0, 12 / 365.0 + 2020, 24 / 365.0 + 2020]) if he5_script else None    # the scripts' decimal dates


def old_fit(t, y):
    """The velocity as the scripts' JSON computed it for a point with gaps: lstsq over the dates it has."""
    return np.linalg.lstsq(np.vstack([t, np.ones(len(t))]).T, y, rcond=None)[0][0]


def write_he5(path):
    """A radar-coded (no X_STEP) HDF-EOS5 file of 3 x 4 cells and 3 dates, as MintPy writes it: cell (0, 1) masked,
    cell (2, 3) missing at the last date."""
    length, width = 3, 4
    disp = np.arange(len(DATES) * length * width, dtype=np.float32).reshape(len(DATES), length, width) / 1000
    disp[2, 2, 3] = np.nan
    mask = np.ones((length, width), dtype=bool)
    mask[0, 1] = False
    lat, lon = np.meshgrid(np.linspace(-0.80, -0.82, length), np.linspace(-91.10, -91.13, width), indexing='ij')
    with h5py.File(path, 'w') as f:
        for k, v in {'LENGTH': str(length), 'WIDTH': str(width), 'PROJECT_NAME': 'unittestGalapagosSenDT128',
                     'mission': 'S1', 'REF_LAT': '-0.81', 'REF_LON': '-91.11', 'first_date': '2020-01-01',
                     'last_date': '2020-01-25', 'processing_software': 'isce'}.items():
            f.attrs[k] = v
        g = f.create_group('HDFEOS/GRIDS/timeseries')
        g.create_dataset('observation/date', data=np.array(DATES, dtype='S8'))
        g.create_dataset('observation/bperp', data=np.zeros(len(DATES), dtype=np.float32))
        g.create_dataset('observation/displacement', data=disp)
        g.create_dataset('quality/mask', data=mask)
        g.create_dataset('geometry/latitude', data=lat.astype(np.float32))
        g.create_dataset('geometry/longitude', data=lon.astype(np.float32))
    return disp, mask


@unittest.skipIf(he5_script is None, 'needs the ingest environment (%s)' % (None if he5_script else IMPORT_ERROR))
class Step1(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)

    def run_script(self, module, args):
        with mock.patch.object(sys, 'argv', ['script'] + args), \
                mock.patch.object(module.geocoder, 'google', side_effect=RuntimeError('no network in tests')), \
                mock.patch('os.system') as system:
            module.main()
        system.assert_not_called()                      # tippecanoe or anything else
        out = args[1]
        names = sorted(os.listdir(out))
        self.assertEqual([n for n in names if not n.startswith('attr_')],
                         ['d.npy', 'lat.npy', 'lon.npy', 'm.npy', 'metadata.pickle', 'p.npy', 'points.json'])
        with open(os.path.join(out, 'metadata.pickle'), 'rb') as f:
            return pb.read_points(out), pickle.load(f)

    def check_he5(self, module, out):
        he5 = os.path.join(self.dir, 'S1_IW1_128_20200101_20200125.he5')
        disp, mask = write_he5(he5)
        (manifest, p, lon, lat, m, d, attrs), meta = self.run_script(module, [he5, out, '--num-workers', '1'])
        self.assertEqual(meta['area'], 'S1_IW1_128_20200101_20200125')
        self.assertEqual(meta['string_dates_sql'], '{' + ','.join(DATES) + '}')
        self.assertEqual((manifest['points'], manifest['dates'], manifest['d_dtype']), (11, 3, 'float32'))
        cells = [k for k in range(12) if k != 1]        # 12 cells, (0, 1) masked
        np.testing.assert_array_equal(d, disp.reshape(3, -1)[:, cells].T)       # the file's values, NaN included
        k = cells.index(11)                             # cell (2, 3): missing at the last date
        self.assertTrue(math.isnan(d[k, 2]))
        self.assertEqual(m[k], old_fit(T[:2], d[k, :2].astype(np.float64)))
        self.assertEqual(int(np.isnan(m).sum()), 0)     # every point's m is fitted over the dates it has
        return p

    def test_he5_with_hdfeos5_2json_mbtiles(self):
        p = self.check_he5(he5_script, os.path.join(self.dir, 'JSON'))
        self.assertEqual(list(p), list(range(11)))      # counted from the first cell of each 20,000

    def test_he5_with_hdfeos5_or_csv_2json_mbtiles(self):
        p = self.check_he5(csv_script, os.path.join(self.dir, 'JSON'))
        self.assertEqual(list(p), [0] + list(range(2, 12)))         # the flat index

    def test_csv(self):
        csv = os.path.join(self.dir, 'TSX_036_20200101_20200125_N2579W08019.csv')
        with open(csv, 'w') as f:
            f.write('X,Y,coherence,D20200101,D20200113,D20200125\n'
                    '-80.19,25.79,0.9,0.0,1.5,3.0\n'
                    '-80.18,25.78,0.8,0.0,-2.0,-4.0\n')
        out = os.path.join(self.dir, 'JSON')
        (manifest, p, lon, lat, m, d, attrs), meta = self.run_script(csv_script, [csv, out, '--num-workers', '1'])
        self.assertEqual(meta['area'], 'TSX_036_20200101_20200125_N2579W08019')
        self.assertEqual(list(p), [0, 1])
        self.assertEqual(list(d[0]), [0.0, 0.0015, 0.003])                       # mm to m, float64
        self.assertEqual(manifest['d_dtype'], 'float64')
        self.assertEqual(attrs['coherence'][1], 0.8)
        self.assertEqual(list(lon), [-80.19, -80.18])

    def test_csv_with_gaps_and_quality_columns(self):
        csv = os.path.join(self.dir, 'TSX_036_20200101_20200125_N2579W08019_gaps.csv')
        with open(csv, 'w') as f:
            f.write('X,Y,coherence,point_ID,D20200101,D20200113,D20200125\n'
                    '-80.19,25.79,0.9,11,0.0,1.5,3.0\n'
                    '-80.18,25.78,,12,,-2.0,-4.0\n'          # no first value: a point now; no coherence
                    '-80.17,25.77,0.7,13,,,5.0\n'            # one value: not a point
                    '-80.16,25.76,0.6,,1.0,2.0,\n')          # no point_ID, no last value
        out = os.path.join(self.dir, 'JSON')
        (manifest, p, lon, lat, m, d, attrs), meta = self.run_script(csv_script, [csv, out])
        self.assertEqual(list(p), [0, 1, 3])                    # the CSV's rows
        self.assertEqual(manifest['attributes'], ['coherence', 'point_ID'])
        self.assertEqual(manifest['attribute_types'], {'coherence': 'double precision', 'point_ID': 'integer'})
        self.assertTrue(math.isnan(d[1, 0]) and math.isnan(d[2, 2]))
        self.assertTrue(math.isnan(attrs['coherence'][1]) and math.isnan(attrs['point_ID'][2]))
        self.assertEqual(attrs['point_ID'][0], 11.0)
        self.assertEqual(m[1], old_fit(T[1:], d[1, 1:]))


if __name__ == '__main__':
    unittest.main()
