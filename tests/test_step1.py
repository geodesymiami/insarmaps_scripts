"""Step 1 end to end on tiny inputs: hdfeos5_2json_mbtiles.py and hdfeos5_or_csv_2json_mbtiles.py write chunk_*.json
and metadata.pickle, and nothing else: no .mbtiles, no tippecanoe. No network (the country lookup is stubbed).
Needs the ingest's Python environment (mintpy, h5py, pandas, geocoder). Run from the repository's root:
python3 -m unittest discover -s tests"""
import glob
import json
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
except ImportError as e:
    if (e.name or '').split('.')[0] not in OPTIONAL:
        raise
    he5_script = csv_script = None
    IMPORT_ERROR = e

DATES = ['20200101', '20200113', '20200125']


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


def features(folder):
    out = []
    for name in sorted(glob.glob(os.path.join(folder, 'chunk_*.json'))):
        with open(name) as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


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
        self.assertEqual(sorted(n for n in os.listdir(out) if not n.startswith('chunk_')), ['metadata.pickle'])
        with open(os.path.join(out, 'metadata.pickle'), 'rb') as f:
            return features(out), pickle.load(f)

    def check_he5(self, module, out):
        he5 = os.path.join(self.dir, 'S1_IW1_128_20200101_20200125.he5')
        write_he5(he5)
        feats, meta = self.run_script(module, [he5, out, '--num-workers', '1'])
        self.assertEqual(meta['area'], 'S1_IW1_128_20200101_20200125')
        self.assertEqual(meta['string_dates_sql'], '{' + ','.join(DATES) + '}')
        self.assertEqual(len(feats), 11)                # 12 cells, one masked
        for ft in feats:
            self.assertEqual(len(ft['properties']['d']), 3)
        self.assertEqual(len({ft['properties']['p'] for ft in feats}), 11)
        return feats

    def test_he5_with_hdfeos5_2json_mbtiles(self):
        feats = self.check_he5(he5_script, os.path.join(self.dir, 'JSON'))
        gap = [ft['properties'] for ft in feats if any(v != v for v in ft['properties']['d'])]
        self.assertEqual(len(gap), 1)                   # the cell missing at the last date: NaN there, NaN m
        self.assertTrue(math.isnan(gap[0]['d'][2]) and math.isnan(gap[0]['m']))

    def test_he5_with_hdfeos5_or_csv_2json_mbtiles(self):
        feats = self.check_he5(csv_script, os.path.join(self.dir, 'JSON'))
        gap = [ft['properties'] for ft in feats if None in ft['properties']['d']]
        self.assertEqual(len(gap), 1)                   # null there, m fitted over the other dates
        self.assertIsNone(gap[0]['d'][2])
        self.assertTrue(math.isfinite(gap[0]['m']))

    def test_csv(self):
        csv = os.path.join(self.dir, 'TSX_036_20200101_20200125_N2579W08019.csv')
        with open(csv, 'w') as f:
            f.write('X,Y,coherence,D20200101,D20200113,D20200125\n'
                    '-80.19,25.79,0.9,0.0,1.5,3.0\n'
                    '-80.18,25.78,0.8,0.0,-2.0,-4.0\n')
        out = os.path.join(self.dir, 'JSON')
        feats, meta = self.run_script(csv_script, [csv, out, '--num-workers', '1'])
        self.assertEqual(meta['area'], 'TSX_036_20200101_20200125_N2579W08019')
        self.assertEqual([ft['properties']['p'] for ft in feats], [0, 1])
        self.assertEqual(feats[0]['properties']['d'], [0.0, 0.0015, 0.003])      # mm to m
        self.assertEqual(feats[1]['properties']['coherence'], 0.8)


if __name__ == '__main__':
    unittest.main()
