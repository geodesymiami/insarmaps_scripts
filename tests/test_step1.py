"""Step 1 end to end on tiny inputs: insarmaps_ingester.py writes the points (points_bundle.py) and metadata.pickle, and
nothing else: no JSON, no .mbtiles, no tippecanoe. An HDF-EOS5 file gets the points and metadata
hdfeos5_2json_mbtiles.py wrote (with --he5-like-csv, what hdfeos5_or_csv_2json_mbtiles.py wrote for one) and its
temporal coherence as the points' attribute coherence, a CSV what hdfeos5_or_csv_2json_mbtiles.py wrote. No network
(the country lookup is stubbed).
Needs the ingest's Python environment (mintpy, h5py, pandas, geocoder). Run from the repository's root:
python3 -m unittest discover -s tests"""
import io
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
    import insarmaps_ingester as ingester
    import points_bundle as pb
except ImportError as e:
    if (e.name or '').split('.')[0] not in OPTIONAL:
        raise
    ingester = None
    IMPORT_ERROR = e

DATES = ['20200101', '20200113', '20200125']
T = np.array([2020.0, 12 / 365.0 + 2020, 24 / 365.0 + 2020]) if ingester else None    # the scripts' decimal dates
GRID = {'X_FIRST': '-91.10', 'Y_FIRST': '-0.80', 'X_STEP': '-0.01', 'Y_STEP': '-0.01'}
CELLS = [k for k in range(12) if k != 1]            # write_he5's 12 cells but (0, 1), which is masked: its points
# a temporalCoherence layer for write_he5, as MintPy writes one: low at the cell its mask hides, (0, 1); here also NaN
# at the point (1, 2)
COHERENCE = np.array([[0.91, 0.32, 0.88, 0.97],
                      [0.75, 0.62, np.nan, 0.99],
                      [0.58, 0.83, 0.95, 0.71]], dtype=np.float32) if ingester else None


def old_fit(t, y):
    """The velocity as the scripts' JSON computed it for a point with gaps: lstsq over the dates it has."""
    return np.linalg.lstsq(np.vstack([t, np.ones(len(t))]).T, y, rcond=None)[0][0]


def write_he5(path, extra_attributes=None, coherence=None):
    """An HDF-EOS5 file of 3 x 4 cells and 3 dates, as MintPy writes it, radar-coded (no X_STEP) unless
    extra_attributes adds GRID: cell (0, 1) masked, cell (2, 3) missing at the last date; coherence, when given, is
    its temporalCoherence layer, which it has none of otherwise."""
    length, width = 3, 4
    disp = np.arange(len(DATES) * length * width, dtype=np.float32).reshape(len(DATES), length, width) / 1000
    disp[2, 2, 3] = np.nan
    mask = np.ones((length, width), dtype=bool)
    mask[0, 1] = False
    lat, lon = np.meshgrid(np.linspace(-0.80, -0.82, length), np.linspace(-91.10, -91.13, width), indexing='ij')
    attributes = {'LENGTH': str(length), 'WIDTH': str(width), 'PROJECT_NAME': 'unittestGalapagosSenDT128',
                  'mission': 'S1', 'REF_LAT': '-0.81', 'REF_LON': '-91.11', 'first_date': '2020-01-01',
                  'last_date': '2020-01-25', 'processing_software': 'isce', 'unwrap_method': 'Unknown'}
    attributes.update(extra_attributes or {})
    with h5py.File(path, 'w') as f:
        for k, v in attributes.items():
            f.attrs[k] = v
        g = f.create_group('HDFEOS/GRIDS/timeseries')
        g.create_dataset('observation/date', data=np.array(DATES, dtype='S8'))
        g.create_dataset('observation/bperp', data=np.zeros(len(DATES), dtype=np.float32))
        g.create_dataset('observation/displacement', data=disp)
        g.create_dataset('quality/mask', data=mask)
        if coherence is not None:
            g.create_dataset('quality/temporalCoherence', data=coherence)
        g.create_dataset('geometry/latitude', data=lat.astype(np.float32))
        g.create_dataset('geometry/longitude', data=lon.astype(np.float32))
    return disp, lat.astype(np.float32), lon.astype(np.float32)


@unittest.skipIf(ingester is None, 'needs the ingest environment (%s)' % (None if ingester else IMPORT_ERROR))
class Step1(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)

    def run_script(self, args):
        """Runs the script; returns the points and the metadata, and keeps what it printed in self.printed."""
        with mock.patch.object(sys, 'argv', ['insarmaps_ingester.py'] + args), \
                mock.patch.object(ingester.geocoder, 'google', side_effect=RuntimeError('no network in tests')), \
                mock.patch('os.system') as system, mock.patch('sys.stdout', new_callable=io.StringIO) as printed:
            ingester.main()
        self.printed = printed.getvalue()
        system.assert_not_called()                      # tippecanoe or anything else
        out = args[1]
        names = sorted(os.listdir(out))
        self.assertEqual([n for n in names if not n.startswith('attr_')],
                         ['d.npy', 'lat.npy', 'lon.npy', 'm.npy', 'metadata.pickle', 'p.npy', 'points.json'])
        with open(os.path.join(out, 'metadata.pickle'), 'rb') as f:
            return pb.read_points(out), pickle.load(f)

    def run_he5(self, extra_attributes=None, options=('--num-workers', '1'), coherence=None):
        """Runs the script on write_he5's file; checks what every HDF-EOS5 run writes the same way and returns p, the
        metadata, and the file's lat/lon grids. Keeps the manifest and the points' attributes in self.manifest and
        self.attrs: none for a file without a temporalCoherence layer."""
        he5 = os.path.join(self.dir, 'S1_IW1_128_20200101_20200125.he5')
        disp, lat, lon = write_he5(he5, extra_attributes, coherence)
        (manifest, p, _, _, m, d, attrs), meta = self.run_script([he5, os.path.join(self.dir, 'JSON'), *options])
        self.manifest, self.attrs = manifest, attrs
        self.assertEqual(meta['area'], 'S1_IW1_128_20200101_20200125')
        self.assertEqual(meta['string_dates_sql'], '{' + ','.join(DATES) + '}')
        self.assertEqual((manifest['points'], manifest['dates'], manifest['d_dtype']), (11, 3, 'float32'))
        np.testing.assert_array_equal(d, disp.reshape(3, -1)[:, CELLS].T)       # the file's values, NaN included
        k = CELLS.index(11)                             # cell (2, 3): missing at the last date
        self.assertTrue(math.isnan(d[k, 2]))
        self.assertEqual(m[k], old_fit(T[:2], d[k, :2].astype(np.float64)))
        self.assertEqual(int(np.isnan(m).sum()), 0)     # every point's m is fitted over the dates it has
        if coherence is None:
            self.assertEqual(attrs, {})
            self.assertIn('temporal coherence: none in the file', self.printed)
        return list(p), meta, lat, lon

    def check_like_hdfeos5_2json_mbtiles(self, p, meta):
        """The points and metadata hdfeos5_2json_mbtiles.py wrote: p counted from the first cell of each 20,000; no
        "dem", no unwrap_method (its list had lost it to a missing comma), no country when the lookup fails."""
        self.assertEqual(p, list(range(11)))
        self.assertIsNone(meta['country'])
        self.assertNotIn('dem', meta['attribute_keys'])
        self.assertNotIn('unwrap_method', meta['needed_attributes'])
        self.assertEqual(meta['attribute_values'], [meta['attributes'][k] for k in meta['attribute_keys']])
        self.assertNotIn('collection', meta['attributes'])

    def check_like_hdfeos5_or_csv_2json_mbtiles(self, p, meta, lat, lon):
        """What hdfeos5_or_csv_2json_mbtiles.py wrote for an HDF-EOS5 file: p the flat index, and a CSV's metadata:
        "dem" last, the mean position of all the cells, "None" when the country lookup fails."""
        self.assertEqual(p, CELLS)
        self.assertEqual((meta['attribute_keys'][-1], meta['attribute_values'][-1]), ('dem', '1'))
        self.assertEqual(meta['attribute_values'][:-1], [meta['attributes'][k] for k in meta['attribute_keys'][:-1]])
        self.assertEqual(meta['country'], 'None')
        self.assertEqual(meta['attributes']['collection'], 'hdfeos5')
        self.assertEqual((meta['mid_lat'], meta['mid_long']),
                         (float(np.nanmean(lat.reshape(-1))), float(np.nanmean(lon.reshape(-1)))))

    def test_he5_radar_coded(self):
        p, meta, lat, lon = self.run_he5()
        self.check_like_hdfeos5_2json_mbtiles(p, meta)
        self.assertEqual(set(meta['attribute_keys']), {'LENGTH', 'WIDTH', 'mission', 'REF_LAT', 'REF_LON',
                                                       'first_date', 'last_date', 'processing_software'})
        self.assertEqual((meta['mid_lat'], meta['mid_long']), (float(lat[1, 2]), float(lon[1, 2])))    # middle cell

    def test_he5_geocoded(self):
        p, meta, _, _ = self.run_he5(GRID)
        self.check_like_hdfeos5_2json_mbtiles(p, meta)
        self.assertEqual(set(meta['attribute_keys']), {'LENGTH', 'WIDTH', 'mission', 'REF_LAT', 'REF_LON',
                                                       'first_date', 'last_date', 'processing_software', *GRID})
        self.assertEqual((meta['mid_lat'], meta['mid_long']), (-0.80 + 1.5 * -0.01, -91.10 + 2 * -0.01))  # grid's

    def test_he5_like_csv_radar_coded(self):
        p, meta, lat, lon = self.run_he5(options=['--he5-like-csv'])
        self.check_like_hdfeos5_or_csv_2json_mbtiles(p, meta, lat, lon)
        self.assertEqual(set(meta['attribute_keys'][:-1]), {'mission', 'REF_LAT', 'REF_LON', 'first_date',
                                                            'last_date', 'processing_software', 'unwrap_method'})

    def test_he5_like_csv_geocoded(self):
        p, meta, lat, lon = self.run_he5(GRID, options=['--he5-like-csv'])
        self.check_like_hdfeos5_or_csv_2json_mbtiles(p, meta, lat, lon)
        self.assertEqual(set(meta['attribute_keys'][:-1]), {'LENGTH', 'WIDTH', 'mission', 'REF_LAT', 'REF_LON',
                                                            'first_date', 'last_date', 'processing_software',
                                                            'unwrap_method', *GRID})

    def test_he5_temporal_coherence_is_the_points_coherence(self):
        """The points' attribute coherence is their cells' temporalCoherence, as the file holds it, NaN where a cell
        has none, with either numbering; the masked cell's value is no point's."""
        for options, numbers in ((('--num-workers', '1'), list(range(11))), (('--he5-like-csv',), CELLS)):
            with self.subTest(options=options):
                p, meta, _, _ = self.run_he5(GRID, options=options, coherence=COHERENCE)
                self.assertEqual(p, numbers)
                self.assertEqual(self.manifest['attributes'], ['coherence'])
                self.assertEqual(self.manifest['attribute_types'], {'coherence': 'double precision'})
                np.testing.assert_array_equal(self.attrs['coherence'],
                                              COHERENCE.reshape(-1)[CELLS].astype(np.float64))
                self.assertTrue(math.isnan(self.attrs['coherence'][CELLS.index(6)]))       # cell (1, 2)
                self.assertNotIn(float(COHERENCE[0, 1]), list(self.attrs['coherence']))
                self.assertIn('temporal coherence: kept as the points\' attribute coherence', self.printed)
                self.assertNotIn('coherence', meta['attribute_keys'])           # not an attribute of the dataset

    def test_he5_without_a_coherence_value_at_its_points_gets_no_coherence(self):
        """No attribute when the file's temporalCoherence is all NaN (as in the horizontal and vertical
        decompositions), has values only where no point is, or is not of the grid's shape."""
        only_masked = np.full((3, 4), np.nan, dtype=np.float32)
        only_masked[0, 1] = 0.32
        for what, layer, message in (
                ('all NaN', np.full((3, 4), np.nan, dtype=np.float32), 'temporal coherence: no point has a value'),
                ('only where the mask hides the cell', only_masked, 'temporal coherence: no point has a value'),
                ('of another shape', COHERENCE[:2], 'temporalCoherence is (2, 4), not the grid\'s (3, 4)')):
            with self.subTest(what):
                p, _, _, _ = self.run_he5(coherence=layer)
                self.assertEqual(p, list(range(11)))
                self.assertEqual((self.manifest['attributes'], self.attrs), ([], {}))
                self.assertIn(message, self.printed)

    def test_csv(self):
        csv = os.path.join(self.dir, 'TSX_036_20200101_20200125_N2579W08019.csv')
        with open(csv, 'w') as f:
            f.write('X,Y,coherence,D20200101,D20200113,D20200125\n'
                    '-80.19,25.79,0.9,0.0,1.5,3.0\n'
                    '-80.18,25.78,0.8,0.0,-2.0,-4.0\n')
        out = os.path.join(self.dir, 'JSON')
        (manifest, p, lon, lat, m, d, attrs), meta = self.run_script([csv, out, '--num-workers', '1'])
        self.assertEqual(meta['area'], 'TSX_036_20200101_20200125_N2579W08019')
        self.assertEqual(list(p), [0, 1])
        self.assertEqual(list(d[0]), [0.0, 0.0015, 0.003])                       # mm to m, float64
        self.assertEqual(manifest['d_dtype'], 'float64')
        self.assertEqual(attrs['coherence'][1], 0.8)
        self.assertEqual(list(lon), [-80.19, -80.18])
        # the metadata hdfeos5_or_csv_2json_mbtiles.py wrote: "dem" last, the mean position, "None" for no country
        self.assertEqual((meta['attribute_keys'][-1], meta['attribute_values'][-1]), ('dem', '1'))
        self.assertEqual(meta['country'], 'None')
        self.assertAlmostEqual(meta['mid_lat'], 25.785)
        self.assertAlmostEqual(meta['mid_long'], -80.185)
        self.assertEqual(meta['project_name'], 'CSV_IMPORT')
        self.assertIn('unwrap_method', meta['needed_attributes'])
        self.assertFalse({'WIDTH', 'LENGTH', 'X_STEP'} & meta['needed_attributes'])

    def test_csv_with_gaps_and_quality_columns(self):
        csv = os.path.join(self.dir, 'TSX_036_20200101_20200125_N2579W08019_gaps.CSV')     # any case of .csv
        with open(csv, 'w') as f:
            f.write('X,Y,coherence,point_ID,D20200101,D20200113,D20200125\n'
                    '-80.19,25.79,0.9,11,0.0,1.5,3.0\n'
                    '-80.18,25.78,,12,,-2.0,-4.0\n'          # no first value: a point now; no coherence
                    '-80.17,25.77,0.7,13,,,5.0\n'            # one value: not a point
                    '-80.16,25.76,0.6,,1.0,2.0,\n')          # no point_ID, no last value
        out = os.path.join(self.dir, 'JSON')
        (manifest, p, lon, lat, m, d, attrs), meta = self.run_script([csv, out])
        self.assertEqual(list(p), [0, 1, 3])                    # the CSV's rows
        self.assertEqual(manifest['attributes'], ['coherence', 'point_ID'])
        self.assertEqual(manifest['attribute_types'], {'coherence': 'double precision', 'point_ID': 'integer'})
        self.assertTrue(math.isnan(d[1, 0]) and math.isnan(d[2, 2]))
        self.assertTrue(math.isnan(attrs['coherence'][1]) and math.isnan(attrs['point_ID'][2]))
        self.assertEqual(attrs['point_ID'][0], 11.0)
        self.assertEqual(m[1], old_fit(T[1:], d[1, 1:]))

    def test_csv_without_a_numeric_point_id_gets_no_point_id_column(self):
        csv = os.path.join(self.dir, 'EGMS_L2a_124_0851_IW1_VV_2020_2024_1.csv')
        with open(csv, 'w') as f:                       # EGMS: pid is text, dates are digit columns
            f.write('pid,latitude,longitude,height_ortho,temporal_coherence,20200104,20200110,20200116\n'
                    '1Y9bj6a9w0,37.637,15.149,246.4,0.67,0.0,1.0,2.0\n'
                    '1Y9bj6a9w1,37.638,15.150,250.1,0.71,0.0,-1.0,-2.0\n')
        out = os.path.join(self.dir, 'JSON')
        (manifest, p, lon, lat, m, d, attrs), meta = self.run_script([csv, out])
        self.assertEqual(manifest['attributes'], ['elevation', 'coherence'])     # as the old JSON's properties
        self.assertEqual(list(attrs['elevation']), [246.4, 250.1])


if __name__ == '__main__':
    unittest.main()
