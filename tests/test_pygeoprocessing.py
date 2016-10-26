"""Module to test PyGeoprocessing module."""
import collections
import unittest
import os
import tempfile
import shutil

from osgeo import gdal
from osgeo import osr
import numpy

import pygeoprocessing.testing.assertions
import pygeoprocessing.testing.sampledata
from pygeoprocessing.testing import scm

TEST_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'pygeoprocessing-test-data')


class PyGeoprocessingTest(unittest.TestCase):
    """Class to test PyGeoprocessing's functions."""

    def setUp(self):
        """Setup workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_map_dataset_to_value_nodata_undefined(self):
        """PyGeoprocessing: test map_dataset_to_value missing nodata."""
        import pygeoprocessing

        n_rows, n_cols = 4, 4
        driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'test.tif')

        new_raster = driver.Create(
            raster_path, n_cols, n_rows, 1, gdal.GDT_Int32)
        band = new_raster.GetRasterBand(1)
        band.WriteArray(numpy.ones((n_rows, n_cols), dtype=numpy.int32))
        band.FlushCache()
        band = None
        new_raster = None

        out_nodata = -1.0
        value_map = {1: 100.0}
        raster_out_path = os.path.join(self.workspace_dir, 'test_out.tif')
        pygeoprocessing.reclassify_dataset_uri(
            raster_path, value_map, raster_out_path, gdal.GDT_Float64,
            out_nodata)

        raster_out = gdal.Open(raster_out_path)
        raster_out_band = raster_out.GetRasterBand(1)
        out_array = numpy.unique(raster_out_band.ReadAsArray())
        raster_out_band = None
        raster_out = None
        self.assertTrue(len(out_array))
        self.assertEqual(out_array[0], 100.0)

    def test_map_dataset_to_value(self):
        """PyGeoprocessing: test map_dataset_to_value for general case."""
        import pygeoprocessing

        n_rows, n_cols = 4, 4
        driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'test.tif')
        new_raster = driver.Create(
            raster_path, n_cols, n_rows, 1, gdal.GDT_Int32)
        band = new_raster.GetRasterBand(1)
        band.WriteArray(numpy.ones((n_rows, n_cols), dtype=numpy.int32))
        band.SetNoDataValue(-1)
        band.FlushCache()
        band = None
        new_raster = None

        out_nodata = -1.0
        value_map = {1: 100.0}
        raster_out_path = os.path.join(self.workspace_dir, 'test_out.tif')
        pygeoprocessing.reclassify_dataset_uri(
            raster_path, value_map, raster_out_path, gdal.GDT_Float64,
            out_nodata)

        raster_out = gdal.Open(raster_out_path)
        raster_out_band = raster_out.GetRasterBand(1)
        out_array = numpy.unique(raster_out_band.ReadAsArray())
        self.assertTrue(len(out_array))
        self.assertEqual(out_array[0], 100.0)

    def test_transform_bounding_box(self):
        """PyGeoprocessing: test bounding box transform."""
        import pygeoprocessing

        vector_extent = [
            440446.6938076447695494, 4800590.4052893081679940,
            606196.6938076447695494, 5087540.4052893081679940]
        expected_extents = [
            -123.76825632966793, 43.350664712678984, -121.63016515055192,
            45.941400531740214]
        # test from UTM 10N to WGS84
        base_ref = osr.SpatialReference()
        base_ref.ImportFromEPSG(26910)

        new_ref = osr.SpatialReference()
        new_ref.ImportFromEPSG(4326)
        actual_extents = pygeoprocessing.transform_bounding_box(
            vector_extent, base_ref.ExportToWkt(), new_ref.ExportToWkt(),
            edge_samples=11)
        numpy.testing.assert_array_almost_equal(
            expected_extents, actual_extents)

    @scm.skip_if_data_missing(TEST_DATA)
    def test_convolve_2d(self):
        """PyGeoprocessing: test convolve 2D regression."""
        import pygeoprocessing

        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        output_path = os.path.join(self.workspace_dir, 'output.tif')
        signal_array = numpy.ones([600, 600])
        kernel_array = numpy.ones([3000, 3000])

        # the following rasters use an arbitrary coordinate system simply
        # to get the rasters to create on disk to invoke `convolve_2d_uri`
        # the numerical output is not otherwise affected
        cs_wkt = pygeoprocessing.testing.sampledata.projection_wkt(3157)
        pygeoprocessing.testing.sampledata.create_raster_on_disk(
            [signal_array], (0, 0), cs_wkt, -1, (30, 30),
            format='GTiff', filename=signal_path)
        pygeoprocessing.testing.sampledata.create_raster_on_disk(
            [kernel_array], (0, 0), cs_wkt, -1, (30, 30),
            format='GTiff', filename=kernel_path)

        pygeoprocessing.convolve_2d_uri(
            signal_path, kernel_path, output_path)

        expected_output_path = os.path.join(
            TEST_DATA, 'convolution_2d_test_data', 'expected_output.tif')

        pygeoprocessing.testing.assertions.assert_rasters_equal(
            output_path, expected_output_path, 1e-6)

    @scm.skip_if_data_missing(TEST_DATA)
    def test_aggregate_raster_values(self):
        """PyGeoprocessing: test aggregate raster values."""
        import pygeoprocessing

        base_raster_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data', 'base_raster.tif')

        shapefile_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data', 'watershed.shp')

        result = pygeoprocessing.aggregate_raster_values_uri(
            base_raster_path, shapefile_path, shapefile_field='DN',
            all_touched=False, polygons_might_overlap=False)

        expected_result = pygeoprocessing.AggregatedValues(
            total={1: 3.0, 2: 398425.0},
            pixel_mean={1: 1.0, 2: 1.0},
            hectare_mean={1: 11.111111110950143, 2: 11.111111110937156},
            n_pixels={1: 3.0, 2: 398425.0},
            pixel_min={1: 1.0, 2: 1.0},
            pixel_max={1: 1.0, 2: 1.0}
        )

        for metric in [
                'total', 'pixel_mean', 'hectare_mean', 'n_pixels',
                'pixel_min', 'pixel_max']:
            _assert_deep_almost_equal(
                self, getattr(expected_result, metric),
                getattr(result, metric), places=6)

    @scm.skip_if_data_missing(TEST_DATA)
    def test_aggregate_raster_values_include_nodata(self):
        """PyGeoprocessing: test aggregate raster values, include nodata."""
        import pygeoprocessing

        base_raster_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data', 'base_raster.tif')

        shapefile_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data',
            'overlap_watershed.shp')

        result = pygeoprocessing.aggregate_raster_values_uri(
            base_raster_path, shapefile_path, shapefile_field='DN',
            all_touched=False, polygons_might_overlap=True,
            ignore_nodata=False)

        expected_result = pygeoprocessing.AggregatedValues(
            total={1: 3.0, 2: 398425.0, 3: 5.0},
            pixel_mean={1: 1.0, 2: 1.0, 3: 0.41666666666666669},
            hectare_mean={
                1: 11.111111110950143,
                2: 11.111111110937156,
                3: 3.6282805923682009},
            n_pixels={1: 3.0, 2: 398425.0, 3: 12.0},
            pixel_min={1: 1.0, 2: 1.0, 3: 1.0},
            pixel_max={1: 1.0, 2: 1.0, 3: 1.0}
        )

        for metric in [
                'total', 'pixel_mean', 'hectare_mean', 'n_pixels',
                'pixel_min', 'pixel_max']:
            _assert_deep_almost_equal(
                self, getattr(expected_result, metric),
                getattr(result, metric), places=6)

    @scm.skip_if_data_missing(TEST_DATA)
    def test_aggregate_raster_values_overlap(self):
        """PyGeoprocessing: test aggregate raster values for overlap poly."""
        import pygeoprocessing

        base_raster_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data', 'base_raster.tif')

        shapefile_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data',
            'overlap_watershed.shp')

        result = pygeoprocessing.aggregate_raster_values_uri(
            base_raster_path, shapefile_path, shapefile_field='DN',
            all_touched=False, polygons_might_overlap=True)

        expected_result = pygeoprocessing.AggregatedValues(
            total={1: 3.0, 2: 398425.0, 3: 5.0},
            pixel_mean={1: 1.0, 2: 1.0, 3: 1.0},
            hectare_mean={
                1: 11.111111110950143,
                2: 11.111111110937156,
                3: 3.6282805923682009},
            n_pixels={1: 3.0, 2: 398425.0, 3: 5.0},
            pixel_min={1: 1.0, 2: 1.0, 3: 1.0},
            pixel_max={1: 1.0, 2: 1.0, 3: 1.0}
        )

        for metric in [
                'total', 'pixel_mean', 'hectare_mean', 'n_pixels',
                'pixel_min', 'pixel_max']:
            _assert_deep_almost_equal(
                self, getattr(expected_result, metric),
                getattr(result, metric), places=6)

    @scm.skip_if_data_missing(TEST_DATA)
    def test_aggregate_raster_values_all_touched(self):
        """PyGeoprocessing: test aggregate raster values all touching poly."""
        import pygeoprocessing

        base_raster_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data', 'base_raster.tif')

        shapefile_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data',
            'overlap_watershed.shp')

        result = pygeoprocessing.aggregate_raster_values_uri(
            base_raster_path, shapefile_path, shapefile_field='DN',
            all_touched=True, polygons_might_overlap=True)

        expected_result = pygeoprocessing.AggregatedValues(
            total={1: 3.0, 2: 398425.0, 3: 8.0},
            pixel_mean={1: 1.0, 2: 1.0, 3: 1.0},
            hectare_mean={
                1: 11.111111110950143,
                2: 11.111111110937156,
                3: 5.8052489477891207},
            n_pixels={1: 3.0, 2: 398425.0, 3: 8.0},
            pixel_min={1: 1.0, 2: 1.0, 3: 1.0},
            pixel_max={1: 1.0, 2: 1.0, 3: 1.0}
        )

        for metric in [
                'total', 'pixel_mean', 'hectare_mean', 'n_pixels',
                'pixel_min', 'pixel_max']:
            _assert_deep_almost_equal(
                self, getattr(expected_result, metric),
                getattr(result, metric), places=6)

    @scm.skip_if_data_missing(TEST_DATA)
    def test_aggregate_raster_values_missing_fid(self):
        """PyGeoprocessing: test aggregate raster field id incorrect."""
        import pygeoprocessing

        base_raster_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data', 'base_raster.tif')

        shapefile_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data',
            'watershed.shp')

        with self.assertRaises(AttributeError):
            pygeoprocessing.aggregate_raster_values_uri(
                base_raster_path, shapefile_path, shapefile_field='badname',
                all_touched=True, polygons_might_overlap=True)

    @scm.skip_if_data_missing(TEST_DATA)
    def test_aggregate_raster_values_all_polys(self):
        """PyGeoprocessing: test aggregate raster values as a lump."""
        import pygeoprocessing

        base_raster_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data', 'base_raster.tif')
        shapefile_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data',
            'watershed.shp')

        result = pygeoprocessing.aggregate_raster_values_uri(
            base_raster_path, shapefile_path)

        expected_result = pygeoprocessing.AggregatedValues(
            total={9999: 398428.0},
            pixel_mean={9999: 1.0},
            hectare_mean={9999: 11.111194773692587},
            n_pixels={9999: 398428.0},
            pixel_min={9999: 1.0},
            pixel_max={9999: 1.0}
        )

        for metric in [
                'total', 'pixel_mean', 'hectare_mean', 'n_pixels',
                'pixel_min', 'pixel_max']:
            _assert_deep_almost_equal(
                self, getattr(expected_result, metric),
                getattr(result, metric), places=6)

    @scm.skip_if_data_missing(TEST_DATA)
    def test_aggregate_raster_bad_fid_type(self):
        """PyGeoprocessing: test aggregate raster bad fieldtype."""
        import pygeoprocessing

        base_raster_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data', 'base_raster.tif')

        shapefile_path = os.path.join(
            TEST_DATA, 'aggregate_raster_values_data',
            'watershed.shp')

        with self.assertRaises(TypeError):
            pygeoprocessing.aggregate_raster_values_uri(
                base_raster_path, shapefile_path,
                shapefile_field='stringfiel', all_touched=True,
                polygons_might_overlap=True)

def _assert_deep_almost_equal(test_case, expected, actual, *args, **kwargs):
    """Assert that two complex structures have almost equal contents.

    I ripped this from this stackoverflow post:
    http://stackoverflow.com/a/23550280/42897

    Compares lists, dicts and tuples recursively. Checks numeric values
    using test_case's :py:meth:`unittest.TestCase.assertAlmostEqual` and
    checks all other values with :py:meth:`unittest.TestCase.assertEqual`.
    Accepts additional positional and keyword arguments and pass those
    intact to assertAlmostEqual() (that's how you specify comparison
    precision).

    :param test_case: TestCase object on which we can call all of the basic
    'assert' methods.
    :type test_case: :py:class:`unittest.TestCase` object
    """
    is_root = '__trace' not in kwargs
    trace = kwargs.pop('__trace', 'ROOT')
    if isinstance(expected, (int, float, long, complex)):
        test_case.assertAlmostEqual(expected, actual, *args, **kwargs)
    elif isinstance(expected, (list, tuple, numpy.ndarray)):
        test_case.assertEqual(len(expected), len(actual))
        for index in xrange(len(expected)):
            v1, v2 = expected[index], actual[index]
            _assert_deep_almost_equal(
                test_case, v1, v2, __trace=repr(index), *args, **kwargs)
    elif isinstance(expected, dict):
        test_case.assertEqual(set(expected), set(actual))
        for key in expected:
            _assert_deep_almost_equal(
                test_case, expected[key], actual[key], __trace=repr(key),
                *args, **kwargs)
    else:
        test_case.assertEqual(expected, actual)
