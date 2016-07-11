"""Module to test PyGeoprocessing module."""
import unittest
import os
import tempfile
import shutil

from osgeo import gdal
from osgeo import osr
import numpy

import pygeoprocessing.testing.assertions
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
        PyGeoprocessingTest._create_raster_on_disk(
            signal_path, signal_array, -1)
        PyGeoprocessingTest._create_raster_on_disk(
            kernel_path, kernel_array, -1)

        pygeoprocessing.convolve_2d_uri(
            signal_path, kernel_path, output_path)

        expected_output_path = os.path.join(
            TEST_DATA, 'convolution_2d_test_data', 'expected_output.tif')

        pygeoprocessing.testing.assertions.assert_rasters_equal(
            output_path, expected_output_path, 1e-6)

    @staticmethod
    def _create_raster_on_disk(file_path, data_array, nodata_value):
        """Create a raster on disk with the provided data array.

        Parameters:
            file_path (string): path to created raster on disk
            data_array (numpy.ndarray): two dimension array representing pixel
                values of the raster.
            nodata_value (float): desired nodata value of the output raster.

        Returns:
            None
        """
        # Create a raster given the shape of the pixels given the input driver
        n_rows, n_cols = data_array.shape
        driver = gdal.GetDriverByName('GTiff')
        new_raster = driver.Create(
            file_path, n_cols, n_rows, 1, gdal.GDT_Float32)

        # create some projection information based on the GDAL tutorial at
        # http://www.gdal.org/gdal_tutorial.html
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)  # UTM 10N
        new_raster.SetProjection(srs.ExportToWkt())
        pixel_size = 30
        new_raster.SetGeoTransform([0, pixel_size, 0, 0, 0, -pixel_size])
        band = new_raster.GetRasterBand(1)
        band.SetNoDataValue(nodata_value)
        band.WriteArray(data_array)
        band.FlushCache()
        band = None
        new_raster = None
