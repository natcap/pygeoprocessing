"""Module to test PyGeoprocessing module."""
import unittest
import os
import tempfile
import shutil

from osgeo import gdal
import numpy


class PyGeoprocessingTest(unittest.TestCase):
    """Class to test PyGeoprocessing's functions."""

    def setUp(self):
        """Setup workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace."""
        shutil.rmtree(self.workspace_dir)

    def test_map_dataset_to_value_nodata_undefined(self):
        """Test map_dataset_to_value if raster is missing a nodata value."""
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
        """Test map_dataset_to_value for general case."""
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
