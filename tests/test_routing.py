"""PyGeoprocessing 1.0 test suite."""
import tempfile
import unittest
import shutil
import os

from osgeo import gdal
import numpy
import numpy.testing


class TestRouting(unittest.TestCase):
    """Tests for pygeoprocessing.routing."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_pit_filling(self):
        """PGP.routing: test pitfilling."""
        import pygeoprocessing.routing
        driver = gdal.GetDriverByName('GTiff')
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        dem_array = numpy.zeros((11, 11))
        dem_array[3:8, 3:8] = -1.0
        dem_array[0, 0] = -1.0
        raster = driver.Create(
            base_path, dem_array.shape[1], dem_array.shape[0], 1,
            gdal.GDT_Float32)
        band = raster.GetRasterBand(1)
        band.WriteArray(dem_array)
        band.FlushCache()
        band = None
        raster = None
        fill_path = os.path.join(self.workspace_dir, 'filled.tif')
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')

        pygeoprocessing.routing.fill_pits(
            (base_path, 1), fill_path, working_dir=self.workspace_dir)

        result_raster = gdal.OpenEx(fill_path, gdal.OF_RASTER)
        result_band = result_raster.GetRasterBand(1)
        result_array = result_band.ReadAsArray()
        result_band = None
        result_raster = None
        self.assertEqual(result_array.dtype, numpy.float32)
        # the expected result is that the pit is filled in
        dem_array[3:8, 3:8] = 0.0
        numpy.testing.assert_almost_equal(result_array, dem_array)

    def test_pit_filling_nodata_int(self):
        """PGP.routing: test pitfilling with nodata value."""
        import pygeoprocessing.routing
        driver = gdal.GetDriverByName('GTiff')
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        dem_array = numpy.zeros((11, 11), dtype=numpy.int32)
        nodata = 9999
        dem_array[3:8, 3:8] = -1
        dem_array[0, 0] = -1
        dem_array[1, 1] = nodata
        raster = driver.Create(
            base_path, dem_array.shape[1], dem_array.shape[0], 1,
            gdal.GDT_Int32)
        band = raster.GetRasterBand(1)
        band.WriteArray(dem_array)
        band.FlushCache()
        band = None
        raster = None
        fill_path = os.path.join(self.workspace_dir, 'filled.tif')
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')

        pygeoprocessing.routing.fill_pits(
            (base_path, 1), fill_path, working_dir=self.workspace_dir)

        result_raster = gdal.OpenEx(fill_path, gdal.OF_RASTER)
        result_band = result_raster.GetRasterBand(1)
        result_array = result_band.ReadAsArray()
        result_band = None
        result_raster = None
        self.assertEqual(result_array.dtype, numpy.int32)
        # the expected result is that the pit is filled in
        dem_array[3:8, 3:8] = 0.0
        numpy.testing.assert_almost_equal(result_array, dem_array)

    def test_d8_flow_dir(self):
        """PGP.routing: test D8 flow."""
        import pygeoprocessing.routing

        driver = gdal.GetDriverByName('GTiff')
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        dem_array = numpy.zeros((11, 11))
        dem_raster = driver.Create(
            dem_path, dem_array.shape[1], dem_array.shape[0], 1,
            gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))

        dem_band = dem_raster.GetRasterBand(1)
        dem_band.WriteArray(dem_array)
        dem_band.FlushCache()
        dem_band = None
        dem_raster = None

        target_flow_dir_path = os.path.join(
            self.workspace_dir, 'flow_dir.tif')

        pygeoprocessing.routing.flow_dir_d8(
            (dem_path, 1), target_flow_dir_path,
            working_dir=self.workspace_dir)

        flow_dir_raster = gdal.OpenEx(target_flow_dir_path, gdal.OF_RASTER)
        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_array = flow_dir_band.ReadAsArray()
        flow_dir_band = None
        flow_dir_raster = None
        self.assertEqual(flow_array.dtype, numpy.uint8)
        expected_result = numpy.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
            [4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
            [4, 4, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [4, 4, 4, 2, 2, 2, 2, 2, 0, 0, 0],
            [4, 4, 4, 4, 2, 2, 2, 0, 0, 0, 0],
            [4, 4, 4, 4, 4, 2, 0, 0, 0, 0, 0],
            [4, 4, 4, 4, 4, 6, 0, 0, 0, 0, 0],
            [4, 4, 4, 4, 6, 6, 6, 0, 0, 0, 0],
            [4, 4, 4, 6, 6, 6, 6, 6, 0, 0, 0],
            [4, 4, 6, 6, 6, 6, 6, 6, 6, 0, 0],
            [4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]])
        numpy.testing.assert_almost_equal(flow_array, expected_result)

    def test_d8_flow_accum(self):
        """PGP.routing: test D8 flow accum."""
        import pygeoprocessing.routing

        driver = gdal.GetDriverByName('GTiff')
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        flow_dir_array = numpy.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
            [4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
            [4, 4, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [4, 4, 4, 2, 2, 2, 2, 2, 0, 0, 0],
            [4, 4, 4, 4, 2, 2, 2, 0, 0, 0, 0],
            [4, 4, 4, 4, 4, 2, 0, 0, 0, 0, 0],
            [4, 4, 4, 4, 4, 6, 0, 0, 0, 0, 0],
            [4, 4, 4, 4, 6, 6, 6, 0, 0, 0, 0],
            [4, 4, 4, 6, 6, 6, 6, 6, 0, 0, 0],
            [4, 4, 6, 6, 6, 6, 6, 6, 6, 0, 0],
            [4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]])
        flow_dir_raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0],
            1, gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))

        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_dir_band.WriteArray(flow_dir_array)
        flow_dir_band.FlushCache()
        flow_dir_band = None
        flow_dir_raster = None

        target_flow_accum_path = 'flow_accum.tif' #os.path.join(self.workspace_dir, 'flow_accum.tif')

        pygeoprocessing.routing.flow_accumulation_d8(
            (flow_dir_path, 1), target_flow_accum_path)

        flow_accum_raster = gdal.OpenEx(target_flow_accum_path, gdal.OF_RASTER)
        flow_accum_band = flow_accum_raster.GetRasterBand(1)
        flow_accum_array = flow_accum_band.ReadAsArray()
        flow_accum_band = None
        flow_accum_raster = None
        self.assertEqual(flow_accum_array.dtype, numpy.int32)

        expected_result = numpy.array(
            [[1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1],
             [1, 1, 2, 3, 4, 5, 4, 3, 2, 1, 1],
             [2, 1, 1, 2, 3, 4, 3, 2, 1, 1, 2],
             [3, 2, 1, 1, 2, 3, 2, 1, 1, 2, 3],
             [4, 3, 2, 1, 1, 2, 1, 1, 2, 3, 4],
             [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5],
             [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5],
             [4, 3, 2, 1, 1, 2, 1, 1, 2, 3, 4],
             [3, 2, 1, 1, 2, 3, 2, 1, 1, 2, 3],
             [2, 1, 1, 2, 3, 4, 3, 2, 1, 1, 2],
             [1, 1, 2, 3, 4, 5, 4, 3, 2, 1, 1]])

        numpy.testing.assert_almost_equal(flow_accum_array, expected_result)


    def test_multiple_flow_dir(self):
        """PGP.routing: test multiple flow dir."""
        import pygeoprocessing.routing

        driver = gdal.GetDriverByName('GTiff')
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        dem_array = numpy.zeros((11, 11))
        dem_raster = driver.Create(
            dem_path, dem_array.shape[1], dem_array.shape[0], 1,
            gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))

        dem_band = dem_raster.GetRasterBand(1)
        dem_band.WriteArray(dem_array)
        dem_band.FlushCache()
        dem_band = None
        dem_raster = None

        target_flow_dir_path = os.path.join(
            self.workspace_dir, 'flow_dir.tif')

        pygeoprocessing.routing.flow_dir_multiple_flow(
            (dem_path, 1), target_flow_dir_path,
            working_dir=self.workspace_dir)
