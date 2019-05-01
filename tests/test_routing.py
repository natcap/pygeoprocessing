"""PyGeoprocessing 1.0 test suite."""
import tempfile
import unittest
import shutil
import os

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import numpy
import numpy.testing
import shapely.geometry
import shapely.wkb

import logging

LOGGER = logging.getLogger(__name__)


class TestRouting(unittest.TestCase):
    """Tests for pygeoprocessing.routing."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        #self.workspace_dir = tempfile.mkdtemp()
        self.workspace_dir = 'delineation_tests'
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

    def tearDown(self):
        """Clean up remaining files."""
        #shutil.rmtree(self.workspace_dir)

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

    def test_pit_filling_path_band_checking(self):
        """PGP.routing: test pitfilling catches path-band formatting errors."""
        import pygeoprocessing.routing

        with self.assertRaises(ValueError):
            pygeoprocessing.routing.fill_pits(
                ('invalid path', 1), 'foo')

        with self.assertRaises(ValueError):
            pygeoprocessing.routing.fill_pits(
                'invalid path', 'foo')

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

    def test_flow_dir_d8(self):
        """PGP.routing: test D8 flow."""
        import pygeoprocessing.routing

        driver = gdal.GetDriverByName('GTiff')
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        dem_array = numpy.zeros((11, 11))
        dem_raster = driver.Create(
            dem_path, dem_array.shape[1], dem_array.shape[0], 1,
            gdal.GDT_Float32, options=(
                'TILED=NO', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=11', 'BLOCKYSIZE=1'))

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
        # this is a regression result saved by hand
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

    def test_flow_accum_d8(self):
        """PGP.routing: test D8 flow accum."""
        import pygeoprocessing.routing

        driver = gdal.GetDriverByName('GTiff')
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        # this was generated from a pre-calculated plateau drain dem
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

        target_flow_accum_path = os.path.join(
            self.workspace_dir, 'flow_accum.tif')

        pygeoprocessing.routing.flow_accumulation_d8(
            (flow_dir_path, 1), target_flow_accum_path)

        flow_accum_raster = gdal.OpenEx(
            target_flow_accum_path, gdal.OF_RASTER)
        flow_accum_band = flow_accum_raster.GetRasterBand(1)
        flow_accum_array = flow_accum_band.ReadAsArray()
        flow_accum_band = None
        flow_accum_raster = None
        self.assertEqual(flow_accum_array.dtype, numpy.float64)

        # this is a regression result saved by hand
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

    def test_flow_accum_d8_flow_weights(self):
        """PGP.routing: test D8 flow accum with flow weights."""
        import pygeoprocessing.routing

        driver = gdal.GetDriverByName('GTiff')
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        # this was generated from a pre-calculated plateau drain dem
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

        flow_weight_raster_path = os.path.join(
            self.workspace_dir, 'flow_weights.tif')
        flow_weight_array = numpy.empty(
            flow_dir_array.shape)
        flow_weight_constant = 2.7
        flow_weight_array[:] = flow_weight_constant

        flow_weight_raster = driver.Create(
            flow_weight_raster_path, flow_weight_array.shape[1],
            flow_weight_array.shape[0], 1, gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        flow_weight_band = flow_weight_raster.GetRasterBand(1)
        flow_weight_band.WriteArray(flow_weight_array)
        flow_weight_band.FlushCache()
        flow_weight_band = None
        flow_weight_raster = None

        target_flow_accum_path = os.path.join(
            self.workspace_dir, 'flow_accum.tif')

        pygeoprocessing.routing.flow_accumulation_d8(
            (flow_dir_path, 1), target_flow_accum_path,
            weight_raster_path_band=(flow_weight_raster_path, 1))

        flow_accum_raster = gdal.OpenEx(
            target_flow_accum_path, gdal.OF_RASTER)
        flow_accum_band = flow_accum_raster.GetRasterBand(1)
        flow_accum_array = flow_accum_band.ReadAsArray()
        flow_accum_band = None
        flow_accum_raster = None
        self.assertEqual(flow_accum_array.dtype, numpy.float64)

        # this is a regression result saved by hand from a simple run but
        # multiplied by the flow weight constant so we know flow weights work.
        expected_result = flow_weight_constant * numpy.array(
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
             [1, 1, 2, 3, 4, 5, 4, 3, 2, 1, 1]], dtype=numpy.float64)

        numpy.testing.assert_almost_equal(
            flow_accum_array, expected_result, 6)

        pygeoprocessing.routing.flow_accumulation_d8(
            (flow_dir_path, 1), target_flow_accum_path,
            weight_raster_path_band=(flow_weight_raster_path, 1))

        flow_accum_raster = gdal.OpenEx(
            target_flow_accum_path, gdal.OF_RASTER)
        flow_accum_band = flow_accum_raster.GetRasterBand(1)
        flow_accum_array = flow_accum_band.ReadAsArray()
        flow_accum_band = None
        flow_accum_raster = None
        self.assertEqual(flow_accum_array.dtype, numpy.float64)

        # this is a regression result saved by hand from a simple run but
        # multiplied by the flow weight constant so we know flow weights work.
        zero_array = numpy.zeros(flow_dir_array.shape)
        zero_raster_path = os.path.join(self.workspace_dir, 'zero.tif')
        zero_raster = driver.Create(
            zero_raster_path, zero_array.shape[1],
            zero_array.shape[0], 1, gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        zero_band = zero_raster.GetRasterBand(1)
        zero_band.WriteArray(zero_array)
        # doing this on purpose to make the weights as complicated as possible
        zero_band.SetNoDataValue(0)
        zero_band.FlushCache()
        zero_band = None
        zero_raster = None

        pygeoprocessing.routing.flow_accumulation_d8(
            (flow_dir_path, 1), target_flow_accum_path,
            weight_raster_path_band=(zero_raster_path, 1))
        flow_accum_raster = gdal.OpenEx(
            target_flow_accum_path, gdal.OF_RASTER)
        flow_accum_band = flow_accum_raster.GetRasterBand(1)
        flow_accum_array = flow_accum_band.ReadAsArray()
        flow_accum_band = None
        flow_accum_raster = None
        self.assertEqual(flow_accum_array.dtype, numpy.float64)

        numpy.testing.assert_almost_equal(
            flow_accum_array, zero_array, 6)

    def test_flow_dir_mfd(self):
        """PGP.routing: test multiple flow dir."""
        import pygeoprocessing.routing

        driver = gdal.GetDriverByName('GTiff')
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        # this makes a flat raster with a left-to-right central channel
        dem_array = numpy.zeros((11, 11))
        dem_array[5, :] = -1
        dem_raster = driver.Create(
            dem_path, dem_array.shape[1], dem_array.shape[0], 1,
            gdal.GDT_Float32, options=(
                'TILED=NO', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=11', 'BLOCKYSIZE=1'))

        dem_band = dem_raster.GetRasterBand(1)
        dem_band.WriteArray(dem_array)
        dem_band.FlushCache()
        dem_band = None
        dem_raster = None

        target_flow_dir_path = os.path.join(
            self.workspace_dir, 'flow_dir.tif')

        pygeoprocessing.routing.flow_dir_mfd(
            (dem_path, 1), target_flow_dir_path,
            working_dir=self.workspace_dir)

        flow_dir_raster = gdal.OpenEx(target_flow_dir_path, gdal.OF_RASTER)
        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_array = flow_dir_band.ReadAsArray()
        flow_dir_band = None
        flow_dir_raster = None
        self.assertEqual(flow_array.dtype, numpy.int32)

        # this was generated from a hand checked result
        expected_result = numpy.array([
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [4603904, 983040, 983040, 983040, 983040, 524296, 15, 15, 15, 15,
             1073741894],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880]])

        numpy.testing.assert_almost_equal(flow_array, expected_result)

    def test_flow_accum_mfd(self):
        """PGP.routing: test flow accumulation for multiple flow."""
        import pygeoprocessing.routing
        driver = gdal.GetDriverByName('GTiff')

        n = 11
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        dem_array = numpy.zeros((n, n))
        dem_raster = driver.Create(
            dem_path, dem_array.shape[1], dem_array.shape[0], 1,
            gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))

        dem_array[int(n/2), :] = -1

        dem_band = dem_raster.GetRasterBand(1)
        dem_band.WriteArray(dem_array)
        dem_band.FlushCache()
        dem_band = None
        dem_raster = None

        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        pygeoprocessing.routing.flow_dir_mfd(
            (dem_path, 1), flow_dir_path,
            working_dir=self.workspace_dir)

        target_flow_accum_path = os.path.join(
            self.workspace_dir, 'flow_accum_mfd.tif')

        pygeoprocessing.routing.flow_accumulation_mfd(
            (flow_dir_path, 1), target_flow_accum_path)

        flow_accum_raster = gdal.OpenEx(
            target_flow_accum_path, gdal.OF_RASTER)
        flow_accum_band = flow_accum_raster.GetRasterBand(1)
        flow_array = flow_accum_band.ReadAsArray()
        flow_accum_band = None
        flow_accum_raster = None
        self.assertEqual(flow_array.dtype, numpy.float64)

        # this was generated from a hand-checked result
        expected_result = numpy.array([
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1.88571429, 2.11428571, 2., 2., 2., 2., 2., 2., 2., 2.11428571,
             1.88571429],
            [2.7355102, 3.23183673, 3.03265306, 3., 3., 3., 3., 3.,
             3.03265306, 3.23183673, 2.7355102],
            [3.56468805, 4.34574927, 4.08023324, 4.00932945, 4., 4., 4.,
             4.00932945, 4.08023324, 4.34574927, 3.56468805],
            [4.38045548, 5.45412012, 5.13583673, 5.02692212, 5.00266556, 5.,
             5.00266556, 5.02692212, 5.13583673, 5.45412012, 4.38045548],
            [60.5, 51.12681336, 39.01272503, 27.62141227, 16.519192,
             11.00304635, 16.519192, 27.62141227, 39.01272503, 51.12681336,
             60.5],
            [4.38045548, 5.45412012, 5.13583673, 5.02692212, 5.00266556, 5.,
             5.00266556, 5.02692212, 5.13583673, 5.45412012, 4.38045548],
            [3.56468805, 4.34574927, 4.08023324, 4.00932945, 4., 4., 4.,
             4.00932945, 4.08023324, 4.34574927, 3.56468805],
            [2.7355102, 3.23183673, 3.03265306, 3., 3., 3., 3., 3.,
             3.03265306, 3.23183673, 2.7355102],
            [1.88571429, 2.11428571, 2., 2., 2., 2., 2., 2., 2., 2.11428571,
             1.88571429],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

        numpy.testing.assert_almost_equal(flow_array, expected_result)

    def test_flow_accum_mfd_with_weights(self):
        """PGP.routing: test flow accum for mfd with weights."""
        import pygeoprocessing.routing
        driver = gdal.GetDriverByName('GTiff')

        n = 11
        dem_raster_path = os.path.join(self.workspace_dir, 'dem.tif')
        dem_array = numpy.zeros((n, n))
        dem_raster = driver.Create(
            dem_raster_path, dem_array.shape[1], dem_array.shape[0], 1,
            gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))

        dem_array[int(n/2), :] = -1

        dem_band = dem_raster.GetRasterBand(1)
        dem_band.WriteArray(dem_array)
        dem_band.FlushCache()
        dem_band = None
        dem_raster = None

        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        pygeoprocessing.routing.flow_dir_mfd(
            (dem_raster_path, 1), flow_dir_path,
            working_dir=self.workspace_dir)

        flow_weight_raster_path = os.path.join(
            self.workspace_dir, 'flow_weights.tif')
        flow_weight_array = numpy.empty((n, n))
        flow_weight_constant = 2.7
        flow_weight_array[:] = flow_weight_constant
        pygeoprocessing.new_raster_from_base(
            flow_dir_path, flow_weight_raster_path, gdal.GDT_Float32,
            [-1.0])
        flow_weight_raster = gdal.OpenEx(
            flow_weight_raster_path, gdal.OF_RASTER | gdal.GA_Update)
        flow_weight_band = flow_weight_raster.GetRasterBand(1)
        flow_weight_band.WriteArray(flow_weight_array)
        flow_weight_band.FlushCache()
        flow_weight_band = None
        flow_weight_raster = None

        target_flow_accum_path = os.path.join(
            self.workspace_dir, 'flow_accum_mfd.tif')

        pygeoprocessing.routing.flow_accumulation_mfd(
            (flow_dir_path, 1), target_flow_accum_path,
            weight_raster_path_band=(flow_weight_raster_path, 1))

        flow_accum_raster = gdal.OpenEx(
            target_flow_accum_path, gdal.OF_RASTER)
        flow_accum_band = flow_accum_raster.GetRasterBand(1)
        flow_array = flow_accum_band.ReadAsArray()
        flow_accum_band = None
        flow_accum_raster = None
        self.assertEqual(flow_array.dtype, numpy.float64)

        # this was generated from a hand-checked result with flow weight of
        # 1, so the result should be twice that since we have flow weights
        # of 2.
        expected_result = flow_weight_constant * numpy.array([
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1.88571429, 2.11428571, 2., 2., 2., 2., 2., 2., 2., 2.11428571,
             1.88571429],
            [2.7355102, 3.23183673, 3.03265306, 3., 3., 3., 3., 3.,
             3.03265306, 3.23183673, 2.7355102],
            [3.56468805, 4.34574927, 4.08023324, 4.00932945, 4., 4., 4.,
             4.00932945, 4.08023324, 4.34574927, 3.56468805],
            [4.38045548, 5.45412012, 5.13583673, 5.02692212, 5.00266556, 5.,
             5.00266556, 5.02692212, 5.13583673, 5.45412012, 4.38045548],
            [60.5, 51.12681336, 39.01272503, 27.62141227, 16.519192,
             11.00304635, 16.519192, 27.62141227, 39.01272503, 51.12681336,
             60.5],
            [4.38045548, 5.45412012, 5.13583673, 5.02692212, 5.00266556, 5.,
             5.00266556, 5.02692212, 5.13583673, 5.45412012, 4.38045548],
            [3.56468805, 4.34574927, 4.08023324, 4.00932945, 4., 4., 4.,
             4.00932945, 4.08023324, 4.34574927, 3.56468805],
            [2.7355102, 3.23183673, 3.03265306, 3., 3., 3., 3., 3.,
             3.03265306, 3.23183673, 2.7355102],
            [1.88571429, 2.11428571, 2., 2., 2., 2., 2., 2., 2., 2.11428571,
             1.88571429],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

        numpy.testing.assert_allclose(flow_array, expected_result, rtol=1e-6)

        # try with zero weights
        zero_array = numpy.zeros(expected_result.shape, dtype=numpy.float32)
        zero_raster_path = os.path.join(self.workspace_dir, 'zero.tif')
        zero_raster = driver.Create(
            zero_raster_path, zero_array.shape[1],
            zero_array.shape[0], 1, gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        zero_band = zero_raster.GetRasterBand(1)
        zero_band.WriteArray(zero_array)
        zero_band.SetNoDataValue(0)
        zero_band.FlushCache()
        zero_raster.FlushCache()
        zero_band = None
        zero_raster = None

        pygeoprocessing.routing.flow_accumulation_mfd(
            (flow_dir_path, 1), target_flow_accum_path,
            weight_raster_path_band=(zero_raster_path, 1))
        flow_accum_raster = gdal.OpenEx(
            target_flow_accum_path, gdal.OF_RASTER)
        flow_accum_band = flow_accum_raster.GetRasterBand(1)
        flow_accum_array = flow_accum_band.ReadAsArray()
        flow_accum_band = None
        flow_accum_raster = None
        self.assertEqual(flow_accum_array.dtype, numpy.float64)

        numpy.testing.assert_almost_equal(
            numpy.sum(flow_accum_array), numpy.sum(zero_array), 6)

    def test_extract_streams_mfd(self):
        """PGP.routing: stream extraction on multiple flow direction."""
        import pygeoprocessing.routing
        driver = gdal.GetDriverByName('GTiff')

        n = 11
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        dem_array = numpy.zeros((n, n))
        dem_raster = driver.Create(
            dem_path, dem_array.shape[1], dem_array.shape[0], 1,
            gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        dem_raster.SetGeoTransform([0, 1, 0, 0, 0, -1])
        dem_array[int(n/2), :] = -1
        dem_band = dem_raster.GetRasterBand(1)
        dem_band.WriteArray(dem_array)
        dem_band.FlushCache()
        dem_band = None
        dem_raster = None

        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        pygeoprocessing.routing.flow_dir_mfd(
            (dem_path, 1), flow_dir_path)

        target_flow_accum_path = os.path.join(
            self.workspace_dir, 'flow_accum_mfd.tif')

        pygeoprocessing.routing.flow_accumulation_mfd(
            (flow_dir_path, 1), target_flow_accum_path)
        target_stream_raster_path = os.path.join(
            self.workspace_dir, 'stream.tif')
        pygeoprocessing.routing.extract_streams_mfd(
            (target_flow_accum_path, 1), (flow_dir_path, 1), 30,
            target_stream_raster_path, trace_threshold_proportion=0.5)

        stream_raster = gdal.OpenEx(target_stream_raster_path, gdal.OF_RASTER)
        stream_band = stream_raster.GetRasterBand(1)
        stream_array = stream_band.ReadAsArray()
        stream_band = None
        stream_raster = None
        expected_stream_array = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        numpy.testing.assert_almost_equal(stream_array, expected_stream_array)

    def test_distance_to_channel_d8(self):
        """PGP.routing: test distance to channel D8."""
        import pygeoprocessing.routing
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_d8_path = os.path.join(self.workspace_dir, 'flow_dir.d8_tif')

        # this is a flow direction raster that was created from a plateau drain
        flow_dir_d8_array = numpy.array([
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
        flow_dir_d8_raster = driver.Create(
            flow_dir_d8_path, flow_dir_d8_array.shape[1],
            flow_dir_d8_array.shape[0], 1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        flow_dir_d8_band = flow_dir_d8_raster.GetRasterBand(1)
        flow_dir_d8_band.WriteArray(flow_dir_d8_array)
        flow_dir_d8_band.FlushCache()
        flow_dir_d8_band = None
        flow_dir_d8_raster = None

        # taken from a manual inspection of a flow accumulation run
        channel_path = os.path.join(self.workspace_dir, 'channel.tif')
        channel_array = numpy.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        channel_raster = driver.Create(
            channel_path, channel_array.shape[1],
            channel_array.shape[0], 1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        channel_band = channel_raster.GetRasterBand(1)
        channel_band.WriteArray(channel_array)
        channel_band.FlushCache()
        channel_band = None
        channel_raster = None

        distance_to_channel_d8_path = os.path.join(
            self.workspace_dir, 'distance_to_channel_d8.tif')
        pygeoprocessing.routing.distance_to_channel_d8(
            (flow_dir_d8_path, 1), (channel_path, 1),
            distance_to_channel_d8_path)

        distance_to_channel_d8_raster = gdal.OpenEx(
            distance_to_channel_d8_path, gdal.OF_RASTER)
        distance_to_channel_d8_band = (
            distance_to_channel_d8_raster.GetRasterBand(1))
        distance_to_channel_d8_array = (
            distance_to_channel_d8_band.ReadAsArray())
        distance_to_channel_d8_band = None
        distance_to_channel_d8_raster = None

        expected_result = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0],
             [0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0],
             [0, 0, 1, 2, 4, 4, 4, 2, 1, 0, 0],
             [0, 0, 1, 2, 3, 5, 3, 2, 1, 0, 0],
             [0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0],
             [0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0],
             [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        numpy.testing.assert_almost_equal(
            distance_to_channel_d8_array, expected_result)

    def test_distance_to_channel_d8_with_weights(self):
        """PGP.routing: test distance to channel D8."""
        import pygeoprocessing.routing
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_d8_path = os.path.join(self.workspace_dir, 'flow_dir.d8_tif')

        # this is a flow direction raster that was created from a plateau drain
        flow_dir_d8_array = numpy.array([
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
        flow_dir_d8_raster = driver.Create(
            flow_dir_d8_path, flow_dir_d8_array.shape[1],
            flow_dir_d8_array.shape[0], 1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        flow_dir_d8_band = flow_dir_d8_raster.GetRasterBand(1)
        flow_dir_d8_band.WriteArray(flow_dir_d8_array)
        flow_dir_d8_band.FlushCache()
        flow_dir_d8_band = None
        flow_dir_d8_raster = None

        # taken from a manual inspection of a flow accumulation run
        channel_path = os.path.join(self.workspace_dir, 'channel.tif')
        channel_array = numpy.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        channel_raster = driver.Create(
            channel_path, channel_array.shape[1],
            channel_array.shape[0], 1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        channel_band = channel_raster.GetRasterBand(1)
        channel_band.WriteArray(channel_array)
        channel_band.FlushCache()
        channel_band = None
        channel_raster = None

        flow_weight_array = numpy.empty(flow_dir_d8_array.shape)
        weight_factor = 2.0
        flow_weight_array[:] = weight_factor
        flow_dir_d8_weight_path = os.path.join(
            self.workspace_dir, 'flow_dir_d8.tif')
        flow_dir_d8_weight_raster = driver.Create(
            flow_dir_d8_weight_path, flow_weight_array.shape[1],
            flow_weight_array.shape[0], 1, gdal.GDT_Int32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        flow_dir_d8_weight_band = flow_dir_d8_weight_raster.GetRasterBand(1)
        flow_dir_d8_weight_band.WriteArray(flow_weight_array)
        flow_dir_d8_weight_band.FlushCache()
        flow_dir_d8_weight_band = None
        flow_dir_d8_weight_raster = None

        distance_to_channel_d8_path = os.path.join(
            self.workspace_dir, 'distance_to_channel_d8.tif')
        pygeoprocessing.routing.distance_to_channel_d8(
            (flow_dir_d8_path, 1), (channel_path, 1),
            distance_to_channel_d8_path,
            weight_raster_path_band=(flow_dir_d8_weight_path, 1))

        distance_to_channel_d8_raster = gdal.OpenEx(
            distance_to_channel_d8_path, gdal.OF_RASTER)
        distance_to_channel_d8_band = (
            distance_to_channel_d8_raster.GetRasterBand(1))
        distance_to_channel_d8_array = (
            distance_to_channel_d8_band.ReadAsArray())
        distance_to_channel_d8_band = None
        distance_to_channel_d8_raster = None

        expected_result = weight_factor * numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0],
             [0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0],
             [0, 0, 1, 2, 4, 4, 4, 2, 1, 0, 0],
             [0, 0, 1, 2, 3, 5, 3, 2, 1, 0, 0],
             [0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0],
             [0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0],
             [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        numpy.testing.assert_almost_equal(
            distance_to_channel_d8_array, expected_result)

        # try with zero weights
        zero_array = numpy.zeros(
            distance_to_channel_d8_array.shape, dtype=numpy.float32)
        zero_raster_path = os.path.join(self.workspace_dir, 'zero.tif')
        zero_raster = driver.Create(
            zero_raster_path, zero_array.shape[1],
            zero_array.shape[0], 1, gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        zero_band = zero_raster.GetRasterBand(1)
        zero_band.WriteArray(zero_array)
        zero_band.SetNoDataValue(0)
        zero_band.FlushCache()
        zero_raster.FlushCache()
        zero_band = None
        zero_raster = None
        pygeoprocessing.routing.distance_to_channel_d8(
            (flow_dir_d8_path, 1), (channel_path, 1),
            distance_to_channel_d8_path,
            weight_raster_path_band=(zero_raster_path, 1))

        distance_to_channel_d8_raster = gdal.OpenEx(
            distance_to_channel_d8_path, gdal.OF_RASTER)
        distance_to_channel_d8_band = (
            distance_to_channel_d8_raster.GetRasterBand(1))
        distance_to_channel_d8_array = (
            distance_to_channel_d8_band.ReadAsArray())
        distance_to_channel_d8_band = None
        distance_to_channel_d8_raster = None

        numpy.testing.assert_almost_equal(
            distance_to_channel_d8_array, zero_array)

    def test_distance_to_channel_mfd(self):
        """PGP.routing: test distance to channel mfd."""
        import pygeoprocessing.routing
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_mfd_path = os.path.join(
            self.workspace_dir, 'flow_dir_mfd.tif')
        flow_dir_mfd_array = numpy.array([
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [4603904, 983040, 983040, 983040, 983040, 524296, 15, 15, 15, 15,
             1073741894],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880]])
        flow_dir_mfd_raster = driver.Create(
            flow_dir_mfd_path, flow_dir_mfd_array.shape[1],
            flow_dir_mfd_array.shape[0], 1, gdal.GDT_Int32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        flow_dir_mfd_band = flow_dir_mfd_raster.GetRasterBand(1)
        flow_dir_mfd_band.WriteArray(flow_dir_mfd_array)
        flow_dir_mfd_band.FlushCache()
        flow_dir_mfd_band = None
        flow_dir_mfd_raster = None

        # taken from a manual inspection of a flow accumulation run
        channel_path = os.path.join(self.workspace_dir, 'channel.tif')
        channel_array = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        channel_raster = driver.Create(
            channel_path, channel_array.shape[1],
            channel_array.shape[0], 1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        channel_band = channel_raster.GetRasterBand(1)
        channel_band.WriteArray(channel_array)
        channel_band.FlushCache()
        channel_band = None
        channel_raster = None

        distance_to_channel_mfd_path = os.path.join(
            self.workspace_dir, 'distance_to_channel_mfd.tif')
        pygeoprocessing.routing.distance_to_channel_mfd(
            (flow_dir_mfd_path, 1), (channel_path, 1),
            distance_to_channel_mfd_path)

        distance_to_channel_mfd_raster = gdal.OpenEx(
            distance_to_channel_mfd_path, gdal.OF_RASTER)
        distance_to_channel_mfd_band = (
            distance_to_channel_mfd_raster.GetRasterBand(1))
        distance_to_channel_mfd_array = (
            distance_to_channel_mfd_band.ReadAsArray())
        distance_to_channel_mfd_band = None
        distance_to_channel_mfd_raster = None

        # this is a regression result copied by hand
        expected_result = numpy.array(
            [[5.98240137, 6.10285187, 6.15935357, 6.1786881, 6.18299413,
              6.18346732, 6.18299413, 6.1786881, 6.15935357, 6.10285187,
              5.98240137],
             [4.77092897, 4.88539641, 4.93253084, 4.94511769, 4.94677386,
              4.94677386, 4.94677386, 4.94511769, 4.93253084, 4.88539641,
              4.77092897],
             [3.56278943, 3.66892471, 3.70428382, 3.71008039, 3.71008039,
              3.71008039, 3.71008039, 3.71008039, 3.70428382, 3.66892471,
              3.56278943],
             [2.35977407, 2.45309892, 2.47338693, 2.47338693, 2.47338693,
              2.47338693, 2.47338693, 2.47338693, 2.47338693, 2.45309892,
              2.35977407],
             [1.16568542, 1.23669346, 1.23669346, 1.23669346, 1.23669346,
              1.23669346, 1.23669346, 1.23669346, 1.23669346, 1.23669346,
              1.16568542],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1.16568542, 1.23669346, 1.23669346, 1.23669346, 1.23669346,
              1.23669346, 1.23669346, 1.23669346, 1.23669346, 1.23669346,
              1.16568542],
             [2.35977407, 2.45309892, 2.47338693, 2.47338693, 2.47338693,
              2.47338693, 2.47338693, 2.47338693, 2.47338693, 2.45309892,
              2.35977407],
             [3.56278943, 3.66892471, 3.70428382, 3.71008039, 3.71008039,
              3.71008039, 3.71008039, 3.71008039, 3.70428382, 3.66892471,
              3.56278943],
             [4.77092897, 4.88539641, 4.93253084, 4.94511769, 4.94677386,
              4.94677386, 4.94677386, 4.94511769, 4.93253084, 4.88539641,
              4.77092897],
             [5.98240137, 6.10285187, 6.15935357, 6.1786881, 6.18299413,
              6.18346732, 6.18299413, 6.1786881, 6.15935357, 6.10285187,
              5.98240137]])

        numpy.testing.assert_almost_equal(
            distance_to_channel_mfd_array, expected_result)

    def test_distance_to_channel_mfd_with_weights(self):
        """PGP.routing: test distance to channel mfd with weights."""
        import pygeoprocessing.routing
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_mfd_path = os.path.join(
            self.workspace_dir, 'flow_dir_mfd.tif')
        flow_dir_mfd_array = numpy.array([
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [1761607680, 1178599424, 1178599424, 1178599424, 1178599424,
             1178599424, 1178599424, 1178599424, 1178599424, 1178599424,
             157286400],
            [4603904, 983040, 983040, 983040, 983040, 524296, 15, 15, 15, 15,
             1073741894],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880],
            [2400, 17984, 17984, 17984, 17984, 17984, 17984, 17984, 17984,
             17984, 26880]])
        flow_dir_mfd_raster = driver.Create(
            flow_dir_mfd_path, flow_dir_mfd_array.shape[1],
            flow_dir_mfd_array.shape[0], 1, gdal.GDT_Int32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        flow_dir_mfd_band = flow_dir_mfd_raster.GetRasterBand(1)
        flow_dir_mfd_band.WriteArray(flow_dir_mfd_array)
        flow_dir_mfd_band.FlushCache()
        flow_dir_mfd_band = None
        flow_dir_mfd_raster = None

        flow_weight_array = numpy.empty(flow_dir_mfd_array.shape)
        flow_weight_array[:] = 2.0
        flow_dir_mfd_weight_path = os.path.join(
            self.workspace_dir, 'flow_dir_mfd_weights.tif')
        flow_dir_mfd_weight_raster = driver.Create(
            flow_dir_mfd_weight_path, flow_weight_array.shape[1],
            flow_weight_array.shape[0], 1, gdal.GDT_Int32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        flow_dir_mfd_weight_band = flow_dir_mfd_weight_raster.GetRasterBand(1)
        flow_dir_mfd_weight_band.WriteArray(flow_weight_array)
        flow_dir_mfd_weight_band.FlushCache()
        flow_dir_mfd_weight_band = None
        flow_dir_mfd_weight_raster = None

        # taken from a manual inspection of a flow accumulation run
        channel_path = os.path.join(self.workspace_dir, 'channel.tif')
        channel_array = numpy.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        channel_raster = driver.Create(
            channel_path, channel_array.shape[1],
            channel_array.shape[0], 1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        channel_band = channel_raster.GetRasterBand(1)
        channel_band.WriteArray(channel_array)
        channel_band.FlushCache()
        channel_band = None
        channel_raster = None

        distance_to_channel_mfd_path = os.path.join(
            self.workspace_dir, 'distance_to_channel_mfd.tif')
        pygeoprocessing.routing.distance_to_channel_mfd(
            (flow_dir_mfd_path, 1), (channel_path, 1),
            distance_to_channel_mfd_path,
            weight_raster_path_band=(flow_dir_mfd_weight_path, 1))

        distance_to_channel_mfd_raster = gdal.OpenEx(
            distance_to_channel_mfd_path, gdal.OF_RASTER)
        distance_to_channel_mfd_band = (
            distance_to_channel_mfd_raster.GetRasterBand(1))
        distance_to_channel_mfd_array = (
            distance_to_channel_mfd_band.ReadAsArray())
        distance_to_channel_mfd_band = None
        distance_to_channel_mfd_raster = None

        # this is a regression result copied by hand
        expected_result = numpy.array(
            [
             [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
             [8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.],
             [6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.],
             [4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
             [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
             [4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
             [6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.],
             [8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.],
             [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
            ])

        numpy.testing.assert_almost_equal(
            distance_to_channel_mfd_array, expected_result)

        # try with zero weights
        zero_array = numpy.zeros(
            expected_result.shape, dtype=numpy.float32)
        zero_raster_path = os.path.join(self.workspace_dir, 'zero.tif')
        zero_raster = driver.Create(
            zero_raster_path, zero_array.shape[1],
            zero_array.shape[0], 1, gdal.GDT_Float32, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        zero_band = zero_raster.GetRasterBand(1)
        zero_band.WriteArray(zero_array)
        zero_band.SetNoDataValue(0)
        zero_band.FlushCache()
        zero_raster.FlushCache()
        zero_band = None
        zero_raster = None
        pygeoprocessing.routing.distance_to_channel_mfd(
            (flow_dir_mfd_path, 1), (channel_path, 1),
            distance_to_channel_mfd_path,
            weight_raster_path_band=(zero_raster_path, 1))

        distance_to_channel_d8_raster = gdal.OpenEx(
            distance_to_channel_mfd_path, gdal.OF_RASTER)
        distance_to_channel_d8_band = (
            distance_to_channel_d8_raster.GetRasterBand(1))
        distance_to_channel_d8_array = (
            distance_to_channel_d8_band.ReadAsArray())
        distance_to_channel_d8_band = None
        distance_to_channel_d8_raster = None

        numpy.testing.assert_almost_equal(
            distance_to_channel_d8_array, zero_array)

    @unittest.skip('deprecated')
    def test_watershed_delineation(self):
        """PGP.routing: test delineation of disjoint watersheds."""
        import pygeoprocessing.routing
        import pygeoprocessing.testing

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731) # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        flow_dir_array = numpy.array([
            [0, 0, 0, 2, 4, 4, 6],
            [6, 0, 0, 2, 4, 4, 6],
            [6, 6, 0, 2, 4, 6, 6],
            [4, 4, 4, 2, 0, 0, 0],
            [2, 2, 0, 6, 4, 2, 2],
            [2, 0, 0, 6, 4, 4, 2],
            [2, 0, 0, 6, 4, 4, 4]])

        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        flow_dir_raster.SetProjection(srs_wkt)
        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_dir_band.WriteArray(flow_dir_array)
        flow_dir_geotransform = [2, 2, 0, -2, 0, -2]
        flow_dir_raster.SetGeoTransform(flow_dir_geotransform)
        flow_dir_raster = None

        work_dir = os.path.join(self.workspace_dir, 'scratch')

        # TODO: What about when points are in a different projection?

        outflow_points = os.path.join(self.workspace_dir,
                                      'outflow_points.gpkg')
        points_geometry = [
            shapely.geometry.Point(3, -9),
            shapely.geometry.Point(9, -3),
            shapely.geometry.Point(15, -9),
            shapely.geometry.Point(9, -15)]
        pygeoprocessing.testing.create_vector_on_disk(
            points_geometry, srs_wkt, vector_format='GPKG',
            filename=outflow_points)

        target_watersheds_vector = os.path.join(self.workspace_dir,
                                                'sheds.gpkg')

        pygeoprocessing.routing.delineate_watersheds_d8(
            (flow_dir_path, 1), outflow_points, target_watersheds_vector,
            work_dir)

        try:
            vector = ogr.Open(target_watersheds_vector)
            self.assertEqual(vector.GetLayerCount(), 1)

            geometries = []
            layer = vector.GetLayer()
            self.assertEqual(layer.GetFeatureCount(), 4)
            for watershed_feature in layer:
                geometry = shapely.wkb.loads(
                    watershed_feature.GetGeometryRef().ExportToWkb())
                geometries.append(geometry)

                # TODO: Assert that fields have been copied over
        finally:
            vector = None

        # Check the areas of each individual polygon
        for ws_index, expected_area in enumerate([60.0, 40.0, 40.0, 56.0]):
            self.assertEqual(geometries[ws_index].area, expected_area)

        # Assert that sum of areas match the area of the raster.
        raster_area = ((flow_dir_geotransform[1]*flow_dir_array.shape[1]) *
                       (flow_dir_geotransform[5]*flow_dir_array.shape[0]))
        self.assertEqual(sum(geometry.area for geometry in geometries),
                         abs(raster_area))

    @unittest.skip('deprecated')
    def test_watershed_delineation_nested(self):
        """PGP.routing: test delineation of nested watersheds."""
        import pygeoprocessing.routing
        import pygeoprocessing.testing

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        flow_dir_array = numpy.array([
            [255, 0, 0],
            [0, 1, 2],
            [1, 2, 2]])
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        flow_dir_raster.SetProjection(srs_wkt)
        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_dir_band.SetNoDataValue(255)
        flow_dir_band.WriteArray(flow_dir_array)
        flow_dir_geotransform = [2, 2, 0, -2, 0, -2]
        flow_dir_raster.SetGeoTransform(flow_dir_geotransform)
        flow_dir_raster = None

        outflow_points = os.path.join(self.workspace_dir, 'outflow.gpkg')
        points_geometry = [
            shapely.geometry.Point(7, -3),
            shapely.geometry.Point(5, -5),
            shapely.geometry.Point(3, -7),
            shapely.geometry.Point(0, 0),  # off edge of raster; skipped.
            shapely.geometry.Point(3, -3),  # over nodata, still valid.
        ]
        pygeoprocessing.testing.create_vector_on_disk(
            points_geometry, srs_wkt, vector_format='GPKG',
            filename=outflow_points)

        target_watersheds_vector = os.path.join(self.workspace_dir,
                                                'sheds.gpkg')
        pygeoprocessing.routing.delineate_watersheds_d8(
            (flow_dir_path, 1), outflow_points,
            target_watersheds_vector,
            os.path.join(self.workspace_dir, 'scratch'))

        try:
            watersheds_vector = ogr.Open(target_watersheds_vector)
            watersheds_layer = watersheds_vector.GetLayer()
            self.assertEqual(watersheds_layer.GetFeatureCount(), 4)

            expected_ws_id_to_upstream_watersheds = {
                1: '2',
                2: '3',
                3: '',  # indicates no upstream watersheds
                5: '',
            }
            ws_id_to_upstream_ws_id = dict(
                (f.GetField('ws_id'), f.GetField('upstream_fragments'))
                for f in watersheds_vector.GetLayer())
            self.assertEqual(expected_ws_id_to_upstream_watersheds,
                             ws_id_to_upstream_ws_id)

            geometries = {}
            for watershed_feature in watersheds_vector.GetLayer():
                ws_id = watershed_feature.GetField('ws_id')
                shapely_geometry = shapely.wkb.loads(
                    watershed_feature.GetGeometryRef().ExportToWkb())
                geometries[ws_id] = shapely_geometry

            expected_areas = {
                1: 16,
                2: 12,
                3: 4,
                5: 4,
            }
            for ws_index, expected_area in expected_areas.items():
                self.assertEqual(geometries[ws_index].area, expected_area)
        finally:
            watersheds_layer = None
            watersheds_vector = None

    def test_join_watershed_fragments(self):
        """PGP.routing: test joining of watershed fragments."""
        import pygeoprocessing.routing
        import pygeoprocessing.testing

        fragment_a = shapely.geometry.Polygon([
            (0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        fragment_b = shapely.geometry.Polygon([
            (1, 0), (1, 1), (2, 1), (2, 0), (1, 0)])
        fragment_c = shapely.geometry.Polygon([
            (0, 1), (0, 2), (2, 2), (2, 1), (0, 1)])
        fragment_d = shapely.geometry.Polygon([
            (0, 2), (0, 3), (2, 3), (2, 2), (0, 2)])
        fragment_e = shapely.geometry.Polygon([
            (2, 2), (2, 3), (3, 3), (3, 2), (2, 2)])
        fragment_f = shapely.geometry.Polygon([
            (3, 1), (3, 2), (4, 2), (4, 1), (3, 1)])
        fragment_g = shapely.geometry.Polygon([
            (4, 0), (4, 1), (5, 1), (5, 0), (4, 0)])

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731) # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        fragments_vector_path = os.path.join(self.workspace_dir, 'fragments.gpkg')
        pygeoprocessing.testing.create_vector_on_disk(
            [fragment_a, fragment_b, fragment_c, fragment_d, fragment_e,
             fragment_f, fragment_g],
            srs_wkt,
            fields={'ws_id': 'int',
                    'upstream_fragments': 'string',
                    'other': 'real'},
            attributes=[
                {'other': 1.2, 'ws_id': 0, 'upstream_fragments': ''},
                {'other': 2.3, 'ws_id': 1, 'upstream_fragments': ''},
                {'other': 3.4, 'ws_id': 2, 'upstream_fragments': '0,1'},
                {'other': 4.5, 'ws_id': 3, 'upstream_fragments': '2,4'},
                {'other': 4.5, 'ws_id': 3, 'upstream_fragments': '2,4'},
                {'other': 5.6, 'ws_id': 4, 'upstream_fragments': '5'},
                {'other': 6.7, 'ws_id': 5, 'upstream_fragments': ''},
            ],
            filename=fragments_vector_path)

        joined_vector_path = os.path.join(self.workspace_dir, 'joined.gpkg')
        pygeoprocessing.routing.join_watershed_fragments(
            fragments_vector_path, joined_vector_path)

        # Maps WS_ID to expected geometries
        expected_geoms = {
            0: fragment_a,
            1: fragment_b,
            2: shapely.ops.cascaded_union([fragment_a, fragment_b,
                                           fragment_c]),
            3: shapely.ops.cascaded_union([fragment_a, fragment_b,
                                           fragment_c, fragment_d,
                                           fragment_e, fragment_f,
                                           fragment_g]),
            4: shapely.ops.cascaded_union([fragment_f, fragment_g]),
            5: fragment_g,
        }

        expected_other_values = {
            0: 1.2,
            1: 2.3,
            2: 3.4,
            3: 4.5,
            4: 5.6,
            5: 6.7,
        }

        try:
            joined_vector = ogr.Open(joined_vector_path)
            joined_layer = joined_vector.GetLayer()
            for joined_feature in joined_layer:
                ws_id = joined_feature.GetField('ws_id')
                joined_geometry = shapely.wkb.loads(
                    joined_feature.GetGeometryRef().ExportToWkb())
                self.assertTrue(expected_geoms[ws_id].equals(joined_geometry))

                # Verify all fields are copied over.
                self.assertEqual(joined_feature.GetField('other'),
                                 expected_other_values[ws_id])
        finally:
            joined_vector = None
            joined_layer = None

    def test_join_watersheds_with_cycles(self):
        """PGP.routing: Test that we can join mutually-upstream watersheds."""
        import pygeoprocessing.testing
        import pygeoprocessing.routing

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        geoms = [
            shapely.geometry.box(0, 0, 1, 1),
            shapely.geometry.box(0, 1, 1, 2),
        ]
        fragments_vector_path = os.path.join(self.workspace_dir,
                                             'ws_fragments.gpkg')
        pygeoprocessing.testing.create_vector_on_disk(
            geoms, srs_wkt, vector_format='GPKG',
            filename=fragments_vector_path,
            fields={'ws_id': 'int',
                    'upstream_fragments': 'string'},
            attributes=[
                {'ws_id': 0, 'upstream_fragments': '1'},
                {'ws_id': 1, 'upstream_fragments': '0'}])

        watersheds_vector_path = os.path.join(self.workspace_dir,
                                              'watersheds.gpkg')
        pygeoprocessing.routing.join_watershed_fragments(
            fragments_vector_path, watersheds_vector_path)

        unioned_geometry = shapely.ops.cascaded_union(geoms)
        watersheds_vector = gdal.OpenEx(watersheds_vector_path, gdal.OF_VECTOR)
        watersheds_layer = watersheds_vector.GetLayer()
        try:
            for feature in watersheds_layer:
                shapely_geom = shapely.wkb.loads(feature.GetGeometryRef().ExportToWkb())
                self.assertEqual(
                    0.0, unioned_geometry.difference(shapely_geom).area)
        finally:
            watersheds_layer = None
            watersheds_vector = None

    @unittest.skip('deprecated')
    def test_watershed_delineation_lakes(self):
        """PGP.routing: Test that we can delineate nested polygons/lakes."""
        import pygeoprocessing.routing
        import pygeoprocessing.testing

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        flow_dir_array = numpy.zeros((7, 7), dtype=numpy.uint8)
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        flow_dir_raster.SetProjection(srs_wkt)
        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_dir_band.WriteArray(flow_dir_array)
        flow_dir_geotransform = [2, 2, 0, -2, 0, -2]
        flow_dir_raster.SetGeoTransform(flow_dir_geotransform)
        flow_dir_raster = None

        def square(centerpoint_tuple):
            x, y = centerpoint_tuple
            return shapely.geometry.Polygon(
                [(x-1.5, y-1.5),
                 (x-1.5, y+1.5),
                 (x+1.5, y+1.5),
                 (x+1.5, y-1.5),
                 (x-1.5, y-1.5)])

        watershed_geometries = [
            square((16, -8)),
            square((8, -10)),
            square((2, -8)),
            square((14, -12)),
        ]

        outflow_vector = os.path.join(self.workspace_dir, 'outflow.gpkg')
        pygeoprocessing.testing.create_vector_on_disk(
            watershed_geometries, srs_wkt, vector_format='GPKG',
            filename=outflow_vector)

        target_fragments_vector = os.path.join(self.workspace_dir,
                                               'fragments.gpkg')
        pygeoprocessing.routing.delineate_watersheds_d8(
            (flow_dir_path, 1), outflow_vector, target_fragments_vector,
            os.path.join(self.workspace_dir, 'scratch'))

        expected_field_values = [
            {'ws_id': 1, 'upstream_fragments': '2,3'},
            {'ws_id': 2, 'upstream_fragments': '3'},
            {'ws_id': 3, 'upstream_fragments': ''},
            {'ws_id': 4, 'upstream_fragments': '2'},
        ]

        fragments_vector = gdal.OpenEx(target_fragments_vector,
                                       gdal.OF_VECTOR)
        field_values = [feature.items() for feature in
                        fragments_vector.GetLayer()]
        for expected_fields, fields in zip(
                sorted(expected_field_values, key=lambda x: x['ws_id']),
                sorted(field_values, key=lambda x: x['ws_id'])):
            self.assertEqual(expected_fields, fields)

    @unittest.skip('deprecated')
    def test_watershed_delineation_overlapping_points(self):
        """PGP.routing: assert geometries when outflow points overlap."""
        import pygeoprocessing.routing
        import pygeoprocessing.testing

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        flow_dir_array = numpy.zeros((3, 3), dtype=numpy.uint8)
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        flow_dir_raster.SetProjection(srs_wkt)
        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_dir_band.WriteArray(flow_dir_array)
        flow_dir_geotransform = [2, 2, 0, -2, 0, -2]
        flow_dir_raster.SetGeoTransform(flow_dir_geotransform)
        flow_dir_raster = None

        # These points all overlap the same 2x2 pixel.
        # All 4 should be included in the fragments vector.
        watershed_geometries = [
                shapely.geometry.Point(6.5, -4.5),
                shapely.geometry.Point(7.0, -5.0),
                shapely.geometry.Point(7.5, -5.5),
                shapely.geometry.Point(7.5, -5.5),  # duplicate.
        ]

        outflow_vector = os.path.join(self.workspace_dir, 'outflow.gpkg')
        pygeoprocessing.testing.create_vector_on_disk(
            watershed_geometries, srs_wkt, vector_format='GPKG',
            filename=outflow_vector)

        target_fragments_vector = os.path.join(self.workspace_dir,
                                               'fragments.gpkg')
        pygeoprocessing.routing.delineate_watersheds_d8(
            (flow_dir_path, 1), outflow_vector, target_fragments_vector,
            os.path.join(self.workspace_dir, 'scratch'))

        try:
            fragments_vector = gdal.OpenEx(target_fragments_vector,
                                           gdal.OF_VECTOR)
            fragments_layer = fragments_vector.GetLayer()
            self.assertEqual(fragments_layer.GetFeatureCount(), 4)

            expected_geometry = shapely.geometry.box(2, -6, 8, -4)
            for feature in fragments_layer:
                fragment_geometry = shapely.wkb.loads(
                    feature.GetGeometryRef().ExportToWkb())
                self.assertEqual(fragment_geometry.difference(expected_geometry).area, 0)
                self.assertEqual(fragment_geometry.symmetric_difference(expected_geometry).area, 0)
        finally:
            fragments_layer = None
            fragments_vector = None

    @unittest.skip('deprecated')
    def test_watershed_delineation_overlapping_polygons(self):
        """PGP.routing: assert geometries when outflow polygons overlap."""
        import pygeoprocessing.routing
        import pygeoprocessing.testing

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        flow_dir_array = numpy.zeros((3, 3), dtype=numpy.uint8)
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        flow_dir_raster.SetProjection(srs_wkt)
        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_dir_band.WriteArray(flow_dir_array)
        flow_dir_geotransform = [2, 2, 0, -2, 0, -2]
        flow_dir_raster.SetGeoTransform(flow_dir_geotransform)
        flow_dir_raster = None

        # These polygons are all identical and all overlap the same 4 pixels.
        # All 4 should be included in the fragments vector.
        watershed_geometries = [
            shapely.geometry.box(4.1, -7.9, 7.9, -4.1) for i in range(4)]

        outflow_vector = os.path.join(self.workspace_dir, 'outflow.gpkg')
        pygeoprocessing.testing.create_vector_on_disk(
            watershed_geometries, srs_wkt, vector_format='GPKG',
            filename=outflow_vector)

        target_fragments_vector = os.path.join(self.workspace_dir,
                                               'fragments.gpkg')
        pygeoprocessing.routing.delineate_watersheds_d8(
            (flow_dir_path, 1), outflow_vector, target_fragments_vector,
            os.path.join(self.workspace_dir, 'scratch'))

        try:
            fragments_vector = gdal.OpenEx(target_fragments_vector,
                                           gdal.OF_VECTOR)
            fragments_layer = fragments_vector.GetLayer()
            self.assertEqual(fragments_layer.GetFeatureCount(), 4)

            expected_geometry = shapely.geometry.box(2, -8, 8, -4)
            for feature in fragments_layer:
                fragment_geometry = shapely.wkb.loads(
                    feature.GetGeometryRef().ExportToWkb())
                self.assertEqual(fragment_geometry.difference(expected_geometry).area, 0)
                self.assertEqual(fragment_geometry.symmetric_difference(expected_geometry).area, 0)
        finally:
            fragments_layer = None
            fragments_vector = None

    @unittest.skip('deprecated')
    def test_watershed_delineation_overlapping_linestrings(self):
        """PGP.routing: assert geometries when outflow linestrings overlap."""
        import pygeoprocessing.routing
        import pygeoprocessing.testing

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        flow_dir_array = numpy.zeros((4, 4), dtype=numpy.uint8)
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        flow_dir_raster.SetProjection(srs_wkt)
        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_dir_band.WriteArray(flow_dir_array)
        flow_dir_geotransform = [2, 2, 0, -2, 0, -2]
        flow_dir_raster.SetGeoTransform(flow_dir_geotransform)
        flow_dir_raster = None

        # These two linestrings are different.
        # They do, however, overlap at multiple points and should have the same
        # watershed geometries.
        watershed_geometries = [
            shapely.geometry.LineString(
                [(5, -3), (5, -9), (9, -9), (9, -3)]),
            shapely.geometry.LineString(
                [(9, -3), (7, -7), (9, -7), (9, -9)])
        ]

        outflow_vector = os.path.join(self.workspace_dir, 'outflow.gpkg')
        pygeoprocessing.testing.create_vector_on_disk(
            watershed_geometries, srs_wkt, vector_format='GPKG',
            filename=outflow_vector)

        target_fragments_vector = os.path.join(self.workspace_dir,
                                               'fragments.gpkg')
        pygeoprocessing.routing.delineate_watersheds_d8(
            (flow_dir_path, 1), outflow_vector, target_fragments_vector,
            os.path.join(self.workspace_dir, 'scratch'))

        try:
            fragments_vector = gdal.OpenEx(target_fragments_vector,
                                           gdal.OF_VECTOR)
            fragments_layer = fragments_vector.GetLayer()
            self.assertEqual(fragments_layer.GetFeatureCount(), 2)

            expected_geometry = shapely.geometry.box(2, -10, 10, -2)
            for feature in fragments_layer:
                fragment_geometry = shapely.wkb.loads(
                    feature.GetGeometryRef().ExportToWkb())
                self.assertEqual(fragment_geometry.difference(expected_geometry).area, 0)
                self.assertEqual(fragment_geometry.symmetric_difference(expected_geometry).area, 0)
        finally:
            fragments_layer = None
            fragments_vector = None

    @unittest.skip('deprecated')
    def test_watershed_delineation_multipolygons(self):
        """PGP.routing: assert watershed when given a multipolygon."""
        import pygeoprocessing.routing
        import pygeoprocessing.testing

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        flow_dir_array = numpy.zeros((3, 3), dtype=numpy.uint8)
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        flow_dir_raster.SetProjection(srs_wkt)
        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_dir_band.WriteArray(flow_dir_array)
        flow_dir_geotransform = [2, 2, 0, -2, 0, -2]
        flow_dir_raster.SetGeoTransform(flow_dir_geotransform)
        flow_dir_raster = None

        # The two component geometries of this multipolygon are discontinuous
        watershed_geometries = [
            shapely.geometry.MultiPolygon([
                shapely.geometry.box(6.1, -3.9, 7.9, -2.1),
                shapely.geometry.box(6.1, -7.9, 7.9, -6.1),
            ])]

        outflow_vector = os.path.join(self.workspace_dir, 'outflow.gpkg')
        pygeoprocessing.testing.create_vector_on_disk(
            watershed_geometries, srs_wkt, vector_format='GPKG',
            filename=outflow_vector)

        target_fragments_vector = os.path.join(self.workspace_dir,
                                               'fragments.gpkg')
        pygeoprocessing.routing.delineate_watersheds_d8(
            (flow_dir_path, 1), outflow_vector, target_fragments_vector,
            os.path.join(self.workspace_dir, 'scratch'))

        try:
            fragments_vector = gdal.OpenEx(target_fragments_vector,
                                           gdal.OF_VECTOR)
            fragments_layer = fragments_vector.GetLayer()
            self.assertEqual(fragments_layer.GetFeatureCount(), 2)

            expected_geometries = [
                shapely.geometry.box(2, -4, 8, -2),
                shapely.geometry.box(2, -8, 8, -6),
            ]
            for feature, expected_geometry in zip(fragments_layer,
                                                  expected_geometries):
                fragment_geometry = shapely.wkb.loads(
                    feature.GetGeometryRef().ExportToWkb())
                self.assertEqual(fragment_geometry.difference(expected_geometry).area, 0)
                self.assertEqual(fragment_geometry.symmetric_difference(expected_geometry).area, 0)
        finally:
            fragments_layer = None
            fragments_vector = None


class TestWatershedDelineation(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    def test_watershed_delineation_polygons(self):
        """PGP.routing: Test that we can delineate nested polygons."""
        import pygeoprocessing.routing
        import pygeoprocessing.testing

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        pixels_array = numpy.arange(49, dtype=numpy.uint8).reshape((7, 7))
        pixels_path = os.path.join(self.workspace_dir, 'pixels.tif')
        driver = gdal.GetDriverByName('GTiff')
        pixels_raster = driver.Create(
            pixels_path, pixels_array.shape[1], pixels_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        pixels_raster.SetProjection(srs_wkt)
        pixels_band = pixels_raster.GetRasterBand(1)
        pixels_band.WriteArray(pixels_array)
        pixels_geotransform = [2, 2, 0, -2, 0, -2]
        pixels_raster.SetGeoTransform(pixels_geotransform)
        pixels_raster = None

        flow_dir_array = numpy.zeros((7, 7), dtype=numpy.uint8)
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        flow_dir_raster.SetProjection(srs_wkt)
        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_dir_band.WriteArray(flow_dir_array)
        flow_dir_geotransform = [2, 2, 0, -2, 0, -2]
        flow_dir_raster.SetGeoTransform(flow_dir_geotransform)
        flow_dir_raster = None

        def square(centerpoint_tuple):
            x, y = centerpoint_tuple
            return shapely.geometry.Polygon(
                [(x-1.5, y-1.5),
                 (x-1.5, y+1.5),
                 (x+1.5, y+1.5),
                 (x+1.5, y-1.5),
                 (x-1.5, y-1.5)])

        watershed_geometries = [
            square((14, -6)),
            square((8, -10)),
            square((2, -8)),
            square((14, -12)),
        ]

        outflow_vector = os.path.join(self.workspace_dir, 'outflow.gpkg')
        pygeoprocessing.testing.create_vector_on_disk(
            watershed_geometries, srs_wkt, vector_format='GPKG',
            filename=outflow_vector)

        target_fragments_vector = os.path.join(self.workspace_dir,
                                               'fragments.gpkg')
        pygeoprocessing.routing.delineate_watersheds_d8(
            (flow_dir_path, 1), outflow_vector, target_fragments_vector,
            os.path.join(self.workspace_dir, 'scratch'),
            starting_ws_id=5)

        # Now, join up the watershed fragments so we can verify everything's
        # been delineated as expected.
        target_watersheds_vector = os.path.join(self.workspace_dir,
                                                'watersheds.gpkg')
        pygeoprocessing.routing.join_watershed_fragments_d8(
            target_fragments_vector, target_watersheds_vector)

        # Mapping ws_id to geometry type
        expected_geometries = {
            5: shapely.geometry.box(2, -4, 16, -8),
            6: shapely.geometry.box(2, -8, 10, -12),
            7: shapely.geometry.box(2, -6, 4, -10),
            8: shapely.geometry.box(2, -10, 16, -14),
        }
        watersheds_vector = gdal.OpenEx(target_watersheds_vector, gdal.OF_VECTOR)
        watersheds_layer = watersheds_vector.GetLayer('watersheds')
        # TODO: ensure that the temporary working layer has been removed.
        self.assertEqual(watersheds_layer.GetFeatureCount(), len(watershed_geometries))

        for watershed_feature in watersheds_layer:
            ws_id = watershed_feature.GetField('__ws_id__')
            shapely_geom = shapely.wkb.loads(watershed_feature.GetGeometryRef().ExportToWkb())
            self.assertEqual(
                expected_geometries[ws_id].intersection(shapely_geom).area,
                shapely_geom.area)
            self.assertEqual(
                expected_geometries[ws_id].difference(shapely_geom).area, 0)

    def test_watershed_delineation_invalid_starting_ws_id(self):
        """PGP.routing: error when invalid starting ws_id."""
        import pygeoprocessing.routing
        import pygeoprocessing.testing

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        pixels_array = numpy.arange(49, dtype=numpy.uint8).reshape((7, 7))
        pixels_path = os.path.join(self.workspace_dir, 'pixels.tif')
        driver = gdal.GetDriverByName('GTiff')
        pixels_raster = driver.Create(
            pixels_path, pixels_array.shape[1], pixels_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        pixels_raster.SetProjection(srs_wkt)
        pixels_band = pixels_raster.GetRasterBand(1)
        pixels_band.WriteArray(pixels_array)
        pixels_geotransform = [2, 2, 0, -2, 0, -2]
        pixels_raster.SetGeoTransform(pixels_geotransform)
        pixels_raster = None

        flow_dir_array = numpy.zeros((7, 7), dtype=numpy.uint8)
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        flow_dir_raster.SetProjection(srs_wkt)
        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_dir_band.WriteArray(flow_dir_array)
        flow_dir_geotransform = [2, 2, 0, -2, 0, -2]
        flow_dir_raster.SetGeoTransform(flow_dir_geotransform)
        flow_dir_raster = None

        def square(centerpoint_tuple):
            x, y = centerpoint_tuple
            return shapely.geometry.Polygon(
                [(x-1.5, y-1.5),
                 (x-1.5, y+1.5),
                 (x+1.5, y+1.5),
                 (x+1.5, y-1.5),
                 (x-1.5, y-1.5)])

        watershed_geometries = [
            square((14, -6)),
            square((8, -10)),
            square((2, -8)),
            square((14, -12)),
        ]

        outflow_vector = os.path.join(self.workspace_dir, 'outflow.gpkg')
        pygeoprocessing.testing.create_vector_on_disk(
            watershed_geometries, srs_wkt, vector_format='GPKG',
            filename=outflow_vector)

        target_fragments_vector = os.path.join(self.workspace_dir,
                                               'fragments.gpkg')
        for invalid_value in (-100, 'foobar', 123.456):
            with self.assertRaises(ValueError) as cm:
                pygeoprocessing.routing.delineate_watersheds_d8(
                    (flow_dir_path, 1), outflow_vector, target_fragments_vector,
                    os.path.join(self.workspace_dir, 'scratch'),
                    starting_ws_id=-100)
            self.assertTrue('positive, nonzero integer'
                            in repr(cm.exception).lower())

    def test_watershed_runthrough(self):
        import pygeoprocessing.routing
        import pygeoprocessing.testing
        srs = osr.SpatialReference()

        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        flow_dir_array = numpy.zeros((1, 7), dtype=numpy.uint8)
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        driver = gdal.GetDriverByName('GTiff')
        flow_dir_raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0],
            1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        flow_dir_raster.SetProjection(srs_wkt)
        flow_dir_band = flow_dir_raster.GetRasterBand(1)
        flow_dir_band.WriteArray(flow_dir_array)
        flow_dir_geotransform = [2, 2, 0, -2, 0, -2]
        flow_dir_raster.SetGeoTransform(flow_dir_geotransform)
        flow_dir_raster = None

        watershed_geometries = [
            shapely.geometry.Point(2*n+1, -3) for n in range(1, 8)]

        outflow_vector_path = os.path.join(self.workspace_dir, 'outflow.gpkg')
        pygeoprocessing.testing.create_vector_on_disk(
            watershed_geometries, srs_wkt, vector_format='GPKG',
            filename=outflow_vector_path)

        target_fragments_vector_path = os.path.join(self.workspace_dir,
                                               'fragments.gpkg')

        pygeoprocessing.routing.delineate_watersheds_d8(
            (flow_dir_path, 1), outflow_vector_path, target_fragments_vector_path)

        target_watersheds_path = os.path.join(self.workspace_dir, 'watersheds.gpkg')
        pygeoprocessing.routing.join_watershed_fragments_d8(
            target_fragments_vector_path, target_watersheds_path)

        watersheds_vector = gdal.OpenEx(target_watersheds_path, gdal.OF_VECTOR)
        watersheds_layer = watersheds_vector.GetLayer('watersheds')
        self.assertEqual(watersheds_layer.GetFeatureCount(), 7)

        # The watersheds should be cumulative, each one bigger than the
        # previous.
        for index, feature in enumerate(watersheds_layer, start=1):
            geometry = feature.GetGeometryRef()
            self.assertEqual(geometry.Area(), index*4)
