"""pygeoprocessing.routing test suite."""
import os
import shutil
import tempfile
import unittest

from osgeo import gdal
from osgeo import osr
import numpy
import numpy.testing

import pygeoprocessing
import pygeoprocessing.routing
from test_geoprocessing import _array_to_raster


class TestRouting(unittest.TestCase):
    """Tests for pygeoprocessing.routing."""
    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_pit_filling(self):
        """PGP.routing: test pitfilling."""
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        dem_array = numpy.zeros((11, 11), dtype=numpy.float32)
        dem_array[3:8, 3:8] = -1.0
        dem_array[0, 0] = -1.0
        _array_to_raster(dem_array, None, base_path)
        fill_path = os.path.join(self.workspace_dir, 'filled.tif')
        pygeoprocessing.routing.fill_pits(
            (base_path, 1), fill_path, working_dir=self.workspace_dir)
        result_array = pygeoprocessing.raster_to_numpy_array(fill_path)
        dem_array[3:8, 3:8] = 0.0
        numpy.testing.assert_almost_equal(result_array, dem_array)

    def test_pit_filling_path_band_checking(self):
        """PGP.routing: test pitfilling catches path-band formatting errors."""
        with self.assertRaises(ValueError):
            pygeoprocessing.routing.fill_pits(
                ('invalid path', 1), 'foo')

        with self.assertRaises(ValueError):
            pygeoprocessing.routing.fill_pits(
                'invalid path', 'foo')

    def test_pit_filling_nodata_int(self):
        """PGP.routing: test pitfilling with nodata value."""
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        dem_array = numpy.zeros((11, 11), dtype=numpy.int32)
        nodata = 9999
        dem_array[3:8, 3:8] = -1
        dem_array[0, 0] = -1
        dem_array[1, 1] = nodata
        _array_to_raster(dem_array, nodata, base_path)

        fill_path = os.path.join(self.workspace_dir, 'filled.tif')
        pygeoprocessing.routing.fill_pits(
            (base_path, 1), fill_path, working_dir=self.workspace_dir)

        result_array = pygeoprocessing.raster_to_numpy_array(fill_path)
        self.assertEqual(result_array.dtype, numpy.int32)
        # the expected result is that the pit is filled in
        dem_array[3:8, 3:8] = 0.0
        numpy.testing.assert_almost_equal(result_array, dem_array)

    def test_flow_dir_d8(self):
        """PGP.routing: test D8 flow."""
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        dem_array = numpy.zeros((11, 11), dtype=numpy.float32)
        _array_to_raster(dem_array, None, dem_path)

        target_flow_dir_path = os.path.join(
            self.workspace_dir, 'flow_dir.tif')

        pygeoprocessing.routing.flow_dir_d8(
            (dem_path, 1), target_flow_dir_path,
            working_dir=self.workspace_dir)

        flow_array = pygeoprocessing.raster_to_numpy_array(
            target_flow_dir_path)
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
            [4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]], dtype=numpy.uint8)

        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        _array_to_raster(flow_dir_array, None, flow_dir_path)

        target_flow_accum_path = os.path.join(
            self.workspace_dir, 'flow_accum.tif')

        pygeoprocessing.routing.flow_accumulation_d8(
            (flow_dir_path, 1), target_flow_accum_path)

        flow_accum_array = pygeoprocessing.raster_to_numpy_array(
            target_flow_accum_path)
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
            [4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]], dtype=numpy.uint8)

        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        _array_to_raster(flow_dir_array, None, flow_dir_path)

        flow_weight_raster_path = os.path.join(
            self.workspace_dir, 'flow_weights.tif')
        flow_weight_array = numpy.empty(
            flow_dir_array.shape, dtype=numpy.float32)
        flow_weight_constant = 2.7
        flow_weight_array[:] = flow_weight_constant
        _array_to_raster(flow_weight_array, None, flow_weight_raster_path)

        target_flow_accum_path = os.path.join(
            self.workspace_dir, 'flow_accum.tif')

        pygeoprocessing.routing.flow_accumulation_d8(
            (flow_dir_path, 1), target_flow_accum_path,
            weight_raster_path_band=(flow_weight_raster_path, 1))

        flow_accum_array = pygeoprocessing.raster_to_numpy_array(
            target_flow_accum_path)
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

        flow_accum_array = pygeoprocessing.raster_to_numpy_array(
            target_flow_accum_path)
        self.assertEqual(flow_accum_array.dtype, numpy.float64)

        # this is a regression result saved by hand from a simple run but
        # multiplied by the flow weight constant so we know flow weights work.
        zero_array = numpy.zeros(flow_dir_array.shape, dtype=numpy.float32)
        zero_raster_path = os.path.join(self.workspace_dir, 'zero.tif')
        _array_to_raster(zero_array, None, zero_raster_path)

        pygeoprocessing.routing.flow_accumulation_d8(
            (flow_dir_path, 1), target_flow_accum_path,
            weight_raster_path_band=(zero_raster_path, 1))
        flow_accum_array = pygeoprocessing.raster_to_numpy_array(
            target_flow_accum_path)
        self.assertEqual(flow_accum_array.dtype, numpy.float64)

        numpy.testing.assert_almost_equal(flow_accum_array, zero_array, 6)

    def test_flow_dir_mfd(self):
        """PGP.routing: test multiple flow dir."""
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        # this makes a flat raster with a left-to-right central channel
        dem_array = numpy.zeros((11, 11))
        dem_array[5, :] = -1

        _array_to_raster(dem_array, None, dem_path)

        target_flow_dir_path = os.path.join(
            self.workspace_dir, 'flow_dir.tif')

        pygeoprocessing.routing.flow_dir_mfd(
            (dem_path, 1), target_flow_dir_path,
            working_dir=self.workspace_dir)

        flow_array = pygeoprocessing.raster_to_numpy_array(target_flow_dir_path)
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
        driver = gdal.GetDriverByName('GTiff')
        n = 11
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        dem_array = numpy.zeros((n, n), dtype=numpy.float32)
        dem_array[int(n/2), :] = -1
        _array_to_raster(dem_array, None, dem_path)

        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        pygeoprocessing.routing.flow_dir_mfd(
            (dem_path, 1), flow_dir_path,
            working_dir=self.workspace_dir)

        target_flow_accum_path = os.path.join(
            self.workspace_dir, 'flow_accum_mfd.tif')

        pygeoprocessing.routing.flow_accumulation_mfd(
            (flow_dir_path, 1), target_flow_accum_path)

        flow_array = pygeoprocessing.raster_to_numpy_array(
            target_flow_accum_path)
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
        n = 11
        dem_raster_path = os.path.join(self.workspace_dir, 'dem.tif')
        dem_array = numpy.zeros((n, n), dtype=numpy.float32)
        dem_array[int(n/2), :] = -1

        _array_to_raster(dem_array, None, dem_raster_path)

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

        flow_array = pygeoprocessing.raster_to_numpy_array(
            target_flow_accum_path)
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

        _array_to_raster(zero_array, None, zero_raster_path)

        pygeoprocessing.routing.flow_accumulation_mfd(
            (flow_dir_path, 1), target_flow_accum_path,
            weight_raster_path_band=(zero_raster_path, 1))
        flow_accum_array = pygeoprocessing.raster_to_numpy_array(
            target_flow_accum_path)
        self.assertEqual(flow_accum_array.dtype, numpy.float64)

        numpy.testing.assert_almost_equal(
            numpy.sum(flow_accum_array), numpy.sum(zero_array), 6)

    def test_extract_streams_mfd(self):
        """PGP.routing: stream extraction on multiple flow direction."""
        n = 11
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        dem_array = numpy.zeros((n, n), dtype=numpy.float32)
        dem_array[int(n/2), :] = -1

        _array_to_raster(dem_array, None, dem_path)

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

        stream_array = pygeoprocessing.raster_to_numpy_array(
            target_stream_raster_path)
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
            [4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]], dtype=numpy.uint8)

        _array_to_raster(flow_dir_d8_array, None, flow_dir_d8_path)

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
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=numpy.uint8)
        _array_to_raster(channel_array, None, channel_path)

        distance_to_channel_d8_path = os.path.join(
            self.workspace_dir, 'distance_to_channel_d8.tif')
        pygeoprocessing.routing.distance_to_channel_d8(
            (flow_dir_d8_path, 1), (channel_path, 1),
            distance_to_channel_d8_path)

        distance_to_channel_d8_array = pygeoprocessing.raster_to_numpy_array(
            distance_to_channel_d8_path)

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
            [4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]], dtype=numpy.uint8)
        _array_to_raster(flow_dir_d8_array, None, flow_dir_d8_path)

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
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=numpy.uint8)

        _array_to_raster(channel_array, None, channel_path)

        flow_weight_array = numpy.empty(
            flow_dir_d8_array.shape, dtype=numpy.int32)
        weight_factor = 2.0
        flow_weight_array[:] = weight_factor
        flow_dir_d8_weight_path = os.path.join(
            self.workspace_dir, 'flow_dir_d8.tif')

        _array_to_raster(flow_weight_array, None, flow_dir_d8_weight_path)

        distance_to_channel_d8_path = os.path.join(
            self.workspace_dir, 'distance_to_channel_d8.tif')
        pygeoprocessing.routing.distance_to_channel_d8(
            (flow_dir_d8_path, 1), (channel_path, 1),
            distance_to_channel_d8_path,
            weight_raster_path_band=(flow_dir_d8_weight_path, 1))

        distance_to_channel_d8_array = pygeoprocessing.raster_to_numpy_array(
            distance_to_channel_d8_path)

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

        _array_to_raster(zero_array, None, zero_raster_path)

        pygeoprocessing.routing.distance_to_channel_d8(
            (flow_dir_d8_path, 1), (channel_path, 1),
            distance_to_channel_d8_path,
            weight_raster_path_band=(zero_raster_path, 1))

        distance_to_channel_d8_array = pygeoprocessing.raster_to_numpy_array(
            distance_to_channel_d8_path)

        numpy.testing.assert_almost_equal(
            distance_to_channel_d8_array, zero_array)

    def test_distance_to_channel_mfd(self):
        """PGP.routing: test distance to channel mfd."""
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
             17984, 26880]], dtype=numpy.int32)

        flow_dir_mfd_path = os.path.join(
            self.workspace_dir, 'flow_dir_mfd.tif')
        _array_to_raster(flow_dir_mfd_array, None, flow_dir_mfd_path)

        # taken from a manual inspection of a flow accumulation run
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
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=numpy.uint8)
        channel_path = os.path.join(self.workspace_dir, 'channel.tif')

        _array_to_raster(channel_array, None, channel_path)

        distance_to_channel_mfd_path = os.path.join(
            self.workspace_dir, 'distance_to_channel_mfd.tif')
        pygeoprocessing.routing.distance_to_channel_mfd(
            (flow_dir_mfd_path, 1), (channel_path, 1),
            distance_to_channel_mfd_path)

        distance_to_channel_mfd_array = pygeoprocessing.raster_to_numpy_array(
            distance_to_channel_mfd_path)

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
             17984, 26880]], dtype=numpy.int32)

        flow_dir_mfd_path = os.path.join(
            self.workspace_dir, 'flow_dir_mfd.tif')
        _array_to_raster(flow_dir_mfd_array, None, flow_dir_mfd_path)

        flow_weight_array = numpy.empty(
            flow_dir_mfd_array.shape, dtype=numpy.int32)
        flow_weight_array[:] = 2.0
        flow_dir_mfd_weight_path = os.path.join(
            self.workspace_dir, 'flow_dir_mfd_weights.tif')

        _array_to_raster(flow_weight_array, None, flow_dir_mfd_weight_path)

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
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=numpy.uint8)
        _array_to_raster(channel_array, None, channel_path)

        distance_to_channel_mfd_path = os.path.join(
            self.workspace_dir, 'distance_to_channel_mfd.tif')
        pygeoprocessing.routing.distance_to_channel_mfd(
            (flow_dir_mfd_path, 1), (channel_path, 1),
            distance_to_channel_mfd_path,
            weight_raster_path_band=(flow_dir_mfd_weight_path, 1))

        distance_to_channel_mfd_array = pygeoprocessing.raster_to_numpy_array(
            distance_to_channel_mfd_path)

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
        _array_to_raster(zero_array, 0, zero_raster_path)

        pygeoprocessing.routing.distance_to_channel_mfd(
            (flow_dir_mfd_path, 1), (channel_path, 1),
            distance_to_channel_mfd_path,
            weight_raster_path_band=(zero_raster_path, 1))

        distance_to_channel_mfd_array = pygeoprocessing.raster_to_numpy_array(
            distance_to_channel_mfd_path)

        numpy.testing.assert_almost_equal(
            distance_to_channel_mfd_array, zero_array)

    def test_flow_dir_mfd_plateau(self):
        """PGP.routing: MFD on a plateau."""
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        # this makes a flat raster
        n = 100
        dem_array = numpy.zeros((n, n))
        dem_nodata = -1
        dem_array[2, :] = 1e-12
        dem_array[n//2, :] = 1e-12
        dem_array[3*n//4, :] = 1e-12

        _array_to_raster(dem_array, dem_nodata, dem_path)

        target_flow_dir_path = os.path.join(
            self.workspace_dir, 'flow_dir.tif')

        pygeoprocessing.routing.flow_dir_mfd(
            (dem_path, 1), target_flow_dir_path,
            working_dir=self.workspace_dir)

        flow_dir_nodata = pygeoprocessing.get_raster_info(
            target_flow_dir_path)['nodata'][0]

        flow_dir_array = pygeoprocessing.raster_to_numpy_array(
            target_flow_dir_path)

        self.assertTrue(not numpy.isclose(
            flow_dir_array[1:-1, 1: -1], flow_dir_nodata).any(),
            'all flow directions should be defined')

    def test_extract_strahler_streams_d8(self):
        """PGP.routing: test Strahler stream extraction."""
        local_workspace = self.workspace_dir
        # make a herringbone style DEM that will have a main central
        # river and little tributaries every other pixel to the west
        n = 50
        dem_array = numpy.zeros((n, 3))
        dem_array[0, 1] = -0.5
        # make notches every other row for both columns
        dem_array[1::2, 0::2] = 1
        dem_path = os.path.join(local_workspace, 'dem.tif')
        pygeoprocessing.numpy_array_to_raster(
            dem_array, -1, (1, -1), (0, 0), None, dem_path)

        filled_pits_path = os.path.join(local_workspace, 'filled_pits.tif')
        pygeoprocessing.routing.fill_pits(
            (dem_path, 1), filled_pits_path)

        flow_dir_d8_path = os.path.join(local_workspace, 'd8.tif')
        pygeoprocessing.routing.flow_dir_d8(
            (filled_pits_path, 1), flow_dir_d8_path,
            working_dir=local_workspace)

        flow_accum_d8_path = os.path.join(local_workspace, 'flow_accum.tif')
        pygeoprocessing.routing.flow_accumulation_d8(
            (flow_dir_d8_path, 1), flow_accum_d8_path)

        stream_vector_path = os.path.join(local_workspace, 'stream.gpkg')
        pygeoprocessing.routing.extract_strahler_streams_d8(
            (flow_dir_d8_path, 1),
            (flow_accum_d8_path, 1),
            (filled_pits_path, 1),
            stream_vector_path,
            min_flow_accum_threshold=2)

        # n-1 streams
        stream_vector = gdal.OpenEx(stream_vector_path, gdal.OF_VECTOR)
        stream_layer = stream_vector.GetLayer()
        self.assertEqual(stream_layer.GetFeatureCount(), (n//2)*2-1)
        stream_layer.SetAttributeFilter(f'"outlet"=1')
        outlet_feature = next(iter(stream_layer))
        self.assertEqual(outlet_feature.GetField('order'), 2)
