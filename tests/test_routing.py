"""pygeoprocessing.routing test suite."""
import os
import shutil
import tempfile
import unittest

import numpy
import numpy.testing
import pygeoprocessing
import pygeoprocessing.routing
import scipy.interpolate
from osgeo import gdal
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

    def test_pit_filling_large_border(self):
        """PGP.routing: test pitfilling with large nodata border."""
        os.makedirs(self.workspace_dir, exist_ok=True)
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        nodata = -1.0
        n = 30
        dem_array = numpy.full((n, n), nodata, dtype=numpy.float32)
        dem_array[n//10:n-n//10,n//10:n-n//10] = nodata
        # make a pour point
        dem_array[n//10+1, n//10+1] = 8
        # make a pit
        dem_array[n//10+2:n//10-2+10, n//10+2:n//10-2+10] = 8
        dem_array[n//10+3:n//10-3+10, n//10+3:n//10-3+10] = 7

        _array_to_raster(dem_array, nodata, base_path)
        fill_path = os.path.join(self.workspace_dir, 'filled.tif')
        pygeoprocessing.routing.fill_pits(
            (base_path, 1), fill_path, working_dir=self.workspace_dir)
        result_array = pygeoprocessing.raster_to_numpy_array(fill_path)
        expected_result = numpy.copy(dem_array)
        expected_result[n//10+2:n//10-2+10, n//10+2:n//10-2+10] = 8
        expected_path = os.path.join(self.workspace_dir, 'expected.tif')
        _array_to_raster(expected_result, nodata, expected_path)
        numpy.testing.assert_almost_equal(result_array, expected_result)

    def test_pit_filling_small_delta(self):
        """PGP.routing: test pitfilling on small delta."""
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        dem_array = numpy.empty((4, 4), dtype=numpy.float32)
        # these values came from a real world dem that failed
        lower_val = 272.53228759765625
        higher_val = 272.5325012207031
        dem_array[:] = higher_val
        dem_array[2, 2] = lower_val

        expected_result = numpy.empty((4, 4), numpy.float32)
        expected_result[:] = higher_val
        _array_to_raster(dem_array, None, base_path)
        fill_path = os.path.join(self.workspace_dir, 'filled.tif')
        pygeoprocessing.routing.fill_pits(
            (base_path, 1), fill_path, working_dir=self.workspace_dir)
        result_array = pygeoprocessing.raster_to_numpy_array(fill_path)
        self.assertTrue(
            (result_array == expected_result).all(),
            result_array == expected_result)

    def test_pit_filling_ignore_large_pit(self):
        """PGP.routing: test pitfilling but ignore large pits."""
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        n = 256
        # create a big pit
        grid_x, grid_y = numpy.mgrid[0:n, 0:n]
        values = numpy.array([10, 10, 10, 10, 0])
        points = numpy.array(
            [(0, 0),  (0, n-1),  (n-1, 0),  (n-1, n-1), (n//2, n//2)])
        pit_dem_array = scipy.interpolate.griddata(
            points, values, (grid_x, grid_y), method='linear')

        _array_to_raster(pit_dem_array, None, base_path)
        fill_path = os.path.join(self.workspace_dir, 'filled.tif')

        # First limit fill size to 100 pixels, should not fill the pit
        pygeoprocessing.routing.fill_pits(
            (base_path, 1), fill_path, working_dir=self.workspace_dir,
            max_pixel_fill_count=100)
        result_array = pygeoprocessing.raster_to_numpy_array(fill_path)
        self.assertTrue(
            (result_array == pit_dem_array).all(),
            result_array == pit_dem_array)

        # Let pit fill all the way
        pygeoprocessing.routing.fill_pits(
            (base_path, 1), fill_path, working_dir=self.workspace_dir,
            max_pixel_fill_count=1000000)
        filled_array = numpy.full((n, n), 10.0)
        result_array = pygeoprocessing.raster_to_numpy_array(fill_path)
        self.assertTrue(
            (numpy.isclose(result_array, filled_array)).all(),
            f'{result_array == filled_array}, {result_array} {filled_array}')

    def test_pit_filling_path_band_checking(self):
        """PGP.routing: test pitfilling catches path-band formatting errors."""
        with self.assertRaises(RuntimeError):
            pygeoprocessing.routing.fill_pits(
                ('invalid path', 1), 'foo')

        with self.assertRaises(RuntimeError):
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

    def test_pit_filling_nodata_nan(self):
        """PGP.routing: test pitfilling with nan nodata value."""
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        dem_array = numpy.zeros((11, 11), dtype=numpy.float32)
        nodata = numpy.nan
        dem_array[3:8, 3:8] = -1
        dem_array[0, 0] = -1
        dem_array[1, 1] = nodata
        _array_to_raster(dem_array, nodata, base_path)

        fill_path = os.path.join(self.workspace_dir, 'filled.tif')
        pygeoprocessing.routing.fill_pits(
            (base_path, 1), fill_path, working_dir=self.workspace_dir)

        result_array = pygeoprocessing.raster_to_numpy_array(fill_path)
        self.assertEqual(result_array.dtype, numpy.float32)
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

    def test_invalid_mode_detect_outlets(self):
        """PGP.routing: ensure invalid mode caught when detecting outlets."""
        flow_dir_d8 = numpy.full((512, 512), 128, dtype=numpy.uint8)
        flow_dir_d8[0:4, 0:4] = [
            [2, 2, 2, 2],
            [2, 2, 2, 0],
            [4, 128, 2, 2],
            [2, 2, 6, 2]]
        flow_dir_d8[-1, -1] = 0
        flow_dir_d8_path = os.path.join(self.workspace_dir, 'd8.tif')
        _array_to_raster(flow_dir_d8, 128, flow_dir_d8_path)
        outlet_vector_path = os.path.join(
            self.workspace_dir, 'outlets.gpkg')

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.routing.detect_outlets(
                (flow_dir_d8_path, 1), 'bad_mode', outlet_vector_path)
        expected_message = (
            'expected flow dir type of either d8 or mfd but got bad_mode')
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

    def test_detect_outlets_d8(self):
        """PGP.routing: test detect outlets for D8."""
        flow_dir_d8 = numpy.full((512, 512), 128, dtype=numpy.uint8)
        flow_dir_d8[0:4, 0:4] = [
            [2, 2, 2, 2],
            [2, 2, 2, 0],
            [4, 128, 2, 2],
            [2, 2, 6, 2]]
        flow_dir_d8[-1, -1] = 0
        flow_dir_d8_path = os.path.join(self.workspace_dir, 'd8.tif')
        _array_to_raster(flow_dir_d8, 128, flow_dir_d8_path)
        outlet_vector_path = os.path.join(
            self.workspace_dir, 'outlets.gpkg')
        pygeoprocessing.routing.detect_outlets(
            (flow_dir_d8_path, 1), 'd8', outlet_vector_path)
        outlet_vector = gdal.OpenEx(
            outlet_vector_path, gdal.OF_VECTOR)
        outlet_layer = outlet_vector.GetLayer()
        outlet_ij_set = set()
        id_list = []
        for outlet_feature in outlet_layer:
            outlet_ij_set.add(
                (outlet_feature.GetField('i'),
                 outlet_feature.GetField('j')))
            id_list.append(outlet_feature.GetField('ID'))
        # We know the expected outlets because we constructed them above
        expected_outlet_ij_set = {
            (0, 0), (1, 0), (2, 0), (3, 0),
            (3, 1),
            (0, 2),
            (1, 3), (2, 3),
            (511, 511)}
        self.assertEqual(outlet_ij_set, expected_outlet_ij_set)
        self.assertEqual(
            sorted(id_list), list(range(len(expected_outlet_ij_set))))

    def test_detect_outlets_mfd(self):
        """PGP.routing: test detect outlets for MFD."""
        d8_nodata = 128
        flow_dir_mfd = numpy.full((512, 512), d8_nodata, dtype=numpy.int32)
        flow_dir_mfd[0:4, 0:4] = [
            [2, 2, 2, 2],
            [2, 2, 2, 0],
            [4, d8_nodata, 2, 2],
            [2, 2, 6, 2]]
        flow_dir_mfd[-1, -1] = 0
        nodata_mask = flow_dir_mfd == d8_nodata
        flow_dir_mfd[~nodata_mask] = (1 << (flow_dir_mfd[~nodata_mask]*4))
        flow_dir_mfd[nodata_mask] = 0  # set to MFD nodata
        flow_dir_mfd_path = os.path.join(self.workspace_dir, 'mfd.tif')
        _array_to_raster(flow_dir_mfd, 0, flow_dir_mfd_path)
        outlet_vector_path = os.path.join(
            self.workspace_dir, 'outlets.gpkg')
        pygeoprocessing.routing.detect_outlets(
            (flow_dir_mfd_path, 1), 'mfd', outlet_vector_path)
        outlet_vector = gdal.OpenEx(
            outlet_vector_path, gdal.OF_VECTOR)
        outlet_layer = outlet_vector.GetLayer()
        outlet_ij_set = set()
        id_list = []
        for outlet_feature in outlet_layer:
            outlet_ij_set.add(
                (outlet_feature.GetField('i'),
                 outlet_feature.GetField('j')))
            id_list.append(outlet_feature.GetField('ID'))
        # We know the expected outlets because we constructed them above
        expected_outlet_ij_set = {
            (0, 0), (1, 0), (2, 0), (3, 0),
            (3, 1),
            (0, 2),
            (1, 3), (2, 3),
            (511, 511)}
        self.assertEqual(outlet_ij_set, expected_outlet_ij_set)
        self.assertEqual(
            sorted(id_list), list(range(len(expected_outlet_ij_set))))

    def test_detect_outlets_by_block(self):
        """PGP: test detect_outlets by memory block for border cases."""
        nodata = 128  # nodata value
        flow_dir_array = numpy.array([
            [0, 0, 0, 0, 7, 7, 7, 1, 6, 6],
            [2, 3, 4, 5, 6, 7, 0, 1, 1, 2],
            [2, 2, 2, 2, 0, nodata, nodata, 3, 3, nodata],
            [2, 1, 1, 1, 2, 6, 4, 1, nodata, nodata],
            [1, 1, 0, 0, 0, 0, nodata, nodata, nodata, nodata]
        ], dtype=numpy.uint8)
        expected_outlet_ij_set = {(7, 0), (5, 1), (4, 2), (5, 4)}

        d8_flow_dir_raster_path = os.path.join(self.workspace_dir, 'd8.tif')
        pygeoprocessing.numpy_array_to_raster(
            flow_dir_array, nodata, (1, 1), (0, 0), None,
            d8_flow_dir_raster_path)
        outlet_vector_path = os.path.join(self.workspace_dir, 'outlets.gpkg')

        # Mock iterblocks so that we can test with an array smaller than
        # gets pour points on block edges e.g. flow_dir_array[2, 4]
        def mock_iterblocks(*args, **kwargs):
            xoffs = [0, 4, 8]
            win_xsizes = [4, 4, 2]
            for xoff, win_xsize in zip(xoffs, win_xsizes):
                yield {
                    'xoff': xoff,
                    'yoff': 0,
                    'win_xsize': win_xsize,
                    'win_ysize': 5}

        with unittest.mock.patch(
                'pygeoprocessing.iterblocks',
                mock_iterblocks):
            pygeoprocessing.routing.detect_outlets(
                (d8_flow_dir_raster_path, 1), 'd8', outlet_vector_path)

        outlet_vector = gdal.OpenEx(outlet_vector_path, gdal.OF_VECTOR)
        outlet_layer = outlet_vector.GetLayer()
        outlet_ij_set = set()
        id_list = []
        for outlet_feature in outlet_layer:
            outlet_ij_set.add(
                (outlet_feature.GetField('i'),
                 outlet_feature.GetField('j')))
            id_list.append(outlet_feature.GetField('ID'))
        # We know the expected outlets because we constructed them above
        self.assertEqual(outlet_ij_set, expected_outlet_ij_set)

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
             1178599424, 26880]], dtype=numpy.int32)

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
              6.18346732, 6.18299413, 6.1786881, 6.15935357, -1,
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


    def test_distance_to_channel_mfd_no_stream(self):
        """PGP.routing: MFD stream distance including area that doesn't drain to stream."""
        stream_array = numpy.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1]], dtype=numpy.int8)
        flow_dir_nodata = 0
        flow_dir_array = numpy.array([
            [268435458, 1, 16777216],
            [16777216, 285212672, 16777216],
            [16777216, 16777216, 16777216]], dtype=numpy.int32)
        # this MFD array is equivalent to these flow direction weights:
        #
        # [0 0 0] [0 0 0] [0 0 0]
        # [0 a 2] [0 b 1] [0 c 0]
        # [0 0 1] [0 0 0] [0 1 0]
        #
        # [0 0 0] [0 0 0] [0 0 0]
        # [0 d 0] [0 e 0] [0 f 0]
        # [0 1 0] [0 1 1] [0 1 0]
        #
        # [0 0 0] [0 0 0] [0 0 0]
        # [0 g 0] [0 h 0] [0 i 0]
        # [0 1 0] [0 1 0] [0 1 0]
        #
        # - pixel i is the only stream
        # - g, h, and i flow off the bottom edge
        # - flow paths are a->b->c->f->i, a->e->i, a->e->h, and d->g
        # - d, g, and h are nodata because none of their flow reaches a stream
        # - part of a and e's flow doesn't reach a stream. their distances
        #   only measure the fraction that does reach the stream.
        nodata = -1
        expected_distance = numpy.array([
            [(8+2*2**.5)/3, 3,      2],
            [nodata,        2**.5,  1],
            [nodata,        nodata, 0]])

        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        stream_path = os.path.join(self.workspace_dir, 'stream.tif')
        target_path = os.path.join(self.workspace_dir, 'distance.tif')
        _array_to_raster(flow_dir_array, flow_dir_nodata, flow_dir_path)
        _array_to_raster(stream_array, None, stream_path)
        pygeoprocessing.routing.distance_to_channel_mfd(
            (flow_dir_path, 1), (stream_path, 1), target_path)

        distance = pygeoprocessing.raster_to_numpy_array(target_path)
        numpy.testing.assert_allclose(expected_distance, distance)


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

    def test_extract_strahler_streams_watersheds_d8(self):
        """PGP.routing: test Strahler stream and subwatershed creation."""
        # make a long canyon herringbone style DEM that will have a main
        # central river and single pixel tributaries every other pixel to
        # the west and east as one steps north/south through the canyon
        target_nodata = -1
        n = 53
        dem_array = numpy.zeros((n, 3))
        # make notches every other row for both columns
        dem_array[1::2, 0::2] = 1
        # near the downstream end, set values in such a way that a nodata
        # pixel would otherwise be treated as a stream seed point if nodata
        # is not properly masked.
        dem_array[1, 1] = -0.5  # a drain
        dem_array[0, :] = 1  # two high points that drain into nodata pixel
        dem_array[0, 1] = target_nodata

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        pygeoprocessing.numpy_array_to_raster(
            dem_array, target_nodata, (1, -1), (0, 0), None, dem_path)

        filled_pits_path = os.path.join(self.workspace_dir, 'filled_pits.tif')
        pygeoprocessing.routing.fill_pits(
            (dem_path, 1), filled_pits_path)

        flow_dir_d8_path = os.path.join(self.workspace_dir, 'd8.tif')
        pygeoprocessing.routing.flow_dir_d8(
            (filled_pits_path, 1), flow_dir_d8_path,
            working_dir=self.workspace_dir)

        flow_accum_d8_path = os.path.join(self.workspace_dir, 'flow_accum.tif')
        pygeoprocessing.routing.flow_accumulation_d8(
            (flow_dir_d8_path, 1), flow_accum_d8_path)

        no_autotune_stream_vector_path = os.path.join(
            self.workspace_dir, 'no_autotune_stream.gpkg')
        pygeoprocessing.routing.extract_strahler_streams_d8(
            (flow_dir_d8_path, 1),
            (flow_accum_d8_path, 1),
            (filled_pits_path, 1),
            no_autotune_stream_vector_path,
            autotune_flow_accumulation=False,
            min_flow_accum_threshold=1)

        stream_vector = gdal.OpenEx(
            no_autotune_stream_vector_path, gdal.OF_VECTOR)
        stream_layer = stream_vector.GetLayer()
        self.assertEqual(stream_layer.GetFeatureCount(), n*2+1)
        stream_layer = None
        stream_vector = None

        autotune_stream_vector_path = os.path.join(
            self.workspace_dir, 'autotune_stream.gpkg')
        pygeoprocessing.routing.extract_strahler_streams_d8(
            (flow_dir_d8_path, 1),
            (flow_accum_d8_path, 1),
            (filled_pits_path, 1),
            autotune_stream_vector_path,
            autotune_flow_accumulation=True,
            min_flow_accum_threshold=2)

        stream_vector = gdal.OpenEx(
            autotune_stream_vector_path, gdal.OF_VECTOR)
        stream_layer = stream_vector.GetLayer()
        self.assertEqual(stream_layer.GetFeatureCount(), n-3)

        # this gets just the single outlet feature
        stream_layer.SetAttributeFilter(f'"outlet"=1')
        outlet_feature = next(iter(stream_layer))

        # known to be order 2 because none of the streams can branch more
        # than once
        self.assertEqual(outlet_feature.GetField('order'), 2)
        stream_vector = None
        stream_layer = None

        watershed_confluence_vector_path = os.path.join(
            self.workspace_dir, 'watershed_confluence.gpkg')
        pygeoprocessing.routing.calculate_subwatershed_boundary(
            (flow_dir_d8_path, 1), autotune_stream_vector_path,
            watershed_confluence_vector_path, outlet_at_confluence=True)

        watershed_vector = gdal.OpenEx(
            watershed_confluence_vector_path, gdal.OF_VECTOR)
        watershed_layer = watershed_vector.GetLayer()
        # there should be exactly an integer half number of watersheds as
        # the length of the canyon; -1 for the special configuration
        # around the nodata pixel.
        self.assertEqual(watershed_layer.GetFeatureCount(), n//2 - 1)
        watershed_vector = None
        watershed_layer = None

        watershed_confluence_vector_path = os.path.join(
            self.workspace_dir, 'watershed_confluence.gpkg')
        pygeoprocessing.routing.calculate_subwatershed_boundary(
            (flow_dir_d8_path, 1), autotune_stream_vector_path,
            watershed_confluence_vector_path, outlet_at_confluence=False)

        watershed_vector = gdal.OpenEx(
            watershed_confluence_vector_path, gdal.OF_VECTOR)
        watershed_layer = watershed_vector.GetLayer()
        self.assertEqual(watershed_layer.GetFeatureCount(), n-4)
        watershed_vector = None
        watershed_layer = None

    def test_single_drain_point(self):
        """PGP.routing: test single_drain_point pitfill."""
        dem_array = numpy.zeros((11, 11), dtype=numpy.float32)
        dem_array[0, 0] = -1.0
        dem_array[1:8, 1:8] = -2.0
        dem_array[10, 7] = -4.0
        dem_array[8:11, 8:11] = 2.0
        dem_array[9:11, 9:11] = 1.0
        dem_array[10, 10] = -7.0
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        _array_to_raster(dem_array, None, dem_path)

        # outlet tuple at 0,0, just drain one edge
        expected_array_0_0 = numpy.copy(dem_array)
        expected_array_0_0[1:8, 1:8] = -1
        expected_array_0_0[10, 7] = 0
        expected_array_0_0[8:11, 8:11] = 2.0

        # output tuple at 5,5, it's a massive pit so drain the edges
        expected_array_5_5 = numpy.copy(dem_array)
        expected_array_5_5[8:11, 8:11] = 2.0
        expected_array_5_5[10, 7] = 0

        for output_tuple, expected_array, fill_dist in [
                ((0, 0), expected_array_0_0, -1),
                ((5, 5), expected_array_5_5, -1),
                ((0, 0), dem_array, 1),
                ((5, 5), dem_array, 1),
                ]:
            fill_path = os.path.join(self.workspace_dir, 'filled.tif')
            pygeoprocessing.routing.fill_pits(
                (dem_path, 1), fill_path,
                single_outlet_tuple=output_tuple,
                max_pixel_fill_count=fill_dist,
                working_dir=self.workspace_dir)
            result_array = pygeoprocessing.raster_to_numpy_array(fill_path)
            numpy.testing.assert_almost_equal(
                result_array, expected_array)

    def test_detect_lowest_drain_and_sink(self):
        """PGP.routing: test detect_lowest_sink_and_drain."""
        dem_array = numpy.zeros((11, 11), dtype=numpy.float32)
        dem_array[3:8, 3:8] = -1.0
        dem_array[0, 0] = -1.0
        dem_array[10, 10] = -1.0

        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        _array_to_raster(dem_array, None, dem_path)

        drain_pixel, drain_height, sink_pixel, sink_height = \
            pygeoprocessing.routing.detect_lowest_drain_and_sink(
                (dem_path, 1))

        expected_drain_pixel = (0, 0)
        expected_drain_height = -1
        expected_sink_pixel = (3, 3)
        expected_sink_height = -1

        self.assertEqual(drain_pixel, expected_drain_pixel)
        self.assertEqual(drain_height, expected_drain_height)
        self.assertEqual(sink_pixel, expected_sink_pixel)
        self.assertEqual(sink_height, expected_sink_height)

    def test_channel_not_exist_distance(self):
        """PGP.routing: test for nodata result if channel doesn't exist."""
        from osgeo import osr
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        projection_wkt = srs.ExportToWkt()

        flow_dir = numpy.ones((10, 10))
        streams = numpy.zeros((10, 10))
        nodata = -1
        expected_result = numpy.full((10, 10), nodata)

        flow_dir_path = os.path.join(
            self.workspace_dir, 'test_stream_distance_flow_dir.tif')
        streams_path = os.path.join(
            self.workspace_dir, 'test_stream_distance_streams.tif')
        distance_path = os.path.join(
            self.workspace_dir, 'test_stream_distance_output.tif')
        pygeoprocessing.numpy_array_to_raster(
            flow_dir, nodata, (10, -10), (1000, 1000), projection_wkt,
            flow_dir_path)
        pygeoprocessing.numpy_array_to_raster(
            streams, nodata, (10, -10), (1000, 1000), projection_wkt,
            streams_path)

        pygeoprocessing.routing.distance_to_channel_d8(
            (flow_dir_path, 1), (streams_path, 1), distance_path)
        numpy.testing.assert_almost_equal(
            pygeoprocessing.raster_to_numpy_array(distance_path),
            expected_result)

        pygeoprocessing.routing.distance_to_channel_mfd(
            (flow_dir_path, 1), (streams_path, 1), distance_path)
        numpy.testing.assert_almost_equal(
            pygeoprocessing.raster_to_numpy_array(distance_path),
            expected_result)

    def test_extract_streams_d8(self):
        """PGP.routing: test d8 stream thresholding."""
        # test without nodata
        flow_accum_no_nodata_path = os.path.join(
            self.workspace_dir, 'flow_accum_no_nodata.tif')
        flow_accum = numpy.array([
            [2, 3, 4, 4, 4, 4],
            [3, 5, 6, 2, 3, 4],
            [2, 3, 4, 1, 2, 3],
            [0, 0, 0, 4, 4, 0]], dtype=numpy.int32)
        _array_to_raster(
            flow_accum, target_nodata=None,
            target_path=flow_accum_no_nodata_path)

        expected_array = numpy.array([
            [0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0]], dtype=numpy.uint8)
        target_streams_path = os.path.join(
            self.workspace_dir, 'streams.tif')
        pygeoprocessing.routing.extract_streams_d8(
            (flow_accum_no_nodata_path, 1), 3.5, target_streams_path)
        numpy.testing.assert_array_equal(
            pygeoprocessing.raster_to_numpy_array(target_streams_path),
            expected_array)

        # test with nodata
        n = 255  # nodata value
        expected_array = numpy.array([
            [0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [n, n, n, 1, 1, n]], dtype=numpy.uint8)
        _array_to_raster(
            flow_accum, target_nodata=0,
            target_path=flow_accum_no_nodata_path)
        pygeoprocessing.routing.extract_streams_d8(
            (flow_accum_no_nodata_path, 1), 3.5, target_streams_path)
        numpy.testing.assert_array_equal(
            pygeoprocessing.raster_to_numpy_array(target_streams_path),
            expected_array)

    def test_flow_accum_d8_with_decay(self):
        """PGP.routing: test d8 flow accumulation with decay."""
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

        # Test with scalar decay factor and also with a raster of scalar values
        const_decay_factor = 0.5
        decay_factor_path = os.path.join(
            self.workspace_dir, 'decay_factor.tif')
        decay_array = numpy.full(flow_dir_array.shape, const_decay_factor,
                                 dtype=numpy.float32)
        _array_to_raster(decay_array, None, decay_factor_path)

        for decay_factor in (const_decay_factor, (decay_factor_path, 1)):
            pygeoprocessing.routing.flow_accumulation_d8(
                (flow_dir_path, 1), target_flow_accum_path,
                custom_decay_factor=decay_factor)

            flow_accum_array = pygeoprocessing.raster_to_numpy_array(
                target_flow_accum_path)
            self.assertEqual(flow_accum_array.dtype, numpy.float64)

            # This array is a regression result saved by hand, but
            # because this flow accumulation doesn't have any joining flow
            # paths we can calculate weighted flow accumulation with the
            # closed form of the summation:
            #   decayed_accum = 2 - decay_factor ** (flow_accum - 1)
            expected_result = 2 - const_decay_factor ** (numpy.array(
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
                 [1, 1, 2, 3, 4, 5, 4, 3, 2, 1, 1]]) - 1)

            numpy.testing.assert_almost_equal(
                flow_accum_array, expected_result)

    def test_flow_accum_with_decay_merging_flow(self):
        """PGP.routing: test d8 flow accum with decay and merged flowpath."""
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        _array_to_raster(
            numpy.array([
                [255, 0, 0, 0, 0],
                [255, 0, 0, 0, 2]], dtype=numpy.uint8), 255, flow_dir_path)

        flow_accum_path = os.path.join(self.workspace_dir, 'flow_accum.tif')
        pygeoprocessing.routing.flow_accumulation_d8(
            (flow_dir_path, 1), flow_accum_path, custom_decay_factor=0.5)

        fnodata = -1.23789789e29  # copied from routing.pyx
        expected_array = numpy.array([
            [fnodata, 1, 1.5, 1.75, 2.8125],
            [fnodata, 1, 1.5, 1.75, 1.875]], dtype=numpy.float64)
        numpy.testing.assert_allclose(
            pygeoprocessing.raster_to_numpy_array(flow_accum_path),
            expected_array)
