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
            (base_path, 1), fill_path, flow_dir_path,
            temp_dir_path=self.workspace_dir)

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
            (base_path, 1), fill_path, flow_dir_path,
            temp_dir_path=self.workspace_dir)

        result_raster = gdal.OpenEx(fill_path, gdal.OF_RASTER)
        result_band = result_raster.GetRasterBand(1)
        result_array = result_band.ReadAsArray()
        result_band = None
        result_raster = None
        self.assertEqual(result_array.dtype, numpy.int32)
        # the expected result is that the pit is filled in
        dem_array[3:8, 3:8] = 0.0
        numpy.testing.assert_almost_equal(result_array, dem_array)

    def test_flow_accumulation(self):
        """PGP.routing: test flow accumulation."""
        import pygeoprocessing.routing

        driver = gdal.GetDriverByName('GTiff')
        flow_path = os.path.join(self.workspace_dir, 'flow.tif')
        # everything flows to the right
        flow_array = numpy.zeros((11, 11), dtype=numpy.int32)
        flow_array[:, -1] = 6  # last col flows down
        raster = driver.Create(
            flow_path, flow_array.shape[1], flow_array.shape[0], 1,
            gdal.GDT_Byte)
        band = raster.GetRasterBand(1)
        band.WriteArray(flow_array)
        band.FlushCache()
        band = None
        raster = None
        flow_accum_path = os.path.join(self.workspace_dir, 'flow_accum.tif')

        pygeoprocessing.routing.flow_accmulation(
            (flow_path, 1), flow_accum_path,
            weight_raster_path_band=None, temp_dir_path=None)

        result_raster = gdal.OpenEx(flow_accum_path, gdal.OF_RASTER)
        result_band = result_raster.GetRasterBand(1)
        result_array = result_band.ReadAsArray()
        result_band = None
        result_raster = None
        self.assertEqual(result_array.dtype, numpy.float64)
        # the expected result is that the pit is filled in
        self.assertEqual(result_array[-1, -1], 11*11)

    def test_flow_accumulation_with_weights(self):
        """PGP.routing: test flow accumulation."""
        import pygeoprocessing.routing

        driver = gdal.GetDriverByName('GTiff')
        flow_path = os.path.join(self.workspace_dir, 'flow.tif')
        # everything flows to the right
        flow_array = numpy.zeros((11, 11), dtype=numpy.int32)
        flow_array[:, -1] = 6  # last col flows down
        raster = driver.Create(
            flow_path, flow_array.shape[1], flow_array.shape[0], 1,
            gdal.GDT_Byte)
        band = raster.GetRasterBand(1)
        band.WriteArray(flow_array)
        band.FlushCache()

        weight_path = os.path.join(self.workspace_dir, 'weights.tif')
        raster = driver.Create(
            weight_path, flow_array.shape[1],
            flow_array.shape[0], 1, gdal.GDT_Byte)
        band = raster.GetRasterBand(1)
        band.Fill(2)  # use 2 as the weight accum, not 1
        band.FlushCache()

        band = None
        raster = None
        flow_accum_path = os.path.join(self.workspace_dir, 'flow_accum.tif')

        pygeoprocessing.routing.flow_accmulation(
            (flow_path, 1), flow_accum_path,
            weight_raster_path_band=(weight_path, 1),
            temp_dir_path=self.workspace_dir)

        result_raster = gdal.OpenEx(flow_accum_path, gdal.OF_RASTER)
        result_band = result_raster.GetRasterBand(1)
        result_array = result_band.ReadAsArray()
        result_band = None
        result_raster = None
        self.assertEqual(result_array.dtype, numpy.float64)
        # the expected result is that the pit is filled in multiplied by 2
        # because of the weight raster
        self.assertEqual(result_array[-1, -1], 2.0 * 11*11)

    def test_downstream(self):
        """PGP.routing: test distance to downstream."""
        import pygeoprocessing.routing

        driver = gdal.GetDriverByName('GTiff')
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        # everything flows to the right
        flow_dir_array = numpy.zeros((11, 11), dtype=numpy.int32)
        flow_dir_array[:, -1] = 6  # last col flows down
        raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0], 1,
            gdal.GDT_Byte)
        band = raster.GetRasterBand(1)
        band.WriteArray(flow_dir_array)
        band.FlushCache()

        flow_accum_path = os.path.join(self.workspace_dir, 'flow_accum.tif')
        # everything flows to the right
        flow_accum_array = numpy.zeros((11, 11), dtype=numpy.int32)
        flow_accum_array[:, -1] = 1  # last col is stream
        raster = driver.Create(
            flow_accum_path, flow_accum_array.shape[1],
            flow_accum_array.shape[0], 1, gdal.GDT_Byte)
        band = raster.GetRasterBand(1)
        band.WriteArray(flow_accum_array)
        band.FlushCache()

        band = None
        raster = None

        target_flow_length_raster_path = os.path.join(
            self.workspace_dir, 'flow_dist.tif')

        pygeoprocessing.routing.downstream_flow_length(
            (flow_dir_path, 1), (flow_accum_path, 1),
            1.0, target_flow_length_raster_path)

        result_raster = gdal.OpenEx(
            target_flow_length_raster_path, gdal.OF_RASTER)
        result_band = result_raster.GetRasterBand(1)
        result_array = result_band.ReadAsArray()
        result_band = None
        result_raster = None
        self.assertEqual(result_array.dtype, numpy.float64)
        # the expected result is that the pit is filled in
        flow_dist_expected_array = numpy.empty((11, 11))
        flow_dist_expected_array[:] = numpy.array(list(reversed(range(11))))
        numpy.testing.assert_almost_equal(
            result_array, flow_dist_expected_array)

    def test_downstream_with_weights(self):
        """PGP.routing: test distance to downstream with weights."""
        import pygeoprocessing.routing

        driver = gdal.GetDriverByName('GTiff')
        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        # everything flows to the right
        flow_dir_array = numpy.zeros((11, 11), dtype=numpy.int32)
        flow_dir_array[:, -1] = 6  # last col flows down
        raster = driver.Create(
            flow_dir_path, flow_dir_array.shape[1], flow_dir_array.shape[0], 1,
            gdal.GDT_Byte)
        band = raster.GetRasterBand(1)
        band.WriteArray(flow_dir_array)
        band.FlushCache()

        flow_accum_path = os.path.join(self.workspace_dir, 'flow_accum.tif')
        # everything flows to the right
        flow_accum_array = numpy.zeros((11, 11), dtype=numpy.int32)
        flow_accum_array[:, -1] = 1  # last col is stream
        raster = driver.Create(
            flow_accum_path, flow_accum_array.shape[1],
            flow_accum_array.shape[0], 1, gdal.GDT_Byte)
        band = raster.GetRasterBand(1)
        band.WriteArray(flow_accum_array)
        band.FlushCache()

        weight_path = os.path.join(self.workspace_dir, 'weights.tif')
        raster = driver.Create(
            weight_path, flow_accum_array.shape[1],
            flow_accum_array.shape[0], 1, gdal.GDT_Byte)
        band = raster.GetRasterBand(1)
        band.Fill(2)  # use 2 as the weight accum, not 1
        band.FlushCache()

        band = None
        raster = None

        target_flow_length_raster_path = os.path.join(
            self.workspace_dir, 'flow_dist.tif')

        pygeoprocessing.routing.downstream_flow_length(
            (flow_dir_path, 1), (flow_accum_path, 1),
            1.0, target_flow_length_raster_path,
            weight_raster_path_band=(weight_path, 1),
            temp_dir_path=self.workspace_dir)

        result_raster = gdal.OpenEx(
            target_flow_length_raster_path, gdal.OF_RASTER)
        result_band = result_raster.GetRasterBand(1)
        result_array = result_band.ReadAsArray()
        result_band = None
        result_raster = None
        self.assertEqual(result_array.dtype, numpy.float64)
        # the expected result is that the pit is filled in
        flow_dist_expected_array = numpy.empty((11, 11))
        # multiply by 2 because that's the weight
        flow_dist_expected_array[:] = 2 * numpy.array(
            list(reversed(range(11))))
        numpy.testing.assert_almost_equal(
            result_array, flow_dist_expected_array)
