"""PyGeoprocessing 1.0 test suite."""
import tempfile
import unittest
import shutil

from osgeo import gdal
import numpy


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
        pygeoprocessing.routing.fill_pits

        driver = gdal.GetDriverByName('GTiff')
        base_path = 'base.tif'#os.path.join(self.workspace_dir, 'base.tif')
        dem_array = numpy.zeros((11, 11))
        dem_array[3:8, 3:8] = -1.0
        dem_array[0, 0] = -1.0
        raster = driver.Create(
            base_path, dem_array.shape[1], dem_array.shape[0], 1,
            gdal.GDT_Float32)
        band = raster.GetRasterBand(1)
        band.WriteArray(dem_array)
        band.FlushCache()
        fill_path = 'filled.tif'
        flow_dir_path = 'flow_dir.tif'

        pygeoprocessing.routing.fill_pits(
            (base_path, 1), fill_path, flow_dir_path,
            temp_dir_path=self.workspace_dir)
