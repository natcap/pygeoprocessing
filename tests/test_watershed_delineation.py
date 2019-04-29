import os
import shutil
import tempfile
import unittest

import numpy
import shapely.geometry
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

class WatershedDelineationTests(unittest.TestCase):
    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    def test_watersheds(self):
        import pygeoprocessing.testing
        import pygeoprocessing.routing

        flow_dir_array = numpy.array([
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 255],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 255],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
            dtype=numpy.int8)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

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
        flow_dir_band.SetNoDataValue(255)
        flow_dir_geotransform = [2, 2, 0, -2, 0, -2]
        flow_dir_raster.SetGeoTransform(flow_dir_geotransform)
        flow_dir_raster = None

        horizontal_line = shapely.geometry.LineString([(19, -11), (21, -11)])
        vertical_line = shapely.geometry.LineString([(21, -9), (21, -13)])
        square = shapely.geometry.box(17, -13, 21, -9)
        point = shapely.geometry.Point(21, -11)

        outflow_vector_path = os.path.join(self.workspace_dir, 'outflow.gpkg')
        pygeoprocessing.testing.create_vector_on_disk(
            [horizontal_line, vertical_line, square, point],
            srs_wkt, vector_format='GPKG', filename=outflow_vector_path)

        target_watersheds_path = os.path.join(self.workspace_dir, 'watersheds.gpkg')

        pygeoprocessing.routing.delineate_watersheds_trivial_d8(
            (flow_dir_path, 1), outflow_vector_path, target_watersheds_path)

        watersheds_vector = gdal.OpenEx(target_watersheds_path, gdal.OF_VECTOR)
        watersheds_layer = watersheds_vector.GetLayer('watersheds')
        self.assertEqual(watersheds_layer.GetFeatureCount(), 4)
