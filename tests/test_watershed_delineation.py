import collections
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

        # These geometries test:
        #  * Delineation works with varying geometry types
        #  * That we exclude seed pixels that are over nodata
        #  * That we exclude seed pixels off the bounds of the raster
        horizontal_line = shapely.geometry.LineString([(19, -11), (25, -11)])
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

        # All features should have the same watersheds, both in area and
        # geometry.
        flow_dir_bbox = pygeoprocessing.get_raster_info(flow_dir_path)['bounding_box']
        expected_watershed_geometry = shapely.geometry.box(*flow_dir_bbox)
        expected_watershed_geometry = expected_watershed_geometry.difference(
            shapely.geometry.box(20, -2, 22, -10))
        expected_watershed_geometry = expected_watershed_geometry.difference(
            shapely.geometry.box(20, -12, 22, -22))
        pygeoprocessing.testing.create_vector_on_disk(
            [expected_watershed_geometry],
            srs_wkt, vector_format='GPKG', filename=os.path.join(self.workspace_dir, 'foo.gpkg'))
        for feature in watersheds_layer:
            geometry = feature.GetGeometryRef()
            shapely_geom = shapely.wkb.loads(geometry.ExportToWkb())
            self.assertEqual(shapely_geom.area, expected_watershed_geometry.area)
            self.assertEqual(
                shapely_geom.intersection(
                    expected_watershed_geometry).area,
                expected_watershed_geometry.area)
            self.assertEqual(
                shapely_geom.difference(
                    expected_watershed_geometry).area, 0)

    def test_fragment_aggregation(self):
        import pygeoprocessing.routing
        nodata = 255
        flow_dir_array= numpy.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, nodata, nodata, nodata],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 2]], dtype=numpy.uint8)
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

        # numpy coordinates go (row, col)
        # managed_raster coordinates go (x, y), and that's what the function expects.
        seeds_to_ws_ids = {
            (5, 0): frozenset([1]),
            (6, 0): frozenset([1]),
            (5, 1): frozenset([1]),
            (6, 1): frozenset([1]),
            (0, 0): frozenset([2]),
            (0, 1): frozenset([2]),
            (0, 2): frozenset([2]),
            (2, 2): frozenset([3]),
            (3, 2): frozenset([3]),
            (2, 3): frozenset([3]),
            (3, 3): frozenset([3]),
            (5, 3): frozenset([4]),
            (6, 3): frozenset([4]),
            (5, 4): frozenset([4]),
            (6, 4): frozenset([4]),
        }

        seed_ids, nested_seeds = pygeoprocessing.routing.group_seeds_into_fragments_d8(
            (flow_dir_path, 1), seeds_to_ws_ids)

        # The order of the seed IDs could be different, so what really matters
        # is that the correct seeds are grouped together under the same ID and
        # that there are only 6 fragment IDs (1-6, inclusive).
        seed_ids_to_seeds = collections.defaultdict(set)
        for seed, seed_id in seed_ids.items():
            seed_ids_to_seeds[seed_id].add(seed)
        seed_ids_to_seeds = dict(seed_ids_to_seeds)

        seed_groupings = set([frozenset(s) for s in seed_ids_to_seeds.values()])

        # Expected groupings of seeds per fragment (the fragment ID used doesn't
        # matter, just that these seeds have the correct groupings)
        expected_seed_groupings = set([
            frozenset([(5, 0), (6, 0), (5, 1), (6, 1)]),  # using set here to make it easier to test membership.
            frozenset([(0, 0), (0, 1)]),
            frozenset([(0, 2)]),
            frozenset([(2, 2), (3, 2)]),
            frozenset([(2, 3), (3, 3)]),
            frozenset([(5, 3), (6, 3), (5, 4), (6, 4)]),
        ])
        self.assertEqual(sorted(set(seed_ids.values())), list(range(1, 7)))
