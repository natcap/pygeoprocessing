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

    def test_watersheds_trivial(self):
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

        seed_ids = pygeoprocessing.routing.group_seeds_into_fragments_d8(
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
        self.assertEqual(seed_groupings, expected_seed_groupings)
        self.assertEqual(sorted(set(seed_ids.values())), list(range(1, 7)))
        #TODO make sure we're testing multiple seeds upstream of a seed.

    def test_fragment_aggregation_2(self):
        import pygeoprocessing.routing
        nodata = 255
        flow_dir_array= numpy.array([
            [4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4]], dtype=numpy.uint8)
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
            (1, 0): frozenset([1]),
            (2, 0): frozenset([1]),
            (2, 1): frozenset([1]),
            (3, 1): frozenset([1]),
            (3, 2): frozenset([1]),
            (4, 2): frozenset([1]),
            (4, 3): frozenset([1]),
            (5, 3): frozenset([1]),
            (5, 4): frozenset([1]),
        }
        seed_ids = pygeoprocessing.routing.group_seeds_into_fragments_d8(
            (flow_dir_path, 1), seeds_to_ws_ids)

        # Expected groupings of seeds per fragment (the fragment ID used doesn't
        # matter, just that these seeds have the correct groupings)
        self.assertEqual(set(seed_ids.keys()), set(seeds_to_ws_ids.keys()))
        self.assertEqual(set(seed_ids.values()), set([1]))
        #TODO make sure we're testing multiple seeds upstream of a seed.

    def test_fragment_aggregation_3(self):
        import pygeoprocessing.routing
        nodata = 255
        flow_dir_array= numpy.array([
            [3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 2, 2]], dtype=numpy.uint8)
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

        seed_ids = pygeoprocessing.routing.group_seeds_into_fragments_d8(
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
        #TODO make sure we're testing multiple seeds upstream of a seed.
        # TODO HOW IS THIS PASSING?


    def test_split_vector_into_seeds(self):
        import pygeoprocessing.routing

        # Use a more interesting SRS than the one I've been using for some time (at (0, 0))
        # make a flow direction raster, anything should do fine, but include some nodata.
        # make some interesting outflow geometries as a geojson vector (where all geometries are in one layer)
        # run
        # assert seeds have expected membership.

        nodata = 255
        flow_dir_array= numpy.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [nodata, nodata, nodata, nodata, nodata, nodata, nodata],
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

        outflow_geometries = []

        # Make several points that all overlap the same pixel
        outflow_geometries.append(shapely.geometry.Point(2.5, -2.5))
        outflow_geometries.append(shapely.geometry.Point(3.0, -3.0))
        outflow_geometries.append(shapely.geometry.Point(3.5, -3.5))

        # Make a few polygons that overlap
        outflow_geometries.append(shapely.geometry.box(
            2.1, -5.9, 5.9, -2.1))
        outflow_geometries.append(shapely.geometry.box(
            4.1, -7.9, 7.9, -4.1))
        outflow_geometries.append(shapely.geometry.box(
            6.1, -9.9, 9.9, -6.1))

        # Make a few lines that don't intersect but that overlap the same pixels
        outflow_geometries.append(shapely.geometry.LineString(
            [(8.1, -2), (8.1, -9.9)]))
        outflow_geometries.append(shapely.geometry.LineString(
            [(9, -2), (9, -9.9)]))
        outflow_geometries.append(shapely.geometry.LineString(
            [(9.9, -2), (9.9, -9.9)]))

        geojson_driver = gdal.GetDriverByName('GeoJSON')
        target_outflow_vector_path = os.path.join(self.workspace_dir, 'outflow.geojson')
        outflow_vector = geojson_driver.Create(
            target_outflow_vector_path, 0, 0, 0, gdal.GDT_Unknown)
        outflow_layer = outflow_vector.CreateLayer(
            'geometries', srs, ogr.wkbUnknown)

        for shapely_geometry in outflow_geometries:
            ogr_geom = ogr.CreateGeometryFromWkb(shapely_geometry.wkb)
            feature = ogr.Feature(outflow_layer.GetLayerDefn())
            feature.SetGeometry(ogr_geom)
            outflow_layer.CreateFeature(feature)

        outflow_layer = None
        outflow_vector = None

        seed_watersheds = pygeoprocessing.routing.split_vector_into_seeds(
            target_outflow_vector_path, (flow_dir_path, 1),
            working_dir=self.workspace_dir, remove=False)

        # expected seed_watersheds
        # Feature IDs are used for watershed IDs, so if we assume that the
        # features are created with sequential FIDs in the order in which they
        # are created, we can assert the whole data structure.
        # If this turns out to not be the case, I'll need to find a new way to
        # assert these outputs.
        expected_seed_watersheds = {
            (0, 0): {0, 1, 2, 3},
            (1, 0): {3},
            (3, 0): {6, 7, 8},
            (0, 1): {3},
            (1, 1): {3, 4},
            (2, 1): {4},
            (3, 1): {6, 7, 8},
            (2, 3): {5},
            (3, 3): {5, 6, 7, 8}
        }
        self.assertEqual(seed_watersheds, expected_seed_watersheds)





