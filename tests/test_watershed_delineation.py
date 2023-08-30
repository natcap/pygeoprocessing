"""pygeoprocessing.watersheds testing suite."""
import glob
import os
import shutil
import tempfile
import unittest

import numpy
import pygeoprocessing
import pygeoprocessing.routing
import shapely.geometry
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from pygeoprocessing.routing import watershed


class WatershedDelineationTests(unittest.TestCase):
    """Main Watershed test module."""

    def setUp(self):
        """Create empty workspace dir."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Delete workspace dir."""
        shutil.rmtree(self.workspace_dir)

    def test_watersheds_diagnostic_vector(self):
        """PGP watersheds: test diagnostic vector."""
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
            dtype=numpy.uint8)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
        pygeoprocessing.numpy_array_to_raster(
            base_array=flow_dir_array,
            target_nodata=255,
            pixel_size=(2, -2),
            origin=(2, -2),
            projection_wkt=srs_wkt,
            target_path=flow_dir_path)

        # These geometries test:
        #  * Delineation works with varying geometry types
        #  * That we exclude seed pixels that are over nodata
        #  * That we exclude seed pixels off the bounds of the raster
        horizontal_line = shapely.geometry.LineString([(19, -11), (25, -11)])
        vertical_line = shapely.geometry.LineString([(21, -9), (21, -13)])
        square = shapely.geometry.box(17, -13, 21, -9)
        point = shapely.geometry.Point(21, -11)

        outflow_vector_path = os.path.join(self.workspace_dir, 'outflow.gpkg')
        pygeoprocessing.shapely_geometry_to_vector(
            [horizontal_line, vertical_line, square, point],
            outflow_vector_path, srs_wkt,
            'GPKG',
            {
                'polygon_id': ogr.OFTInteger,
                'field_string': ogr.OFTString,
                'other': ogr.OFTReal
            },
            [
                {'polygon_id': 1, 'field_string': 'hello world',
                 'other': 1.111},
                {'polygon_id': 2, 'field_string': 'hello foo', 'other': 2.222},
                {'polygon_id': 3, 'field_string': 'hello bar', 'other': 3.333},
                {'polygon_id': 4, 'field_string': 'hello baz', 'other': 4.444}
            ],
            ogr_geom_type=ogr.wkbUnknown)

        target_watersheds_path = os.path.join(
            self.workspace_dir, 'watersheds.gpkg')

        pygeoprocessing.routing.delineate_watersheds_d8(
            (flow_dir_path, 1), outflow_vector_path, target_watersheds_path,
            write_diagnostic_vector=True, working_dir=self.workspace_dir,
            remove_temp_files=False)

        # I'm deliberately only testing that the diagnostic files exist, not
        # the contents.  The diagnostic files should be for debugging only,
        # so I just want to make sure that they're created.
        num_diagnostic_files = len(
            glob.glob(os.path.join(self.workspace_dir, '**/*_seeds.gpkg')))
        self.assertEqual(num_diagnostic_files, 3)  # 3 features valid

    def test_watersheds_trivial(self):
        """PGP watersheds: test trivial delineation."""
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
            dtype=numpy.uint8)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        for srs in (srs_wkt, None):
            flow_dir_path = os.path.join(self.workspace_dir, 'flow_dir.tif')
            pygeoprocessing.numpy_array_to_raster(
                base_array=flow_dir_array,
                target_nodata=255,
                pixel_size=(2, -2),
                origin=(2, -2),
                projection_wkt=srs,
                target_path=flow_dir_path)

            # These geometries test:
            #  * Delineation works with varying geometry types
            #  * That we exclude seed pixels that are over nodata
            #  * That we exclude seed pixels off the bounds of the raster
            horizontal_line = shapely.geometry.LineString(
                [(19, -11), (25, -11)])
            vertical_line = shapely.geometry.LineString([(21, -9), (21, -13)])
            square = shapely.geometry.box(17, -13, 21, -9)
            point = shapely.geometry.Point(21, -11)

            outflow_vector_path = os.path.join(
                self.workspace_dir, 'outflow.gpkg')
            pygeoprocessing.shapely_geometry_to_vector(
                [horizontal_line, vertical_line, square, point],
                outflow_vector_path, srs_wkt,
                'GPKG',
                {
                    'polygon_id': ogr.OFTInteger,
                    'field_string': ogr.OFTString,
                    'other': ogr.OFTReal,
                    # We use ws_id internally, so make sure that this field is
                    # copied over into the final vector since it's present in
                    # the source vector.
                    'ws_id': ogr.OFTInteger,
                },
                [
                    {'polygon_id': 1, 'field_string': 'hello world',
                     'other': 1.111, 'ws_id': 1},
                    {'polygon_id': 2, 'field_string': 'hello foo',
                     'other': 2.222, 'ws_id': 2},
                    {'polygon_id': 3, 'field_string': 'hello bar',
                     'other': 3.333, 'ws_id': 3},
                    {'polygon_id': 4, 'field_string': 'hello baz',
                     'other': 4.444, 'ws_id': 4},
                ],
                ogr_geom_type=ogr.wkbUnknown)

            target_watersheds_path = os.path.join(
                self.workspace_dir, 'watersheds.gpkg')

            pygeoprocessing.routing.delineate_watersheds_d8(
                (flow_dir_path, 1), outflow_vector_path,
                target_watersheds_path,
                target_layer_name='watersheds_something')

            try:
                watersheds_vector = gdal.OpenEx(target_watersheds_path,
                                                gdal.OF_VECTOR)
                watersheds_layer = watersheds_vector.GetLayer(
                    'watersheds_something')
                self.assertEqual(watersheds_layer.GetFeatureCount(), 4)

                # All features should have the same watersheds, both in area
                # and geometry.
                flow_dir_bbox = pygeoprocessing.get_raster_info(
                    flow_dir_path)['bounding_box']
                expected_watershed_geometry = shapely.geometry.box(
                    *flow_dir_bbox)
                expected_watershed_geometry = (
                    expected_watershed_geometry.difference(
                        shapely.geometry.box(20, -2, 22, -10)))
                expected_watershed_geometry = (
                    expected_watershed_geometry.difference(
                        shapely.geometry.box(20, -12, 22, -22)))
                pygeoprocessing.shapely_geometry_to_vector(
                    [expected_watershed_geometry],
                    os.path.join(self.workspace_dir, 'foo.gpkg'), srs_wkt,
                    'GPKG', ogr_geom_type=ogr.wkbGeometryCollection)

                id_to_fields = {}
                for feature in watersheds_layer:
                    geometry = feature.GetGeometryRef()
                    shapely_geom = shapely.wkb.loads(
                        bytes(geometry.ExportToWkb()))
                    self.assertEqual(
                        shapely_geom.area, expected_watershed_geometry.area)
                    self.assertEqual(
                        shapely_geom.intersection(
                            expected_watershed_geometry).area,
                        expected_watershed_geometry.area)
                    self.assertEqual(
                        shapely_geom.difference(
                            expected_watershed_geometry).area, 0)

                    field_values = feature.items()
                    id_to_fields[field_values['polygon_id']] = field_values
            finally:
                watersheds_vector = None
                watersheds_layer = None

            try:
                outflow_vector = gdal.OpenEx(
                    outflow_vector_path, gdal.OF_VECTOR)
                outflow_layer = outflow_vector.GetLayer()
                found_ws_ids = set()  # make sure the ws_id field is copied
                for feature in outflow_layer:
                    self.assertEqual(
                        id_to_fields[feature.GetField('polygon_id')],
                        feature.items())
                    found_ws_ids.add(feature.GetField('ws_id'))
            finally:
                outflow_layer = None
                outflow_vector = None
            self.assertEqual(found_ws_ids, set([1, 2, 3, 4]))

    def test_split_geometry_into_seeds(self):
        """PGP watersheds: Test geometry-to-seed extraction."""
        nodata = 255
        flow_dir_array = numpy.array([
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
        flow_dir_info = pygeoprocessing.get_raster_info(flow_dir_path)

        point = shapely.geometry.Point(2.5, -2.5)
        linestring = shapely.geometry.LineString([(10, -2), (10, -9.9)])
        box = shapely.geometry.box(4.1, -7.9, 7.9, -4.1)

        for index, (geometry, expected_seeds) in enumerate((
                (point, set([(0, 0)])),
                # includes nodata pixel
                (linestring, set([(4, 0), (4, 1), (4, 2), (4, 3)])),
                # includes nodata pixels
                (box, set([(1, 1), (2, 1), (1, 2), (2, 2)])))):

            raster_path = os.path.join(self.workspace_dir, '%s.tif' % index)
            diagnostic_path = os.path.join(
                self.workspace_dir, '%s.gpkg' % index)
            result_seeds = watershed._split_geometry_into_seeds(
                geometry.wkb, flow_dir_info['geotransform'], srs,
                flow_dir_array.shape[1], flow_dir_array.shape[0],
                raster_path, diagnostic_path)

            self.assertEqual(result_seeds, expected_seeds)

    def test_split_geometry_into_seeds_willamette(self):
        """PGP watersheds: Test geometry-to-seed extraction in Willamette."""
        flow_dir_array = numpy.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]], dtype=numpy.uint8)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3157)  # UTM zone 17N
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
        pixel_xsize = 30
        pixel_ysize = -30
        flow_dir_geotransform = [
            443723, pixel_xsize, 0,
            4956546, 0, pixel_ysize]
        flow_dir_raster.SetGeoTransform(flow_dir_geotransform)
        flow_dir_raster = None
        flow_dir_info = pygeoprocessing.get_raster_info(flow_dir_path)

        pixel_indexes_array = numpy.arange(
            flow_dir_array.size).reshape(flow_dir_array.shape)
        pixel_indexes_path = os.path.join(
            self.workspace_dir, 'pixel_indexes.tif')
        driver = gdal.GetDriverByName('GTiff')
        pixel_indexes_raster = driver.Create(
            pixel_indexes_path, pixel_indexes_array.shape[1],
            pixel_indexes_array.shape[0], 1, gdal.GDT_Byte, options=(
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
        pixel_indexes_raster.SetProjection(srs_wkt)
        pixel_indexes_band = pixel_indexes_raster.GetRasterBand(1)
        pixel_indexes_band.WriteArray(pixel_indexes_array)
        pixel_indexes_band.SetNoDataValue(255)
        pixel_xsize = 30
        pixel_ysize = -30
        pixel_indexes_geotransform = [
            443723, pixel_xsize, 0,
            4956546, 0, pixel_ysize]
        pixel_indexes_raster.SetGeoTransform(pixel_indexes_geotransform)
        pixel_indexes_raster = None

        point = shapely.geometry.Point(
            flow_dir_geotransform[0] + pixel_xsize / 2.,
            flow_dir_geotransform[3] + pixel_ysize / 2.)
        linestring = shapely.geometry.LineString([
            (flow_dir_geotransform[0] + pixel_xsize * 4,
             # extend beyond y boundary
             flow_dir_geotransform[3] - pixel_ysize * 2),
            (flow_dir_geotransform[0] + pixel_xsize * 4,
             flow_dir_geotransform[3] + pixel_ysize * 5)])
        box = shapely.geometry.box(
            flow_dir_geotransform[0] + pixel_xsize * 2.1,
            flow_dir_geotransform[3] + pixel_ysize * 4.1,
            flow_dir_geotransform[0] + pixel_xsize * 3.9,
            flow_dir_geotransform[3] + pixel_ysize * 2.1)

        for index, (geometry, expected_seeds) in enumerate((
                (point, set([(0, 0)])),
                (linestring, set([(4, 0), (4, 1), (4, 2), (4, 3)])),
                (box, set([(2, 2), (2, 3), (3, 2), (3, 3)])))):

            result_seeds = watershed._split_geometry_into_seeds(
                geometry.wkb, flow_dir_info['geotransform'], srs,
                flow_dir_array.shape[1], flow_dir_array.shape[0],
                os.path.join(self.workspace_dir, '%s.tif' % index),
                os.path.join(self.workspace_dir, '%s.gpkg' % index))

            self.assertEqual(result_seeds, expected_seeds)
