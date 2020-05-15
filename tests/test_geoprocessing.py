"""pygeoprocessing.geoprocessing test suite."""
import os
import shutil
import tempfile
import time
import types
import unittest
import unittest.mock

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import scipy.ndimage
import shapely.geometry
import shapely.wkt

import pygeoprocessing
import pygeoprocessing.symbolic
from pygeoprocessing.geoprocessing_core import \
    DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS

_DEFAULT_ORIGIN = (444720, 3751320)
_DEFAULT_PIXEL_SIZE = (30, -30)
_DEFAULT_EPSG = 3116


def passthrough(x):
    """Use in testing simple raster calculator calls."""
    return x


def _geometry_to_vector(
        geometry_list, target_vector_path, projection_epsg=3116,
        vector_format='GeoJSON', fields=None, attribute_list=None,
        ogr_geom_type=ogr.wkbPolygon):
    """Passthrough to pygeoprocessing.shapely_geometry_to_vector."""
    projection = osr.SpatialReference()
    projection.ImportFromEPSG(projection_epsg)
    pygeoprocessing.shapely_geometry_to_vector(
        geometry_list, target_vector_path, projection.ExportToWkt(),
        vector_format, fields=fields, attribute_list=attribute_list,
        ogr_geom_type=ogr.wkbPolygon)


def _array_to_raster(
        base_array, target_nodata, target_path,
        creation_options=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1],
        pixel_size=_DEFAULT_PIXEL_SIZE, projection_epsg=_DEFAULT_EPSG,
        origin=_DEFAULT_ORIGIN):
    """Passthrough to pygeoprocessing.array_to_raster."""
    projection = osr.SpatialReference()
    projection_wkt = None
    if projection_epsg is not None:
        projection.ImportFromEPSG(projection_epsg)
        projection_wkt = projection.ExportToWkt()
    pygeoprocessing.numpy_array_to_raster(
        base_array, target_nodata, pixel_size, origin, projection_wkt,
        target_path, raster_driver_creation_tuple=('GTiff', creation_options))


class PyGeoprocessing10(unittest.TestCase):
    """Tests for the PyGeoprocesing 1.0 refactor."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_star_import(self):
        """PGP: verify we can use *-import statement."""
        # Actually trying out the `from pygeoprocessing import *` notation here
        # raises a SyntaxWarning.  Instead, I'll just ensure that every
        # attribute in pygeoprocessing.__all__ is a function that is available
        # at the pygeoprocessing level.
        import inspect
        for attrname in pygeoprocessing.__all__:
            try:
                func = getattr(pygeoprocessing, attrname)
                self.assertTrue(
                    isinstance(func, (
                        types.FunctionType, types.BuiltinFunctionType)) or
                    inspect.isroutine(func))
            except AttributeError:
                self.fail(('Function %s is in pygeoprocessing.__all__ but '
                           'is not exposed at the package level') % attrname)

    def test_version_loaded(self):
        """PGP: verify we can load the version."""
        try:
            # Verifies that there's a version attribute and it has a value.
            self.assertTrue(len(pygeoprocessing.__version__) > 0)
        except Exception:
            self.fail('Could not load pygeoprocessing version.')

    def test_reclassify_raster_missing_pixel_value(self):
        """PGP.geoprocessing: test reclassify raster with missing value."""
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        pixel_matrix[-1, 0] = test_value - 1  # making a bad value
        target_nodata = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path)

        value_map = {
            test_value: 100,
        }
        target_nodata = -1
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.reclassify_raster(
                (raster_path, 1), value_map, target_path, gdal.GDT_Float32,
                target_nodata, values_required=True)
        expected_message = (
            'The following 1 raster values [-0.5] from "%s" do not have ' %
            (raster_path,))
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_reclassify_raster(self):
        """PGP.geoprocessing: test reclassify raster."""
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        target_nodata = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path)

        value_map = {
            test_value: 100,
        }
        target_nodata = -1
        pygeoprocessing.reclassify_raster(
            (raster_path, 1), value_map, target_path, gdal.GDT_Float32,
            target_nodata, values_required=True)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        self.assertAlmostEqual(
            numpy.sum(target_array), n_pixels**2 * value_map[test_value])

    def test_reclassify_raster_no_raster_path_band(self):
        """PGP.geoprocessing: test reclassify raster is path band aware."""
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        target_nodata = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path)

        value_map = {
            test_value: 100,
        }
        target_nodata = -1
        # we expect a value error because we didn't pass a (path, band)
        # for the first argument
        with self.assertRaises(ValueError):
            pygeoprocessing.reclassify_raster(
                raster_path, value_map, target_path, gdal.GDT_Float32,
                target_nodata, values_required=True)

    def test_reclassify_raster_empty_value_map(self):
        """PGP.geoprocessing: test reclassify raster."""
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        target_nodata = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path)

        empty_value_map = {
        }
        target_nodata = -1
        with self.assertRaises(ValueError):
            pygeoprocessing.reclassify_raster(
                (raster_path, 1), empty_value_map, target_path,
                gdal.GDT_Float32, target_nodata, values_required=False)

    def test_reproject_vector(self):
        """PGP.geoprocessing: test reproject vector."""
        # Create polygon shapefile to reproject
        base_srs = osr.SpatialReference()
        # NAD83(CSRS) / UTM zone 10N
        base_srs.ImportFromEPSG(3157)
        extents = [
            443723.1273278, 4956276.905980, 443993.1273278, 4956546.905980]

        polygon_a = shapely.geometry.box(*extents)

        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.shp')
        field_name = 'id'
        test_driver = ogr.GetDriverByName('ESRI Shapefile')
        test_vector = test_driver.CreateDataSource(base_vector_path)
        test_layer = test_vector.CreateLayer('base_layer', srs=base_srs)

        field_defn = ogr.FieldDefn(field_name, ogr.OFTInteger)
        test_layer.CreateField(field_defn)
        layer_defn = test_layer.GetLayerDefn()

        test_feature = ogr.Feature(layer_defn)
        test_geometry = ogr.CreateGeometryFromWkb(polygon_a.wkb)
        test_feature.SetGeometry(test_geometry)
        test_feature.SetField(field_name, 0)
        test_layer.CreateFeature(test_feature)

        test_layer = None
        test_vector = None
        test_driver = None

        target_reference = osr.SpatialReference()
        # UTM zone 18N
        target_reference.ImportFromEPSG(26918)

        target_vector_path = os.path.join(
            self.workspace_dir, 'target_vector.shp')
        # create the file first so the model needs to deal with that
        target_file = open(target_vector_path, 'w')
        target_file.close()
        pygeoprocessing.reproject_vector(
            base_vector_path, target_reference.ExportToWkt(),
            target_vector_path, layer_id=0)

        vector = ogr.Open(target_vector_path)
        layer = vector.GetLayer()
        result_reference = layer.GetSpatialRef()

        layer = None
        vector = None

        self.assertTrue(
            osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                osr.SpatialReference(target_reference.ExportToWkt())))

    def test_reproject_vector_partial_fields(self):
        """PGP.geoprocessing: reproject vector with partial field copy."""
        # Create polygon shapefile to reproject
        base_srs = osr.SpatialReference()
        # NAD83(CSRS) / UTM zone 10N
        base_srs.ImportFromEPSG(3157)
        extents = [
            443723.1273278, 4956276.905980, 443993.1273278, 4956546.905980]

        polygon_a = shapely.geometry.box(*extents)

        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.shp')
        fields = {'id': 0, 'foo': 'bar'}
        ogr_types = {'id': ogr.OFTInteger, 'foo': ogr.OFTString}
        test_driver = ogr.GetDriverByName('ESRI Shapefile')
        test_vector = test_driver.CreateDataSource(base_vector_path)
        test_layer = test_vector.CreateLayer('base_layer', srs=base_srs)

        for field_name in fields.keys():
            field_defn = ogr.FieldDefn(field_name, ogr_types[field_name])
            test_layer.CreateField(field_defn)
        layer_defn = test_layer.GetLayerDefn()

        test_feature = ogr.Feature(layer_defn)
        test_geometry = ogr.CreateGeometryFromWkb(polygon_a.wkb)
        test_feature.SetGeometry(test_geometry)
        for field_name, field_val in fields.items():
            test_feature.SetField(field_name, field_val)
        test_layer.CreateFeature(test_feature)

        test_layer = None
        test_vector = None
        test_driver = None

        target_reference = osr.SpatialReference()
        # UTM zone 18N
        target_reference.ImportFromEPSG(26918)

        target_vector_path = os.path.join(
            self.workspace_dir, 'target_vector.shp')
        # create the file first so the model needs to deal with that
        target_file = open(target_vector_path, 'w')
        target_file.close()
        pygeoprocessing.reproject_vector(
            base_vector_path, target_reference.ExportToWkt(),
            target_vector_path, layer_id=0, copy_fields=['id'])

        vector = ogr.Open(target_vector_path)
        layer = vector.GetLayer()
        result_reference = layer.GetSpatialRef()
        layer_defn = layer.GetLayerDefn()
        self.assertTrue(
            osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                osr.SpatialReference(target_reference.ExportToWkt())))
        self.assertTrue(layer_defn.GetFieldCount(), 1)
        self.assertEqual(layer_defn.GetFieldIndex('id'), 0)

        target_vector_no_fields_path = os.path.join(
            self.workspace_dir, 'target_vector_no_fields.shp')
        pygeoprocessing.reproject_vector(
            base_vector_path, target_reference.ExportToWkt(),
            target_vector_no_fields_path, layer_id=0, copy_fields=False)
        layer = None
        vector = None

        vector = ogr.Open(target_vector_no_fields_path)
        layer = vector.GetLayer()
        result_reference = layer.GetSpatialRef()
        layer_defn = layer.GetLayerDefn()
        self.assertTrue(
            osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                osr.SpatialReference(target_reference.ExportToWkt())))
        self.assertTrue(layer_defn.GetFieldCount(), 0)
        layer = None
        vector = None

    def test_reproject_vector_latlon_to_utm(self):
        """PGP.geoprocessing: reproject vector from lat/lon to utm."""
        # Create polygon shapefile to reproject
        base_srs = osr.SpatialReference()
        # WGS84
        base_srs.ImportFromEPSG(4326)
        extents = [-123.71107369, 44.7600990, -121.71107369, 43.7600990]

        polygon_a = shapely.geometry.box(*extents)

        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.shp')
        field_name = 'id'
        test_driver = ogr.GetDriverByName('ESRI Shapefile')
        test_vector = test_driver.CreateDataSource(base_vector_path)
        test_layer = test_vector.CreateLayer('base_layer', srs=base_srs)

        field_defn = ogr.FieldDefn(field_name, ogr.OFTInteger)
        test_layer.CreateField(field_defn)
        layer_defn = test_layer.GetLayerDefn()

        test_feature = ogr.Feature(layer_defn)
        test_geometry = ogr.CreateGeometryFromWkb(polygon_a.wkb)
        test_feature.SetGeometry(test_geometry)
        test_feature.SetField(field_name, 0)
        test_layer.CreateFeature(test_feature)

        test_layer = None
        test_vector = None
        test_driver = None

        target_reference = osr.SpatialReference()
        # UTM zone 10N
        target_reference.ImportFromEPSG(3157)

        target_vector_path = os.path.join(
            self.workspace_dir, 'target_vector.shp')
        # create the file first so the model needs to deal with that
        target_file = open(target_vector_path, 'w')
        target_file.close()
        pygeoprocessing.reproject_vector(
            base_vector_path, target_reference.ExportToWkt(),
            target_vector_path, layer_id=0, copy_fields=['id'])

        vector = ogr.Open(target_vector_path)
        layer = vector.GetLayer()
        result_reference = layer.GetSpatialRef()
        layer_defn = layer.GetLayerDefn()
        self.assertTrue(
            osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                osr.SpatialReference(target_reference.ExportToWkt())))
        self.assertTrue(layer_defn.GetFieldCount(), 1)
        self.assertEqual(layer_defn.GetFieldIndex('id'), 0)

    def test_reproject_vector_utm_to_latlon(self):
        """PGP.geoprocessing: reproject vector from utm to lat/lon."""
        # Create polygon shapefile to reproject
        base_srs = osr.SpatialReference()
        # NAD83(CSRS) / UTM zone 10N
        base_srs.ImportFromEPSG(3157)
        extents = [
            443723.1273278, 4956276.905980, 443993.1273278, 4956546.905980]

        polygon_a = shapely.geometry.box(*extents)

        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.shp')
        field_name = 'id'
        test_driver = ogr.GetDriverByName('ESRI Shapefile')
        test_vector = test_driver.CreateDataSource(base_vector_path)
        test_layer = test_vector.CreateLayer('base_layer', srs=base_srs)

        field_defn = ogr.FieldDefn(field_name, ogr.OFTInteger)
        test_layer.CreateField(field_defn)
        layer_defn = test_layer.GetLayerDefn()

        test_feature = ogr.Feature(layer_defn)
        test_geometry = ogr.CreateGeometryFromWkb(polygon_a.wkb)
        test_feature.SetGeometry(test_geometry)
        test_feature.SetField(field_name, 0)
        test_layer.CreateFeature(test_feature)

        test_layer = None
        test_vector = None
        test_driver = None

        # Lat/Lon WGS84
        target_reference = osr.SpatialReference()
        target_reference.ImportFromEPSG(4326)

        target_vector_path = os.path.join(
            self.workspace_dir, 'target_vector.shp')
        # create the file first so the model needs to deal with that
        target_file = open(target_vector_path, 'w')
        target_file.close()
        pygeoprocessing.reproject_vector(
            base_vector_path, target_reference.ExportToWkt(),
            target_vector_path, layer_id=0, copy_fields=['id'])

        vector = ogr.Open(target_vector_path)
        layer = vector.GetLayer()
        result_reference = layer.GetSpatialRef()
        layer_defn = layer.GetLayerDefn()
        self.assertTrue(
            osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                osr.SpatialReference(target_reference.ExportToWkt())))
        self.assertTrue(layer_defn.GetFieldCount(), 1)
        self.assertEqual(layer_defn.GetFieldIndex('id'), 0)

    def test_reproject_vector_latlon_to_latlon(self):
        """PGP.geoprocessing: reproject vector from lat/lon to utm."""
        # Create polygon shapefile to reproject
        base_srs = osr.SpatialReference()
        # WGS84
        base_srs.ImportFromEPSG(4326)
        extents = [-123.71107369, 44.7600990, -121.71107369, 43.7600990]

        polygon_a = shapely.geometry.box(*extents)

        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.shp')
        field_name = 'id'
        test_driver = ogr.GetDriverByName('ESRI Shapefile')
        test_vector = test_driver.CreateDataSource(base_vector_path)
        test_layer = test_vector.CreateLayer('base_layer', srs=base_srs)

        field_defn = ogr.FieldDefn(field_name, ogr.OFTInteger)
        test_layer.CreateField(field_defn)
        layer_defn = test_layer.GetLayerDefn()

        test_feature = ogr.Feature(layer_defn)
        test_geometry = ogr.CreateGeometryFromWkb(polygon_a.wkb)
        test_feature.SetGeometry(test_geometry)
        test_feature.SetField(field_name, 0)
        test_layer.CreateFeature(test_feature)

        test_layer = None
        test_vector = None
        test_driver = None

        # Lat/Lon WGS84
        target_reference = osr.SpatialReference()
        target_reference.ImportFromEPSG(4326)

        target_vector_path = os.path.join(
            self.workspace_dir, 'target_vector.shp')
        # create the file first so the model needs to deal with that
        target_file = open(target_vector_path, 'w')
        target_file.close()
        pygeoprocessing.reproject_vector(
            base_vector_path, target_reference.ExportToWkt(),
            target_vector_path, layer_id=0)

        vector = ogr.Open(target_vector_path)
        layer = vector.GetLayer()
        result_reference = layer.GetSpatialRef()
        layer_defn = layer.GetLayerDefn()
        self.assertTrue(
            osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                osr.SpatialReference(target_reference.ExportToWkt())))
        # Since projecting to the same SRS, the vectors should be identical
        target_vector = gdal.OpenEx(target_vector_path, gdal.OF_VECTOR)
        target_layer = target_vector.GetLayer()
        self.assertEqual(target_layer.GetFeatureCount(), 1)
        feature = next(iter(target_layer))
        feature_geom = shapely.wkt.loads(
            feature.GetGeometryRef().ExportToWkt())
        self.assertTrue(feature_geom.almost_equals(polygon_a))

    def test_reproject_vector_utm_to_utm(self):
        """PGP.geoprocessing: reproject vector from utm to utm."""
        # Create polygon shapefile to reproject
        base_srs = osr.SpatialReference()
        # NAD83(CSRS) / UTM zone 10N
        base_srs.ImportFromEPSG(3157)
        extents = [
            443723.1273278, 4956276.905980, 443993.1273278, 4956546.905980]

        polygon_a = shapely.geometry.box(*extents)

        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.shp')
        field_name = 'id'
        test_driver = ogr.GetDriverByName('ESRI Shapefile')
        test_vector = test_driver.CreateDataSource(base_vector_path)
        test_layer = test_vector.CreateLayer('base_layer', srs=base_srs)

        field_defn = ogr.FieldDefn(field_name, ogr.OFTInteger)
        test_layer.CreateField(field_defn)
        layer_defn = test_layer.GetLayerDefn()

        test_feature = ogr.Feature(layer_defn)
        test_geometry = ogr.CreateGeometryFromWkb(polygon_a.wkb)
        test_feature.SetGeometry(test_geometry)
        test_feature.SetField(field_name, 0)
        test_layer.CreateFeature(test_feature)

        test_layer = None
        test_vector = None
        test_driver = None

        target_reference = osr.SpatialReference()
        # NAD83 / UTM 10N
        target_reference.ImportFromEPSG(26910)

        target_vector_path = os.path.join(
            self.workspace_dir, 'target_vector.shp')
        # create the file first so the model needs to deal with that
        target_file = open(target_vector_path, 'w')
        target_file.close()
        pygeoprocessing.reproject_vector(
            base_vector_path, target_reference.ExportToWkt(),
            target_vector_path, layer_id=0)

        vector = ogr.Open(target_vector_path)
        layer = vector.GetLayer()
        result_reference = layer.GetSpatialRef()
        layer_defn = layer.GetLayerDefn()
        self.assertTrue(
            osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                osr.SpatialReference(target_reference.ExportToWkt())))

    def test_calculate_disjoint_polygon_set(self):
        """PGP.geoprocessing: test calc_disjoint_poly no/intersection."""
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        layer.CreateField(ogr.FieldDefn('expected_value', ogr.OFTInteger))
        layer_defn = layer.GetLayerDefn()

        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1.0
        subpixel_size = 1./5. * pixel_size
        origin_x = 1.0
        origin_y = -1.0
        n = 1
        layer.StartTransaction()
        for row_index in range(n * 2):
            for col_index in range(n * 2):
                x_pos = origin_x + (
                    col_index*2 + 1 + col_index // 2) * subpixel_size
                y_pos = origin_y - (
                    row_index*2 + 1 + row_index // 2) * subpixel_size
                shapely_feature = shapely.geometry.Polygon([
                    (x_pos, y_pos),
                    (x_pos+subpixel_size, y_pos),
                    (x_pos+subpixel_size, y_pos-subpixel_size),
                    (x_pos, y_pos-subpixel_size),
                    (x_pos, y_pos)])
                new_feature = ogr.Feature(layer_defn)
                new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
                new_feature.SetGeometry(new_geometry)
                expected_value = row_index // 2 * n + col_index // 2
                new_feature.SetField('expected_value', expected_value)
                layer.CreateFeature(new_feature)
        layer.CommitTransaction()
        layer.SyncToDisk()

        result = pygeoprocessing.calculate_disjoint_polygon_set(
            vector_path, bounding_box=[-10, -10, -9, -9])
        self.assertTrue(not result)

        # otherwise none overlap:
        result = pygeoprocessing.calculate_disjoint_polygon_set(vector_path)
        self.assertEqual(len(result), 1, result)

    def test_zonal_stats_for_small_polygons(self):
        """PGP.geoprocessing: test small polygons for zonal stats."""
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        layer.CreateField(ogr.FieldDefn('expected_value', ogr.OFTInteger))
        layer_defn = layer.GetLayerDefn()

        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1.0
        subpixel_size = 1./5. * pixel_size
        origin_x = 1.0
        origin_y = -1.0
        n = 16
        layer.StartTransaction()
        for row_index in range(n * 2):
            for col_index in range(n * 2):
                x_pos = origin_x + (
                    col_index*2 + 1 + col_index // 2) * subpixel_size
                y_pos = origin_y - (
                    row_index*2 + 1 + row_index // 2) * subpixel_size
                shapely_feature = shapely.geometry.Polygon([
                    (x_pos, y_pos),
                    (x_pos+subpixel_size, y_pos),
                    (x_pos+subpixel_size, y_pos-subpixel_size),
                    (x_pos, y_pos-subpixel_size),
                    (x_pos, y_pos)])
                new_feature = ogr.Feature(layer_defn)
                new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
                new_feature.SetGeometry(new_geometry)
                expected_value = row_index // 2 * n + col_index // 2
                new_feature.SetField('expected_value', expected_value)
                layer.CreateFeature(new_feature)
        layer.CommitTransaction()
        layer.SyncToDisk()

        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        _array_to_raster(
            numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n)),
            -1, raster_path, projection_epsg=4326, origin=(origin_x, origin_y),
            pixel_size=(pixel_size, -pixel_size))

        zonal_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path)
        self.assertEqual(len(zonal_stats), 4*n*n)
        for poly_id in zonal_stats:
            feature = layer.GetFeature(poly_id)
            self.assertEqual(
                feature.GetField('expected_value'),
                zonal_stats[poly_id]['sum'])

    def test_zonal_stats_no_bb_overlap(self):
        """PGP.geoprocessing: test no vector bb raster overlap."""
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        layer_defn = layer.GetLayerDefn()

        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1.0
        subpixel_size = 1./5. * pixel_size
        origin_x = 1.0
        origin_y = -1.0
        n = 16
        layer.StartTransaction()
        x_pos = origin_x - n
        y_pos = origin_y + n
        shapely_feature = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)
        layer.CreateFeature(new_feature)
        layer.CommitTransaction()
        layer.SyncToDisk()
        layer = None
        vector = None

        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        _array_to_raster(
            numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n)),
            -1, raster_path)

        zonal_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path)
        for poly_id in zonal_stats:
            self.assertEqual(zonal_stats[poly_id]['sum'], 0.0)

    def test_zonal_stats_all_outside(self):
        """PGP.geoprocessing: test vector all outside raster."""
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        layer_defn = layer.GetLayerDefn()

        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1.0
        subpixel_size = 1./5. * pixel_size
        origin_x = 1.0
        origin_y = -1.0
        n = 16
        layer.StartTransaction()
        x_pos = origin_x - n
        y_pos = origin_y + n
        shapely_feature = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)
        layer.CreateFeature(new_feature)

        x_pos = origin_x + n*2
        y_pos = origin_y - n*2
        shapely_feature = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)

        x_pos = origin_x - subpixel_size*.99
        y_pos = origin_y + subpixel_size*.99
        shapely_feature = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)
        layer.CreateFeature(new_feature)

        x_pos = origin_x + (n+.99)
        y_pos = origin_y - (n+.99)
        shapely_feature = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)
        layer.CreateFeature(new_feature)

        layer.CommitTransaction()
        layer.SyncToDisk()

        # this will catch a polygon that barely intersects the upper left
        # hand corner but is nodata.
        array = numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))
        array[0, 0] = -1
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        _array_to_raster(array, -1, raster_path)

        zonal_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path)
        for poly_id in zonal_stats:
            self.assertEqual(zonal_stats[poly_id]['sum'], 0.0)

        # this will catch a polygon that barely intersects the upper left
        # hand corner but is nodata.
        raster_path = os.path.join(
            self.workspace_dir, 'nonodata_small_raster.tif')
        array = numpy.fliplr(numpy.flipud(
            numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))))
        _array_to_raster(
            array, None, raster_path, projection_epsg=4326,
            origin=(origin_x+n, origin_y-n), pixel_size=(-1, 1))

        zonal_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path)
        for poly_id in zonal_stats:
            self.assertEqual(zonal_stats[poly_id]['sum'], 0.0)

    def test_mask_raster(self):
        """PGP.geoprocessing: test mask raster."""
        origin_x = 1.0
        origin_y = -1.0
        n = 16
        test_val = 2
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        shapely_feature = shapely.geometry.Polygon([
            (origin_x, origin_y),
            (origin_x+n, origin_y),
            (origin_x+n, origin_y-n//2),
            (origin_x, origin_y-n//2),
            (origin_x, origin_y)])
        _geometry_to_vector(
            [shapely_feature], vector_path, projection_epsg=4326,
            vector_format='GPKG')

        array = numpy.empty((n, n), dtype=numpy.int32)
        array[:] = test_val
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        _array_to_raster(
            array, None, raster_path, projection_epsg=4326,
            origin=(origin_x, origin_y), pixel_size=(1, -1))

        target_mask_raster_path = os.path.join(
            self.workspace_dir, 'test_mask.tif')
        pygeoprocessing.mask_raster(
            (raster_path, 1), vector_path, target_mask_raster_path,
            target_mask_value=None, working_dir=self.workspace_dir)

        mask_array = pygeoprocessing.raster_to_numpy_array(
            target_mask_raster_path)
        expected_result = numpy.empty((n, n))
        expected_result[0:8, :] = test_val
        expected_result[8::, :] = 0
        self.assertTrue(
            numpy.count_nonzero(numpy.isclose(
                mask_array, expected_result)) == n**2,
            msg=f'expected: {expected_result}\ngot: {mask_array}')

        pygeoprocessing.mask_raster(
            (raster_path, 1), vector_path, target_mask_raster_path,
            target_mask_value=12, working_dir=self.workspace_dir)

        mask_array = pygeoprocessing.raster_to_numpy_array(
            target_mask_raster_path)
        expected_result = numpy.empty((16, 16))
        expected_result[0:8, :] = 2
        expected_result[8::, :] = 12
        self.assertTrue(
            numpy.count_nonzero(numpy.isclose(
                mask_array, expected_result)) == 16**2,
            msg=f'expected: {expected_result}\ngot: {mask_array}')

    def test_zonal_statistics(self):
        """PGP.geoprocessing: test zonal stats function."""
        # create aggregating polygon
        pixel_size = 30.0
        n_pixels = 9
        origin = (444720, 3751320)
        polygon_a = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels, origin[1]),
            (origin[0], origin[1])])
        origin = (444720, 3751320)
        polygon_b = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size+origin[1]),
            (origin[0]+pixel_size, -pixel_size+origin[1]),
            (origin[0]+pixel_size, origin[1]),
            (origin[0], origin[1])])
        polygon_c = shapely.geometry.Polygon([
            (origin[1]*2, origin[1]*3),
            (origin[1]*2, -pixel_size+origin[1]*3),
            (origin[1]*2+pixel_size,
             -pixel_size+origin[1]*3),
            (origin[1]*2+pixel_size, origin[1]*3),
            (origin[1]*2, origin[1]*3)])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector')
        _geometry_to_vector(
            [polygon_a, polygon_b, polygon_c], aggregating_vector_path)
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        target_nodata = None
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path)
        result = pygeoprocessing.zonal_statistics(
            (raster_path, 1), aggregating_vector_path,
            aggregate_layer_name=None,
            ignore_nodata=True,
            polygons_might_overlap=True)
        expected_result = {
            0: {
                'count': 81,
                'max': 1.0,
                'min': 1.0,
                'nodata_count': 0,
                'sum': 81.0},
            1: {
                'count': 1,
                'max': 1.0,
                'min': 1.0,
                'nodata_count': 0,
                'sum': 1.0},
            2: {
                'min': None,
                'max': None,
                'count': 0,
                'nodata_count': 0,
                'sum': 0.0}}
        self.assertEqual(result, expected_result)

    def test_zonal_statistics_nodata(self):
        """PGP.geoprocessing: test zonal stats function with non-overlap."""
        # create aggregating polygon
        pixel_size = 30.0
        n_pixels = 9
        origin = (444720, 3751320)
        polygon_a = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels, origin[1]),
            (origin[0], origin[1])])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector')
        aggregate_field_name = 'id'
        _geometry_to_vector(
            [polygon_a], aggregating_vector_path,
            fields={'id': ogr.OFTInteger}, attribute_list=[
                {aggregate_field_name: 0}])
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        target_nodata = -1
        pixel_matrix[:] = target_nodata
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path)
        result = pygeoprocessing.zonal_statistics(
            (raster_path, 1), aggregating_vector_path,
            aggregate_layer_name=None,
            ignore_nodata=True,
            polygons_might_overlap=False)
        expected_result = {
            0: {
                'count': 0,
                'max': None,
                'min': None,
                'nodata_count': 81,
                'sum': 0.0}}
        self.assertEqual(result, expected_result)

    def test_zonal_statistics_nodata_is_zero(self):
        """PGP.geoprocessing: test zonal stats function w/ nodata set to 0."""
        # create aggregating polygon
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        origin_x = 1.0
        origin_y = -1.0
        n = 2
        _geometry_to_vector(
            [shapely.geometry.Polygon([
                 (origin_x, origin_y),
                 (origin_x+n, origin_y),
                 (origin_x+n, origin_y-n),
                 (origin_x, origin_y-n),
                 (origin_x, origin_y)])],
            vector_path, projection_epsg=4326, vector_format='gpkg')

        # create raster with nodata value of 0
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        _array_to_raster(
            numpy.array([[1, 0], [1, 0]], dtype=numpy.int32), 0, raster_path,
            origin=(origin_x, origin_y), pixel_size=(1.0, -1.0))

        result = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path,
            aggregate_layer_name=None,
            ignore_nodata=True,
            polygons_might_overlap=False)
        expected_result = {
            1: {
                'count': 2,
                'max': 1,
                'min': 1,
                'nodata_count': 2,
                'sum': 2.0}}
        self.assertEqual(result, expected_result)

    def test_zonal_statistics_named_layer(self):
        """PGP.geoprocessing: test zonal stats with named layer."""
        # create aggregating polygon
        pixel_size = 30.0
        n_pixels = 9
        origin = (444720, 3751320)
        polygon_a = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels, origin[1]),
            (origin[0], origin[1])])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector.shp')
        _geometry_to_vector(
            [polygon_a], aggregating_vector_path,
            vector_format='ESRI Shapefile')
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        target_nodata = None
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path)
        result = pygeoprocessing.zonal_statistics(
            (raster_path, 1), aggregating_vector_path,
            aggregate_layer_name='aggregate_vector',
            ignore_nodata=True,
            polygons_might_overlap=True)
        expected_result = {
            0: {
                'count': 81,
                'max': 1.0,
                'min': 1.0,
                'nodata_count': 0,
                'sum': 81.0}}
        self.assertEqual(result, expected_result)

    def test_zonal_statistics_bad_vector(self):
        """PGP.geoprocessing: zonal stats raises exception on bad vectors."""
        # create aggregating polygon
        n_pixels = 9
        missing_aggregating_vector_path = os.path.join(
            self.workspace_dir, 'not_exists.shp')
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        target_nodata = None
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path)
        with self.assertRaises(RuntimeError) as cm:
            _ = pygeoprocessing.zonal_statistics(
                (raster_path, 1), missing_aggregating_vector_path,
                ignore_nodata=True,
                polygons_might_overlap=True)
        expected_message = 'Could not open aggregate vector'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        pixel_size = 30.0
        origin = (444720, 3751320)
        polygon_a = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels, origin[1]),
            (origin[0], origin[1])])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector.shp')
        _geometry_to_vector(
            [polygon_a], aggregating_vector_path,
            vector_format='ESRI Shapefile')
        with self.assertRaises(RuntimeError) as cm:
            _ = pygeoprocessing.zonal_statistics(
                (raster_path, 1), aggregating_vector_path,
                ignore_nodata=True,
                aggregate_layer_name='not a layer name',
                polygons_might_overlap=True)
        expected_message = 'Could not open layer not a layer name'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_zonal_statistics_bad_raster_path_band(self):
        """PGP.geoprocessing: test zonal stats with bad raster/path type."""
        pixel_size = 30.0
        n_pixels = 9
        origin = (444720, 3751320)
        polygon_a = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels, origin[1]),
            (origin[0], origin[1])])
        origin = (444720, 3751320)
        polygon_b = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size+origin[1]),
            (origin[0]+pixel_size, -pixel_size+origin[1]),
            (origin[0]+pixel_size, origin[1]),
            (origin[0], origin[1])])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector')
        aggregate_field_name = 'id'
        _geometry_to_vector(
            [polygon_a, polygon_b], aggregating_vector_path,
            fields={'id': ogr.OFTString}, attribute_list=[
                {aggregate_field_name: '0'}, {aggregate_field_name: '1'}])
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        target_nodata = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path)
        with self.assertRaises(ValueError):
            # intentionally not passing a (path, band) tuple as first arg
            _ = pygeoprocessing.zonal_statistics(
                raster_path, aggregating_vector_path,
                aggregate_layer_name=None,
                ignore_nodata=True,
                polygons_might_overlap=True)

    def test_interpolate_points(self):
        """PGP.geoprocessing: test interpolate points feature."""
        # construct a point shapefile
        origin = (444720, 3751320)
        point_a = shapely.geometry.Point(
            origin[0] + 30 * 9 / 2, origin[1])
        point_b = shapely.geometry.Point(
            origin[0] + 30 * 9 / 2, origin[1] + -30 * 9)
        source_vector_path = os.path.join(self.workspace_dir, 'sample_vector')
        _geometry_to_vector(
            [point_a, point_b], source_vector_path,
            fields={'value': ogr.OFTInteger},
            attribute_list=[{'value': 0}, {'value': 1}])
        # construct a raster
        pixel_matrix = numpy.ones((9, 9), numpy.float32)
        target_nodata = -1
        result_path = os.path.join(self.workspace_dir, 'result.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, result_path)

        # interpolate
        pygeoprocessing.interpolate_points(
            source_vector_path, 'value', (result_path, 1), 'near')

        # verify that result is expected
        result_array = pygeoprocessing.raster_to_numpy_array(result_path)

        # we expect the first 4 rows to be 0, then the last ones to be 1
        expected_result = numpy.ones((9, 9), numpy.float32)
        expected_result[:5, :] = 0

        numpy.testing.assert_array_equal(result_array, expected_result)

    def test_invoke_timed_callback(self):
        """PGP.geoprocessing: cover a timed callback."""
        reference_time = time.time()
        time.sleep(0.1)
        new_time = pygeoprocessing.geoprocessing._invoke_timed_callback(
            reference_time, lambda: None, 0.05)
        self.assertNotEqual(reference_time, new_time)

    def test_warp_raster(self):
        """PGP.geoprocessing: warp raster test."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path)

        target_raster_path = os.path.join(self.workspace_dir, 'target_a.tif')
        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        pygeoprocessing.warp_raster(
            base_a_path, base_a_raster_info['pixel_size'], target_raster_path,
            'near', n_threads=1)

        self.assertTrue(
            numpy.isclose(
                pygeoprocessing.raster_to_numpy_array(base_a_path),
                pygeoprocessing.raster_to_numpy_array(target_raster_path)).all())

    def test_warp_raster_unusual_pixel_size(self):
        """PGP.geoprocessing: warp on unusual pixel types and sizes."""
        pixel_a_matrix = numpy.ones((1, 1), numpy.byte)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path,
            creation_options=['PIXELTYPE=SIGNEDBYTE'], pixel_size=(20, -20),
            projection_epsg=4326)

        target_raster_path = os.path.join(self.workspace_dir, 'target_a.tif')

        # convert 1x1 pixel to a 30x30m pixel
        wgs84_projection = osr.SpatialReference()
        wgs84_projection.ImportFromEPSG(4326)
        pygeoprocessing.warp_raster(
            base_a_path, [-30, 30], target_raster_path,
            'near', target_projection_wkt=wgs84_projection.ExportToWkt())

        expected_raster_path = os.path.join(
            self.workspace_dir, 'expected.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, expected_raster_path,
            creation_options=['PIXELTYPE=SIGNEDBYTE'], pixel_size=(30, -30),
            projection_epsg=4326)

        self.assertTrue(
            numpy.isclose(
                pygeoprocessing.raster_to_numpy_array(base_a_path),
                pygeoprocessing.raster_to_numpy_array(expected_raster_path)).all())

    def test_warp_raster_0x0_size(self):
        """PGP.geoprocessing: test warp where so small it would be 0x0."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path)

        target_raster_path = os.path.join(self.workspace_dir, 'target_a.tif')
        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)
        target_bb = base_a_raster_info['bounding_box']
        # pick a tiny tiny bounding box in the middle (less than a pixel big)
        target_bb[0] = (target_bb[0] + target_bb[2]) / 2.0
        target_bb[1] = (target_bb[1] + target_bb[3]) / 2.0
        target_bb[2] = target_bb[0]
        target_bb[3] = target_bb[1]
        pygeoprocessing.warp_raster(
            base_a_path, base_a_raster_info['pixel_size'], target_raster_path,
            'near', target_bb=target_bb)

        expected_raster_path = os.path.join(
            self.workspace_dir, 'expected.tif')
        expected_matrix = numpy.ones((1, 1), numpy.int16)
        _array_to_raster(
            expected_matrix, target_nodata, expected_raster_path)

        self.assertTrue(
            numpy.isclose(
                pygeoprocessing.raster_to_numpy_array(base_a_path),
                pygeoprocessing.raster_to_numpy_array(expected_raster_path)).all())

    def test_align_and_resize_raster_stack_bad_values(self):
        """PGP.geoprocessing: align/resize raster bad base values."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path)

        base_raster_path_list = [base_a_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b']]

        resample_method_list = ['near']
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        with self.assertRaises(ValueError) as cm:
            # here base_raster_path_list is length 1 but others are length 2
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, target_raster_path_list,
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)
        expected_message = 'must be the same length'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            # here pixel size is an invalid type
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, ['target_a.tif'],
                resample_method_list,
                100.0, bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)
        expected_message = 'target_pixel_size is not a tuple'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            # here pixel size has invalid values
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, ['target_a.tif'],
                resample_method_list,
                [100.0, "ten"], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)
        expected_message = 'Invalid value for `target_pixel_size`'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            # here pixel size is too long
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, ['target_a.tif'],
                resample_method_list,
                [100.0, 100.0, 100.0], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)
        expected_message = 'Invalid value for `target_pixel_size`'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_align_and_resize_raster_stack_duplicate_outputs(self):
        """PGP.geoprocessing: align/resize raster duplicate outputs."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path)

        base_raster_path_list = [base_a_path, base_a_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'a']]

        resample_method_list = ['near'] * 2
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        with self.assertRaises(ValueError) as cm:
            # here base_raster_path_list is length 1 but others are length 2
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, target_raster_path_list,
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)

        expected_message = 'There are duplicated paths on the target list.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_align_and_resize_raster_stack_bad_mode(self):
        """PGP.geoprocessing: align/resize raster bad bounding box mode."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path)

        base_raster_path_list = [base_a_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_a.tif')]

        resample_method_list = ['near']
        bounding_box_mode = 'bad_mode'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        with self.assertRaises(ValueError):
            # here base_raster_path_list is length 1 but others are length 2
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, target_raster_path_list,
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)

    def test_align_and_resize_raster_stack_bad_index(self):
        """PGP.geoprocessing: align/resize raster test intersection."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path)

        base_raster_path_list = [base_a_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_a.tif')]

        resample_method_list = ['near']
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        with self.assertRaises(ValueError):
            # here align index is -1 which is invalid
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, target_raster_path_list,
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=-1)

    def test_align_and_resize_raster_stack_int(self):
        """PGP.geoprocessing: align/resize raster test intersection."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path)

        pixel_b_matrix = numpy.ones((15, 15), numpy.int16)
        target_nodata = -1
        base_b_path = os.path.join(self.workspace_dir, 'base_b.tif')
        _array_to_raster(
            pixel_b_matrix, target_nodata, base_b_path)

        pixel_c_matrix = numpy.ones((15, 5), numpy.int16)
        target_nodata = -1
        base_c_path = os.path.join(self.workspace_dir, 'base_c.tif')
        _array_to_raster(
            pixel_c_matrix, target_nodata, base_c_path, pixel_size=(45, -45))

        pixel_d_matrix = numpy.ones((5, 10), numpy.int16)
        target_nodata = -1
        base_d_path = os.path.join(self.workspace_dir, 'base_d.tif')
        _array_to_raster(pixel_d_matrix, target_nodata, base_d_path)

        base_raster_path_list = [
            base_a_path, base_b_path, base_c_path, base_d_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b', 'c', 'd']]

        resample_method_list = ['near'] * len(target_raster_path_list)
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list, target_raster_path_list,
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            base_vector_path_list=None, raster_align_index=0)

        for raster_index in range(len(target_raster_path_list)):
            target_raster_info = pygeoprocessing.get_raster_info(
                target_raster_path_list[raster_index])
            target_array = pygeoprocessing.raster_to_numpy_array(
                target_raster_path_list[raster_index])
            numpy.testing.assert_array_equal(pixel_a_matrix, target_array)
            self.assertEqual(
                target_raster_info['pixel_size'],
                base_a_raster_info['pixel_size'])

    def test_align_and_resize_raster_stack_int_with_vectors(self):
        """PGP.geoprocessing: align/resize raster test inters. w/ vectors."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path)

        pixel_b_matrix = numpy.ones((15, 15), numpy.int16)
        target_nodata = -1
        base_b_path = os.path.join(self.workspace_dir, 'base_b.tif')
        _array_to_raster(
            pixel_b_matrix, target_nodata, base_b_path)

        base_raster_path_list = [base_a_path, base_b_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b']]

        resample_method_list = ['near'] * 2
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        # make a vector whose bounding box is 1 pixel large
        point_a = shapely.geometry.Point(
            _DEFAULT_ORIGIN[0], _DEFAULT_ORIGIN[1])
        point_b = shapely.geometry.Point(
            _DEFAULT_ORIGIN[0] + _DEFAULT_PIXEL_SIZE[0],
            _DEFAULT_ORIGIN[1] + _DEFAULT_PIXEL_SIZE[1])
        single_pixel_path = os.path.join(self.workspace_dir, 'single_pixel')
        _geometry_to_vector(
            [point_a, point_b], single_pixel_path,
            fields={'value': ogr.OFTInteger},
            attribute_list=[{'value': 0}, {'value': 1}])

        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list, target_raster_path_list,
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            raster_align_index=0, base_vector_path_list=[single_pixel_path])

        expected_matrix = numpy.ones((1, 1), numpy.int16)
        for raster_index in range(2):
            target_raster_info = pygeoprocessing.get_raster_info(
                target_raster_path_list[raster_index])
            target_array = pygeoprocessing.raster_to_numpy_array(
                target_raster_path_list[raster_index])
            numpy.testing.assert_array_equal(expected_matrix, target_array)
            self.assertEqual(
                target_raster_info['pixel_size'],
                base_a_raster_info['pixel_size'])

    def test_align_and_resize_raster_stack_manual_projection(self):
        """PGP.geoprocessing: align/resize with manual projections."""
        base_raster_path = os.path.join(self.workspace_dir, 'base_raster.tif')
        pixel_matrix = numpy.ones((1, 1), numpy.int16)
        _array_to_raster(
            pixel_matrix, -1, base_raster_path, projection_epsg=4326,
            origin=(1, 1), pixel_size=(1, -1))
        utm_30n_sr = osr.SpatialReference()
        utm_30n_sr.ImportFromEPSG(32630)
        wgs84_sr = osr.SpatialReference()
        wgs84_sr.ImportFromEPSG(4326)

        target_raster_path = os.path.join(self.workspace_dir, 'target.tif')
        target_pixel_size = (112000/4, -112000/4)

        pygeoprocessing.align_and_resize_raster_stack(
            [base_raster_path], [target_raster_path],
            ['near'], target_pixel_size, 'intersection',
            raster_align_index=0,
            base_projection_wkt_list=[wgs84_sr.ExportToWkt()],
            target_projection_wkt=utm_30n_sr.ExportToWkt())

        target_array = pygeoprocessing.raster_to_numpy_array(
            target_raster_path)
        numpy.testing.assert_almost_equal(
            target_array, numpy.ones((4, 4)))

    def test_align_and_resize_raster_stack_no_base_projection(self):
        """PGP.geoprocessing: align raise error if no base projection."""
        base_raster_path = os.path.join(self.workspace_dir, 'base_raster.tif')
        pixel_matrix = numpy.ones((1, 1), numpy.int16)
        _array_to_raster(
            pixel_matrix, -1, base_raster_path, projection_epsg=None,
            origin=(1, 1), pixel_size=(1, -1))

        utm_30n_sr = osr.SpatialReference()
        utm_30n_sr.ImportFromEPSG(32630)
        wgs84_sr = osr.SpatialReference()
        wgs84_sr.ImportFromEPSG(4326)

        target_raster_path = os.path.join(self.workspace_dir, 'target.tif')
        target_pixel_size = (112000/4, -112000/4)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.align_and_resize_raster_stack(
                [base_raster_path], [target_raster_path],
                ['near'], target_pixel_size, 'intersection',
                raster_align_index=0,
                base_projection_wkt_list=[None],
                target_projection_wkt=utm_30n_sr.ExportToWkt())
            expected_message = "no projection for raster"
            actual_message = str(cm.exception)
            self.assertTrue(
                expected_message in actual_message, actual_message)

    def test_align_and_resize_raster_stack_no_overlap(self):
        """PGP.geoprocessing: align/resize raster no intersection error."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path, origin=[-10*30, 10*30])

        pixel_b_matrix = numpy.ones((15, 15), numpy.int16)
        target_nodata = -1
        base_b_path = os.path.join(self.workspace_dir, 'base_b.tif')
        _array_to_raster(pixel_b_matrix, target_nodata, base_b_path)

        base_raster_path_list = [base_a_path, base_b_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b']]

        resample_method_list = ['near'] * 2
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        # make a vector whose bounding box is 1 pixel large
        point_a = shapely.geometry.Point(
            _DEFAULT_ORIGIN[0], _DEFAULT_ORIGIN[1])
        point_b = shapely.geometry.Point(
            _DEFAULT_ORIGIN[0] + _DEFAULT_PIXEL_SIZE[0],
            _DEFAULT_ORIGIN[1] + _DEFAULT_PIXEL_SIZE[1])
        single_pixel_path = os.path.join(self.workspace_dir, 'single_pixel')
        _geometry_to_vector(
            [point_a, point_b], single_pixel_path,
            fields={'value': ogr.OFTInteger},
            attribute_list=[{'value': 0}, {'value': 1}])

        with self.assertRaises(ValueError):
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, target_raster_path_list,
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                raster_align_index=0,
                base_vector_path_list=[single_pixel_path])

    def test_align_and_resize_raster_stack_union(self):
        """PGP.geoprocessing: align/resize raster test union."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path, pixel_size=(30, -30))

        pixel_b_matrix = numpy.ones((10, 10), numpy.int16)
        target_nodata = -1
        base_b_path = os.path.join(self.workspace_dir, 'base_b.tif')
        _array_to_raster(
            pixel_b_matrix, target_nodata, base_b_path, pixel_size=(60, -60))

        base_raster_path_list = [base_a_path, base_b_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b']]

        resample_method_list = ['near'] * 2
        bounding_box_mode = 'union'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list, target_raster_path_list,
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            base_vector_path_list=None, raster_align_index=0)

        # we expect this to be twice as big since second base raster has a
        # pixel size twice that of the first.
        expected_matrix_a = numpy.ones((20, 20), numpy.int16)
        expected_matrix_a[5:, :] = target_nodata
        expected_matrix_a[:, 5:] = target_nodata

        target_array = pygeoprocessing.raster_to_numpy_array(
            target_raster_path_list[0])
        numpy.testing.assert_array_equal(expected_matrix_a, target_array)

    def test_align_and_resize_raster_stack_bb(self):
        """PGP.geoprocessing: align/resize raster test bounding box."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path, pixel_size=(30, -30))

        pixel_b_matrix = numpy.ones((10, 10), numpy.int16)
        target_nodata = -1
        base_b_path = os.path.join(self.workspace_dir, 'base_b.tif')
        _array_to_raster(
            pixel_b_matrix, target_nodata, base_b_path, pixel_size=(30, -30))

        base_raster_path_list = [base_a_path, base_b_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b']]

        resample_method_list = ['near'] * 2
        # format is xmin,ymin,xmax,ymax; since y pixel size is negative it
        # goes first in the following bounding box construction
        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list, target_raster_path_list,
            resample_method_list,
            base_a_raster_info['pixel_size'], 'intersection',
            base_vector_path_list=None, raster_align_index=0)

        # we expect this to be twice as big since second base raster has a
        # pixel size twice that of the first.
        target_array = pygeoprocessing.raster_to_numpy_array(
            target_raster_path_list[0])
        numpy.testing.assert_array_equal(pixel_a_matrix, target_array)

    def test_raster_calculator(self):
        """PGP.geoprocessing: raster_calculator identity test."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        _array_to_raster(pixel_matrix, target_nodata, base_path)

        target_path = os.path.join(self.workspace_dir, 'subdir', 'target.tif')

        pygeoprocessing.raster_calculator(
            [(base_path, 1)], passthrough, target_path,
            gdal.GDT_Int32, target_nodata, calc_raster_stats=True)

        self.assertTrue(
            numpy.isclose(
                pygeoprocessing.raster_to_numpy_array(base_path),
                pygeoprocessing.raster_to_numpy_array(target_path)).all())

    def test_raster_calculator_bad_target_type(self):
        """PGP.geoprocessing: raster_calculator bad target type value."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        _array_to_raster(pixel_matrix, target_nodata, base_path)

        target_path = os.path.join(
            self.workspace_dir, 'subdir', 'target.tif')
        # intentionally reversing `target_nodata` and `gdal.GDT_Int32`,
        # a value of -1 should be a value error for the target
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(base_path, 1)], passthrough, target_path,
                target_nodata, gdal.GDT_Int32)
        expected_message = (
            'Invalid target type, should be a gdal.GDT_* type')
        actual_message = str(cm.exception)
        self.assertTrue(
            expected_message in actual_message, actual_message)
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        _array_to_raster(pixel_matrix, target_nodata, base_path)

        target_path = os.path.join(
            self.workspace_dir, 'target.tif')
        for bad_raster_path_band_list in [
                [base_path], [(base_path, "1")], [(1, 1)],
                [(base_path, 1, base_path, 2)], base_path]:
            with self.assertRaises(ValueError) as cm:
                pygeoprocessing.raster_calculator(
                    bad_raster_path_band_list, passthrough, target_path,
                    gdal.GDT_Int32, target_nodata, calc_raster_stats=True)
            expected_message = (
                'Expected a sequence of path / integer band tuples, '
                'ndarrays, ')
            actual_message = str(cm.exception)
            self.assertTrue(
                expected_message in actual_message, actual_message)

    def test_raster_calculator_no_path(self):
        """PGP.geoprocessing: raster_calculator raise ex. on bad file path."""
        target_nodata = -1
        nonexistant_path = os.path.join(self.workspace_dir, 'nofile.tif')
        target_path = os.path.join(
            self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(nonexistant_path, 1)], passthrough, target_path,
                gdal.GDT_Int32, target_nodata, calc_raster_stats=True)
        expected_message = (
            "The following files were expected but do not exist on the ")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_raster_calculator_nodata(self):
        """PGP.geoprocessing: raster_calculator test with all nodata."""
        pixel_matrix = numpy.empty((5, 5), numpy.int16)
        target_nodata = -1
        pixel_matrix[:] = target_nodata
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        _array_to_raster(pixel_matrix, target_nodata, base_path)

        target_path = os.path.join(
            self.workspace_dir, 'target.tif')
        pygeoprocessing.raster_calculator(
            [(base_path, 1)], passthrough, target_path,
            gdal.GDT_Int32, target_nodata, calc_raster_stats=True)

        self.assertTrue(
            numpy.isclose(
                pygeoprocessing.raster_to_numpy_array(base_path),
                pygeoprocessing.raster_to_numpy_array(target_path)).all())

    def test_rs_calculator_output_alias(self):
        """PGP.geoprocessing: rs_calculator expected error for aliasing."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        nodata_base = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        _array_to_raster(pixel_matrix, nodata_base, base_path)

        with self.assertRaises(ValueError) as cm:
            # intentionally passing target path as base path to raise error
            pygeoprocessing.raster_calculator(
                [(base_path, 1)], passthrough, base_path,
                gdal.GDT_Int32, nodata_base, calc_raster_stats=True)
        expected_message = 'is used as a target path, but it is also in the '
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_rs_calculator_bad_overlap(self):
        """PGP.geoprocessing: rs_calculator expected error on bad overlap."""
        pixel_matrix_a = numpy.ones((5, 5), numpy.int16)
        nodata_base = -1
        base_path_a = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_matrix_a, nodata_base, base_path_a)

        pixel_matrix_b = numpy.ones((4, 5), numpy.int16)
        base_path_b = os.path.join(self.workspace_dir, 'base_b.tif')
        _array_to_raster(pixel_matrix_b, nodata_base, base_path_b)

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(base_path_a, 1), (base_path_b, 1)], passthrough,
                target_path, gdal.GDT_Int32, nodata_base,
                calc_raster_stats=True)
        expected_message = 'Input Rasters are not the same dimensions.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_raster_calculator_constant_args_error(self):
        """PGP.geoprocessing: handle empty input arrays."""
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError) as cm:
            # no input args should cause a ValueError
            pygeoprocessing.raster_calculator(
                [], lambda: None, target_path,
                gdal.GDT_Float32, None)
        expected_message = '`base_raster_path_band_const_list` is empty'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_raster_calculator_invalid_numpy_array(self):
        """PGP.geoprocessing: handle invalid numpy array sizes."""
        target_path = os.path.join(self.workspace_dir, 'target.tif')

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [numpy.empty((3, 3, 3))], lambda x: None, target_path,
                gdal.GDT_Float32, None)
        expected_message = 'Numpy array inputs must be 2 dimensions or less'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_raster_calculator_invalid_band_numbers(self):
        """PGP.geoprocessing: ensure invalid band numbers fail."""
        driver = gdal.GetDriverByName("GTiff")
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        new_raster = driver.Create(
            base_path, 128, 128, 1, gdal.GDT_Int32,
            options=(
                'TILED=YES', 'BLOCKXSIZE=16', 'BLOCKYSIZE=16'))
        new_raster.FlushCache()
        new_raster = None

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError) as cm:
            # no input args should cause a ValueError
            pygeoprocessing.raster_calculator(
                [(base_path, 2)], lambda: None, target_path,
                gdal.GDT_Float32, None)
        expected_message = "do not contain requested band "
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

        with self.assertRaises(ValueError) as cm:
            # no input args should cause a ValueError
            pygeoprocessing.raster_calculator(
                [(base_path, 0)], lambda: None, target_path,
                gdal.GDT_Float32, None)
        expected_message = "do not contain requested band "
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_raster_calculator_unbroadcastable_array(self):
        """PGP.geoprocessing: incompatable array sizes raise error."""
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        a_arg = 3
        x_arg = numpy.array(range(2))
        y_arg = numpy.array(range(3)).reshape((3, 1))
        z_arg = numpy.ones((4, 4))
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(a_arg, 'raw'), x_arg, y_arg, z_arg],
                lambda a, x, y, z: a*x*y*z, target_path, gdal.GDT_Float32,
                None)
        expected_message = "inputs cannot be broadcast into a single shape"
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_raster_calculator_array_raster_mismatch(self):
        """PGP.geoprocessing: bad array shape with raster raise error."""
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        _array_to_raster(
            numpy.ones((128, 128), dtype=numpy.int32), -1, base_path)

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        z_arg = numpy.ones((4, 4))
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(base_path, 1), z_arg], lambda a, z: a*z,
                target_path, gdal.GDT_Float32, None)
        expected_message = (
            'Raster size (128, 128) cannot be broadcast '
            'to numpy shape (4')
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        y_arg = numpy.ones((4,))
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(base_path, 1), y_arg], lambda a, y: a*y,
                target_path, gdal.GDT_Float32, None)
        expected_message = (
            'Raster size (128, 128) cannot be broadcast '
            'to numpy shape (4')
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_raster_calculator_bad_raw_args(self):
        """PGP.geoprocessing: tuples that don't match (x, 'raw')."""
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        _array_to_raster(
            numpy.ones((128, 128), dtype=numpy.int32), -1, base_path)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(base_path, 1), ("raw",)], lambda a, z: a*z,
                target_path, gdal.GDT_Float32, None)
        expected_message = 'Expected a sequence of path / integer band tuples'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_raster_calculator_constant_args(self):
        """PGP.geoprocessing: test constant arguments of raster calc."""
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        a_arg = 3
        x_arg = numpy.array(range(2))
        y_arg = numpy.array(range(3)).reshape((3, 1))
        z_arg = numpy.ones((3, 2))
        list_arg = [1, 1, 1, -1]
        pygeoprocessing.raster_calculator(
            [(a_arg, 'raw'), x_arg, y_arg, z_arg], lambda a, x, y, z: a*x*y*z,
            target_path, gdal.GDT_Float32, 0)

        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        expected_result = numpy.array([[0, 0], [0, 3], [0, 6]])
        numpy.testing.assert_array_almost_equal(target_array, expected_result)

        target_path = os.path.join(self.workspace_dir, 'target_a.tif')
        with self.assertRaises(ValueError) as cm:
            # this will return a scalar, when it should return 2d array
            pygeoprocessing.raster_calculator(
                [(a_arg, 'raw')], lambda a: a, target_path,
                gdal.GDT_Float32, None)
        expected_message = (
            "Only (object, 'raw') values have been passed.")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            # this will return a scalar, when it should return 2d array
            pygeoprocessing.raster_calculator(
                [x_arg], lambda x: 0.0, target_path,
                gdal.GDT_Float32, None)
        expected_message = (
            "Expected `local_op` to return a numpy.ndarray")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        target_path = os.path.join(
            self.workspace_dir, 'target_1d_2darray.tif')
        pygeoprocessing.raster_calculator(
            [y_arg], lambda y: y, target_path,
            gdal.GDT_Float32, None)

        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        numpy.testing.assert_array_almost_equal(target_array, y_arg)

        target_path = os.path.join(self.workspace_dir, 'target_1d_only.tif')
        pygeoprocessing.raster_calculator(
            [x_arg], lambda x: x, target_path,
            gdal.GDT_Float32, None)

        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        numpy.testing.assert_array_almost_equal(
            target_array, x_arg.reshape((1, x_arg.size)))

        target_path = os.path.join(self.workspace_dir, 'raw_args.tif')
        pygeoprocessing.raster_calculator(
            [x_arg, (list_arg, 'raw')], lambda x, y_list: x * y_list[3],
            target_path, gdal.GDT_Float32, None)

        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        numpy.testing.assert_array_almost_equal(
            target_array, -x_arg.reshape((1, x_arg.size)))

        target_path = os.path.join(self.workspace_dir, 'raw_numpy_args.tif')
        pygeoprocessing.raster_calculator(
            [x_arg, (numpy.array(list_arg), 'raw')],
            lambda x, y_list: x * y_list[3], target_path, gdal.GDT_Float32,
            None)

        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        numpy.testing.assert_array_almost_equal(
            target_array, -x_arg.reshape((1, x_arg.size)))

    def test_combined_constant_args_raster(self):
        """PGP.geoprocessing: test raster calc with constant args."""
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        nodata = 0
        raster_array = numpy.ones((128, 128), dtype=numpy.int32)
        raster_array[127, 127] = nodata
        _array_to_raster(
            raster_array, nodata, base_path, projection_epsg=4326,
            origin=(0.1, 0), pixel_size=(1, -1))

        target_path = os.path.join(self.workspace_dir, 'target.tif')

        # making a local op that needs a valid mask to ensure that `col_array`
        # is tiled out correctly
        def local_op(scalar, raster_array, col_array):
            valid_mask = raster_array != nodata
            result = numpy.empty_like(raster_array)
            result[:] = nodata
            result[valid_mask] = (
                scalar * raster_array[valid_mask] * col_array[valid_mask])
            return result

        pygeoprocessing.raster_calculator(
            [(10, 'raw'), (base_path, 1), numpy.array(range(128))],
            local_op, target_path, gdal.GDT_Float32, None, largest_block=0)

        result = pygeoprocessing.raster_to_numpy_array(target_path)
        expected_result = (
            10 * numpy.ones((128, 128)) * numpy.array(range(128)))
        # we expect one pixel to have been masked out
        expected_result[127, 127] = nodata
        numpy.testing.assert_allclose(result, expected_result)

    def test_new_raster_from_base_unsigned_byte(self):
        """PGP.geoprocessing: test that signed byte rasters copy over."""
        pixel_array = numpy.ones((128, 128), numpy.byte)
        pixel_array[0, 0] = 255  # 255 ubyte is -1 byte
        nodata_base = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        _array_to_raster(
            pixel_array, nodata_base, base_path,
            creation_options=['PIXELTYPE=SIGNEDBYTE'])

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        # 255 should convert to -1 with signed bytes
        pygeoprocessing.new_raster_from_base(
            base_path, target_path, gdal.GDT_Byte, [0],
            fill_value_list=[255])

        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        # we expect a negative result even though we put in a positive because
        # we know signed bytes will convert
        self.assertEqual(target_array[0, 0], -1)

    def test_new_raster_from_base_nodata_not_set(self):
        """PGP.geoprocessing: test new raster with nodata not set."""
        driver = gdal.GetDriverByName('GTiff')
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        new_raster = driver.Create(base_path, 128, 128, 1, gdal.GDT_Int32)
        del new_raster

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        # 255 should convert to -1 with signed bytes
        pygeoprocessing.new_raster_from_base(
            base_path, target_path, gdal.GDT_Byte, [None],
            fill_value_list=[None],
            raster_driver_creation_tuple=('GTiff', [
                'PIXELTYPE=SIGNEDBYTE',
                ]))

        raster_properties = pygeoprocessing.get_raster_info(target_path)
        self.assertEqual(raster_properties['nodata'], [None])

    def test_create_raster_from_vector_extents(self):
        """PGP.geoprocessing: test creation of raster from vector extents."""
        point_a = shapely.geometry.Point(
            _DEFAULT_ORIGIN[0], _DEFAULT_ORIGIN[1])
        n_pixels_x = 9
        n_pixels_y = 19
        point_b = shapely.geometry.Point(
            _DEFAULT_ORIGIN[0] +
            _DEFAULT_PIXEL_SIZE[0] * n_pixels_x,
            _DEFAULT_ORIGIN[1] +
            _DEFAULT_PIXEL_SIZE[1] * n_pixels_y)
        source_vector_path = os.path.join(self.workspace_dir, 'sample_vector')
        _geometry_to_vector(
            [point_a, point_b], source_vector_path,
            fields={'value': ogr.OFTInteger},
            attribute_list=[{'value': 0}, {'value': 1}])
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        target_nodata = -1
        target_pixel_type = gdal.GDT_Int16
        pygeoprocessing.create_raster_from_vector_extents(
            source_vector_path, target_raster_path, _DEFAULT_PIXEL_SIZE,
            target_pixel_type, target_nodata)

        raster_properties = pygeoprocessing.get_raster_info(
            target_raster_path)
        self.assertEqual(raster_properties['raster_size'][0], n_pixels_x)
        self.assertEqual(raster_properties['raster_size'][1], n_pixels_y)

    def test_create_raster_from_vector_extents_invalid_pixeltype(self):
        """PGP.geoprocessing: raster from vector with bad datatype."""
        point_a = shapely.geometry.Point(
            _DEFAULT_ORIGIN[0], _DEFAULT_ORIGIN[1])
        n_pixels_x = 9
        n_pixels_y = 19
        point_b = shapely.geometry.Point(
            _DEFAULT_ORIGIN[0] +
            _DEFAULT_PIXEL_SIZE[0] * n_pixels_x,
            _DEFAULT_ORIGIN[1] +
            _DEFAULT_PIXEL_SIZE[1] * n_pixels_y)
        source_vector_path = os.path.join(self.workspace_dir, 'sample_vector')
        _geometry_to_vector(
            [point_a, point_b], source_vector_path,
            fields={'value': ogr.OFTInteger},
            attribute_list=[{'value': 0}, {'value': 1}])
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        target_nodata = -1
        target_pixel_type = gdal.GDT_Int16
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.create_raster_from_vector_extents(
                source_vector_path, target_raster_path, _DEFAULT_PIXEL_SIZE,
                target_nodata, target_pixel_type)
            expected_message = (
                'Invalid target type, should be a gdal.GDT_* type')
            actual_message = str(cm.exception)
            self.assertTrue(
                expected_message in actual_message, actual_message)

    def test_create_raster_from_vector_extents_odd_pixel_shapes(self):
        """PGP.geoprocessing: create raster vector ext. w/ odd pixel size."""
        point_a = shapely.geometry.Point(
            _DEFAULT_ORIGIN[0], _DEFAULT_ORIGIN[1])
        pixel_x_size = -10
        pixel_y_size = 20
        n_pixels_x = 9
        n_pixels_y = 19
        point_b = shapely.geometry.Point(
            _DEFAULT_ORIGIN[0] + pixel_x_size * n_pixels_x,
            _DEFAULT_ORIGIN[1] + pixel_y_size * n_pixels_y)
        source_vector_path = os.path.join(self.workspace_dir, 'sample_vector')
        _geometry_to_vector(
            [point_a, point_b], source_vector_path,
            fields={'value': ogr.OFTInteger},
            attribute_list=[{'value': 0}, {'value': 1}])
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        target_pixel_size = [pixel_x_size, pixel_y_size]
        target_nodata = -1
        target_pixel_type = gdal.GDT_Int16
        pygeoprocessing.create_raster_from_vector_extents(
            source_vector_path, target_raster_path, target_pixel_size,
            target_pixel_type, target_nodata)

        raster_properties = pygeoprocessing.get_raster_info(
            target_raster_path)
        self.assertEqual(raster_properties['raster_size'][0], n_pixels_x)
        self.assertEqual(raster_properties['raster_size'][1], n_pixels_y)

    def test_create_raster_from_vector_extents_linestring_no_width(self):
        """PGP.geoprocessing: create raster from v. ext with no geom width."""
        point_a = shapely.geometry.LineString(
            [(_DEFAULT_ORIGIN[0], _DEFAULT_ORIGIN[1]),
             (_DEFAULT_ORIGIN[0], _DEFAULT_ORIGIN[1] + 100)])
        pixel_x_size = -10
        pixel_y_size = 20
        n_pixels_x = 1
        n_pixels_y = 5
        source_vector_path = os.path.join(self.workspace_dir, 'sample_vector')
        _geometry_to_vector(
            [point_a], source_vector_path, fields={'value': ogr.OFTInteger},
            attribute_list=[{'value': 0}])
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        target_pixel_size = [pixel_x_size, pixel_y_size]
        target_nodata = -1
        target_pixel_type = gdal.GDT_Int16
        pygeoprocessing.create_raster_from_vector_extents(
            source_vector_path, target_raster_path, target_pixel_size,
            target_pixel_type, target_nodata)

        raster_properties = pygeoprocessing.get_raster_info(
            target_raster_path)
        self.assertEqual(raster_properties['raster_size'][0], n_pixels_x)
        self.assertEqual(raster_properties['raster_size'][1], n_pixels_y)

    def test_create_raster_from_vector_extents_linestring_no_height(self):
        """PGP.geoprocessing: create raster from v. ext with no geom height."""
        point_a = shapely.geometry.LineString(
            [(_DEFAULT_ORIGIN[0], _DEFAULT_ORIGIN[1]),
             (_DEFAULT_ORIGIN[0] + 100, _DEFAULT_ORIGIN[1])])
        pixel_x_size = -10
        pixel_y_size = 20
        n_pixels_x = 10
        n_pixels_y = 1
        source_vector_path = os.path.join(self.workspace_dir, 'sample_vector')
        _geometry_to_vector(
            [point_a], source_vector_path, fields={'value': ogr.OFTInteger},
            attribute_list=[{'value': 0}])
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        target_pixel_size = [pixel_x_size, pixel_y_size]
        target_nodata = -1
        target_pixel_type = gdal.GDT_Int16
        pygeoprocessing.create_raster_from_vector_extents(
            source_vector_path, target_raster_path, target_pixel_size,
            target_pixel_type, target_nodata)

        raster_properties = pygeoprocessing.get_raster_info(
            target_raster_path)
        self.assertEqual(raster_properties['raster_size'][0], n_pixels_x)
        self.assertEqual(raster_properties['raster_size'][1], n_pixels_y)

    def test_create_raster_from_vector_extents_bad_geometry(self):
        """PGP.geoprocessing: create raster from v. ext. with bad geometry."""
        vector_driver = ogr.GetDriverByName('GeoJSON')
        source_vector_path = os.path.join(self.workspace_dir, 'vector.json')
        source_vector = vector_driver.CreateDataSource(source_vector_path)
        projection = osr.SpatialReference()
        projection.ImportFromEPSG(_DEFAULT_EPSG)
        source_layer = source_vector.CreateLayer('vector', srs=projection)

        layer_defn = source_layer.GetLayerDefn()

        point_a = shapely.geometry.Point(
            _DEFAULT_ORIGIN[0], _DEFAULT_ORIGIN[1])
        mean_pixel_size = 30
        n_pixels_x = 9
        n_pixels_y = 19
        point_b = shapely.geometry.Point(
            _DEFAULT_ORIGIN[0] + _DEFAULT_PIXEL_SIZE[0] * n_pixels_x,
            _DEFAULT_ORIGIN[1] + _DEFAULT_PIXEL_SIZE[1] * n_pixels_y)

        for point in [point_a, point_b]:
            feature = ogr.Feature(layer_defn)
            feature_geometry = ogr.CreateGeometryFromWkb(point.wkb)
            feature.SetGeometry(feature_geometry)
            source_layer.CreateFeature(feature)
        null_feature = ogr.Feature(layer_defn)
        source_layer.CreateFeature(null_feature)
        source_layer.SyncToDisk()
        source_layer = None
        ogr.DataSource.__swig_destroy__(source_vector)
        source_vector = None

        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        target_pixel_size = [mean_pixel_size, -mean_pixel_size]
        target_nodata = -1
        target_pixel_type = gdal.GDT_Int16
        pygeoprocessing.create_raster_from_vector_extents(
            source_vector_path, target_raster_path, target_pixel_size,
            target_pixel_type, target_nodata, fill_value=0)

        raster_properties = pygeoprocessing.get_raster_info(
            target_raster_path)
        self.assertEqual(raster_properties['raster_size'][0], n_pixels_x)
        self.assertEqual(raster_properties['raster_size'][1], n_pixels_y)
        expected_result = numpy.zeros((19, 9))
        result = pygeoprocessing.raster_to_numpy_array(target_raster_path)
        numpy.testing.assert_array_equal(expected_result, result)

    def test_transform_box(self):
        """PGP.geoprocessing: test geotransforming lat/lng box to UTM10N."""
        # Willamette valley in lat/lng
        bounding_box = [-123.587984, 44.415778, -123.397976, 44.725814]
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)  # WGS84 EPSG

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(26910)  # UTM10N EPSG

        result = pygeoprocessing.transform_bounding_box(
            bounding_box, wgs84_srs.ExportToWkt(), target_srs.ExportToWkt())
        # I have confidence this function works by taking the result and
        # plotting it in a GIS polygon, so the expected result below is
        # regression data
        expected_result = [
            453188.671769, 4918131.799327, 468483.727558, 4952661.553935]
        self.assertIs(
            numpy.testing.assert_allclose(
                result, expected_result), None)

        # this test case came up where the y coordinates got flipped
        gibraltar_bb = [6598990.0, 15315600.0, 7152690.0, 16058800.0]
        utm_30n_srs = osr.SpatialReference()
        utm_30n_srs.ImportFromEPSG(32630)
        gibraltar_bb_wgs84 = pygeoprocessing.transform_bounding_box(
            gibraltar_bb, utm_30n_srs.ExportToWkt(), wgs84_srs.ExportToWkt())
        self.assertTrue(
            gibraltar_bb_wgs84[0] < gibraltar_bb_wgs84[2] and
            gibraltar_bb_wgs84[1] < gibraltar_bb_wgs84[3],
            'format should be [xmin, ymin, xmax, ymax]: '
            '%s' % gibraltar_bb_wgs84)

    def test_transform_box_latlon_to_utm(self):
        """PGP.geoprocessing: test geotransforming lat/lon box to UTM19N."""
        # lat/lon bounds for Coast of New England from shapefile created in
        # QGIS 3.10.2 running GDAL 3.0.3
        bounding_box = [-72.3638439, 40.447948, -68.0041948, 43.1441579]

        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)  # WGS84 EPSG

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(32619)  # UTM19N EPSG

        result = pygeoprocessing.transform_bounding_box(
            bounding_box, wgs84_srs.ExportToWkt(), target_srs.ExportToWkt())
        # Expected result taken from QGIS UTM19N - WGS84 reference and
        # converting extents from above bounding box (extents) of shapefile
        expected_result = [
            214722.122449, 4477484.382162, 584444.275934, 4782318.029707]

        self.assertIs(
            numpy.testing.assert_allclose(
                result, expected_result), None)

    def test_transform_box_utm_to_latlon(self):
        """PGP.geoprocessing: test geotransforming UTM19N box to lat/lon."""
        # UTM19N bounds for Coast of New England
        bounding_box = [
            214722.122449, 4477484.382162, 584444.275934, 4782318.029707]

        utm19n_srs = osr.SpatialReference()
        utm19n_srs.ImportFromEPSG(32619)  # UTM19N EPSG

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)  # WGS84 EPSG

        result = pygeoprocessing.transform_bounding_box(
            bounding_box, utm19n_srs.ExportToWkt(), target_srs.ExportToWkt())
        # Expected result taken from QGIS UTM19N - WGS84 reference and
        # converting extents from above bounding box (extents) of shapefile
        expected_result = [-72.507803,  40.399122, -67.960794,  43.193562]

        self.assertIs(
            numpy.testing.assert_allclose(
                result, expected_result), None)

    def test_transform_box_latlon_to_latlon(self):
        """PGP.geoprocessing: test geotransforming lat/lon box to lat/lon."""
        # lat/lon bounds for Coast of New England from shapefile created in
        # QGIS 3.10.2 running GDAL 3.0.3
        bounding_box = [-72.3638439, 40.447948, -68.0041948, 43.1441579]

        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)  # WGS84 EPSG

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)  # WGS84 EPSG

        result = pygeoprocessing.transform_bounding_box(
            bounding_box, wgs84_srs.ExportToWkt(), target_srs.ExportToWkt())
        # Expected result should be identical
        expected_result = [-72.3638439, 40.447948, -68.0041948, 43.1441579]

        self.assertIs(
            numpy.testing.assert_allclose(
                result, expected_result), None)

    def test_transform_box_utm_to_utm(self):
        """PGP.geoprocessing: test geotransforming utm box to utm."""
        # UTM19N bounds for Coast of New England
        bounding_box = [
            214722.122449, 4477484.382162, 584444.275934, 4782318.029707]

        utm19n_srs = osr.SpatialReference()
        utm19n_srs.ImportFromEPSG(32619)  # UTM19N EPSG

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(32619)  # UTM19N EPSG

        result = pygeoprocessing.transform_bounding_box(
            bounding_box, utm19n_srs.ExportToWkt(), target_srs.ExportToWkt())
        # Expected result should be identical
        expected_result = [
            214722.122449, 4477484.382162, 584444.275934, 4782318.029707]

        self.assertIs(
            numpy.testing.assert_allclose(
                result, expected_result), None)

    def test_iterblocks(self):
        """PGP.geoprocessing: test iterblocks."""
        n_pixels = 100
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        target_nodata = None
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path, creation_options=[
                'TILED=YES',
                'BLOCKXSIZE=64',
                'BLOCKYSIZE=64'])

        total = 0
        for _, block in pygeoprocessing.iterblocks(
                (raster_path, 1), largest_block=0):
            total += numpy.sum(block)
        self.assertAlmostEqual(total, test_value * n_pixels**2)

    def test_iterblocks_bad_raster_band(self):
        """PGP.geoprocessing: test iterblocks."""
        n_pixels = 100
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        target_nodata = None
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path, creation_options=[
                'TILED=YES',
                'BLOCKXSIZE=64',
                'BLOCKYSIZE=64'])

        total = 0
        with self.assertRaises(ValueError) as cm:
            for _, block in pygeoprocessing.iterblocks(
                    raster_path, largest_block=0):
                total += numpy.sum(block)
        expected_message = (
            "`raster_path_band` not formatted as expected.")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_iterblocks_unsigned_byte(self):
        """PGP.geoprocessing: test iterblocks with unsigned byte."""
        n_pixels = 100
        pixel_matrix = numpy.empty((n_pixels, n_pixels), numpy.uint8)
        test_value = 255
        pixel_matrix[:] = test_value
        target_nodata = None
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path, creation_options=[
                'TILED=YES',
                'BLOCKXSIZE=64',
                'BLOCKYSIZE=64'])

        total = 0
        for _, block in pygeoprocessing.iterblocks(
                (raster_path, 1), largest_block=0):
            total += numpy.sum(block)
        self.assertEqual(total, test_value * n_pixels**2)

    def test_convolve_2d_single_thread(self):
        """PGP.geoprocessing: test convolve 2d (single thread)."""
        n_pixels = 100
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        target_nodata = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        _array_to_raster(signal_array, target_nodata, signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((3, 3), numpy.float32)
        _array_to_raster(kernel_array, target_nodata, kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            n_threads=1, ignore_nodata_and_edges=False)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)

        # calculate expected result by adding up all squares, subtracting off
        # the sides and realizing diagonals got subtracted twice
        expected_result = test_value * (
            n_pixels ** 2 * 9 - n_pixels * 4 * 3 + 4)
        numpy.testing.assert_allclose(numpy.sum(target_array), expected_result)

    def test_convolve_2d_multiprocess(self):
        """PGP.geoprocessing: test convolve 2d (multiprocess)."""
        n_pixels = 100
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        target_nodata = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        _array_to_raster(signal_array, target_nodata, signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((3, 3), numpy.float32)
        _array_to_raster(kernel_array, target_nodata, kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            n_threads=3)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)

        # calculate expected result by adding up all squares, subtracting off
        # the sides and realizing diagonals got subtracted twice
        expected_result = test_value * (
            n_pixels ** 2 * 9 - n_pixels * 4 * 3 + 4)
        numpy.testing.assert_allclose(numpy.sum(target_array), expected_result)

    def test_convolve_2d_normalize_ignore_nodata(self):
        """PGP.geoprocessing: test convolve 2d w/ normalize and ignore."""
        n_pixels = 100
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        target_nodata = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        _array_to_raster(signal_array, target_nodata, signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((3, 3), numpy.float32)
        _array_to_raster(kernel_array, target_nodata, kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            mask_nodata=False, ignore_nodata_and_edges=True,
            normalize_kernel=True)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        expected_result = test_value * n_pixels ** 2
        numpy.testing.assert_allclose(numpy.sum(target_array),
                                      expected_result)

    def test_convolve_2d_ignore_nodata(self):
        """PGP.geoprocessing: test convolve 2d w/ normalize and ignore."""
        n_pixels = 3
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        target_nodata = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        _array_to_raster(signal_array, target_nodata, signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((3, 3), numpy.float32)
        _array_to_raster(kernel_array, target_nodata, kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            ignore_nodata_and_edges=True)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)

        # calculate by working on some graph paper
        expected_result = 9*9*.5
        self.assertAlmostEqual(numpy.sum(target_array), expected_result,
                               places=5)

    def test_convolve_2d_normalize(self):
        """PGP.geoprocessing: test convolve 2d w/ normalize."""
        n_pixels = 3
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        target_nodata = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        _array_to_raster(signal_array, target_nodata, signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((3, 3), numpy.float32)
        _array_to_raster(kernel_array, target_nodata, kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            normalize_kernel=True)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)

        # I calculated this by manually doing a grid on graph paper
        expected_result = .5 + 4 * 5./9.
        self.assertAlmostEqual(numpy.sum(target_array), expected_result)

    def test_convolve_2d_missing_nodata(self):
        """PGP.geoprocessing: test convolve2d if target type but no nodata."""
        n_pixels = 100
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        target_nodata = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        _array_to_raster(signal_array, target_nodata, signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((3, 3), numpy.float32)
        _array_to_raster(kernel_array, target_nodata, kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.convolve_2d(
                (signal_path, 1), (kernel_path, 1), target_path,
                target_datatype=gdal.GDT_Int32)
        expected_message = (
            "`target_datatype` is set, but `target_nodata` is None. ")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_convolve_2d_reverse(self):
        """PGP.geoprocessing: test convolve 2d reversed."""
        n_pixels = 3
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        target_nodata = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        _array_to_raster(signal_array, target_nodata, signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((100, 100), numpy.float32)
        _array_to_raster(kernel_array, target_nodata, kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)

        # calculate expected result by adding up all squares, subtracting off
        # the sides and realizing diagonals got subtracted twice
        expected_result = test_value * (n_pixels ** 4)
        self.assertAlmostEqual(numpy.sum(target_array), expected_result,
                               places=4)

    def test_convolve_2d_large(self):
        """PGP.geoprocessing: test convolve 2d with large kernel & signal."""
        n_pixels = 100
        n_kernel_pixels = 1750
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        target_nodata = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        _array_to_raster(signal_array, target_nodata, signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.zeros(
            (n_kernel_pixels, n_kernel_pixels), numpy.float32)
        kernel_array[int(n_kernel_pixels/2), int(n_kernel_pixels/2)] = 1
        _array_to_raster(kernel_array, target_nodata, kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)

        # calculate expected result by adding up all squares, subtracting off
        # the sides and realizing diagonals got subtracted twice
        expected_result = test_value * (n_pixels ** 2)
        numpy.testing.assert_allclose(numpy.sum(target_array),
                                      expected_result, rtol=1e-6)

    def test_convolve_2d_numerical_zero(self):
        """PGP.geoprocessing: test convolve 2d for numerical 0.0 set to 0.0."""
        # set tiny signal with one pixel on so we get lots of numerical noise
        n_pixels = 100
        n_kernel_pixels = 100
        signal_array = numpy.zeros((n_pixels, n_pixels), numpy.float32)
        signal_array[n_pixels//2, int(0.05*n_pixels)] = 1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')

        _array_to_raster(
            signal_array, None, signal_path, projection_epsg=4326,
            origin=(1, 1), pixel_size=(1, -1))

        # make a linear decay kernel
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_x, kernel_y = numpy.meshgrid(
            range(n_kernel_pixels), range(n_kernel_pixels))
        kernel_radius = n_kernel_pixels//2
        dist_array = 1.0 - numpy.sqrt(
            (kernel_x-kernel_radius)**2 +
            (kernel_y-kernel_radius)**2)/kernel_radius
        dist_array[dist_array < 0] = 0
        kernel_array = dist_array / numpy.sum(dist_array)

        _array_to_raster(
            kernel_array, None, kernel_path, projection_epsg=4326,
            origin=(1, 1), pixel_size=(1, -1))

        # ensure non-tolerance has some negative noise
        raw_result_path = os.path.join(self.workspace_dir, 'raw_result.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), raw_result_path,
            set_tol_to_zero=None)
        raw_array = pygeoprocessing.raster_to_numpy_array(raw_result_path)
        self.assertTrue(
            numpy.count_nonzero(raw_array < 0) != 0.0,
            msg='we expect numerical noise in this result')

        # ensure tolerant clamped has no negative noise
        tol_result_path = os.path.join(self.workspace_dir, 'tol_result.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), tol_result_path)
        tol_array = pygeoprocessing.raster_to_numpy_array(tol_result_path)
        self.assertTrue(
            numpy.count_nonzero(tol_array < 0) == 0.0,
            msg='we expect no noise in this result')

    def test_convolve_2d_ignore_undefined_nodata(self):
        """PGP.geoprocessing: test convolve 2d ignore nodata when None."""
        # set tiny signal with one pixel on so we get lots of numerical noise
        n_pixels = 100
        n_kernel_pixels = 100
        signal_array = numpy.zeros((n_pixels, n_pixels), numpy.float32)
        signal_array[n_pixels//2, int(0.05*n_pixels)] = 1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')

        _array_to_raster(
            signal_array, -1, signal_path, projection_epsg=4326,
            origin=(1, 1), pixel_size=(1, -1))

        signal_nodata_none_path = os.path.join(
            self.workspace_dir, 'signal_none.tif')
        _array_to_raster(
            signal_array, None, signal_nodata_none_path, projection_epsg=4326,
            origin=(1, 1), pixel_size=(1, -1))

        # make a linear decay kernel
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_x, kernel_y = numpy.meshgrid(
            range(n_kernel_pixels), range(n_kernel_pixels))
        kernel_radius = n_kernel_pixels//2
        dist_array = 1.0 - numpy.sqrt(
            (kernel_x-kernel_radius)**2 +
            (kernel_y-kernel_radius)**2)/kernel_radius
        dist_array[dist_array < 0] = 0
        kernel_array = dist_array / numpy.sum(dist_array)

        _array_to_raster(
            kernel_array, None, kernel_path, projection_epsg=4326,
            origin=(1, 1), pixel_size=(1, -1))

        nodata_result_path = os.path.join(
            self.workspace_dir, 'nodata_result.tif')
        none_result_path = os.path.join(self.workspace_dir, 'none_result.tif')

        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), nodata_result_path,
            ignore_nodata_and_edges=True)
        signal_nodata_array = pygeoprocessing.raster_to_numpy_array(
            nodata_result_path)

        pygeoprocessing.convolve_2d(
            (signal_nodata_none_path, 1), (kernel_path, 1), none_result_path,
            ignore_nodata_and_edges=True)
        signal_nodata_none_array = pygeoprocessing.raster_to_numpy_array(
            none_result_path)

        self.assertTrue(
            numpy.isclose(signal_nodata_array, signal_nodata_none_array).all(),
            'signal with nodata should be the same as signal with none')

    def test_calculate_slope(self):
        """PGP.geoprocessing: test calculate slope."""
        n_pixels = 9
        dem_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        dem_array[:] = numpy.arange((n_pixels))
        nodata_value = -1
        # make a nodata hole in the middle to test boundary cases
        dem_array[int(n_pixels/2), int(n_pixels/2)] = nodata_value
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        target_slope_path = os.path.join(self.workspace_dir, 'slope.tif')
        _array_to_raster(
            dem_array, nodata_value, dem_path, projection_epsg=4326,
            pixel_size=(1, -1), origin=(0.1, 0))

        pygeoprocessing.calculate_slope((dem_path, 1), target_slope_path)
        slope_raster = gdal.OpenEx(target_slope_path, gdal.OF_RASTER)
        slope_band = slope_raster.GetRasterBand(1)
        target_nodata = slope_band.GetNoDataValue()
        count = 0
        expected_slope = 100.0
        for _, band_data in pygeoprocessing.iterblocks(
                (target_slope_path, 1)):
            block = band_data.astype(numpy.float32)
            bad_mask = (
                ~numpy.isclose(block, target_nodata) &
                (block != expected_slope))
            if numpy.any(bad_mask):
                self.fail(
                    "Unexpected value in slope raster: %s" % block[bad_mask])
            count += numpy.count_nonzero(block == expected_slope)
        # all slopes should be 1 except center pixel
        self.assertEqual(count, n_pixels**2 - 1)

    def test_calculate_slope_undefined_nodata(self):
        """PGP.geoprocessing: test calculate slope with no nodata."""
        n_pixels = 9
        dem_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        target_slope_path = os.path.join(self.workspace_dir, 'slope.tif')
        _array_to_raster(
            dem_array, None, dem_path, projection_epsg=4326,
            pixel_size=(1, -1), origin=(0.1, 0))

        pygeoprocessing.calculate_slope((dem_path, 1), target_slope_path)

        actual_slope = pygeoprocessing.raster_to_numpy_array(target_slope_path)
        expected_slope = numpy.zeros((n_pixels, n_pixels), numpy.float32)
        numpy.testing.assert_almost_equal(expected_slope, actual_slope)

    def test_rasterize(self):
        """PGP.geoprocessing: test rasterize."""
        n_pixels = 3
        target_raster_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        target_raster_array[:] = test_value
        target_nodata = -1
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        _array_to_raster(
            target_raster_array, target_nodata, target_raster_path)

        pixel_size = 30.0
        origin = (444720, 3751320)
        polygon = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels, origin[1]),
            (origin[0], origin[1])])
        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.json')
        _geometry_to_vector(
            [polygon], base_vector_path, fields={'id': ogr.OFTInteger},
            attribute_list=[{'id': 5}], vector_format='GeoJSON')

        pygeoprocessing.rasterize(
            base_vector_path, target_raster_path, [test_value], None,
            layer_id=0)
        result = pygeoprocessing.raster_to_numpy_array(target_raster_path)
        self.assertTrue((result == test_value).all())

        pygeoprocessing.rasterize(
            base_vector_path, target_raster_path, None,
            ["ATTRIBUTE=id"], layer_id=0)
        result = pygeoprocessing.raster_to_numpy_array(
            target_raster_path)
        self.assertTrue((result == 5).all())

    def test_rasterize_error(self):
        """PGP.geoprocessing: test rasterize when error encountered."""
        n_pixels = 3
        target_raster_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        target_raster_array[:] = test_value
        target_nodata = -1
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        _array_to_raster(
            target_raster_array, target_nodata, target_raster_path)

        pixel_size = 30.0
        origin = (444720, 3751320)
        polygon = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels, origin[1]),
            (origin[0], origin[1])])
        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.json')

        _geometry_to_vector(
            [polygon], base_vector_path, fields={'id': ogr.OFTInteger},
            attribute_list=[{'id': 5}], vector_format='GeoJSON')

        with self.assertRaises(RuntimeError) as cm:
            # Patching the function that makes a logger callback so that
            # it will raise an exception (ZeroDivisionError in this case,
            # but any exception should do).
            with unittest.mock.patch(
                    'pygeoprocessing.geoprocessing._make_logger_callback',
                    return_value=lambda x, y, z: 1/0.):
                pygeoprocessing.rasterize(
                    base_vector_path, target_raster_path, [test_value], None,
                    layer_id=0)

        self.assertTrue('nonzero exit code' in str(cm.exception))

    def test_rasterize_missing_file(self):
        """PGP.geoprocessing: test rasterize with no target raster."""
        n_pixels = 3
        target_raster_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        target_raster_array[:] = test_value
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')

        # intentionally not making the raster on disk
        pixel_size = 30.0
        origin = (444720, 3751320)
        polygon = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels, origin[1]),
            (origin[0], origin[1])])
        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.json')
        _geometry_to_vector(
            [polygon], base_vector_path, fields={'id': ogr.OFTInteger},
            attribute_list=[{'id': 5}], vector_format='GeoJSON')

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.rasterize(
                base_vector_path, target_raster_path, [test_value], None,
                layer_id=0)
        expected_message = (
            "%s doesn't exist, but needed to rasterize." % target_raster_path)
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_rasterize_error_handling(self):
        """PGP.geoprocessing: test rasterize error handling."""
        n_pixels = 3
        target_raster_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        target_raster_array[:] = test_value
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        _array_to_raster(target_raster_array, -1, target_raster_path)

        # intentionally not making the raster on disk
        pixel_size = 30.0
        origin = (444720, 3751320)
        polygon = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels, origin[1]),
            (origin[0], origin[1])])
        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.json')
        _geometry_to_vector(
            [polygon], base_vector_path, fields={'id': ogr.OFTInteger},
            attribute_list=[{'id': 5}], vector_format='GeoJSON')

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.rasterize(
                base_vector_path, target_raster_path, None, None,
                layer_id=0)
        expected_message = (
            "Neither `burn_values` nor `option_list` is set")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.rasterize(
                base_vector_path, target_raster_path, 1, None,
                layer_id=0)
        expected_message = "`burn_values` is not a list/tuple"
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.rasterize(
                base_vector_path, target_raster_path, None, "ATTRIBUTE=id",
                layer_id=0)
        expected_message = "`option_list` is not a list/tuple"
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_distance_transform_edt(self):
        """PGP.geoprocessing: test distance transform EDT."""
        n_pixels = 1000
        target_nodata = 0
        base_raster_array = numpy.zeros(
            (n_pixels, n_pixels), dtype=numpy.int32)
        base_raster_array[:, n_pixels//2:] = target_nodata
        base_raster_array[int(n_pixels/2), int(n_pixels/2)] = 1
        base_raster_array[0, 0] = 1
        base_raster_array[0, n_pixels-1] = 1
        base_raster_array[3, 4] = 1
        base_raster_array[3, 5] = 1
        base_raster_array[3, 6] = 1
        base_raster_array[4, 4] = 1
        base_raster_array[4, 5] = 1
        base_raster_array[4, 6] = 1
        base_raster_array[5, 4] = 1
        base_raster_array[5, 5] = 1
        base_raster_array[5, 6] = 1
        base_raster_array[n_pixels-1, 0] = 1
        base_raster_array[n_pixels-1, n_pixels-1] = 1
        base_raster_array[int(n_pixels/2), int(n_pixels/2)] = 1
        base_raster_array[int(n_pixels/2), int(n_pixels/4)] = 1
        base_raster_array[int(n_pixels/2), int((3*n_pixels)/4)] = 1
        base_raster_path = os.path.join(self.workspace_dir, 'base_raster.tif')
        _array_to_raster(base_raster_array, target_nodata, base_raster_path)

        target_distance_raster_path = os.path.join(
            self.workspace_dir, 'target_distance.tif')

        for sampling_distance in [(200.0, 1.5), (1.5, 200.0)]:
            pygeoprocessing.distance_transform_edt(
                (base_raster_path, 1), target_distance_raster_path,
                sampling_distance=sampling_distance,
                working_dir=self.workspace_dir)
            target_array = pygeoprocessing.raster_to_numpy_array(
                target_distance_raster_path)
            expected_result = scipy.ndimage.morphology.distance_transform_edt(
                1 - (base_raster_array == 1), sampling=(
                    sampling_distance[1], sampling_distance[0]))
            numpy.testing.assert_array_almost_equal(
                target_array, expected_result, decimal=2)

        base_raster_path = os.path.join(
            self.workspace_dir, 'undefined_nodata_base_raster.tif')
        _array_to_raster(base_raster_array, None, base_raster_path)
        pygeoprocessing.distance_transform_edt(
            (base_raster_path, 1), target_distance_raster_path,
            sampling_distance=sampling_distance,
            working_dir=self.workspace_dir)
        target_array = pygeoprocessing.raster_to_numpy_array(
            target_distance_raster_path)
        numpy.testing.assert_array_almost_equal(
            target_array, expected_result, decimal=2)

    def test_distance_transform_edt_small_sample_distance(self):
        """PGP.geoprocessing: test distance transform w/ small sample dist."""
        n_pixels = 10
        target_nodata = None
        base_raster_array = numpy.zeros(
            (n_pixels, n_pixels), dtype=numpy.int32)
        base_raster_array[n_pixels//2:, :] = 1
        base_raster_path = os.path.join(self.workspace_dir, 'base_raster.tif')
        _array_to_raster(base_raster_array, target_nodata, base_raster_path)

        target_distance_raster_path = os.path.join(
            self.workspace_dir, 'target_distance.tif')

        sampling_distance = (0.1, 0.1)
        pygeoprocessing.distance_transform_edt(
            (base_raster_path, 1), target_distance_raster_path,
            sampling_distance=sampling_distance,
            working_dir=self.workspace_dir)
        target_array = pygeoprocessing.raster_to_numpy_array(
            target_distance_raster_path)
        expected_result = scipy.ndimage.morphology.distance_transform_edt(
            1 - (base_raster_array == 1), sampling=(
                sampling_distance[1], sampling_distance[0]))
        numpy.testing.assert_array_almost_equal(
            target_array, expected_result, decimal=2)

    def test_distance_transform_edt_bad_data(self):
        """PGP.geoprocessing: test distance transform EDT with bad values."""
        n_pixels = 10
        base_raster_array = numpy.zeros(
            (n_pixels, n_pixels), dtype=numpy.int32)
        base_raster_array[int(n_pixels/2), int(n_pixels/2)] = 1
        base_raster_array[0, 0] = 1
        base_raster_array[0, n_pixels-1] = 1
        base_raster_array[3, 4] = 1
        base_raster_array[3, 5] = 1
        base_raster_array[3, 6] = 1
        base_raster_array[4, 4] = 1
        base_raster_array[4, 5] = 1
        base_raster_array[4, 6] = 1
        base_raster_array[5, 4] = 1
        base_raster_array[5, 5] = 1
        base_raster_array[5, 6] = 1
        base_raster_array[n_pixels-1, 0] = 1
        base_raster_array[n_pixels-1, n_pixels-1] = 1
        base_raster_array[int(n_pixels/2), int(n_pixels/2)] = 1
        base_raster_array[int(n_pixels/2), int(n_pixels/4)] = 1
        base_raster_array[int(n_pixels/2), int((3*n_pixels)/4)] = 1
        base_raster_path = os.path.join(self.workspace_dir, 'base_raster.tif')
        _array_to_raster(base_raster_array, None, base_raster_path)

        target_distance_raster_path = os.path.join(
            self.workspace_dir, 'target_distance.tif')

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.distance_transform_edt(
                (base_raster_path, 1), target_distance_raster_path,
                working_dir=self.workspace_dir,
                sampling_distance=1.0)
        expected_message = '`sampling_distance` should be a tuple/list'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.distance_transform_edt(
                (base_raster_path, 1), target_distance_raster_path,
                working_dir=self.workspace_dir,
                sampling_distance=(1.0, -1.0))
        expected_message = 'Sample distances must be > 0.0'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_next_regular(self):
        """PGP.geoprocessing: test next regular number generator."""
        # just test the first few numbers in the A051037 series
        regular_ints = [
            1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30,
            32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100,
            108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192, 200, 216,
            225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 384,
            400, 405]

        next_int = 0
        for regular_int in regular_ints:
            next_int = pygeoprocessing.geoprocessing._next_regular(next_int+1)
            self.assertEqual(next_int, regular_int)

    def test_merge_rasters(self):
        """PGP.geoprocessing: test merge_rasters."""
        driver = gdal.GetDriverByName('GTiff')

        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)  # WGS84 EPSG

        # the following creates a checkerboard of upper left square raster
        # defined, lower right, and equal sized nodata chunks on the other
        # blocks.

        raster_a_path = os.path.join(self.workspace_dir, 'raster_a.tif')
        # everything flows to the right
        raster_a_array = numpy.zeros((128, 128), dtype=numpy.int32)
        raster_a_array[:] = 10
        raster_a = driver.Create(
            raster_a_path, raster_a_array.shape[1], raster_a_array.shape[0],
            2, gdal.GDT_Int32)
        raster_a_geotransform = [0.1, 1., 0., 0., 0., -1.]
        raster_a.SetGeoTransform(raster_a_geotransform)
        raster_a.SetProjection(wgs84_ref.ExportToWkt())
        band = raster_a.GetRasterBand(1)
        band.WriteArray(raster_a_array)
        band.FlushCache()
        band = None
        raster_a = None

        raster_b_path = os.path.join(self.workspace_dir, 'raster_b.tif')
        raster_b_array = numpy.zeros((128, 128), dtype=numpy.int32)
        raster_b_array[:] = 20
        raster_b = driver.Create(
            raster_b_path, raster_b_array.shape[1], raster_b_array.shape[0],
            2, gdal.GDT_Int32)
        raster_b.SetProjection(wgs84_ref.ExportToWkt())
        raster_b_geotransform = [128.1, 1, 0, -128, 0, -1]
        raster_b.SetGeoTransform(raster_b_geotransform)
        band = raster_b.GetRasterBand(1)
        band.WriteArray(raster_b_array)
        band.FlushCache()
        raster_b = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        pygeoprocessing.merge_rasters(
            [raster_a_path, raster_b_path], target_path)

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        self.assertEqual(target_band.GetNoDataValue(), None)
        target_array = target_band.ReadAsArray()
        target_band = None
        expected_array = numpy.zeros((256, 256))
        expected_array[0:128, 0:128] = 10
        expected_array[128:, 128:] = 20
        numpy.testing.assert_almost_equal(target_array, expected_array)

        target_band = target_raster.GetRasterBand(2)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        expected_array = numpy.zeros((256, 256))
        numpy.testing.assert_almost_equal(target_array, expected_array)

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        pygeoprocessing.merge_rasters(
            [raster_a_path, raster_b_path], target_path,
            bounding_box=[4, -6, 6, -4])

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        self.assertEqual(target_band.GetNoDataValue(), None)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        expected_array = numpy.empty((2, 2))
        expected_array[:] = 10
        numpy.testing.assert_almost_equal(target_array, expected_array)

    def test_merge_rasters_target_nodata(self):
        """PGP.geoprocessing: test merge_rasters with defined nodata."""
        driver = gdal.GetDriverByName('GTiff')

        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)  # WGS84 EPSG

        # the following creates a checkerboard of upper left square raster
        # defined, lower right, and equal sized nodata chunks on the other
        # blocks.

        raster_a_path = os.path.join(self.workspace_dir, 'raster_a.tif')
        # everything flows to the right
        raster_a_array = numpy.zeros((11, 11), dtype=numpy.int32)
        raster_a_array[:] = 10
        raster_a = driver.Create(
            raster_a_path, raster_a_array.shape[1], raster_a_array.shape[0],
            2, gdal.GDT_Int32)
        raster_a_geotransform = [0.1, 1., 0., 0., 0., -1.]
        raster_a.SetGeoTransform(raster_a_geotransform)
        raster_a.SetProjection(wgs84_ref.ExportToWkt())
        band = raster_a.GetRasterBand(1)
        band.WriteArray(raster_a_array)
        band.FlushCache()
        band = None
        raster_a = None

        raster_b_path = os.path.join(self.workspace_dir, 'raster_b.tif')
        raster_b_array = numpy.zeros((11, 11), dtype=numpy.int32)
        raster_b_array[:] = 20
        raster_b = driver.Create(
            raster_b_path, raster_b_array.shape[1], raster_b_array.shape[0],
            2, gdal.GDT_Int32)
        raster_b.SetProjection(wgs84_ref.ExportToWkt())
        raster_b_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_b.SetGeoTransform(raster_b_geotransform)
        band = raster_b.GetRasterBand(1)
        band.WriteArray(raster_b_array)
        band.FlushCache()
        band = None
        raster_b = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        pygeoprocessing.merge_rasters(
            [raster_a_path, raster_b_path], target_path, target_nodata=0)

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        nodata_value = target_raster.GetRasterBand(2).GetNoDataValue()
        target_band = None
        target_raster = None
        expected_array = numpy.zeros((22, 22))
        expected_array[0:11, 0:11] = 10
        expected_array[11:, 11:] = 20

        numpy.testing.assert_almost_equal(target_array, expected_array)
        self.assertEqual(nodata_value, 0)

    def test_merge_rasters_exception_cover(self):
        """PGP.geoprocessing: test merge_rasters with bad data."""
        driver = gdal.GetDriverByName('GTiff')

        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)  # WGS84 EPSG

        # the following creates a checkerboard of upper left square raster
        # defined, lower right, and equal sized nodata chunks on the other
        # blocks.

        raster_a_path = os.path.join(self.workspace_dir, 'raster_a.tif')
        # everything flows to the right
        raster_a_array = numpy.zeros((11, 11), dtype=numpy.int32)
        raster_a_array[:] = 10
        raster_a = driver.Create(
            raster_a_path, raster_a_array.shape[1], raster_a_array.shape[0],
            2, gdal.GDT_Int32)
        raster_a_geotransform = [0.1, 1., 0., 0., 0., -1.]
        raster_a.SetGeoTransform(raster_a_geotransform)
        raster_a.SetProjection(wgs84_ref.ExportToWkt())
        band = raster_a.GetRasterBand(1)
        band.WriteArray(raster_a_array)
        band.FlushCache()
        band = None
        raster_a = None

        raster_b_path = os.path.join(self.workspace_dir, 'raster_b.tif')
        raster_b_array = numpy.zeros((11, 11), dtype=numpy.int32)
        raster_b_array[:] = 20
        raster_b = driver.Create(
            raster_b_path, raster_b_array.shape[1], raster_b_array.shape[0],
            2, gdal.GDT_Int32)
        raster_b.SetProjection(wgs84_ref.ExportToWkt())
        raster_b_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_b.SetGeoTransform(raster_b_geotransform)
        band = raster_b.GetRasterBand(1)
        band.SetNoDataValue(-1.0)
        band.WriteArray(raster_b_array)
        band.FlushCache()
        band = None
        raster_b = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.merge_rasters(
                [raster_a_path, raster_b_path], target_path)
        expected_message = 'Nodata per raster are not the same'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        raster_c_path = os.path.join(self.workspace_dir, 'raster_c.tif')
        raster_c_array = numpy.zeros((11, 11), dtype=numpy.int32)
        raster_c_array[:] = 20
        raster_c = driver.Create(
            raster_c_path, raster_c_array.shape[1], raster_c_array.shape[0],
            1, gdal.GDT_Int32)
        raster_c.SetProjection(wgs84_ref.ExportToWkt())
        raster_c_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_c.SetGeoTransform(raster_c_geotransform)
        band = raster_c.GetRasterBand(1)
        band.SetNoDataValue(-1.0)
        band.WriteArray(raster_c_array)
        band.FlushCache()
        band = None
        raster_c = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.merge_rasters(
                [raster_a_path, raster_c_path], target_path)
        expected_message = 'Number of bands per raster are not the same.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        raster_d_path = os.path.join(self.workspace_dir, 'raster_d.tif')
        raster_d_array = numpy.zeros((11, 11), dtype=numpy.float32)
        raster_d_array[:] = 20
        raster_d = driver.Create(
            raster_d_path, raster_d_array.shape[1], raster_d_array.shape[0],
            2, gdal.GDT_Float32)
        raster_d.SetProjection(wgs84_ref.ExportToWkt())
        raster_d_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_d.SetGeoTransform(raster_d_geotransform)
        band = raster_d.GetRasterBand(1)
        band.SetNoDataValue(-1.0)
        band.WriteArray(raster_d_array)
        band.FlushCache()
        band = None
        raster_d = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.merge_rasters(
                [raster_a_path, raster_d_path], target_path)
        expected_message = 'Rasters have different datatypes.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        raster_e_path = os.path.join(self.workspace_dir, 'raster_e.tif')
        raster_e_array = numpy.zeros((11, 11), dtype=numpy.int32)
        raster_e_array[:] = 20
        raster_e = driver.Create(
            raster_e_path, raster_e_array.shape[1], raster_e_array.shape[0],
            2, gdal.GDT_Int32)
        utm10_ref = osr.SpatialReference()
        utm10_ref.ImportFromEPSG(26910)
        raster_e.SetProjection(utm10_ref.ExportToWkt())
        raster_e_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_e.SetGeoTransform(raster_e_geotransform)
        band = raster_e.GetRasterBand(1)
        band.WriteArray(raster_e_array)
        band.FlushCache()
        band = None
        raster_e = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.merge_rasters(
                [raster_a_path, raster_e_path], target_path)
        expected_message = 'Projections are not identical.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        raster_f_path = os.path.join(self.workspace_dir, 'raster_f.tif')
        raster_f_array = numpy.zeros((11, 11), dtype=numpy.uint8)
        raster_f_array[:] = 20
        raster_f = driver.Create(
            raster_f_path, raster_f_array.shape[1], raster_f_array.shape[0],
            2, gdal.GDT_Byte)
        utm10_ref = osr.SpatialReference()
        utm10_ref.ImportFromEPSG(26910)
        raster_f.SetProjection(utm10_ref.ExportToWkt())
        raster_f_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_f.SetGeoTransform(raster_f_geotransform)
        band = raster_f.GetRasterBand(1)
        band.WriteArray(raster_f_array)
        band.FlushCache()
        band = None
        raster_f = None

        raster_g_path = os.path.join(self.workspace_dir, 'raster_g.tif')
        raster_g_array = numpy.zeros((11, 11), dtype=numpy.int8)
        raster_g_array[:] = 20
        raster_g = driver.Create(
            raster_g_path, raster_g_array.shape[1], raster_g_array.shape[0],
            2, gdal.GDT_Byte, options=['PIXELTYPE=SIGNEDBYTE'])
        utm10_ref = osr.SpatialReference()
        utm10_ref.ImportFromEPSG(26910)
        raster_g.SetProjection(utm10_ref.ExportToWkt())
        raster_g_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_g.SetGeoTransform(raster_g_geotransform)
        band = raster_g.GetRasterBand(1)
        band.WriteArray(raster_g_array)
        band.FlushCache()
        band = None
        raster_g = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.merge_rasters(
                [raster_f_path, raster_g_path], target_path)
        expected_message = 'PIXELTYPE different between rasters'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        raster_h_path = os.path.join(self.workspace_dir, 'raster_h.tif')
        raster_h_array = numpy.zeros((11, 11), dtype=numpy.int8)
        raster_h_array[:] = 20
        raster_h = driver.Create(
            raster_h_path, raster_h_array.shape[1], raster_h_array.shape[0],
            2, gdal.GDT_Int32)
        utm10_ref = osr.SpatialReference()
        raster_h.SetProjection(wgs84_ref.ExportToWkt())
        raster_h_geotransform = [11.1, 2, 0, -11, 0, -2]
        raster_h.SetGeoTransform(raster_h_geotransform)
        band = raster_h.GetRasterBand(1)
        band.WriteArray(raster_h_array)
        band.FlushCache()
        band = None
        raster_h = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.merge_rasters(
                [raster_a_path, raster_h_path], target_path)
        expected_message = 'Pixel sizes of all rasters are not the same.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_align_with_target_sr(self):
        """PGP: test align_and_resize_raster_stack with a target sr."""
        wgs84_sr = osr.SpatialReference()
        wgs84_sr.ImportFromEPSG(4326)  # WGS84 EPSG

        driver = gdal.GetDriverByName("GTiff")
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        new_raster = driver.Create(base_path, 10, 10, 1, gdal.GDT_Int32)
        new_raster.FlushCache()

        new_raster.SetProjection(wgs84_sr.ExportToWkt())
        new_raster.SetGeoTransform([
            -123.587984, 1.0, 0.0, 44.725814, 0.0, -1.0])
        new_raster = None

        target_path = os.path.join(self.workspace_dir, 'target.tif')

        target_ref = osr.SpatialReference()
        target_ref.ImportFromEPSG(26910)  # UTM10N EPSG

        pygeoprocessing.align_and_resize_raster_stack(
            [base_path], [target_path], ['near'],
            (3e4, -3e4), 'intersection',
            target_projection_wkt=target_ref.ExportToWkt())

        target_raster_info = pygeoprocessing.get_raster_info(target_path)

        # I have confidence in the upper left bounding box coordinate
        # hardcoded below because I plotted the WGS84 and UTM10N rasters on
        # top of each other and moused over the upper right hand corner.
        # Note the warping of wgs84 to utm will cause distortion and this
        # function attempts to make a raster that bounds that distortion
        # without losing any data, so a direct mapping from the lat/lng
        # coordinate to the one below will be slightly of because the
        # warped raster's bounding box will be a little larger.
        self.assertIs(
            numpy.testing.assert_allclose(
                [446166.79245811916, 5012714.829567],
                [target_raster_info['bounding_box'][0],
                 target_raster_info['bounding_box'][3]]), None)

    def test_get_raster_info_error_handling(self):
        """PGP: test that bad data raise good errors in get_raster_info."""
        # check for missing file
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.get_raster_info(
                os.path.join(self.workspace_dir, 'not_a_file.tif'))
        expected_message = 'Could not open'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        # check that file exists but is not a raster.
        not_a_raster_path = os.path.join(
            self.workspace_dir, 'not_a_raster.tif')
        with open(not_a_raster_path, 'w') as not_a_raster_file:
            not_a_raster_file.write("this is not a raster.\n")
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.get_raster_info(not_a_raster_path)
        expected_message = 'Could not open'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_get_vector_info_error_handling(self):
        """PGP: test that bad data raise good errors in get_vector_info."""
        # check for missing file
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.get_vector_info(
                os.path.join(self.workspace_dir, 'not_a_file.tif'))
        expected_message = 'Could not open'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        # check that file exists but is not a vector
        not_a_vector_path = os.path.join(
            self.workspace_dir, 'not_a_vector')
        os.makedirs(not_a_vector_path)
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.get_raster_info(not_a_vector_path)
        expected_message = 'Could not open'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_merge_bounding_box_list(self):
        """PGP: test merge_bounding_box_list."""
        bb_a = (-1, -1, 1, 1)
        bb_b = (-.9, -1.5, 1.5, 1)

        union_bb = pygeoprocessing.merge_bounding_box_list(
            [bb_a, bb_b], 'union')
        numpy.testing.assert_array_almost_equal(
            union_bb, [-1, -1.5, 1.5, 1])
        intersection_bb = pygeoprocessing.merge_bounding_box_list(
            [bb_a, bb_b], 'intersection')
        numpy.testing.assert_array_almost_equal(
            intersection_bb, [-.9, -1, 1, 1])

    def test_align_and_resize_raster_stack_int_with_vector_mask(self):
        """PGP.geoprocessing: align/resize raster w/ vector mask."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(pixel_a_matrix, target_nodata, base_a_path)

        resample_method_list = ['near']
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        # make a vector whose bounding box is 1 pixel large
        poly_a = shapely.geometry.box(
            _DEFAULT_ORIGIN[0],
            _DEFAULT_ORIGIN[1],
            _DEFAULT_ORIGIN[0] + _DEFAULT_PIXEL_SIZE[0],
            _DEFAULT_ORIGIN[1] + _DEFAULT_PIXEL_SIZE[1])
        poly_b = shapely.geometry.box(
            _DEFAULT_ORIGIN[0] + 2*_DEFAULT_PIXEL_SIZE[0],
            _DEFAULT_ORIGIN[1] + 2*_DEFAULT_PIXEL_SIZE[1],
            _DEFAULT_ORIGIN[0] + 3*_DEFAULT_PIXEL_SIZE[0],
            _DEFAULT_ORIGIN[1] + 3*_DEFAULT_PIXEL_SIZE[1])

        dual_poly_path = os.path.join(self.workspace_dir, 'dual_poly.gpkg')
        _geometry_to_vector(
            [poly_a, poly_b], dual_poly_path, fields={'value': ogr.OFTInteger},
            attribute_list=[{'value': 100}, {'value': 1}])

        target_path = os.path.join(self.workspace_dir, 'target_a.tif')
        pygeoprocessing.align_and_resize_raster_stack(
            [base_a_path], [target_path],
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            raster_align_index=0,
            vector_mask_options={
                'mask_vector_path': dual_poly_path,
                'mask_layer_name': 'dual_poly',
            },
            gdal_warp_options=["CUTLINE_ALL_TOUCHED=FALSE"])

        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        # the first pass doesn't do any filtering, so we should have 2 pixels
        self.assertEqual(
            numpy.count_nonzero(target_array[target_array == 1]), 2)

        # now test where only one of the polygons match
        pygeoprocessing.align_and_resize_raster_stack(
            [base_a_path], [target_path],
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            raster_align_index=0,
            vector_mask_options={
                'mask_vector_path': dual_poly_path,
                'mask_layer_name': 'dual_poly',
                'mask_vector_where_filter': 'value=1'
            })

        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        # we should have only one pixel left
        self.assertEqual(
            numpy.count_nonzero(target_array[target_array == 1]), 1)

    def test_align_and_resize_raster_stack_int_with_bad_vector_mask(self):
        """PGP.geoprocessing: align/resize raster w/ bad vector mask."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pixel_size = 30
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path, origin=[0.1, 0.1],
            pixel_size=(pixel_size, -pixel_size))

        resample_method_list = ['near']
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        # make a vector whose bounding box is 1 pixel large
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(1e3, 0)
        ring.AddPoint(1e3 + pixel_size, 0)
        ring.AddPoint(1e3 + pixel_size, pixel_size)
        ring.AddPoint(1e3 + pixel_size, 0)
        ring.AddPoint(1e3, 0)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        dual_poly_path = os.path.join(self.workspace_dir, 'dual_poly')
        vector_driver = gdal.GetDriverByName('ESRI Shapefile')
        poly_vector = vector_driver.Create(
            dual_poly_path, 0, 0, 0, gdal.GDT_Unknown)
        poly_layer = poly_vector.CreateLayer(
            'poly_vector', None, ogr.wkbPolygon)
        poly_feature = ogr.Feature(poly_layer.GetLayerDefn())
        poly_feature.SetGeometry(poly)
        poly_layer.CreateFeature(poly_feature)
        poly_layer.SyncToDisk()
        poly_layer = None
        poly_vector = None

        target_path = os.path.join(self.workspace_dir, 'target_a.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.align_and_resize_raster_stack(
                [base_a_path], [target_path],
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                raster_align_index=0,
                vector_mask_options={
                    'mask_vector_path': dual_poly_path,
                    'mask_layer_name': 'dual_poly',
                })
        expected_message = 'Bounding boxes do not intersect'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.align_and_resize_raster_stack(
                [base_a_path], [target_path],
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                raster_align_index=0,
                vector_mask_options={
                    'bad_mask_vector_path': dual_poly_path,
                    'mask_layer_name': 'dual_poly',
                })
        expected_message = 'no value for "mask_vector_path"'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.warp_raster(
                base_a_path, base_a_raster_info['pixel_size'],
                target_path, 'near',
                vector_mask_options={
                    'bad_mask_vector_path': dual_poly_path,
                    'mask_layer_name': 'dual_poly',
                })
        expected_message = 'no value for "mask_vector_path"'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.warp_raster(
                base_a_path, base_a_raster_info['pixel_size'],
                target_path, 'near',
                vector_mask_options={
                    'mask_vector_path': 'not_a_file.shp',
                    'mask_layer_name': 'dual_poly',
                })
        expected_message = 'was not found'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_disjoint_polygon_set_no_bounding_box(self):
        """PGP.geoprocessing: check disjoint sets."""
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
            square((2, -8)),  # overlap with FID 4
            square((2, -7)),  # overlap with FID 3
            square((14, -12)),
        ]

        outflow_vector = os.path.join(self.workspace_dir, 'outflow.gpkg')
        _geometry_to_vector(
            watershed_geometries, outflow_vector, projection_epsg=32731,
            vector_format='GPKG')

        disjoint_sets = pygeoprocessing.calculate_disjoint_polygon_set(
            outflow_vector)
        self.assertEqual(
            disjoint_sets,
            [set([1, 2, 3, 5]), set([4])])

    def test_disjoint_polygon_set_no_features_error(self):
        """PGP.geoprocessing: raise an error when a vector has no features."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4623)

        empty_vector_path = os.path.join(self.workspace_dir, 'empty.geojson')
        _geometry_to_vector(
            [], empty_vector_path, projection_epsg=4623)

        with self.assertRaises(RuntimeError) as cm:
            pygeoprocessing.calculate_disjoint_polygon_set(empty_vector_path)

        self.assertTrue('Vector must have geometries but does not'
                        in str(cm.exception))

    def test_assert_is_valid_pixel_size(self):
        """PGP: geoprocessing test to cover valid pixel size."""
        self.assertTrue(pygeoprocessing._assert_is_valid_pixel_size(
            (-10.5, 18282828228)))
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing._assert_is_valid_pixel_size(
                (-238.2, 'eleventeen'))
        expected_message = 'Invalid value for'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing._assert_is_valid_pixel_size(
                (-238.2, (10.2,)))
        expected_message = 'Invalid value for'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_percentile_long_type(self):
        """PGP: test percentile with long type."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        int_raster_path = os.path.join(self.workspace_dir, 'int_raster.tif')
        n_length = 10
        # I made this array from a random set and since it's 100 elements long
        # I know exactly the percentile cutoffs.
        array = numpy.array([
            1975, 153829, 346236, 359534, 372568, 432350, 468065, 620239,
            757710, 835119, 870788, 880695, 899211, 939183, 949597, 976023,
            1210404, 1242155, 1395436, 1484104, 1563806, 1787749, 2001579,
            2015145, 2080141, 2107594, 2331278, 2335667, 2508967, 2513463,
            2529240, 2764320, 2782388, 2892567, 3131013, 3242402, 3313283,
            3353958, 3427341, 3473886, 3507842, 3552610, 3730904, 3800470,
            3871533, 3955725, 4114781, 4326231, 4333170, 4464510, 4585432,
            4632068, 4671364, 4770370, 4927815, 4962157, 4974890, 5153019,
            5370756, 5592526, 5611672, 5688083, 5746114, 5833862, 5890515,
            5948526, 6030964, 6099825, 6162147, 6169317, 6181528, 6186133,
            6225623, 6732204, 6800472, 7059916, 7097505, 7112239, 7435668,
            7438680, 7713058, 7759246, 7878338, 7882983, 7974409, 8223956,
            8226559, 8355570, 8433741, 8523959, 8853540, 8999076, 9109444,
            9250199, 9262560, 9365311, 9404229, 9529068, 9597598,
            2**31], dtype=numpy.uint32)
        _array_to_raster(
            array.reshape((n_length, n_length)), -1, int_raster_path)

        percentile_cutoffs = [0.0, 22.5, 72.1, 99.0, 100.0]
        # manually rounding up the percentiles
        expected_int_percentiles = [
            array[0], array[23], array[73], array[99], array[99]]
        working_dir = os.path.join(
            self.workspace_dir, 'percentile_working_dir')
        actual_int_percentiles = pygeoprocessing.raster_band_percentile(
            (int_raster_path, 1), working_dir, percentile_cutoffs,
            heap_buffer_size=8, ffi_buffer_size=4)
        numpy.testing.assert_almost_equal(
            actual_int_percentiles, expected_int_percentiles)
        self.assertTrue(
            not os.path.exists(working_dir), 'working dir was not deleted')

    def test_percentile_double_type(self):
        """PGP: test percentile function with double type."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        percentile_cutoffs = [0.0, 22.5, 72.1, 99.0, 100.0]
        array = numpy.array([
            0.003998113607125986, 0.012483605193988612, 0.015538926080136628,
            0.0349541783138948, 0.056811563936455145, 0.06472245939357957,
            0.06763766500876733, 0.0996146617328485, 0.10319174490493743,
            0.1108529662149651, 0.11748524088704182, 0.13932099810203546,
            0.14593634331220395, 0.17290496444623904, 0.18410163405268687,
            0.19228618118906593, 0.19472498766411306, 0.19600894348473485,
            0.19675234705931377, 0.22294217186343712, 0.24301516135472034,
            0.25174824011708297, 0.25961212269403156, 0.2633977735981743,
            0.26432729041437664, 0.26786682579209775, 0.29237261233784806,
            0.29255695849184316, 0.2964090298109583, 0.2990174003705779,
            0.30405728527347686, 0.3264688470028486, 0.33514871371769506,
            0.3482838254608601, 0.35647966026887656, 0.3610103066480047,
            0.36266883466382505, 0.3722525921166677, 0.3732773924434396,
            0.37359492466545774, 0.3782442911035093, 0.38183103184230927,
            0.4061775627324341, 0.40752141481722104, 0.42563138319552407,
            0.45240943914984344, 0.48131663894772847, 0.48452027730035463,
            0.5080370178708488, 0.5160581721673511, 0.5207327746991738,
            0.5218827923543758, 0.5254400558377796, 0.5314284222888134,
            0.5399960806407419, 0.5540251419037007, 0.567883875779636,
            0.5759479782760882, 0.5762026663686868, 0.5851386281066929,
            0.6023424727834, 0.6224012318616832, 0.6349951577963391,
            0.6352127038584446, 0.6361159542649262, 0.6369708440504545,
            0.6432382687009855, 0.6449485473328685, 0.6458541196589433,
            0.6638401540497775, 0.6810857637187034, 0.6914374635530586,
            0.7146862236370655, 0.7335122551062899, 0.7380305619344611,
            0.7481552829167317, 0.7534333422153502, 0.7659278079221241,
            0.7802925160056647, 0.7840515443802779, 0.7858175566560684,
            0.7882952522603599, 0.7931210734787487, 0.8054471062280362,
            0.8369260071883123, 0.8448121201845042, 0.8457743106122408,
            0.8725394176743159, 0.8776084968191854, 0.8932892524100567,
            0.8974703081229631, 0.9246294314690737, 0.9470450112295367,
            0.9497456418201979, 0.9599420128556164, 0.9777130042139013,
            0.9913972371243881, 0.9930411737585775, 0.9963741185277734,
            0.9971933068336024], dtype=numpy.float32)
        double_raster_path = os.path.join(
            self.workspace_dir, 'double_raster.tif')
        n_length = 10
        _array_to_raster(
            array.reshape((n_length, n_length)), -1, double_raster_path)

        expected_float_percentiles = [
            array[0], array[23], array[73], array[99], array[99]]
        actual_percentiles = pygeoprocessing.raster_band_percentile(
            (double_raster_path, 1), self.workspace_dir, percentile_cutoffs,
            heap_buffer_size=0)
        numpy.testing.assert_almost_equal(
            actual_percentiles, expected_float_percentiles)
        # ensure heapfiles were removed
        self.assertEqual(
            len([path for path in os.listdir(self.workspace_dir)]), 1,
            "Expected only one file in the workspace directory after "
            "the call")

    def test_percentile_unsupported_type(self):
        """PGP: test percentile with unsupported type."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        gtiff_driver = gdal.GetDriverByName('GTiff')
        percentile_cutoffs = [0.0, 22.5, 72.1, 99.0, 100.0]
        cdouble_raster_path = os.path.join(
            self.workspace_dir, 'cdouble_raster.tif')
        n_length = 10
        cdouble_raster = gtiff_driver.Create(
            cdouble_raster_path, n_length, n_length, 1, gdal.GDT_CFloat32,
            options=['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                     'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])
        cdouble_raster.SetProjection(srs.ExportToWkt())
        cdouble_raster.SetGeoTransform([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
        cdouble_band = cdouble_raster.GetRasterBand(1)
        cdouble_band.SetNoDataValue(-1)
        cdouble_band = None
        cdouble_raster = None

        with self.assertRaises(ValueError) as cm:
            _ = pygeoprocessing.raster_band_percentile(
                (cdouble_raster_path, 1), self.workspace_dir,
                percentile_cutoffs)
        expected_message = 'Cannot process raster type'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_evaluate_raster_calculator_expression(self):
        """PGP: test evaluate raster symbolic expression."""
        n = 10
        raster_a_path = os.path.join(self.workspace_dir, 'a.tif')
        raster_b_path = os.path.join(self.workspace_dir, 'b.tif')
        val_array = numpy.array(
            range(n*n), dtype=numpy.float32).reshape((n, n))
        nodata_val = None
        _array_to_raster(
            val_array, nodata_val, raster_a_path)
        _array_to_raster(
            val_array, nodata_val, raster_b_path)

        raster_c_path = os.path.join(self.workspace_dir, 'c.tif')
        val_array = numpy.array(
            range(n*n), dtype=numpy.float32).reshape((n, n))
        c_d_nodata = -1
        val_array[0, 0] = c_d_nodata  # set as nodata
        _array_to_raster(
            val_array, c_d_nodata, raster_c_path)
        raster_d_path = os.path.join(self.workspace_dir, 'd.tif')
        val_array = numpy.array(
            range(n*n), dtype=numpy.float32).reshape((n, n))
        val_array[-1, -1] = c_d_nodata
        _array_to_raster(
            val_array, c_d_nodata, raster_d_path)

        zero_array = numpy.zeros((n, n), dtype=numpy.float32)
        raster_zero_path = os.path.join(self.workspace_dir, 'zero.tif')
        _array_to_raster(
            zero_array, nodata_val, raster_zero_path)

        ones_array = numpy.ones((n, n), dtype=numpy.float32)
        raster_ones_path = os.path.join(self.workspace_dir, 'ones.tif')
        _array_to_raster(
            ones_array, nodata_val, raster_ones_path)

        bytes_array = numpy.ones((n, n), dtype=numpy.int8) * -1
        bytes_path = os.path.join(self.workspace_dir, 'bytes.tif')
        _array_to_raster(
            bytes_array, nodata_val, bytes_path,
            creation_options=['PIXELTYPE=SIGNEDBYTE'])

        # test regular addition
        sum_expression = 'a+b'
        symbol_to_path_band_map = {
            'a': (raster_a_path, 1),
            'b': (raster_b_path, 1),
            'c': (raster_c_path, 1),
            'd': (raster_d_path, 1),
            'all_ones': (raster_ones_path, 1),
            'all_zeros': (raster_zero_path, 1),
            'byte_val': (bytes_path, 1),
        }
        target_nodata = None
        target_raster_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.symbolic.evaluate_raster_calculator_expression(
            sum_expression, symbol_to_path_band_map, target_nodata,
            target_raster_path)
        target_array = pygeoprocessing.raster_to_numpy_array(
            target_raster_path)
        numpy.testing.assert_almost_equal(
            target_array, 2*numpy.array(range(n*n)).reshape((n, n)))

        # test with two values as nodata
        target_nodata = -1
        mult_expression = 'c*d'
        pygeoprocessing.symbolic.evaluate_raster_calculator_expression(
            mult_expression, symbol_to_path_band_map, target_nodata,
            target_raster_path)
        target_array = pygeoprocessing.raster_to_numpy_array(
            target_raster_path)
        expected_array = val_array * val_array
        expected_array[0, 0] = -1
        expected_array[-1, -1] = -1
        numpy.testing.assert_almost_equal(target_array, expected_array)

        # test the case where the input raster has a nodata value but the
        # target nodata is None -- this should be an error.
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.symbolic.evaluate_raster_calculator_expression(
                mult_expression, symbol_to_path_band_map, None,
                target_raster_path)
        expected_message = '`target_nodata` is undefined (None)'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        # test divide by zero
        divide_by_zero_expr = 'all_ones / all_zeros'
        target_raster_path = os.path.join(self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.symbolic.evaluate_raster_calculator_expression(
                divide_by_zero_expr, symbol_to_path_band_map, target_nodata,
                target_raster_path)
        expected_message = 'Encountered inf in calculation'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        pygeoprocessing.symbolic.evaluate_raster_calculator_expression(
            divide_by_zero_expr, symbol_to_path_band_map, target_nodata,
            target_raster_path, default_inf=-9999)
        expected_array = numpy.empty(val_array.shape)
        expected_array[:] = -9999
        target_array = pygeoprocessing.raster_to_numpy_array(
            target_raster_path)
        numpy.testing.assert_almost_equal(target_array, expected_array)

        zero_by_zero_expr = 'all_zeros / a'
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.symbolic.evaluate_raster_calculator_expression(
                zero_by_zero_expr, symbol_to_path_band_map, target_nodata,
                target_raster_path)
        expected_message = 'Encountered NaN in calculation'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        pygeoprocessing.symbolic.evaluate_raster_calculator_expression(
            zero_by_zero_expr, symbol_to_path_band_map, target_nodata,
            target_raster_path, default_nan=-9999)
        expected_array = numpy.zeros(val_array.shape)
        expected_array[0, 0] = -9999
        target_array = pygeoprocessing.raster_to_numpy_array(
            target_raster_path)
        numpy.testing.assert_almost_equal(target_array, expected_array)
        # ensure it's a float32
        self.assertEqual(pygeoprocessing.get_raster_info(
            target_raster_path)['numpy_type'], numpy.float32)

        target_byte_raster_path = os.path.join(
            self.workspace_dir, 'byte.tif')
        byte_expression = '1+byte_val'
        pygeoprocessing.symbolic.evaluate_raster_calculator_expression(
            byte_expression, symbol_to_path_band_map, target_nodata,
            target_byte_raster_path)
        # we should get out a *signed* byte
        self.assertEqual(
            pygeoprocessing.get_raster_info(
                target_byte_raster_path)['numpy_type'], numpy.int8)

    def test_evaluate_symbolic_bad_type(self):
        """PGP: evaluate raster calculator expression gets the right type."""
        not_a_str_expression = False
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.symbolic.evaluate_raster_calculator_expression(
                not_a_str_expression, {}, None, 'raster.tif')
        actual_message = str(cm.exception)
        self.assertTrue(
            'Expected type `str` for `expression`' in actual_message,
            actual_message)

    def test_get_gis_type(self):
        """PGP: test geoprocessing type."""
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        layer.CreateField(ogr.FieldDefn('expected_value', ogr.OFTInteger))
        layer_defn = layer.GetLayerDefn()

        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1.0
        subpixel_size = 1./5. * pixel_size
        origin_x = 1.0
        origin_y = -1.0
        n = 16
        layer.StartTransaction()
        for row_index in range(n * 2):
            for col_index in range(n * 2):
                x_pos = origin_x + (
                    col_index*2 + 1 + col_index // 2) * subpixel_size
                y_pos = origin_y - (
                    row_index*2 + 1 + row_index // 2) * subpixel_size
                shapely_feature = shapely.geometry.Polygon([
                    (x_pos, y_pos),
                    (x_pos+subpixel_size, y_pos),
                    (x_pos+subpixel_size, y_pos-subpixel_size),
                    (x_pos, y_pos-subpixel_size),
                    (x_pos, y_pos)])
                new_feature = ogr.Feature(layer_defn)
                new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
                new_feature.SetGeometry(new_geometry)
                expected_value = row_index // 2 * n + col_index // 2
                new_feature.SetField('expected_value', expected_value)
                layer.CreateFeature(new_feature)
        layer.CommitTransaction()
        layer.SyncToDisk()
        layer = None
        vector = None

        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        array = numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))
        _array_to_raster(array, -1, raster_path)

        text_file_path = os.path.join(self.workspace_dir, 'text_file.txt')
        with open(text_file_path, 'w') as text_file:
            text_file.write('test')

        self.assertEqual(
            pygeoprocessing.get_gis_type(text_file_path),
            pygeoprocessing.UNKNOWN_TYPE)
        self.assertEqual(
            pygeoprocessing.get_gis_type(raster_path),
            pygeoprocessing.RASTER_TYPE)
        self.assertEqual(
            pygeoprocessing.get_gis_type(vector_path),
            pygeoprocessing.VECTOR_TYPE)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.get_gis_type('totally_fake_file')
        actual_message = str(cm.exception)
        self.assertTrue('does not exist' in actual_message, actual_message)

    def test_get_raster_info_type(self):
        """PGP: test get_raster_info's type."""
        gdal_type_numpy_pairs = (
            ('int16.tif', gdal.GDT_Int16, numpy.int16),
            ('uint16.tif', gdal.GDT_UInt16, numpy.uint16),
            ('int32.tif', gdal.GDT_Int32, numpy.int32),
            ('uint32.tif', gdal.GDT_UInt32, numpy.uint32),
            ('float32.tif', gdal.GDT_Float32, numpy.float32),
            ('float64.tif', gdal.GDT_Float64, numpy.float64),
            ('cfloat32.tif', gdal.GDT_CFloat32, numpy.csingle),
            ('cfloat64.tif', gdal.GDT_CFloat64, numpy.complex64))

        for raster_filename, gdal_type, numpy_type in gdal_type_numpy_pairs:
            raster_path = os.path.join(self.workspace_dir, raster_filename)
            array = numpy.array([[1]], dtype=numpy_type)
            _array_to_raster(array, None, raster_path)
            raster_info = pygeoprocessing.get_raster_info(raster_path)
            self.assertEqual(raster_info['numpy_type'], numpy_type)

    def test_non_geotiff_raster_types(self):
        """PGP: test mixed GTiff and gpkg raster types."""
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        n = 5
        array = numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))
        _array_to_raster(array, -1, raster_path)

        target_path = os.path.join(self.workspace_dir, 'target.gpkg')
        pygeoprocessing.raster_calculator(
            ((raster_path, 1), (raster_path, 1)), lambda a, b: a+b,
            target_path, gdal.GDT_Byte, None,
            raster_driver_creation_tuple=['gpkg', ()])
        target_raster = gdal.OpenEx(target_path)
        target_driver = target_raster.GetDriver()
        self.assertEqual(target_driver.GetDescription().lower(), 'gpkg')
        target_band = target_raster.GetRasterBand(1)
        numpy.testing.assert_array_equal(
            target_band.ReadAsArray(), array*2)

    def test_get_file_info(self):
        """PGP: geoprocessing test for `file_list` in the get_*_info ops."""
        gtiff_driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'test.tif')
        raster = gtiff_driver.Create(raster_path, 1, 1, 1, gdal.GDT_Int32)
        raster.FlushCache()
        raster_file_list = raster.GetFileList()
        raster = None
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        self.assertEqual(raster_info['file_list'], raster_file_list)

        gpkg_driver = gdal.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        vector = gpkg_driver.Create(vector_path, 0, 0, 0, gdal.GDT_Unknown)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        del layer
        vector_file_list = vector.GetFileList()
        vector = None
        vector_info = pygeoprocessing.get_vector_info(vector_path)
        self.assertEqual(vector_info['file_list'], vector_file_list)

    def test_iterblocks_bad_raster(self):
        """PGP: tests iterblocks presents useful error on missing raster."""
        with self.assertRaises(ValueError) as cm:
            _ = list(pygeoprocessing.iterblocks(('fake_file.tif', 1)))
        expected_message = 'could not be opened'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_warp_raster_signedbyte(self):
        """PGP.geoprocessing: warp raster test."""
        pixel_a_matrix = numpy.full((5, 5), -1, numpy.int8)
        target_nodata = -127
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path,
            creation_options=['PIXELTYPE=SIGNEDBYTE'], projection_epsg=4326,
            pixel_size=(1, -1), origin=(1, 1))

        wgs84_sr = osr.SpatialReference()
        wgs84_sr.ImportFromEPSG(4326)
        wgs84_wkt = wgs84_sr.ExportToWkt()

        target_raster_path = os.path.join(self.workspace_dir, 'target_a.tif')
        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)
        pygeoprocessing.warp_raster(
            base_a_path, base_a_raster_info['pixel_size'], target_raster_path,
            'near', target_projection_wkt=wgs84_wkt, n_threads=1)

        base_array = pygeoprocessing.raster_to_numpy_array(base_a_path)
        numpy.testing.assert_array_equal(pixel_a_matrix, base_array)

        array = pygeoprocessing.raster_to_numpy_array(target_raster_path)
        numpy.testing.assert_array_equal(pixel_a_matrix, array)

    def test_convolve_2d_bad_path_bands(self):
        """PGP.geoprocessing: test convolve 2d bad raster path bands."""
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.convolve_2d(
                signal_path, kernel_path, target_path)
        actual_message = str(cm.exception)
        # we expect an error about both signal and kernel
        self.assertTrue('signal' in actual_message)
        self.assertTrue('kernel' in actual_message)

    def test_convolve_2d_nodata(self):
        """PGP.geoprocessing: test convolve 2d (single thread)."""
        n_pixels = 100
        signal_array = numpy.empty((n_pixels//10, n_pixels//10), numpy.float32)
        base_nodata = -1
        signal_array[:] = base_nodata
        signal_array[n_pixels//20, n_pixels//20] = 0
        signal_array[0, 0] = 1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        _array_to_raster(signal_array, base_nodata, signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        _array_to_raster(kernel_array, base_nodata, kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            n_threads=1, ignore_nodata_and_edges=True, mask_nodata=False)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)

        expected_output = numpy.empty(signal_array.shape, numpy.float32)
        expected_output[:] = n_pixels**2 // 2
        numpy.testing.assert_allclose(target_array, expected_output)
