"""pygeoprocessing.geoprocessing test suite."""
import contextlib
import itertools
import logging
import logging.handlers
import os
import pathlib
import queue
import shutil
import sys
import tempfile
import time
import types
import unittest
import unittest.mock
import warnings

import numpy
import packaging.version
import pygeoprocessing
import pygeoprocessing.multiprocessing
import pygeoprocessing.symbolic
import scipy.ndimage
import shapely.geometry
import shapely.wkt
from numpy.random import MT19937
from numpy.random import RandomState
from numpy.random import SeedSequence
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from pygeoprocessing.geoprocessing_core import DEFAULT_CREATION_OPTIONS
from pygeoprocessing.geoprocessing_core import \
    DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
from pygeoprocessing.geoprocessing_core import gdal_use_exceptions
from pygeoprocessing.geoprocessing_core import INT8_CREATION_OPTIONS
from pygeoprocessing.geoprocessing_core import \
    INT8_GTIFF_CREATION_TUPLE_OPTIONS

_DEFAULT_ORIGIN = (444720, 3751320)
_DEFAULT_PIXEL_SIZE = (30, -30)
_DEFAULT_EPSG = 3116

# Numpy changed their division-by-zero warning message in numpy 1.23.0
if (packaging.version.parse(numpy.__version__) <
        packaging.version.parse('1.23.0')):
    NUMPY_DIV_BY_ZERO_MSG = 'divide by zero encountered in true_divide'
    NUMPY_DIV_INVALID_VAL_MSG = 'invalid value encountered in true_divide'
else:
    NUMPY_DIV_BY_ZERO_MSG = 'divide by zero encountered in divide'
    NUMPY_DIV_INVALID_VAL_MSG = 'invalid value encountered in divide'


def passthrough(x):
    """Use in testing simple raster calculator calls."""
    return x


def arithmetic_wrangle(x):
    """Do some non trivial arithmetic."""
    result = -x**2/x**0.2
    return result


@gdal_use_exceptions
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


@gdal_use_exceptions
def _array_to_raster(
        base_array, target_nodata, target_path,
        creation_options=DEFAULT_CREATION_OPTIONS,
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


@contextlib.contextmanager
def capture_logging(logger):
    """Capture logging within a context manager.

    Args:
        logger (logging.Logger): The logger that should be monitored for
            log records within the scope of the context manager.

    Yields:
        log_records (list): A list of logging.LogRecord objects.  This list is
            yielded early in the execution, and may have logging progressively
            added to it until the context manager is exited.
    Returns:
        ``None``
    """
    message_queue = queue.Queue()
    queuehandler = logging.handlers.QueueHandler(message_queue)
    logger.addHandler(queuehandler)
    log_records = []
    yield log_records
    logger.removeHandler(queuehandler)

    # Append log records to the existing log_records list.
    while True:
        try:
            log_records.append(message_queue.get_nowait())
        except queue.Empty:
            break


class TestGeoprocessing(unittest.TestCase):
    """Tests for pygeoprocessing.geoprocessing."""

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
        for attrname in pygeoprocessing.__all__:
            try:
                func = getattr(pygeoprocessing, attrname)
                try:
                    _ = getattr(func, '__call__')
                except AttributeError:
                    self.fail(('Function %s is in pygeoprocessing.__all__ but '
                               'is not a callable') % attrname)
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
        with self.assertRaises(
                pygeoprocessing.ReclassificationMissingValuesError) as cm:
            pygeoprocessing.reclassify_raster(
                (raster_path, 1), value_map, target_path, gdal.GDT_Float32,
                target_nodata, values_required=True)
        expected_message = 'The following 1 raster values [-0.5]'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

    def test_reclassify_raster_nonnumeric_key(self):
        """PGP.geoprocessing: test reclassify raster with non-numeric key
        in value_map."""
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        target_nodata = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path)

        value_map = {1: 2, None: 3, "s": 4, numpy.nan: 5, numpy.float32(99): 6}
        with self.assertRaises(TypeError) as e:
            pygeoprocessing.reclassify_raster(
                (raster_path, 1), value_map, target_path, gdal.GDT_Float32,
                target_nodata, values_required=False)
        expected_message = "Non-numeric key(s) in value map: [None, 's']"
        actual_message = str(e.exception)
        self.assertIn(expected_message, actual_message)

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
        with self.assertRaises(ValueError):
            pygeoprocessing.reclassify_raster(
                (raster_path, 1), empty_value_map, target_path,
                gdal.GDT_Float32, target_nodata, values_required=False)

    def test_reclassify_raster_int_to_float(self):
        """PGP.geoprocessing: test reclassify to float types."""
        pixel_matrix = numpy.array([
            [2, 2],
            [3, 3]
        ], dtype=numpy.int8)
        target_nodata = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        _array_to_raster(
            pixel_matrix, target_nodata, raster_path)

        value_map = {
            2: 3.1,
            3: 4.567
        }
        expected = numpy.array([
            [3.1, 3.1],
            [4.567, 4.567]
        ])
        pygeoprocessing.reclassify_raster(
            (raster_path, 1), value_map, target_path,
            gdal.GDT_Float32, target_nodata, values_required=True)
        actual = pygeoprocessing.raster_to_numpy_array(target_path)
        numpy.testing.assert_allclose(actual, expected)

    def test_reclassify_raster_reclass_nodata(self):
        """PGP.geoprocessing: test reclassifying nodata value."""
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        nodata = -1
        pixel_matrix[0,0] = nodata
        pixel_matrix[5,7] = nodata
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        _array_to_raster(
            pixel_matrix, nodata, raster_path)

        value_map = {
            test_value: 0,
            nodata: 1,
        }
        target_nodata = -1
        pygeoprocessing.reclassify_raster(
            (raster_path, 1), value_map, target_path, gdal.GDT_Float32,
            target_nodata, values_required=True)
        target_info = pygeoprocessing.get_raster_info(target_path)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        self.assertAlmostEqual(numpy.sum(target_array), 2)
        self.assertAlmostEqual(target_info['nodata'][0], target_nodata)

    def test_reclassify_raster_reclass_max_nodata(self):
        """PGP.geoprocessing: test reclassifying max nodata value."""
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        nodata = numpy.finfo(numpy.float32).max - 1
        pixel_matrix[0, 0] = nodata
        pixel_matrix[5, 7] = nodata
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        _array_to_raster(
            pixel_matrix, nodata, raster_path)

        value_map = {
            test_value: 0,
            nodata: 1,
        }
        target_nodata = -1
        pygeoprocessing.reclassify_raster(
            (raster_path, 1), value_map, target_path, gdal.GDT_Float32,
            target_nodata, values_required=True)
        target_info = pygeoprocessing.get_raster_info(target_path)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        self.assertAlmostEqual(numpy.sum(target_array), 2)
        self.assertAlmostEqual(target_info['nodata'][0], target_nodata)

    def test_reclassify_raster_reclass_new_nodata(self):
        """PGP.geoprocessing: test reclassifying None nodata value."""
        n_pixels = 9
        pixel_matrix = numpy.array([[0, 1, -1]], dtype=numpy.int16)
        nodata = 0
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        _array_to_raster(pixel_matrix, nodata, raster_path)

        value_map = {
            1: 5,
            -1: 6,
        }
        target_nodata = 10

        pygeoprocessing.reclassify_raster(
            (raster_path, 1), value_map, target_path, gdal.GDT_Int16,
            target_nodata, values_required=True)
        target_info = pygeoprocessing.get_raster_info(target_path)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        expected = numpy.array([[10, 5, 6]])
        numpy.testing.assert_allclose(target_array, expected)
        self.assertAlmostEqual(target_info['nodata'][0], target_nodata)

    def test_reclassify_raster_reclass_nodata_none(self):
        """PGP.geoprocessing: test reclassifying None target nodata value."""
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        nodata = -1
        pixel_matrix[0,0] = nodata
        pixel_matrix[5,7] = nodata
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        _array_to_raster(
            pixel_matrix, nodata, raster_path)

        value_map = {
            test_value: 0,
            nodata: 1,
        }
        target_nodata = None
        pygeoprocessing.reclassify_raster(
            (raster_path, 1), value_map, target_path, gdal.GDT_Float32,
            target_nodata, values_required=True)
        target_info = pygeoprocessing.get_raster_info(target_path)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        self.assertAlmostEqual(target_array[0,0], 1)
        self.assertAlmostEqual(target_array[5,7], 1)
        self.assertAlmostEqual(target_info['nodata'][0], target_nodata)

    def test_reclassify_raster_reclass_nodata_ambiguity(self):
        """PGP.geoprocessing: test target_nodata=None ambiguity."""
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        nodata = -1
        pixel_matrix[0,0] = nodata
        pixel_matrix[5,7] = nodata
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        _array_to_raster(
            pixel_matrix, nodata, raster_path)

        value_map = {
            test_value: 0,
        }
        target_nodata = None
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.reclassify_raster(
                (raster_path, 1), value_map, target_path, gdal.GDT_Float32,
                target_nodata, values_required=True)
        expected_message = (
            "target_nodata was set to None and the base raster nodata"
            " value was not represented in the value_map")
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

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

        pygeoprocessing_logger = logging.getLogger('pygeoprocessing')
        layer_name = 'my_fun_layer'
        with capture_logging(pygeoprocessing_logger) as captured_logging:
            pygeoprocessing.reproject_vector(
                base_vector_path, target_reference.ExportToWkt(),
                target_vector_path, layer_id=0, target_layer_name=layer_name)
        self.assertEqual(len(captured_logging), 2)
        self.assertIn(f'already exists, removing and overwriting',
                      captured_logging[0].msg)
        self.assertIn(f'Ignoring user-defined layer name {layer_name}',
                      captured_logging[1].msg)
        try:
            vector = ogr.Open(target_vector_path)
            layer = vector.GetLayer()
            result_reference = layer.GetSpatialRef()
            self.assertTrue(
                osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                    osr.SpatialReference(target_reference.ExportToWkt())))
            self.assertEqual(layer.GetLayerDefn().GetName(), 'target_vector')
        finally:
            layer = None
            vector = None

        with capture_logging(pygeoprocessing_logger) as captured_logging:
            target_vector_path = os.path.join(self.workspace_dir, 'test.gpkg')
            pygeoprocessing.reproject_vector(
                base_vector_path, target_reference.ExportToWkt(),
                target_vector_path, layer_id=0, driver_name='GPKG',
                target_layer_name=layer_name)
        self.assertEqual(len(captured_logging), 0)

        try:
            vector = ogr.Open(target_vector_path)
            layer = vector.GetLayer()
            result_reference = layer.GetSpatialRef()
            self.assertTrue(
                osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                    osr.SpatialReference(target_reference.ExportToWkt())))
            self.assertEqual(layer.GetLayerDefn().GetName(), layer_name)
        finally:
            layer = None
            vector = None

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
        self.assertTrue(feature_geom.equals_exact(polygon_a, 1e-6))

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

    def test_reproject_vector_handles_bad_data(self):
        """PGP.geoprocessing: reproject vector with bad data."""
        vector_path = os.path.join(self.workspace_dir, 'bad_vector.shp')
        driver = ogr.GetDriverByName('ESRI Shapefile')
        vector = driver.CreateDataSource(vector_path)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(26710)  # NAD27 / UTM zone 10N
        layer = vector.CreateLayer(
            'bad_layer', srs=srs, geom_type=ogr.wkbPoint)

        # No/empty geometry
        feature = ogr.Feature(layer.GetLayerDefn())
        layer.CreateFeature(feature)

        # Create a point at the centroid of the UTM zone 10N bounding box
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(
            ogr.CreateGeometryFromWkt('POINT (2074757.82 7209331.79)'))

        layer = None
        vector = None

        # Our target UTM zone is 59S (in NZ), so the points should not be
        # usable.
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(2134)  # NZGD2000 / UTM zone 59S
        target_srs_wkt = target_srs.ExportToWkt()

        target_path = os.path.join(self.workspace_dir, 'target_vector.shp')
        pygeoprocessing.reproject_vector(
            base_vector_path=vector_path,
            target_projection_wkt=target_srs_wkt,
            target_path=target_path,
        )

        # verify that both fields were skipped.
        try:
            target_vector = gdal.OpenEx(target_path)
            target_layer = target_vector.GetLayer()
            self.assertEqual(target_layer.GetFeatureCount(), 0)
        finally:
            target_vector = target_layer = None

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
        origin_x = 1.0
        origin_y = -1.0
        n = 1
        layer.StartTransaction()
        for row_index in range(n * 2):
            for col_index in range(n * 2):
                x_pos = origin_x + (
                    col_index*2 + 1 + col_index // 2) * pixel_size
                y_pos = origin_y - (
                    row_index*2 + 1 + row_index // 2) * pixel_size
                shapely_feature = shapely.geometry.Polygon([
                    (x_pos, y_pos),
                    (x_pos+pixel_size, y_pos),
                    (x_pos+pixel_size, y_pos-pixel_size),
                    (x_pos, y_pos-pixel_size),
                    (x_pos, y_pos)])
                new_feature = ogr.Feature(layer_defn)
                new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
                new_feature.SetGeometry(new_geometry)
                expected_value = row_index // 2 * n + col_index // 2
                new_feature.SetField('expected_value', expected_value)
                layer.CreateFeature(new_feature)

        # Create a feature right in the middle of the 4 geometries above
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(
            shapely.geometry.box(3, -4, 4, -3).wkb)
        new_feature.SetField('expected_value', 10)
        new_feature.SetGeometry(new_geometry)
        layer.CreateFeature(new_feature)

        # Now create one additional feature that has no geometry in order to
        # exercise the warning around the feature not having a geometry defined
        # at all.
        new_feature = ogr.Feature(layer_defn)
        new_feature.SetField('expected_value', 0)
        layer.CreateFeature(new_feature)

        # Now create one more feature, but with valid (but empty) geometry.
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkt('GEOMETRYCOLLECTION EMPTY')
        new_feature.SetGeometry(new_geometry)
        new_feature.SetField('expected_value', 1)
        layer.CreateFeature(new_feature)

        layer.CommitTransaction()
        layer.SyncToDisk()

        pygeoprocessing_logger = logging.getLogger('pygeoprocessing')
        # None of the polygons overlap the given bounding box
        with capture_logging(pygeoprocessing_logger) as captured_logging:
            result = pygeoprocessing.calculate_disjoint_polygon_set(
                vector_path, bounding_box=[-10, -10, -9, -9])
        self.assertTrue(not result)
        self.assertEqual(len(captured_logging), 3)
        self.assertIn('no geometry in', captured_logging[0].msg)
        self.assertIn('empty geometry in', captured_logging[1].msg)
        self.assertEqual('no polygons intersected the bounding box',
                         captured_logging[2].msg)

        # 4 polygons touch the center polygon, so 2 groups returned
        with capture_logging(pygeoprocessing_logger) as captured_logging:
            result = pygeoprocessing.calculate_disjoint_polygon_set(vector_path)
        self.assertEqual(len(result), 2, result)
        self.assertEqual(len(captured_logging), 2)
        self.assertIn('no geometry in', captured_logging[0].msg)
        self.assertIn('empty geometry in', captured_logging[1].msg)

        # When polygons are allowed to touch, we end up with 1 group.  The 4
        # outer polygons touch the central polygon at the corner vertices.
        with capture_logging(pygeoprocessing_logger) as captured_logging:
            result = pygeoprocessing.calculate_disjoint_polygon_set(
                vector_path, geometries_may_touch=True)
        self.assertEqual(len(result), 1, result)
        self.assertEqual(len(captured_logging), 2)
        self.assertIn('no geometry in', captured_logging[0].msg)
        self.assertIn('empty geometry in', captured_logging[1].msg)

    def test_zonal_stats_for_small_polygons(self):
        """PGP.geoprocessing: test small polygons for zonal stats."""
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs, geom_type=ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn('expected_value', ogr.OFTInteger))
        layer_defn = layer.GetLayerDefn()

        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1
        subpixel_size = pixel_size / 5
        origin_x = 1
        origin_y = -1
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

        # Add a feature with no geometry to make sure we can handle it.
        new_feature = ogr.Feature(layer_defn)
        new_feature.SetField('expected_value', 0)
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
        self.assertEqual(len(zonal_stats), 4*n*n + 1)
        for poly_id in zonal_stats:
            feature = layer.GetFeature(poly_id)
            self.assertEqual(
                feature.GetField('expected_value'),
                zonal_stats[poly_id]['sum'])

    def test_zonal_stats_no_bb_overlap(self):
        """PGP.geoprocessing: test no vector bb raster overlap."""
        vector_path = os.path.join(self.workspace_dir, 'vector.gpkg')
        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1
        subpixel_size = pixel_size / 5
        origin_x = 1
        origin_y = -1
        n = 16
        x_pos = origin_x - n
        y_pos = origin_y + n
        geom = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])
        _geometry_to_vector([geom], vector_path, projection_epsg=4326)

        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        _array_to_raster(
            numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n)),
            -1, raster_path)

        zonal_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path)
        for poly_id in zonal_stats:
            self.assertEqual(zonal_stats[poly_id]['sum'], 0)

    def test_zonal_stats_all_outside(self):
        """PGP.geoprocessing: test vector all outside raster."""
        vector_path = os.path.join(self.workspace_dir, 'vector.gpkg')

        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1
        subpixel_size = pixel_size / 5
        origin_x = 1
        origin_y = -1
        n = 16
        x_pos = origin_x - n
        y_pos = origin_y + n
        polygon_a = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])

        x_pos = origin_x + n*2
        y_pos = origin_y - n*2
        polygon_b = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])

        x_pos = origin_x - subpixel_size*.99
        y_pos = origin_y + subpixel_size*.99
        polygon_c = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])

        x_pos = origin_x + (n-.01)
        y_pos = origin_y - (n-.01)
        polygon_d = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])

        _geometry_to_vector([polygon_a, polygon_b, polygon_c, polygon_d],
                            vector_path, projection_epsg=4326)

        # this will catch a polygon that barely intersects the upper left
        # hand corner but is nodata.
        array = numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))
        array[0, 0] = -1
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        _array_to_raster(array, -1, raster_path)

        zonal_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path)
        for poly_id in zonal_stats:
            self.assertEqual(zonal_stats[poly_id]['sum'], 0)

        # this will catch a polygon that barely intersects the upper left
        # hand corner but is nodata.
        raster_path = os.path.join(
            self.workspace_dir, 'nonodata_small_raster.tif')
        array = numpy.fliplr(numpy.flipud(
            numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))))
        _array_to_raster(
            array, 255, raster_path, projection_epsg=4326,
            origin=(origin_x+n, origin_y-n), pixel_size=(1, -1))

        zonal_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path)
        for poly_id in zonal_stats:
            self.assertEqual(zonal_stats[poly_id]['sum'], 0)

    def test_zonal_stats_multiple_rasters(self):
        """PGP.geoprocessing: test zonal stats works on a stack of rasters"""
        pixel_size = 30
        n_pixels = 9
        origin = (444720, 3751320)
        polygon_a = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels, origin[1]),
            (origin[0], origin[1])])
        polygon_b = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size+origin[1]),
            (origin[0]+pixel_size, -pixel_size+origin[1]),
            (origin[0]+pixel_size, origin[1]),
            (origin[0], origin[1])])
        aggregate_vector_path = os.path.join(self.workspace_dir, 'aggregate')
        _geometry_to_vector([polygon_a, polygon_b], aggregate_vector_path)
        target_nodata = None
        raster_path_1 = os.path.join(self.workspace_dir, 'raster1.tif')
        raster_path_2 = os.path.join(self.workspace_dir, 'raster2.tif')
        _array_to_raster(
            numpy.ones((n_pixels, n_pixels), numpy.float32),
            target_nodata, raster_path_1)
        _array_to_raster(
            numpy.full((n_pixels, n_pixels), 2, numpy.float32),
            target_nodata, raster_path_2)
        result = pygeoprocessing.zonal_statistics(
            [(raster_path_1, 1), (raster_path_2, 1)],
            aggregate_vector_path,
            aggregate_layer_name=None,
            ignore_nodata=True,
            polygons_might_overlap=True)
        expected_result = [
            {
                0: {
                    'count': 81,
                    'max': 1,
                    'min': 1,
                    'nodata_count': 0,
                    'sum': 81},
                1: {
                    'count': 1,
                    'max': 1,
                    'min': 1,
                    'nodata_count': 0,
                    'sum': 1}
            }, {
                0: {
                    'count': 81,
                    'max': 2,
                    'min': 2,
                    'nodata_count': 0,
                    'sum': 162},
                1: {
                    'count': 1,
                    'max': 2,
                    'min': 2,
                    'nodata_count': 0,
                    'sum': 2}
            }]
        self.assertEqual(result, expected_result)

    def test_zonal_stats_misaligned_rasters(self):
        """PGP.geoprocessing: test zonal stats errors on misaligned rasters"""
        pixel_size = 30
        n_pixels = 9
        origin = (444720, 3751320)
        polygon_a = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+origin[1]),
            (origin[0]+pixel_size * n_pixels, origin[1]),
            (origin[0], origin[1])])
        polygon_b = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size+origin[1]),
            (origin[0]+pixel_size, -pixel_size+origin[1]),
            (origin[0]+pixel_size, origin[1]),
            (origin[0], origin[1])])
        aggregate_vector_path = os.path.join(self.workspace_dir, 'aggregate')
        _geometry_to_vector([polygon_a, polygon_b], aggregate_vector_path)
        target_nodata = None
        raster_path_1 = os.path.join(self.workspace_dir, 'raster1.tif')
        raster_path_2 = os.path.join(self.workspace_dir, 'raster2.tif')
        _array_to_raster(
            numpy.ones((n_pixels, n_pixels), numpy.float32),
            target_nodata, raster_path_1)
        _array_to_raster(
            numpy.full((n_pixels, n_pixels + 1), 2, numpy.float32),
            target_nodata, raster_path_2)
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.zonal_statistics(
                [(raster_path_1, 1), (raster_path_2, 1)],
                aggregate_vector_path)
        actual_message = str(cm.exception)
        self.assertIn(
            ('All input rasters must be aligned. Multiple values of '
             '"bounding_box" were found among the input rasters'),
            actual_message, actual_message)

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

    def test_zonal_statistics_multipolygon(self):
        """PGP.geoprocessing: test zonal stats function with multipolygons."""
        # create aggregating polygon
        pixel_size = 30.0
        n_pixels = 9
        origin = (444720, 3751320)
        polygon_a = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size + origin[1]),
            (origin[0] + pixel_size, -pixel_size + origin[1]),
            (origin[0] + pixel_size, origin[1]),
            (origin[0], origin[1])])
        origin = (origin[0] + pixel_size, origin[1] - pixel_size)
        polygon_b = shapely.geometry.Polygon([
            (origin[0], origin[1]),
            (origin[0], -pixel_size + origin[1]),
            (origin[0] + pixel_size, -pixel_size + origin[1]),
            (origin[0] + pixel_size, origin[1]),
            (origin[0], origin[1])])
        multipolygon = shapely.geometry.MultiPolygon([polygon_a, polygon_b])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector')
        _geometry_to_vector([multipolygon], aggregating_vector_path)
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
        self.assertEqual(result, {
            0: {
                'count': 2,
                'max': 1.0,
                'min': 1.0,
                'nodata_count': 0,
                'sum': 2
        }})

    def test_zonal_statistics_value_counts(self):
        """PGP.geoprocessing: test zonal stats function (value counts)."""
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
        with capture_logging(
                logging.getLogger('pygeoprocessing')) as log_messages:
            result = pygeoprocessing.zonal_statistics(
                (raster_path, 1), aggregating_vector_path,
                aggregate_layer_name=None,
                ignore_nodata=True,
                include_value_counts=True,
                polygons_might_overlap=True)

        # Raster is float32, so we expect a warning to be posted.
        self.assertEqual(len(log_messages), 1)
        self.assertEqual(log_messages[0].levelno, logging.WARNING)
        self.assertIn('Value counts requested on a floating-point raster',
                      log_messages[0].msg)
        expected_result = {
            0: {
                'count': 81,
                'max': 1.0,
                'min': 1.0,
                'nodata_count': 0,
                'sum': 81.0,
                'value_counts': {1.0: 81}},
            1: {
                'count': 1,
                'max': 1.0,
                'min': 1.0,
                'nodata_count': 0,
                'sum': 1.0,
                'value_counts': {1.0: 1}},
            2: {
                'min': None,
                'max': None,
                'count': 0,
                'nodata_count': 0,
                'sum': 0.0,
                'value_counts': {}}
        }
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
            origin=(origin_x, origin_y), pixel_size=(1.0, -1.0),
            projection_epsg=4326)

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
        self.assertIn('No such file or directory', str(cm.exception))

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
        self.assertIn(expected_message, actual_message)

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

    def test_timed_logging_adapter(self):
        """PGP.geoprocessing: check timed logging."""
        from pygeoprocessing.geoprocessing import TimedLoggingAdapter

        pygeoprocessing_logger = logging.getLogger('pygeoprocessing')
        with capture_logging(pygeoprocessing_logger) as captured_logging:
            timed_logger = TimedLoggingAdapter(interval_s=0.1)
            time.sleep(0.1)
            timed_logger.warning('message 1')  # Logged
            timed_logger.warning('message 2')  # Skipped
            pygeoprocessing_logger.warning('normal 1')  # logged
            time.sleep(0.1)
            timed_logger.warning('message 3')  # logged
            pygeoprocessing_logger.warning('normal 2')  # logged
            timed_logger.warning('message 4')  # skipped
            pygeoprocessing_logger.warning('normal 3')  # logged

        self.assertEqual(len(captured_logging), 5)
        expected_messages = [
            'message 1', 'normal 1', 'message 3', 'normal 2', 'normal 3']
        for record, expected_message in zip(captured_logging,
                                            expected_messages):
            # Check the message string
            self.assertEqual(record.msg, expected_message)

            # check that the function name logged is the name of this test
            # (which is the calling function).  This is only possible on python
            # 3.8 or later.
            if sys.version_info >= (3, 8):
                self.assertEqual(
                    record.funcName, self.test_timed_logging_adapter.__name__)

        # The rest of this test only applies to python 3.8+
        if sys.version_info < (3, 8):
            return

        # Check the custom stack frame adjustment
        with capture_logging(pygeoprocessing_logger) as captured_logging:
            timed_logger = TimedLoggingAdapter(interval_s=0.1)

            def _sub_stackframe():
                time.sleep(0.5)  # make sure we pass the interval threshold
                # The 4 is 1 more than the default, so the message SHOULD
                # report that the test called it.
                timed_logger.critical('DANGER', stacklevel=4)

            _sub_stackframe()

        self.assertEqual(len(captured_logging), 1)

        record = captured_logging[0]
        self.assertEqual(record.msg, 'DANGER')

        # check that the function name logged is the name of this test
        # (which is the parent of the calling function)
        self.assertEqual(
            record.funcName, self.test_timed_logging_adapter.__name__)

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
                pygeoprocessing.raster_to_numpy_array(
                    target_raster_path)).all())

    def test_warp_raster_invalid_paths(self):
        """PGP.geoprocessing: error on invalid raster paths."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path)

        target_raster_path = os.path.join(self.workspace_dir, 'target_a.tif')
        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        for invalid_source_path in [(base_a_path,), (base_a_path, 1)]:
            with self.assertRaises(ValueError):
                pygeoprocessing.warp_raster(
                    invalid_source_path, base_a_raster_info['pixel_size'],
                    target_raster_path, 'near', n_threads=1)

        for invalid_target_path in [(target_raster_path,),
                                    (target_raster_path, 1)]:
            with self.assertRaises(ValueError):
                pygeoprocessing.warp_raster(
                    base_a_path, base_a_raster_info['pixel_size'],
                    invalid_target_path, 'near', n_threads=1)

    def test_warp_raster_overview_level(self):
        """PGP.geoprocessing: warp raster overview test."""
        # This array is big enough that build_overviews will render several
        # overview levels.
        pixel_a_matrix = numpy.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ], dtype=numpy.float32)

        # Using overview level 0 (the base raster), we should have the same
        # output when we warp the array that has and does not have overviews
        # present.
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path)
        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)
        warped_a_path = os.path.join(self.workspace_dir, 'warped_a.tif')
        pygeoprocessing.warp_raster(
            base_a_path, base_a_raster_info['pixel_size'], warped_a_path,
            'bilinear', use_overview_level=-1)

        base_b_path = os.path.join(self.workspace_dir, 'base_b.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_b_path)
        warped_b_path = os.path.join(self.workspace_dir, 'warped_b.tif')
        pygeoprocessing.build_overviews(
            base_b_path, levels=[2, 4], resample_method='bilinear')
        pygeoprocessing.warp_raster(
            base_b_path, base_a_raster_info['pixel_size'], warped_b_path,
            'bilinear', use_overview_level=-1)

        warped_a_array = pygeoprocessing.raster_to_numpy_array(warped_a_path)
        warped_b_array = pygeoprocessing.raster_to_numpy_array(warped_b_path)
        numpy.testing.assert_allclose(warped_a_array, warped_b_array)

        # Force warping using a higher overview level.
        # Overview level 2 really means the 2nd overview in the stack, which is
        # where 4 pixels are aggregated into 1 value.
        target_raster_path = os.path.join(self.workspace_dir, 'target_c.tif')
        pygeoprocessing.warp_raster(
            base_b_path, base_a_raster_info['pixel_size'], target_raster_path,
            'bilinear', n_threads=1, use_overview_level=1)
        array = pygeoprocessing.raster_to_numpy_array(target_raster_path)
        numpy.testing.assert_allclose(array, 2.5)

    def test_warp_raster_invalid_resample_alg(self):
        """PGP.geoprocessing: error on invalid resample algorithm."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path)

        target_raster_path = os.path.join(self.workspace_dir, 'target_a.tif')
        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        with self.assertRaises(RuntimeError) as context:
            pygeoprocessing.warp_raster(
                base_a_path, base_a_raster_info['pixel_size'],
                target_raster_path, 'not_an_algorithm', n_threads=1)

        self.assertIn('Unknown resampling method', str(context.exception))

    def test_warp_raster_unusual_pixel_size(self):
        """PGP.geoprocessing: warp on unusual pixel types and sizes."""
        pixel_a_matrix = numpy.ones((1, 1), numpy.byte)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path,
            creation_options=INT8_CREATION_OPTIONS, pixel_size=(20, -20),
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
            creation_options=INT8_CREATION_OPTIONS, pixel_size=(30, -30),
            projection_epsg=4326)

        self.assertTrue(
            numpy.isclose(
                pygeoprocessing.raster_to_numpy_array(base_a_path),
                pygeoprocessing.raster_to_numpy_array(
                    expected_raster_path)).all())

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
                pygeoprocessing.raster_to_numpy_array(
                    expected_raster_path)).all())

    def test_warp_raster_mask_raster(self):
        """PGP.geoprocessing: test warp when provided a mask raster."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path)

        mask_matrix = numpy.ones((5, 5), numpy.int16)
        mask_matrix[0, 0] = 0
        mask_raster_path = os.path.join(self.workspace_dir, 'mask.tif')
        _array_to_raster(
            mask_matrix, target_nodata, mask_raster_path)

        target_raster_path = os.path.join(self.workspace_dir, 'target_a.tif')
        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)
        pygeoprocessing.warp_raster(
            base_a_path, base_a_raster_info['pixel_size'], target_raster_path,
            'near', mask_options={'mask_raster_path': mask_raster_path})

        expected_matrix = numpy.ones((5, 5), numpy.int16)
        expected_matrix[0, 0] = target_nodata

        target_array = pygeoprocessing.raster_to_numpy_array(
            target_raster_path)
        numpy.testing.assert_allclose(target_array, expected_matrix)

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
        self.assertIn(expected_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            # here pixel size is an invalid type
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, ['target_a.tif'],
                resample_method_list,
                100.0, bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)
        expected_message = 'target_pixel_size is not a tuple'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            # here pixel size has invalid values
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, ['target_a.tif'],
                resample_method_list,
                [100.0, "ten"], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)
        expected_message = 'Invalid value for `target_pixel_size`'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            # here pixel size is too long
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, ['target_a.tif'],
                resample_method_list,
                [100.0, 100.0, 100.0], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)
        expected_message = 'Invalid value for `target_pixel_size`'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

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
        self.assertIn(expected_message, actual_message)

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
            self.assertIn(expected_message, actual_message)

    def test_align_and_resize_raster_stack_no_overlap(self):
        """PGP.geoprocessing: align/resize raster no intersection error."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path, origin=[-10*30, 10*30])

        pixel_b_matrix = numpy.ones((15, 15), numpy.int16)
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
            gdal.GDT_Int32, target_nodata, calc_raster_stats=True,
            use_shared_memory=True)

        self.assertTrue(
            numpy.isclose(
                pygeoprocessing.raster_to_numpy_array(base_path),
                pygeoprocessing.raster_to_numpy_array(target_path)).all())

    def test_raster_calculator_multiprocessing(self):
        """PGP.geoprocessing: raster_calculator identity test."""
        pixel_matrix = numpy.ones((1024, 1024), numpy.int16)
        target_nodata = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        _array_to_raster(pixel_matrix, target_nodata, base_path)

        target_path = os.path.join(self.workspace_dir, 'subdir', 'target.tif')

        pygeoprocessing.multiprocessing.raster_calculator(
            [(base_path, 1)], arithmetic_wrangle, target_path,
            gdal.GDT_Int32, target_nodata, calc_raster_stats=True,
            use_shared_memory=True)

        self.assertTrue(
            numpy.isclose(
                arithmetic_wrangle(pixel_matrix),
                pygeoprocessing.raster_to_numpy_array(target_path)).all())

    def test_raster_calculator_multiprocessing_cwd(self):
        """PGP.geoprocessing: raster_calculator identity test in cwd."""
        pixel_matrix = numpy.ones((1024, 1024), numpy.int16)
        target_nodata = -1
        try:
            cwd = os.getcwd()
            os.chdir(self.workspace_dir)
            base_path = 'base.tif'
            _array_to_raster(pixel_matrix, target_nodata, base_path)

            target_path = 'target.tif'
            pygeoprocessing.multiprocessing.raster_calculator(
                [(base_path, 1)], arithmetic_wrangle, target_path,
                gdal.GDT_Int32, target_nodata, calc_raster_stats=True,
                use_shared_memory=True)

            self.assertTrue(
                numpy.isclose(
                    arithmetic_wrangle(pixel_matrix),
                    pygeoprocessing.raster_to_numpy_array(target_path)).all())
        finally:
            os.chdir(cwd)

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
        self.assertIn(expected_message, actual_message)
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
            self.assertIn(expected_message, actual_message)

    def test_raster_calculator_no_path(self):
        """PGP.geoprocessing: raster_calculator raise ex. on bad file path."""
        target_nodata = -1
        nonexistant_path = os.path.join(self.workspace_dir, 'nofile.tif')
        target_path = os.path.join(
            self.workspace_dir, 'target.tif')
        with self.assertRaises(RuntimeError) as cm:
            pygeoprocessing.raster_calculator(
                [(nonexistant_path, 1)], passthrough, target_path,
                gdal.GDT_Int32, target_nodata, calc_raster_stats=True)
        self.assertIn('No such file or directory', str(cm.exception))

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
        self.assertIn(expected_message, actual_message)

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
        self.assertIn(expected_message, actual_message)

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
        self.assertIn(expected_message, actual_message)

    def test_raster_calculator_invalid_numpy_array(self):
        """PGP.geoprocessing: handle invalid numpy array sizes."""
        target_path = os.path.join(self.workspace_dir, 'target.tif')

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [numpy.empty((3, 3, 3))], lambda x: None, target_path,
                gdal.GDT_Float32, None)
        expected_message = 'Numpy array inputs must be 2 dimensions or less'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

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
        with self.assertRaises(RuntimeError) as cm:
            # no input args should cause a ValueError
            pygeoprocessing.raster_calculator(
                [(base_path, 2)], lambda: None, target_path,
                gdal.GDT_Float32, None)
        self.assertIn("Illegal band #", str(cm.exception))

        with self.assertRaises(RuntimeError) as cm:
            # no input args should cause a ValueError
            pygeoprocessing.raster_calculator(
                [(base_path, 0)], lambda: None, target_path,
                gdal.GDT_Float32, None)
        self.assertIn("Illegal band #", str(cm.exception))

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
        self.assertIn(expected_message, actual_message)

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
        self.assertIn(expected_message, actual_message)

        y_arg = numpy.ones((4,))
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(base_path, 1), y_arg], lambda a, y: a*y,
                target_path, gdal.GDT_Float32, None)
        expected_message = (
            'Raster size (128, 128) cannot be broadcast '
            'to numpy shape (4')
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

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
        self.assertIn(expected_message, actual_message)

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
        self.assertIn(expected_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            # this will return a scalar, when it should return 2d array
            pygeoprocessing.raster_calculator(
                [x_arg], lambda x: 0.0, target_path,
                gdal.GDT_Float32, None)
        expected_message = (
            "Expected `local_op` to return a numpy.ndarray")
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

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

    @unittest.skipIf(pygeoprocessing.geoprocessing.GDAL_VERSION >= (3, 7, 0),
                     "not supported in this library version")
    def test_raster_calculator_signed_byte(self):
        """PGP.geoprocessing: test that signed byte pixels interpreted."""
        pixel_array = numpy.ones((128, 128), numpy.uint8)
        # ArcGIS will create a signed byte raster with an unsigned value of 255
        # that actually is supposed to represent -1, even though the nodata
        # value will be set as -1.
        pixel_array[0, 0] = 255  # 255 ubyte is -1 byte
        nodata_base = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        _array_to_raster(
            pixel_array, nodata_base, base_path,
            creation_options=INT8_CREATION_OPTIONS)

        # The local_op should receive at least one value less than 0
        def local_op(byte_values):
            byte_values[byte_values < 0] = 2
            return byte_values

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.raster_calculator(
            [(base_path, 1)], local_op, target_path, gdal.GDT_Byte, None)

        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        # We expect that any values less than 0 are converted to 2.
        # This ensures that we expect a negative result even though we
        # put in a positive because we know signed bytes will convert.
        self.assertEqual(target_array[0, 0], 2)
        self.assertEqual(target_array.sum(), 128 * 128 + 1)
        self.assertEqual(list(numpy.unique(target_array)), [1, 2])

        # Nodata value is not set, so it should come back as ``None``.
        try:
            target_raster = gdal.OpenEx(target_path)
            target_band = target_raster.GetRasterBand(1)
            self.assertEqual(target_band.GetNoDataValue(), None)
        finally:
            target_band = None
            target_raster = None

    @unittest.skipIf(pygeoprocessing.geoprocessing.GDAL_VERSION >= (3, 7, 0),
                     "not supported in this library version")
    def test_new_raster_from_base_unsigned_byte(self):
        """PGP.geoprocessing: test that signed byte rasters copy over."""
        pixel_array = numpy.ones((128, 128), numpy.uint8)
        pixel_array[0, 0] = 255  # 255 ubyte is -1 byte
        nodata_base = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        _array_to_raster(
            pixel_array, nodata_base, base_path,
            creation_options=INT8_CREATION_OPTIONS)

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        # 255 should convert to -1 with signed bytes
        pygeoprocessing.new_raster_from_base(
            base_path, target_path, gdal.GDT_Byte, [0],
            fill_value_list=[255],
            raster_driver_creation_tuple=INT8_GTIFF_CREATION_TUPLE_OPTIONS)

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
            raster_driver_creation_tuple=INT8_GTIFF_CREATION_TUPLE_OPTIONS)

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
        # older versions of GDAL don't properly raise this exception
        if packaging.version.parse(gdal.__version__) >= packaging.version.parse('3.3.0'):
            with self.assertRaises(ValueError) as cm:
                pygeoprocessing.create_raster_from_vector_extents(
                    source_vector_path, target_raster_path, _DEFAULT_PIXEL_SIZE,
                    target_nodata, target_pixel_type)
            self.assertIn('Invalid value for GDALDataType', str(cm.exception))

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

    def test_create_raster_from_bounding_box(self):
        """PGP.geoprocessing: test create raster from bbox."""
        bbox = [_DEFAULT_ORIGIN[0], _DEFAULT_ORIGIN[1],
                _DEFAULT_ORIGIN[0] + 100, _DEFAULT_ORIGIN[1] + 145]
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(_DEFAULT_EPSG)
        target_wkt = target_srs.ExportToWkt()

        target_raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_nodata = -12345
        fill_value = 5678
        pixel_size = (10, -10)
        pygeoprocessing.create_raster_from_bounding_box(
            bbox, target_raster_path, pixel_size, gdal.GDT_Int32,
            target_wkt, target_nodata=target_nodata, fill_value=fill_value)

        info = pygeoprocessing.get_raster_info(target_raster_path)
        self.assertEqual(info['pixel_size'], pixel_size)
        self.assertEqual(info['raster_size'], (10, 15))
        self.assertEqual(info['nodata'], [target_nodata])
        self.assertEqual(info['datatype'], gdal.GDT_Int32)

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

    def test_transform_bounding_box_out_of_bounds(self):
        """PGP.geoprocessing: test that bad transform raises an exception."""
        # going to 91 degrees north to make an error
        bounding_box_lat_lng_oob = [-45, 89, -43, 91]

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(32619)  # UTM19N EPSG

        with self.assertRaises(Exception) as cm:
            result = pygeoprocessing.transform_bounding_box(
                bounding_box_lat_lng_oob, osr.SRS_WKT_WGS84_LAT_LONG,
                target_srs.ExportToWkt())
        self.assertTrue(
            'Invalid latitude' in str(cm.exception) or
            'Invalid coordinate' in str(cm.exception) or
            'Some transformed coordinates are not finite' in str(cm.exception))

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
        self.assertIn(expected_message, actual_message)

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
            ignore_nodata_and_edges=False)
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
            mask_nodata=True, ignore_nodata_and_edges=True,
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
        self.assertIn(expected_message, actual_message)

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

    def test_convolve_2d_error_on_worker_timeout(self):
        """PGP.geoprocessing: test convolve 2d error when worker times out."""
        n_pixels = 10000
        n_kernel_pixels = 17500
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
        with self.assertRaises(RuntimeError):
            pygeoprocessing.convolve_2d(
                (signal_path, 1), (kernel_path, 1), target_path,
                max_timeout=0.5)

        # Wait for the worker thread to catch up
        # Hacky, but should be enough to avoid test failures.
        time.sleep(0.5)

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

    def test_calculate_slope_nan_nodata(self):
        """PGP.geoprocessing: test calculate slope with NaN nodata."""
        n_pixels = 9
        dem_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        dem_array[0, 0] = numpy.nan
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        target_slope_path = os.path.join(self.workspace_dir, 'slope.tif')
        _array_to_raster(
            dem_array, numpy.nan, dem_path, projection_epsg=4326,
            pixel_size=(1, -1), origin=(0.1, 0))

        pygeoprocessing.calculate_slope((dem_path, 1), target_slope_path)

        actual_slope = pygeoprocessing.raster_to_numpy_array(target_slope_path)
        expected_slope = numpy.zeros((n_pixels, n_pixels), numpy.float32)
        expected_slope[0, 0] = pygeoprocessing.get_raster_info(
            target_slope_path)['nodata'][0]
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

        self.assertIn('nonzero exit code', str(cm.exception))

    def test_rasterize_missing_file(self):
        """PGP.geoprocessing: test rasterize with no target raster."""
        n_pixels = 3
        test_value = 0.5
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

        with self.assertRaises(RuntimeError) as cm:
            pygeoprocessing.rasterize(
                base_vector_path, target_raster_path, [test_value], None,
                layer_id=0)
        self.assertIn('No such file or directory', str(cm.exception))

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
        self.assertIn(expected_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.rasterize(
                base_vector_path, target_raster_path, 1, None,
                layer_id=0)
        expected_message = "`burn_values` is not a list/tuple"
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.rasterize(
                base_vector_path, target_raster_path, None, "ATTRIBUTE=id",
                layer_id=0)
        expected_message = "`option_list` is not a list/tuple"
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

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
            expected_result = scipy.ndimage.distance_transform_edt(
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
        expected_result = scipy.ndimage.distance_transform_edt(
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
        self.assertIn(expected_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.distance_transform_edt(
                (base_raster_path, 1), target_distance_raster_path,
                working_dir=self.workspace_dir,
                sampling_distance=(1.0, -1.0))
        expected_message = 'Sample distances must be > 0.0'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

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

    def test_stitch_rasters_area_change(self):
        """PGP.geoprocessing: test stitch_rasters accounting for area."""
        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)  # WGS84 EPSG

        raster_a_path = os.path.join(self.workspace_dir, 'raster_a.tif')
        raster_a_array = numpy.zeros((1, 1), dtype=numpy.int32)
        raster_a_array[:] = 1

        utm_31n_ref = osr.SpatialReference()
        utm_31n_ref.ImportFromEPSG(32631)
        # 277438.26, 110597.97 is the easting/northing of UTM 31N
        # make a raster with a single pixel 10km X 10km
        pygeoprocessing.numpy_array_to_raster(
            raster_a_array, None, (10000, -10000), (277438.26, 110597.97),
            utm_31n_ref.ExportToWkt(), raster_a_path)

        # create a raster in wgs84 space that has a lot of pixel coverage
        # of the above raster
        target_stitch_path = os.path.join(
            self.workspace_dir, 'stitch_by_area.tif')
        pygeoprocessing.numpy_array_to_raster(
            numpy.full((1000, 1000), -1.0), -1, (0.0001, -0.0001), (1, 1),
            wgs84_ref.ExportToWkt(), target_stitch_path)

        pygeoprocessing.stitch_rasters(
            [(raster_a_path, 1)],
            ['near'], (target_stitch_path, 1),
            overlap_algorithm='etch',
            area_weight_m2_to_wgs84=True)

        target_stitch_array = pygeoprocessing.raster_to_numpy_array(
            target_stitch_path)
        # add all the non-nodata values
        valid_sum = numpy.sum(target_stitch_array[target_stitch_array != -1])
        # the result shoudl be pretty close to 1.0. it's not exact because
        # there's a lot of numerical noise introduced when slicing up pixels
        # but that's fine for what we're trying to achieve here.
        numpy.testing.assert_almost_equal(valid_sum, 1.0, decimal=4)

    def test_stitch_rasters(self):
        """PGP.geoprocessing: test stitch_rasters."""
        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)  # WGS84 EPSG

        # the following creates an overlapping set of squares of
        # left square raster at 10 and middle square of 20 with nodata
        # everywhere else
        raster_a_path = os.path.join(self.workspace_dir, 'raster_a.tif')
        raster_a_array = numpy.zeros((128, 128), dtype=numpy.int32)
        raster_a_array[:] = 10
        pygeoprocessing.numpy_array_to_raster(
            raster_a_array, None, (1, -1), (0, 0), wgs84_ref.ExportToWkt(),
            raster_a_path)
        raster_b_path = os.path.join(self.workspace_dir, 'raster_b.tif')
        raster_b_array = numpy.zeros((128, 128), dtype=numpy.int32)
        raster_b_array[:] = 20
        pygeoprocessing.numpy_array_to_raster(
            raster_b_array, None, (1, -1), (64, -64),
            wgs84_ref.ExportToWkt(), raster_b_path)

        # Test etch
        stitch_by_etch_target_path = os.path.join(
            self.workspace_dir, 'stitch_by_etch.tif')
        pygeoprocessing.numpy_array_to_raster(
            numpy.full((256, 256), -1, dtype=numpy.float32),
            -1, (1, -1), (0, 0),
            wgs84_ref.ExportToWkt(), stitch_by_etch_target_path)
        pygeoprocessing.stitch_rasters(
            [(raster_a_path, 1), (raster_b_path, 1)],
            ['near', 'near'], (stitch_by_etch_target_path, 1),
            overlap_algorithm='etch')
        target_etch_array = pygeoprocessing.raster_to_numpy_array(
            stitch_by_etch_target_path)
        # since we etch, "20" will get overwritten so write it first
        expected_etch_array = numpy.full((256, 256), -1)
        expected_etch_array[64:64+128, 64:64+128] = 20
        expected_etch_array[0:128, 0:128] = 10
        numpy.testing.assert_almost_equal(
            target_etch_array, expected_etch_array)

        # Test replace:
        stitch_by_replace_target_path = os.path.join(
            self.workspace_dir, 'stitch_by_etch.tif')
        pygeoprocessing.numpy_array_to_raster(
            numpy.full((256, 256), -1, dtype=numpy.float32),
            -1, (1, -1), (0, 0), wgs84_ref.ExportToWkt(),
            stitch_by_replace_target_path)
        pygeoprocessing.stitch_rasters(
            [(raster_a_path, 1), (raster_b_path, 1)],
            ['near', 'near'], (stitch_by_replace_target_path, 1),
            overlap_algorithm='replace')
        target_replace_array = pygeoprocessing.raster_to_numpy_array(
            stitch_by_replace_target_path)
        # since we replace we write in order 10 then 20
        expected_replace_array = numpy.full((256, 256), -1)
        expected_replace_array[0:128, 0:128] = 10
        expected_replace_array[64:64+128, 64:64+128] = 20
        numpy.testing.assert_almost_equal(
            target_replace_array, expected_replace_array)

        # Test add
        stitch_by_add_target_path = os.path.join(
            self.workspace_dir, 'stitch_by_add.tif')
        pygeoprocessing.numpy_array_to_raster(
            numpy.full((256, 256), -1, dtype=numpy.float32),
            -1, (1, -1), (0, 0), wgs84_ref.ExportToWkt(),
            stitch_by_add_target_path)
        pygeoprocessing.stitch_rasters(
            [(raster_a_path, 1), (raster_b_path, 1)],
            ['near', 'near'], (stitch_by_add_target_path, 1),
            overlap_algorithm='add')
        target_add_array = pygeoprocessing.raster_to_numpy_array(
            stitch_by_add_target_path)
        # we add on the add
        expected_add_array = numpy.full((256, 256), -1)
        expected_add_array[0:128, 0:128] = 10
        expected_add_array[64:64+128, 64:64+128] = numpy.where(
            expected_add_array[64:64+128, 64:64+128] == -1, 20,
            expected_add_array[64:64+128, 64:64+128]+20)
        numpy.testing.assert_almost_equal(
            target_add_array, expected_add_array)

    def test_stitch_rasters_error_handling(self):
        """PGP: test stich_rasters error handling."""
        regular_file_raster_path = os.path.join(
            self.workspace_dir, 'file.txt')
        pathlib.Path(regular_file_raster_path).touch()

        nodata_undefined_raster_path = os.path.join(
            self.workspace_dir, 'nodata_undefined.tif')
        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)
        pygeoprocessing.numpy_array_to_raster(
            numpy.array([[1.0]], dtype=numpy.float32), None, (1, -1), (0, 0),
            wgs84_ref.ExportToWkt(), nodata_undefined_raster_path)

        stitch_raster_path = os.path.join(self.workspace_dir, 'stitch.tif')
        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)
        pygeoprocessing.numpy_array_to_raster(
            numpy.array([[1.0]], dtype=numpy.float32), -1, (1, -1), (0, 0),
            wgs84_ref.ExportToWkt(), stitch_raster_path)

        a_raster_path = os.path.join(self.workspace_dir, 'a.tif')
        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)
        pygeoprocessing.numpy_array_to_raster(
            numpy.array([[1.0]], dtype=numpy.float32), None, (1, -1), (0, 0),
            wgs84_ref.ExportToWkt(), a_raster_path)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.stitch_rasters(
                [(a_raster_path, 1)], ['near'], (stitch_raster_path, 1),
                overlap_algorithm='bad algo')
        self.assertIn('overlap algorithm', str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.stitch_rasters(
                [(a_raster_path, 1)], ['near'], stitch_raster_path,
                overlap_algorithm='etch')
        self.assertIn(
            'Expected raster path/band tuple for '
            'target_stitch_raster_path_band',
            str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.stitch_rasters(
                [(a_raster_path, 1)], ['near'], (stitch_raster_path, 2),
                overlap_algorithm='add')
        expected_message = (
            'target_stitch_raster_path_band refers to a band that '
            'exceeds')
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.stitch_rasters(
                [(a_raster_path, 1)], ['near']*2, (stitch_raster_path, 1),
                overlap_algorithm='add')
        expected_message = 'Expected same number of elements in'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

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
        with self.assertRaises(RuntimeError) as cm:
            pygeoprocessing.get_raster_info(
                os.path.join(self.workspace_dir, 'not_a_file.tif'))
        self.assertIn('No such file or directory', str(cm.exception))

        # check that file exists but is not a raster.
        not_a_raster_path = os.path.join(
            self.workspace_dir, 'not_a_raster.tif')
        with open(not_a_raster_path, 'w') as not_a_raster_file:
            not_a_raster_file.write("this is not a raster.\n")
        with self.assertRaises(RuntimeError) as cm:
            pygeoprocessing.get_raster_info(not_a_raster_path)
        self.assertRegex(
            str(cm.exception),
            r'not recognized as [a-z ]* supported file format')

    def test_get_vector_info_error_handling(self):
        """PGP: test that bad data raise good errors in get_vector_info."""
        # check for missing file
        with self.assertRaises(RuntimeError) as cm:
            pygeoprocessing.get_vector_info(
                os.path.join(self.workspace_dir, 'not_a_file.tif'))
        self.assertIn('No such file or directory', str(cm.exception))

        # check that file exists but is not a vector
        not_a_vector_path = os.path.join(
            self.workspace_dir, 'not_a_vector')
        with open(not_a_vector_path, 'w') as not_a_vector_file:
            not_a_vector_file.write("this is not a vector.\n")
        with self.assertRaises(RuntimeError) as cm:
            pygeoprocessing.get_vector_info(not_a_vector_path)
        self.assertRegex(
            str(cm.exception),
            r'not recognized as [a-z ]* supported file format')

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
            mask_options={
                'mask_vector_path': dual_poly_path,
                'mask_layer_name': 'dual_poly',
            },
            gdal_warp_options=["CUTLINE_ALL_TOUCHED=FALSE"])

        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        # the first pass doesn't do any filtering, so we should have 2 pixels
        self.assertEqual(
            numpy.count_nonzero(target_array[target_array == 1]), 2)

        # now test where only one of the polygons match
        mask_raster_path = os.path.join(self.workspace_dir, 'mask.tif')
        pygeoprocessing.align_and_resize_raster_stack(
            [base_a_path], [target_path],
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            raster_align_index=0,
            mask_options={
                'mask_vector_path': dual_poly_path,
                'mask_layer_name': 'dual_poly',
                'mask_vector_where_filter': 'value=1',
                'mask_raster_path': mask_raster_path,
            })

        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        # we should have only one pixel left
        self.assertEqual(
            numpy.count_nonzero(target_array[target_array == 1]), 1)

        # There should also now be a mask raster with a mask pixel where we
        # have valid data.
        self.assertTrue(os.path.exists(mask_raster_path))
        mask_array = pygeoprocessing.raster_to_numpy_array(mask_raster_path)
        self.assertEqual(mask_array.sum(), 1)
        numpy.testing.assert_allclose(target_array[mask_array.astype(bool)], 1)

    def test_align_and_resize_raster_stack_int_with_vector_mask_bb(self):
        """PGP.geoprocessing: align/resize raster w/ vector mask."""
        os.makedirs(self.workspace_dir, exist_ok=True)
        pixel_a_matrix = numpy.ones((180, 360), numpy.int16)
        target_nodata = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path,
            pixel_size=(1, -1), projection_epsg=4326,
            origin=(-180, 90))

        # make a vector whose bounding box is 1 degree  large
        poly_a = shapely.geometry.box(0, 0, 1, 1)
        poly_b = shapely.geometry.box(-180, -90, 180, 90)

        poly_path = os.path.join(self.workspace_dir, 'poly.gpkg')
        _geometry_to_vector(
            [poly_a, poly_b], poly_path, fields={'value': ogr.OFTInteger},
            attribute_list=[{'value': 100}, {'value': 1}],
            projection_epsg=4326)

        utm_31w_srs = osr.SpatialReference()
        utm_31w_srs.ImportFromEPSG(32631)

        poly_bb = [0, 0, 2, 2]
        poly_bb_transform = pygeoprocessing.transform_bounding_box(
            poly_bb, osr.SRS_WKT_WGS84_LAT_LONG,
            utm_31w_srs.ExportToWkt())

        target_path = os.path.join(self.workspace_dir, 'target_a.tif')

        pygeoprocessing.align_and_resize_raster_stack(
            [base_a_path], [target_path],
            ['near'],
            (111000/2, -111000/2), poly_bb_transform,
            raster_align_index=0,
            target_projection_wkt=utm_31w_srs.ExportToWkt(),
            mask_options={
                'mask_vector_path': poly_path,
                'mask_layer_name': 'poly',
                'mask_vector_where_filter': 'value=100'
            })

        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        # chose half of 111km so that's 4 pixels in 1 degree
        self.assertEqual(
            numpy.count_nonzero(target_array[target_array == 1]), 4)

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
                mask_options={
                    'mask_vector_path': dual_poly_path,
                    'mask_layer_name': 'dual_poly',
                })
        expected_message = 'Bounding boxes do not intersect'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.align_and_resize_raster_stack(
                [base_a_path], [target_path],
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                raster_align_index=0,
                mask_options={
                    'bad_mask_vector_path': dual_poly_path,
                    'mask_layer_name': 'dual_poly',
                })
        expected_message = 'no value for "mask_vector_path"'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.warp_raster(
                base_a_path, base_a_raster_info['pixel_size'],
                target_path, 'near',
                mask_options={
                    'bad_mask_vector_path': dual_poly_path,
                    'mask_layer_name': 'dual_poly',
                })
        expected_message = 'no value for "mask_vector_path"'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.warp_raster(
                base_a_path, base_a_raster_info['pixel_size'],
                target_path, 'near',
                mask_options={
                    'mask_vector_path': 'not_a_file.shp',
                    'mask_layer_name': 'dual_poly',
                })
        expected_message = 'was not found'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

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

        self.assertIn(
            'Vector must have geometries but does not', str(cm.exception))

    def test_assert_is_valid_pixel_size(self):
        """PGP: geoprocessing test to cover valid pixel size."""
        self.assertTrue(pygeoprocessing._assert_is_valid_pixel_size(
            (-10.5, 18282828228)))
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing._assert_is_valid_pixel_size(
                (-238.2, 'eleventeen'))
        expected_message = 'Invalid value for'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing._assert_is_valid_pixel_size(
                (-238.2, (10.2,)))
        expected_message = 'Invalid value for'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

    def test_raster_band_percentile_warning(self):
        """PGP: test raster_band_percentile geographic CRS warning."""
        geo_raster_path = os.path.join(self.workspace_dir, 'geo_raster.tif')
        _array_to_raster(
            numpy.ones((10,10)), 0, geo_raster_path,
            projection_epsg=4326) # Geographic CRS
        proj_raster_path = os.path.join(self.workspace_dir, 'proj_raster.tif')
        _array_to_raster(
            numpy.ones((10,10)), 0, proj_raster_path,
            projection_epsg=3116) # Projected CRS

        percentile_cutoffs = [0.0]*5
        working_dir = os.path.join(
            self.workspace_dir, 'percentile_working_dir')

        with capture_logging(
                logging.getLogger('pygeoprocessing')) as log_messages:
            pygeoprocessing.raster_band_percentile(
                (geo_raster_path, 1), working_dir, percentile_cutoffs,
                heap_buffer_size=8, ffi_buffer_size=4,
                geographic_crs_warn=True)
        self.assertEqual(len(log_messages), 1)
        self.assertEqual(log_messages[0].levelno, logging.WARNING)
        self.assertIn('geographic CRS', log_messages[0].msg)

        with capture_logging(
                logging.getLogger('pygeoprocessing')) as log_messages:
            pygeoprocessing.raster_band_percentile(
                (proj_raster_path, 1), working_dir, percentile_cutoffs,
                heap_buffer_size=8, ffi_buffer_size=4,
                geographic_crs_warn=True)
        self.assertEqual(len(log_messages), 0)

        self.assertTrue(
            not os.path.exists(working_dir), 'working dir was not deleted')

    def test_percentile_int_type(self):
        """PGP: test percentile with int type."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        int_raster_path = os.path.join(self.workspace_dir, 'int_raster.tif')
        n_length = 10
        # I made this array from a random set and since it's 100 elements long
        # I know exactly the percentile cutoffs.
        array = numpy.array([
            0, 153829, 346236, 359534, 372568, 432350, 468065, 620239,
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
            array.reshape((n_length, n_length)), 0, int_raster_path)

        percentile_cutoffs = [0.0, 22.5, 72.1, 99.0, 100.0]
        # manually rounding up the percentiles
        expected_int_percentiles = [
            array[1], array[24], array[73], array[99]]
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
            numpy.nan, -1, 0.015538926080136628,
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
            array[2], array[25], array[73], array[99]]
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

    def test_percentile_int_type_undefined_nodata(self):
        """PGP: test percentile with int type with undefined nodata."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        int_raster_path = os.path.join(self.workspace_dir, 'int_raster.tif')
        n_length = 10
        # I made this array from a random set and since it's 100 elements int
        # I know exactly the percentile cutoffs.
        array = numpy.array([
            0, 153829, 346236, 359534, 372568, 432350, 468065, 620239,
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
            array.reshape((n_length, n_length)), None, int_raster_path)

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

    def test_percentile_double_type_undefined_nodata(self):
        """PGP: test percentile function with double and undefined nodata."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        percentile_cutoffs = [0.0, 22.5, 72.1, 99.0, 100.0]
        array = numpy.array([
            0.0032453543, 0.012483605193988612, 0.015538926080136628,
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
            array.reshape((n_length, n_length)), None, double_raster_path)

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
        self.assertIn(expected_message, actual_message)

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
            creation_options=INT8_CREATION_OPTIONS)

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
        self.assertIn(expected_message, actual_message)

        # test that divide by zero yields an inf
        divide_by_zero_expr = 'all_ones / all_zeros'
        target_raster_path = os.path.join(self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError) as cm:
            with warnings.catch_warnings():
                # Ignore the specific divide-by-zero warning we expect.
                warnings.filterwarnings(
                    action="ignore",
                    message=NUMPY_DIV_BY_ZERO_MSG,
                    category=RuntimeWarning
                )
                pygeoprocessing.symbolic.evaluate_raster_calculator_expression(
                    divide_by_zero_expr, symbol_to_path_band_map,
                    target_nodata, target_raster_path)
        expected_message = 'Encountered inf in calculation'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

        with warnings.catch_warnings():
            # Ignore the specific divide-by-zero warning we expect.
            warnings.filterwarnings(
                action="ignore",
                message=NUMPY_DIV_BY_ZERO_MSG,
                category=RuntimeWarning
            )
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
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    message=NUMPY_DIV_INVALID_VAL_MSG,
                    category=RuntimeWarning
                )
                pygeoprocessing.symbolic.evaluate_raster_calculator_expression(
                    zero_by_zero_expr, symbol_to_path_band_map, target_nodata,
                    target_raster_path)
        expected_message = 'Encountered NaN in calculation'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message=NUMPY_DIV_INVALID_VAL_MSG,
                category=RuntimeWarning
            )
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
        self.assertIn(
            'Expected type `str` for `expression`', actual_message)

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
            pygeoprocessing.get_gis_type(raster_path),
            pygeoprocessing.RASTER_TYPE)
        self.assertEqual(
            pygeoprocessing.get_gis_type(vector_path),
            pygeoprocessing.VECTOR_TYPE)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.get_gis_type(text_file_path)
        actual_message = str(cm.exception)
        self.assertIn('Could not open', actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.get_gis_type('totally_fake_file')
        actual_message = str(cm.exception)
        self.assertIn('Could not open', actual_message)

    def test_get_raster_info_type(self):
        """PGP: test get_raster_info's type."""
        gdal_type_numpy_pairs = [
            ('int16.tif', gdal.GDT_Int16, numpy.int16),
            ('uint16.tif', gdal.GDT_UInt16, numpy.uint16),
            ('int32.tif', gdal.GDT_Int32, numpy.int32),
            ('uint32.tif', gdal.GDT_UInt32, numpy.uint32),
            ('float32.tif', gdal.GDT_Float32, numpy.float32),
            ('float64.tif', gdal.GDT_Float64, numpy.float64),
            ('cfloat32.tif', gdal.GDT_CFloat32, numpy.csingle),
            ('cfloat64.tif', gdal.GDT_CFloat64, numpy.complex64)]

        if pygeoprocessing.geoprocessing.GDAL_VERSION >= (3, 5, 0):
            gdal_type_numpy_pairs.append(
                ('int64.tif', gdal.GDT_Int64, numpy.int64))
            gdal_type_numpy_pairs.append(
                ('uint64.tif', gdal.GDT_UInt64, numpy.uint64))
        if pygeoprocessing.geoprocessing.GDAL_VERSION >= (3, 7, 0):
            gdal_type_numpy_pairs.append(
                ('int8.tif', gdal.GDT_Int8, numpy.int8))
            gdal_type_numpy_pairs.append(
                ('uint8.tif', gdal.GDT_Byte, numpy.uint8))

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
        with self.assertRaises(RuntimeError) as cm:
            _ = list(pygeoprocessing.iterblocks(('fake_file.tif', 1)))
        self.assertIn('No such file or directory', str(cm.exception))

    def test_warp_raster_signedbyte(self):
        """PGP.geoprocessing: warp raster test."""
        pixel_a_matrix = numpy.full((5, 5), -1, numpy.int8)
        target_nodata = -127
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        _array_to_raster(
            pixel_a_matrix, target_nodata, base_a_path,
            creation_options=INT8_CREATION_OPTIONS, projection_epsg=4326,
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

    def test_convolve_2d_bad_input_data(self):
        """PGP.geoprocessing: test convolve 2d for programmer error."""
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.convolve_2d(
                signal_path, kernel_path, target_path)
        actual_message = str(cm.exception)
        # we expect an error about both signal and kernel
        self.assertIn('signal', actual_message)
        self.assertIn('kernel', actual_message)

    def test_convolve_2d_interpolate_nodata(self):
        """PGP.geoprocessing: test ability to fill nodata holes."""
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')

        n = 10
        # make all 1s but one nodata hole
        signal_nodata = -1
        signal_array = numpy.ones((n, n))
        signal_array[n//2, n//2] = signal_nodata
        pygeoprocessing.numpy_array_to_raster(
            signal_array, signal_nodata, (1, -1),
            (0, 0), None, signal_path)

        kernel_array = numpy.ones((n//4, n//4))
        pygeoprocessing.numpy_array_to_raster(
            kernel_array, None, (1, -1),
            (0, 0), None, kernel_path)

        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1),
            target_path, ignore_nodata_and_edges=True,
            mask_nodata=False, normalize_kernel=True)
        result = pygeoprocessing.raster_to_numpy_array(
            target_path)
        # the nodata hole is now filled with valid data
        self.assertAlmostEqual(
            pygeoprocessing.raster_to_numpy_array(
                signal_path)[n//2, n//2], signal_nodata)
        self.assertAlmostEqual(result[n//2, n//2], 1.0)

    def test_convolve_2d_nodata(self):
        """PGP.geoprocessing: test convolve 2d (single thread)."""
        n_pixels = 100
        signal_array = numpy.empty((n_pixels//10, n_pixels//10), numpy.float32)
        base_nodata = -1
        signal_array[:] = base_nodata
        signal_array[0, 0] = 1
        signal_array[-1, -1] = 0
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        _array_to_raster(signal_array, base_nodata, signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        _array_to_raster(kernel_array, base_nodata, kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            ignore_nodata_and_edges=False, mask_nodata=True)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)
        target_nodata = pygeoprocessing.get_raster_info(
            target_path)['nodata'][0]
        # target_nodata should be assigned this value when defaults are used
        self.assertAlmostEqual(
            target_nodata, float(numpy.finfo(numpy.float32).min))

        expected_output = numpy.empty(signal_array.shape, numpy.float64)
        expected_output[:] = target_nodata
        expected_output[0, 0] = 1
        expected_output[-1, -1] = 1
        numpy.testing.assert_allclose(target_array, expected_output)

    def test_convolve_2d_gaussian(self):
        """PGP.geoprocessing: test convolve 2d with large gaussian kernel."""
        # choosing twice the max memory block
        n_pixels = 256*2
        # this is a fun seed random seed
        random_state = RandomState(MT19937(SeedSequence(123456789)))
        signal_array = random_state.random((n_pixels, n_pixels))

        base_nodata = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        _array_to_raster(signal_array, base_nodata, signal_path)

        kernel_seed = numpy.zeros((n_pixels, n_pixels))
        kernel_seed[n_pixels//2, n_pixels//2] = 1
        kernel_array = scipy.ndimage.gaussian_filter(kernel_seed, 1.0)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        _array_to_raster(kernel_array, base_nodata, kernel_path)

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            ignore_nodata_and_edges=False, mask_nodata=True)
        target_array = pygeoprocessing.raster_to_numpy_array(target_path)

        # gaussian filter with constant is the same as bleeding off the edges
        expected_output = scipy.ndimage.gaussian_filter(
            signal_array, 1.0, mode='constant')

        numpy.testing.assert_allclose(
            target_array, expected_output, rtol=1e-6, atol=1e-6)

    def test_empty_vector_extent_creation(self):
        """PGP: test that empty vector extend closes nicely."""
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromWkt(osr.SRS_WKT_WGS84_LAT_LONG)

        # Make a vector with no features.
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'empty.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)
        layer = vector.CreateLayer('empty', wgs84_srs)
        layer = None
        vector = None

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.create_raster_from_vector_extents(
                vector_path, 'out_raster.tif', (20, -20), gdal.GDT_Byte, 255)
        expected_message = 'has no geometry'
        actual_message = str(cm.exception)
        self.assertIn(expected_message, actual_message)

    def test_convolve_2d_non_square_blocksizes(self):
        """PGP.geo: test that convolve 2d errors on non-square blocksizes."""
        a_path = os.path.join(self.workspace_dir, 'a.tif')
        b_path = os.path.join(self.workspace_dir, 'b.tif')
        c_path = os.path.join(self.workspace_dir, 'c.tif')
        n_pixels = 100
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        kernel_array = numpy.ones((3, 3), numpy.float32)
        target_nodata = -1
        pixel_size = (1, -1)
        origin = (0, 0)

        # make row block (can't do a column block because geotiffs force a
        # larger blocksize in that case)
        pygeoprocessing.numpy_array_to_raster(
            signal_array, target_nodata, pixel_size, origin,
            osr.SRS_WKT_WGS84_LAT_LONG,
            a_path,
            raster_driver_creation_tuple=(
                'GTIFF', (f'BLOCKXSIZE={n_pixels}', 'BLOCKYSIZE=1')))
        pygeoprocessing.numpy_array_to_raster(
            kernel_array, target_nodata, pixel_size, origin,
            osr.SRS_WKT_WGS84_LAT_LONG,
            b_path,
            raster_driver_creation_tuple=(
                'GTIFF', (f'BLOCKXSIZE={n_pixels}', f'BLOCKYSIZE={n_pixels}')))

        target_path = os.path.join(self.workspace_dir, 'target.tif')

        # test both combinations of a, b
        for signal_path, kernel_path in itertools.permutations(
                [a_path, b_path], 2):
            with self.assertRaises(ValueError) as cm:
                pygeoprocessing.convolve_2d(
                    (signal_path, 1), (kernel_path, 1), target_path,
                    ignore_nodata_and_edges=False)
            expected_message = 'has a row blocksize'
            actual_message = str(cm.exception)
            self.assertIn(expected_message, actual_message)

    def test_convolve_with_byte_kernel(self):
        """PGP: test that byte kernel can still convolve."""
        array = numpy.ones((10, 10), dtype=numpy.float32)
        kernel = numpy.ones((3, 3))
        signal_path = os.path.join(
            self.workspace_dir, 'test_convolve_2d_signal.tif')
        float_kernel_path = os.path.join(
            self.workspace_dir, 'test_convolve_2d_float_kernel.tif')
        int_kernel_path = os.path.join(
            self.workspace_dir, 'test_convolve_2d_int_kernel.tif')
        float_out_path = os.path.join(
            self.workspace_dir, 'test_convolve_2d_float_out.tif')
        int_out_path = os.path.join(
            self.workspace_dir, 'test_convolve_2d_int_out.tif')

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        projection_wkt = srs.ExportToWkt()

        pygeoprocessing.numpy_array_to_raster(
            array, 255, (20, -20), (0, 0), projection_wkt, signal_path)
        pygeoprocessing.numpy_array_to_raster(
            kernel.astype(numpy.uint8), 255, (20, -20), (0, 0), projection_wkt,
            int_kernel_path)

        pygeoprocessing.convolve_2d(
            (signal_path, 1),
            (int_kernel_path, 1),
            int_out_path,
            ignore_nodata_and_edges=True,
            normalize_kernel=True)

        # the above configuration should leave the signal intact except for
        # numerical noise
        numpy.testing.assert_almost_equal(
            pygeoprocessing.raster_to_numpy_array(int_out_path),
            array)

    def test_geometry_field_mismatch(self):
        """PGP: test exception in field name mismatch."""
        projection = osr.SpatialReference()
        projection.ImportFromEPSG(3116)
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.shapely_geometry_to_vector(
                [shapely.geometry.Point(1, -1)],
                os.path.join(self.workspace_dir, 'vector.gpkg'),
                projection.ExportToWkt(),
                'GPKG',
                fields={'field_a': ogr.OFTReal},
                attribute_list=[{'field_b': 123}],
                ogr_geom_type=ogr.wkbPoint)

        self.assertIn(
            "The fields and attributes for feature 0", str(cm.exception))

    def test_raster_map_multiply_by_scalar(self):
        """PGP: raster_map can multiply raster by scalar."""
        array = numpy.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=numpy.float32)
        path = os.path.join(self.workspace_dir, 'a.tif')
        out_path = os.path.join(self.workspace_dir, 'out.tif')
        _array_to_raster(array, -1, path)

        pygeoprocessing.raster_map(lambda a: a * 2, [path], out_path)

        out_array = pygeoprocessing.raster_to_numpy_array(out_path)
        numpy.testing.assert_allclose(out_array, array * 2)

    def test_raster_map_sum_series(self):
        """PGP: raster_map can sum a series of rasters."""
        a_array = numpy.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=numpy.float32)
        b_array = a_array * 10
        c_array = b_array * 10
        a_path = os.path.join(self.workspace_dir, 'a.tif')
        b_path = os.path.join(self.workspace_dir, 'b.tif')
        c_path = os.path.join(self.workspace_dir, 'c.tif')
        out_path = os.path.join(self.workspace_dir, 'out.tif')
        _array_to_raster(a_array, -1, a_path)
        _array_to_raster(b_array, -1, b_path)
        _array_to_raster(c_array, -1, c_path)

        pygeoprocessing.raster_map(
            lambda *xs: numpy.sum(xs, axis=0),
            [a_path, b_path, c_path], out_path)

        out_array = pygeoprocessing.raster_to_numpy_array(out_path)
        expected_array = a_array + b_array + c_array
        numpy.testing.assert_allclose(out_array, expected_array)

    def test_raster_map_multi_part_arithmetic(self):
        """PGP: raster_map can do raster arithmetic and numpy functions."""
        eff_array = numpy.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]], dtype=numpy.float32)
        ic_array = numpy.array([
            [0.1, 0.1, 0.3],
            [0.3, 0.3, 0.4],
            [0.6, 0.9, 1]], dtype=numpy.float32)
        eff_path = os.path.join(self.workspace_dir, 'eff.tif')
        ic_path = os.path.join(self.workspace_dir, 'ic.tif')
        out_path = os.path.join(self.workspace_dir, 'out.tif')
        _array_to_raster(eff_array, -1, eff_path)
        _array_to_raster(ic_array, -1, ic_path)

        def ndr(eff, ic):
            return (1 - eff) / (1 + numpy.exp((0.5 - ic) / 2))

        pygeoprocessing.raster_map(ndr, [eff_path, ic_path], out_path)

        out_array = pygeoprocessing.raster_to_numpy_array(out_path)
        expected_array = ndr(eff_array, ic_array)
        numpy.testing.assert_allclose(out_array, expected_array)

    def test_raster_map_nodata_propagates(self):
        """PGP: raster_map result is valid where all inputs are valid."""
        a_array = numpy.array([
            [1, 1, 255],
            [1, 255, 1],
            [1, 1, 1]], dtype=numpy.uint8)
        b_array = numpy.array([
            [10, 10, 10],
            [10, -1, -1],
            [10, 10, 10]], dtype=numpy.float32)
        expected_array = numpy.array([
            [11, 11, -1],
            [11, -1, -1],
            [11, 11, 11]])
        a_path = os.path.join(self.workspace_dir, 'a.tif')
        b_path = os.path.join(self.workspace_dir, 'b.tif')
        out_path = os.path.join(self.workspace_dir, 'out.tif')
        _array_to_raster(a_array, 255, a_path)
        _array_to_raster(b_array, -1, b_path)

        pygeoprocessing.raster_map(
            lambda a, b: a + b, [a_path, b_path], out_path,
            target_nodata=-1)

        out_array = pygeoprocessing.raster_to_numpy_array(out_path)
        numpy.testing.assert_allclose(out_array, expected_array)

    def test_raster_map_nodata(self):
        """PGP: raster_map uses target nodata value or chooses one for you."""
        nodata = 21
        array = numpy.array([[1, 1], [1, nodata]], dtype=numpy.uint8)
        path = os.path.join(self.workspace_dir, 'a.tif')
        out_path = os.path.join(self.workspace_dir, 'out.tif')
        _array_to_raster(array, nodata, path)

        # set target nodata
        # output raster should have that nodata value
        target_nodata = 5
        pygeoprocessing.raster_map(
            lambda a: a, [path], out_path, target_nodata=target_nodata)
        out_array = pygeoprocessing.raster_to_numpy_array(out_path)
        expected_array = numpy.array(
            [[1, 1], [1, target_nodata]], dtype=numpy.uint8)
        numpy.testing.assert_allclose(out_array, expected_array)

        # set target nodata value that can't fit in the dtype
        # output raster should have that nodata value
        with self.assertRaises(ValueError):
            pygeoprocessing.raster_map(
                lambda a: a, [path], out_path, target_nodata=-5)

        # don't set target nodata
        # an appropriate value should be chosen automatically
        pygeoprocessing.raster_map(lambda a: a, [path], out_path)
        out_array = pygeoprocessing.raster_to_numpy_array(out_path)
        expected_nodata = numpy.iinfo(numpy.uint8).max
        expected_array = numpy.array(
            [[1, 1], [1, expected_nodata]], dtype=numpy.uint8)
        numpy.testing.assert_allclose(out_array, expected_array)

    def test_raster_map_dtype(self):
        """PGP: raster_map uses target dtype value or chooses one for you."""
        array = numpy.array([[1, 1], [1, -1]], dtype=numpy.float32)
        path = os.path.join(self.workspace_dir, 'a.tif')
        out_path = os.path.join(self.workspace_dir, 'out.tif')
        _array_to_raster(array, -1, path)

        # set target dtype to uint8
        # output dtype should be uint8
        # nodata should automatically be selected: max uint8 value
        pygeoprocessing.raster_map(
            lambda a: a, [path], out_path, target_dtype=numpy.uint8)
        out_array = pygeoprocessing.raster_to_numpy_array(out_path)
        expected_nodata = numpy.iinfo(numpy.uint8).max
        expected_array = numpy.array(
            [[1, 1], [1, expected_nodata]], dtype=numpy.uint8)
        numpy.testing.assert_allclose(out_array, expected_array)
        self.assertEqual(out_array.dtype, expected_array.dtype)

        # don't set target dtype
        # output dtype should be same as input
        # nodata should automatically be selected: max float32 value
        pygeoprocessing.raster_map(lambda a: a, [path], out_path)
        out_array = pygeoprocessing.raster_to_numpy_array(out_path)
        expected_nodata = numpy.finfo(numpy.float32).max
        expected_array = numpy.array(
            [[1, 1], [1, expected_nodata]], dtype=numpy.float32)
        numpy.testing.assert_allclose(out_array, expected_array)
        self.assertEqual(out_array.dtype, expected_array.dtype)

    def test_raster_map_nodata_and_dtype(self):
        """PGP: raster_map uses nodata, dtype values or chooses for you."""
        a_array = numpy.array([[1, 1], [1, 255]], dtype=numpy.uint8)
        b_array = numpy.array([[.5, .5], [.5, -1]], dtype=numpy.float32)
        a_path = os.path.join(self.workspace_dir, 'a.tif')
        b_path = os.path.join(self.workspace_dir, 'b.tif')
        out_path = os.path.join(self.workspace_dir, 'out.tif')
        _array_to_raster(a_array, 255, a_path)
        _array_to_raster(b_array, -1, b_path)

        # can set a target nodata and dtype together
        pygeoprocessing.raster_map(
            lambda a, b: a * b, [a_path, b_path], out_path,
            target_nodata=-5, target_dtype=numpy.float64)
        out_array = pygeoprocessing.raster_to_numpy_array(out_path)
        expected_array = numpy.array(
            [[.5, .5], [.5, -5]], dtype=numpy.float64)
        numpy.testing.assert_allclose(out_array, expected_array)
        self.assertEqual(out_array.dtype, expected_array.dtype)

        # can set a smaller target dtype, even if it loses information
        pygeoprocessing.raster_map(
            lambda a, b: a * b, [a_path, b_path], out_path,
            target_nodata=-5, target_dtype=numpy.int16)
        out_array = pygeoprocessing.raster_to_numpy_array(out_path)
        expected_array = numpy.array(
            [[0, 0], [0, -5]], dtype=numpy.int16)
        numpy.testing.assert_allclose(out_array, expected_array)
        self.assertEqual(out_array.dtype, expected_array.dtype)

        # error is raised if nodata value and dtype conflict
        with self.assertRaises(ValueError):
            pygeoprocessing.raster_map(
                lambda a, b: a * b, [a_path], out_path,
                target_nodata=.1, target_dtype=numpy.int16)

    def test_raster_map_handles_int8(self):
        """PGP: raster_map can accept and output signed int8 raster."""
        a_array = numpy.array([[1, 1], [1, -1]], dtype=numpy.int8)
        a_path = os.path.join(self.workspace_dir, 'a.tif')
        out_path = os.path.join(self.workspace_dir, 'out.tif')
        _array_to_raster(
            a_array, -1, a_path, creation_options=INT8_CREATION_OPTIONS)

        # automatically keeps signed int8 type
        # automatically uses int8 max value for nodata
        pygeoprocessing.raster_map(lambda a: a, [a_path], out_path)
        out_array = pygeoprocessing.raster_to_numpy_array(out_path)
        expected_array = numpy.array([[1, 1], [1, 127]], dtype=numpy.int8)
        numpy.testing.assert_allclose(out_array, expected_array)
        self.assertEqual(out_array.dtype, expected_array.dtype)

    def test_raster_map_warn_on_multiband(self):
        """PGP: raster_map raises a warning when given a multiband raster."""
        band_1_array = numpy.array([[1, 1], [1, 1]], dtype=numpy.float32)
        band_2_array = numpy.array([[2, 2], [2, 2]], dtype=numpy.float32)
        target_path = os.path.join(self.workspace_dir, 'multiband.tif')

        driver = gdal.GetDriverByName('GTIFF')
        raster = driver.Create(
            target_path, 2, 2, 2, gdal.GDT_Float32,
            options=DEFAULT_CREATION_OPTIONS)

        projection = osr.SpatialReference()
        projection.ImportFromEPSG(_DEFAULT_EPSG)
        projection_wkt = projection.ExportToWkt()
        raster.SetProjection(projection_wkt)
        raster.SetGeoTransform(
            [_DEFAULT_ORIGIN[0], _DEFAULT_PIXEL_SIZE[0], 0,
             _DEFAULT_ORIGIN[1], 0, _DEFAULT_PIXEL_SIZE[1]])

        band_1 = raster.GetRasterBand(1)
        band_1.WriteArray(band_1_array)
        band_2 = raster.GetRasterBand(1)
        band_2.WriteArray(band_2_array)
        band_1, band_2, raster = None, None, None

        with capture_logging(
                logging.getLogger('pygeoprocessing')) as log_messages:
            pygeoprocessing.raster_map(
                lambda a: a, [target_path],
                os.path.join(self.workspace_dir, 'out.tif'))
        self.assertEqual(len(log_messages), 1)
        self.assertEqual(log_messages[0].levelno, logging.WARNING)
        self.assertIn('has more than one band', log_messages[0].msg)

    def test_raster_map_different_nodata_and_array_dtypes(self):
        """PGP: raster_map can handle similar dtypes for nodata and arrays."""
        band_1_array = numpy.array([[1, 1], [1, 1]], dtype=numpy.float32)
        nodata = float(numpy.finfo(numpy.float32).min)  # this is a float64
        source_path = os.path.join(self.workspace_dir, 'float32.tif')
        _array_to_raster(band_1_array, nodata, source_path)

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.raster_map(lambda a: a, [source_path], target_path,
                                   target_nodata=nodata)

        expected_array = numpy.array(band_1_array, dtype=numpy.float32)
        numpy.testing.assert_allclose(
            expected_array, pygeoprocessing.raster_to_numpy_array(target_path))

    def test_choose_dtype(self):
        """PGP: choose_dtype picks smallest safe dtype for raster output."""
        uint8_raster = os.path.join(self.workspace_dir, 'uint8.tif')
        int16_raster = os.path.join(self.workspace_dir, 'int16.tif')
        int32_raster = os.path.join(self.workspace_dir, 'int32.tif')
        float32_raster = os.path.join(self.workspace_dir, 'float32.tif')
        float64_raster = os.path.join(self.workspace_dir, 'float64.tif')
        for path, array in [
                (uint8_raster, numpy.array([[1]], dtype=numpy.uint8)),
                (int16_raster, numpy.array([[1]], dtype=numpy.int16)),
                (int32_raster, numpy.array([[1]], dtype=numpy.int32)),
                (float32_raster, numpy.array([[1]], dtype=numpy.float32)),
                (float64_raster, numpy.array([[1]], dtype=numpy.float64))]:
            _array_to_raster(array, -1, path)

        self.assertEqual(
            pygeoprocessing.choose_dtype(uint8_raster, uint8_raster),
            numpy.uint8)
        self.assertEqual(
            pygeoprocessing.choose_dtype(uint8_raster, int16_raster),
            numpy.int16)
        self.assertEqual(
            pygeoprocessing.choose_dtype(float32_raster, float32_raster),
            numpy.float32)
        self.assertEqual(
            pygeoprocessing.choose_dtype(float64_raster, float32_raster),
            numpy.float64)
        self.assertEqual(
            pygeoprocessing.choose_dtype(int32_raster, float32_raster),
            numpy.float64)
        self.assertEqual(
            pygeoprocessing.choose_dtype(
                uint8_raster, float32_raster, int16_raster),
            numpy.float32)
        if pygeoprocessing.geoprocessing.GDAL_VERSION >= (3, 7, 0):
            int64_raster = os.path.join(self.workspace_dir, 'int64.tif')
            _array_to_raster(
                numpy.array([[1]], dtype=numpy.int64), -1, int64_raster)
            self.assertEqual(
                pygeoprocessing.choose_dtype(int64_raster, float64_raster),
                numpy.float64)

    def test_raster_reduce(self):
        """PGP: test raster_reduce can calculate a sum."""
        block_size = 256
        array = numpy.ones((block_size * 2, block_size * 2))
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        _array_to_raster(array, -1, raster_path)

        def sum_blocks(total, block): return total + numpy.sum(block)
        spy_sum_blocks = unittest.mock.Mock(wraps=sum_blocks)
        result = pygeoprocessing.raster_reduce(spy_sum_blocks, (raster_path, 1), 0)
        self.assertEqual(result, numpy.sum(array))

        # assert sum_blocks was called with the correct arguments each time
        # default block size is 256x256 resulting in four calls
        for i, (_, (total, block), _) in enumerate(spy_sum_blocks.mock_calls):
            self.assertEqual(total, i * block_size * block_size)
            numpy.testing.assert_array_equal(
                block, numpy.ones(block_size ** 2))  # flattened block

    def test_raster_reduce_mask_nodata(self):
        """PGP: test raster_reduce can mask out nodata by default."""
        block_size = 256
        nodata = -2
        array = numpy.ones((block_size * 2, block_size * 2))
        array[0][0] = nodata  # set a pixel to nodata
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        _array_to_raster(array, nodata, raster_path)

        def sum_blocks(total, block): return total + numpy.sum(block)

        # by default, nodata should be masked out
        result = pygeoprocessing.raster_reduce(sum_blocks, (raster_path, 1), 0)
        # the nodata pixel should not be counted
        self.assertEqual(result, array.size - 1)

        # set mask_nodata=False to allow nodata
        result = pygeoprocessing.raster_reduce(
            sum_blocks, (raster_path, 1), 0, mask_nodata=False)
        # the nodata pixel should be counted
        self.assertEqual(result, array.size - 3)

    def test_build_overviews_gtiff(self):
        """PGP: test raster overviews."""
        array = numpy.ones((2000, 1000), dtype=numpy.byte)

        for internal, expected_filecount, levels in (
                (True, 1, 'auto'), (False, 2, [2, 4])):
            # Rewriting raster on each iteration to ensure we're working with a
            # fresh raster each time.
            raster_path = os.path.join(self.workspace_dir, 'raster.tif')
            _array_to_raster(array, 255, raster_path)
            pygeoprocessing.build_overviews(raster_path,
                                            internal=internal,
                                            resample_method='near',
                                            levels=levels)

            # internal overviews mean only 1 file in raster
            self.assertEqual(len(os.listdir(self.workspace_dir)),
                             expected_filecount)
            try:
                raster = gdal.Open(raster_path)
                band = raster.GetRasterBand(1)
                self.assertEqual(band.GetOverviewCount(), 2)

                for ovr_index, (shape_x, shape_y) in enumerate(
                        [(500, 1000), (250, 500)]):
                    overview = band.GetOverview(ovr_index)
                    self.assertEqual(overview.XSize, shape_x)
                    self.assertEqual(overview.YSize, shape_y)
                    numpy.testing.assert_array_equal(
                        overview.ReadAsArray(),
                        numpy.ones((shape_y, shape_x), dtype=array.dtype))
            finally:
                band = None
                raster = None

        # Test that an error was raised if we try to re-build overviews on a
        # raster that already has them.
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.build_overviews(raster_path)
        self.assertIn("Raster already has overviews", str(cm.exception))

        # Forcibly overwriting the overviews should work fine, though.
        pygeoprocessing.build_overviews(raster_path, overwrite=True)

        # Check that we can catch invalid resample methods
        with self.assertRaises(RuntimeError) as cm:
            pygeoprocessing.build_overviews(
                raster_path, overwrite=True,
                resample_method='invalid choice')
        self.assertIn('Unsupported resampling method', str(cm.exception))

    def test_get_raster_info_overviews(self):
        """PGP: raster info about overviews."""
        array = numpy.ones((2000, 1000), dtype=numpy.byte)
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        _array_to_raster(array, 255, raster_path)
        pygeoprocessing.build_overviews(
            raster_path, resample_method='near')

        raster_info = pygeoprocessing.get_raster_info(raster_path)

        # The ordering of x, y is reversed from the numpy array shape.
        self.assertEqual(
            raster_info['overviews'], [(500, 1000), (250, 500)])

    def test_integer_array(self):
        """PGP: array_equals_nodata returns integer array as expected."""
        nodata_values = [9, 9.0]

        int_array = numpy.array(
            [[4, 2, 9], [1, 9, 3], [9, 6, 1]], dtype=numpy.int16)

        expected_array = numpy.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

        for nodata in nodata_values:
            result_array = pygeoprocessing.array_equals_nodata(int_array, nodata)
            numpy.testing.assert_array_equal(result_array, expected_array)

    def test_nan_nodata_array(self):
        """PGP: test array_equals_nodata with numpy.nan nodata values."""
        array = numpy.array(
            [[4, 2, numpy.nan], [1, numpy.nan, 3], [numpy.nan, 6, 1]])

        expected_array = numpy.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

        result_array = pygeoprocessing.array_equals_nodata(array, numpy.nan)
        numpy.testing.assert_array_equal(result_array, expected_array)

    def test_none_nodata(self):
        """PGP: test array_equals_nodata when nodata is undefined (None)."""
        array = numpy.array(
            [[4, 2, numpy.nan], [1, numpy.nan, 3], [numpy.nan, 6, 1]])
        result_array = pygeoprocessing.array_equals_nodata(array, None)
        numpy.testing.assert_array_equal(
            result_array, numpy.zeros(array.shape, dtype=bool))

    def test_align_bbox(self):
        """PGP: test align_bbox expands bbox to align with grid."""
        self.assertEqual(
            pygeoprocessing.align_bbox(
                [0, 1, 0, 0, 0, 1],  # origin (0, 0), pixel width 1, pixel height 1
                [0.5, 1.7, 2.1, 3.4]),
            [0, 1, 3, 4])

        self.assertEqual(
            pygeoprocessing.align_bbox(
                [0, -1, 0, 0, 0, -1],  # origin (0, 0), pixel width -1, pixel height -1
                [-2.1, -3.4, -0.5, -1.7]),
            [-3, -4, 0, -1])

        self.assertEqual(
            pygeoprocessing.align_bbox(
                [460633.493, 30, 0, 4932268.39, 0, -30],
                [464935, 4928100, 465000, 4928139]),
            [464923.493, 4928098.39, 465013.493, 4928158.39])

        self.assertEqual(
            pygeoprocessing.align_bbox(
                [500000, -50, 0, 5000000, 0, 50],
                [499999, 5000001, 499999, 5000001]),
            [499950, 5000000, 500000, 5000050])
