import unittest
import os
import subprocess
import shutil
import tempfile

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
from shapely.geometry import Polygon
import mock

import pygeoprocessing
import pygeoprocessing.testing


class RasterCreationTest(unittest.TestCase):
    def setUp(self):
        """Pre-test setUp function.

        Overridden from unittest.TestCase.setUp()
        """
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        """Post-test teardown function.

        Overridden from unittest.TestCase.tearDown()
        """
        shutil.rmtree(self.workspace)

    def test_init(self):
        from pygeoprocessing.testing import create_raster_on_disk
        from pygeoprocessing.testing.sampledata import SRS_COLOMBIA
        pixels = numpy.ones((4, 4), numpy.byte)
        nodata = 0
        reference = SRS_COLOMBIA
        filename = os.path.join(self.workspace, 'foo.tif')

        create_raster_on_disk([pixels], reference.origin,
                              reference.projection,
                              nodata, reference.pixel_size(30),
                              datatype=gdal.GDT_Byte, format='GTiff',
                              filename=filename)

        self.assertTrue(os.path.exists(filename))

        dataset = gdal.Open(filename)
        self.assertEqual(dataset.RasterXSize, 4)
        self.assertEqual(dataset.RasterYSize, 4)

        band = dataset.GetRasterBand(1)
        band_nodata = band.GetNoDataValue()
        self.assertEqual(band_nodata, nodata)

        dataset_sr = osr.SpatialReference()
        dataset_sr.ImportFromWkt(dataset.GetProjection())
        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(reference.projection)
        self.assertTrue(dataset_sr.IsSame(source_sr))

    def test_bad_driver(self):
        from pygeoprocessing.testing import create_raster_on_disk
        from pygeoprocessing.testing.sampledata import SRS_COLOMBIA
        reference = SRS_COLOMBIA

        with self.assertRaises(RuntimeError):
            create_raster_on_disk([numpy.ones((4, 4))],
                reference.origin, reference.projection, 0,
                reference.pixel_size(30), format='foo')

    def test_raster_autodtype(self):
        from pygeoprocessing.testing import create_raster_on_disk
        from pygeoprocessing.testing.sampledata import SRS_COLOMBIA
        pixels = numpy.ones((4, 4), numpy.uint16)
        nodata = 0
        reference = SRS_COLOMBIA
        filename = pygeoprocessing.temporary_filename()

        create_raster_on_disk([pixels], reference.origin,
                              reference.projection,
                              nodata, reference.pixel_size(30),
                              datatype='auto',
                              filename=filename)

        dataset = gdal.Open(filename)
        band = dataset.GetRasterBand(1)
        band_dtype = band.DataType

        # numpy.uint16 should translate to gdal.GDT_UInt16
        self.assertEqual(band_dtype, gdal.GDT_UInt16)

    def test_invalid_raster_bands(self):
        from pygeoprocessing.testing import create_raster_on_disk
        from pygeoprocessing.testing.sampledata import SRS_WILLAMETTE
        pixels = numpy.ones((4, 4), numpy.uint16)
        nodata = 0
        reference = SRS_WILLAMETTE
        filename = pygeoprocessing.temporary_filename()

        # Error raised when `pixels` is not a list.  List of 2D matrices
        # expected.
        with self.assertRaises(TypeError):
            create_raster_on_disk(pixels, reference.origin,
                 reference.projection, nodata,
                 reference.pixel_size(30), datatype='auto',
                 filename=filename)

    def test_multi_bands(self):
        from pygeoprocessing.testing import create_raster_on_disk
        from pygeoprocessing.testing.sampledata import SRS_WILLAMETTE
        pixels = [
            numpy.ones((5, 5)),
            numpy.zeros((5, 5)),
            numpy.multiply(numpy.ones((5, 5)), 3),
        ]
        nodata = 0
        reference = SRS_WILLAMETTE
        filename = pygeoprocessing.temporary_filename()

        create_raster_on_disk(pixels, reference.origin,
                              reference.projection, nodata,
                              reference.pixel_size(30),
                              datatype='auto', filename=filename)

        # check that the three bands have been written properly.
        dataset = gdal.Open(filename)
        for band_num, input_matrix in zip(range(1, 4), pixels):
            band = dataset.GetRasterBand(band_num)
            written_matrix = band.ReadAsArray()
            numpy.testing.assert_almost_equal(input_matrix, written_matrix)

    def test_mismatched_bands(self):
        """When band sizes are mismatched, TypeError should be raised"""
        from pygeoprocessing.testing import create_raster_on_disk
        from pygeoprocessing.testing.sampledata import SRS_WILLAMETTE
        pixels = [
            numpy.ones((5, 5)),
            numpy.ones((4, 4)),
            numpy.ones((7, 7))
        ]
        nodata = 0
        reference = SRS_WILLAMETTE
        filename = pygeoprocessing.temporary_filename()
        with self.assertRaises(TypeError):
            create_raster_on_disk(
                pixels, reference.origin, reference.projection, nodata,
                reference.pixel_size(30), datatype='auto',
                filename=filename)

    def test_raster_nodata_notset(self):
        """When nodata=None, a nodata value should not be set."""
        from pygeoprocessing.testing import create_raster_on_disk
        from pygeoprocessing.testing.sampledata import SRS_WILLAMETTE
        pixels = [numpy.array([[0]])]
        nodata = None
        reference = SRS_WILLAMETTE
        filename = pygeoprocessing.temporary_filename()
        create_raster_on_disk(
            pixels, reference.origin, reference.projection, nodata,
            reference.pixel_size(30), datatype='auto', filename=filename)

        set_nodata_value = pygeoprocessing.get_nodata_from_uri(filename)
        self.assertEqual(set_nodata_value, None)

    def test_raster_bad_matrix_iterable_input(self):
        """Verify TypeError raised when band_matrices not a list."""
        from pygeoprocessing.testing import create_raster_on_disk
        from pygeoprocessing.testing.sampledata import SRS_WILLAMETTE
        pixels = set([1])
        nodata = None
        reference = SRS_WILLAMETTE
        filename = pygeoprocessing.temporary_filename()
        with self.assertRaises(TypeError):
            create_raster_on_disk(
                pixels, reference.origin, reference.projection, nodata,
                reference.pixel_size(30), datatype='auto', filename=filename)

    def test_raster_multiple_dtypes(self):
        """Verify TypeError raised when matrix band dtypes are mismatched."""
        from pygeoprocessing.testing import create_raster_on_disk
        from pygeoprocessing.testing.sampledata import SRS_WILLAMETTE
        pixels = [numpy.array([[0]], dtype=numpy.int),
                  numpy.array([[0]], dtype=numpy.float)]
        nodata = None
        reference = SRS_WILLAMETTE
        filename = pygeoprocessing.temporary_filename()
        with self.assertRaises(TypeError):
            create_raster_on_disk(
                pixels, reference.origin, reference.projection, nodata,
                reference.pixel_size(30), datatype='auto', filename=filename)


class VectorCreationTest(unittest.TestCase):
    def test_init(self):
        from pygeoprocessing.testing import create_vector_on_disk
        from pygeoprocessing.testing.sampledata import SRS_COLOMBIA
        polygons = [
            Polygon([(0, 0), (1, 0), (0.5, 1), (0, 0)]),
        ]
        reference = SRS_COLOMBIA

        filename = create_vector_on_disk(polygons, reference.projection)

        vector = ogr.Open(filename)
        layer = vector.GetLayer()

        features = layer.GetFeatureCount()
        self.assertEqual(features, 1)

    def test_mismatched_geoms_attrs(self):
        from pygeoprocessing.testing import create_vector_on_disk
        from pygeoprocessing.testing.sampledata import SRS_COLOMBIA
        polygons = [
            Polygon([(0, 0), (1, 0), (0.5, 1), (0, 0)]),
        ]
        reference = SRS_COLOMBIA
        fields = {'foo': 'int'}
        attrs = []
        with self.assertRaises(AssertionError):
            create_vector_on_disk(polygons, reference.projection, fields,
                                  attrs)

    def test_wrong_field_type(self):
        from pygeoprocessing.testing import create_vector_on_disk
        from pygeoprocessing.testing.sampledata import SRS_WILLAMETTE
        polygons = []
        reference = SRS_WILLAMETTE
        fields = {'foo': 'bar'}
        with self.assertRaises(AssertionError):
            create_vector_on_disk(polygons, reference.projection, fields)

    def test_wrong_driver(self):
        from pygeoprocessing.testing import create_vector_on_disk
        from pygeoprocessing.testing.sampledata import SRS_WILLAMETTE
        with self.assertRaises(AssertionError):
            create_vector_on_disk([], SRS_WILLAMETTE.projection,
                                  vector_format='foobar')

class GISBrowserTest(unittest.TestCase):

    """Test fixture for opening a default GIS browser."""

    def test_qgis_called(self):
        """Verify QGIS is called with the correct CLI args."""
        from pygeoprocessing.testing.sampledata import \
            open_files_in_gis_browser
        file_list = ['foo']
        with mock.patch('subprocess.call'):
            open_files_in_gis_browser(file_list)
            self.assertTrue(subprocess.call.called)
            self.assertEqual(subprocess.call.call_args[0][0],
                             ['qgis'] + file_list)


class CleanupTest(unittest.TestCase):

    """Test fixture for file cleanup functionality."""

    def test_cleanup_dir(self):
        """Verify shutil.rmtree is called by cleanup()."""
        from pygeoprocessing.testing.sampledata import cleanup
        with mock.patch('shutil.rmtree'):
            cleanup('/')
            self.assertTrue(shutil.rmtree.called)

    def test_cleanup_file(self):
        """Verify os.remove is called by cleanup()."""
        from pygeoprocessing.testing.sampledata import cleanup
        with mock.patch('os.remove'):
            cleanup('/foo')
            self.assertTrue(os.remove.called)
