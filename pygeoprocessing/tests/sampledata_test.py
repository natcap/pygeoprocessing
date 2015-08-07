import unittest
import os

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
from shapely.geometry import Polygon, Point

import pygeoprocessing
import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata

class RasterFactoryTest(unittest.TestCase):
    def test_init_noargs(self):
        factory = sampledata.RasterFactory()

        # Insufficient arguments.
        self.assertRaises(TypeError, factory.new)

        # verify we can make a new raster.
        reference = sampledata.SRS_COLOMBIA
        filename = factory.new(
            band_matrix=numpy.ones((4, 4), numpy.byte),
            origin=(0, 0),
            projection_wkt=reference.projection,
            nodata=0,
            pixel_size=reference.pixel_size(30)
        )
        self.assertTrue(os.path.exists(filename))

class VectorFactoryTest(unittest.TestCase):
    def test_init_noargs(self):
        factory = sampledata.VectorFactory()

        # Insufficient arguments
        self.assertRaises(TypeError, factory.new)

        #Verify we can make a new vector
        _ = factory.new([Point([(1, 1)])], sampledata.SRS_COLOMBIA.projection)

class RasterTest(unittest.TestCase):
    def test_init(self):
        pixels = numpy.ones((4, 4), numpy.byte)
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        filename = pygeoprocessing.temporary_filename()

        sampledata.raster(pixels, reference.origin, reference.projection,
                          nodata, reference.pixel_size(30), datatype=gdal.GDT_Byte,
                          format='GTiff', filename=filename)

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
        reference = sampledata.SRS_COLOMBIA
        self.assertRaises(RuntimeError, sampledata.raster, numpy.ones((4, 4)),
                          reference.origin, reference.projection, 0,
                          reference.pixel_size(30), format='foo')

    def test_raster_autodtype(self):
        pixels = numpy.ones((4, 4), numpy.uint16)
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        filename = pygeoprocessing.temporary_filename()

        sampledata.raster(pixels, reference.origin, reference.projection,
                          nodata, reference.pixel_size(30), datatype='auto',
                          filename=filename)

        dataset = gdal.Open(filename)
        band = dataset.GetRasterBand(1)
        band_dtype = band.DataType

        # numpy.uint16 should translate to gdal.GDT_UInt16
        self.assertEqual(band_dtype, gdal.GDT_UInt16)


class VectorTest(unittest.TestCase):
    def test_init(self):
        polygons = [
            Polygon([(0, 0), (1, 0), (0.5, 1), (0, 0)]),
        ]
        reference = sampledata.SRS_COLOMBIA

        filename = sampledata.vector(polygons, reference.projection)

        vector = ogr.Open(filename)
        layer = vector.GetLayer()

        features = layer.GetFeatureCount()
        self.assertEqual(features, 1)


