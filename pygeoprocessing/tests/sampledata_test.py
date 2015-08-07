import unittest
import os

from osgeo import gdal
from osgeo import ogr
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
        factory.new(numpy.ones((4, 4)), 0,
                    sampledata.SRS_COLOMBIA_30M)

class VectorFactoryTest(unittest.TestCase):
    def test_init_noargs(self):
        factory = sampledata.VectorFactory()

        # Insufficient arguments
        self.assertRaises(TypeError, factory.new)

        #Verify we can make a new vector
        _ = factory.new([Point([(1, 1)])], sampledata.SRS_COLOMBIA_30M)

class RasterTest(unittest.TestCase):
    def test_init(self):
        pixels = numpy.ones((4, 4), numpy.byte)
        nodata = 0
        reference = sampledata.COLOMBIA_30M
        filename = pygeoprocessing.temporary_filename()

        sampledata.raster(pixels, reference.origin, reference.projection,
                          nodata, reference.pixel_size, datatype=gdal.GDT_Byte,
                          format='GTiff', filename=filename)

        self.assertTrue(os.path.exists(filename))

        dataset = gdal.Open(filename)
        self.assertEqual(dataset.RasterXSize, 4)
        self.assertEqual(dataset.RasterYSize, 4)

        band = dataset.GetRasterBand(1)
        band_nodata = band.GetNoDataValue()
        self.assertEqual(band_nodata, nodata)

    def test_bad_driver(self):
        reference = sampledata.COLOMBIA_30M
        self.assertRaises(RuntimeError, sampledata.raster, numpy.ones((4, 4)),
                          reference.origin, reference.projection, 0,
                          reference.pixel_size, format='foo')

    def test_raster_autodtype(self):
        pixels = numpy.ones((4, 4), numpy.uint16)
        nodata = 0
        reference = sampledata.COLOMBIA_30M
        filename = pygeoprocessing.temporary_filename()

        sampledata.raster(pixels, reference.origin, reference.projection,
                          nodata, reference.pixel_size, datatype='auto',
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
        reference = sampledata.SRS_COLOMBIA_30M

        filename = sampledata.vector(polygons, reference)

        vector = ogr.Open(filename)
        layer = vector.GetLayer()

        features = layer.GetFeatureCount()
        self.assertEqual(features, 1)


