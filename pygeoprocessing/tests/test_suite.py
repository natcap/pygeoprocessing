"""Smoke test to make sure basic construction of the project is correct."""

import os
import unittest

import numpy

import pygeoprocessing
import pygeoprocessing.tests


class TestRasterFunctions(unittest.TestCase):
    def setUp(self):
        self.raster_filename = pygeoprocessing.temporary_filename()

    def tearDown(self):
        os.remove(self.raster_filename)

    def test_get_nodata(self):
        """Test nodata values get set and read"""

        pixel_matrix = numpy.ones((5, 5))
        geotransform = pygeoprocessing.tests.COLOMBIA_GEOTRANSFORM(1, -1)
        for nodata in [5, 10, -5, 9999]:
            pygeoprocessing.tests.create_raster_from_array(
                pixel_matrix, pygeoprocessing.tests.COLOMBIA_SRS, geotransform,
                nodata, filename=self.raster_filename)
            raster_nodata = pygeoprocessing.get_nodata_from_uri(
                self.raster_filename)
            self.assertEqual(raster_nodata, nodata)


