"""Smoke test to make sure basic construction of the project is correct."""

import os
import unittest

import numpy

import pygeoprocessing
import pygeoprocessing.tests
import pygeoprocessing.routing


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


class TestRoutingFunctions(unittest.TestCase):
    def setUp(self):
        self.dem_filename = pygeoprocessing.temporary_filename()
        self.flow_direction_filename = pygeoprocessing.temporary_filename()

    def tearDown(self):
        os.remove(self.dem_filename)
        os.remove(self.flow_direction_filename)

    def test_simple_route(self):
        """simple sanity dinf test"""

        pixel_matrix = numpy.ones((5, 5))
        geotransform = pygeoprocessing.tests.COLOMBIA_GEOTRANSFORM(1, -1)
        nodata = -9999

        for row_index in range(pixel_matrix.shape[0]):
            pixel_matrix[row_index, :] = row_index

        pygeoprocessing.tests.create_raster_from_array(
            pixel_matrix, pygeoprocessing.tests.COLOMBIA_SRS, geotransform,
            nodata, filename=self.dem_filename)

        pygeoprocessing.routing.flow_direction_d_inf(
            self.dem_filename, self.flow_direction_filename)

if __name__ == '__main__':
    unittest.main()
