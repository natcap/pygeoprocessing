"""Smoke test to make sure basic construction of the project is correct."""

import os
import unittest

import gdal
import numpy

import pygeoprocessing
import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata
import pygeoprocessing.routing


class TestRasterFunctions(unittest.TestCase):
    def setUp(self):
        self.raster_filename = pygeoprocessing.temporary_filename()

    def tearDown(self):
        os.remove(self.raster_filename)

    def test_get_nodata(self):
        """Test nodata values get set and read"""

        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        for nodata in [5, 10, -5, 9999]:
            pygeoprocessing.testing.create_raster_on_disk(
                [pixel_matrix], reference.origin, reference.projection, nodata,
                reference.pixel_size(30), filename=self.raster_filename)

            raster_nodata = pygeoprocessing.get_nodata_from_uri(
                self.raster_filename)
            self.assertEqual(raster_nodata, nodata)

    def test_vectorize_datasets_identity(self):
        """Verify lambda x:x is correct in vectorize_datasets"""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=self.raster_filename)

        out_filename = pygeoprocessing.temporary_filename()
        pygeoprocessing.vectorize_datasets(
            [self.raster_filename], lambda x: x, out_filename, gdal.GDT_Int32,
            nodata, 30, 'intersection')

        pygeoprocessing.testing.assert_rasters_equal(self.raster_filename,
                                                     out_filename)

    def test_memblock_generator(self):
        """
        Verify that a raster iterator works and we can sum the correct value.
        """
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=self.raster_filename)

        sum = 0
        for block_data, memblock in pygeoprocessing.iterblocks(self.raster_filename):
            sum += memblock.sum()

        self.assertEqual(sum, 1000000)

    def test_iterblocks_multiband(self):
        """
        Verify that iterblocks() works when we operate on a raster with multiple bands.
        """
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix, pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=self.raster_filename)

        for data_dict, band_1, band_2 in pygeoprocessing.iterblocks(self.raster_filename):
            numpy.testing.assert_almost_equal(band_1, band_2)


class TestRoutingFunctions(unittest.TestCase):
    def setUp(self):
        self.dem_filename = pygeoprocessing.temporary_filename()
        self.flow_direction_filename = pygeoprocessing.temporary_filename()

    def tearDown(self):
        os.remove(self.dem_filename)
        os.remove(self.flow_direction_filename)

    def test_simple_route(self):
        """simple sanity dinf test"""

        pixel_matrix = numpy.ones((5, 5), numpy.byte)
        reference = sampledata.SRS_COLOMBIA
        nodata = -9999

        for row_index in range(pixel_matrix.shape[0]):
            pixel_matrix[row_index, :] = row_index

        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=self.dem_filename)

        pygeoprocessing.routing.flow_direction_d_inf(
            self.dem_filename, self.flow_direction_filename)

if __name__ == '__main__':
    unittest.main()
