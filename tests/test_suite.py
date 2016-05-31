"""Smoke test to make sure basic construction of the project is correct."""

import os
import unittest

import gdal
import numpy

import pygeoprocessing
import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata
import pygeoprocessing.routing


class TestProjectionFunctions(unittest.TestCase):
    def test_projection_wkt_import_from_epsg(self):
        projection_wkt = sampledata.projection_wkt(4326)
        self.assertNotEqual(projection_wkt, None)

    def test_projection_wkt_import_from_epsg_invalid(self):
        self.assertRaises(RuntimeError, sampledata.projection_wkt, -1)


class TestDataComplexity(unittest.TestCase):
    """
    A test class for pygeoprocessing.testing.sampledata's data complexity
    checking.  This is used for checking that the user's sample raster matrices
    have a datatype that coincides with an acceptable GDAL datatype so that
    data is not lost when writing a numpy matrix to a GDAL raster.

    For checking this, pygeoprocessing.testing.sampledata.DTYPES is a list of
    tuples associating numpy datatypes with their corresponding GDAL datatypes.
    The relative index of the list indicates their relative complexity.
    A higher index indicates greater complexity (Float64 is the highest).
    Lower indices indicate lesser complexity (Byte is the lowest).

    The index of a GDAL or numpy datatype is most conveniently fetched with
    pygeoprocessing.testing.sampledata.dtype_index().
    """
    def test_gdal_dtype_index(self):
        """Verify that the GDAL byte is at index 0 in DTYPES"""
        self.assertEqual(sampledata.dtype_index(gdal.GDT_Byte), 0)

    def test_numpy_dtype_index(self):
        """Verify that numpy's int32 is at index 4 in DTYPES"""
        self.assertEqual(sampledata.dtype_index(numpy.int32), 4)

    def test_invalid_dtype(self):
        """Verify that an invalid datatype raises a RuntimeError"""
        self.assertRaises(RuntimeError, sampledata.dtype_index, 'foobar')


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
                                                     out_filename, tolerance=1e-9)

    def test_memblock_generator(self):
        """
        Verify that a raster iterator works and we can sum the correct value.
        """
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=self.raster_filename,
            dataset_opts=['TILED=YES'])

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
            [pixel_matrix, pixel_matrix], reference.origin,
            reference.projection, nodata,
            reference.pixel_size(30), filename=self.raster_filename,
            dataset_opts=['TILED=YES'])

        for data_dict, band_1_block, band_2_block in \
                pygeoprocessing.iterblocks(self.raster_filename):
            numpy.testing.assert_almost_equal(band_1_block, band_2_block)

    def test_default_blocksizes_tiled(self):
        """
        Verify that block size is set properly when default tilesize is used.
        """
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), dataset_opts=['TILED=YES'],
            filename=self.raster_filename)

        ds = gdal.Open(self.raster_filename)
        band = ds.GetRasterBand(1)
        block_size = band.GetBlockSize()

        # default geotiff block size is 256x256
        # Testing here that the block size is square, instead of a single
        # strip, which is what would have happened if the raster was not
        # created with TILED=YES.
        self.assertTrue(block_size[0] == block_size[1])

    def test_default_blocksizes_striped(self):
        """
        Verify that block size is set properly when default stripe is used.
        """
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), dataset_opts=['TILED=NO'],
            filename=self.raster_filename)

        ds = gdal.Open(self.raster_filename)
        band = ds.GetRasterBand(1)
        block_size = band.GetBlockSize()

        # If a raster is forced to be un-tiled (striped), the raster's blocks
        # will be accessed line-by-line.
        self.assertEqual(block_size[0], 1000)  # 1000 is num. columns
        self.assertEqual(block_size[1], 1)  # block is 1 pixel tall

    def test_custom_blocksizes(self):
        """
        Verify that block size is set properly.
        """
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), dataset_opts=['TILED=YES',
                                                    'BLOCKXSIZE=128',
                                                    'BLOCKYSIZE=256'],
            filename=self.raster_filename)

        ds = gdal.Open(self.raster_filename)
        band = ds.GetRasterBand(1)
        block_size = band.GetBlockSize()

        self.assertEqual(block_size, [128, 256])

    def test_custom_blocksizes_multiband(self):
        """
        Verify that block size is set properly.
        """
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix, pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), dataset_opts=['TILED=YES',
                                                    'BLOCKXSIZE=128',
                                                    'BLOCKYSIZE=256'],
            filename=self.raster_filename)

        ds = gdal.Open(self.raster_filename)
        for band_index in [1, 2]:
            band = ds.GetRasterBand(band_index)

            # Not sure why the BlockSize is a band attribute, as it's set
            # at the dataset level.
            block_size = band.GetBlockSize()

            self.assertEqual(block_size, [128, 256])


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
