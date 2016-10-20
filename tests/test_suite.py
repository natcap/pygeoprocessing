"""Smoke test to make sure basic construction of the project is correct."""

import os
import unittest

import gdal
import numpy

from shapely.geometry import Polygon
import pygeoprocessing
import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata
import pygeoprocessing.routing


class TestDataComplexity(unittest.TestCase):
    """A test class for pygeoprocessing.testing.sampledata's data complexity.

    This is used for checking that the user's sample raster matrices
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
        """PGP.geoprocessing: Verify GDAL byte is at index 0 in DTYPES."""
        self.assertEqual(sampledata.dtype_index(gdal.GDT_Byte), 0)

    def test_numpy_dtype_index(self):
        """PGP.geoprocessing: Verify numpy's int32 is at index 4 in DTYPES."""
        self.assertEqual(sampledata.dtype_index(numpy.int32), 4)

    def test_invalid_dtype(self):
        """PGP.geoprocessing: Verify invalid datatype raises RuntimeError."""
        self.assertRaises(RuntimeError, sampledata.dtype_index, 'foobar')


class TestRasterFunctions(unittest.TestCase):
    """Tests for raster based functionality."""

    def setUp(self):
        """Predefine filename as something temporary."""
        self.raster_filename = pygeoprocessing.temporary_filename()
        self.aoi_filename = pygeoprocessing.temporary_filename()
        os.remove(self.aoi_filename)

    def tearDown(self):
        """Clean up temporary file made in setup."""
        os.remove(self.raster_filename)
        if os.path.exists(self.aoi_filename):
            os.remove(self.aoi_filename)

    def test_get_nodata(self):
        """PGP.geoprocessing: Test nodata values get set and read."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        for nodata in [5, 10, -5, 9999]:
            pygeoprocessing.testing.create_raster_on_disk(
                [pixel_matrix], reference.origin, reference.projection, nodata,
                reference.pixel_size(30), filename=self.raster_filename)

            raster_nodata = pygeoprocessing.get_nodata_from_uri(
                self.raster_filename)
            self.assertEqual(raster_nodata, nodata)

    def test_vect_datasets_(self):
        """PGP.geoprocessing: vect..._datasets expected error for non-list."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=self.raster_filename)

        out_filename = pygeoprocessing.temporary_filename()
        with self.assertRaises(ValueError):
            # intentionally passing a filename rather than a list of files
            # to get an expected exception
            pygeoprocessing.vectorize_datasets(
                self.raster_filename, lambda x: x, out_filename,
                gdal.GDT_Int32, nodata, 30, 'intersection')

    def test_vect_datasets_identity(self):
        """PGP.geoprocessing: vectorize_datasets f(x)=x."""
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

        pygeoprocessing.testing.assert_rasters_equal(
            self.raster_filename, out_filename, rel_tol=1e-9)

    def test_vect_datasets_identity_aoi(self):
        """PGP.geoprocessing: vectorize_datasets f(x)=x with AOI."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=self.raster_filename)

        polygons = [
            Polygon([
                (reference.origin[0] + reference.pixel_size(30)[0] * 0,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                (reference.origin[0] + reference.pixel_size(30)[0] * 5,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                (reference.origin[0] + reference.pixel_size(30)[0] * 5,
                 reference.origin[1] + reference.pixel_size(30)[1] * 5),
                (reference.origin[0] + reference.pixel_size(30)[0] * 0,
                 reference.origin[1] + reference.pixel_size(30)[1] * 5),
                (reference.origin[0] + reference.pixel_size(30)[0] * 0,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                ]),
        ]
        pygeoprocessing.testing.create_vector_on_disk(
            polygons, reference.projection, filename=self.aoi_filename)

        out_filename = pygeoprocessing.temporary_filename()
        pygeoprocessing.vectorize_datasets(
            [self.raster_filename], lambda x: x, out_filename, gdal.GDT_Int32,
            nodata, 30, 'intersection', aoi_uri=self.aoi_filename)

        pygeoprocessing.testing.assert_rasters_equal(
            self.raster_filename, out_filename, rel_tol=1e-9)

    def test_iterblocks(self):
        """PGP.geoprocessing: Sum a 1000**2 raster using iterblocks."""
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=self.raster_filename,
            dataset_opts=['TILED=YES'])

        raster_sum = 0
        for _, memblock in pygeoprocessing.iterblocks(
                self.raster_filename):
            raster_sum += memblock.sum()

        self.assertEqual(raster_sum, 1000000)

    def test_iterblocks_multiband(self):
        """PGP.geoprocessing: multiband iterblocks on identical blocks."""
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        # double one value so we can ensure we're getting out different bands
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix, 2 * pixel_matrix], reference.origin,
            reference.projection, nodata,
            reference.pixel_size(30), filename=self.raster_filename,
            dataset_opts=['TILED=YES'])

        for _, band_1_block, band_2_block in \
                pygeoprocessing.iterblocks(self.raster_filename):
            numpy.testing.assert_almost_equal(band_1_block * 2, band_2_block)

    def test_default_blocksizes_tiled(self):
        """PGP.geoprocessing: Verify block size is set on default tilesize."""
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), dataset_opts=['TILED=YES'],
            filename=self.raster_filename)

        raster = gdal.Open(self.raster_filename)
        band = raster.GetRasterBand(1)
        block_size = band.GetBlockSize()

        # default geotiff block size is 256x256
        # Testing here that the block size is square, instead of a single
        # strip, which is what would have happened if the raster was not
        # created with TILED=YES.
        self.assertTrue(block_size[0] == block_size[1])

    def test_default_blocksizes_striped(self):
        """PGP.geoprocessing: Verify block size is rows not tiled."""
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), dataset_opts=['TILED=NO'],
            filename=self.raster_filename)

        raster = gdal.Open(self.raster_filename)
        band = raster.GetRasterBand(1)
        block_size = band.GetBlockSize()

        # If a raster is forced to be un-tiled (striped), the raster's blocks
        # will be accessed line-by-line.
        self.assertEqual(block_size[0], 1000)  # 1000 is num. columns
        self.assertEqual(block_size[1], 1)  # block is 1 pixel tall

    def test_custom_blocksizes(self):
        """PGP.geoprocessing:  Verify that custom block size is set."""
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), dataset_opts=[
                'TILED=YES', 'BLOCKXSIZE=128', 'BLOCKYSIZE=256'],
            filename=self.raster_filename)
        raster = gdal.Open(self.raster_filename)
        band = raster.GetRasterBand(1)
        block_size = band.GetBlockSize()

        self.assertEqual(block_size, [128, 256])

    def test_custom_blocksizes_multiband(self):
        """PGP.geoprocessing:  Verify block sizes are set on multibands."""
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix, pixel_matrix], reference.origin,
            reference.projection, nodata, reference.pixel_size(30),
            dataset_opts=['TILED=YES', 'BLOCKXSIZE=128', 'BLOCKYSIZE=256'],
            filename=self.raster_filename)

        raster = gdal.Open(self.raster_filename)
        for band_index in [1, 2]:
            band = raster.GetRasterBand(band_index)
            # Not sure why the BlockSize is a band attribute, as it's set
            # at the dataset level.
            block_size = band.GetBlockSize()
            self.assertEqual(block_size, [128, 256])

if __name__ == '__main__':
    unittest.main()
