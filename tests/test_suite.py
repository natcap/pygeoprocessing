"""Smoke test to make sure basic construction of the project is correct."""

import tempfile
import os
import unittest
import mock
import shutil

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


class TestPyGeoprocessing(unittest.TestCase):
    """Tests for raster based functionality."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_get_raster_info(self):
        """PGP.geoprocessing: test for raster property info."""
        pixel_matrix = numpy.ones((5, 10), numpy.int16)
        pixel_matrix[2:4:, 2:4] = 2
        reference = sampledata.SRS_COLOMBIA
        nodata = -1.0
        pixel_matrix[0, 0] = nodata
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

        raster_info = pygeoprocessing.get_raster_info(raster_filename)

        expected_results = {
            'pixel_size': (30.0, -30.0),
            'mean_pixel_size': 30.0,
            'raster_size': (10, 5),
            'nodata': nodata,
            'n_bands': 1,
            'bounding_box': [
                reference.origin[0], reference.origin[1],
                reference.origin[0] + reference.pixel_size(30)[0] * 10,
                reference.origin[1] + reference.pixel_size(30)[1] * 5],
            'geotransform': (
                reference.origin[0], reference.pixel_size(30)[0], 0.0,
                reference.origin[1], 0, reference.pixel_size(30)[1]),
        }

        self.assertEqual(expected_results, raster_info)

    def test_raster_info_nodata_undefied(self):
        """PGP.geoprocessing: covers info case when nodata is undefined."""
        raster_matrix = numpy.empty((5, 5), numpy.int8)
        reference = sampledata.SRS_COLOMBIA
        nodata = None
        for row_index in xrange(raster_matrix.shape[1]):
            raster_matrix[row_index, :] = row_index
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [raster_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(1), filename=raster_path,
            datatype=gdal.GDT_Byte)

        fetched_nodata_value = pygeoprocessing.get_raster_info(
            raster_path)['nodata']

        self.assertEquals(fetched_nodata_value, nodata)


    def test_gdal_to_numpy_type_signedbyte(self):
        """PGP.geoprocessing _gdal_numpy_type gives good byte value."""
        from pygeoprocessing import geoprocessing
        raster_matrix = numpy.empty((5, 5), numpy.int8)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        for row_index in xrange(raster_matrix.shape[1]):
            raster_matrix[row_index, :] = row_index
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [raster_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(1), filename=raster_path,
            datatype=gdal.GDT_Byte)

        raster = gdal.Open(raster_path)
        band = raster.GetRasterBand(1)
        raster = None
        numpy_type = geoprocessing._gdal_to_numpy_type(band)
        band = None
        self.assertEqual(numpy_type, numpy.uint8)

    def test_gdal_to_numpy_type_unsignedbyte(self):
        """PGP.geoprocessing _gdal_numpy_type gives good ubyte value."""
        from pygeoprocessing import geoprocessing
        raster_matrix = numpy.empty((5, 5), numpy.int8)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        for row_index in xrange(raster_matrix.shape[1]):
            raster_matrix[row_index, :] = row_index
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [raster_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(1), filename=raster_path,
            datatype=gdal.GDT_Byte)

        raster = gdal.Open(raster_path, gdal.GA_Update)
        band = raster.GetRasterBand(1)
        raster = None
        band.SetMetadataItem('IMAGE_STRUCTURE', 'PIXELTYPE', 'SIGNEDBYTE')
        numpy_type = geoprocessing._gdal_to_numpy_type(band)
        band = None
        self.assertEqual(numpy_type, numpy.uint8)

    def test_gdal_to_numpy_type_complex(self):
        """PGP.geoprocessing _gdal_numpy_type ValueError on complex type."""
        from pygeoprocessing import geoprocessing
        raster_matrix = numpy.empty((5, 5), numpy.cfloat)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        for row_index in xrange(raster_matrix.shape[1]):
            raster_matrix[row_index, :] = row_index
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [raster_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(1), filename=raster_path,
            datatype=gdal.GDT_CFloat32)

        raster = gdal.Open(raster_path)
        band = raster.GetRasterBand(1)
        raster = None
        with self.assertRaises(ValueError):
            geoprocessing._gdal_to_numpy_type(band)
        band = None

    def test_calculate_slope(self):
        """PGP.geoprocessing: test slope calculation."""
        dem_matrix = numpy.empty((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        for row_index in xrange(dem_matrix.shape[1]):
            dem_matrix[row_index, :] = row_index
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [dem_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(1), filename=dem_path)
        slope_path = os.path.join(self.workspace_dir, 'slope.tif')
        pygeoprocessing.calculate_slope(dem_path, slope_path)

        slope_raster = gdal.Open(slope_path)
        slope_band = slope_raster.GetRasterBand(1)
        slope_array = slope_band.ReadAsArray()
        slope_band = None
        slope_raster = None

        # just check the inner pixels; outer are questionable corner cases
        # that will be addressed in this issue: https://bitbucket.org/richpsharp/pygeoprocessing/issues/67/test-slope-calculation-against-saga-gis
        numpy.testing.assert_array_almost_equal(
            slope_array[1:-1, 1:-1], 100 * numpy.ones((3, 3)))

    def test_distance_transform_mocked_os_remove(self):
        """PGP.geoprocessing: ensure OSError is tolerated."""
        from scipy.ndimage import morphology
        mask_matrix = numpy.ones((500, 500), numpy.float32)
        mask_matrix[:] = 0.0
        mask_matrix[mask_matrix.shape[0] / 2, mask_matrix.shape[1] / 2] = 1.0
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        mask_path = os.path.join(self.workspace_dir, 'mask.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [mask_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=mask_path)
        output_path = os.path.join(self.workspace_dir, 'output.tif')
        with mock.patch.object(
                os, 'remove', return_value=None) as os_remove_mock:
            try:
                os_remove_mock.side_effect = OSError('Mock OSError')
                pygeoprocessing.distance_transform_edt(mask_path, output_path)
            except OSError as error:
                self.fail("Unexpected OSError was raised %s" % error)
        output_raster = gdal.Open(output_path)
        output_band = output_raster.GetRasterBand(1)
        output_array = output_band.ReadAsArray()
        output_band = None
        output_raster = None
        inverse_mask = numpy.ones(mask_matrix.shape, numpy.float32)
        inverse_mask[mask_matrix.shape[0] / 2, mask_matrix.shape[1] / 2] = 0.0

        expected = morphology.distance_transform_edt(inverse_mask)
        numpy.testing.assert_array_almost_equal(output_array, expected)


    def test_distance_transform(self):
        """PGP.geoprocessing: distance transform coverage test."""
        from scipy.ndimage import morphology
        mask_matrix = numpy.ones((500, 500), numpy.float32)
        mask_matrix[:] = 0.0
        mask_matrix[mask_matrix.shape[0] / 2, mask_matrix.shape[1] / 2] = 1.0
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        mask_path = os.path.join(self.workspace_dir, 'mask.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [mask_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=mask_path)
        output_path = os.path.join(self.workspace_dir, 'output.tif')
        pygeoprocessing.distance_transform_edt(mask_path, output_path)
        output_raster = gdal.Open(output_path)
        output_band = output_raster.GetRasterBand(1)
        output_array = output_band.ReadAsArray()
        output_band = None
        output_raster = None
        inverse_mask = numpy.ones(mask_matrix.shape, numpy.float32)
        inverse_mask[mask_matrix.shape[0] / 2, mask_matrix.shape[1] / 2] = 0.0

        expected = morphology.distance_transform_edt(inverse_mask)
        numpy.testing.assert_array_almost_equal(output_array, expected)

    def test_convolve_2d_uri_flip_signal(self):
        """PGP.geoprocessing: convolve 2D case when kernel > signal."""
        signal_matrix = numpy.ones((1, 1), numpy.float32)
        kernel_matrix = numpy.ones((5, 5), numpy.float32)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [signal_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [kernel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=kernel_path)
        output_path = os.path.join(self.workspace_dir, 'output.tif')
        pygeoprocessing.convolve_2d_uri(
            signal_path, kernel_path, output_path)
        output_raster = gdal.Open(output_path)
        output_band = output_raster.GetRasterBand(1)
        output_array = output_band.ReadAsArray()
        output_band = None
        output_raster = None
        self.assertEquals(numpy.sum(output_array), 1)

    def test_convolve_2d_simple_uri(self):
        """PGP.geoprocessing: test convolve 2D with 5x5s."""
        signal_array = numpy.ones([5, 5], numpy.float32)
        kernel_array = numpy.ones([5, 5], numpy.float32)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [signal_array], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [kernel_array], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=kernel_path)

        output_path = os.path.join(self.workspace_dir, 'output.tif')
        pygeoprocessing.convolve_2d_uri(
            signal_path, kernel_path, output_path)

        output_raster = gdal.Open(output_path)
        output_band = output_raster.GetRasterBand(1)
        output_array = output_band.ReadAsArray()
        output_band = None
        output_raster = None
        self.assertEquals(numpy.sum(output_array), 361)

    def test_convolve_2d_uri(self):
        """PGP.geoprocessing: test convolve 2D with large array."""
        signal_array = numpy.ones([225, 250], dtype=numpy.float32)
        kernel_array = numpy.ones([1000, 1000], dtype=numpy.float32)

        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [signal_array], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [kernel_array], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=kernel_path)

        output_path = os.path.join(self.workspace_dir, 'output.tif')
        pygeoprocessing.convolve_2d_uri(
            signal_path, kernel_path, output_path)

        output_raster = gdal.Open(output_path)
        output_band = output_raster.GetRasterBand(1)
        output_array = output_band.ReadAsArray()
        output_band = None
        output_raster = None
        self.assertAlmostEqual(
            numpy.sum(output_array),
            (signal_array.shape[0] * signal_array.shape[1])**2)

    def test_calculate_disjoint_polygon_set(self):
        """PGP.geoprocessing; test disjoing polygon set."""
        reference = sampledata.SRS_COLOMBIA

        polygons = [
            Polygon([
                (reference.origin[0] + reference.pixel_size(30)[0] * 0,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                (reference.origin[0] + reference.pixel_size(30)[0] * 2,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                (reference.origin[0] + reference.pixel_size(30)[0] * 2,
                 reference.origin[1] + reference.pixel_size(30)[1] * 5),
                (reference.origin[0] + reference.pixel_size(30)[0] * 0,
                 reference.origin[1] + reference.pixel_size(30)[1] * 5),
                (reference.origin[0] + reference.pixel_size(30)[0] * 0,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                ]),
            Polygon([
                (reference.origin[0] + reference.pixel_size(30)[0] * 3,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                (reference.origin[0] + reference.pixel_size(30)[0] * 5,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                (reference.origin[0] + reference.pixel_size(30)[0] * 5,
                 reference.origin[1] + reference.pixel_size(30)[1] * 5),
                (reference.origin[0] + reference.pixel_size(30)[0] * 3,
                 reference.origin[1] + reference.pixel_size(30)[1] * 5),
                (reference.origin[0] + reference.pixel_size(30)[0] * 3,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                ]),
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

        aoi_path = os.path.join(self.workspace_dir, 'overlap_aoi.json')
        pygeoprocessing.testing.create_vector_on_disk(
            polygons, reference.projection, filename=aoi_path)

        disjoint_set = pygeoprocessing.calculate_disjoint_polygon_set(
            aoi_path)
        self.assertEquals(len(disjoint_set), 2)
        for pair in [set([0, 1]), set([2])]:
            self.assertTrue(pair in disjoint_set)

    def test_rasterize_layer(self):
        """PGP.geoprocessing: test rasterize layer."""
        pixel_matrix = numpy.empty((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        pixel_matrix[:] = nodata
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_path)

        polygons = [
            Polygon([
                (reference.origin[0] + reference.pixel_size(30)[0] * 0,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                (reference.origin[0] + reference.pixel_size(30)[0] * 4,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                (reference.origin[0] + reference.pixel_size(30)[0] * 4,
                 reference.origin[1] + reference.pixel_size(30)[1] * 4),
                (reference.origin[0] + reference.pixel_size(30)[0] * 0,
                 reference.origin[1] + reference.pixel_size(30)[1] * 4),
                (reference.origin[0] + reference.pixel_size(30)[0] * 0,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                ]),
        ]
        aoi_path = os.path.join(self.workspace_dir, 'aoi')

        pygeoprocessing.testing.create_vector_on_disk(
            polygons, reference.projection, filename=aoi_path,
            fields={'id': 'int'}, attributes=[{'id': 1}])

        pygeoprocessing.rasterize_layer_uri(
            raster_path, aoi_path, option_list=["ATTRIBUTE=ID"])

        raster = gdal.Open(raster_path)
        band = raster.GetRasterBand(1)
        values = band.ReadAsArray()
        band = None
        raster = None

        self.assertAlmostEqual(4**2, numpy.sum(values[values != nodata]))

    def test_reclassify_dataset(self):
        """PGP.geoprocessing: test for reclassify dataset."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        pixel_matrix[2:4:, 2:4] = 2
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        pixel_matrix[0, 0] = nodata
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

        raster_out_uri = os.path.join(self.workspace_dir, 'reclassified.tif')
        value_map = {1: 0.5, 2: 2.5}
        out_datatype = gdal.GDT_Float32
        out_nodata = -1.0

        pygeoprocessing.reclassify_dataset_uri(
            raster_filename, value_map, raster_out_uri, out_datatype,
            out_nodata, exception_flag='values_required',
            assert_dataset_projected=True)

        out_raster = gdal.Open(raster_out_uri)
        out_band = out_raster.GetRasterBand(1)
        out_array = out_band.ReadAsArray()

        self.assertEqual(
            numpy.sum(out_array), 0.5 * 20 + 2.5 * 4 + out_nodata)

    def test_reclassify_dataset_bad_mode(self):
        """PGP.geoprocessing: test bad mode in reclassify dataset."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        pixel_matrix[2:4:, 2:4] = 2
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        pixel_matrix[0, 0] = nodata
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

        raster_out_uri = os.path.join(self.workspace_dir, 'reclassified.tif')
        value_map = {1: 0.5, 2: 2.5}
        out_datatype = gdal.GDT_Float32
        out_nodata = -1.0

        with self.assertRaises(ValueError):
            pygeoprocessing.reclassify_dataset_uri(
                raster_filename, value_map, raster_out_uri, out_datatype,
                out_nodata, exception_flag='bad_mode',
                assert_dataset_projected=True)

    def test_reclassify_dataset_missing_code(self):
        """PGP.geoprocessing: missing lookup code in reclassify dataset."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        pixel_matrix[2:4:, 2:4] = 2
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        pixel_matrix[0, 0] = nodata
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

        raster_out_uri = os.path.join(self.workspace_dir, 'reclassified.tif')
        value_map = {1: 0.5}  # missing an entry for code 2
        out_datatype = gdal.GDT_Float32
        out_nodata = -1.0

        with self.assertRaises(ValueError):
            # there's a missing entry for code 2, should raise a ValueError
            pygeoprocessing.reclassify_dataset_uri(
                raster_filename, value_map, raster_out_uri, out_datatype,
                out_nodata, exception_flag='values_required',
                assert_dataset_projected=True)

    def test_agg_raster_values_ignore_nodata(self):
        """PGP.geoprocessing: test for agg raster values ignore nodata."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        pixel_matrix[0, 0] = nodata
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

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
        aoi_filename = os.path.join(self.workspace_dir, 'aoi')
        pygeoprocessing.testing.create_vector_on_disk(
            polygons, reference.projection, filename=aoi_filename)

        result = pygeoprocessing.aggregate_raster_values_uri(
            raster_filename, aoi_filename, shapefile_field=None,
            ignore_nodata=False, all_touched=False,
            polygons_might_overlap=True)

        # there are 25 pixels fully covered
        self.assertAlmostEqual(result.total[9999], 24)

    def test_agg_raster_values_oserror(self):
        """PGP.geoprocessing: test for agg raster values oserror on remove."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

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
        aoi_filename = os.path.join(self.workspace_dir, 'aoi')
        pygeoprocessing.testing.create_vector_on_disk(
            polygons, reference.projection, filename=aoi_filename)

        with mock.patch.object(
                os, 'remove', return_value=None) as os_remove_mock:
            try:
                os_remove_mock.side_effect = OSError('Mock OSError')
                result = pygeoprocessing.aggregate_raster_values_uri(
                    raster_filename, aoi_filename, shapefile_field=None,
                    ignore_nodata=True, all_touched=False,
                    polygons_might_overlap=True)
                # there are 25 pixels fully covered
                self.assertAlmostEqual(result.total[9999], 25)
            except OSError as error:
                self.fail("Unexpected OSError was raised %s" % error)

    def test_agg_raster_values(self):
        """PGP.geoprocessing: basic unit test for aggregate raster values."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

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
        aoi_filename = os.path.join(self.workspace_dir, 'aoi')
        pygeoprocessing.testing.create_vector_on_disk(
            polygons, reference.projection, filename=aoi_filename)

        result = pygeoprocessing.aggregate_raster_values_uri(
            raster_filename, aoi_filename, shapefile_field=None,
            ignore_nodata=True, all_touched=False,
            polygons_might_overlap=True)

        # there are 25 pixels fully covered
        self.assertAlmostEqual(result.total[9999], 25)

    def test_agg_raster_values_with_id(self):
        """PGP.geoprocessing: aggregate raster values test with feature id."""
        pixel_matrix = numpy.ones((5000, 5000), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        pixel_matrix[0, 0] = nodata
        pixel_matrix[-1, -1] = nodata
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

        polygons = [
            Polygon([
                (reference.origin[0] + reference.pixel_size(30)[0] * 250,
                 reference.origin[1] + reference.pixel_size(30)[1] * 250),
                (reference.origin[0] + reference.pixel_size(30)[0] * 750,
                 reference.origin[1] + reference.pixel_size(30)[1] * 250),
                (reference.origin[0] + reference.pixel_size(30)[0] * 750,
                 reference.origin[1] + reference.pixel_size(30)[1] * 750),
                (reference.origin[0] + reference.pixel_size(30)[0] * 250,
                 reference.origin[1] + reference.pixel_size(30)[1] * 750),
                (reference.origin[0] + reference.pixel_size(30)[0] * 250,
                 reference.origin[1] + reference.pixel_size(30)[1] * 250),
                ]),
        ]
        aoi_filename = os.path.join(self.workspace_dir, 'aoi.json')

        pygeoprocessing.testing.create_vector_on_disk(
            polygons, reference.projection, fields={'id': 'int'},
            attributes=[{'id': 1}], filename=aoi_filename)

        result = pygeoprocessing.aggregate_raster_values_uri(
            raster_filename, aoi_filename, shapefile_field='id',
            ignore_nodata=True, all_touched=False,
            polygons_might_overlap=False)

        self.assertAlmostEqual(result.total[1], 250000)

    def test_agg_raster_values_with_bad_id(self):
        """PGP.geoprocessing: aggregate raster values test with bad id."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

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
        aoi_filename = os.path.join(self.workspace_dir, 'aoi.json')

        pygeoprocessing.testing.create_vector_on_disk(
            polygons, reference.projection, fields={'id': 'int'},
            attributes=[{'id': 1}], filename=aoi_filename)

        with self.assertRaises(AttributeError):
            # bad_id is not defined as an id
            pygeoprocessing.aggregate_raster_values_uri(
                raster_filename, aoi_filename, shapefile_field='bad_id',
                ignore_nodata=True, all_touched=False,
                polygons_might_overlap=False)

    def test_agg_raster_values_with_bad_id_type(self):
        """PGP.geoprocessing: agg raster values test with bad id type."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

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
        aoi_filename = os.path.join(self.workspace_dir, 'aoi.json')

        pygeoprocessing.testing.create_vector_on_disk(
            polygons, reference.projection, fields={'id': 'real'},
            attributes=[{'id': 1.0}], filename=aoi_filename)

        with self.assertRaises(TypeError):
            # 'id' field is a string and that's not an int that can be
            # rasterized
            pygeoprocessing.aggregate_raster_values_uri(
                raster_filename, aoi_filename, shapefile_field='id',
                ignore_nodata=True, all_touched=False,
                polygons_might_overlap=False)

    def test_agg_raster_values_with_overlap(self):
        """PGP.geoprocessing: test agg raster values w/ overlap features."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

        polygons = [
            Polygon([
                (reference.origin[0] + reference.pixel_size(30)[0] * 0,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                (reference.origin[0] + reference.pixel_size(30)[0] * 2,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                (reference.origin[0] + reference.pixel_size(30)[0] * 2,
                 reference.origin[1] + reference.pixel_size(30)[1] * 5),
                (reference.origin[0] + reference.pixel_size(30)[0] * 0,
                 reference.origin[1] + reference.pixel_size(30)[1] * 5),
                (reference.origin[0] + reference.pixel_size(30)[0] * 0,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                ]),
            Polygon([
                (reference.origin[0] + reference.pixel_size(30)[0] * 2,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                (reference.origin[0] + reference.pixel_size(30)[0] * 5,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                (reference.origin[0] + reference.pixel_size(30)[0] * 5,
                 reference.origin[1] + reference.pixel_size(30)[1] * 5),
                (reference.origin[0] + reference.pixel_size(30)[0] * 2,
                 reference.origin[1] + reference.pixel_size(30)[1] * 5),
                (reference.origin[0] + reference.pixel_size(30)[0] * 2,
                 reference.origin[1] + reference.pixel_size(30)[1] * 0),
                ]),
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
        aoi_filename = os.path.join(self.workspace_dir, 'overlap_aoi.json')

        pygeoprocessing.testing.create_vector_on_disk(
            polygons, reference.projection, fields={'id': 'int'},
            attributes=[{'id': 1}, {'id': 2}, {'id': 3}],
            filename=aoi_filename)

        result = pygeoprocessing.aggregate_raster_values_uri(
            raster_filename, aoi_filename, shapefile_field='id',
            ignore_nodata=True, all_touched=False,
            polygons_might_overlap=True)

        self.assertAlmostEqual(result.total[1], 10)
        self.assertAlmostEqual(result.total[2], 15)
        self.assertAlmostEqual(result.total[3], 25)

    def test_align_dataset_list_different_arg_lengths(self):
        """PGP.geoprocessing: align dataset expect error on unequal lists."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_a_filename = os.path.join(self.workspace_dir, 'a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_a_filename)
        pixel_matrix = numpy.ones((15, 15), numpy.int16)
        raster_b_filename = os.path.join(self.workspace_dir, 'b.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_b_filename)
        out_a_filename = os.path.join(self.workspace_dir, 'a_out.tif')
        out_b_filename = os.path.join(self.workspace_dir, 'b_out.tif')

        with self.assertRaises(ValueError):
            # Too few intersection lists
            pygeoprocessing.align_dataset_list(
                [raster_a_filename, raster_b_filename],
                [out_a_filename, out_b_filename], ['nearest'],
                30, 'intersection', 0,
                dataset_to_bound_index=None, aoi_uri=None,
                assert_datasets_projected=True, all_touched=False)

            # Too few input lists
            pygeoprocessing.align_dataset_list(
                [raster_b_filename],
                [out_a_filename, out_b_filename], ['nearest', 'nearest'],
                30, 'intersection', 0,
                dataset_to_bound_index=None, aoi_uri=None,
                assert_datasets_projected=True, all_touched=False)

            # Too few output lists
            pygeoprocessing.align_dataset_list(
                [raster_a_filename, raster_b_filename],
                [out_b_filename], ['nearest', 'nearest'],
                30, 'intersection', 0,
                dataset_to_bound_index=None, aoi_uri=None,
                assert_datasets_projected=True, all_touched=False)

    def test_align_dataset_bad_mode(self):
        """PGP.geoprocessing: align dataset expect error on bad mode."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_a_filename = os.path.join(self.workspace_dir, 'a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_a_filename)
        pixel_matrix = numpy.ones((15, 15), numpy.int16)
        raster_b_filename = os.path.join(self.workspace_dir, 'b.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_b_filename)
        out_a_filename = os.path.join(self.workspace_dir, 'a_out.tif')
        out_b_filename = os.path.join(self.workspace_dir, 'b_out.tif')
        with self.assertRaises(ValueError):
            # intentionally bad mode passed
            pygeoprocessing.align_dataset_list(
                [raster_a_filename, raster_b_filename],
                [out_a_filename, out_b_filename], ['nearest', 'nearest'],
                30, 'bad_mode', 0,
                dataset_to_bound_index=None, aoi_uri=None,
                assert_datasets_projected=True, all_touched=False)

    def test_align_dataset_list_intersection(self):
        """PGP.geoprocessing: double raster align dataset test intersect."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_a_filename = os.path.join(self.workspace_dir, 'a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_a_filename)
        pixel_matrix = numpy.ones((15, 15), numpy.int16)
        raster_b_filename = os.path.join(self.workspace_dir, 'b.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_b_filename)

        out_a_filename = os.path.join(self.workspace_dir, 'a_out.tif')
        out_b_filename = os.path.join(self.workspace_dir, 'b_out.tif')

        pygeoprocessing.align_dataset_list(
            [raster_a_filename, raster_b_filename],
            [out_a_filename, out_b_filename], ['nearest', 'nearest'],
            30, 'intersection', 0,
            dataset_to_bound_index=None, aoi_uri=None,
            assert_datasets_projected=True, all_touched=False)

        # both output rasters should the the same as input 'a'
        pygeoprocessing.testing.assert_rasters_equal(
            raster_a_filename, out_a_filename, rel_tol=1e-9)
        pygeoprocessing.testing.assert_rasters_equal(
            raster_a_filename, out_b_filename, rel_tol=1e-9)

    def test_align_dataset_list_dataset_mode(self):
        """PGP.geoprocessing: raster align dataset test on dataset mode."""
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        pixel_matrix[:] = nodata
        raster_a_filename = os.path.join(self.workspace_dir, 'a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_a_filename)
        pixel_matrix = numpy.ones((15, 15), numpy.int16)
        pixel_matrix[:] = nodata
        raster_b_filename = os.path.join(self.workspace_dir, 'b.tif')
        b_origin = (reference.origin[0] + 30, reference.origin[1] - 30)
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], b_origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_b_filename)

        out_a_filename = os.path.join(self.workspace_dir, 'a_out.tif')
        out_b_filename = os.path.join(self.workspace_dir, 'b_out.tif')

        pygeoprocessing.align_dataset_list(
            [raster_a_filename, raster_b_filename],
            [out_a_filename, out_b_filename], ['nearest', 'nearest'],
            30, 'dataset', 0,
            dataset_to_bound_index=1, aoi_uri=None,
            assert_datasets_projected=True, all_touched=False)

        # both output rasters should the the same as input 'a'
        pygeoprocessing.testing.assert_rasters_equal(
            raster_b_filename, out_a_filename, rel_tol=1e-9)
        pygeoprocessing.testing.assert_rasters_equal(
            raster_b_filename, out_b_filename, rel_tol=1e-9)

    def test_align_dataset_list_dataset_bad_index(self):
        """PGP.geoprocessing: align dataset expect error on bad dset index."""
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        pixel_matrix[:] = nodata
        raster_a_filename = os.path.join(self.workspace_dir, 'a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_a_filename)
        pixel_matrix = numpy.ones((15, 15), numpy.int16)
        pixel_matrix[:] = nodata
        raster_b_filename = os.path.join(self.workspace_dir, 'b.tif')
        b_origin = (reference.origin[0] + 30, reference.origin[1] - 30)
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], b_origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_b_filename)

        out_a_filename = os.path.join(self.workspace_dir, 'a_out.tif')
        out_b_filename = os.path.join(self.workspace_dir, 'b_out.tif')

        with self.assertRaises(ValueError):
            # reference dataset 3 which doesn't exist
            pygeoprocessing.align_dataset_list(
                [raster_a_filename, raster_b_filename],
                [out_a_filename, out_b_filename], ['nearest', 'nearest'],
                30, 'dataset', 0,
                dataset_to_bound_index=3, aoi_uri=None,
                assert_datasets_projected=True, all_touched=False)

    def test_align_dataset_list_non_aligned_rasters(self):
        """PGP.geoprocessing: align dataset error on misaligned rasters."""
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        pixel_matrix[:] = nodata
        raster_a_filename = os.path.join(self.workspace_dir, 'a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_a_filename)
        pixel_matrix = numpy.ones((15, 15), numpy.int16)
        pixel_matrix[:] = nodata
        raster_b_filename = os.path.join(self.workspace_dir, 'b.tif')
        b_origin = (reference.origin[0] + 3000, reference.origin[1] - 3000)
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], b_origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_b_filename)

        out_a_filename = os.path.join(self.workspace_dir, 'a_out.tif')
        out_b_filename = os.path.join(self.workspace_dir, 'b_out.tif')

        with self.assertRaises(ValueError):
            # raster a and b don't overlap
            pygeoprocessing.align_dataset_list(
                [raster_a_filename, raster_b_filename],
                [out_a_filename, out_b_filename], ['nearest', 'nearest'],
                30, 'intersection', 0,
                dataset_to_bound_index=None, aoi_uri=None,
                assert_datasets_projected=True, all_touched=False)

    def test_align_dataset_list_union(self):
        """PGP.geoprocessing: double raster align dataset test intersect."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_a_filename = os.path.join(self.workspace_dir, 'a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_a_filename)
        pixel_matrix = numpy.ones((15, 15), numpy.int16)
        # slice up the matrix so that it matches what a.tif will look like
        # when extended to b's range
        pixel_matrix[5:, :] = nodata
        pixel_matrix[:, 5:] = nodata
        raster_b_filename = os.path.join(self.workspace_dir, 'b.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_b_filename)

        out_a_filename = os.path.join(self.workspace_dir, 'a_out.tif')
        out_b_filename = os.path.join(self.workspace_dir, 'b_out.tif')

        pygeoprocessing.align_dataset_list(
            [raster_a_filename, raster_b_filename],
            [out_a_filename, out_b_filename], ['nearest', 'nearest'],
            30, 'union', 0,
            dataset_to_bound_index=None, aoi_uri=None,
            assert_datasets_projected=True, all_touched=False)

        # both output rasters should the the same as input 'a'
        pygeoprocessing.testing.assert_rasters_equal(
            raster_b_filename, out_a_filename, rel_tol=1e-9)
        pygeoprocessing.testing.assert_rasters_equal(
            raster_b_filename, out_b_filename, rel_tol=1e-9)

    def test_vect_datasets_bad_filelist(self):
        """PGP.geoprocessing: vect..._datasets expected error for non-list."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

        out_filename = pygeoprocessing.temporary_filename()
        with self.assertRaises(ValueError):
            # intentionally passing a filename rather than a list of files
            # to get an expected exception
            pygeoprocessing.vectorize_datasets(
                raster_filename, lambda x: x, out_filename,
                gdal.GDT_Int32, nodata, 30, 'intersection')

    def test_vect_datasets_output_alias(self):
        """PGP.geoprocessing: vect..._datasets expected error for aliasing."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

        with self.assertRaises(ValueError):
            # intentionally passing a filename rather than a list of files
            # to get an expected exception
            pygeoprocessing.vectorize_datasets(
                [raster_filename], lambda x: x, raster_filename,
                gdal.GDT_Int32, nodata, 30, 'intersection')

    def test_vec_datasets_oserror(self):
        """PGP.geoprocessing: vec_datasets os.remove OSError handling."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

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
        aoi_filename = os.path.join(self.workspace_dir, 'aoi')
        pygeoprocessing.testing.create_vector_on_disk(
            polygons, reference.projection, filename=aoi_filename)

        out_filename = os.path.join(self.workspace_dir, 'out.tif')
        with mock.patch.object(
                os, 'remove', return_value=None) as os_remove_mock:
            try:
                os_remove_mock.side_effect = OSError('Mock OSError')
                pygeoprocessing.vectorize_datasets(
                    [raster_filename], lambda x: x, out_filename,
                    gdal.GDT_Int32, nodata, 30, 'intersection',
                    aoi_uri=aoi_filename)
            except OSError as error:
                self.fail("Unexpected OSError was raised %s" % error)

        pygeoprocessing.testing.assert_rasters_equal(
            raster_filename, out_filename, rel_tol=1e-9)

    def test_vect_datasets_bad_bbs(self):
        """PGP.geoprocessing: vect..._datasets expected error on bad BBox."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

        out_filename = pygeoprocessing.temporary_filename()
        with self.assertRaises(ValueError):
            # intentionally passing a filename rather than a list of files
            # to get an expected exception
            pygeoprocessing.vectorize_datasets(
                [raster_filename], lambda x: x, out_filename,
                gdal.GDT_Int32, nodata, 30, 'bad_mode')

    def test_vect_datasets_identity(self):
        """PGP.geoprocessing: vectorize_datasets f(x)=x."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

        out_filename = os.path.join(self.workspace_dir, 'out.tif')
        pygeoprocessing.vectorize_datasets(
            [raster_filename], lambda x: x, out_filename, gdal.GDT_Int32,
            nodata, 30, 'intersection')

        pygeoprocessing.testing.assert_rasters_equal(
            raster_filename, out_filename, rel_tol=1e-9)

    def test_vect_datasets_identity_aoi(self):
        """PGP.geoprocessing: vectorize_datasets f(x)=x with AOI."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata = -1
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename)

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
        aoi_filename = os.path.join(self.workspace_dir, 'aoi')
        pygeoprocessing.testing.create_vector_on_disk(
            polygons, reference.projection, filename=aoi_filename)

        out_filename = os.path.join(self.workspace_dir, 'out.tif')
        pygeoprocessing.vectorize_datasets(
            [raster_filename], lambda x: x, out_filename, gdal.GDT_Int32,
            nodata, 30, 'intersection', aoi_uri=aoi_filename)

        pygeoprocessing.testing.assert_rasters_equal(
            raster_filename, out_filename, rel_tol=1e-9)

    def test_iterblocks(self):
        """PGP.geoprocessing: Sum a 1000**2 raster using iterblocks."""
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename,
            dataset_opts=['TILED=YES'])

        raster_sum = 0
        for _, memblock in pygeoprocessing.iterblocks(
                raster_filename):
            raster_sum += memblock.sum()

        self.assertEqual(raster_sum, 1000000)

    def test_iterblocks_astype(self):
        """PGP.geoprocessing: test iterblocks astype flag."""
        pixel_matrix = numpy.empty((1000, 1000))
        pixel_matrix[:] = 1.4
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename,
            dataset_opts=['TILED=YES'])

        raster_sum = 0
        for _, memblock in pygeoprocessing.iterblocks(
                raster_filename, astype=numpy.int):
            raster_sum += memblock.sum()

        self.assertEqual(raster_sum, 1000000)

    def test_iterblocks_multiband(self):
        """PGP.geoprocessing: multiband iterblocks on identical blocks."""
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        # double one value so we can ensure we're getting out different bands
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix, 2 * pixel_matrix], reference.origin,
            reference.projection, nodata,
            reference.pixel_size(30), filename=raster_filename,
            dataset_opts=['TILED=YES'])

        for _, band_1_block, band_2_block in \
                pygeoprocessing.iterblocks(raster_filename):
            numpy.testing.assert_almost_equal(band_1_block * 2, band_2_block)

    def test_default_blocksizes_tiled(self):
        """PGP.geoprocessing: Verify block size is set on default tilesize."""
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), dataset_opts=['TILED=YES'],
            filename=raster_filename)

        raster = gdal.Open(raster_filename)
        band = raster.GetRasterBand(1)
        block_size = band.GetBlockSize()
        band = None
        raster = None

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
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), dataset_opts=['TILED=NO'],
            filename=raster_filename)

        raster = gdal.Open(raster_filename)
        band = raster.GetRasterBand(1)
        block_size = band.GetBlockSize()
        band = None
        raster = None

        # If a raster is forced to be un-tiled (striped), the raster's blocks
        # will be accessed line-by-line.
        self.assertEqual(block_size[0], 1000)  # 1000 is num. columns
        self.assertEqual(block_size[1], 1)  # block is 1 pixel tall

    def test_custom_blocksizes(self):
        """PGP.geoprocessing:  Verify that custom block size is set."""
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection, nodata,
            reference.pixel_size(30), dataset_opts=[
                'TILED=YES', 'BLOCKXSIZE=128', 'BLOCKYSIZE=256'],
            filename=raster_filename)
        raster = gdal.Open(raster_filename)
        band = raster.GetRasterBand(1)
        block_size = band.GetBlockSize()
        band = None
        raster = None

        self.assertEqual(block_size, [128, 256])

    def test_custom_blocksizes_multiband(self):
        """PGP.geoprocessing:  Verify block sizes are set on multibands."""
        pixel_matrix = numpy.ones((1000, 1000))
        nodata = 0
        reference = sampledata.SRS_COLOMBIA
        raster_filename = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix, pixel_matrix], reference.origin,
            reference.projection, nodata, reference.pixel_size(30),
            dataset_opts=['TILED=YES', 'BLOCKXSIZE=128', 'BLOCKYSIZE=256'],
            filename=raster_filename)

        for band_index in [1, 2]:
            raster = gdal.Open(raster_filename)
            band = raster.GetRasterBand(band_index)
            # Not sure why the BlockSize is a band attribute, as it's set
            # at the dataset level.
            block_size = band.GetBlockSize()
            band = None
            raster = None
            self.assertEqual(block_size, [128, 256])

if __name__ == '__main__':
    unittest.main()
