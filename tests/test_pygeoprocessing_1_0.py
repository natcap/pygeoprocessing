"""PyGeoprocessing 1.0 test suite."""
import time
import tempfile
import os
import unittest
import shutil

from osgeo import gdal
import numpy
import pygeoprocessing
import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata
import pygeoprocessing.routing
import shapely.geometry

class PyGeoprocessing10(unittest.TestCase):
    """Tests for the PyGeoprocesing 1.0 refactor."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_interpolate_points(self):
        """PGP.geoprocessing: test interpolate points feature."""
        # construct a point shapefile
        reference = sampledata.SRS_COLOMBIA
        point_a = shapely.geometry.Point(
            reference.origin[0] + reference.pixel_size(30)[0] * 9 / 2,
            reference.origin[1])
        point_b = shapely.geometry.Point(
            reference.origin[0] + reference.pixel_size(30)[0] * 9 / 2,
            reference.origin[1] + reference.pixel_size(30)[1] * 9)
        source_vector_path = os.path.join(self.workspace_dir, 'sample_vector')
        pygeoprocessing.testing.create_vector_on_disk(
            [point_a, point_b], reference.projection, fields={'value': 'int'},
            attributes=[{'value': 0}, {'value': 1}], vector_format='GeoJSON',
            filename=source_vector_path)
        # construct a raster
        pixel_matrix = numpy.ones((9, 9), numpy.float32)
        nodata_target = -1
        result_path = os.path.join(self.workspace_dir, 'result.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=result_path)

        # interpolate
        pygeoprocessing.interpolate_points(
            source_vector_path, 'value', (result_path, 1), 'nearest')

        # verify that result is expected
        result_raster = gdal.Open(result_path)
        result_band = result_raster.GetRasterBand(1)
        result_array = result_band.ReadAsArray()
        result_band = None
        result_raster = None

        # we expect the first 4 rows to be 0, then the last ones to be 1
        expected_result = numpy.ones((9, 9), numpy.float32)
        expected_result[:5, :] = 0

        numpy.testing.assert_array_equal(result_array, expected_result)

    def test_invoke_timed_callback(self):
        """PGP.geoprocessing: cover a timed callback."""
        import pygeoprocessing.geoprocessing
        reference_time = time.time()
        time.sleep(0.1)
        new_time = pygeoprocessing.geoprocessing._invoke_timed_callback(
            reference_time, lambda: None, 0.05)
        self.assertNotEqual(reference_time, new_time)

    def test_warp_raster(self):
        """PGP.geoprocessing: align/resize raster test reprojection."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_a_path)

        target_raster_path = os.path.join(self.workspace_dir, 'target_a.tif')
        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        pygeoprocessing.warp_raster(
            base_a_path, base_a_raster_info['pixel_size'], target_raster_path,
            'nearest', target_sr_wkt=reference.projection,
            gtiff_creation_options=['TILED=NO'])

        pygeoprocessing.testing.assert_rasters_equal(
            base_a_path, target_raster_path)

    def test_align_and_resize_raster_stack_bad_lengths(self):
        """PGP.geoprocessing: align/resize raster bad list lengths."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_a_path)

        base_raster_path_list = [base_a_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b']]

        resample_method_list = ['nearest'] * 2
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        with self.assertRaises(ValueError):
            # here base_raster_path_list is length 1 but others are length 2
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, target_raster_path_list,
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)

    def test_align_and_resize_raster_stack_bad_mode(self):
        """PGP.geoprocessing: align/resize raster bad bounding box mode."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_a_path)

        base_raster_path_list = [base_a_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_a.tif')]

        resample_method_list = ['nearest']
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
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_a_path)

        base_raster_path_list = [base_a_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_a.tif')]

        resample_method_list = ['nearest']
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
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_a_path)

        pixel_b_matrix = numpy.ones((15, 15), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_b_path = os.path.join(self.workspace_dir, 'base_b.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_b_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(60), filename=base_b_path)

        base_raster_path_list = [base_a_path, base_b_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b']]

        resample_method_list = ['nearest'] * 2
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list, target_raster_path_list,
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            base_vector_path_list=None, raster_align_index=0)

        for raster_index in xrange(2):
            target_raster_info = pygeoprocessing.get_raster_info(
                target_raster_path_list[raster_index])
            target_raster = gdal.Open(target_raster_path_list[raster_index])
            target_band = target_raster.GetRasterBand(1)
            target_array = target_band.ReadAsArray()
            numpy.testing.assert_array_equal(pixel_a_matrix, target_array)
            self.assertEqual(
                target_raster_info['pixel_size'],
                base_a_raster_info['pixel_size'])

    def test_align_and_resize_raster_stack_int_with_vectors(self):
        """PGP.geoprocessing: align/resize raster test inters. w/ vectors."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_a_path)

        pixel_b_matrix = numpy.ones((15, 15), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_b_path = os.path.join(self.workspace_dir, 'base_b.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_b_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(60), filename=base_b_path)

        base_raster_path_list = [base_a_path, base_b_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b']]

        resample_method_list = ['nearest'] * 2
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        # make a vector whose bounding box is 1 pixel large
        point_a = shapely.geometry.Point(
            reference.origin[0], reference.origin[1])
        point_b = shapely.geometry.Point(
            reference.origin[0] + reference.pixel_size(30)[0],
            reference.origin[1] + reference.pixel_size(30)[1])
        single_pixel_path = os.path.join(self.workspace_dir, 'single_pixel')
        pygeoprocessing.testing.create_vector_on_disk(
            [point_a, point_b], reference.projection, fields={'value': 'int'},
            attributes=[{'value': 0}, {'value': 1}], vector_format='GeoJSON',
            filename=single_pixel_path)

        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list, target_raster_path_list,
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            raster_align_index=0, base_vector_path_list=[single_pixel_path])

        expected_matrix = numpy.ones((1, 1), numpy.int16)
        for raster_index in xrange(2):
            target_raster_info = pygeoprocessing.get_raster_info(
                target_raster_path_list[raster_index])
            target_raster = gdal.Open(target_raster_path_list[raster_index])
            target_band = target_raster.GetRasterBand(1)
            target_array = target_band.ReadAsArray()
            numpy.testing.assert_array_equal(expected_matrix, target_array)
            self.assertEqual(
                target_raster_info['pixel_size'],
                base_a_raster_info['pixel_size'])

    def test_align_and_resize_raster_stack_no_overlap(self):
        """PGP.geoprocessing: align/resize raster no intersection error."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix],
            [reference.origin[0]-10*30, reference.origin[1]+10*30],
            reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_a_path)

        pixel_b_matrix = numpy.ones((15, 15), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_b_path = os.path.join(self.workspace_dir, 'base_b.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_b_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(60), filename=base_b_path)

        base_raster_path_list = [base_a_path, base_b_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b']]

        resample_method_list = ['nearest'] * 2
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        # make a vector whose bounding box is 1 pixel large
        point_a = shapely.geometry.Point(
            reference.origin[0], reference.origin[1])
        point_b = shapely.geometry.Point(
            reference.origin[0] + reference.pixel_size(30)[0],
            reference.origin[1] + reference.pixel_size(30)[1])
        single_pixel_path = os.path.join(self.workspace_dir, 'single_pixel')
        pygeoprocessing.testing.create_vector_on_disk(
            [point_a, point_b], reference.projection, fields={'value': 'int'},
            attributes=[{'value': 0}, {'value': 1}], vector_format='GeoJSON',
            filename=single_pixel_path)

        with self.assertRaises(ValueError):
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, target_raster_path_list,
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                raster_align_index=0, base_vector_path_list=[single_pixel_path])

    def test_align_and_resize_raster_stack_union(self):
        """PGP.geoprocessing: align/resize raster test union."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_a_path)

        pixel_b_matrix = numpy.ones((10, 10), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_b_path = os.path.join(self.workspace_dir, 'base_b.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_b_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(60), filename=base_b_path)

        base_raster_path_list = [base_a_path, base_b_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b']]

        resample_method_list = ['nearest'] * 2
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
        expected_matrix_a[5:, :] = nodata_target
        expected_matrix_a[:, 5:] = nodata_target

        target_raster = gdal.Open(target_raster_path_list[0])
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        numpy.testing.assert_array_equal(expected_matrix_a, target_array)

    def test_align_and_resize_raster_stack_bb(self):
        """PGP.geoprocessing: align/resize raster test bounding box."""
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_a_path)

        pixel_b_matrix = numpy.ones((10, 10), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_b_path = os.path.join(self.workspace_dir, 'base_b.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_b_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(60), filename=base_b_path)

        base_raster_path_list = [base_a_path, base_b_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b']]

        resample_method_list = ['nearest'] * 2
        # format is xmin,ymin,xmax,ymax; since y pixel size is negative it
        # goes first in the following bounding box construction
        bounding_box_mode = 'bb=[%d,%d,%d,%d]' % (
            reference.origin[0],
            reference.origin[1] + reference.pixel_size(30)[1] * 5,
            reference.origin[0] + reference.pixel_size(30)[0] * 5,
            reference.origin[1])

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list, target_raster_path_list,
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            base_vector_path_list=None, raster_align_index=0)

        # we expect this to be twice as big since second base raster has a
        # pixel size twice that of the first.
        target_raster = gdal.Open(target_raster_path_list[0])
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        numpy.testing.assert_array_equal(pixel_a_matrix, target_array)

    def test_raster_calculator(self):
        """PGP.geoprocessing: raster_calculator identity test."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_path)

        target_path = os.path.join(
            self.workspace_dir, 'target.tif')
        pygeoprocessing.raster_calculator(
            [(base_path, 1)], lambda x: x, target_path,
            gdal.GDT_Int32, nodata_target, calc_raster_stats=True)
        pygeoprocessing.testing.assert_rasters_equal(base_path, target_path)

    def test_raster_calculator_no_path(self):
        """PGP.geoprocessing: raster_calculator raise ex. on bad file path."""
        nodata_target = -1
        nonexistant_path = os.path.join(self.workspace_dir, 'nofile.tif')
        target_path = os.path.join(
            self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError):
            pygeoprocessing.raster_calculator(
                [(nonexistant_path, 1)], lambda x: x, target_path,
                gdal.GDT_Int32, nodata_target, calc_raster_stats=True)

    def test_raster_calculator_nodata(self):
        """PGP.geoprocessing: raster_calculator test with all nodata."""
        pixel_matrix = numpy.empty((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        pixel_matrix[:] = nodata_target
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_path)

        target_path = os.path.join(
            self.workspace_dir, 'target.tif')
        pygeoprocessing.raster_calculator(
            [(base_path, 1)], lambda x: x, target_path,
            gdal.GDT_Int32, nodata_target, calc_raster_stats=True)
        pygeoprocessing.testing.assert_rasters_equal(base_path, target_path)

    def test_rs_calculator_output_alias(self):
        """PGP.geoprocessing: rs_calculator expected error for aliasing."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_base = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_base, reference.pixel_size(30), filename=base_path)

        with self.assertRaises(ValueError):
            # intentionally passing target path as base path to raise error
            pygeoprocessing.raster_calculator(
                [(base_path, 1)], lambda x: x, base_path,
                gdal.GDT_Int32, nodata_base, calc_raster_stats=True)

    def test_rs_calculator_bad_overlap(self):
        """PGP.geoprocessing: rs_calculator expected error on bad overlap."""
        pixel_matrix_a = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_base = -1
        base_path_a = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix_a], reference.origin, reference.projection,
            nodata_base, reference.pixel_size(30), filename=base_path_a)

        pixel_matrix_b = numpy.ones((4, 5), numpy.int16)
        base_path_b = os.path.join(self.workspace_dir, 'base_b.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix_b], reference.origin, reference.projection,
            nodata_base, reference.pixel_size(30), filename=base_path_b)

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError):
            # intentionally passing a filename rather than a list of files
            # to get an expected exception
            pygeoprocessing.raster_calculator(
                [(base_path_a, 1), (base_path_b, 1)], lambda x: x,
                target_path, gdal.GDT_Int32, nodata_base,
                gtiff_creation_options=None, calc_raster_stats=True)

    def test_new_raster_from_base_unsigned_byte(self):
        """PGP.geoprocessing: test that signed byte rasters copy over."""
        pixel_matrix = numpy.ones((5, 5), numpy.byte)
        pixel_matrix[0, 0] = 255  # 255 ubyte is -1 byte
        reference = sampledata.SRS_COLOMBIA
        nodata_base = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_base, reference.pixel_size(30), datatype=gdal.GDT_Byte,
            filename=base_path,
            dataset_opts=['PIXELTYPE=SIGNEDBYTE'])

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        # 255 should convert to -1 with signed bytes
        pygeoprocessing.new_raster_from_base(
            base_path, target_path, gdal.GDT_Byte, [-1],
            fill_value_list=[255],
            gtiff_creation_options=['PIXELTYPE=SIGNEDBYTE'])

        target_raster = gdal.Open(target_path)
        target_band = target_raster.GetRasterBand(1)
        target_matrix = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        # we expect a negative result even though we put in a positive because
        # we know signed bytes will convert
        self.assertEqual(target_matrix[0, 0], -1)

    def test_calculate_raster_stats_empty(self):
        """PGP.geoprocessing: test empty rasters don't calculate stats."""
        pixel_matrix = numpy.ones((5, 5), numpy.byte)
        pixel_matrix[0, 0] = 255  # 255 ubyte is -1 byte
        reference = sampledata.SRS_COLOMBIA
        nodata_base = -1
        pixel_matrix[:] = nodata_base
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_base, reference.pixel_size(30), datatype=gdal.GDT_Byte,
            filename=base_path, dataset_opts=['PIXELTYPE=SIGNEDBYTE'])

        # this used to cause an error to be printed, now it won't though it
        # doesn't bother setting any values in the raster
        pygeoprocessing.calculate_raster_stats(base_path)
        self.assertTrue(True)
