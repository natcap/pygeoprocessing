"""PyGeoprocessing 1.0 test suite."""

import tempfile
import os
import unittest
import shutil

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
from shapely.geometry import Polygon
import pygeoprocessing
import pygeoprocessing.testing
from pygeoprocessing.testing import sampledata
import pygeoprocessing.routing


class PyGeoprocessing10(unittest.TestCase):
    """Tests for the PyGeoprocesing 1.0 refactor."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_align_and_resize_raster_stack_int(self):
        """PGP.geoprocessing; align/resize raster test intersection."""
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

    def test_align_and_resize_raster_stack_union(self):
        """PGP.geoprocessing; align/resize raster test union."""
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
        """PGP.geoprocessing; align/resize raster test bounding box."""
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
        bounding_box_mode = 'bb=[%d,%d,%d,%d]' % (
            reference.origin[0], reference.origin[1],
            reference.origin[0] + reference.pixel_size(30)[0] * 5,
            reference.origin[1] + reference.pixel_size(30)[1] * 5)

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
            gdal.GDT_Int32, nodata_target, dataset_options=None,
            calc_raster_stats=True)
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
                gdal.GDT_Int32, nodata_base, dataset_options=None,
                calc_raster_stats=True)

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
            nodata_base, reference.pixel_size(30), filename=base_path_a)

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError):
            # intentionally passing a filename rather than a list of files
            # to get an expected exception
            pygeoprocessing.raster_calculator(
                [(base_path_a, 1), (base_path_b, 1)], lambda x: x,
                target_path, gdal.GDT_Int32, nodata_base,
                dataset_options=None, calc_raster_stats=True)
