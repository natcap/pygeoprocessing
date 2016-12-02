"""PyGeoprocessing 1.0 test suite."""

import tempfile
import os
import unittest
import shutil

import mock
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

    def test_raster_stack_calculator(self):
        """PGP.geoprocessing: raster_stack_calculator identity test."""
        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_path)

        target_path = os.path.join(
            self.workspace_dir, 'target.tif')
        pygeoprocessing.raster_stack_calculator(
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
            pygeoprocessing.raster_stack_calculator(
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
            pygeoprocessing.raster_stack_calculator(
                [(base_path_a, 1), (base_path_b, 1)], lambda x: x,
                target_path, gdal.GDT_Int32, nodata_base,
                dataset_options=None, calc_raster_stats=True)
