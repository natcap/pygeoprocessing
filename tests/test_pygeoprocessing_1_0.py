"""PyGeoprocessing 1.0 test suite."""
import time
import tempfile
import os
import unittest
import shutil

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
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

    def test_reclassify_raster_missing_pixel_value(self):
        """PGP.geoprocessing: test reclassify raster with missing value."""
        reference = sampledata.SRS_COLOMBIA
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        pixel_matrix[-1, 0] = test_value - 1  # making a bad value
        nodata_target = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path)

        value_map = {
            test_value: 100,
        }
        target_nodata = -1
        with self.assertRaises(ValueError):
            pygeoprocessing.reclassify_raster(
                raster_path, value_map, target_path, gdal.GDT_Float32,
                target_nodata, exception_flag='values_required', band_index=1)

    def test_reclassify_raster_bad_mode(self):
        """PGP.geoprocessing: test reclassify raster with bad flag."""
        reference = sampledata.SRS_COLOMBIA
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        nodata_target = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path)

        value_map = {
            test_value: 100,
        }
        target_nodata = -1
        with self.assertRaises(ValueError):
            pygeoprocessing.reclassify_raster(
                raster_path, value_map, target_path, gdal.GDT_Float32,
                target_nodata, exception_flag='BAD FLAG', band_index=1)

    def test_reclassify_raster(self):
        """PGP.geoprocessing: test reclassify raster."""
        reference = sampledata.SRS_COLOMBIA
        n_pixels = 9
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        nodata_target = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path)

        value_map = {
            test_value: 100,
        }
        target_nodata = -1
        pygeoprocessing.reclassify_raster(
            raster_path, value_map, target_path, gdal.GDT_Float32,
            target_nodata, exception_flag='values_required', band_index=1)
        target_raster = gdal.Open(target_path)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        self.assertEqual(
            numpy.sum(target_array), n_pixels**2 * value_map[test_value])

    def test_reproject_vector(self):
        """PGP.geoprocessing: test reproject vector."""
        reference = sampledata.SRS_WILLAMETTE
        pixel_size = 30.0
        n_pixels = 9
        polygon_a = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.json')
        aggregate_field_name = 'id'
        pygeoprocessing.testing.create_vector_on_disk(
            [polygon_a], reference.projection,
            fields={'id': 'int'}, attributes=[
                {aggregate_field_name: 0}],
            vector_format='GeoJSON', filename=base_vector_path)

        target_reference = osr.SpatialReference()
        # UTM zone 18N
        target_reference.ImportFromEPSG(26918)

        target_vector_path = os.path.join(
            self.workspace_dir, 'target_vector.shp')
        # create the file first so the model needs to deal with that
        target_file = open(target_vector_path, 'w')
        target_file.close()
        pygeoprocessing.reproject_vector(
            base_vector_path, target_reference.ExportToWkt(),
            target_vector_path, layer_index=0)

        vector = ogr.Open(target_vector_path)
        layer = vector.GetLayer()
        result_reference = layer.GetSpatialRef()
        layer = None
        vector = None
        self.assertTrue(
            osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                osr.SpatialReference(target_reference.ExportToWkt())))


    def test_zonal_statistics(self):
        """PGP.geoprocessing: test zonal stats function."""
        # create aggregating polygon
        reference = sampledata.SRS_COLOMBIA
        pixel_size = 30.0
        n_pixels = 9
        polygon_a = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        polygon_b = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size+reference.origin[1]),
            (reference.origin[0]+pixel_size, -pixel_size+reference.origin[1]),
            (reference.origin[0]+pixel_size, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector')
        aggregate_field_name = 'id'
        pygeoprocessing.testing.create_vector_on_disk(
            [polygon_a, polygon_b], reference.projection,
            fields={'id': 'int'}, attributes=[
                {aggregate_field_name: 0}, {aggregate_field_name: 1}],
            vector_format='GeoJSON', filename=aggregating_vector_path)
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        nodata_target = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path)
        result = pygeoprocessing.zonal_statistics(
            (raster_path, 1), aggregating_vector_path,
            aggregate_field_name, aggregate_layer_name=None,
            ignore_nodata=True, all_touched=False,
            polygons_might_overlap=True)
        expected_result = {
            0: {
                'count': 81,
                'max': 1.0,
                'min': 1.0,
                'nodata_count': 0,
                'sum': 81.0},
            1: {
                'count': 1,
                'max': 1.0,
                'min': 1.0,
                'nodata_count': 0,
                'sum': 1.0}}
        self.assertEqual(result, expected_result)

    def test_zonal_statistics_nodata(self):
        """PGP.geoprocessing: test zonal stats function with non-overlap."""
        # create aggregating polygon
        reference = sampledata.SRS_COLOMBIA
        pixel_size = 30.0
        n_pixels = 9
        polygon_a = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector')
        aggregate_field_name = 'id'
        pygeoprocessing.testing.create_vector_on_disk(
            [polygon_a], reference.projection,
            fields={'id': 'int'}, attributes=[
                {aggregate_field_name: 0}],
            vector_format='GeoJSON', filename=aggregating_vector_path)
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        nodata_target = -1
        pixel_matrix[:] = nodata_target
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path)
        result = pygeoprocessing.zonal_statistics(
            (raster_path, 1), aggregating_vector_path,
            aggregate_field_name, aggregate_layer_name=None,
            ignore_nodata=True, all_touched=False,
            polygons_might_overlap=False)
        expected_result = {
            0: {
                'count': 0,
                'max': None,
                'min': None,
                'nodata_count': 81,
                'sum': 0.0}}
        self.assertEqual(result, expected_result)

    def test_zonal_statistics_named_layer(self):
        """PGP.geoprocessing: test zonal stats with named layer."""
        # create aggregating polygon
        reference = sampledata.SRS_COLOMBIA
        pixel_size = 30.0
        n_pixels = 9
        polygon_a = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector.shp')
        aggregate_field_name = 'id'
        pygeoprocessing.testing.create_vector_on_disk(
            [polygon_a], reference.projection,
            fields={'id': 'int'}, attributes=[
                {aggregate_field_name: 0}],
            vector_format='ESRI Shapefile', filename=aggregating_vector_path)
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        nodata_target = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path)
        result = pygeoprocessing.zonal_statistics(
            (raster_path, 1), aggregating_vector_path,
            aggregate_field_name, aggregate_layer_name='aggregate_vector',
            ignore_nodata=True, all_touched=False,
            polygons_might_overlap=True)
        expected_result = {
            0: {
                'count': 81,
                'max': 1.0,
                'min': 1.0,
                'nodata_count': 0,
                'sum': 81.0}}
        self.assertEqual(result, expected_result)

    def test_zonal_statistics_missing_id(self):
        """PGP.geoprocessing: test zonal stats function with missing id."""
        # create aggregating polygon
        reference = sampledata.SRS_COLOMBIA
        pixel_size = 30.0
        n_pixels = 9
        polygon_a = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        polygon_b = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size+reference.origin[1]),
            (reference.origin[0]+pixel_size, -pixel_size+reference.origin[1]),
            (reference.origin[0]+pixel_size, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector')
        aggregate_field_name = 'id'
        pygeoprocessing.testing.create_vector_on_disk(
            [polygon_a, polygon_b], reference.projection,
            fields={'id': 'int'}, attributes=[
                {aggregate_field_name: 0}, {aggregate_field_name: 1}],
            vector_format='GeoJSON', filename=aggregating_vector_path)
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        nodata_target = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path)
        with self.assertRaises(ValueError):
            _ = pygeoprocessing.zonal_statistics(
                (raster_path, 1), aggregating_vector_path,
                'BAD ID', aggregate_layer_name=None,
                ignore_nodata=True, all_touched=False,
                polygons_might_overlap=False)

    def test_zonal_statistics_bad_aggregate_type(self):
        """PGP.geoprocessing: test zonal stats function with bad agg type."""
        # create aggregating polygon
        reference = sampledata.SRS_COLOMBIA
        pixel_size = 30.0
        n_pixels = 9
        polygon_a = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        polygon_b = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size+reference.origin[1]),
            (reference.origin[0]+pixel_size, -pixel_size+reference.origin[1]),
            (reference.origin[0]+pixel_size, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector')
        aggregate_field_name = 'id'
        pygeoprocessing.testing.create_vector_on_disk(
            [polygon_a, polygon_b], reference.projection,
            fields={'id': 'string'}, attributes=[
                {aggregate_field_name: '0'}, {aggregate_field_name: '1'}],
            vector_format='GeoJSON', filename=aggregating_vector_path)
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        nodata_target = -1
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path)
        with self.assertRaises(TypeError):
            _ = pygeoprocessing.zonal_statistics(
                (raster_path, 1), aggregating_vector_path,
                aggregate_field_name, aggregate_layer_name=None,
                ignore_nodata=True, all_touched=False,
                polygons_might_overlap=True)

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
        """PGP.geoprocessing: warp raster test."""
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
            'nearest', target_sr_wkt=reference.projection)

        pygeoprocessing.testing.assert_rasters_equal(
            base_a_path, target_raster_path)

    def test_warp_raster_unusual_pixel_size(self):
        """PGP.geoprocessing: warp on unusual pixel types and sizes."""
        pixel_a_matrix = numpy.ones((1, 1), numpy.byte)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(20), filename=base_a_path,
            dataset_opts=['PIXELTYPE=SIGNEDBYTE'])

        target_raster_path = os.path.join(self.workspace_dir, 'target_a.tif')

        # convert 1x1 pixel to a 30x30m pixel
        pygeoprocessing.warp_raster(
            base_a_path, [-30, 30], target_raster_path,
            'nearest', target_sr_wkt=reference.projection)

        expected_raster_path = os.path.join(
            self.workspace_dir, 'expected.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30),
            filename=expected_raster_path)

        pygeoprocessing.testing.assert_rasters_equal(
            expected_raster_path, target_raster_path)

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
        pixel_matrix = numpy.ones((128, 128), numpy.byte)
        pixel_matrix[0, 0] = 255  # 255 ubyte is -1 byte
        reference = sampledata.SRS_COLOMBIA
        nodata_base = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_base, reference.pixel_size(30), datatype=gdal.GDT_Byte,
            filename=base_path,
            dataset_opts=[
                'PIXELTYPE=SIGNEDBYTE',
                'TILED=YES',
                'BLOCKXSIZE=64',
                'BLOCKYSIZE=64',
                ])

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        # 255 should convert to -1 with signed bytes
        pygeoprocessing.new_raster_from_base(
            base_path, target_path, gdal.GDT_Byte, [-1],
            fill_value_list=[255],
            gtiff_creation_options=[
                'PIXELTYPE=SIGNEDBYTE',
                ])

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

    def test_new_raster_from_base_nodata_not_set(self):
        """PGP.geoprocessing: test new raster with nodata not set."""
        driver = gdal.GetDriverByName('GTiff')
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        new_raster = driver.Create(base_path, 128, 128, 1, gdal.GDT_Int32)
        new_raster = None

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        # 255 should convert to -1 with signed bytes
        pygeoprocessing.new_raster_from_base(
            base_path, target_path, gdal.GDT_Byte, [None],
            fill_value_list=[None],
            gtiff_creation_options=[
                'PIXELTYPE=SIGNEDBYTE',
                ])

        raster_properties = pygeoprocessing.get_raster_info(target_path)
        self.assertEqual(raster_properties['nodata'], [None])

    def test_create_raster_from_vector_extents(self):
        """PGP.geoprocessing: test creation of raster from vector extents."""
        reference = sampledata.SRS_COLOMBIA
        point_a = shapely.geometry.Point(
            reference.origin[0], reference.origin[1])
        mean_pixel_size = 30
        n_pixels_x = 9
        n_pixels_y = 19
        point_b = shapely.geometry.Point(
            reference.origin[0] +
            reference.pixel_size(mean_pixel_size)[0] * n_pixels_x,
            reference.origin[1] +
            reference.pixel_size(mean_pixel_size)[1] * n_pixels_y)
        source_vector_path = os.path.join(self.workspace_dir, 'sample_vector')
        pygeoprocessing.testing.create_vector_on_disk(
            [point_a, point_b], reference.projection, fields={'value': 'int'},
            attributes=[{'value': 0}, {'value': 1}], vector_format='GeoJSON',
            filename=source_vector_path)
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        target_pixel_size = [mean_pixel_size, -mean_pixel_size]
        target_nodata = -1
        target_pixel_type = gdal.GDT_Int16
        pygeoprocessing.create_raster_from_vector_extents(
            source_vector_path, target_raster_path, target_pixel_size,
            target_pixel_type, target_nodata)

        raster_properties = pygeoprocessing.get_raster_info(
            target_raster_path)
        self.assertEqual(raster_properties['raster_size'][0], n_pixels_x)
        self.assertEqual(raster_properties['raster_size'][1], n_pixels_y)

    def test_create_raster_from_vector_extents_odd_pixel_shapes(self):
        """PGP.geoprocessing: create raster vector ext. w/ odd pixel size."""
        reference = sampledata.SRS_COLOMBIA
        point_a = shapely.geometry.Point(
            reference.origin[0], reference.origin[1])
        pixel_x_size = -10
        pixel_y_size = 20
        n_pixels_x = 9
        n_pixels_y = 19
        point_b = shapely.geometry.Point(
            reference.origin[0] +
            pixel_x_size * n_pixels_x,
            reference.origin[1] +
            pixel_y_size * n_pixels_y)
        source_vector_path = os.path.join(self.workspace_dir, 'sample_vector')
        pygeoprocessing.testing.create_vector_on_disk(
            [point_a, point_b], reference.projection, fields={'value': 'int'},
            attributes=[{'value': 0}, {'value': 1}], vector_format='GeoJSON',
            filename=source_vector_path)
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        target_pixel_size = [pixel_x_size, pixel_y_size]
        target_nodata = -1
        target_pixel_type = gdal.GDT_Int16
        pygeoprocessing.create_raster_from_vector_extents(
            source_vector_path, target_raster_path, target_pixel_size,
            target_pixel_type, target_nodata)

        raster_properties = pygeoprocessing.get_raster_info(
            target_raster_path)
        self.assertEqual(raster_properties['raster_size'][0], n_pixels_x)
        self.assertEqual(raster_properties['raster_size'][1], n_pixels_y)

    def test_create_raster_from_vector_extents_bad_geometry(self):
        """PGP.geoprocessing: create raster from v. ext. with bad geometry."""
        reference = sampledata.SRS_COLOMBIA
        vector_driver = ogr.GetDriverByName('GeoJSON')
        source_vector_path = os.path.join(self.workspace_dir, 'vector.json')
        source_vector = vector_driver.CreateDataSource(source_vector_path)
        srs = osr.SpatialReference(reference.projection)
        source_layer = source_vector.CreateLayer(
            'vector', srs=srs)

        layer_defn = source_layer.GetLayerDefn()

        point_a = shapely.geometry.Point(
            reference.origin[0], reference.origin[1])
        mean_pixel_size = 30
        n_pixels_x = 9
        n_pixels_y = 19
        point_b = shapely.geometry.Point(
            reference.origin[0] +
            reference.pixel_size(mean_pixel_size)[0] * n_pixels_x,
            reference.origin[1] +
            reference.pixel_size(mean_pixel_size)[1] * n_pixels_y)

        for point in [point_a, point_b]:
            feature = ogr.Feature(layer_defn)
            feature_geometry = ogr.CreateGeometryFromWkb(point.wkb)
            feature.SetGeometry(feature_geometry)
            source_layer.CreateFeature(feature)
        null_feature = ogr.Feature(layer_defn)
        source_layer.CreateFeature(null_feature)
        source_layer.SyncToDisk()
        source_layer = None
        source_vector = None

        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        target_pixel_size = [mean_pixel_size, -mean_pixel_size]
        target_nodata = -1
        target_pixel_type = gdal.GDT_Int16
        pygeoprocessing.create_raster_from_vector_extents(
            source_vector_path, target_raster_path, target_pixel_size,
            target_pixel_type, target_nodata, fill_value=0)

        raster_properties = pygeoprocessing.get_raster_info(
            target_raster_path)
        self.assertEqual(raster_properties['raster_size'][0], n_pixels_x)
        self.assertEqual(raster_properties['raster_size'][1], n_pixels_y)
        expected_result = numpy.zeros((19, 9))
        raster = gdal.Open(target_raster_path)
        band = raster.GetRasterBand(1)
        result = band.ReadAsArray()
        band = None
        raster = None
        numpy.testing.assert_array_equal(expected_result, result)

    def test_find_int_not_in_array(self):
        """PGP.geoprocessing: test find int not in array."""
        import pygeoprocessing.geoprocessing
        for array in [[1, 2, 3],
                      [1, 2, 3, 4, 5, 6, 8],
                      [-10, 0, 1000],
                      [0],
                      [numpy.iinfo(numpy.int32).min,
                       numpy.iinfo(numpy.int32).min+1],
                      [numpy.iinfo(numpy.int32).min,
                       numpy.iinfo(numpy.int32).max]]:
            value = pygeoprocessing.geoprocessing._find_int_not_in_array(
                numpy.array(array))
            self.assertTrue(value not in array)

    def test_transform_box(self):
        """PGP.geoprocessing: test geotransforming lat/lng box to UTM10N."""
        # Willamette valley in lat/lng
        bounding_box = [-123.587984, 44.415778, -123.397976, 44.725814]
        base_ref = osr.SpatialReference()
        base_ref.ImportFromEPSG(4326)  # WGS84 EPSG

        target_ref = osr.SpatialReference()
        target_ref.ImportFromEPSG(26910)  # UTM10N EPSG

        result = pygeoprocessing.transform_bounding_box(
            bounding_box, base_ref.ExportToWkt(), target_ref.ExportToWkt())
        # I have confidence this function works by taking the result and
        # plotting it in a GIS polygon, so the expected result below is
        # regression data
        expected_result = [
            453189.3366727062, 4918131.085894576,
            468484.1637522648, 4952660.678869661]
        self.assertIs(
            numpy.testing.assert_allclose(
                result, expected_result), None)

    def test_iterblocks(self):
        """PGP.geoprocessing: test iterblocks."""
        reference = sampledata.SRS_COLOMBIA
        n_pixels = 100
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        pixel_matrix[:] = test_value
        nodata_target = None
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path,
            dataset_opts=[
                'TILED=YES',
                'BLOCKXSIZE=64',
                'BLOCKYSIZE=64'])

        total = 0
        for _, block in pygeoprocessing.iterblocks(
                raster_path, largest_block=0):
            total += numpy.sum(block)
        self.assertEqual(total, test_value * n_pixels**2)
