"""PyGeoprocessing 1.0 test suite."""
import time
import tempfile
import os
import unittest
import shutil
import types

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import scipy.ndimage
import shapely.geometry
import mock

try:
    from builtins import reload
except ImportError:
    from imp import reload


def passthrough(x):
    """Use in testing simple raster calculator calls."""
    return x


class PyGeoprocessing10(unittest.TestCase):
    """Tests for the PyGeoprocesing 1.0 refactor."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_star_import(self):
        """PGP: verify we can use *-import statement."""
        # Actually trying out the `from pygeoprocessing import *` notation here
        # raises a SyntaxWarning.  Instead, I'll just ensure that every
        # attribute in pygeoprocessing.__all__ is a function that is available
        # at the pygeoprocessing level.
        import pygeoprocessing
        import inspect
        for attrname in pygeoprocessing.__all__:
            try:
                func = getattr(pygeoprocessing, attrname)
                self.assertTrue(
                    isinstance(func, (
                        types.FunctionType, types.BuiltinFunctionType)) or
                    inspect.isroutine(func))
            except AttributeError:
                self.fail(('Function %s is in pygeoprocessing.__all__ but '
                           'is not exposed at the package level') % attrname)

    def test_version_loaded(self):
        """PGP: verify we can load the version."""
        try:
            import pygeoprocessing
            # Verifies that there's a version attribute and it has a value.
            self.assertTrue(len(pygeoprocessing.__version__) > 0)
        except Exception:
            self.fail('Could not load pygeoprocessing version.')

    def test_version_not_loaded(self):
        """PGP: verify exception when not installed."""
        from pkg_resources import DistributionNotFound
        import pygeoprocessing

        with mock.patch('pygeoprocessing.pkg_resources.get_distribution',
                        side_effect=DistributionNotFound('pygeoprocessing')):
            with self.assertRaises(RuntimeError):
                # RuntimeError is a side effect of `import pygeoprocessing`,
                # so we reload it to retrigger the metadata load.
                pygeoprocessing = reload(pygeoprocessing)

    def test_reclassify_raster_missing_pixel_value(self):
        """PGP.geoprocessing: test reclassify raster with missing value."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.reclassify_raster(
                (raster_path, 1), value_map, target_path, gdal.GDT_Float32,
                target_nodata, values_required=True)
        expected_message = (
            'The following 1 raster values [-0.5] from "%s" do not have ' %
            (raster_path,))
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_reclassify_raster(self):
        """PGP.geoprocessing: test reclassify raster."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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
            (raster_path, 1), value_map, target_path, gdal.GDT_Float32,
            target_nodata, values_required=True)
        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        self.assertEqual(
            numpy.sum(target_array), n_pixels**2 * value_map[test_value])

    def test_reclassify_raster_no_raster_path_band(self):
        """PGP.geoprocessing: test reclassify raster is path band aware."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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
        # we expect a value error because we didn't pass a (path, band)
        # for the first argument
        with self.assertRaises(ValueError):
            pygeoprocessing.reclassify_raster(
                raster_path, value_map, target_path, gdal.GDT_Float32,
                target_nodata, values_required=True)

    def test_reclassify_raster_empty_value_map(self):
        """PGP.geoprocessing: test reclassify raster."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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

        empty_value_map = {
        }
        target_nodata = -1
        with self.assertRaises(ValueError):
            pygeoprocessing.reclassify_raster(
                (raster_path, 1), empty_value_map, target_path,
                gdal.GDT_Float32, target_nodata, values_required=False)

    def test_reproject_vector(self):
        """PGP.geoprocessing: test reproject vector."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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
            target_vector_path, layer_id=0)

        vector = ogr.Open(target_vector_path)
        layer = vector.GetLayer()
        result_reference = layer.GetSpatialRef()
        layer = None
        vector = None
        self.assertTrue(
            osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                osr.SpatialReference(target_reference.ExportToWkt())))

    def test_calculate_disjoint_polygon_set(self):
        """PGP.geoprocessing: test calc_disjoint_poly no/intersection."""
        import pygeoprocessing
        import pygeoprocessing.testing
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        layer.CreateField(ogr.FieldDefn('expected_value', ogr.OFTInteger))
        layer_defn = layer.GetLayerDefn()

        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1.0
        subpixel_size = 1./5. * pixel_size
        origin_x = 1.0
        origin_y = -1.0
        n = 1
        layer.StartTransaction()
        for row_index in range(n * 2):
            for col_index in range(n * 2):
                x_pos = origin_x + (
                    col_index*2 + 1 + col_index // 2) * subpixel_size
                y_pos = origin_y - (
                    row_index*2 + 1 + row_index // 2) * subpixel_size
                shapely_feature = shapely.geometry.Polygon([
                    (x_pos, y_pos),
                    (x_pos+subpixel_size, y_pos),
                    (x_pos+subpixel_size, y_pos-subpixel_size),
                    (x_pos, y_pos-subpixel_size),
                    (x_pos, y_pos)])
                new_feature = ogr.Feature(layer_defn)
                new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
                new_feature.SetGeometry(new_geometry)
                expected_value = row_index // 2 * n + col_index // 2
                new_feature.SetField('expected_value', expected_value)
                layer.CreateFeature(new_feature)
        layer.CommitTransaction()
        layer.SyncToDisk()

        result = pygeoprocessing.calculate_disjoint_polygon_set(
            vector_path, bounding_box=[-10, -10, -9, -9])
        self.assertTrue(not result)

        # otherwise none overlap:
        result = pygeoprocessing.calculate_disjoint_polygon_set(vector_path)
        self.assertEqual(len(result), 1, result)

    def test_zonal_stats_for_small_polygons(self):
        """PGP.geoprocessing: test small polygons for zonal stats."""
        import pygeoprocessing
        import pygeoprocessing.testing
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        layer.CreateField(ogr.FieldDefn('expected_value', ogr.OFTInteger))
        layer_defn = layer.GetLayerDefn()

        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1.0
        subpixel_size = 1./5. * pixel_size
        origin_x = 1.0
        origin_y = -1.0
        n = 16
        layer.StartTransaction()
        for row_index in range(n * 2):
            for col_index in range(n * 2):
                x_pos = origin_x + (
                    col_index*2 + 1 + col_index // 2) * subpixel_size
                y_pos = origin_y - (
                    row_index*2 + 1 + row_index // 2) * subpixel_size
                shapely_feature = shapely.geometry.Polygon([
                    (x_pos, y_pos),
                    (x_pos+subpixel_size, y_pos),
                    (x_pos+subpixel_size, y_pos-subpixel_size),
                    (x_pos, y_pos-subpixel_size),
                    (x_pos, y_pos)])
                new_feature = ogr.Feature(layer_defn)
                new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
                new_feature.SetGeometry(new_geometry)
                expected_value = row_index // 2 * n + col_index // 2
                new_feature.SetField('expected_value', expected_value)
                layer.CreateFeature(new_feature)
        layer.CommitTransaction()
        layer.SyncToDisk()

        gtiff_driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        new_raster = gtiff_driver.Create(
            raster_path, n, n, 1, gdal.GDT_Int32, options=[
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])
        new_raster.SetProjection(srs.ExportToWkt())
        new_raster.SetGeoTransform([origin_x, 1.0, 0.0, origin_y, 0.0, -1.0])
        new_band = new_raster.GetRasterBand(1)
        new_band.SetNoDataValue(-1)
        array = numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))
        new_band.WriteArray(array)
        new_raster.FlushCache()
        new_band = None
        new_raster = None

        zonal_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path)
        self.assertEqual(len(zonal_stats), 4*n*n)
        for poly_id in zonal_stats:
            feature = layer.GetFeature(poly_id)
            self.assertEqual(
                feature.GetField('expected_value'),
                zonal_stats[poly_id]['sum'])

    def test_zonal_stats_no_bb_overlap(self):
        """PGP.geoprocessing: test no vector bb raster overlap."""
        import pygeoprocessing
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        layer_defn = layer.GetLayerDefn()

        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1.0
        subpixel_size = 1./5. * pixel_size
        origin_x = 1.0
        origin_y = -1.0
        n = 16
        layer.StartTransaction()
        x_pos = origin_x - n
        y_pos = origin_y + n
        shapely_feature = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)
        layer.CreateFeature(new_feature)
        layer.CommitTransaction()
        layer.SyncToDisk()
        layer = None
        vector = None

        gtiff_driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        new_raster = gtiff_driver.Create(
            raster_path, n, n, 1, gdal.GDT_Int32, options=[
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])
        new_raster.SetProjection(srs.ExportToWkt())
        new_raster.SetGeoTransform([origin_x, 1.0, 0.0, origin_y, 0.0, -1.0])
        new_band = new_raster.GetRasterBand(1)
        new_band.SetNoDataValue(-1)
        array = numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))
        new_band.WriteArray(array)
        new_raster.FlushCache()
        new_band = None
        new_raster = None

        zonal_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path)
        for poly_id in zonal_stats:
            self.assertEqual(zonal_stats[poly_id]['sum'], 0.0)

    def test_zonal_stats_all_outside(self):
        """PGP.geoprocessing: test vector all outside raster."""
        import pygeoprocessing
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        layer_defn = layer.GetLayerDefn()

        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1.0
        subpixel_size = 1./5. * pixel_size
        origin_x = 1.0
        origin_y = -1.0
        n = 16
        layer.StartTransaction()
        x_pos = origin_x - n
        y_pos = origin_y + n
        shapely_feature = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)
        layer.CreateFeature(new_feature)

        x_pos = origin_x + n*2
        y_pos = origin_y - n*2
        shapely_feature = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)

        x_pos = origin_x - subpixel_size*.99
        y_pos = origin_y + subpixel_size*.99
        shapely_feature = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)
        layer.CreateFeature(new_feature)

        x_pos = origin_x + (n+.99)
        y_pos = origin_y - (n+.99)
        shapely_feature = shapely.geometry.Polygon([
            (x_pos, y_pos),
            (x_pos+subpixel_size, y_pos),
            (x_pos+subpixel_size, y_pos-subpixel_size),
            (x_pos, y_pos-subpixel_size),
            (x_pos, y_pos)])
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)
        layer.CreateFeature(new_feature)

        layer.CommitTransaction()
        layer.SyncToDisk()

        gtiff_driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        new_raster = gtiff_driver.Create(
            raster_path, n, n, 1, gdal.GDT_Int32, options=[
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])
        new_raster.SetProjection(srs.ExportToWkt())
        new_raster.SetGeoTransform([origin_x, 1.0, 0.0, origin_y, 0.0, -1.0])
        new_band = new_raster.GetRasterBand(1)
        new_band.SetNoDataValue(-1)
        array = numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))
        # this will catch a polygon that barely intersects the upper left
        # hand corner but is nodata.
        array[0, 0] = -1
        new_band.WriteArray(array)
        new_raster.FlushCache()
        new_band = None
        new_raster = None

        zonal_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path)
        for poly_id in zonal_stats:
            self.assertEqual(zonal_stats[poly_id]['sum'], 0.0)

        raster_path = os.path.join(
            self.workspace_dir, 'nonodata_small_raster.tif')
        new_raster = gtiff_driver.Create(
            raster_path, n, n, 1, gdal.GDT_Int32, options=[
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])
        new_raster.SetProjection(srs.ExportToWkt())
        new_raster.SetGeoTransform(
            [origin_x+n, -1.0, 0.0, origin_y-n, 0.0, 1.0])
        new_band = new_raster.GetRasterBand(1)
        array = numpy.fliplr(numpy.flipud(
            numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))))
        # this will catch a polygon that barely intersects the upper left
        # hand corner but is nodata.
        new_band.WriteArray(array)
        new_raster.FlushCache()
        new_band = None
        new_raster = None

        zonal_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path)
        for poly_id in zonal_stats:
            self.assertEqual(zonal_stats[poly_id]['sum'], 0.0)

    def test_mask_raster(self):
        """PGP.geoprocessing: test mask raster."""
        import pygeoprocessing
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        layer_defn = layer.GetLayerDefn()

        origin_x = 1.0
        origin_y = -1.0
        n = 16

        layer.StartTransaction()
        shapely_feature = shapely.geometry.Polygon([
            (origin_x, origin_y),
            (origin_x+n, origin_y),
            (origin_x+n, origin_y-n//2),
            (origin_x, origin_y-n//2),
            (origin_x, origin_y)])
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)
        layer.CreateFeature(new_feature)
        layer.CommitTransaction()
        layer.SyncToDisk()

        gtiff_driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        new_raster = gtiff_driver.Create(
            raster_path, n, n, 1, gdal.GDT_Int32, options=[
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])
        new_raster.SetProjection(srs.ExportToWkt())
        new_raster.SetGeoTransform([origin_x, 1.0, 0.0, origin_y, 0.0, -1.0])
        new_band = new_raster.GetRasterBand(1)
        new_band.Fill(2)
        new_raster.FlushCache()
        new_band = None
        new_raster = None

        target_mask_raster_path = os.path.join(
            self.workspace_dir, 'test_mask.tif')
        pygeoprocessing.mask_raster(
            (raster_path, 1), vector_path, target_mask_raster_path,
            target_mask_value=None, working_dir=self.workspace_dir)

        mask_raster = gdal.OpenEx(target_mask_raster_path, gdal.OF_RASTER)
        mask_band = mask_raster.GetRasterBand(1)
        mask_array = mask_band.ReadAsArray()
        expected_result = numpy.empty((16, 16))
        expected_result[0:8, :] = 2
        expected_result[8::, :] = 0
        self.assertTrue(
            numpy.count_nonzero(numpy.isclose(
                mask_array, expected_result)) == 16**2)

        pygeoprocessing.mask_raster(
            (raster_path, 1), vector_path, target_mask_raster_path,
            target_mask_value=12, working_dir=self.workspace_dir)

        mask_raster = gdal.OpenEx(target_mask_raster_path, gdal.OF_RASTER)
        mask_band = mask_raster.GetRasterBand(1)
        mask_array = mask_band.ReadAsArray()
        expected_result = numpy.empty((16, 16))
        expected_result[0:8, :] = 2
        expected_result[8::, :] = 12
        self.assertTrue(
            numpy.count_nonzero(numpy.isclose(
                mask_array, expected_result)) == 16**2)

    def test_reproject_vector_partial_fields(self):
        """PGP.geoprocessing: reproject vector with partial field copy."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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
            fields={
                'id': 'int',
                'foo': 'string'},
            attributes=[{aggregate_field_name: 0}],
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
            target_vector_path, layer_id=0, copy_fields=['id'])

        vector = ogr.Open(target_vector_path)
        layer = vector.GetLayer()
        result_reference = layer.GetSpatialRef()
        layer_defn = layer.GetLayerDefn()
        self.assertTrue(
            osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                osr.SpatialReference(target_reference.ExportToWkt())))
        self.assertTrue(layer_defn.GetFieldCount(), 1)
        self.assertEqual(layer_defn.GetFieldIndex('id'), 0)

        target_vector_no_fields_path = os.path.join(
            self.workspace_dir, 'target_vector_no_fields.shp')
        pygeoprocessing.reproject_vector(
            base_vector_path, target_reference.ExportToWkt(),
            target_vector_no_fields_path, layer_id=0, copy_fields=False)
        layer = None
        vector = None

        vector = ogr.Open(target_vector_no_fields_path)
        layer = vector.GetLayer()
        result_reference = layer.GetSpatialRef()
        layer_defn = layer.GetLayerDefn()
        self.assertTrue(
            osr.SpatialReference(result_reference.ExportToWkt()).IsSame(
                osr.SpatialReference(target_reference.ExportToWkt())))
        self.assertTrue(layer_defn.GetFieldCount(), 0)
        layer = None
        vector = None

    def test_zonal_statistics(self):
        """PGP.geoprocessing: test zonal stats function."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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
        polygon_c = shapely.geometry.Polygon([
            (reference.origin[1]*2, reference.origin[1]*3),
            (reference.origin[1]*2, -pixel_size+reference.origin[1]*3),
            (reference.origin[1]*2+pixel_size,
             -pixel_size+reference.origin[1]*3),
            (reference.origin[1]*2+pixel_size, reference.origin[1]*3),
            (reference.origin[1]*2, reference.origin[1]*3)])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector')
        pygeoprocessing.testing.create_vector_on_disk(
            [polygon_a, polygon_b, polygon_c], reference.projection,
            vector_format='GeoJSON', filename=aggregating_vector_path)
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        nodata_target = None
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path)
        result = pygeoprocessing.zonal_statistics(
            (raster_path, 1), aggregating_vector_path,
            aggregate_layer_name=None,
            ignore_nodata=True,
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
                'sum': 1.0},
            2: {
                'min': None,
                'max': None,
                'count': 0,
                'nodata_count': 0,
                'sum': 0.0}}
        self.assertEqual(result, expected_result)

    def test_zonal_statistics_nodata(self):
        """PGP.geoprocessing: test zonal stats function with non-overlap."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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
            aggregate_layer_name=None,
            ignore_nodata=True,
            polygons_might_overlap=False)
        expected_result = {
            0: {
                'count': 0,
                'max': None,
                'min': None,
                'nodata_count': 81,
                'sum': 0.0}}
        self.assertEqual(result, expected_result)

    def test_zonal_statistics_nodata_is_zero(self):
        """PGP.geoprocessing: test zonal stats function w/ nodata set to 0."""
        import pygeoprocessing
        import pygeoprocessing.testing
        # create aggregating polygon
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        layer_defn = layer.GetLayerDefn()

        origin_x = 1.0
        origin_y = -1.0
        n = 2

        layer.StartTransaction()
        shapely_feature = shapely.geometry.Polygon([
            (origin_x, origin_y),
            (origin_x+n, origin_y),
            (origin_x+n, origin_y-n),
            (origin_x, origin_y-n),
            (origin_x, origin_y)])
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)
        layer.CreateFeature(new_feature)
        layer.CommitTransaction()
        layer.SyncToDisk()

        layer = None
        vector = None

        # create raster with nodata value of 0
        gtiff_driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        new_raster = gtiff_driver.Create(
            raster_path, n, n, 1, gdal.GDT_Int32, options=[
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])
        new_raster.SetProjection(srs.ExportToWkt())
        new_raster.SetGeoTransform([origin_x, 1.0, 0.0, origin_y, 0.0, -1.0])
        new_band = new_raster.GetRasterBand(1)
        new_band.WriteArray(numpy.array([[1, 0], [1, 0]]))
        new_band.SetNoDataValue(0)
        new_raster.FlushCache()
        new_band = None
        new_raster = None

        result = pygeoprocessing.zonal_statistics(
            (raster_path, 1), vector_path,
            aggregate_layer_name=None,
            ignore_nodata=True,
            polygons_might_overlap=False)
        expected_result = {
            1: {
                'count': 2,
                'max': 1,
                'min': 1,
                'nodata_count': 2,
                'sum': 2.0}}
        self.assertEqual(result, expected_result)

    def test_zonal_statistics_named_layer(self):
        """PGP.geoprocessing: test zonal stats with named layer."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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
        pygeoprocessing.testing.create_vector_on_disk(
            [polygon_a], reference.projection,
            vector_format='ESRI Shapefile', filename=aggregating_vector_path)
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        nodata_target = None
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path)
        result = pygeoprocessing.zonal_statistics(
            (raster_path, 1), aggregating_vector_path,
            aggregate_layer_name='aggregate_vector',
            ignore_nodata=True,
            polygons_might_overlap=True)
        expected_result = {
            0: {
                'count': 81,
                'max': 1.0,
                'min': 1.0,
                'nodata_count': 0,
                'sum': 81.0}}
        self.assertEqual(result, expected_result)

    def test_zonal_statistics_bad_vector(self):
        """PGP.geoprocessing: zonal stats raises exception on bad vectors."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
        # create aggregating polygon
        reference = sampledata.SRS_COLOMBIA
        n_pixels = 9
        missing_aggregating_vector_path = os.path.join(
            self.workspace_dir, 'not_exists.shp')
        pixel_matrix = numpy.ones((n_pixels, n_pixels), numpy.float32)
        nodata_target = None
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path)
        with self.assertRaises(RuntimeError) as cm:
            _ = pygeoprocessing.zonal_statistics(
                (raster_path, 1), missing_aggregating_vector_path,
                ignore_nodata=True,
                polygons_might_overlap=True)
        expected_message = 'Could not open aggregate vector'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        pixel_size = 30.0
        polygon_a = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        aggregating_vector_path = os.path.join(
            self.workspace_dir, 'aggregate_vector.shp')
        pygeoprocessing.testing.create_vector_on_disk(
            [polygon_a], reference.projection,
            vector_format='ESRI Shapefile', filename=aggregating_vector_path)
        with self.assertRaises(RuntimeError) as cm:
            _ = pygeoprocessing.zonal_statistics(
                (raster_path, 1), aggregating_vector_path,
                ignore_nodata=True,
                aggregate_layer_name='not a layer name',
                polygons_might_overlap=True)
        expected_message = 'Could not open layer not a layer name'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_zonal_statistics_bad_raster_path_band(self):
        """PGP.geoprocessing: test zonal stats with bad raster/path type."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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
        with self.assertRaises(ValueError):
            # intentionally not passing a (path, band) tuple as first arg
            _ = pygeoprocessing.zonal_statistics(
                raster_path, aggregating_vector_path,
                aggregate_layer_name=None,
                ignore_nodata=True,
                polygons_might_overlap=True)

    def test_interpolate_points(self):
        """PGP.geoprocessing: test interpolate points feature."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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
            source_vector_path, 'value', (result_path, 1), 'near')

        # verify that result is expected
        result_raster = gdal.OpenEx(result_path, gdal.OF_RASTER)
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
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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
            'near', target_sr_wkt=reference.projection, n_threads=1)

        pygeoprocessing.testing.assert_rasters_equal(
            base_a_path, target_raster_path)

    def test_warp_raster_unusual_pixel_size(self):
        """PGP.geoprocessing: warp on unusual pixel types and sizes."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
        pixel_a_matrix = numpy.ones((1, 1), numpy.byte)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(20), filename=base_a_path,
            raster_driver_creation_tuple=('GTiff', [
                'PIXELTYPE=SIGNEDBYTE']))

        target_raster_path = os.path.join(self.workspace_dir, 'target_a.tif')

        # convert 1x1 pixel to a 30x30m pixel
        pygeoprocessing.warp_raster(
            base_a_path, [-30, 30], target_raster_path,
            'near', target_sr_wkt=reference.projection)

        expected_raster_path = os.path.join(
            self.workspace_dir, 'expected.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30),
            filename=expected_raster_path)

        pygeoprocessing.testing.assert_rasters_equal(
            expected_raster_path, target_raster_path)

    def test_warp_raster_0x0_size(self):
        """PGP.geoprocessing: test warp where so small it would be 0x0."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_a_path)

        target_raster_path = os.path.join(self.workspace_dir, 'target_a.tif')
        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)
        target_bb = base_a_raster_info['bounding_box']
        # pick a tiny tiny bounding box in the middle (less than a pixel big)
        target_bb[0] = (target_bb[0] + target_bb[2]) / 2.0
        target_bb[1] = (target_bb[1] + target_bb[3]) / 2.0
        target_bb[2] = target_bb[0]
        target_bb[3] = target_bb[1]
        pygeoprocessing.warp_raster(
            base_a_path, base_a_raster_info['pixel_size'], target_raster_path,
            'near', target_bb=target_bb,
            target_sr_wkt=reference.projection)

        expected_raster_path = os.path.join(
            self.workspace_dir, 'expected.tif')
        expected_matrix = numpy.ones((1, 1), numpy.int16)
        pygeoprocessing.testing.create_raster_on_disk(
            [expected_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30),
            filename=expected_raster_path)

        pygeoprocessing.testing.assert_rasters_equal(
            expected_raster_path, target_raster_path)

    def test_align_and_resize_raster_stack_bad_values(self):
        """PGP.geoprocessing: align/resize raster bad base values."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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

        resample_method_list = ['near']
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        with self.assertRaises(ValueError) as cm:
            # here base_raster_path_list is length 1 but others are length 2
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, target_raster_path_list,
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)
        expected_message = 'must be the same length'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            # here pixel size is an invalid type
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, ['target_a.tif'],
                resample_method_list,
                100.0, bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)
        expected_message = 'target_pixel_size is not a tuple'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            # here pixel size has invalid values
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, ['target_a.tif'],
                resample_method_list,
                [100.0, "ten"], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)
        expected_message = 'Invalid value for `target_pixel_size`'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            # here pixel size is too long
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, ['target_a.tif'],
                resample_method_list,
                [100.0, 100.0, 100.0], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)
        expected_message = 'Invalid value for `target_pixel_size`'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_align_and_resize_raster_stack_duplicate_outputs(self):
        """PGP.geoprocessing: align/resize raster duplicate outputs."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_a_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_a_path)

        base_raster_path_list = [base_a_path, base_a_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'a']]

        resample_method_list = ['near'] * 2
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        with self.assertRaises(ValueError) as cm:
            # here base_raster_path_list is length 1 but others are length 2
            pygeoprocessing.align_and_resize_raster_stack(
                base_raster_path_list, target_raster_path_list,
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                base_vector_path_list=None, raster_align_index=0)

        expected_message = 'There are duplicated paths on the target list.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_align_and_resize_raster_stack_bad_mode(self):
        """PGP.geoprocessing: align/resize raster bad bounding box mode."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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

        resample_method_list = ['near']
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
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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

        resample_method_list = ['near']
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
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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

        pixel_c_matrix = numpy.ones((15, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_c_path = os.path.join(self.workspace_dir, 'base_c.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_c_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(45), filename=base_c_path)

        pixel_d_matrix = numpy.ones((5, 10), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_d_path = os.path.join(self.workspace_dir, 'base_d.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_d_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(45), filename=base_d_path)

        base_raster_path_list = [
            base_a_path, base_b_path, base_c_path, base_d_path]
        target_raster_path_list = [
            os.path.join(self.workspace_dir, 'target_%s.tif' % char)
            for char in ['a', 'b', 'c', 'd']]

        resample_method_list = ['near'] * len(target_raster_path_list)
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list, target_raster_path_list,
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            base_vector_path_list=None, raster_align_index=0)

        for raster_index in range(len(target_raster_path_list)):
            target_raster_info = pygeoprocessing.get_raster_info(
                target_raster_path_list[raster_index])
            target_raster = gdal.OpenEx(
                target_raster_path_list[raster_index], gdal.OF_RASTER)
            target_band = target_raster.GetRasterBand(1)
            target_array = target_band.ReadAsArray()
            numpy.testing.assert_array_equal(pixel_a_matrix, target_array)
            self.assertEqual(
                target_raster_info['pixel_size'],
                base_a_raster_info['pixel_size'])

    def test_align_and_resize_raster_stack_int_with_vectors(self):
        """PGP.geoprocessing: align/resize raster test inters. w/ vectors."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata
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

        resample_method_list = ['near'] * 2
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
        for raster_index in range(2):
            target_raster_info = pygeoprocessing.get_raster_info(
                target_raster_path_list[raster_index])
            target_raster = gdal.OpenEx(
                target_raster_path_list[raster_index], gdal.OF_RASTER)
            target_band = target_raster.GetRasterBand(1)
            target_array = target_band.ReadAsArray()
            numpy.testing.assert_array_equal(expected_matrix, target_array)
            self.assertEqual(
                target_raster_info['pixel_size'],
                base_a_raster_info['pixel_size'])

    def test_align_and_resize_raster_stack_manual_projection(self):
        """PGP.geoprocessing: align/resize with manual projections."""
        import pygeoprocessing

        geotiff_driver = gdal.GetDriverByName('GTiff')
        base_raster_path = os.path.join(self.workspace_dir, 'base_raster.tif')
        base_raster = geotiff_driver.Create(
            base_raster_path, 1, 1, 1, gdal.GDT_Byte)
        base_raster.SetGeoTransform([0.1, 1, 0, 0.1, 0, -1])
        base_band = base_raster.GetRasterBand(1)
        pixel_matrix = numpy.ones((1, 1), numpy.int16)
        base_band.WriteArray(pixel_matrix)
        base_band = None
        base_raster = None

        utm_30n_sr = osr.SpatialReference()
        utm_30n_sr.ImportFromEPSG(32630)
        wgs84_sr = osr.SpatialReference()
        wgs84_sr.ImportFromEPSG(4326)

        target_raster_path = os.path.join(self.workspace_dir, 'target.tif')
        target_pixel_size = (112000/4, -112000/4)

        pygeoprocessing.align_and_resize_raster_stack(
            [base_raster_path], [target_raster_path],
            ['near'], target_pixel_size, 'intersection',
            raster_align_index=0,
            base_sr_wkt_list=[wgs84_sr.ExportToWkt()],
            target_sr_wkt=utm_30n_sr.ExportToWkt())

        target_raster = gdal.OpenEx(target_raster_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        numpy.testing.assert_almost_equal(
            target_array, numpy.ones((4, 4)))

    def test_align_and_resize_raster_stack_no_base_projection(self):
        """PGP.geoprocessing: align raise error if no base projection."""
        import pygeoprocessing

        geotiff_driver = gdal.GetDriverByName('GTiff')
        base_raster_path = os.path.join(self.workspace_dir, 'base_raster.tif')
        base_raster = geotiff_driver.Create(
            base_raster_path, 1, 1, 1, gdal.GDT_Byte)
        base_raster.SetGeoTransform([0.1, 1, 0, 0.1, 0, -1])
        base_band = base_raster.GetRasterBand(1)
        pixel_matrix = numpy.ones((1, 1), numpy.int16)
        base_band.WriteArray(pixel_matrix)
        base_band = None
        base_raster = None

        utm_30n_sr = osr.SpatialReference()
        utm_30n_sr.ImportFromEPSG(32630)
        wgs84_sr = osr.SpatialReference()
        wgs84_sr.ImportFromEPSG(4326)

        target_raster_path = os.path.join(self.workspace_dir, 'target.tif')
        target_pixel_size = (112000/4, -112000/4)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.align_and_resize_raster_stack(
                [base_raster_path], [target_raster_path],
                ['near'], target_pixel_size, 'intersection',
                raster_align_index=0,
                base_sr_wkt_list=[None],
                target_sr_wkt=utm_30n_sr.ExportToWkt())
            expected_message = "no projection for raster"
            actual_message = str(cm.exception)
            self.assertTrue(
                expected_message in actual_message, actual_message)

    def test_align_and_resize_raster_stack_no_overlap(self):
        """PGP.geoprocessing: align/resize raster no intersection error."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

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

        resample_method_list = ['near'] * 2
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
                raster_align_index=0,
                base_vector_path_list=[single_pixel_path])

    def test_align_and_resize_raster_stack_union(self):
        """PGP.geoprocessing: align/resize raster test union."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

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

        resample_method_list = ['near'] * 2
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

        target_raster = gdal.OpenEx(target_raster_path_list[0], gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        numpy.testing.assert_array_equal(expected_matrix_a, target_array)

    def test_align_and_resize_raster_stack_bb(self):
        """PGP.geoprocessing: align/resize raster test bounding box."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

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

        resample_method_list = ['near'] * 2
        # format is xmin,ymin,xmax,ymax; since y pixel size is negative it
        # goes first in the following bounding box construction
        bounding_box_mode = [
            reference.origin[0],
            reference.origin[1] + reference.pixel_size(30)[1] * 5,
            reference.origin[0] + reference.pixel_size(30)[0] * 5,
            reference.origin[1]]

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list, target_raster_path_list,
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            base_vector_path_list=None, raster_align_index=0)

        # we expect this to be twice as big since second base raster has a
        # pixel size twice that of the first.
        target_raster = gdal.OpenEx(target_raster_path_list[0], gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        numpy.testing.assert_array_equal(pixel_a_matrix, target_array)

    def test_raster_calculator(self):
        """PGP.geoprocessing: raster_calculator identity test."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_path)

        target_path = os.path.join(
            self.workspace_dir, 'subdir', 'target.tif')
        pygeoprocessing.raster_calculator(
            [(base_path, 1)], passthrough, target_path,
            gdal.GDT_Int32, nodata_target, calc_raster_stats=True)
        pygeoprocessing.testing.assert_rasters_equal(base_path, target_path)

    def test_raster_calculator_bad_target_type(self):
        """PGP.geoprocessing: raster_calculator bad target type value."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_path)

        target_path = os.path.join(
            self.workspace_dir, 'subdir', 'target.tif')
        # intentionally reversing `nodata_target` and `gdal.GDT_Int32`,
        # a value of -1 should be a value error for the target
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(base_path, 1)], passthrough, target_path,
                nodata_target, gdal.GDT_Int32)
        expected_message = (
            'Invalid target type, should be a gdal.GDT_* type')
        actual_message = str(cm.exception)
        self.assertTrue(
            expected_message in actual_message, actual_message)
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=base_path)

        target_path = os.path.join(
            self.workspace_dir, 'target.tif')
        for bad_raster_path_band_list in [
                [base_path], [(base_path, "1")], [(1, 1)],
                [(base_path, 1, base_path, 2)], base_path]:
            with self.assertRaises(ValueError) as cm:
                pygeoprocessing.raster_calculator(
                    bad_raster_path_band_list, passthrough, target_path,
                    gdal.GDT_Int32, nodata_target, calc_raster_stats=True)
            expected_message = (
                'Expected a sequence of path / integer band tuples, '
                'ndarrays, ')
            actual_message = str(cm.exception)
            self.assertTrue(
                expected_message in actual_message, actual_message)

    def test_raster_calculator_no_path(self):
        """PGP.geoprocessing: raster_calculator raise ex. on bad file path."""
        import pygeoprocessing

        nodata_target = -1
        nonexistant_path = os.path.join(self.workspace_dir, 'nofile.tif')
        target_path = os.path.join(
            self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(nonexistant_path, 1)], passthrough, target_path,
                gdal.GDT_Int32, nodata_target, calc_raster_stats=True)
        expected_message = (
            "The following files were expected but do not exist on the ")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_raster_calculator_nodata(self):
        """PGP.geoprocessing: raster_calculator test with all nodata."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

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
            [(base_path, 1)], passthrough, target_path,
            gdal.GDT_Int32, nodata_target, calc_raster_stats=True)
        pygeoprocessing.testing.assert_rasters_equal(base_path, target_path)

    def test_rs_calculator_output_alias(self):
        """PGP.geoprocessing: rs_calculator expected error for aliasing."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        pixel_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_base = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_base, reference.pixel_size(30), filename=base_path)

        with self.assertRaises(ValueError) as cm:
            # intentionally passing target path as base path to raise error
            pygeoprocessing.raster_calculator(
                [(base_path, 1)], passthrough, base_path,
                gdal.GDT_Int32, nodata_base, calc_raster_stats=True)
        expected_message = 'is used as a target path, but it is also in the '
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_rs_calculator_bad_overlap(self):
        """PGP.geoprocessing: rs_calculator expected error on bad overlap."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

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
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(base_path_a, 1), (base_path_b, 1)], passthrough,
                target_path, gdal.GDT_Int32, nodata_base,
                calc_raster_stats=True)
        expected_message = 'Input Rasters are not the same dimensions.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_raster_calculator_constant_args_error(self):
        """PGP.geoprocessing: handle empty input arrays."""
        import pygeoprocessing

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError) as cm:
            # no input args should cause a ValueError
            pygeoprocessing.raster_calculator(
                [], lambda: None, target_path,
                gdal.GDT_Float32, None)
        expected_message = '`base_raster_path_band_const_list` is empty'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_raster_calculator_invalid_numpy_array(self):
        """PGP.geoprocessing: handle invalid numpy array sizes."""
        import pygeoprocessing

        target_path = os.path.join(self.workspace_dir, 'target.tif')

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [numpy.empty((3, 3, 3))], lambda x: None, target_path,
                gdal.GDT_Float32, None)
        expected_message = 'Numpy array inputs must be 2 dimensions or less'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_raster_calculator_invalid_band_numbers(self):
        """PGP.geoprocessing: ensure invalid band numbers fail."""
        import pygeoprocessing

        driver = gdal.GetDriverByName("GTiff")
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        new_raster = driver.Create(
            base_path, 128, 128, 1, gdal.GDT_Int32,
            options=(
                'TILED=YES', 'BLOCKXSIZE=16', 'BLOCKYSIZE=16'))
        new_raster.FlushCache()
        new_raster = None

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError) as cm:
            # no input args should cause a ValueError
            pygeoprocessing.raster_calculator(
                [(base_path, 2)], lambda: None, target_path,
                gdal.GDT_Float32, None)
        expected_message = "do not contain requested band "
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

        with self.assertRaises(ValueError) as cm:
            # no input args should cause a ValueError
            pygeoprocessing.raster_calculator(
                [(base_path, 0)], lambda: None, target_path,
                gdal.GDT_Float32, None)
        expected_message = "do not contain requested band "
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_raster_calculator_unbroadcastable_array(self):
        """PGP.geoprocessing: incompatable array sizes raise error."""
        import pygeoprocessing

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        a_arg = 3
        x_arg = numpy.array(range(2))
        y_arg = numpy.array(range(3)).reshape((3, 1))
        z_arg = numpy.ones((4, 4))
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(a_arg, 'raw'), x_arg, y_arg, z_arg],
                lambda a, x, y, z: a*x*y*z, target_path, gdal.GDT_Float32,
                None)
        expected_message = "inputs cannot be broadcast into a single shape"
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_raster_calculator_array_raster_mismatch(self):
        """PGP.geoprocessing: bad array shape with raster raise error."""
        import pygeoprocessing

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        driver = gdal.GetDriverByName('GTiff')
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        new_raster = driver.Create(
            base_path, 128, 128, 1, gdal.GDT_Int32,
            options=(
                'TILED=YES', 'BLOCKXSIZE=16', 'BLOCKYSIZE=16'))
        new_raster.GetRasterBand(1).WriteArray(
            numpy.ones((128, 128)))
        new_raster.FlushCache()
        new_raster = None

        z_arg = numpy.ones((4, 4))
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(base_path, 1), z_arg], lambda a, z: a*z,
                target_path, gdal.GDT_Float32, None)
        expected_message = (
            'Raster size (128, 128) cannot be broadcast '
            'to numpy shape (4')
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        y_arg = numpy.ones((4,))
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(base_path, 1), y_arg], lambda a, y: a*y,
                target_path, gdal.GDT_Float32, None)
        expected_message = (
            'Raster size (128, 128) cannot be broadcast '
            'to numpy shape (4')
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_raster_calculator_bad_raw_args(self):
        """PGP.geoprocessing: tuples that don't match (x, 'raw')."""
        import pygeoprocessing

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        driver = gdal.GetDriverByName('GTiff')
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        new_raster = driver.Create(
            base_path, 128, 128, 1, gdal.GDT_Int32,
            options=(
                'TILED=YES', 'BLOCKXSIZE=16', 'BLOCKYSIZE=16'))
        new_raster.GetRasterBand(1).WriteArray(
            numpy.ones((128, 128)))
        new_raster.FlushCache()
        new_raster = None

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.raster_calculator(
                [(base_path, 1), ("raw",)], lambda a, z: a*z,
                target_path, gdal.GDT_Float32, None)
        expected_message = 'Expected a sequence of path / integer band tuples'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_raster_calculator_constant_args(self):
        """PGP.geoprocessing: test constant arguments of raster calc."""
        import pygeoprocessing

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        a_arg = 3
        x_arg = numpy.array(range(2))
        y_arg = numpy.array(range(3)).reshape((3, 1))
        z_arg = numpy.ones((3, 2))
        list_arg = [1, 1, 1, -1]
        pygeoprocessing.raster_calculator(
            [(a_arg, 'raw'), x_arg, y_arg, z_arg], lambda a, x, y, z: a*x*y*z,
            target_path, gdal.GDT_Float32, 0)

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_array = target_raster.GetRasterBand(1).ReadAsArray()
        target_raster = None
        expected_result = numpy.array([[0, 0], [0, 3], [0, 6]])
        numpy.testing.assert_array_almost_equal(target_array, expected_result)

        target_path = os.path.join(self.workspace_dir, 'target_a.tif')
        with self.assertRaises(ValueError) as cm:
            # this will return a scalar, when it should return 2d array
            pygeoprocessing.raster_calculator(
                [(a_arg, 'raw')], lambda a: a, target_path,
                gdal.GDT_Float32, None)
        expected_message = (
            "Only (object, 'raw') values have been passed.")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            # this will return a scalar, when it should return 2d array
            pygeoprocessing.raster_calculator(
                [x_arg], lambda x: 0.0, target_path,
                gdal.GDT_Float32, None)
        expected_message = (
            "Expected `local_op` to return a numpy.ndarray")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        target_path = os.path.join(
            self.workspace_dir, 'target_1d_2darray.tif')
        pygeoprocessing.raster_calculator(
            [y_arg], lambda y: y, target_path,
            gdal.GDT_Float32, None)

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_array = target_raster.GetRasterBand(1).ReadAsArray()
        target_raster = None
        numpy.testing.assert_array_almost_equal(target_array, y_arg)

        target_path = os.path.join(self.workspace_dir, 'target_1d_only.tif')
        pygeoprocessing.raster_calculator(
            [x_arg], lambda x: x, target_path,
            gdal.GDT_Float32, None)

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_array = target_raster.GetRasterBand(1).ReadAsArray()
        target_raster = None
        numpy.testing.assert_array_almost_equal(
            target_array, x_arg.reshape((1, x_arg.size)))

        target_path = os.path.join(self.workspace_dir, 'raw_args.tif')
        pygeoprocessing.raster_calculator(
            [x_arg, (list_arg, 'raw')], lambda x, y_list: x * y_list[3],
            target_path, gdal.GDT_Float32, None)

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_array = target_raster.GetRasterBand(1).ReadAsArray()
        target_raster = None
        numpy.testing.assert_array_almost_equal(
            target_array, -x_arg.reshape((1, x_arg.size)))

        target_path = os.path.join(self.workspace_dir, 'raw_numpy_args.tif')
        pygeoprocessing.raster_calculator(
            [x_arg, (numpy.array(list_arg), 'raw')],
            lambda x, y_list: x * y_list[3], target_path, gdal.GDT_Float32,
            None)

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_array = target_raster.GetRasterBand(1).ReadAsArray()
        target_raster = None
        numpy.testing.assert_array_almost_equal(
            target_array, -x_arg.reshape((1, x_arg.size)))

    def test_combined_constant_args_raster(self):
        """PGP.geoprocessing: test raster calc with constant args."""
        import pygeoprocessing

        driver = gdal.GetDriverByName('GTiff')
        base_path = os.path.join(self.workspace_dir, 'base.tif')

        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)  # WGS84 EPSG

        new_raster = driver.Create(
            base_path, 128, 128, 1, gdal.GDT_Int32,
            options=(
                'TILED=YES', 'BLOCKXSIZE=32', 'BLOCKYSIZE=32'))
        geotransform = [0.1, 1., 0., 0., 0., -1.]
        new_raster.SetGeoTransform(geotransform)
        new_raster.SetProjection(wgs84_ref.ExportToWkt())
        new_band = new_raster.GetRasterBand(1)

        nodata = 0
        new_band.SetNoDataValue(nodata)
        raster_array = numpy.ones((128, 128), dtype=numpy.int32)
        raster_array[127, 127] = nodata
        new_band.WriteArray(raster_array)
        new_band.FlushCache()
        new_raster.FlushCache()
        new_band = None
        new_raster = None

        target_path = os.path.join(self.workspace_dir, 'target.tif')

        # making a local op that needs a valid mask to ensure that `col_array`
        # is tiled out correctly
        def local_op(scalar, raster_array, col_array):
            valid_mask = raster_array != nodata
            result = numpy.empty_like(raster_array)
            result[:] = nodata
            result[valid_mask] = (
                scalar * raster_array[valid_mask] * col_array[valid_mask])
            return result

        pygeoprocessing.raster_calculator(
            [(10, 'raw'), (base_path, 1), numpy.array(range(128))],
            local_op, target_path, gdal.GDT_Float32, None, largest_block=0)

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        result = target_raster.GetRasterBand(1).ReadAsArray()

        expected_result = (
            10 * numpy.ones((128, 128)) * numpy.array(range(128)))
        # we expect one pixel to have been masked out
        expected_result[127, 127] = nodata
        numpy.testing.assert_allclose(result, expected_result)

    def test_new_raster_from_base_unsigned_byte(self):
        """PGP.geoprocessing: test that signed byte rasters copy over."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        pixel_matrix = numpy.ones((128, 128), numpy.byte)
        pixel_matrix[0, 0] = 255  # 255 ubyte is -1 byte
        reference = sampledata.SRS_COLOMBIA
        nodata_base = -1
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_base, reference.pixel_size(30), datatype=gdal.GDT_Byte,
            filename=base_path,
            raster_driver_creation_tuple=('GTiff', [
                'PIXELTYPE=SIGNEDBYTE',
                'TILED=YES',
                'BLOCKXSIZE=64',
                'BLOCKYSIZE=64',
                ]))

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        # 255 should convert to -1 with signed bytes
        pygeoprocessing.new_raster_from_base(
            base_path, target_path, gdal.GDT_Byte, [0],
            fill_value_list=[255])

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_matrix = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        # we expect a negative result even though we put in a positive because
        # we know signed bytes will convert
        self.assertEqual(target_matrix[0, 0], -1)

    def test_new_raster_from_base_nodata_not_set(self):
        """PGP.geoprocessing: test new raster with nodata not set."""
        import pygeoprocessing

        driver = gdal.GetDriverByName('GTiff')
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        new_raster = driver.Create(base_path, 128, 128, 1, gdal.GDT_Int32)
        del new_raster

        target_path = os.path.join(self.workspace_dir, 'target.tif')
        # 255 should convert to -1 with signed bytes
        pygeoprocessing.new_raster_from_base(
            base_path, target_path, gdal.GDT_Byte, [None],
            fill_value_list=[None],
            raster_driver_creation_tuple=('GTiff', [
                'PIXELTYPE=SIGNEDBYTE',
                ]))

        raster_properties = pygeoprocessing.get_raster_info(target_path)
        self.assertEqual(raster_properties['nodata'], [None])

    def test_create_raster_from_vector_extents(self):
        """PGP.geoprocessing: test creation of raster from vector extents."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

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

    def test_create_raster_from_vector_extents_invalid_pixeltype(self):
        """PGP.geoprocessing: raster from vector with bad datatype."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

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
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.create_raster_from_vector_extents(
                source_vector_path, target_raster_path, target_pixel_size,
                target_nodata, target_pixel_type)
            expected_message = (
                'Invalid target type, should be a gdal.GDT_* type')
            actual_message = str(cm.exception)
            self.assertTrue(
                expected_message in actual_message, actual_message)

    def test_create_raster_from_vector_extents_odd_pixel_shapes(self):
        """PGP.geoprocessing: create raster vector ext. w/ odd pixel size."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

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

    def test_create_raster_from_vector_extents_linestring_no_width(self):
        """PGP.geoprocessing: create raster from v. ext with no geom width."""
        import pygeoprocessing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        point_a = shapely.geometry.LineString(
            [(reference.origin[0], reference.origin[1]),
             (reference.origin[0], reference.origin[1] + 100)])
        pixel_x_size = -10
        pixel_y_size = 20
        n_pixels_x = 1
        n_pixels_y = 5
        source_vector_path = os.path.join(self.workspace_dir, 'sample_vector')
        pygeoprocessing.testing.create_vector_on_disk(
            [point_a], reference.projection, fields={'value': 'int'},
            attributes=[{'value': 0}], vector_format='GeoJSON',
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

    def test_create_raster_from_vector_extents_linestring_no_height(self):
        """PGP.geoprocessing: create raster from v. ext with no geom height."""
        import pygeoprocessing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        point_a = shapely.geometry.LineString(
            [(reference.origin[0], reference.origin[1]),
             (reference.origin[0] + 100, reference.origin[1])])
        pixel_x_size = -10
        pixel_y_size = 20
        n_pixels_x = 10
        n_pixels_y = 1
        source_vector_path = os.path.join(self.workspace_dir, 'sample_vector')
        pygeoprocessing.testing.create_vector_on_disk(
            [point_a], reference.projection, fields={'value': 'int'},
            attributes=[{'value': 0}], vector_format='GeoJSON',
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
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        vector_driver = ogr.GetDriverByName('GeoJSON')
        source_vector_path = os.path.join(self.workspace_dir, 'vector.json')
        source_vector = vector_driver.CreateDataSource(source_vector_path)
        srs = osr.SpatialReference(reference.projection)
        source_layer = source_vector.CreateLayer('vector', srs=srs)

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
        ogr.DataSource.__swig_destroy__(source_vector)
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
        raster = gdal.OpenEx(target_raster_path, gdal.OF_RASTER)
        band = raster.GetRasterBand(1)
        result = band.ReadAsArray()
        band = None
        raster = None
        numpy.testing.assert_array_equal(expected_result, result)

    def test_transform_box(self):
        """PGP.geoprocessing: test geotransforming lat/lng box to UTM10N."""
        import pygeoprocessing

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
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

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
            raster_driver_creation_tuple=('GTiff', [
                'TILED=YES',
                'BLOCKXSIZE=64',
                'BLOCKYSIZE=64']))

        total = 0
        for _, block in pygeoprocessing.iterblocks(
                (raster_path, 1), largest_block=0):
            total += numpy.sum(block)
        self.assertEqual(total, test_value * n_pixels**2)

    def test_iterblocks_bad_raster_band(self):
        """PGP.geoprocessing: test iterblocks."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

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
            raster_driver_creation_tuple=('GTiff', [
                'TILED=YES',
                'BLOCKXSIZE=64',
                'BLOCKYSIZE=64']))

        total = 0
        with self.assertRaises(ValueError) as cm:
            for _, block in pygeoprocessing.iterblocks(
                    raster_path, largest_block=0):
                total += numpy.sum(block)
        expected_message = (
            "`raster_path_band` not formatted as expected.")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_iterblocks_unsigned_byte(self):
        """PGP.geoprocessing: test iterblocks with unsigned byte."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 100
        pixel_matrix = numpy.empty((n_pixels, n_pixels), numpy.uint8)
        test_value = 255
        pixel_matrix[:] = test_value
        nodata_target = None
        raster_path = os.path.join(self.workspace_dir, 'raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [pixel_matrix], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=raster_path,
            raster_driver_creation_tuple=('GTiff', [
                'TILED=YES',
                'BLOCKXSIZE=64',
                'BLOCKYSIZE=64']))

        total = 0
        for _, block in pygeoprocessing.iterblocks(
                (raster_path, 1), largest_block=0):
            total += numpy.sum(block)
        self.assertEqual(total, test_value * n_pixels**2)

    def test_convolve_2d_single_thread(self):
        """PGP.geoprocessing: test convolve 2d (single thread)."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 100
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        nodata_target = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [signal_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((3, 3), numpy.float32)
        pygeoprocessing.testing.create_raster_on_disk(
            [kernel_array], reference.origin, reference.projection,
            None, reference.pixel_size(30), filename=kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            n_threads=1, ignore_nodata=False)
        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None

        # calculate expected result by adding up all squares, subtracting off
        # the sides and realizing diagonals got subtracted twice
        expected_result = test_value * (
            n_pixels ** 2 * 9 - n_pixels * 4 * 3 + 4)
        self.assertEqual(numpy.sum(target_array), expected_result)

    def test_convolve_2d_multiprocess(self):
        """PGP.geoprocessing: test convolve 2d (multiprocess)."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 100
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        nodata_target = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [signal_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((3, 3), numpy.float32)
        pygeoprocessing.testing.create_raster_on_disk(
            [kernel_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            n_threads=3)
        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None

        # calculate expected result by adding up all squares, subtracting off
        # the sides and realizing diagonals got subtracted twice
        expected_result = test_value * (
            n_pixels ** 2 * 9 - n_pixels * 4 * 3 + 4)
        self.assertEqual(numpy.sum(target_array), expected_result)

    def test_convolve_2d_normalize_ignore_nodata(self):
        """PGP.geoprocessing: test convolve 2d w/ normalize and ignore."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 100
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        nodata_target = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [signal_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((3, 3), numpy.float32)
        pygeoprocessing.testing.create_raster_on_disk(
            [kernel_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            mask_nodata=False, ignore_nodata=True, normalize_kernel=True)
        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        expected_result = test_value * n_pixels ** 2
        self.assertEqual(numpy.sum(target_array), expected_result)

    def test_convolve_2d_ignore_nodata(self):
        """PGP.geoprocessing: test convolve 2d w/ normalize and ignore."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 3
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        nodata_target = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [signal_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((3, 3), numpy.float32)
        pygeoprocessing.testing.create_raster_on_disk(
            [kernel_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            ignore_nodata=True)
        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None

        # calculate by working on some graph paper
        expected_result = 9*9*.5
        self.assertEqual(numpy.sum(target_array), expected_result)

    def test_convolve_2d_normalize(self):
        """PGP.geoprocessing: test convolve 2d w/ normalize."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 3
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        nodata_target = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [signal_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((3, 3), numpy.float32)
        pygeoprocessing.testing.create_raster_on_disk(
            [kernel_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path,
            normalize_kernel=True)
        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None

        # I calculated this by manually doing a grid on graph paper
        expected_result = .5 + 4 * 5./9.
        self.assertAlmostEqual(numpy.sum(target_array), expected_result)

    def test_convolve_2d_missing_nodata(self):
        """PGP.geoprocessing: test convolve2d if target type but no nodata."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 100
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        nodata_target = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [signal_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((3, 3), numpy.float32)
        pygeoprocessing.testing.create_raster_on_disk(
            [kernel_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.convolve_2d(
                (signal_path, 1), (kernel_path, 1), target_path,
                target_datatype=gdal.GDT_Int32)
        expected_message = (
            "`target_datatype` is set, but `target_nodata` is None. ")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message)

    def test_convolve_2d_reverse(self):
        """PGP.geoprocessing: test convolve 2d reversed."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 3
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        nodata_target = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [signal_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.ones((100, 100), numpy.float32)
        pygeoprocessing.testing.create_raster_on_disk(
            [kernel_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path)
        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None

        # calculate expected result by adding up all squares, subtracting off
        # the sides and realizing diagonals got subtracted twice
        expected_result = test_value * (n_pixels ** 4)
        self.assertEqual(numpy.sum(target_array), expected_result)

    def test_convolve_2d_large(self):
        """PGP.geoprocessing: test convolve 2d with large kernel & signal."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 100
        n_kernel_pixels = 1750
        signal_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        signal_array[:] = test_value
        nodata_target = -1
        signal_path = os.path.join(self.workspace_dir, 'signal.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [signal_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=signal_path)
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        kernel_array = numpy.zeros(
            (n_kernel_pixels, n_kernel_pixels), numpy.float32)
        kernel_array[int(n_kernel_pixels/2), int(n_kernel_pixels/2)] = 1
        pygeoprocessing.testing.create_raster_on_disk(
            [kernel_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30), filename=kernel_path)
        target_path = os.path.join(self.workspace_dir, 'target.tif')
        pygeoprocessing.convolve_2d(
            (signal_path, 1), (kernel_path, 1), target_path)
        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None

        # calculate expected result by adding up all squares, subtracting off
        # the sides and realizing diagonals got subtracted twice
        expected_result = test_value * (n_pixels ** 2)
        self.assertEqual(numpy.sum(target_array), expected_result)

    def test_calculate_slope(self):
        """PGP.geoprocessing: test calculate slope."""
        import pygeoprocessing

        n_pixels = 9
        dem_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        dem_array[:] = numpy.arange((n_pixels))
        nodata_value = -1
        # make a nodata hole in the middle to test boundary cases
        dem_array[int(n_pixels/2), int(n_pixels/2)] = nodata_value
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        target_slope_path = os.path.join(self.workspace_dir, 'slope.tif')
        driver = gdal.GetDriverByName('GTiff')
        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)  # WGS84 EPSG
        dem_raster = driver.Create(
            dem_path, dem_array.shape[1], dem_array.shape[0],
            2, gdal.GDT_Int32)
        dem_raster_geotransform = [0.1, 1., 0., 0., 0., -1.]
        dem_raster.SetGeoTransform(dem_raster_geotransform)
        dem_raster.SetProjection(wgs84_ref.ExportToWkt())
        dem_band = dem_raster.GetRasterBand(1)
        dem_band.SetNoDataValue(nodata_value)
        dem_band.WriteArray(dem_array)
        dem_band.FlushCache()
        dem_band = None
        dem_raster = None

        pygeoprocessing.calculate_slope((dem_path, 1), target_slope_path)
        slope_raster = gdal.OpenEx(target_slope_path, gdal.OF_RASTER)
        slope_band = slope_raster.GetRasterBand(1)
        target_nodata = slope_band.GetNoDataValue()
        count = 0
        expected_slope = 100.0
        for _, band_data in pygeoprocessing.iterblocks(
                (target_slope_path, 1)):
            block = band_data.astype(numpy.float32)
            bad_mask = (
                ~numpy.isclose(block, target_nodata) &
                (block != expected_slope))
            if numpy.any(bad_mask):
                self.fail(
                    "Unexpected value in slope raster: %s" % block[bad_mask])
            count += numpy.count_nonzero(block == expected_slope)
        # all slopes should be 1 except center pixel
        self.assertEqual(count, n_pixels**2 - 1)

    def test_calculate_slope_undefined_nodata(self):
        """PGP.geoprocessing: test calculate slope with no nodata."""
        import pygeoprocessing

        n_pixels = 9
        dem_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        dem_path = os.path.join(self.workspace_dir, 'dem.tif')
        target_slope_path = os.path.join(self.workspace_dir, 'slope.tif')
        driver = gdal.GetDriverByName('GTiff')
        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)  # WGS84 EPSG
        dem_raster = driver.Create(
            dem_path, dem_array.shape[1], dem_array.shape[0],
            1, gdal.GDT_Int32)
        dem_raster_geotransform = [0.1, 1., 0., 0., 0., -1.]
        dem_raster.SetGeoTransform(dem_raster_geotransform)
        dem_raster.SetProjection(wgs84_ref.ExportToWkt())
        dem_band = dem_raster.GetRasterBand(1)
        dem_band.WriteArray(dem_array)
        dem_band.FlushCache()
        dem_band = None
        dem_raster = None

        pygeoprocessing.calculate_slope((dem_path, 1), target_slope_path)
        slope_raster = gdal.OpenEx(target_slope_path, gdal.OF_RASTER)
        slope_band = slope_raster.GetRasterBand(1)
        actual_slope = slope_band.ReadAsArray()
        slope_band = None
        slope_raster = None
        expected_slope = numpy.zeros((n_pixels, n_pixels), numpy.float32)
        numpy.testing.assert_almost_equal(expected_slope, actual_slope)

    def test_rasterize(self):
        """PGP.geoprocessing: test rasterize."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 3
        target_raster_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        target_raster_array[:] = test_value
        nodata_target = -1
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [target_raster_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30),
            filename=target_raster_path)

        reference = sampledata.SRS_COLOMBIA
        pixel_size = 30.0
        polygon = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.json')
        pygeoprocessing.testing.create_vector_on_disk(
            [polygon], reference.projection,
            fields={'id': 'int'}, attributes=[{'id': 5}],
            vector_format='GeoJSON', filename=base_vector_path)

        pygeoprocessing.rasterize(
            base_vector_path, target_raster_path, [test_value], None,
            layer_id=0)

        target_raster = gdal.OpenEx(target_raster_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        result = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        self.assertTrue((result == test_value).all())

        pygeoprocessing.rasterize(
            base_vector_path, target_raster_path, None,
            ["ATTRIBUTE=id"], layer_id=0)
        target_raster = gdal.OpenEx(target_raster_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        result = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        self.assertTrue((result == 5).all())

    def test_rasterize_error(self):
        """PGP.geoprocessing: test rasterize when error encountered."""
        import pygeoprocessing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 3
        target_raster_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        target_raster_array[:] = test_value
        nodata_target = -1
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [target_raster_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30),
            filename=target_raster_path)

        reference = sampledata.SRS_COLOMBIA
        pixel_size = 30.0
        polygon = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.json')

        pygeoprocessing.testing.create_vector_on_disk(
            [polygon], reference.projection,
            fields={'id': 'int'}, attributes=[{'id': 5}],
            vector_format='GeoJSON', filename=base_vector_path)

        with self.assertRaises(RuntimeError) as cm:
            # Patching the function that makes a logger callback so that
            # it will raise an exception (ZeroDivisionError in this case,
            # but any exception should do).
            with mock.patch(
                    'pygeoprocessing.geoprocessing._make_logger_callback',
                    return_value=lambda x, y, z: 1/0.):
                pygeoprocessing.rasterize(
                    base_vector_path, target_raster_path, [test_value], None,
                    layer_id=0)

        self.assertTrue('nonzero exit code' in str(cm.exception))

    def test_rasterize_missing_file(self):
        """PGP.geoprocessing: test rasterize with no target raster."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 3
        target_raster_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        target_raster_array[:] = test_value
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')

        # intentionally not making the raster on disk
        reference = sampledata.SRS_COLOMBIA
        pixel_size = 30.0
        polygon = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.json')
        pygeoprocessing.testing.create_vector_on_disk(
            [polygon], reference.projection,
            fields={'id': 'int'}, attributes=[{'id': 5}],
            vector_format='GeoJSON', filename=base_vector_path)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.rasterize(
                base_vector_path, target_raster_path, [test_value], None,
                layer_id=0)
        expected_message = (
            "%s doesn't exist, but needed to rasterize." % target_raster_path)
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_rasterize_error_handling(self):
        """PGP.geoprocessing: test rasterize error handling."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 3
        target_raster_array = numpy.ones((n_pixels, n_pixels), numpy.float32)
        test_value = 0.5
        target_raster_array[:] = test_value
        target_raster_path = os.path.join(
            self.workspace_dir, 'target_raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [target_raster_array], reference.origin, reference.projection,
            -1, reference.pixel_size(30),
            filename=target_raster_path)

        # intentionally not making the raster on disk
        reference = sampledata.SRS_COLOMBIA
        pixel_size = 30.0
        polygon = shapely.geometry.Polygon([
            (reference.origin[0], reference.origin[1]),
            (reference.origin[0], -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels,
             -pixel_size * n_pixels+reference.origin[1]),
            (reference.origin[0]+pixel_size * n_pixels, reference.origin[1]),
            (reference.origin[0], reference.origin[1])])
        base_vector_path = os.path.join(
            self.workspace_dir, 'base_vector.json')
        pygeoprocessing.testing.create_vector_on_disk(
            [polygon], reference.projection,
            fields={'id': 'int'}, attributes=[{'id': 5}],
            vector_format='GeoJSON', filename=base_vector_path)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.rasterize(
                base_vector_path, target_raster_path, None, None,
                layer_id=0)
        expected_message = (
            "Neither `burn_values` nor `option_list` is set")
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.rasterize(
                base_vector_path, target_raster_path, 1, None,
                layer_id=0)
        expected_message = "`burn_values` is not a list/tuple"
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.rasterize(
                base_vector_path, target_raster_path, None, "ATTRIBUTE=id",
                layer_id=0)
        expected_message = "`option_list` is not a list/tuple"
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_distance_transform_edt(self):
        """PGP.geoprocessing: test distance transform EDT."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 1000
        nodata_target = 0
        base_raster_array = numpy.zeros(
            (n_pixels, n_pixels), dtype=numpy.int)
        base_raster_array[:, n_pixels//2:] = nodata_target
        base_raster_array[int(n_pixels/2), int(n_pixels/2)] = 1
        base_raster_array[0, 0] = 1
        base_raster_array[0, n_pixels-1] = 1
        base_raster_array[3, 4] = 1
        base_raster_array[3, 5] = 1
        base_raster_array[3, 6] = 1
        base_raster_array[4, 4] = 1
        base_raster_array[4, 5] = 1
        base_raster_array[4, 6] = 1
        base_raster_array[5, 4] = 1
        base_raster_array[5, 5] = 1
        base_raster_array[5, 6] = 1
        base_raster_array[n_pixels-1, 0] = 1
        base_raster_array[n_pixels-1, n_pixels-1] = 1
        base_raster_array[int(n_pixels/2), int(n_pixels/2)] = 1
        base_raster_array[int(n_pixels/2), int(n_pixels/4)] = 1
        base_raster_array[int(n_pixels/2), int((3*n_pixels)/4)] = 1
        base_raster_path = os.path.join(self.workspace_dir, 'base_raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [base_raster_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30),
            filename=base_raster_path)

        target_distance_raster_path = os.path.join(
            self.workspace_dir, 'target_distance.tif')

        for sampling_distance in [(200.0, 1.5), (1.5, 200.0)]:
            pygeoprocessing.distance_transform_edt(
                (base_raster_path, 1), target_distance_raster_path,
                sampling_distance=sampling_distance,
                working_dir=self.workspace_dir)
            target_raster = gdal.OpenEx(
                target_distance_raster_path, gdal.OF_RASTER)
            target_band = target_raster.GetRasterBand(1)
            target_array = target_band.ReadAsArray()
            target_band = None
            target_raster = None
            expected_result = scipy.ndimage.morphology.distance_transform_edt(
                1 - (base_raster_array == 1), sampling=(
                    sampling_distance[1], sampling_distance[0]))
            numpy.testing.assert_array_almost_equal(
                target_array, expected_result, decimal=2)

        base_raster_path = os.path.join(
            self.workspace_dir, 'undefined_nodata_base_raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [base_raster_array], reference.origin, reference.projection,
            None, reference.pixel_size(30),
            filename=base_raster_path)
        pygeoprocessing.distance_transform_edt(
            (base_raster_path, 1), target_distance_raster_path,
            sampling_distance=sampling_distance,
            working_dir=self.workspace_dir)
        target_raster = gdal.OpenEx(
            target_distance_raster_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        numpy.testing.assert_array_almost_equal(
            target_array, expected_result, decimal=2)

    def test_distance_transform_edt_small_sample_distance(self):
        """PGP.geoprocessing: test distance transform w/ small sample dist."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 10
        nodata_target = None
        base_raster_array = numpy.zeros(
            (n_pixels, n_pixels), dtype=numpy.int)
        base_raster_array[n_pixels//2:, :] = 1
        base_raster_path = os.path.join(self.workspace_dir, 'base_raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [base_raster_array], reference.origin, reference.projection,
            nodata_target, reference.pixel_size(30),
            filename=base_raster_path)

        target_distance_raster_path = os.path.join(
            self.workspace_dir, 'target_distance.tif')

        sampling_distance = (0.1, 0.1)
        pygeoprocessing.distance_transform_edt(
            (base_raster_path, 1), target_distance_raster_path,
            sampling_distance=sampling_distance,
            working_dir=self.workspace_dir)
        target_raster = gdal.OpenEx(
            target_distance_raster_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        expected_result = scipy.ndimage.morphology.distance_transform_edt(
            1 - (base_raster_array == 1), sampling=(
                sampling_distance[1], sampling_distance[0]))
        numpy.testing.assert_array_almost_equal(
            target_array, expected_result, decimal=2)

    def test_distance_transform_edt_bad_data(self):
        """PGP.geoprocessing: test distance transform EDT with bad values."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        reference = sampledata.SRS_COLOMBIA
        n_pixels = 10
        base_raster_array = numpy.zeros(
            (n_pixels, n_pixels), dtype=numpy.int)
        base_raster_array[int(n_pixels/2), int(n_pixels/2)] = 1
        base_raster_array[0, 0] = 1
        base_raster_array[0, n_pixels-1] = 1
        base_raster_array[3, 4] = 1
        base_raster_array[3, 5] = 1
        base_raster_array[3, 6] = 1
        base_raster_array[4, 4] = 1
        base_raster_array[4, 5] = 1
        base_raster_array[4, 6] = 1
        base_raster_array[5, 4] = 1
        base_raster_array[5, 5] = 1
        base_raster_array[5, 6] = 1
        base_raster_array[n_pixels-1, 0] = 1
        base_raster_array[n_pixels-1, n_pixels-1] = 1
        base_raster_array[int(n_pixels/2), int(n_pixels/2)] = 1
        base_raster_array[int(n_pixels/2), int(n_pixels/4)] = 1
        base_raster_array[int(n_pixels/2), int((3*n_pixels)/4)] = 1
        base_raster_path = os.path.join(self.workspace_dir, 'base_raster.tif')
        pygeoprocessing.testing.create_raster_on_disk(
            [base_raster_array], reference.origin, reference.projection,
            None, reference.pixel_size(30),
            filename=base_raster_path)

        target_distance_raster_path = os.path.join(
            self.workspace_dir, 'target_distance.tif')

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.distance_transform_edt(
                (base_raster_path, 1), target_distance_raster_path,
                working_dir=self.workspace_dir,
                sampling_distance=1.0)
        expected_message = '`sampling_distance` should be a tuple/list'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.distance_transform_edt(
                (base_raster_path, 1), target_distance_raster_path,
                working_dir=self.workspace_dir,
                sampling_distance=(1.0, -1.0))
        expected_message = 'Sample distances must be > 0.0'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_next_regular(self):
        """PGP.geoprocessing: test next regular number generator."""
        import pygeoprocessing.geoprocessing
        # just test the first few numbers in the A051037 series
        regular_ints = [
            1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30,
            32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100,
            108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192, 200, 216,
            225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 384,
            400, 405]

        next_int = 0
        for regular_int in regular_ints:
            next_int = pygeoprocessing.geoprocessing._next_regular(next_int+1)
            self.assertEqual(next_int, regular_int)

    def test_merge_rasters(self):
        """PGP.geoprocessing: test merge_rasters."""
        import pygeoprocessing

        driver = gdal.GetDriverByName('GTiff')

        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)  # WGS84 EPSG

        # the following creates a checkerboard of upper left square raster
        # defined, lower right, and equal sized nodata chunks on the other
        # blocks.

        raster_a_path = os.path.join(self.workspace_dir, 'raster_a.tif')
        # everything flows to the right
        raster_a_array = numpy.zeros((128, 128), dtype=numpy.int32)
        raster_a_array[:] = 10
        raster_a = driver.Create(
            raster_a_path, raster_a_array.shape[1], raster_a_array.shape[0],
            2, gdal.GDT_Int32)
        raster_a_geotransform = [0.1, 1., 0., 0., 0., -1.]
        raster_a.SetGeoTransform(raster_a_geotransform)
        raster_a.SetProjection(wgs84_ref.ExportToWkt())
        band = raster_a.GetRasterBand(1)
        band.WriteArray(raster_a_array)
        band.FlushCache()
        band = None
        raster_a = None

        raster_b_path = os.path.join(self.workspace_dir, 'raster_b.tif')
        raster_b_array = numpy.zeros((128, 128), dtype=numpy.int32)
        raster_b_array[:] = 20
        raster_b = driver.Create(
            raster_b_path, raster_b_array.shape[1], raster_b_array.shape[0],
            2, gdal.GDT_Int32)
        raster_b.SetProjection(wgs84_ref.ExportToWkt())
        raster_b_geotransform = [128.1, 1, 0, -128, 0, -1]
        raster_b.SetGeoTransform(raster_b_geotransform)
        band = raster_b.GetRasterBand(1)
        band.WriteArray(raster_b_array)
        band.FlushCache()
        raster_b = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        pygeoprocessing.merge_rasters(
            [raster_a_path, raster_b_path], target_path)

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        self.assertEqual(target_band.GetNoDataValue(), None)
        target_array = target_band.ReadAsArray()
        target_band = None
        expected_array = numpy.zeros((256, 256))
        expected_array[0:128, 0:128] = 10
        expected_array[128:, 128:] = 20
        numpy.testing.assert_almost_equal(target_array, expected_array)

        target_band = target_raster.GetRasterBand(2)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        expected_array = numpy.zeros((256, 256))
        numpy.testing.assert_almost_equal(target_array, expected_array)

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        pygeoprocessing.merge_rasters(
            [raster_a_path, raster_b_path], target_path,
            bounding_box=[4, -6, 6, -4])

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        self.assertEqual(target_band.GetNoDataValue(), None)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        expected_array = numpy.empty((2, 2))
        expected_array[:] = 10
        numpy.testing.assert_almost_equal(target_array, expected_array)

    def test_merge_rasters_target_nodata(self):
        """PGP.geoprocessing: test merge_rasters with defined nodata."""
        import pygeoprocessing

        driver = gdal.GetDriverByName('GTiff')

        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)  # WGS84 EPSG

        # the following creates a checkerboard of upper left square raster
        # defined, lower right, and equal sized nodata chunks on the other
        # blocks.

        raster_a_path = os.path.join(self.workspace_dir, 'raster_a.tif')
        # everything flows to the right
        raster_a_array = numpy.zeros((11, 11), dtype=numpy.int32)
        raster_a_array[:] = 10
        raster_a = driver.Create(
            raster_a_path, raster_a_array.shape[1], raster_a_array.shape[0],
            2, gdal.GDT_Int32)
        raster_a_geotransform = [0.1, 1., 0., 0., 0., -1.]
        raster_a.SetGeoTransform(raster_a_geotransform)
        raster_a.SetProjection(wgs84_ref.ExportToWkt())
        band = raster_a.GetRasterBand(1)
        band.WriteArray(raster_a_array)
        band.FlushCache()
        band = None
        raster_a = None

        raster_b_path = os.path.join(self.workspace_dir, 'raster_b.tif')
        raster_b_array = numpy.zeros((11, 11), dtype=numpy.int32)
        raster_b_array[:] = 20
        raster_b = driver.Create(
            raster_b_path, raster_b_array.shape[1], raster_b_array.shape[0],
            2, gdal.GDT_Int32)
        raster_b.SetProjection(wgs84_ref.ExportToWkt())
        raster_b_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_b.SetGeoTransform(raster_b_geotransform)
        band = raster_b.GetRasterBand(1)
        band.WriteArray(raster_b_array)
        band.FlushCache()
        band = None
        raster_b = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        pygeoprocessing.merge_rasters(
            [raster_a_path, raster_b_path], target_path, target_nodata=0)

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        nodata_value = target_raster.GetRasterBand(2).GetNoDataValue()
        target_band = None
        target_raster = None
        expected_array = numpy.zeros((22, 22))
        expected_array[0:11, 0:11] = 10
        expected_array[11:, 11:] = 20

        numpy.testing.assert_almost_equal(target_array, expected_array)
        self.assertEqual(nodata_value, 0)

    def test_merge_rasters_exception_cover(self):
        """PGP.geoprocessing: test merge_rasters with bad data."""
        import pygeoprocessing

        driver = gdal.GetDriverByName('GTiff')

        wgs84_ref = osr.SpatialReference()
        wgs84_ref.ImportFromEPSG(4326)  # WGS84 EPSG

        # the following creates a checkerboard of upper left square raster
        # defined, lower right, and equal sized nodata chunks on the other
        # blocks.

        raster_a_path = os.path.join(self.workspace_dir, 'raster_a.tif')
        # everything flows to the right
        raster_a_array = numpy.zeros((11, 11), dtype=numpy.int32)
        raster_a_array[:] = 10
        raster_a = driver.Create(
            raster_a_path, raster_a_array.shape[1], raster_a_array.shape[0],
            2, gdal.GDT_Int32)
        raster_a_geotransform = [0.1, 1., 0., 0., 0., -1.]
        raster_a.SetGeoTransform(raster_a_geotransform)
        raster_a.SetProjection(wgs84_ref.ExportToWkt())
        band = raster_a.GetRasterBand(1)
        band.WriteArray(raster_a_array)
        band.FlushCache()
        band = None
        raster_a = None

        raster_b_path = os.path.join(self.workspace_dir, 'raster_b.tif')
        raster_b_array = numpy.zeros((11, 11), dtype=numpy.int32)
        raster_b_array[:] = 20
        raster_b = driver.Create(
            raster_b_path, raster_b_array.shape[1], raster_b_array.shape[0],
            2, gdal.GDT_Int32)
        raster_b.SetProjection(wgs84_ref.ExportToWkt())
        raster_b_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_b.SetGeoTransform(raster_b_geotransform)
        band = raster_b.GetRasterBand(1)
        band.SetNoDataValue(-1.0)
        band.WriteArray(raster_b_array)
        band.FlushCache()
        band = None
        raster_b = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.merge_rasters(
                [raster_a_path, raster_b_path], target_path)
        expected_message = 'Nodata per raster are not the same'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        raster_c_path = os.path.join(self.workspace_dir, 'raster_c.tif')
        raster_c_array = numpy.zeros((11, 11), dtype=numpy.int32)
        raster_c_array[:] = 20
        raster_c = driver.Create(
            raster_c_path, raster_c_array.shape[1], raster_c_array.shape[0],
            1, gdal.GDT_Int32)
        raster_c.SetProjection(wgs84_ref.ExportToWkt())
        raster_c_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_c.SetGeoTransform(raster_c_geotransform)
        band = raster_c.GetRasterBand(1)
        band.SetNoDataValue(-1.0)
        band.WriteArray(raster_c_array)
        band.FlushCache()
        band = None
        raster_c = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.merge_rasters(
                [raster_a_path, raster_c_path], target_path)
        expected_message = 'Number of bands per raster are not the same.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        raster_d_path = os.path.join(self.workspace_dir, 'raster_d.tif')
        raster_d_array = numpy.zeros((11, 11), dtype=numpy.float32)
        raster_d_array[:] = 20
        raster_d = driver.Create(
            raster_d_path, raster_d_array.shape[1], raster_d_array.shape[0],
            2, gdal.GDT_Float32)
        raster_d.SetProjection(wgs84_ref.ExportToWkt())
        raster_d_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_d.SetGeoTransform(raster_d_geotransform)
        band = raster_d.GetRasterBand(1)
        band.SetNoDataValue(-1.0)
        band.WriteArray(raster_d_array)
        band.FlushCache()
        band = None
        raster_d = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.merge_rasters(
                [raster_a_path, raster_d_path], target_path)
        expected_message = 'Rasters have different datatypes.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        raster_e_path = os.path.join(self.workspace_dir, 'raster_e.tif')
        raster_e_array = numpy.zeros((11, 11), dtype=numpy.int32)
        raster_e_array[:] = 20
        raster_e = driver.Create(
            raster_e_path, raster_e_array.shape[1], raster_e_array.shape[0],
            2, gdal.GDT_Int32)
        utm10_ref = osr.SpatialReference()
        utm10_ref.ImportFromEPSG(26910)
        raster_e.SetProjection(utm10_ref.ExportToWkt())
        raster_e_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_e.SetGeoTransform(raster_e_geotransform)
        band = raster_e.GetRasterBand(1)
        band.WriteArray(raster_e_array)
        band.FlushCache()
        band = None
        raster_e = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.merge_rasters(
                [raster_a_path, raster_e_path], target_path)
        expected_message = 'Projections are not identical.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        raster_f_path = os.path.join(self.workspace_dir, 'raster_f.tif')
        raster_f_array = numpy.zeros((11, 11), dtype=numpy.uint8)
        raster_f_array[:] = 20
        raster_f = driver.Create(
            raster_f_path, raster_f_array.shape[1], raster_f_array.shape[0],
            2, gdal.GDT_Byte)
        utm10_ref = osr.SpatialReference()
        utm10_ref.ImportFromEPSG(26910)
        raster_f.SetProjection(utm10_ref.ExportToWkt())
        raster_f_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_f.SetGeoTransform(raster_f_geotransform)
        band = raster_f.GetRasterBand(1)
        band.WriteArray(raster_f_array)
        band.FlushCache()
        band = None
        raster_f = None

        raster_g_path = os.path.join(self.workspace_dir, 'raster_g.tif')
        raster_g_array = numpy.zeros((11, 11), dtype=numpy.int8)
        raster_g_array[:] = 20
        raster_g = driver.Create(
            raster_g_path, raster_g_array.shape[1], raster_g_array.shape[0],
            2, gdal.GDT_Byte, options=['PIXELTYPE=SIGNEDBYTE'])
        utm10_ref = osr.SpatialReference()
        utm10_ref.ImportFromEPSG(26910)
        raster_g.SetProjection(utm10_ref.ExportToWkt())
        raster_g_geotransform = [11.1, 1, 0, -11, 0, -1]
        raster_g.SetGeoTransform(raster_g_geotransform)
        band = raster_g.GetRasterBand(1)
        band.WriteArray(raster_g_array)
        band.FlushCache()
        band = None
        raster_g = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.merge_rasters(
                [raster_f_path, raster_g_path], target_path)
        expected_message = 'PIXELTYPE different between rasters'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        raster_h_path = os.path.join(self.workspace_dir, 'raster_h.tif')
        raster_h_array = numpy.zeros((11, 11), dtype=numpy.int8)
        raster_h_array[:] = 20
        raster_h = driver.Create(
            raster_h_path, raster_h_array.shape[1], raster_h_array.shape[0],
            2, gdal.GDT_Int32)
        utm10_ref = osr.SpatialReference()
        raster_h.SetProjection(wgs84_ref.ExportToWkt())
        raster_h_geotransform = [11.1, 2, 0, -11, 0, -2]
        raster_h.SetGeoTransform(raster_h_geotransform)
        band = raster_h.GetRasterBand(1)
        band.WriteArray(raster_h_array)
        band.FlushCache()
        band = None
        raster_h = None

        target_path = os.path.join(self.workspace_dir, 'merged.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.merge_rasters(
                [raster_a_path, raster_h_path], target_path)
        expected_message = 'Pixel sizes of all rasters are not the same.'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_align_with_target_sr(self):
        """PGP: test align_and_resize_raster_stack with a target sr."""
        import pygeoprocessing

        wgs84_sr = osr.SpatialReference()
        wgs84_sr.ImportFromEPSG(4326)  # WGS84 EPSG

        driver = gdal.GetDriverByName("GTiff")
        base_path = os.path.join(self.workspace_dir, 'base.tif')
        new_raster = driver.Create(base_path, 10, 10, 1, gdal.GDT_Int32)
        new_raster.FlushCache()

        new_raster.SetProjection(wgs84_sr.ExportToWkt())
        new_raster.SetGeoTransform([
            -123.587984, 1.0, 0.0, 44.725814, 0.0, -1.0])
        new_raster = None

        target_path = os.path.join(self.workspace_dir, 'target.tif')

        target_ref = osr.SpatialReference()
        target_ref.ImportFromEPSG(26910)  # UTM10N EPSG

        pygeoprocessing.align_and_resize_raster_stack(
            [base_path], [target_path], ['near'],
            (3e4, -3e4), 'intersection',
            target_sr_wkt=target_ref.ExportToWkt())

        target_raster_info = pygeoprocessing.get_raster_info(target_path)

        # I have confidence in the upper left bounding box coordinate
        # hardcoded below because I plotted the WGS84 and UTM10N rasters on
        # top of each other and moused over the upper right hand corner.
        # Note the warping of wgs84 to utm will cause distortion and this
        # function attempts to make a raster that bounds that distortion
        # without losing any data, so a direct mapping from the lat/lng
        # coordinate to the one below will be slightly of because the
        # warped raster's bounding box will be a little larger.
        self.assertIs(
            numpy.testing.assert_allclose(
                [446166.79245811916, 5012714.829567],
                [target_raster_info['bounding_box'][0],
                 target_raster_info['bounding_box'][3]]), None)

    def test_get_raster_info_error_handling(self):
        """PGP: test that bad data raise good errors in get_raster_info."""
        import pygeoprocessing

        # check for missing file
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.get_raster_info(
                os.path.join(self.workspace_dir, 'not_a_file.tif'))
        expected_message = 'Could not open'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        # check that file exists but is not a raster.
        not_a_raster_path = os.path.join(
            self.workspace_dir, 'not_a_raster.tif')
        with open(not_a_raster_path, 'w') as not_a_raster_file:
            not_a_raster_file.write("this is not a raster.\n")
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.get_raster_info(not_a_raster_path)
        expected_message = 'Could not open'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_get_vector_info_error_handling(self):
        """PGP: test that bad data raise good errors in get_vector_info."""
        import pygeoprocessing

        # check for missing file
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.get_vector_info(
                os.path.join(self.workspace_dir, 'not_a_file.tif'))
        expected_message = 'Could not open'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        # check that file exists but is not a vector
        not_a_vector_path = os.path.join(
            self.workspace_dir, 'not_a_vector')
        os.makedirs(not_a_vector_path)
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.get_raster_info(not_a_vector_path)
        expected_message = 'Could not open'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_merge_bounding_box_list(self):
        """PGP: test merge_bounding_box_list."""
        import pygeoprocessing

        bb_a = (-1, -1, 1, 1)
        bb_b = (-.9, -1.5, 1.5, 1)

        union_bb = pygeoprocessing.merge_bounding_box_list(
            [bb_a, bb_b], 'union')
        numpy.testing.assert_array_almost_equal(
            union_bb, [-1, -1.5, 1.5, 1])
        intersection_bb = pygeoprocessing.merge_bounding_box_list(
            [bb_a, bb_b], 'intersection')
        numpy.testing.assert_array_almost_equal(
            intersection_bb, [-.9, -1, 1, 1])

    def test_align_and_resize_raster_stack_int_with_vector_mask(self):
        """PGP.geoprocessing: align/resize raster w/ vector mask."""
        import pygeoprocessing
        import pygeoprocessing.testing
        from pygeoprocessing.testing import sampledata

        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        reference = sampledata.SRS_COLOMBIA
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')

        geotiff_driver = gdal.GetDriverByName('GTiff')
        base_raster = geotiff_driver.Create(
            base_a_path, 5, 5, 1, gdal.GDT_Byte)
        pixel_size = 30
        base_raster.SetGeoTransform(
            [reference.origin[0], pixel_size, 0,
             reference.origin[1], 0, -pixel_size])
        base_raster.SetProjection(reference.projection)
        base_band = base_raster.GetRasterBand(1)
        base_band.WriteArray(pixel_a_matrix)
        base_band.SetNoDataValue(nodata_target)
        base_band.FlushCache()
        base_raster.FlushCache()
        base_band = None
        base_raster = None

        resample_method_list = ['near']
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        # make a vector whose bounding box is 1 pixel large
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(reference.origin[0], reference.origin[1])
        ring.AddPoint(
            reference.origin[0] + reference.pixel_size(30)[0],
            reference.origin[1])
        ring.AddPoint(
            reference.origin[0] + reference.pixel_size(30)[0],
            reference.origin[1] + reference.pixel_size(30)[1])
        ring.AddPoint(
            reference.origin[0], reference.origin[1] +
            reference.pixel_size(30)[1])
        ring.AddPoint(reference.origin[0], reference.origin[1])
        poly_a = ogr.Geometry(ogr.wkbPolygon)
        poly_a.AddGeometry(ring)

        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(
            reference.origin[0] + 2*reference.pixel_size(30)[0],
            reference.origin[1] + 2*reference.pixel_size(30)[1])
        ring.AddPoint(
            reference.origin[0] + 3*reference.pixel_size(30)[0],
            reference.origin[1] + 2*reference.pixel_size(30)[1])
        ring.AddPoint(
            reference.origin[0] + 3*reference.pixel_size(30)[0],
            reference.origin[1] + 3*reference.pixel_size(30)[1])
        ring.AddPoint(
            reference.origin[0] + 2*reference.pixel_size(30)[0],
            reference.origin[1] + 3*reference.pixel_size(30)[1])
        ring.AddPoint(
            reference.origin[0] + 2*reference.pixel_size(30)[0],
            reference.origin[1] + 2*reference.pixel_size(30)[1])
        poly_b = ogr.Geometry(ogr.wkbPolygon)
        poly_b.AddGeometry(ring)

        dual_poly_path = os.path.join(self.workspace_dir, 'dual_poly.gpkg')
        vector_driver = gdal.GetDriverByName('GPKG')
        poly_vector = vector_driver.Create(
            dual_poly_path, 0, 0, 0, gdal.GDT_Unknown)
        reference_srs = osr.SpatialReference()
        reference_srs.ImportFromWkt(reference.projection)
        poly_layer = poly_vector.CreateLayer(
            'dual_poly', reference_srs, ogr.wkbPolygon)
        poly_layer.CreateField(ogr.FieldDefn('value', ogr.OFTInteger))
        poly_feature = ogr.Feature(poly_layer.GetLayerDefn())
        poly_feature.SetGeometry(poly_a)
        poly_feature.SetField('value', 100)
        poly_layer.CreateFeature(poly_feature)

        poly_feature = ogr.Feature(poly_layer.GetLayerDefn())
        poly_feature.SetGeometry(poly_b)
        poly_feature.SetField('value', 1)
        poly_layer.CreateFeature(poly_feature)
        poly_layer.SyncToDisk()
        poly_vector.FlushCache()
        poly_layer = None
        poly_vector = None

        target_path = os.path.join(self.workspace_dir, 'target_a.tif')
        pygeoprocessing.align_and_resize_raster_stack(
            [base_a_path], [target_path],
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            raster_align_index=0,
            target_sr_wkt=reference.projection,
            vector_mask_options={
                'mask_vector_path': dual_poly_path,
                'mask_layer_name': 'dual_poly',
            },
            gdal_warp_options=["CUTLINE_ALL_TOUCHED=FALSE"])

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        # the first pass doesn't do any filtering, so we should have 2 pixels
        self.assertEqual(
            numpy.count_nonzero(target_array[target_array == 1]), 2)

        # now test where only one of the polygons match
        pygeoprocessing.align_and_resize_raster_stack(
            [base_a_path], [target_path],
            resample_method_list,
            base_a_raster_info['pixel_size'], bounding_box_mode,
            raster_align_index=0,
            vector_mask_options={
                'mask_vector_path': dual_poly_path,
                'mask_layer_name': 'dual_poly',
                'mask_vector_where_filter': 'value=1'
            })

        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER)
        target_band = target_raster.GetRasterBand(1)
        target_array = target_band.ReadAsArray()
        target_band = None
        target_raster = None
        # we should have only one pixel left
        self.assertEqual(
            numpy.count_nonzero(target_array[target_array == 1]), 1)

    def test_align_and_resize_raster_stack_int_with_bad_vector_mask(self):
        """PGP.geoprocessing: align/resize raster w/ bad vector mask."""
        import pygeoprocessing

        pixel_a_matrix = numpy.ones((5, 5), numpy.int16)
        nodata_target = -1
        base_a_path = os.path.join(self.workspace_dir, 'base_a.tif')

        geotiff_driver = gdal.GetDriverByName('GTiff')
        base_raster = geotiff_driver.Create(
            base_a_path, 10, 10, 1, gdal.GDT_Byte)
        pixel_size = 30
        base_raster.SetGeoTransform([0.1, pixel_size, 0, 0.1, 0, -pixel_size])
        base_band = base_raster.GetRasterBand(1)
        base_band.WriteArray(pixel_a_matrix)
        base_band.SetNoDataValue(nodata_target)
        base_band.FlushCache()
        base_raster.FlushCache()
        base_band = None
        base_raster = None

        resample_method_list = ['near']
        bounding_box_mode = 'intersection'

        base_a_raster_info = pygeoprocessing.get_raster_info(base_a_path)

        # make a vector whose bounding box is 1 pixel large
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(1e3, 0)
        ring.AddPoint(1e3 + pixel_size, 0)
        ring.AddPoint(1e3 + pixel_size, pixel_size)
        ring.AddPoint(1e3 + pixel_size, 0)
        ring.AddPoint(1e3, 0)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        dual_poly_path = os.path.join(self.workspace_dir, 'dual_poly')
        vector_driver = gdal.GetDriverByName('ESRI Shapefile')
        poly_vector = vector_driver.Create(
            dual_poly_path, 0, 0, 0, gdal.GDT_Unknown)
        poly_layer = poly_vector.CreateLayer(
            'poly_vector', None, ogr.wkbPolygon)
        poly_feature = ogr.Feature(poly_layer.GetLayerDefn())
        poly_feature.SetGeometry(poly)
        poly_layer.CreateFeature(poly_feature)
        poly_layer.SyncToDisk()
        poly_layer = None
        poly_vector = None

        target_path = os.path.join(self.workspace_dir, 'target_a.tif')
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.align_and_resize_raster_stack(
                [base_a_path], [target_path],
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                raster_align_index=0,
                vector_mask_options={
                    'mask_vector_path': dual_poly_path,
                    'mask_layer_name': 'dual_poly',
                })
        expected_message = 'Bounding boxes do not intersect'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.align_and_resize_raster_stack(
                [base_a_path], [target_path],
                resample_method_list,
                base_a_raster_info['pixel_size'], bounding_box_mode,
                raster_align_index=0,
                vector_mask_options={
                    'bad_mask_vector_path': dual_poly_path,
                    'mask_layer_name': 'dual_poly',
                })
        expected_message = 'no value for "mask_vector_path"'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.warp_raster(
                base_a_path, base_a_raster_info['pixel_size'],
                target_path, 'near',
                vector_mask_options={
                    'bad_mask_vector_path': dual_poly_path,
                    'mask_layer_name': 'dual_poly',
                })
        expected_message = 'no value for "mask_vector_path"'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.warp_raster(
                base_a_path, base_a_raster_info['pixel_size'],
                target_path, 'near',
                vector_mask_options={
                    'mask_vector_path': 'not_a_file.shp',
                    'mask_layer_name': 'dual_poly',
                })
        expected_message = 'was not found'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_disjoint_polygon_set_no_bounding_box(self):
        """PGP.geoprocessing: check disjoint sets."""
        import pygeoprocessing

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32731)  # WGS84 / UTM zone 31s
        srs_wkt = srs.ExportToWkt()

        def square(centerpoint_tuple):
            x, y = centerpoint_tuple
            return shapely.geometry.Polygon(
                [(x-1.5, y-1.5),
                 (x-1.5, y+1.5),
                 (x+1.5, y+1.5),
                 (x+1.5, y-1.5),
                 (x-1.5, y-1.5)])

        watershed_geometries = [
            square((16, -8)),
            square((8, -10)),
            square((2, -8)),  # overlap with FID 4
            square((2, -7)),  # overlap with FID 3
            square((14, -12)),
        ]

        outflow_vector = os.path.join(self.workspace_dir, 'outflow.gpkg')
        pygeoprocessing.testing.create_vector_on_disk(
            watershed_geometries, srs_wkt, vector_format='GPKG',
            filename=outflow_vector)

        disjoint_sets = pygeoprocessing.calculate_disjoint_polygon_set(
            outflow_vector)
        self.assertEqual(
            disjoint_sets,
            [set([1, 2, 3, 5]), set([4])])

    def test_disjoint_polygon_set_no_features_error(self):
        """PGP.geoprocessing: raise an error when a vector has no features."""
        import pygeoprocessing
        import pygeoprocessing.testing

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4623)

        empty_vector_path = os.path.join(self.workspace_dir, 'empty.geojson')
        pygeoprocessing.testing.create_vector_on_disk(
            geometries=[],
            projection=srs.ExportToWkt(),
            fields=None,
            vector_format='GeoJSON',
            filename=empty_vector_path)

        with self.assertRaises(RuntimeError) as cm:
            pygeoprocessing.calculate_disjoint_polygon_set(empty_vector_path)

        self.assertTrue('Vector must have geometries but does not'
                        in str(cm.exception))

    def test_assert_is_valid_pixel_size(self):
        """PGP: geoprocessing test to cover valid pixel size."""
        import pygeoprocessing

        self.assertTrue(pygeoprocessing._assert_is_valid_pixel_size(
            (-10.5, 18282828228)))
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing._assert_is_valid_pixel_size(
                (-238.2, 'eleventeen'))
        expected_message = 'Invalid value for'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing._assert_is_valid_pixel_size(
                (-238.2, (10.2,)))
        expected_message = 'Invalid value for'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)

    def test_get_raster_info_type(self):
        """PGP: test get_raster_info's type."""
        import pygeoprocessing
        gdal_type_numpy_pairs = (
            ('int16.tif', gdal.GDT_Int16, numpy.int16),
            ('uint16.tif', gdal.GDT_UInt16, numpy.uint16),
            ('int32.tif', gdal.GDT_Int32, numpy.int32),
            ('uint32.tif', gdal.GDT_UInt32, numpy.uint32),
            ('float32.tif', gdal.GDT_Float32, numpy.float32),
            ('float64.tif', gdal.GDT_Float64, numpy.float64),
            ('cfloat32.tif', gdal.GDT_CFloat32, numpy.csingle),
            ('cfloat64.tif', gdal.GDT_CFloat64, numpy.complex64))

        gtiff_driver = gdal.GetDriverByName('GTiff')
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        wgs84_wkt = srs.ExportToWkt()
        for raster_filename, gdal_type, numpy_type in gdal_type_numpy_pairs:
            raster_path = os.path.join(self.workspace_dir, raster_filename)
            new_raster = gtiff_driver.Create(raster_path, 1, 1, 1, gdal_type)
            new_raster.SetProjection(wgs84_wkt)
            new_raster.SetGeoTransform([1.0, 1.0, 0.0, 1.0, 0.0, -1.0])
            new_band = new_raster.GetRasterBand(1)
            array = numpy.array([[1]], dtype=numpy_type)
            new_band.WriteArray(array)
            new_raster.FlushCache()
            new_band = None
            new_raster = None

            raster_info = pygeoprocessing.get_raster_info(raster_path)
            self.assertEqual(raster_info['numpy_type'], numpy_type)

    def test_non_geotiff_raster_types(self):
        """PGP: test mixed GTiff and gpkg raster types."""
        import pygeoprocessing

        gtiff_driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        n = 5
        new_raster = gtiff_driver.Create(
            raster_path, n, n, 1, gdal.GDT_Byte, options=[
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        new_raster.SetProjection(srs.ExportToWkt())
        new_raster.SetGeoTransform([1.0, 1.0, 0.0, 1.0, 0.0, -1.0])
        new_band = new_raster.GetRasterBand(1)
        new_band.SetNoDataValue(-1)
        array = numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))
        new_band.WriteArray(array)
        new_raster.FlushCache()
        new_band = None
        new_raster = None

        target_path = os.path.join(self.workspace_dir, 'target.gpkg')
        pygeoprocessing.raster_calculator(
            ((raster_path, 1), (raster_path, 1)), lambda a, b: a+b,
            target_path, gdal.GDT_Byte, None,
            raster_driver_creation_tuple=['gpkg', ()])
        target_raster = gdal.OpenEx(target_path)
        target_driver = target_raster.GetDriver()
        self.assertEqual(target_driver.GetDescription().lower(), 'gpkg')
        target_band = target_raster.GetRasterBand(1)
        numpy.testing.assert_array_equal(
            target_band.ReadAsArray(), array*2)

    def test_get_file_info(self):
        """PGP: geoprocessing test for `file_list` in the get_*_info ops."""
        import pygeoprocessing

        gtiff_driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'test.tif')
        raster = gtiff_driver.Create(raster_path, 1, 1, 1, gdal.GDT_Int32)
        raster.FlushCache()
        raster_file_list = raster.GetFileList()
        raster = None
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        self.assertEqual(raster_info['file_list'], raster_file_list)

        gpkg_driver = gdal.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        vector = gpkg_driver.Create(vector_path, 0, 0, 0, gdal.GDT_Unknown)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        del layer
        vector_file_list = vector.GetFileList()
        vector = None
        vector_info = pygeoprocessing.get_vector_info(vector_path)
        self.assertEqual(vector_info['file_list'], vector_file_list)

    def test_get_gis_type(self):
        """PGP: test geoprocessing type."""
        import pygeoprocessing
        gpkg_driver = ogr.GetDriverByName('GPKG')
        vector_path = os.path.join(self.workspace_dir, 'small_vector.gpkg')
        vector = gpkg_driver.CreateDataSource(vector_path)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = vector.CreateLayer('small_vector', srs=srs)
        layer.CreateField(ogr.FieldDefn('expected_value', ogr.OFTInteger))
        layer_defn = layer.GetLayerDefn()

        # make an n x n raster with 2*m x 2*m polygons inside.
        pixel_size = 1.0
        subpixel_size = 1./5. * pixel_size
        origin_x = 1.0
        origin_y = -1.0
        n = 16
        layer.StartTransaction()
        for row_index in range(n * 2):
            for col_index in range(n * 2):
                x_pos = origin_x + (
                    col_index*2 + 1 + col_index // 2) * subpixel_size
                y_pos = origin_y - (
                    row_index*2 + 1 + row_index // 2) * subpixel_size
                shapely_feature = shapely.geometry.Polygon([
                    (x_pos, y_pos),
                    (x_pos+subpixel_size, y_pos),
                    (x_pos+subpixel_size, y_pos-subpixel_size),
                    (x_pos, y_pos-subpixel_size),
                    (x_pos, y_pos)])
                new_feature = ogr.Feature(layer_defn)
                new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
                new_feature.SetGeometry(new_geometry)
                expected_value = row_index // 2 * n + col_index // 2
                new_feature.SetField('expected_value', expected_value)
                layer.CreateFeature(new_feature)
        layer.CommitTransaction()
        layer.SyncToDisk()
        layer = None
        vector = None

        gtiff_driver = gdal.GetDriverByName('GTiff')
        raster_path = os.path.join(self.workspace_dir, 'small_raster.tif')
        new_raster = gtiff_driver.Create(
            raster_path, n, n, 1, gdal.GDT_Int32, options=[
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=16', 'BLOCKYSIZE=16'])
        new_raster.SetProjection(srs.ExportToWkt())
        new_raster.SetGeoTransform([origin_x, 1.0, 0.0, origin_y, 0.0, -1.0])
        new_band = new_raster.GetRasterBand(1)
        new_band.SetNoDataValue(-1)
        array = numpy.array(range(n*n), dtype=numpy.int32).reshape((n, n))
        new_band.WriteArray(array)
        new_raster.FlushCache()
        new_band = None
        new_raster = None

        text_file_path = os.path.join(self.workspace_dir, 'text_file.txt')
        with open(text_file_path, 'w') as text_file:
            text_file.write('test')

        self.assertEqual(
            pygeoprocessing.get_gis_type(text_file_path),
            pygeoprocessing.UNKNOWN_TYPE)
        self.assertEqual(
            pygeoprocessing.get_gis_type(raster_path),
            pygeoprocessing.RASTER_TYPE)
        self.assertEqual(
            pygeoprocessing.get_gis_type(vector_path),
            pygeoprocessing.VECTOR_TYPE)

        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.get_gis_type('totally_fake_file')
        actual_message = str(cm.exception)
        self.assertTrue('does not exist' in actual_message, actual_message)

    def test_iterblocks_bad_raster(self):
        """PGP: tests iterblocks presents useful error on missing raster."""
        import pygeoprocessing
        with self.assertRaises(ValueError) as cm:
            _ = list(pygeoprocessing.iterblocks(('fake_file.tif', 1)))
        expected_message = 'could not be opened'
        actual_message = str(cm.exception)
        self.assertTrue(expected_message in actual_message, actual_message)
