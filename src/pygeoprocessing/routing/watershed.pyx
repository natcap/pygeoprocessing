import logging
import os
import shutil
import tempfile
import time

cimport cython
cimport numpy
numpy.import_array()
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.time cimport time as ctime
from libc.time cimport time_t
from libcpp.list cimport list as clist
from libcpp.map cimport map as cmap
from libcpp.pair cimport pair
from libcpp.queue cimport queue
from libcpp.set cimport set as cset
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import shapely.geometry
import shapely.prepared
import shapely.wkb

import pygeoprocessing
from ..geoprocessing_core import SPARSE_CREATION_OPTIONS
from ..extensions cimport ManagedRaster
from ..utils import gdal_use_exceptions, GDALUseExceptions

LOGGER = logging.getLogger(__name__)

# It's convenient to define a C++ pair here as a pair of longs to represent the
# x,y coordinates of a pixel.  So, CoordinatePair().first is the x coordinate,
# CoordinatePair().second is the y coordinate.  Both are in integer pixel
# coordinates.
ctypedef pair[long, long] CoordinatePair


def _is_raster_path_band_formatted(raster_path_band):
    """Return true if raster path band is a (str, int) tuple/list."""
    if not isinstance(raster_path_band, (list, tuple)):
        return False
    elif len(raster_path_band) != 2:
        return False
    elif not isinstance(raster_path_band[0], basestring):
        return False
    elif not isinstance(raster_path_band[1], int):
        return False
    else:
        return True


cdef cset[CoordinatePair] _c_split_geometry_into_seeds(
        source_geom_wkb, tuple flow_dir_geotransform, flow_dir_srs,
        flow_dir_n_cols, flow_dir_n_rows,
        target_raster_path, diagnostic_vector_path=None):
    """Split a geometry into 'seeds' of (x, y) coordinate pairs.

    Parameters:
        source_geom_wkb (str): A string of bytes in WKB representing
            a geometry. Must be in the same projected coordinate system
            as the flow direction raster from which
            ``flow_dir_geotransform`` and ``flow_dir_srs`` are derived.
        flow_dir_geotransform (list): A 6-element array representing
            the GDAL affine geotransform from the flow direction raster.
        flow_dir_srs (osr.SpatialReference): The OSR SpatialReference
            object from the flow direction raster.
        flow_dir_n_cols (int): the number of columns in the flow
            direction raster.
        flow_dir_n_rows (int): the number of rows in the flow
            direction raster.
        target_raster_path (str): The path to a raster onto which
            the geometry might be rasterized.  If the geometry is small
            enough to be completely contained within a single pixel, no
            raster will be written to this location.
        diagnostic_vector_path (string or None): If a string, a GeoPackage
            vector will be written to this path containing georeferenced
            points representing the 'seeds' determined by this function.
            If ``None``, no vector is created.

    Returns:
        A ``cset[CoordinatePair]`` for use in cython code.

    """
    cdef float minx, miny, maxx, maxy
    cdef double x_origin = flow_dir_geotransform[0]
    cdef double y_origin = flow_dir_geotransform[3]
    cdef double x_pixelwidth = flow_dir_geotransform[1]
    cdef double y_pixelwidth = flow_dir_geotransform[5]
    cdef double flow_dir_maxx = x_origin + (flow_dir_n_cols * x_pixelwidth)
    cdef double flow_dir_miny = y_origin + (flow_dir_n_rows * y_pixelwidth)
    cdef cset[CoordinatePair] seed_set

    geometry = shapely.wkb.loads(source_geom_wkb)

    minx, miny, maxx, maxy = geometry.bounds
    cdef int minx_pixelcoord = <int>((minx - x_origin) // x_pixelwidth)
    cdef int miny_pixelcoord = <int>((miny - y_origin) // y_pixelwidth)
    cdef int maxx_pixelcoord = <int>((maxx - x_origin) // x_pixelwidth)
    cdef int maxy_pixelcoord = <int>((maxy - y_origin) // y_pixelwidth)

    # If the geometry only intersects a single pixel, we can treat it
    # as a single point, which means that we can track it directly in our
    # seeds data structure and not have to include it in the disjoint set
    # determination.
    if minx_pixelcoord == maxx_pixelcoord and miny_pixelcoord == maxy_pixelcoord:
        # If the point is over nodata, skip it.
        seed = CoordinatePair(minx_pixelcoord, miny_pixelcoord)
        seed_set.insert(seed)
        return seed_set

    # If the geometry's bounding box covers more than one pixel, we need to
    # rasterize it to determine which pixels it intersects.
    cdef double minx_aligned = max(
        x_origin + (minx_pixelcoord * x_pixelwidth),
        x_origin)
    cdef double miny_aligned = max(
        y_origin + ((miny_pixelcoord+1) * y_pixelwidth),
        flow_dir_miny)
    cdef double maxx_aligned = min(
        x_origin + ((maxx_pixelcoord+1) * x_pixelwidth),
        flow_dir_maxx)
    cdef double maxy_aligned = min(
        y_origin + (maxy_pixelcoord * y_pixelwidth),
        y_origin)
    cdef int write_diagnostic_vector = 0

    cdef int row, col
    cdef int global_row, global_col
    cdef int seed_raster_origin_col
    cdef int seed_raster_origin_row
    cdef dict block_info
    cdef int block_xoff
    cdef int block_yoff
    cdef numpy.ndarray[numpy.npy_uint8, ndim=2] seed_array

    # It's possible for a perfectly vertical or horizontal line to cover 0 rows
    # or columns, so defaulting to row/col count of 1 in these cases.
    local_n_cols = max(abs(maxx_aligned - minx_aligned) // abs(x_pixelwidth), 1)
    local_n_rows = max(abs(maxy_aligned - miny_aligned) // abs(y_pixelwidth), 1)

    with GDALUseExceptions():
        # The geometry does not fit into a single pixel, so let's create a new
        # raster onto which to rasterize it.
        memory_driver = gdal.GetDriverByName('Memory')
        new_vector = memory_driver.Create('mem', 0, 0, 0, gdal.GDT_Unknown)
        new_layer = new_vector.CreateLayer('user_geometry', flow_dir_srs, ogr.wkbUnknown)
        new_layer.StartTransaction()
        new_feature = ogr.Feature(new_layer.GetLayerDefn())
        new_feature.SetGeometry(ogr.CreateGeometryFromWkb(source_geom_wkb))
        new_layer.CreateFeature(new_feature)
        new_layer.CommitTransaction()

        local_origin_x = max(minx_aligned, x_origin)
        local_origin_y = min(maxy_aligned, y_origin)

        local_geotransform = [
            local_origin_x, flow_dir_geotransform[1], flow_dir_geotransform[2],
            local_origin_y, flow_dir_geotransform[4], flow_dir_geotransform[5]]
        gtiff_driver = gdal.GetDriverByName('GTiff')
        # Raster is sparse, no need to fill.
        raster = gtiff_driver.Create(
            target_raster_path, int(local_n_cols), int(local_n_rows), 1,
            gdal.GDT_Byte, options=SPARSE_CREATION_OPTIONS)

        raster.SetSpatialRef(flow_dir_srs)
        raster.SetGeoTransform(local_geotransform)

        gdal.RasterizeLayer(
            raster, [1], new_layer, burn_values=[1], options=['ALL_TOUCHED=True'])
        raster = None

        diagnostic_vector = None
        diagnostic_layer = None
        if diagnostic_vector_path is not None:
            write_diagnostic_vector = 1

            gpkg_driver = gdal.GetDriverByName('GPKG')
            diagnostic_vector = gpkg_driver.Create(
                diagnostic_vector_path, 0, 0, 0, gdal.GDT_Unknown)
            diagnostic_layer = diagnostic_vector.CreateLayer(
                'seeds', flow_dir_srs, ogr.wkbPoint)
            user_geometry_layer = diagnostic_vector.CreateLayer(
                'user_geometry', flow_dir_srs, ogr.wkbUnknown)
            user_geometry_layer.StartTransaction()
            user_feature = ogr.Feature(user_geometry_layer.GetLayerDefn())
            user_feature.SetGeometry(ogr.CreateGeometryFromWkb(source_geom_wkb))
            user_geometry_layer.CreateFeature(user_feature)
            user_geometry_layer.CommitTransaction()

        seed_raster_origin_col = <int>((local_origin_x - x_origin) // x_pixelwidth)
        seed_raster_origin_row = <int>((local_origin_y - y_origin) // y_pixelwidth)
        for block_info, seed_array in pygeoprocessing.iterblocks(
                (target_raster_path, 1)):
            block_xoff = block_info['xoff']
            block_yoff = block_info['yoff']
            n_rows = seed_array.shape[0]
            n_cols = seed_array.shape[1]

            for row in range(n_rows):
                for col in range(n_cols):
                    with cython.boundscheck(False):
                        # Check if the pixel does not overlap the geometry.
                        if seed_array[row, col] == 0:
                            continue

                    global_row = seed_raster_origin_row + block_yoff + row
                    global_col = seed_raster_origin_col + block_xoff + col

                    if write_diagnostic_vector == 1:
                        diagnostic_layer.StartTransaction()
                        new_feature = ogr.Feature(diagnostic_layer.GetLayerDefn())
                        new_feature.SetGeometry(ogr.CreateGeometryFromWkb(
                            shapely.geometry.Point(
                                x_origin + ((global_col*x_pixelwidth) +
                                            (x_pixelwidth / 2.)),
                                y_origin + ((global_row*y_pixelwidth) +
                                            (y_pixelwidth / 2.))).wkb))
                        diagnostic_layer.CreateFeature(new_feature)
                        diagnostic_layer.CommitTransaction()

                    seed_set.insert(CoordinatePair(global_col, global_row))

        return seed_set


def _split_geometry_into_seeds(
        source_geom_wkb, geotransform, flow_dir_srs,
        flow_dir_n_cols, flow_dir_n_rows, target_raster_path,
        diagnostic_vector_path=None):
    """Split a geometry into 'seeds' of (x, y) coordinate pairs.

    This function is a python wrapper around ``_c_split_geometry_into_seeds``
    that is useful for testing.

    Parameters:
        source_geom_wkb (str): A string of bytes in WKB representing
            a geometry. Must be in the same projected coordinate system
            as the flow direction raster from which
            ``flow_dir_geotransform`` and ``flow_dir_srs`` are derived.
        flow_dir_geotransform (list): A 6-element array representing
            the GDAL affine geotransform from the flow direction raster.
        flow_dir_srs (osr.SpatialReference): The OSR SpatialReference
            object from the flow direction raster.
        flow_dir_n_cols (int): the number of columns in the flow
            direction raster.
        flow_dir_n_rows (int): the number of rows in the flow
            direction raster.
        target_raster_path (str): The path to a raster onto which
            the geometry might be rasterized.  If the geometry is small
            enough to be completely contained within a single pixel, no
            raster will be written to this location.
        diagnostic_vector_path (string or None): If a string, a GeoPackage
            vector will be written to this path containing georeferenced
            points representing the 'seeds' determined by this function.
            If ``None``, no vector is created.

    Returns:
        A python ``set`` of (x-index, y-index) tuples.

    """

    return_set = set()
    cdef cset[CoordinatePair] seeds = _c_split_geometry_into_seeds(
        source_geom_wkb, geotransform, flow_dir_srs, flow_dir_n_cols,
        flow_dir_n_rows, target_raster_path, diagnostic_vector_path)

    cdef CoordinatePair seed
    cdef cset[CoordinatePair].iterator seeds_iterator = seeds.begin()
    while seeds_iterator != seeds.end():
        seed = deref(seeds_iterator)
        inc(seeds_iterator)

        return_set.add((seed.first, seed.second))

    return return_set


@cython.boundscheck(False)
@gdal_use_exceptions
def delineate_watersheds_d8(
        d8_flow_dir_raster_path_band, outflow_vector_path,
        target_watersheds_vector_path, working_dir=None,
        write_diagnostic_vector=False, remove_temp_files=True,
        target_layer_name='watersheds'):
    """Delineate watersheds for a vector of geometries using D8 flow dir.

    Note:
        The ``d8_flow_dir_raster_path_band`` and ``outflow_vector_path`` files
        must have the same spatial reference system.  The output watersheds
        vector will use this same spatial reference system.

    Args:
        d8_flow_dir_raster_path_band (tuple): A (path, band_id) tuple
            to a D8 flow direction raster.  This raster must be a tiled raster
            with block sizes being a power of 2.  The output watersheds vector
            will have its spatial reference copied from this raster. Paths may
            use any GDAL-supported scheme, including virtual file system /vsi schemes.
        outflow_vector_path (str): The path to a vector containing features
            with valid geometries from which watersheds will be delineated.
            Only those parts of the geometry that overlap valid flow direction
            pixels will be included in the output watersheds vector. Paths may
            use any GDAL-supported scheme, including virtual file system /vsi schemes.
        target_watersheds_vector_path (str): The path to a vector where the
            target watersheds will be stored.  Must have the extension ``.gpkg``.
        working_dir=None (str or None): The path to a directory on disk
            within which various intermediate files will be stored.  If None,
            a folder will be created within the system's temp directory.
        write_diagnostic_vector=False (bool): If ``True``, a set of vectors will
            be written to ``working_dir``, one per watershed.  Each vector
            includes geometries for the watershed being represented and
            for the watershed seed pixels the geometry overlaps.  Useful in
            debugging issues with feature overlap of the DEM.  Setting this
            parameter to ``True`` will dramatically increase runtime when
            outflow geometries cover many pixels.
        remove_temp_files=True (bool): Whether to remove the created temp
            directory at the end of the watershed delineation run.
        target_layer_name='watersheds' (str): The string name to use for
            the watersheds layer.  This layer name may be named anything
            except for "polygonized_watersheds".

    Returns:
        ``None``

    """
    try:
        if working_dir is not None:
            os.makedirs(working_dir)
    except OSError:
        pass
    working_dir_path = tempfile.mkdtemp(
        dir=working_dir, prefix='watershed_delineation_trivial_%s_' % time.strftime(
            '%Y-%m-%d_%H_%M_%S', time.gmtime()))

    if (d8_flow_dir_raster_path_band is not None and not
            _is_raster_path_band_formatted(d8_flow_dir_raster_path_band)):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                d8_flow_dir_raster_path_band))

    flow_dir_info = pygeoprocessing.get_raster_info(
        d8_flow_dir_raster_path_band[0])
    cdef int flow_dir_nodata = flow_dir_info['nodata'][0]
    source_gt = flow_dir_info['geotransform']
    cdef double flow_dir_origin_x = source_gt[0]
    cdef double flow_dir_origin_y = source_gt[3]
    cdef double flow_dir_pixelsize_x = source_gt[1]
    cdef double flow_dir_pixelsize_y = source_gt[5]
    cdef int flow_dir_n_cols = flow_dir_info['raster_size'][0]
    cdef int flow_dir_n_rows = flow_dir_info['raster_size'][1]
    cdef int ws_id
    bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = flow_dir_info['bounding_box']
    LOGGER.debug('Creating flow dir bbox')
    flow_dir_bbox = shapely.prepared.prep(
        shapely.geometry.Polygon([
            (bbox_minx, bbox_maxy),
            (bbox_minx, bbox_miny),
            (bbox_maxx, bbox_miny),
            (bbox_maxx, bbox_maxy),
            (bbox_minx, bbox_maxy)]))
    LOGGER.debug('Creating flow dir managed raster')
    flow_dir_managed_raster = ManagedRaster(
        d8_flow_dir_raster_path_band[0].encode('utf-8'),
        d8_flow_dir_raster_path_band[1], False)
    gtiff_driver = gdal.GetDriverByName('GTiff')
    flow_dir_srs = osr.SpatialReference()
    if flow_dir_info['projection_wkt']:
        flow_dir_srs.ImportFromWkt(flow_dir_info['projection_wkt'])

    outflow_vector = gdal.OpenEx(outflow_vector_path, gdal.OF_VECTOR)
    if outflow_vector is None:
        raise ValueError(u'Could not open outflow vector %s' % outflow_vector_path)

    driver = ogr.GetDriverByName('GPKG')
    watersheds_srs = osr.SpatialReference()
    if flow_dir_info['projection_wkt']:
        watersheds_srs.ImportFromWkt(flow_dir_info['projection_wkt'])
    watersheds_vector = driver.CreateDataSource(target_watersheds_vector_path)
    polygonized_watersheds_layer = watersheds_vector.CreateLayer(
        'polygonized_watersheds', watersheds_srs, ogr.wkbPolygon)

    # Using wkbMultiPolygon for this layer because GDAL's polygonize function
    # may create multiple polygons for a single watershed feature, and we want
    # all geometries for a single watershed to be represented in a single
    # feature.
    watersheds_layer = watersheds_vector.CreateLayer(
        target_layer_name, watersheds_srs, ogr.wkbMultiPolygon)
    index_field = ogr.FieldDefn('ws_id', ogr.OFTInteger)
    index_field.SetWidth(24)
    polygonized_watersheds_layer.CreateField(index_field)

    cdef int* reverse_flow = [4, 5, 6, 7, 0, 1, 2, 3]
    cdef int* neighbor_col = [1, 1, 0, -1, -1, -1, 0, 1]
    cdef int* neighbor_row = [0, -1, -1, -1, 0, 1, 1, 1]
    cdef queue[CoordinatePair] process_queue
    cdef cset[CoordinatePair] process_queue_set
    cdef CoordinatePair neighbor_pixel
    cdef int ix_min, iy_min, ix_max, iy_max
    cdef ManagedRaster scratch_managed_raster
    cdef int watersheds_created = 0
    cdef int current_fid, outflow_feature_count
    cdef cset[CoordinatePair].iterator seed_iterator
    cdef cset[CoordinatePair] seeds_in_watershed
    cdef time_t last_log_time
    cdef int n_cells_visited = 0

    LOGGER.info('Delineating watersheds')
    outflow_layer = outflow_vector.GetLayer()
    outflow_feature_count = outflow_layer.GetFeatureCount()
    for feature in outflow_layer:
        # Some vectors start indexing their FIDs at 0.
        # The mask raster input to polygonization, however, only regards pixels
        # as zero or nonzero.  Therefore, to make sure we can use the ws_id as
        # the FID and not maintain a separate mask raster, we'll just add 1.
        current_fid = feature.GetFID()
        ws_id = current_fid + 1
        assert ws_id >= 1, 'WSID <= 1!'

        geom = feature.GetGeometryRef()
        if geom.IsEmpty():
            LOGGER.debug(
                'Outflow feature %s has empty geometry.  Skipping.',
                current_fid)
            continue
        geom_wkb = bytes(geom.ExportToWkb())
        shapely_geom = shapely.wkb.loads(geom_wkb)

        LOGGER.debug('Testing geometry bbox')
        if not flow_dir_bbox.intersects(shapely.geometry.box(*shapely_geom.bounds)):
            LOGGER.debug(
                'Outflow feature %s does not overlap with the flow '
                'direction raster. Skipping.', current_fid)
            continue

        seeds_raster_path = os.path.join(working_dir_path, '%s_rasterized.tif' % ws_id)
        if write_diagnostic_vector:
            diagnostic_vector_path = os.path.join(working_dir_path, '%s_seeds.gpkg' % ws_id)
        else:
            diagnostic_vector_path = None
        seeds_in_watershed = _c_split_geometry_into_seeds(
            geom_wkb,
            source_gt,
            flow_dir_srs=flow_dir_srs,
            flow_dir_n_cols=flow_dir_n_cols,
            flow_dir_n_rows=flow_dir_n_rows,
            target_raster_path=seeds_raster_path,
            diagnostic_vector_path=diagnostic_vector_path
        )

        seed_iterator = seeds_in_watershed.begin()
        while seed_iterator != seeds_in_watershed.end():
            seed = deref(seed_iterator)
            inc(seed_iterator)

            if not 0 <= seed.first < flow_dir_n_cols:
                continue

            if not 0 <= seed.second < flow_dir_n_rows:
                continue

            if flow_dir_managed_raster.get(seed.first, seed.second) == flow_dir_nodata:
                continue

            process_queue.push(seed)
            process_queue_set.insert(seed)

        if process_queue_set.size() == 0:
            LOGGER.debug(
                'Outflow feature %s does not intersect any pixels with '
                'valid flow direction. Skipping.', current_fid)
            continue

        scratch_raster_path = os.path.join(working_dir_path,
                                           '%s_scratch.tif' % ws_id)
        scratch_raster = gtiff_driver.Create(
            scratch_raster_path,
            flow_dir_n_cols,
            flow_dir_n_rows,
            1,  # n bands
            gdal.GDT_UInt32,
            options=SPARSE_CREATION_OPTIONS)
        scratch_raster.SetGeoTransform(source_gt)
        if flow_dir_info['projection_wkt']:
            scratch_raster.SetProjection(flow_dir_info['projection_wkt'])
        # strictly speaking, there's no need to set the nodata value on the band.
        scratch_raster = None

        scratch_managed_raster = ManagedRaster(
            scratch_raster_path.encode('utf-8'), 1, True)
        ix_min = flow_dir_n_cols
        iy_min = flow_dir_n_rows
        ix_max = 0
        iy_max = 0
        n_cells_visited = 0
        LOGGER.info(
            'Delineating watershed %s of %s (ws_id %s)',
            current_fid, outflow_feature_count, ws_id)
        last_log_time = ctime(NULL)
        while not process_queue.empty():
            if ctime(NULL) - last_log_time > 5.0:
                last_log_time = ctime(NULL)
                LOGGER.info(
                    'Delineating watershed %s of %s (ws_id %s), %s pixels '
                    'found so far', current_fid, outflow_feature_count,
                    ws_id, n_cells_visited)

            current_pixel = process_queue.front()
            process_queue_set.erase(current_pixel)
            process_queue.pop()

            scratch_managed_raster.set(current_pixel.first,
                                       current_pixel.second, ws_id)
            n_cells_visited += 1

            # These are for tracking the extents of the raster so we can build
            # a VRT and only polygonize the pixels we need to.
            ix_min = min(ix_min, current_pixel.first)
            iy_min = min(iy_min, current_pixel.second)
            ix_max = max(ix_max, current_pixel.first)
            iy_max = max(iy_max, current_pixel.second)

            for neighbor_index in range(8):
                neighbor_pixel = CoordinatePair(
                    current_pixel.first + neighbor_col[neighbor_index],
                    current_pixel.second + neighbor_row[neighbor_index])

                if not 0 <= neighbor_pixel.first < flow_dir_n_cols:
                    continue

                if not 0 <= neighbor_pixel.second < flow_dir_n_rows:
                    continue

                # If we've already enqueued the neighbor (either it's
                # upstream of another pixel or it's a watershed seed), we
                # don't need to re-enqueue it.
                if (process_queue_set.find(neighbor_pixel) !=
                        process_queue_set.end()):
                    continue

                # If the neighbor is known to be a seed, we don't need to
                # re-enqueue it either.  Either it's been visited already (and
                # may have upstream pixels) or it's going to be.
                if (seeds_in_watershed.find(neighbor_pixel) !=
                        seeds_in_watershed.end()):
                    continue

                # Does the neighbor flow into this pixel?
                # If yes, enqueue it.
                if (reverse_flow[neighbor_index] ==
                        flow_dir_managed_raster.get(
                            neighbor_pixel.first, neighbor_pixel.second)):
                    process_queue.push(neighbor_pixel)
                    process_queue_set.insert(neighbor_pixel)

        watersheds_created += 1
        scratch_managed_raster.close()

        # Build a VRT from the bounds of the affected pixels before
        # rasterizing.  For watersheds significantly smaller than the flow dir
        # raster, this yields a large speedup because we don't have to read in
        # the whole scratch raster in order to polygonize.
        x1 = (flow_dir_origin_x + (max(ix_min-1, 0)*flow_dir_pixelsize_x))  # minx
        y1 = (flow_dir_origin_y + (max(iy_min-1, 0)*flow_dir_pixelsize_y))  # miny
        x2 = (flow_dir_origin_x + (min(ix_max+1, flow_dir_n_cols)*flow_dir_pixelsize_x))  # maxx
        y2 = (flow_dir_origin_y + (min(iy_max+1, flow_dir_n_rows)*flow_dir_pixelsize_y))  # maxy

        vrt_options = gdal.BuildVRTOptions(
            outputBounds=(
                min(x1, x2),
                min(y1, y2),
                max(x1, x2),
                max(y1, y2))
        )
        vrt_path = os.path.join(working_dir_path, '%s_vrt.vrt' % ws_id)
        gdal.BuildVRT(vrt_path, [scratch_raster_path], options=vrt_options)

        # Polygonize this new watershed from the VRT.
        vrt_raster = gdal.OpenEx(vrt_path, gdal.OF_RASTER, allowed_drivers=['VRT'])
        vrt_band = vrt_raster.GetRasterBand(1)
        _ = gdal.Polygonize(
            vrt_band,  # The source band
            vrt_band,  # The mask. Pixels with 0 are invalid, nonzero are valid
            polygonized_watersheds_layer,
            0,  # ws_id field index
            [])  # 8connectedness does not always produce valid geometries.
        _ = None
        vrt_band = None
        vrt_raster = None

        # Removing files as we go to help manage disk space.
        if remove_temp_files:
            os.remove(scratch_raster_path)
            if os.path.exists(seeds_raster_path):
                os.remove(seeds_raster_path)
            os.remove(vrt_path)
            if (diagnostic_vector_path
                    and os.path.exists(diagnostic_vector_path)):
                os.remove(diagnostic_vector_path)

    flow_dir_managed_raster.close()
    LOGGER.info('Finished delineating %s watersheds', watersheds_created)

    # The Polygonization algorithm will sometimes identify regions that
    # should be contiguous in a single polygon, but are not.  For this reason,
    # we need an extra consolidation step here to make sure that we only produce
    # 1 feature per watershed.
    cdef cmap[int, cset[int]] fragments_with_duplicates
    cdef int fid
    for feature in polygonized_watersheds_layer:
        fid = feature.GetFID()
        # ws_id is tracked as 1 more than the FID.  See previous note about why.
        ws_id = feature.GetField('ws_id') - 1
        if (fragments_with_duplicates.find(ws_id)
                == fragments_with_duplicates.end()):
            fragments_with_duplicates[ws_id] = cset[int]()
        fragments_with_duplicates[ws_id].insert(fid)
    polygonized_watersheds_layer.ResetReading()

    LOGGER.info(
        'Consolidating %s fragments and copying field values to '
        'watersheds layer.', polygonized_watersheds_layer.GetFeatureCount())
    source_vector = gdal.OpenEx(outflow_vector_path, gdal.OF_VECTOR)
    source_layer = source_vector.GetLayer()
    watersheds_layer.CreateFields(source_layer.schema)

    watersheds_layer.StartTransaction()
    cdef int duplicate_fid
    cdef cset[int] duplicate_ids_set
    cdef cset[int].iterator duplicate_ids_set_iterator
    cdef cmap[int, cset[int]].iterator fragments_with_duplicates_iterator
    fragments_with_duplicates_iterator = fragments_with_duplicates.begin()
    while fragments_with_duplicates_iterator != fragments_with_duplicates.end():
        ws_id = deref(fragments_with_duplicates_iterator).first
        duplicate_ids_set = deref(fragments_with_duplicates_iterator).second
        inc(fragments_with_duplicates_iterator)

        duplicate_ids_set_iterator = duplicate_ids_set.begin()

        if duplicate_ids_set.size() == 1:
            duplicate_fid = deref(duplicate_ids_set_iterator)
            source_feature = polygonized_watersheds_layer.GetFeature(duplicate_fid)
            new_geometry = source_feature.GetGeometryRef()
        else:
            new_geometry = ogr.Geometry(ogr.wkbMultiPolygon)
            while duplicate_ids_set_iterator != duplicate_ids_set.end():
                duplicate_fid = deref(duplicate_ids_set_iterator)
                inc(duplicate_ids_set_iterator)

                duplicate_feature = polygonized_watersheds_layer.GetFeature(duplicate_fid)
                duplicate_geometry = duplicate_feature.GetGeometryRef()
                new_geometry.AddGeometry(duplicate_geometry)

        watershed_feature = ogr.Feature(watersheds_layer.GetLayerDefn())
        watershed_feature.SetGeometry(ogr.ForceToMultiPolygon(new_geometry))

        source_feature = source_layer.GetFeature(ws_id)
        for field_name, field_value in source_feature.items().items():
            watershed_feature.SetField(field_name, field_value)
        watersheds_layer.CreateFeature(watershed_feature)
    watersheds_layer.CommitTransaction()

    polygonized_watersheds_layer = None
    watersheds_layer = None
    if remove_temp_files:
        watersheds_vector.DeleteLayer('polygonized_watersheds')
    LOGGER.info('Finished vector consolidation')

    watersheds_vector = None
    source_layer = None
    source_vector = None

    if remove_temp_files:
        shutil.rmtree(working_dir_path)
    LOGGER.info('Watershed delineation complete')
