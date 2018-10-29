import time
import os
import logging
import shutil
import tempfile

import numpy
import pygeoprocessing
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import shapely.wkb
import shapely.ops
import rtree

cimport numpy
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.time cimport time_t
from libc.time cimport time as ctime
from libcpp.list cimport list as clist
from libcpp.pair cimport pair
from libcpp.queue cimport queue
from libcpp.deque cimport deque
from libcpp.set cimport set as cset
from libcpp.map cimport map as cmap

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())  # silence logging by default

# This module creates rasters with a memory xy block size of 2**BLOCK_BITS
cdef int BLOCK_BITS = 8

# Number of raster blocks to hold in memory at once per Managed Raster
cdef int MANAGED_RASTER_N_BLOCKS = 2**7

# these are the creation options that'll be used for all the rasters
GTIFF_CREATION_OPTIONS = (
    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
    'SPARSE_OK=YES',
    'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
    'BLOCKYSIZE=%d' % (1 << BLOCK_BITS))

# this is used to calculate the opposite D8 direction interpreting the index
# as a D8 direction
cdef int* D8_REVERSE_DIRECTION = [4, 5, 6, 7, 0, 1, 2, 3]

# this is a least recently used cache written in C++ in an external file,
# exposing here so _ManagedRaster can use it
cdef extern from "LRUCache.h" nogil:
    cdef cppclass LRUCache[KEY_T, VAL_T]:
        LRUCache(int)
        void put(KEY_T&, VAL_T&, clist[pair[KEY_T,VAL_T]]&)
        clist[pair[KEY_T,VAL_T]].iterator begin()
        clist[pair[KEY_T,VAL_T]].iterator end()
        bint exist(KEY_T &)
        VAL_T get(KEY_T &)

# this ctype is used to store the block ID and the block buffer as one object
# inside Managed Raster
ctypedef pair[int, int*] BlockBufferPair

# a class to allow fast random per-pixel access to a raster for both setting
# and reading pixels.
cdef class _ManagedRaster:
    cdef LRUCache[int, int*]* lru_cache
    cdef cset[int] dirty_blocks
    cdef int block_xsize
    cdef int block_ysize
    cdef int block_xmod
    cdef int block_ymod
    cdef int block_xbits
    cdef int block_ybits
    cdef int raster_x_size
    cdef int raster_y_size
    cdef int block_nx
    cdef int block_ny
    cdef int write_mode
    cdef bytes raster_path
    cdef int band_id
    cdef int closed

    def __cinit__(self, raster_path, band_id, write_mode):
        """Create new instance of Managed Raster.

        Parameters:
            raster_path (char*): path to raster that has block sizes that are
                powers of 2. If not, an exception is raised.
            band_id (int): which band in `raster_path` to index. Uses GDAL
                notation that starts at 1.
            write_mode (boolean): if true, this raster is writable and dirty
                memory blocks will be written back to the raster as blocks
                are swapped out of the cache or when the object deconstructs.

        Returns:
            None.
        """
        if not os.path.isfile(raster_path):
            LOGGER.error("%s is not a file.", raster_path)
            return
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        self.raster_x_size, self.raster_y_size = raster_info['raster_size']
        self.block_xsize, self.block_ysize = raster_info['block_size']
        self.block_xmod = self.block_xsize-1
        self.block_ymod = self.block_ysize-1

        if not (1 <= band_id <= raster_info['n_bands']):
            err_msg = (
                "Error: band ID (%s) is not a valid band number. "
                "This exception is happening in Cython, so it will cause a "
                "hard seg-fault, but it's otherwise meant to be a "
                "ValueError." % (band_id))
            print(err_msg)
            raise ValueError(err_msg)
        self.band_id = band_id

        if (self.block_xsize & (self.block_xsize - 1) != 0) or (
                self.block_ysize & (self.block_ysize - 1) != 0):
            # If inputs are not a power of two, this will at least print
            # an error message. Unfortunately with Cython, the exception will
            # present itself as a hard seg-fault, but I'm leaving the
            # ValueError in here at least for readability.
            err_msg = (
                "Error: Block size is not a power of two: "
                "block_xsize: %d, %d, %s. This exception is happening"
                "in Cython, so it will cause a hard seg-fault, but it's"
                "otherwise meant to be a ValueError." % (
                    self.block_xsize, self.block_ysize, raster_path))
            print(err_msg)
            raise ValueError(err_msg)

        self.block_xbits = numpy.log2(self.block_xsize)
        self.block_ybits = numpy.log2(self.block_ysize)
        self.block_nx = (
            self.raster_x_size + (self.block_xsize) - 1) / self.block_xsize
        self.block_ny = (
            self.raster_y_size + (self.block_ysize) - 1) / self.block_ysize

        self.lru_cache = new LRUCache[int, int*](MANAGED_RASTER_N_BLOCKS)
        self.raster_path = <bytes> raster_path
        self.write_mode = write_mode
        self.closed = 0

    def __dealloc__(self):
        """Deallocate _ManagedRaster.

        This operation manually frees memory from the LRUCache and writes any
        dirty memory blocks back to the raster if `self.write_mode` is True.
        """
        self.close()

    def close(self):
        """Close the _ManagedRaster and free up resources.

            This call writes any dirty blocks to disk, frees up the memory
            allocated as part of the cache, and frees all GDAL references.

            Any subsequent calls to any other functions in _ManagedRaster will
            have undefined behavior.
        """
        if self.closed:
            return
        self.closed = 1
        cdef int xi_copy, yi_copy
        cdef numpy.ndarray[int, ndim=2] block_array = numpy.empty(
            (self.block_ysize, self.block_xsize), dtype=numpy.int32)
        cdef int *int_buffer
        cdef int block_xi
        cdef int block_yi
        # initially the win size is the same as the block size unless
        # we're at the edge of a raster
        cdef int win_xsize
        cdef int win_ysize

        # we need the offsets to subtract from global indexes for cached array
        cdef int xoff
        cdef int yoff

        cdef clist[BlockBufferPair].iterator it = self.lru_cache.begin()
        cdef clist[BlockBufferPair].iterator end = self.lru_cache.end()
        if not self.write_mode:
            while it != end:
                # write the changed value back if desired
                PyMem_Free(deref(it).second)
                inc(it)
            return

        raster = gdal.OpenEx(
            self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
        raster_band = raster.GetRasterBand(self.band_id)

        # if we get here, we're in write_mode
        cdef cset[int].iterator dirty_itr
        while it != end:
            int_buffer = deref(it).second
            block_index = deref(it).first

            # write to disk if block is dirty
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr != self.dirty_blocks.end():
                self.dirty_blocks.erase(dirty_itr)
                block_xi = block_index % self.block_nx
                block_yi = block_index / self.block_nx

                # we need the offsets to subtract from global indexes for
                # cached array
                xoff = block_xi << self.block_xbits
                yoff = block_yi << self.block_ybits

                win_xsize = self.block_xsize
                win_ysize = self.block_ysize

                # clip window sizes if necessary
                if xoff+win_xsize > self.raster_x_size:
                    win_xsize = win_xsize - (
                        xoff+win_xsize - self.raster_x_size)
                if yoff+win_ysize > self.raster_y_size:
                    win_ysize = win_ysize - (
                        yoff+win_ysize - self.raster_y_size)

                for xi_copy in xrange(win_xsize):
                    for yi_copy in xrange(win_ysize):
                        block_array[yi_copy, xi_copy] = (
                            int_buffer[
                                (yi_copy << self.block_xbits) + xi_copy])
                raster_band.WriteArray(
                    block_array[0:win_ysize, 0:win_xsize],
                    xoff=xoff, yoff=yoff)
            PyMem_Free(int_buffer)
            inc(it)
        raster_band.FlushCache()
        raster_band = None
        raster = None

    cdef inline void set(self, int xi, int yi, int value):
        """Set the pixel at `xi,yi` to `value`."""
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        self.lru_cache.get(
            block_index)[
                ((yi & (self.block_ymod))<<self.block_xbits) +
                (xi & (self.block_xmod))] = value
        if self.write_mode:
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr == self.dirty_blocks.end():
                self.dirty_blocks.insert(block_index)

    cdef inline int get(self, int xi, int yi):
        """Return the value of the pixel at `xi,yi`."""
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        return self.lru_cache.get(
            block_index)[
                ((yi & (self.block_ymod))<<self.block_xbits) +
                (xi & (self.block_xmod))]

    cdef void _load_block(self, int block_index) except *:
        cdef int block_xi = block_index % self.block_nx
        cdef int block_yi = block_index / self.block_nx

        # we need the offsets to subtract from global indexes for cached array
        cdef int xoff = block_xi << self.block_xbits
        cdef int yoff = block_yi << self.block_ybits

        cdef int xi_copy, yi_copy
        cdef numpy.ndarray[int, ndim=2] block_array
        cdef int *int_buffer
        cdef clist[BlockBufferPair] removed_value_list

        # determine the block aligned xoffset for read as array

        # initially the win size is the same as the block size unless
        # we're at the edge of a raster
        cdef int win_xsize = self.block_xsize
        cdef int win_ysize = self.block_ysize

        # load a new block
        if xoff+win_xsize > self.raster_x_size:
            win_xsize = win_xsize - (xoff+win_xsize - self.raster_x_size)
        if yoff+win_ysize > self.raster_y_size:
            win_ysize = win_ysize - (yoff+win_ysize - self.raster_y_size)

        raster = gdal.OpenEx(self.raster_path, gdal.OF_RASTER)
        raster_band = raster.GetRasterBand(self.band_id)
        block_array = raster_band.ReadAsArray(
            xoff=xoff, yoff=yoff, win_xsize=win_xsize,
            win_ysize=win_ysize).astype(
            numpy.int32)
        raster_band = None
        raster = None
        int_buffer = <int*>PyMem_Malloc(
            (sizeof(int) << self.block_xbits) * win_ysize)
        for xi_copy in xrange(win_xsize):
            for yi_copy in xrange(win_ysize):
                int_buffer[(yi_copy<<self.block_xbits)+xi_copy] = (
                    block_array[yi_copy, xi_copy])
        self.lru_cache.put(
            <int>block_index, <int*>int_buffer, removed_value_list)

        if self.write_mode:
            raster = gdal.OpenEx(
                self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
            raster_band = raster.GetRasterBand(self.band_id)

        block_array = numpy.empty(
            (self.block_ysize, self.block_xsize), dtype=numpy.int)
        while not removed_value_list.empty():
            # write the changed value back if desired
            int_buffer = removed_value_list.front().second

            if self.write_mode:
                block_index = removed_value_list.front().first

                # write back the block if it's dirty
                dirty_itr = self.dirty_blocks.find(block_index)
                if dirty_itr != self.dirty_blocks.end():
                    self.dirty_blocks.erase(dirty_itr)

                    block_xi = block_index % self.block_nx
                    block_yi = block_index / self.block_nx

                    xoff = block_xi << self.block_xbits
                    yoff = block_yi << self.block_ybits

                    win_xsize = self.block_xsize
                    win_ysize = self.block_ysize

                    if xoff+win_xsize > self.raster_x_size:
                        win_xsize = win_xsize - (
                            xoff+win_xsize - self.raster_x_size)
                    if yoff+win_ysize > self.raster_y_size:
                        win_ysize = win_ysize - (
                            yoff+win_ysize - self.raster_y_size)

                    for xi_copy in xrange(win_xsize):
                        for yi_copy in xrange(win_ysize):
                            block_array[yi_copy, xi_copy] = int_buffer[
                                (yi_copy << self.block_xbits) + xi_copy]
                    raster_band.WriteArray(
                        block_array[0:win_ysize, 0:win_xsize],
                        xoff=xoff, yoff=yoff)
            PyMem_Free(int_buffer)
            removed_value_list.pop_front()

        if self.write_mode:
            raster_band = None
            raster = None


# It's convenient to define a C++ pair here as a pair of longs to represent the
# x,y coordinates of a pixel.  So, CoordinatePair().first is the x coordinate,
# CoordinatePair().second is the y coordinate.  Both are in integer pixel
# coordinates.
ctypedef pair[long, long] CoordinatePair


def delineate_watersheds(
        d8_flow_dir_raster_path_band, outflow_points_vector_path,
        target_fragments_vector_path, working_dir=None):
    """Delineate watersheds from a D8 flow direction raster.

    This function produces a vector of watershed fragments, where each fragment
    represents the area that flows into each outflow point.  Nested watersheds
    are represented by the field

    Parameters:
        d8_flow_dir_raster_path_band (tuple): A two-tuple representing the
            string path to the D8 flow direction raster to use and the band
            index to use.
        outflow_points_vector_path (string): Path to a vector of points on
            disk representing outflow points for a watershed.  This vector
            must have one layer, only contain point geometries, and must be
            in the same projection as the flow direction raster.
        target_fragments_vector_path (string): Path to where the watershed
            fragments vector will be stored on disk.  This filepath must end
            with the 'gpkg' extension, and will be created as a GeoPackage.
        working_dir=None (string or None): The path to a directory on disk
            where intermediate files will be stored.  This directory will be
            created if it does not exist, and intermediate files created will
            be removed.  If ``None``, a new temporary folder will be created
            within the system temp directory.

    Returns:
        ``None``

    """
    if (d8_flow_dir_raster_path_band is not None and not
            _is_raster_path_band_formatted(d8_flow_dir_raster_path_band)):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                d8_flow_dir_raster_path_band))

    # TODO: warn against mismatched projections.

    flow_dir_info = pygeoprocessing.get_raster_info(
        d8_flow_dir_raster_path_band[0])
    source_gt = flow_dir_info['geotransform']
    cdef long flow_dir_n_cols = flow_dir_info['raster_size'][0]
    cdef long flow_dir_n_rows = flow_dir_info['raster_size'][1]
    try:
        if working_dir is not None:
            os.makedirs(working_dir)
    except OSError:
        pass
    working_dir_path = tempfile.mkdtemp(
        dir=working_dir, prefix='watershed_delineation_%s_' % time.strftime(
            '%Y-%m-%d_%H_%M_%S', time.gmtime()))

    scratch_raster_path = os.path.join(working_dir_path, 'scratch_raster.tif')
    mask_raster_path = os.path.join(working_dir_path, 'scratch_mask.tif')

    # Create the watershed fragments layer for later.
    gpkg_driver = ogr.GetDriverByName('GPKG')
    flow_dir_srs = osr.SpatialReference()
    flow_dir_srs.ImportFromWkt(flow_dir_info['projection'])
    watershed_fragments_path = os.path.join(working_dir_path, 'watershed_fragments.gpkg')
    watershed_fragments_vector = gpkg_driver.CreateDataSource(watershed_fragments_path)
    watershed_fragments_layer = watershed_fragments_vector.CreateLayer(
        'watershed_fragments', flow_dir_srs, ogr.wkbPolygon)
    ws_id_field = ogr.FieldDefn('ws_id', ogr.OFTInteger)
    ws_id_field.SetWidth(24)
    watershed_fragments_layer.CreateField(ws_id_field)

    flow_dir_bbox_geometry = shapely.geometry.box(*flow_dir_info['bounding_box'])

    source_outlets_vector = ogr.Open(outflow_points_vector_path)

    working_outlets_path = os.path.join(working_dir_path, 'working_outlets.gpkg')
    working_outlets_vector = gpkg_driver.CopyDataSource(source_outlets_vector,
                                                        working_outlets_path)
    working_outlets_layer = working_outlets_vector.GetLayer()
    working_outlets_layer_name = working_outlets_layer.GetName()
    # Add a new field to the clipped_outlets_layer to ensure we know what field
    # values are bing rasterized.
    cdef int ws_id
    ws_id_fieldname = '__ws_id__'
    ws_id_field_defn = ogr.FieldDefn(ws_id_fieldname, ogr.OFTInteger64)
    working_outlets_layer.CreateField(ws_id_field_defn)

    working_outlets_layer.StartTransaction()
    for ws_id, feature in enumerate(working_outlets_layer, start=1):
        feature.SetField(ws_id_fieldname, ws_id)
        working_outlets_layer.SetFeature(feature)
    working_outlets_layer.CommitTransaction()
    feature = None
    ws_id_field_defn = None
    working_outlets_vector.SyncToDisk()
    working_outlets_layer = None
    working_outlets_vector = None

    # Create a new watershed scratch raster the size, shape of the flow dir raster
    # via rasterization.
    # TODO: make a byte and/or int managed raster class
    cdef int NO_WATERSHED = 0
    LOGGER.info('Creating raster for tracking watershed fragments')
    pygeoprocessing.new_raster_from_base(
        d8_flow_dir_raster_path_band[0], scratch_raster_path, gdal.GDT_UInt32,
        [NO_WATERSHED], fill_value_list=[NO_WATERSHED],
        gtiff_creation_options=GTIFF_CREATION_OPTIONS)

    pygeoprocessing.rasterize(
        working_outlets_path, scratch_raster_path, None,
        ['ALL_TOUCHED=TRUE', 'ATTRIBUTE=%s' % ws_id_fieldname],
        layer_index=working_outlets_layer_name)

    # Create a new watershed scratch mask raster the size, shape of the flow dir raster
    LOGGER.info('Creating raster for tracking visited pixels')
    pygeoprocessing.new_raster_from_base(
        d8_flow_dir_raster_path_band[0], mask_raster_path, gdal.GDT_Byte,
        [255], fill_value_list=[0],
        gtiff_creation_options=GTIFF_CREATION_OPTIONS)

    cdef CoordinatePair current_pixel, neighbor_pixel
    cdef queue[CoordinatePair] process_queue
    cdef cset[CoordinatePair] process_queue_set
    cdef cset[int] nested_watershed_ids

    cdef int* neighbor_col = [1, 1, 0, -1, -1, -1, 0, 1]
    cdef int* neighbor_row = [0, -1, -1, -1, 0, 1, 1, 1]
    cdef int* reverse_flow = [4, 5, 6, 7, 0, 1, 2, 3]

    # Map ws_id to the watersheds that nest within.
    nested_watersheds = {}

    # Track outflow geometry FIDs against the WS_ID used.
    ws_id_to_fid = {}

    cdef int pixels_in_watershed
    cdef time_t last_log_time = ctime(NULL)
    cdef time_t last_ws_log_time = ctime(NULL)
    cdef double x_origin = source_gt[0]
    cdef double y_origin = source_gt[3]
    cdef double x_pixelwidth = source_gt[1]
    cdef double y_pixelwidth = source_gt[5]
    cdef int block_index

    # When interleaved is True, coords are in (xmin, ymin, xmax, ymax)
    spatial_index = rtree.index.Index(interleaved=True)
    for block_index, offset_dict in enumerate(
            pygeoprocessing.iterblocks(d8_flow_dir_raster_path_band[0],
                                       largest_block=0,
                                       offset_only=True)):
        origin_xcoord = x_origin + offset_dict['xoff']*x_pixelwidth
        origin_ycoord = y_origin + offset_dict['yoff']*y_pixelwidth
        win_xcoord = origin_xcoord + offset_dict['win_xsize']*x_pixelwidth
        win_ycoord = origin_ycoord + offset_dict['win_ysize']*y_pixelwidth

        spatial_index.insert(block_index, (min(origin_xcoord, win_xcoord),
                                     min(origin_ycoord, win_ycoord),
                                     max(origin_xcoord, win_xcoord),
                                     max(origin_ycoord, win_ycoord)))

    working_outlets_vector = gdal.OpenEx(working_outlets_path,
                                         gdal.OF_VECTOR)
    working_outlets_layer = working_outlets_vector.GetLayer()
    working_outlet_layer_definition = working_outlets_layer.GetLayerDefn()
    cdef int features_in_layer = working_outlets_layer.GetFeatureCount()
    cdef cmap[int, cset[CoordinatePair]] points_in_blocks
    cdef cmap[CoordinatePair, int] point_ws_ids 
    cdef CoordinatePair ws_seed_coord

    for outflow_feature in working_outlets_layer:
        ws_id = int(outflow_feature.GetField(ws_id_fieldname))
        ws_id_to_fid[ws_id] = outflow_feature.GetFID()
        geometry = shapely.wkb.loads(
            outflow_feature.GetGeometryRef().ExportToWkb())
        if not flow_dir_bbox_geometry.contains(geometry):
            geometry = flow_dir_bbox_geometry.intersection(geometry)
            if geometry.is_empty:
                continue

        ws_seed_point = geometry.representative_point()
        ws_seed_coord = CoordinatePair(
            (ws_seed_point.x - x_origin) // x_pixelwidth,
            (ws_seed_point.y - y_origin) // y_pixelwidth)

        block_index = next(spatial_index.nearest(
                (ws_seed_coord.first, ws_seed_coord.second,
                 ws_seed_coord.first, ws_seed_coord.second), num_results=1))

        if (points_in_blocks.find(block_index) ==
                points_in_blocks.end()):
            points_in_blocks[block_index] = cset[CoordinatePair]()

        points_in_blocks[block_index].insert(ws_seed_coord)
        point_ws_ids[ws_seed_coord] = ws_id

    LOGGER.info('Delineating watersheds')
    cdef _ManagedRaster flow_dir_managed_raster, scratch_managed_raster, mask_managed_raster
    flow_dir_managed_raster = _ManagedRaster(d8_flow_dir_raster_path_band[0],
                                             d8_flow_dir_raster_path_band[1],
                                             0)  # read-only
    scratch_managed_raster = _ManagedRaster(scratch_raster_path, 1, 1)
    mask_managed_raster = _ManagedRaster(mask_raster_path, 1, 1)

    cdef int watersheds_started = 0
    cdef cmap[int, cset[CoordinatePair]].iterator block_iterator = points_in_blocks.begin()
    cdef cset[CoordinatePair] coords_in_block
    cdef cset[CoordinatePair].iterator coord_iterator
    cdef int pixels_visited = 0
    cdef int neighbor_ws_id
    cdef unsigned char pixel_visited
    while block_iterator != points_in_blocks.end():
        block_index = deref(block_iterator).first
        coords_in_block = deref(block_iterator).second
        inc(block_iterator)

        coord_iterator = coords_in_block.begin()

        while coord_iterator != coords_in_block.end():
            current_pixel = deref(coord_iterator)
            ws_id = point_ws_ids[current_pixel]
            inc(coord_iterator)

            watersheds_started += 1
            pixels_in_watershed = 0

            if ctime(NULL) - last_log_time > 5.0:
                last_log_time = ctime(NULL)
                LOGGER.info('Delineated %s watersheds of %s so far',
                            watersheds_started, features_in_layer)

            last_ws_log_time = ctime(NULL)
            process_queue.push(current_pixel)
            process_queue_set.insert(current_pixel)
            nested_watershed_ids.clear()  # clear the set for each watershed.

            while not process_queue.empty():
                pixels_visited += 1
                pixels_in_watershed += 1
                if ctime(NULL) - last_ws_log_time > 5.0:
                    last_ws_log_time = ctime(NULL)
                    LOGGER.info('Delineating watershed %i of %i, %i pixels '
                                'found so far.', watersheds_started, features_in_layer,
                                pixels_in_watershed)

                current_pixel = process_queue.front()
                process_queue_set.erase(current_pixel)
                process_queue.pop()

                mask_managed_raster.set(current_pixel.first,
                                        current_pixel.second, 1)
                scratch_managed_raster.set(current_pixel.first,
                                           current_pixel.second, ws_id)

                for neighbor_index in range(8):
                    neighbor_pixel = CoordinatePair(
                        current_pixel.first + neighbor_col[neighbor_index],
                        current_pixel.second + neighbor_row[neighbor_index])

                    if not 0 <= neighbor_pixel.first < flow_dir_n_cols:
                        continue

                    if not 0 <= neighbor_pixel.second < flow_dir_n_rows:
                        continue

                    # Is the neighbor pixel already in the queue?
                    if (process_queue_set.find(neighbor_pixel) !=
                            process_queue_set.end()):
                        continue

                    # Is the neighbor an unvisited lake pixel in this watershed?
                    # If yes, enqueue it.
                    neighbor_ws_id = scratch_managed_raster.get(
                        neighbor_pixel.first, neighbor_pixel.second)
                    pixel_visited = mask_managed_raster.get(
                        neighbor_pixel.first, neighbor_pixel.second)
                    if (neighbor_ws_id == ws_id and pixel_visited == 0):
                        process_queue.push(neighbor_pixel)
                        process_queue_set.insert(neighbor_pixel)
                        continue

                    # Does the neighbor flow into the current pixel?
                    if (reverse_flow[neighbor_index] ==
                            flow_dir_managed_raster.get(
                                neighbor_pixel.first, neighbor_pixel.second)):

                        # Does the neighbor belong to a different outflow geometry?
                        if (neighbor_ws_id != NO_WATERSHED and
                                neighbor_ws_id != ws_id):
                            # If it is, track the watershed connectivity, but otherwise
                            # skip this pixel.  Either we've already processed it, or
                            # else we will soon!
                            nested_watershed_ids.insert(neighbor_ws_id)
                            continue

                        # If the pixel has not yet been visited, enqueue it.
                        if pixel_visited == 0:
                            process_queue.push(neighbor_pixel)
                            process_queue_set.insert(neighbor_pixel)

            nested_watersheds[ws_id] = set(nested_watershed_ids)

    flow_dir_managed_raster.close()  # Don't need this any more.
    scratch_managed_raster.close()  # flush the scratch raster.
    mask_managed_raster.close()  # flush the mask raster

    scratch_raster = gdal.OpenEx(scratch_raster_path, gdal.OF_RASTER)
    scratch_band = scratch_raster.GetRasterBand(1)
    mask_raster = gdal.OpenEx(mask_raster_path, gdal.OF_RASTER)
    mask_band = mask_raster.GetRasterBand(1)

    def _polygonize_callback(df_complete, psz_message, p_progress_arg):
        """Log progress messages during long-running polygonize calls.

        The parameters for this function are defined by GDAL."""
        try:
            current_time = time.time()
            if ((current_time - _polygonize_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and _polygonize_callback.total_time >= 5.0)):
                LOGGER.info(
                    'Fragment polygonization %.1f%% complete %s',
                    df_complete * 100, psz_message)
                _polygonize_callback.last_time = time.time()
                _polygonize_callback.total_time += current_time
        except AttributeError:
            _polygonize_callback.last_time = time.time()
            _polygonize_callback.total_time = 0.0
        except:
            # If an exception is uncaught from this callback, it will kill the
            # polygonize routine.  Just an FYI.
            LOGGER.exception('Error in polygonize progress callback')

    gdal.Polygonize(
        scratch_band,  # the source band to be analyzed
        mask_band,  # the mask band indicating valid pixels
        watershed_fragments_layer,  # polygons are added to this layer
        0,  # field index into which to save the pixel value of watershed
        ['8CONNECTED8'],  # use 8-connectedness algorithm.
        _polygonize_callback
    )

    # create a new vector with new geometries in it
    # If watersheds have nested watersheds, the geometries written should be
    # the union of all upstream sheds.
    # If a watershed is alone (does not contain nested watersheds), the
    # geometry should be written as-is.
    LOGGER.info('Copying fields over to the new fragments vector.')
    watershed_vector = gpkg_driver.CreateDataSource(target_fragments_vector_path)
    watershed_layer = watershed_vector.CreateLayer(
        'watershed_fragments', flow_dir_srs, ogr.wkbPolygon)

    # Create the fields in the target vector that already existed in the
    # outflow points vector
    for index in range(working_outlet_layer_definition.GetFieldCount()):
        field_defn = working_outlet_layer_definition.GetFieldDefn(index)
        if field_defn.GetName() == ws_id_fieldname:
            continue

        field_type = field_defn.GetType()

        if field_type in (ogr.OFTInteger, ogr.OFTReal):
            field_defn.SetWidth(24)
        watershed_layer.CreateField(field_defn)

    upstream_fragments_field = ogr.FieldDefn('upstream_fragments', ogr.OFTString)
    watershed_layer.CreateField(upstream_fragments_field)
    watershed_layer.CreateField(ws_id_field)
    watershed_layer_defn = watershed_layer.GetLayerDefn()

    # Copy over the field values to the target vector
    watershed_layer.StartTransaction()
    for watershed_fragment in watershed_fragments_layer:
        fragment_ws_id = watershed_fragment.GetField('ws_id')
        outflow_geom_fid = ws_id_to_fid[fragment_ws_id]

        watershed_feature = ogr.Feature(watershed_layer_defn)
        watershed_feature.SetGeometry(watershed_fragment.GetGeometryRef())

        outflow_point_feature = working_outlets_layer.GetFeature(outflow_geom_fid)
        for outflow_field_index in range(working_outlet_layer_definition.GetFieldCount()):
            watershed_feature.SetField(
                outflow_field_index,
                outflow_point_feature.GetField(outflow_field_index))

        watershed_feature.SetField('ws_id', float(fragment_ws_id))
        try:
            upstream_fragments = ','.join(
                [str(s) for s in sorted(nested_watersheds[fragment_ws_id])])
        except KeyError:
            upstream_fragments = ''
        watershed_feature.SetField('upstream_fragments', upstream_fragments)

        watershed_layer.CreateFeature(watershed_feature)
    watershed_layer.CommitTransaction()

    scratch_band = None
    scratch_raster = None
    mask_band = None
    mask_raster = None
    watershed_fragments_layer = None
    watershed_fragments_vector = None

    shutil.rmtree(working_dir_path, ignore_errors=True)


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
