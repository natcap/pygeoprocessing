import time
import os
import logging
import shutil
import tempfile
import itertools 
import math
import collections

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

try:
    from shapely.geos import ReadingError
except ImportError:
    from shapely.errors import ReadingError

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
            (self.block_ysize, self.block_xsize), dtype=numpy.int32)
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


def _make_polygonize_callback(logging_adapter):
    """Create a callback function for gdal.Polygonize.
    
    Parameters:
        logging_adapter (logging.Logger): A logging.Logger or context adapter
            to which records will be logged.
            
    Returns:
        A callback compatible with gdal.Polygonize.
    """
    def _polygonize_callback(df_complete, psz_message, p_progress_arg):
        """Log progress messages during long-running polygonize calls.

        The parameters for this function are defined by GDAL."""
        try:
            current_time = time.time()
            if ((current_time - _polygonize_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and _polygonize_callback.total_time >= 5.0)):
                logging_adapter.info(
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
            logging_adapter.exception('Error in polygonize progress callback')
    return _polygonize_callback


class OverlappingWatershedContextAdapter(logging.LoggerAdapter):
    """Contextual logger for noting where we are in disjoint set iteration."""

    def process(self, msg, kwargs):
        """Prepare contextual information for passed log messages.
        
        Parameters:
            msg (string): The templated string log message to prepend to.
            kwargs (dict): Keyword arguments for logging, passed through.
            
        Returns:
            A tuple of (string log message, kwargs).  The log message has some
            extra information about which pass we're in the middle of.
        """
        return (
            '[pass %s of %s] %s' % (
                self.extra['pass_index'], self.extra['n_passes'], msg),
            kwargs)


# It's convenient to define a C++ pair here as a pair of longs to represent the
# x,y coordinates of a pixel.  So, CoordinatePair().first is the x coordinate,
# CoordinatePair().second is the y coordinate.  Both are in integer pixel
# coordinates.
ctypedef pair[long, long] CoordinatePair


# Phase 0: preprocess working geometries
#    * Attempt to repair invalid geometries.
#    * Create new WS_ID field, set field value.
#    * Exclude any geometries that do not intersect the DEM bbox (prepared geometry op)
#
# Phase 1: Prepare working geometries for determining sets of disjoint polygons.
#    * Points and very small geometries can be removed, as we already know their seed coords.
#         * Track these in a {CoordinatePair: set of ws_ids} structure
#    * Alter geometries of remaining polygons and lines as needed
#
# Phase 2: Determine disjoint sets
#    * Write out new vectors
#
# Phase 3: determine seed coordinates for remaining geometries
#    * For each disjoint vector:
#        * Create a new UInt32 raster (new raster is for debugging)
#        * rasterize the vector's ws_id column.
#        * iterblocks over the resulting raster, tracking seed coordinates
#            * If a seed is over nodata, skip it.
#
# Phase 4: Cluster seeds by flow direction block
#    * Use a spatial index to find which block a seed coordinate belongs to.
#
# Phase 5: Iteration.
#    * For each blockwise seed coordinate cluster:
#        * Process neighbors.
#        * If a neighbor is upstream and is a known seed, mark connectivity between seeds.
# 
# Phase 6: Polygonization
#    * Polygonize the seeds fragments.
#
# Phase 7: Copy field values over to the new fragments.


# In join_watershed_fragments:
# Phase 1: Take the union of all fragments with the same ws_id, noting union of all upstream seeds.
# Phase 2: Recurse the data structure, building nested geometries.

def delineate_watersheds_d8(
        d8_flow_dir_raster_path_band, outflow_vector_path,
        target_fragments_vector_path, working_dir=None):
    """Delineate watersheds from a D8 flow direction raster.

    This function produces a vector of watershed fragments, where each fragment
    represents the area that flows into each outflow point.  Nested watersheds
    are represented by the field

    Parameters:
        d8_flow_dir_raster_path_band (tuple): A two-tuple representing the
            string path to the D8 flow direction raster to use and the band
            index to use.
        outflow_vector_path (string): Path to a vector on disk representing
            outflow geometries for watersheds.  This vector must have one layer
            and must be in the same projection as the flow direction raster.
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

    flow_dir_info = pygeoprocessing.get_raster_info(
        d8_flow_dir_raster_path_band[0])
    source_gt = flow_dir_info['geotransform']
    cdef long flow_dir_n_cols = flow_dir_info['raster_size'][0]
    cdef long flow_dir_n_rows = flow_dir_info['raster_size'][1]
    cdef int flow_dir_block_x_size = flow_dir_info['block_size'][0]
    cdef int flow_dir_block_y_size = flow_dir_info['block_size'][1]
    cdef double x_origin = source_gt[0]
    cdef double y_origin = source_gt[3]
    cdef double x_pixelwidth = source_gt[1]
    cdef double y_pixelwidth = source_gt[5]
    cdef double pixel_area = abs(x_pixelwidth * y_pixelwidth)
    try:
        if working_dir is not None:
            os.makedirs(working_dir)
    except OSError:
        pass
    working_dir_path = tempfile.mkdtemp(
        dir=working_dir, prefix='watershed_delineation_%s_' % time.strftime(
            '%Y-%m-%d_%H_%M_%S', time.gmtime()))

    gpkg_driver = gdal.GetDriverByName('GPKG')
    flow_dir_srs = osr.SpatialReference()
    flow_dir_srs.ImportFromWkt(flow_dir_info['projection'])
    flow_dir_bbox_geometry = shapely.prepared.prep(
        shapely.geometry.box(*flow_dir_info['bounding_box']))

    source_outlets_vector = gdal.OpenEx(outflow_vector_path, gdal.OF_VECTOR)
    working_outlets_path = os.path.join(working_dir_path, 'working_outlets.gpkg')
    working_outlets_vector = gpkg_driver.CreateCopy(working_outlets_path,
                                                    source_outlets_vector)
    working_outlets_layer = working_outlets_vector.GetLayer()

    # Add a new field to the clipped_outlets_layer to ensure we know what field
    # values are bing rasterized.
    cdef int ws_id
    ws_id_fieldname = '__ws_id__'
    ws_id_field_defn = ogr.FieldDefn(ws_id_fieldname, ogr.OFTInteger64)
    working_outlets_layer.CreateField(ws_id_field_defn)
    outlets_schema = working_outlets_layer.schema

    # Phase 0: preprocess working geometries.
    #    * Attempt to repair invalid geometries.
    #    * Create new WS_ID field, set field value.
    #    * Exclude any geometries that do not intersect the DEM bbox (prepared geometry op)
    ws_id = 1  # start indexing ws_id at 1
    working_vector_ws_id_to_fid = {}
    working_outlets_layer.StartTransaction()
    # TODO: add progress logging
    LOGGER.info('Preprocessing outflow geometries.')
    for feature in working_outlets_layer:
        ogr_geom = feature.GetGeometryRef()
        try:
            geometry = shapely.wkb.loads(ogr_geom.ExportToWkb())
        except ReadingError:
            # This happens when a polygon isn't a closed ring.
            ogr_geom.CloseRings()
            ogr_geom.Buffer(0)
            geometry = shapely.wkb.loads(ogr_geom.ExportToWkb())
            feature.SetGeometry(ogr.CreateGeometryFromWkb(geometry.wkb))

        fid = feature.GetFID()

        # If the geometry doesn't intersect the flow direction bounding box,
        # no need to include it in any of the watershed processing.
        if not flow_dir_bbox_geometry.intersects(geometry):
            working_outlets_layer.DeleteFeature(fid)
            continue

        working_vector_ws_id_to_fid[ws_id] = fid
        feature.SetField(ws_id_fieldname, ws_id)
        working_outlets_layer.SetFeature(feature)
        ws_id += 1  # only track ws_ids that end up in the vector.
    working_outlets_layer.CommitTransaction()

    # Phase 1: Prepare working geometries for determining sets of disjoint polygons.
    #    * Points and very small geometries can be removed, as we already know their seed coords.
    #         * Track these in a {CoordinatePair: set of ws_ids} structure
    #    * Alter geometries of remaining polygons and lines as needed
    cdef cmap[CoordinatePair, cset[int]] seed_watersheds
    cdef cmap[CoordinatePair, int] seed_ids  # {CoordinatePair: seed id}
    cdef CoordinatePair seed
    cdef int seed_id = 1
    cdef double buffer_dist = math.hypot(x_pixelwidth, y_pixelwidth) / 2. * 1.1

    buffered_working_outlets_path = os.path.join(working_dir_path,
                                                 'working_outlets_buffered.gpkg')
    buffered_working_outlets_vector = gpkg_driver.CreateCopy(buffered_working_outlets_path,
                                                             working_outlets_vector)
    buffered_working_outlets_layer = buffered_working_outlets_vector.GetLayer()
    buffered_working_outlets_layer.StartTransaction()
    # TODO: add progress logging
    LOGGER.info('First pass over outflow features')
    for feature in buffered_working_outlets_layer:
        geometry = shapely.wkb.loads(feature.GetGeometryRef().ExportToWkb())

        minx, miny, maxx, maxy = geometry.bounds
        minx_pixelcoord = (minx - x_origin) // x_pixelwidth
        miny_pixelcoord = (miny - y_origin) // y_pixelwidth
        maxx_pixelcoord = (maxx - x_origin) // x_pixelwidth
        maxy_pixelcoord = (maxy - y_origin) // y_pixelwidth

        # If the geometry only intersects a single pixel, we can treat it
        # as a single point, which means that we can track it directly in our
        # seeds data structure and not have to include it in the disjoint set
        # determination.
        if minx_pixelcoord == maxx_pixelcoord and miny_pixelcoord == maxy_pixelcoord:
            # Repr. point allows this geometry to be of any type.
            geometry = geometry.representative_point()
            ws_id = feature.GetField(ws_id_fieldname)
            seed = CoordinatePair(minx_pixelcoord, miny_pixelcoord)

            if (seed_watersheds.find(seed) == seed_watersheds.end()):
                seed_watersheds[seed] = cset[int]()
            seed_watersheds[seed].insert(ws_id)
            seed_ids[seed] = seed_id
            seed_id += 1

            # Don't need the feature any more!
            buffered_working_outlets_layer.DeleteFeature(feature.GetFID())
            continue

        # If we can't fit the geometry into a single pixel, there remain
        # two special cases that warrant buffering:
        #     * It's a line (lines don't have area). We want to avoid a
        #       situation where multiple lines cover the same pixels
        #       but don't intersect.  Such geometries should be handled
        #       in disjoint sets.
        #     * It's a polygon that has area smaller than a pixel. This
        #       came up in real-world sample data (the Montana Lakes
        #       example, specifically), where some very small lakes were
        #       disjoint but both overlapped only one pixel, the same pixel.
        #       This lead to a race condition in rasterization where only
        #       one of them would be in the output vector.
        elif (geometry.area == 0.0 or geometry.area <= pixel_area):
            new_geometry = geometry.buffer(buffer_dist)
        else:
            # No need to set the geometry, just continue.
            continue

        feature.SetGeometry(ogr.CreateGeometryFromWkb(new_geometry.wkb))
        buffered_working_outlets_layer.SetFeature(feature)

    buffered_working_outlets_layer.CommitTransaction()

    # Phase 2: Determine disjoint sets for any remaining geometries.
    #    * Write out new vectors
    cdef int NO_WATERSHED = 0
    cdef int row, col
    if buffered_working_outlets_layer.GetFeatureCount() > 0:
        LOGGER.info('Determining sets of non-overlapping geometries')
        for set_index, disjoint_polygon_fid_set in enumerate(
                pygeoprocessing.calculate_disjoint_polygon_set(
                    buffered_working_outlets_path,
                    bounding_box=flow_dir_info['bounding_box']),
                start=1):

            LOGGER.info("Creating a vector of %s disjoint geometries",
                        len(disjoint_polygon_fid_set))
            disjoint_vector_path = os.path.join(
                    working_dir_path, 'disjoint_outflow_%s.gpkg' % set_index)
            disjoint_vector = gpkg_driver.Create(disjoint_vector_path, 0, 0, 0,
                                                 gdal.GDT_Unknown)
            disjoint_layer = disjoint_vector.CreateLayer(
                'outlet_geometries', flow_dir_srs, ogr.wkbPolygon)
            disjoint_layer.CreateFields(outlets_schema)

            disjoint_layer.StartTransaction()
            # Though the buffered working layer was used for determining the sets
            # of nonoverlapping polygons, we want to use the *original* outflow
            # geometries for rasterization.  This step gets the appropriate features
            # from the correct vectors so we keep the correct geometries.
            for polygon_fid in disjoint_polygon_fid_set:
                original_feature = buffered_working_outlets_layer.GetFeature(polygon_fid)
                ws_id = original_feature.GetField(ws_id_fieldname)
                new_feature = working_outlets_layer.GetFeature(
                    working_vector_ws_id_to_fid[ws_id]).Clone()
                disjoint_layer.CreateFeature(new_feature)
            disjoint_layer.CommitTransaction()

            disjoint_layer = None
            disjoint_vector = None

            # Phase 3: determine seed coordinates for remaining geometries
            #    * For each disjoint vector:
            #        * Create a new UInt32 raster (new raster is for debugging)
            #        * rasterize the vector's ws_id column.
            #        * iterblocks over the resulting raster, tracking seed coordinates
            #            * If a seed is over nodata, skip it.
            tmp_seed_raster_path = os.path.join(working_dir_path,
                                                'disjoint_outflow_%s.tif' % set_index)
            pygeoprocessing.new_raster_from_base(
                d8_flow_dir_raster_path_band[0], tmp_seed_raster_path,
                gdal.GDT_UInt32, [NO_WATERSHED], fill_value_list=[NO_WATERSHED],
                gtiff_creation_options=GTIFF_CREATION_OPTIONS)

            pygeoprocessing.rasterize(
                disjoint_vector_path, tmp_seed_raster_path, None,
                ['ALL_TOUCHED=TRUE', 'ATTRIBUTE=%s' % ws_id_fieldname],
                layer_index='outlet_geometries')

            for block_info, block in pygeoprocessing.iterblocks(tmp_seed_raster_path):
                for (row, col) in zip(*numpy.nonzero(block)):
                    ws_id = block[row, col]
                    seed = CoordinatePair(col + block_info['xoff'],
                                          row + block_info['yoff'])

                    # Insert the seed into the seed_watersheds structure.
                    if (seed_watersheds.find(seed) == seed_watersheds.end()):
                        seed_watersheds[seed] = cset[int]()
                    seed_watersheds[seed].insert(ws_id)
                    seed_ids[seed] = seed_id
                    seed_id += 1
            del block
            del block_info

        buffered_working_outlets_layer = None
    buffered_working_outlets_vector = None
    del disjoint_polygon_fid_set

    # Phase 4: Cluster seeds by flow direction block
    #    * Use math to find which block a seed coordinate belongs to.
    #      Calculating the index directly is a constant-time operation, which is
    #      much cheaper than using a spatial index.
    #
    # build up a map of which seeds are in which block and
    # identify seeds by an integer index.
    LOGGER.info('Splitting seeds into their blocks.')
    cdef int block_index
    cdef int n_blocks = (
        ((flow_dir_n_cols // flow_dir_block_x_size) + 1) *
        ((flow_dir_n_rows // flow_dir_block_y_size) + 1))
    cdef cmap[int, cset[CoordinatePair]] seeds_in_block
    cdef cmap[CoordinatePair, cset[int]].iterator seeds_in_watersheds_iterator = seed_watersheds.begin()

    # Initialize the seeds_in_block data structure
    for block_index in range(n_blocks):
        seeds_in_block[block_index] = cset[CoordinatePair]()

    while seeds_in_watersheds_iterator != seed_watersheds.end():
        # Only need the seed at present; don't need the member ws_ids.
        seed = deref(seeds_in_watersheds_iterator).first
        inc(seeds_in_watersheds_iterator)

        # Determine the block index mathematically.  We only need to be able to
        # group pixels together, so the specific number used does not matter.
        block_index = (
            (seed.first // flow_dir_block_x_size) * flow_dir_block_x_size + 
            (seed.second // flow_dir_block_y_size))
        seeds_in_block[block_index].insert(seed)

    # Phase 5: Iteration.
    #    * For each blockwise seed coordinate cluster:
    #        * Process neighbors.
    #        * If a neighbor is upstream and is a known seed, mark connectivity between seeds.
    cdef cmap[int, cset[int]] upstream_seed_ids  # {int seed_id: int upstream seed_id}
    cdef cset[int] nested_fragment_ids
    cdef int* neighbor_col = [1, 1, 0, -1, -1, -1, 0, 1]
    cdef int* neighbor_row = [0, -1, -1, -1, 0, 1, 1, 1]
    cdef int* reverse_flow = [4, 5, 6, 7, 0, 1, 2, 3]
    cdef cset[CoordinatePair] process_queue_set
    cdef queue[CoordinatePair] process_queue

    scratch_raster_path = os.path.join(
            working_dir_path, 'scratch_raster.tif')
    pygeoprocessing.new_raster_from_base(
        d8_flow_dir_raster_path_band[0], scratch_raster_path,
        gdal.GDT_UInt32, [NO_WATERSHED], fill_value_list=[NO_WATERSHED],
        gtiff_creation_options=GTIFF_CREATION_OPTIONS)
    scratch_managed_raster = _ManagedRaster(scratch_raster_path, 1, 1)

    mask_raster_path = os.path.join(working_dir_path, 'mask_raster.tif')
    pygeoprocessing.new_raster_from_base(
        d8_flow_dir_raster_path_band[0], mask_raster_path,
        gdal.GDT_Byte, [255], fill_value_list=[0],
        gtiff_creation_options=GTIFF_CREATION_OPTIONS)
    mask_managed_raster = _ManagedRaster(mask_raster_path, 1, 1)

    flow_dir_managed_raster = _ManagedRaster(d8_flow_dir_raster_path_band[0],
                                             d8_flow_dir_raster_path_band[1],
                                             0)  # read-only

    LOGGER.info('Starting delineation from %s seeds', seed_id)
    cdef cmap[int, cset[int]] nested_fragments
    last_log_time = ctime(NULL)
    block_iterator = seeds_in_block.begin()
    cdef cset[CoordinatePair].iterator seed_iterator
    cdef CoordinatePair current_pixel, neighbor_pixel
    for block_index in range(n_blocks):
        seeds_in_current_block = seeds_in_block[block_index]
        seed_iterator = seeds_in_current_block.begin()

        while seed_iterator != seeds_in_current_block.end():
            current_pixel = deref(seed_iterator)
            seed_id = seed_ids[current_pixel]
            inc(seed_iterator)

            last_ws_log_time = ctime(NULL)
            process_queue.push(current_pixel)
            process_queue_set.insert(current_pixel)
            nested_fragment_ids.clear()  # clear the set for each fragment.

            while not process_queue.empty():
                if ctime(NULL) - last_log_time > 5.0:
                    LOGGER.info('Delineating watersheds')
                    last_log_time = ctime(NULL)

                current_pixel = process_queue.front()
                process_queue_set.erase(current_pixel)
                process_queue.pop()

                mask_managed_raster.set(current_pixel.first,
                                        current_pixel.second, 1)
                scratch_managed_raster.set(current_pixel.first,
                                           current_pixel.second, seed_id)

                for neighbor_index in range(8):
                    neighbor_pixel = CoordinatePair(
                        current_pixel.first + neighbor_col[neighbor_index],
                        current_pixel.second + neighbor_row[neighbor_index])

                    # Is the neighbor off the bounds of the raster?
                    # skip if so.
                    if not 0 <= neighbor_pixel.first < flow_dir_n_cols:
                        continue

                    if not 0 <= neighbor_pixel.second < flow_dir_n_rows:
                        continue

                    # Is the neighbor pixel already in the queue?
                    # Skip if so.
                    if (process_queue_set.find(neighbor_pixel) !=
                            process_queue_set.end()):
                        continue

                    # Does the neighbor flow into the current pixel?
                    if (reverse_flow[neighbor_index] ==
                            flow_dir_managed_raster.get(
                                neighbor_pixel.first, neighbor_pixel.second)):

                        # Does the neighbor belong to a different outflow
                        # geometry (is it a seed)?
                        if (seed_ids.find(neighbor_pixel) != seed_ids.end()):
                            # If it is, track the fragment connectivity,
                            # but otherwise skip this pixel.  Either we've
                            # already processed it, or else we will soon!
                            nested_fragment_ids.insert(seed_ids[neighbor_pixel])
                            continue

                        # If the pixel has not yet been visited, enqueue it.
                        pixel_visited = mask_managed_raster.get(
                            neighbor_pixel.first, neighbor_pixel.second)
                        if pixel_visited == 0:
                            process_queue.push(neighbor_pixel)
                            process_queue_set.insert(neighbor_pixel)

            nested_fragments[seed_id] = nested_fragment_ids
    scratch_managed_raster.close()  # flush the scratch raster.
    mask_managed_raster.close()  # flush the mask raster
    flow_dir_managed_raster.close()  # don't need this any longer.

    seeds_in_block.clear()

    # Phase 6: Polygonization
    #    * Polygonize the seeds fragments.
    LOGGER.info('Polygonizing fragments')
    scratch_raster = gdal.OpenEx(scratch_raster_path, gdal.OF_RASTER)
    scratch_band = scratch_raster.GetRasterBand(1)
    mask_raster = gdal.OpenEx(mask_raster_path, gdal.OF_RASTER)
    mask_band = mask_raster.GetRasterBand(1)

    target_fragments_vector = gpkg_driver.Create(target_fragments_vector_path,
                                                 0, 0, 0, gdal.GDT_Unknown)
    if target_fragments_vector is None:
        raise RuntimeError(  # Because I frequently have this open in QGIS when I shouldn't.
            "Could not open target fragments vector for writing. Do you have "
            "access to this path?  Is the file open in another program?")

    # Create a non-spatial database layer for storing all of the watershed
    # attributes without the geometries.  This is important because we need to
    # know the ws_id of each outlet feature.
    # TODO: don't include watersheds that are not represented in the fragments
    target_fragments_watershed_attrs_layer = target_fragments_vector.CreateLayer(
        'watershed_attributes', geom_type=ogr.wkbNone,
        options=['ASPATIAL_VARIANT=OGR_ASPATIAL'])
    target_fragments_watershed_attrs_layer.CreateFields(working_outlets_layer.schema)
    target_fragments_watershed_attrs_layer.StartTransaction()
    working_outlets_layer.ResetReading()
    for working_feature in working_outlets_layer:
        new_feature = ogr.Feature(
            target_fragments_watershed_attrs_layer.GetLayerDefn())
        for attr_name, attr_value in working_feature.items().items():
            new_feature.SetField(attr_name, attr_value)
        target_fragments_watershed_attrs_layer.CreateFeature(new_feature)
    target_fragments_watershed_attrs_layer.CommitTransaction()

    # Create a spatial layer for the fragment geometries.
    # This layer only needs to know which fragments are upstream of a given fragment.
    target_fragments_layer = target_fragments_vector.CreateLayer(
        'watershed_fragments', flow_dir_srs, ogr.wkbPolygon)

    # Create the fields in the target vector that already existed in the
    # outflow points vector
    target_fragments_layer.CreateField(ogr.FieldDefn('fragment_id', ogr.OFTInteger64))
    target_fragments_layer.CreateField(ogr.FieldDefn('upstream_fragments', ogr.OFTString))
    target_fragments_layer.CreateField(ogr.FieldDefn('member_ws_ids', ogr.OFTString))

    gdal.Polygonize(
        scratch_band,  # the source band to be analyzed
        mask_band,  # the mask band indicating valid pixels
        target_fragments_layer,  # polygons are added to this layer
        0,  # field index of 'fragment_id' field.
        ['8CONNECTED8'],  # use 8-connectedness algorithm.
        _make_polygonize_callback(LOGGER)
    )

    # Reverse the {seed: seed ID} mapping for use in retrieving features.
    cdef cmap[int, CoordinatePair] seed_id_to_seed  # {seed id: CoordinatePair}
    cdef cmap[CoordinatePair, int].iterator seed_to_id_iterator = seed_ids.begin()
    while seed_to_id_iterator != seed_ids.end():
        seed = deref(seed_to_id_iterator).first
        seed_id = deref(seed_to_id_iterator).second
        seed_id_to_seed[seed_id] = seed
        inc(seed_to_id_iterator)

    # Polygonization only copies the fragment IDs. Copy over the relevant
    # attributes that will link the fragments into whole watersheds.
    LOGGER.info('Copying fragments to the output vector.')
    target_fragments_layer.StartTransaction()
    cdef int fragment_id
    cdef CoordinatePair fragment_seed
    cdef cset[int].iterator ws_id_iterator
    # TODO: progress logging here.
    for created_fragment in target_fragments_layer:
        fragment_id = created_fragment.GetFieldAsInteger('fragment_id')
        fragment_seed = seed_id_to_seed[fragment_id]
        created_fragment.SetField('upstream_fragments',
            ','.join([str(s) for s in sorted(nested_fragments[fragment_id])]))
        created_fragment.SetField('member_ws_ids',
            ','.join([str(s) for s in sorted(set(seed_watersheds[fragment_seed]))]))
        target_fragments_layer.SetFeature(created_fragment)

    target_fragments_layer.CommitTransaction()

    # Close the watershed fragments vector from this disjoint geometry set.
    disjoint_outlets_layer = None
    disjoint_outlets_vector = None
    watershed_fragments_layer = None
    watershed_fragments_vector = None


def join_watershed_fragments_d8(watershed_fragments_vector, target_watersheds_path):
    fragments_vector = gdal.OpenEx(watershed_fragments_vector, gdal.OF_VECTOR)
    fragments_layer = fragments_vector.GetLayer('watershed_fragments')
    outflow_attributes_layer = fragments_vector.GetLayer('watershed_attributes')
    fragments_srs = fragments_layer.GetSpatialRef()

    driver = gdal.GetDriverByName('GPKG')
    watersheds_vector = driver.Create(target_watersheds_path, 0, 0, 0,
                                      gdal.GDT_Unknown)
    watersheds_layer = watersheds_vector.CreateLayer(
        'watersheds', fragments_srs, ogr.wkbPolygon)
    watersheds_layer_defn = watersheds_layer.GetLayerDefn()

    for field_defn in outflow_attributes_layer.schema:
        field_type = field_defn.GetType()
        if field_type in (ogr.OFTInteger, ogr.OFTReal):
            field_defn.SetWidth(24)
        watersheds_layer.CreateField(field_defn)

    # Phase 1: Load in all of the fragments
    upstream_fragments = collections.defaultdict(list)
    fragment_geometries = {}
    fragments_in_watershed = collections.defaultdict(set)  # {ws_id: set(fragment_ids)}
    for fragment in fragments_layer:
        fragment_id = fragment.GetField('fragment_id')
        member_ws_ids = set([int(ws_id) for ws_id in
                             fragment.GetField('member_ws_ids').split(',')])
        for ws_id in member_ws_ids:
            fragments_in_watershed[ws_id].add(fragment_id)

        try:
            upstream_fragments[fragment_id] = [
                int(f) for f in fragment.GetField('upstream_fragments').split(',')]
        except ValueError:
            # If no upstream fragments the string will be '', and ''.split(',')
            # turns into [''], which crashes when you cast it to an int.
            # We're using a defaultdict(list) here, so no need to do anything.
            # AttributeError when field value is None.
            pass

        fragment_geometries[fragment_id] = shapely.wkb.loads(
            fragment.GetGeometryRef().ExportToWkb())

    # Construct the base geometries for each watershed.
    # Whatever is in this dict is assumed to be a complete geometry for this fragment.
    # Initialize this dict with any fragments that do not have any upstream fragments.
    compiled_upstream_geometries = {}  # {fragment_id: shapely geometry}
    for fragment_id in fragment_geometries.keys():
        if not upstream_fragments[fragment_id]:
            compiled_upstream_geometries[fragment_id] = fragment_geometries[fragment_id]

    # prepopulate the stack. Cast to list for python3 compatibility.
    for starter_fragment_id in fragment_geometries.keys():
        # If the geometry has already been compiled, we can skip the stack step.
        if starter_fragment_id in compiled_upstream_geometries:
            continue
        
        # Copy the upstream fragments list to a new list so we don't mutate the
        # original one.
        stack = upstream_fragments[starter_fragment_id][:]
        geometries = [fragment_geometries[starter_fragment_id]]
        while len(stack) > 0:
            fragment_id = stack.pop()
            
            try:
                # Base case: this fragment already has geometry in compiled dict
                geometries.append(compiled_upstream_geometries[fragment_id])
            except KeyError:
                # Fragment geometry has not yet been compiled.
                # Get the fragment's geometry and push it onto the stack.

                # Are all of the geometries upstream of this fragment in the compiled dict?
                # If yes, cascade-union them and save.
                upstream_fragment_ids = upstream_fragments[fragment_id]
                if all(k in compiled_upstream_geometries
                       for k in upstream_fragment_ids):
                    compiled_upstream_geometries[fragment_id] = (
                        shapely.ops.cascaded_union(
                            [compiled_upstream_geometries[f]
                             for f in upstream_fragment_ids]))
                    geometries.append(compiled_upstream_geometries[fragment_id])
                            
                # If not, push the current fragment to the stack (so we visit it later)
                else:
                    stack.append(fragment_id)
                    stack += upstream_fragment_ids
        compiled_upstream_geometries[starter_fragment_id] = shapely.ops.cascaded_union(geometries)

    # Load the attributes table into a dict for easier accesses
    watershed_attributes = {}
    for feature in outflow_attributes_layer:
        watershed_attributes[feature.GetField('__ws_id__')] = feature.items()

    watersheds_layer.StartTransaction()
    for ws_id, fragments_set in fragments_in_watershed.items():
        target_feature = ogr.Feature(watersheds_layer_defn)

        # Compile the geometry from all member fragments.
        watershed_geometry = shapely.ops.cascaded_union([
            compiled_upstream_geometries[fragment_id]
            for fragment_id in fragments_set])
        target_feature.SetGeometry(
            ogr.CreateGeometryFromWkb(watershed_geometry.wkb))

        # Copy field values over to the new feature.
        for field_name, field_value in watershed_attributes[ws_id].items():
            target_feature.SetField(field_name, field_value)

        watersheds_layer.CreateFeature(target_feature)
    watersheds_layer.CommitTransaction()


def old_delineation(
        d8_flow_dir_raster_path_band, outflow_vector_path,
        target_fragments_vector_path, working_dir=None):
    """Delineate watersheds from a D8 flow direction raster.

    This function produces a vector of watershed fragments, where each fragment
    represents the area that flows into each outflow point.  Nested watersheds
    are represented by the field

    Parameters:
        d8_flow_dir_raster_path_band (tuple): A two-tuple representing the
            string path to the D8 flow direction raster to use and the band
            index to use.
        outflow_vector_path (string): Path to a vector on disk representing
            outflow geometries for watersheds.  This vector must have one layer
            and must be in the same projection as the flow direction raster.
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

    # Create the watershed fragments layer for later.
    gpkg_driver = gdal.GetDriverByName('GPKG')
    flow_dir_srs = osr.SpatialReference()
    flow_dir_srs.ImportFromWkt(flow_dir_info['projection'])
    flow_dir_bbox_geometry = shapely.geometry.box(*flow_dir_info['bounding_box'])

    source_outlets_vector = gdal.OpenEx(outflow_vector_path, gdal.OF_VECTOR)
    working_outlets_path = os.path.join(working_dir_path, 'working_outlets.gpkg')
    working_outlets_vector = gpkg_driver.CreateCopy(working_outlets_path,
                                                    source_outlets_vector)
    working_outlets_layer = working_outlets_vector.GetLayer()

    # Add a new field to the clipped_outlets_layer to ensure we know what field
    # values are bing rasterized.
    cdef int ws_id
    ws_id_fieldname = '__ws_id__'
    ws_id_field_defn = ogr.FieldDefn(ws_id_fieldname, ogr.OFTInteger64)
    working_outlets_layer.CreateField(ws_id_field_defn)

    cdef double x_origin = source_gt[0]
    cdef double y_origin = source_gt[3]
    cdef double x_pixelwidth = source_gt[1]
    cdef double y_pixelwidth = source_gt[5]
    cdef int n_outlet_features = working_outlets_layer.GetFeatureCount()

    # FIDs are unreliable, so we use the ws_id field to identify geometries.
    LOGGER.info("Preprocessing outlet geometries")
    working_outlets_layer.StartTransaction()
    working_vector_ws_id_to_fid = {}
    for ws_id, feature in enumerate(working_outlets_layer, start=1):
        working_vector_ws_id_to_fid[ws_id] = feature.GetFID()
        feature.SetField(ws_id_fieldname, ws_id)

        try:
            ogr_geom = feature.GetGeometryRef()
            geometry = shapely.wkb.loads(ogr_geom.ExportToWkb())
        except ReadingError:
            # Attempt to load the geometry as a shapely object. If this fails, try
            # to fix the geometry.  The most common reason for this to fail is
            # when polygons don't form closed rings.
            ogr_geom.CloseRings()
            ogr_geom.Buffer(0)
            geometry = shapely.wkb.loads(ogr_geom.ExportToWkb())
            feature.SetGeometry(ogr.CreateGeometryFromWkb(geometry.wkb))

        working_outlets_layer.SetFeature(feature)
    working_outlets_layer.CommitTransaction()

    # Because of the realities of real-world data, there are a couple cases
    # that need to be handled before we determine the nonoverlapping sets
    # of geometries:
    #    * Points are unlikely to overlap as-is, so their geometries need to be
    #      buffered.  Otherwise, we won't know which point will end up being
    #      rasterized.  Solution: create a small square in the center of the pixel
    #      that overlaps with the point.
    #      As a convenient optimization, we also track which points have duplicate
    #      geometries so that we avoid making more passes than absolutely necessary.
    #      Duplicate fragments have new rows created for them at the end of delineation.
    #    * Polygons only need some extra attention when they are very small. The
    #      Montana lakes dataset had several examples of very small lake polygons
    #      overlapping the same pixel, resulting in a race condition for which would be
    #      rasterized.  The solution here is to buffer small polygons so we can more
    #      accurately determine sets of disjoint polygons, and then we'll use the actual
    #      geometries for the watershed calculations.
    #
    # Other geometric types are likely to work here, but I don't have sample data for them.
    # If you find something doesn't work right, please email your sample data to
    # jdouglass@stanford.edu.
    duplicate_points = {}  # Map center of points to the ws_id of the first point found.
    duplicate_point_ws_ids = {}  # Map origial ws_id to ws_ids of points over the same pixel.
    duplicate_point_fids = set([])
    buffered_working_outlets_path = os.path.join(working_dir_path, 'working_outlets_buffered.gpkg')
    buffered_working_outlets_vector = gpkg_driver.CreateCopy(buffered_working_outlets_path,
                                                             working_outlets_vector)
    buffered_working_outlets_layer = buffered_working_outlets_vector.GetLayer()
    buffered_working_outlets_layer.StartTransaction()
    cdef int minx_pixelcoord, maxx_pixelcoord, miny_pixelcoord, maxy_pixelcoord
    cdef double pixel_area = abs(x_pixelwidth * y_pixelwidth)
    cdef double half_pixelwidth = x_pixelwidth / 2.
    cdef double half_pixelheight = y_pixelwidth / 2.

    # The disjoint_polygon_set determination process returns sets of polygons
    # where the geometries in the set share no vertices whatsoever. When we
    # calculate sets of disjoint polygons on geometries that overlap exactly
    # one pixel, we need to use geometries that 1) cover almost all of the
    # pixel and 2) do not intersect with the neighboring pixels.
    # This way, we can have include more small geometries into a disjoint set.
    cdef double x_half_boxwidth = half_pixelwidth * 0.95
    cdef double y_half_boxheight = half_pixelheight * 0.95

    # Using slightly more than half the hypotenuse of the pixel's height and
    # width should guarantee that we won't have disjoint polygons that cover
    # the same pixel.
    cdef double buffer_dist = math.hypot(x_pixelwidth, y_pixelwidth) / 2. * 1.1
    cdef time_t last_log_time, last_ws_log_time
    last_log_time = ctime(NULL)
    cdef int n_features_preprocessed = 0

    for feature in buffered_working_outlets_layer:
        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            LOGGER.info('Preprocessing outlet geometries approx %.1f%% complete',
                        (n_features_preprocessed / <float>n_outlet_features) * 100.)
        n_features_preprocessed += 1

        geometry = shapely.wkb.loads(feature.GetGeometryRef().ExportToWkb())

        minx, miny, maxx, maxy = geometry.bounds
        minx_pixelcoord = (minx - x_origin) // x_pixelwidth
        miny_pixelcoord = (miny - y_origin) // y_pixelwidth
        maxx_pixelcoord = (maxx - x_origin) // x_pixelwidth
        maxy_pixelcoord = (maxy - y_origin) // y_pixelwidth

        # If the geometry only intersects a single pixel, we can treat it as a point,
        # which allows us to trivially check for functionally-duplicate geometries and
        # avoid recomputing some watersheds.
        if minx_pixelcoord == maxx_pixelcoord and miny_pixelcoord == maxy_pixelcoord:
            # Snap the point to the nearest Flow Dir pixel and create a small
            # polygon.
            geometry = geometry.representative_point()

            x_coord = geometry.x - ((geometry.x - x_origin) % x_pixelwidth) + x_pixelwidth/2.
            y_coord = geometry.y - ((geometry.y - y_origin) % y_pixelwidth) + y_pixelwidth/2.

            ws_id = feature.GetField(ws_id_fieldname)
            coord_pair = (x_coord, y_coord)
            if coord_pair not in duplicate_points:
                # This is the first occurrance of this point geometry
                duplicate_points[coord_pair] = ws_id
                duplicate_point_ws_ids[ws_id] = []
            else:
                duplicate_point_fids.add(feature.GetFID())
                duplicate_point_ws_ids[duplicate_points[coord_pair]].append(ws_id)
                continue  # no need to create this geometry; we'll copy it over later.

            new_geometry = shapely.geometry.box(
                x_coord - x_half_boxwidth,
                y_coord - y_half_boxheight,
                x_coord + x_half_boxwidth,
                y_coord + y_half_boxheight)

        # If we can't fit the geometry into a single pixel, there remain
        # two special cases that warrant buffering:
        #     * It's a line (lines don't have area). We want to avoid a
        #       situation where multiple lines cover the same pixels
        #       but don't intersect.  Such geometries should be handled
        #       in disjoint sets.
        #     * It's a polygon that has area smaller than a pixel. This
        #       came up in real-world sample data (the Montana Lakes
        #       example, specifically), where some very small lakes were
        #       disjoint but both overlapped only one pixel, the same pixel.
        #       This lead to a race condition in rasterization where only
        #       one of them would be in the output vector.
        elif (geometry.area == 0.0 or geometry.area <= pixel_area):
            new_geometry = geometry.buffer(buffer_dist)
        else:
            # No need to set the geometry, just continue.
            continue

        feature.SetGeometry(ogr.CreateGeometryFromWkb(new_geometry.wkb))
        buffered_working_outlets_layer.SetFeature(feature)

    buffered_working_outlets_layer.CommitTransaction()
    buffered_working_outlets_vector.FlushCache()
    del duplicate_points  # don't need this any more.

    cdef int polygon_fid
    disjoint_vector_paths = []
    working_outlets_vector = gdal.OpenEx(working_outlets_path)
    working_outlets_layer = working_outlets_vector.GetLayer()
    outlets_schema = working_outlets_layer.schema

    LOGGER.info('Determining sets of non-overlapping geometries')
    for set_index, disjoint_polygon_fid_set in enumerate(
            pygeoprocessing.calculate_disjoint_polygon_set(
                buffered_working_outlets_path,
                bounding_box=flow_dir_info['bounding_box']),
            start=1):
        # Remove any of the duplicate points.  This set will be empty if
        # the geometries aren't points or if all points are disjoint.
        disjoint_polygon_fid_set -= duplicate_point_fids
        if not disjoint_polygon_fid_set:
            continue

        LOGGER.info("Creating a vector of %s disjoint geometries",
                    len(disjoint_polygon_fid_set))
        disjoint_vector_path = os.path.join(
                working_dir_path, 'disjoint_outflow_%s.gpkg' % set_index)
        disjoint_vector_paths.append(disjoint_vector_path)
        disjoint_vector = gpkg_driver.Create(disjoint_vector_path, 0, 0, 0,
                                             gdal.GDT_Unknown)
        disjoint_layer = disjoint_vector.CreateLayer(
            'outlet_geometries', flow_dir_srs, ogr.wkbPolygon)
        disjoint_layer.CreateFields(outlets_schema)

        disjoint_layer.StartTransaction()
        # Though the buffered working layer was used for determining the sets
        # of nonoverlapping polygons, we want to use the *original* outflow
        # geometries for rasterization.  This step gets the appropriate features
        # from the correct vectors so we keep the correct geometries.
        for polygon_fid in disjoint_polygon_fid_set:
            original_feature = buffered_working_outlets_layer.GetFeature(polygon_fid)
            ws_id = original_feature.GetField(ws_id_fieldname)
            new_feature = working_outlets_layer.GetFeature(
                working_vector_ws_id_to_fid[ws_id]).Clone()
            disjoint_layer.CreateFeature(new_feature)
        disjoint_layer.CommitTransaction()

        disjoint_layer = None
        disjoint_vector = None
    buffered_working_outlets_layer = None
    buffered_working_outlets_vector = None

    target_fragments_vector = gpkg_driver.Create(target_fragments_vector_path,
                                                 0, 0, 0, gdal.GDT_Unknown)
    if target_fragments_vector is None:
        raise RuntimeError(  # Because I frequently have this open in QGIS when I shouldn't.
            "Could not open target fragments vector for writing. Do you have "
            "access to this path?  Is the file open in another program?")

    target_fragments_layer = target_fragments_vector.CreateLayer(
        'watershed_fragments', flow_dir_srs, ogr.wkbPolygon)

    # Create the fields in the target vector that already existed in the
    # outflow points vector
    target_fragments_layer.CreateFields(
            [f for f in working_outlets_layer.schema if f.GetName() != ws_id_fieldname])
    upstream_fragments_field = ogr.FieldDefn('upstream_fragments', ogr.OFTString)
    target_fragments_layer.CreateField(upstream_fragments_field)
    ws_id_field = ogr.FieldDefn('ws_id', ogr.OFTInteger)
    ws_id_field.SetWidth(24)
    target_fragments_layer.CreateField(ws_id_field)
    target_fragments_layer_defn = target_fragments_layer.GetLayerDefn()

    cdef int NO_WATERSHED = 0
    cdef CoordinatePair current_pixel, neighbor_pixel, ws_seed_coord
    cdef queue[CoordinatePair] process_queue
    cdef cset[CoordinatePair] process_queue_set, coords_in_block
    cdef cset[CoordinatePair].iterator coord_iterator
    cdef cset[int] nested_watershed_ids
    cdef cmap[CoordinatePair, int] point_ws_ids 
    cdef cmap[int, cset[CoordinatePair]] points_in_blocks
    cdef cmap[int, cset[CoordinatePair]].iterator block_iterator
    cdef int* neighbor_col = [1, 1, 0, -1, -1, -1, 0, 1]
    cdef int* neighbor_row = [0, -1, -1, -1, 0, 1, 1, 1]
    cdef int* reverse_flow = [4, 5, 6, 7, 0, 1, 2, 3]
    cdef int pixels_in_watershed
    cdef int block_index
    cdef _ManagedRaster flow_dir_managed_raster, scratch_managed_raster, mask_managed_raster
    cdef int watersheds_started = 0
    cdef int n_disjoint_features = 0
    cdef int n_disjoint_features_started = 0
    cdef int pixels_visited = 0
    cdef int neighbor_ws_id
    cdef unsigned char pixel_visited

    # This builds up a spatial index of raster blocks so we can figure out the
    # order in which to start processing points to try to minimize disk
    # accesses.  There will likely be fewer blocks than there will be outflow
    # geometries.
    # 
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

    # Map ws_id to the watersheds that nest within.
    nested_watersheds = {}

    flow_dir_managed_raster = _ManagedRaster(d8_flow_dir_raster_path_band[0],
                                             d8_flow_dir_raster_path_band[1],
                                             0)  # read-only

    for disjoint_index, disjoint_vector_path in enumerate(
                disjoint_vector_paths, start=1):
        # This contextual logger allows us to have extra information in log messages
        # about where we are in the iteration over n disjoint sets.
        disjoint_logger = OverlappingWatershedContextAdapter(
            LOGGER, {'pass_index': disjoint_index,
                     'n_passes': len(disjoint_vector_paths)})

        disjoint_logger.info('Delineating watersheds for disjoint vector #%s',
                             disjoint_index)
        disjoint_logger.info('Creating raster for tracking watershed fragments')
        scratch_raster_path = os.path.join(
                working_dir_path, 'scratch_raster_%s.tif' % disjoint_index)
        pygeoprocessing.new_raster_from_base(
            d8_flow_dir_raster_path_band[0], scratch_raster_path,
            gdal.GDT_UInt32, [NO_WATERSHED], fill_value_list=[NO_WATERSHED],
            gtiff_creation_options=GTIFF_CREATION_OPTIONS)

        pygeoprocessing.rasterize(
            disjoint_vector_path, scratch_raster_path, None,
            ['ALL_TOUCHED=TRUE', 'ATTRIBUTE=%s' % ws_id_fieldname],
            layer_index='outlet_geometries')

        # Create a new watershed scratch mask raster the size, shape of the
        # flow dir raster
        disjoint_logger.info('Creating raster for tracking visited pixels')
        mask_raster_path = os.path.join(working_dir_path,
                                        'scratch_mask_%s.tif' % disjoint_index)
        pygeoprocessing.new_raster_from_base(
            d8_flow_dir_raster_path_band[0], mask_raster_path, gdal.GDT_Byte,
            [255], fill_value_list=[0],
            gtiff_creation_options=GTIFF_CREATION_OPTIONS)

        # Track outflow geometry FIDs against the WS_ID used.
        ws_id_to_disjoint_fid = {}

        disjoint_outlets_vector = gdal.OpenEx(disjoint_vector_path,
                                             gdal.OF_VECTOR)
        disjoint_outlets_layer = disjoint_outlets_vector.GetLayer()
        n_disjoint_features = disjoint_outlets_layer.GetFeatureCount()

        for outflow_feature in disjoint_outlets_layer:
            ws_id = int(outflow_feature.GetField(ws_id_fieldname))
            ws_id_to_disjoint_fid[ws_id] = outflow_feature.GetFID()
            geometry = shapely.wkb.loads(
                outflow_feature.GetGeometryRef().ExportToWkb())
            if not flow_dir_bbox_geometry.contains(geometry):
                geometry = flow_dir_bbox_geometry.intersection(geometry)
                if geometry.is_empty:
                    continue

            try:
                # If a multi-collection
                geometries = geometry.geoms
            except AttributeError:
                geometries = [geometry]

            for geometry in geometries:
                # Since we know that our polygons don't overlap, any point
                # within a geometry should work as a starting point.
                ws_seed_point = geometry.representative_point()
                ws_seed_coord = CoordinatePair(
                    (ws_seed_point.x - x_origin) // x_pixelwidth,
                    (ws_seed_point.y - y_origin) // y_pixelwidth)

                # This is the numeric index of the raster block that intersects
                # with this seed point.
                block_index = next(spatial_index.nearest(
                        (ws_seed_coord.first, ws_seed_coord.second,
                         ws_seed_coord.first, ws_seed_coord.second),
                        num_results=1))

                if (points_in_blocks.find(block_index) ==
                        points_in_blocks.end()):
                    points_in_blocks[block_index] = cset[CoordinatePair]()
                points_in_blocks[block_index].insert(ws_seed_coord)
                point_ws_ids[ws_seed_coord] = ws_id

        disjoint_logger.info('Delineating watersheds')
        scratch_managed_raster = _ManagedRaster(scratch_raster_path, 1, 1)
        mask_managed_raster = _ManagedRaster(mask_raster_path, 1, 1)

        last_log_time = ctime(NULL)
        block_iterator = points_in_blocks.begin()
        n_disjoint_features_started = 0
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
                n_disjoint_features_started += 1
                pixels_in_watershed = 0

                if ctime(NULL) - last_log_time > 5.0:
                    last_log_time = ctime(NULL)
                    disjoint_logger.info(
                        'Delineated %s watersheds of %s (%s of %s total)',
                        n_disjoint_features_started, n_disjoint_features,
                        watersheds_started, n_outlet_features)

                last_ws_log_time = ctime(NULL)
                process_queue.push(current_pixel)
                process_queue_set.insert(current_pixel)
                nested_watershed_ids.clear()  # clear the set for each watershed.

                while not process_queue.empty():
                    pixels_visited += 1
                    pixels_in_watershed += 1
                    if ctime(NULL) - last_ws_log_time > 5.0:
                        last_ws_log_time = ctime(NULL)
                        disjoint_logger.info(
                            'Delineating watershed %i of %i, %i pixels found '
                            'so far.', n_disjoint_features_started,
                            n_disjoint_features, pixels_in_watershed)

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

                        # Is the neighbor off the bounds of the raster?
                        # skip if so.
                        if not 0 <= neighbor_pixel.first < flow_dir_n_cols:
                            continue

                        if not 0 <= neighbor_pixel.second < flow_dir_n_rows:
                            continue

                        # Is the neighbor pixel already in the queue?
                        # Skip if so.
                        if (process_queue_set.find(neighbor_pixel) !=
                                process_queue_set.end()):
                            continue

                        # Is the neighbor an unvisited lake pixel in this
                        # watershed?  If yes, enqueue it.
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

                            # Does the neighbor belong to a different outflow
                            # geometry?
                            if (neighbor_ws_id != NO_WATERSHED and
                                    neighbor_ws_id != ws_id):
                                # If it is, track the watershed connectivity,
                                # but otherwise skip this pixel.  Either we've
                                # already processed it, or else we will soon!
                                nested_watershed_ids.insert(neighbor_ws_id)
                                continue

                            # If the pixel has not yet been visited, enqueue it.
                            if pixel_visited == 0:
                                process_queue.push(neighbor_pixel)
                                process_queue_set.insert(neighbor_pixel)

                nested_watersheds[ws_id] = set(nested_watershed_ids)
        disjoint_logger.info(
            'Delineated %s watersheds of %s (%s of %s total)',
            n_disjoint_features_started, n_disjoint_features,
            watersheds_started, n_outlet_features)

        points_in_blocks.clear()
        coords_in_block.clear()

        scratch_managed_raster.close()  # flush the scratch raster.
        mask_managed_raster.close()  # flush the mask raster

        disjoint_logger.info('Polygonizing fragments')
        scratch_raster = gdal.OpenEx(scratch_raster_path, gdal.OF_RASTER)
        scratch_band = scratch_raster.GetRasterBand(1)
        mask_raster = gdal.OpenEx(mask_raster_path, gdal.OF_RASTER)
        mask_band = mask_raster.GetRasterBand(1)

        watershed_fragments_path = os.path.join(
            working_dir_path, 'watershed_fragments_%s.gpkg' % disjoint_index)
        watershed_fragments_vector = gpkg_driver.Create(
            watershed_fragments_path, 0, 0, 0, gdal.GDT_Unknown)
        watershed_fragments_layer = watershed_fragments_vector.CreateLayer(
            'watershed_fragments', flow_dir_srs, ogr.wkbPolygon)
        watershed_fragments_layer.CreateField(ws_id_field)

        gdal.Polygonize(
            scratch_band,  # the source band to be analyzed
            mask_band,  # the mask band indicating valid pixels
            watershed_fragments_layer,  # polygons are added to this layer
            0,  # field index into which to save the pixel value of watershed
            ['8CONNECTED8'],  # use 8-connectedness algorithm.
            _make_polygonize_callback(disjoint_logger)
        )

        # Iterate through all of the features we just created in this disjoint
        # fragments vector and copy them over into the target fragments vector.
        disjoint_logger.info('Copying fragments to the output vector.')
        target_fragments_layer.StartTransaction()
        for created_fragment in watershed_fragments_layer:
            fragment_ws_id = created_fragment.GetField('ws_id')
            outflow_geom_fid = ws_id_to_disjoint_fid[fragment_ws_id]

            target_fragment_feature = ogr.Feature(target_fragments_layer_defn)
            target_fragment_feature.SetGeometry(
                created_fragment.GetGeometryRef())

            outflow_point_feature = disjoint_outlets_layer.GetFeature(
                outflow_geom_fid)
            for outflow_fieldname, outflow_fieldvalue in (
                    outflow_point_feature.items().items()):
                if outflow_fieldname == ws_id_fieldname:
                    continue
                target_fragment_feature.SetField(outflow_fieldname,
                                                 outflow_fieldvalue)

            target_fragment_feature.SetField('ws_id', float(fragment_ws_id))
            try:
                upstream_fragments = ','.join(
                    [str(s) for s in sorted(nested_watersheds[fragment_ws_id])])
            except KeyError:
                upstream_fragments = ''
            target_fragment_feature.SetField('upstream_fragments',
                                             upstream_fragments)

            target_fragments_layer.CreateFeature(target_fragment_feature)

            # If there were any duplicate points, copy the geometry
            if fragment_ws_id in duplicate_point_ws_ids:
                for duplicate_ws_id in duplicate_point_ws_ids[fragment_ws_id]:
                    duplicate_feature = ogr.Feature(target_fragments_layer_defn)
                    duplicate_feature.SetGeometry(
                        created_fragment.GetGeometryRef())
                    duplicate_feature.SetField('upstream_fragments',
                                               upstream_fragments)
                    source_feature = working_outlets_layer.GetFeature(
                        working_vector_ws_id_to_fid[duplicate_ws_id])

                    for fieldname, fieldvalue in source_feature.items().items():
                        if fieldname == ws_id_fieldname:
                            continue
                        duplicate_feature.SetField(fieldname, fieldvalue)

                    target_fragments_layer.CreateFeature(duplicate_feature)
        target_fragments_layer.CommitTransaction()

        # Close the watershed fragments vector from this disjoint geometry set.
        disjoint_outlets_layer = None
        disjoint_outlets_vector = None
        watershed_fragments_layer = None
        watershed_fragments_vector = None

    flow_dir_managed_raster.close()
    scratch_band = None
    scratch_raster = None
    mask_band = None
    mask_raster = None
    working_outlets_layer = None
    working_outlets_vector = None

    shutil.rmtree(working_dir_path, ignore_errors=True)


def join_watershed_fragments(watershed_fragments_vector,
                             target_watersheds_vector):
    """Join watershed fragments by their WS IDs.

    This function takes a watershed fragments vector and creates a new vector
    where all geometries represent the full watershed represented by any nested
    watershed fragments contained within the watershed fragments vector.

    Parameters:
        watershed_fragments_vector (string): A path to a vector on disk.  This
            vector must have at least two fields: 'ws_id' (int) and
            'upstream_fragments' (string).  The 'upstream_fragments' field's
            values must be formatted as a list of comma-separated integers
            (example: ``1,2,3,4``), where each integer matches the ``ws_id``
            field of a polygon in this vector.  This field should only contain
            the ``ws_id``s of watershed fragments that directly touch the
            current fragment, as this function will recurse through the
            fragments to determine the correct nested geometry.  If a fragment
            has no nested watersheds, this field's value will have an empty
            string.  The ``ws_id`` field represents a unique integer ID for the
            watershed fragment.
        target_watersheds_vector (string): A path to a vector on disk.  This
            vector will be written as a GeoPackage.

    Returns:
        ``None``

    """
    fragments_vector = gdal.OpenEx(watershed_fragments_vector, gdal.OF_VECTOR)
    fragments_layer = fragments_vector.GetLayer()
    fragments_srs = fragments_layer.GetSpatialRef()

    driver = gdal.GetDriverByName('GPKG')
    watersheds_vector = driver.Create(target_watersheds_vector, 0, 0, 0,
                                      gdal.GDT_Unknown)
    watersheds_layer = watersheds_vector.CreateLayer(
        'watersheds', fragments_srs, ogr.wkbPolygon)
    watersheds_layer_defn = watersheds_layer.GetLayerDefn()

    for field_defn in fragments_layer.schema:
        field_type = field_defn.GetType()
        if field_type in (ogr.OFTInteger, ogr.OFTReal):
            field_defn.SetWidth(24)
        watersheds_layer.CreateField(field_defn)

    upstream_fragments = {}
    fragment_geometries = {}
    fragment_field_values = {}
    LOGGER.info('Loading fragment geometries.')
    for feature in fragments_layer:
        ws_id = feature.GetField('ws_id')
        fragment_field_values[ws_id] = feature.items()
        upstream_fragments_string = feature.GetField('upstream_fragments')
        if upstream_fragments_string:
            upstream_fragments[ws_id] = [int(f) for f in
                                         upstream_fragments_string.split(',')]
        else:
            # If no upstream fragments, the string will be '', and
            # ''.split(',') turns into [''], which crashes when you cast it to
            # an int.
            upstream_fragments[ws_id] = []
        shapely_polygon = shapely.wkb.loads(
            feature.GetGeometryRef().ExportToWkb())
        try:
            fragment_geometries[ws_id].append(shapely_polygon)
        except KeyError:
            fragment_geometries[ws_id] = [shapely_polygon]

    # Create multipolygons from each of the lists of fragment geometries.
    for ws_id in fragment_geometries:
        fragment_geometries[ws_id] = shapely.geometry.MultiPolygon(fragment_geometries[ws_id])

    # Populate the watershed geometries dict with fragments that are as
    # upstream as you can go.
    watershed_geometries = dict(
        (ws_id, fragment_geometries[ws_id]) for (ws_id, upstream_fragments_list)
        in upstream_fragments.items() if len(upstream_fragments_list) == 0)
    LOGGER.info('%s watersheds have no upstream fragments and can be created directly.',
        len(watershed_geometries))

    encountered_ws_ids = set([])
    def _recurse_watersheds(ws_id):
        """Find or build geometries for the given ``ws_id``.

        This is a dynamic programming, recursive approach to determining
        watershed geometries.  If a geometry already exists, we use that.
        If it doesn't, we create the appropriate geometries as needed by
        taking the union of that fragment's upstream fragments with the
        geometry of the fragment itself.

        This function has the side effect of modifying the
        ``watershed_geometries`` dict.

        Parameters:
            ws_id (int): The integer ws_id that identifies the watershed
                geometry to build.

        Returns:
            shapely.geometry.Polygon object of the union of the watershed
                fragment identified by ws_id as well as all watersheds upstream
                of this fragment.

        """

        try:
            geometries = [fragment_geometries[ws_id]]
            encountered_ws_ids.add(ws_id)
        except KeyError:
            LOGGER.warn('Upstream watershed fragment %s not found. '
                        'Do you have overlapping geometries?', ws_id)
            return [shapely.geometry.Polygon([])]

        cdef int upstream_fragment_id
        for upstream_fragment_id in upstream_fragments[ws_id]:
            # If we've already encountered this upstream fragment on this
            # run through the recursion, skip it.
            if upstream_fragment_id in encountered_ws_ids:
                continue

            encountered_ws_ids.add(upstream_fragment_id)
            if upstream_fragment_id not in watershed_geometries:
                watershed_geometries[upstream_fragment_id] = (
                    shapely.ops.cascaded_union(
                        _recurse_watersheds(upstream_fragment_id)))
            geometries.append(watershed_geometries[upstream_fragment_id])
        return geometries

    # Iterate over the ws_ids that have upstream geometries and create the
    # watershed geometries.
    # This iteration must happen in sorted order because real-world watersheds
    # sometimes have thousands of nested watersheds, which will exhaust
    # python's max recursion depth here if we're not careful.
    last_log_time = time.time()
    LOGGER.info('%s geometries have upstream fragments.',
                len(upstream_fragments) - len(watershed_geometries))
    n_watersheds_processed = 0
    n_watersheds_total = len(upstream_fragments) - len(watershed_geometries)
    for ws_id in sorted(
            set(upstream_fragments.keys()).difference(
                set(watershed_geometries.keys())),
            key=lambda ws_id_key: len(upstream_fragments[ws_id_key])):
        current_time = time.time()
        if current_time - last_log_time >= 5.0:
            LOGGER.info("Joined %s of %s watersheds so far (%.1f%%)",
                n_watersheds_processed, n_watersheds_total,
                n_watersheds_processed / float(n_watersheds_total) * 100)
            last_log_time = current_time

        # The presence of a ws_id key in watershed_geometries could be
        # altered during a call to _recurse_watersheds.  This condition
        # ensures that we don't call _recurse_watersheds more than we need to.
        encountered_ws_ids.clear()
        watershed_geometries[ws_id] = shapely.ops.cascaded_union(
            _recurse_watersheds(ws_id))
        n_watersheds_processed +=1 

    # Copy fields from the fragments vector and set the geometries to the
    # newly-created, unioned geometries.
    LOGGER.info('Copying field values to the target vector')
    watersheds_layer.StartTransaction()
    for ws_id, watershed_geometry in sorted(watershed_geometries.items(),
                                            key=lambda x: x[0]):
        # Creating the feature here works properly, unlike
        # fragments_feature.Clone(), which raised SQL errors.
        watershed_feature = ogr.Feature(watersheds_layer_defn)
        try:
            for field_name, field_value in fragment_field_values[ws_id].items():
                watershed_feature.SetField(field_name, field_value)
        except KeyError:
            # KeyError happens when an upstream geometry was expected but
            # wasn't found in the fragments vector.  This typically happens
            # because of a sequence of events:
            #
            #  1. Two nonoverlapping features (with area < 1 pixel) overlap the
            #     same pixel.
            #  2. Because these two features are disjoint, disjoint_polygon_set
            #     includes both of these features ina single disjoint set.
            #  3. The two polygons are rasterized, and there's a race condition
            #     where we don't know which feature will be rasterized on top
            #     of the other.
            #  4. In the delineation of nearby watershed fragments, some
            #     watersheds are determined to be downstream of whichever
            #     feature was rasterized in step 3.
            #  5. At some point, the watershed delineation processes the second
            #     feature that overlaps the pixel, setting that pixel to the new
            #     ws_id, erasing any record of the first feature's ws_id in
            #     the process.
            #  6. Eventually, the fragments vector is polygonized from the
            #     resulting raster.
            #  7. When we come here to join watershed fragments, some fragments
            #     delineated in step 4 look for their upstream fragments, but
            #     because this small fragment was overwritten in step 5
            #     the ws_id was never included in the fragments vector and we
            #     get a KeyError.
            #
            # The solution here is to ensure that very small features are
            # handled properly when determining sets of disjoint polygons.
            #
            # If this warning is ever encountered, send your inputs to
            # jdouglass@stanford.edu so we can fix it!
            LOGGER.warn(
                "Skipping ws_id %s, which shouldn't happen. This means "
                "that at least one watershed was not delineated correctly.",
                ws_id)

        watershed_feature.SetGeometry(ogr.CreateGeometryFromWkb(
            shapely.wkb.dumps(watershed_geometry)))
        watersheds_layer.CreateFeature(watershed_feature)
    watersheds_layer.CommitTransaction()

    fragments_srs = None
    fragments_layer = None
    fragments_vector = None

    watersheds_layer = None
    watersheds_vector = None


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
