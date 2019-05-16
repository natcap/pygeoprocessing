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
import shapely.geometry
import shapely.prepared

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
from libcpp.stack cimport stack as cstack
from libc.math cimport fmod, ceil, floor, fabs

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

def delineate_watersheds_d8_old(
        d8_flow_dir_raster_path_band, outflow_vector_path,
        target_fragments_vector_path, working_dir=None, starting_ws_id=None):
    """Delineate watersheds from a D8 flow direction raster.

    This function produces a vector of watershed fragments, where each fragment
    represents the area that flows into each outflow point.  Nested watersheds
    are represented by the field

    Parameters:
        d8_flow_dir_raster_path_band (tuple): A two-tuple representing the
            string path to the D8 flow direction raster to use and the band
            index to use.
        outflow_vector_path (string): Path to a vector on disk representing
            geometries for features of interest.  This vector must have one
            layer and must be in the same projection as the flow direction
            raster.
        target_fragments_vector_path (string): Path to where the watershed
            fragments vector will be stored on disk.  This filepath must end
            with the 'gpkg' extension, and will be created as a GeoPackage.
        working_dir=None (string or None): The path to a directory on disk
            where intermediate files will be stored.  This directory will be
            created if it does not exist, and intermediate files created will
            be removed.  If ``None``, a new temporary folder will be created
            within the system temp directory.
        starting_ws_id=None (int or None): A numeric identifier to start
            counting outflow geometries from.  Useful when splitting
            delineation across multiple runs and having consistent watershed
            indexes.  If provided, must be a positive, nonzero integer. If
            None, the ``ws_id`` field will begin indexing from 1.

    Returns:
        ``None``

    """
    if (d8_flow_dir_raster_path_band is not None and not
            _is_raster_path_band_formatted(d8_flow_dir_raster_path_band)):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                d8_flow_dir_raster_path_band))

    cdef int ws_id  # start indexing ws_id at 1
    if starting_ws_id is None:
        ws_id = 1
    else:
        if not isinstance(starting_ws_id, int) or starting_ws_id <= 0:
            raise ValueError(
                'starting_ws_id must be a positive, nonzero integer; %s found'
                % starting_ws_id)
        else:
            ws_id = starting_ws_id

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

    source_outlets_vector = gdal.OpenEx(outflow_vector_path, gdal.OF_VECTOR)
    if source_outlets_vector is None:
        raise ValueError(u'Could not open outflow vector %s' % outflow_vector_path)

    working_outlets_path = os.path.join(working_dir_path, 'working_outlets.gpkg')
    LOGGER.info('Copying outlets to %s', working_outlets_path)
    working_outlets_vector = gpkg_driver.CreateCopy(working_outlets_path,
                                                    source_outlets_vector)
    working_outlets_layer = working_outlets_vector.GetLayer()

    # Add a new field to the clipped_outlets_layer to ensure we know what field
    # values are bing rasterized.
    ws_id_fieldname = '__ws_id__'
    ws_id_field_defn = ogr.FieldDefn(ws_id_fieldname, ogr.OFTInteger64)
    working_outlets_layer.CreateField(ws_id_field_defn)
    outlets_schema = working_outlets_layer.schema

    # Phase 0: preprocess working geometries.
    #    * Attempt to repair invalid geometries.
    #    * Create new WS_ID field, set field value.
    #    * Exclude any geometries that do not intersect the DEM bbox (prepared geometry op)
    working_vector_ws_id_to_fid = {}
    working_outlets_layer.StartTransaction()
    cdef int n_features = working_outlets_layer.GetFeatureCount()
    cdef int n_complete = 0
    last_log_time = ctime(NULL)
    LOGGER.info('Preprocessing outflow geometries.')
    for feature in working_outlets_layer:
        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            LOGGER.info('%s of %s features complete (%.3f %%)',
                n_complete, n_features, (float(n_complete)/n_features)*100.0)
        ogr_geom = feature.GetGeometryRef()
        fid = feature.GetFID()
        try:
            geometry = shapely.wkb.loads(ogr_geom.ExportToWkb())
        except ReadingError:
            # This happens when a polygon isn't a closed ring.
            ogr_geom.CloseRings()
            ogr_geom.Buffer(0)
            # If the geometry isn't 'corrected', an exception should be raised
            # here and the function will crash.
            try:
                geometry = shapely.wkb.loads(ogr_geom.ExportToWkb())
                feature.SetGeometry(ogr.CreateGeometryFromWkb(geometry.wkb))
            except ReadingError:
                raise ValueError(
                    'Feature %s has invalid geometry that could not be '
                    'corrected.' % fid)

        n_complete += 1  # always incremement this counter.

        working_vector_ws_id_to_fid[ws_id] = fid
        feature.SetField(ws_id_fieldname, ws_id)
        working_outlets_layer.SetFeature(feature)
        ws_id += 1
    working_outlets_layer.CommitTransaction()
    LOGGER.info('Preprocessing complete')

    # Phase 1: Prepare working geometries for determining sets of disjoint polygons.
    #    * Points and very small geometries can be removed, as we already know their seed coords.
    #         * Track these in a {CoordinatePair: set of ws_ids} structure
    #    * Alter geometries of remaining polygons and lines as needed
    cdef cmap[CoordinatePair, cset[int]] seed_watersheds
    cdef cmap[CoordinatePair, int] seed_ids  # {CoordinatePair: seed id}
    cdef cmap[int, CoordinatePair] seed_id_to_seed  # {seed id: CoordinatePair}
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
            seed_id_to_seed[seed_id] = seed
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
    cdef int row, col, neighbor_x, neighbor_y
    cdef int* neighbor_col = [1, 1, 0, -1, -1, -1, 0, 1]
    cdef int* neighbor_row = [0, -1, -1, -1, 0, 1, 1, 1]
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

            mr = _ManagedRaster(tmp_seed_raster_path, 1, 0)
            for block_info, block in pygeoprocessing.iterblocks(tmp_seed_raster_path):
                for (row, col) in zip(*numpy.nonzero(block)):
                    ws_id = block[row, col]
                    seed = CoordinatePair(col + block_info['xoff'],
                                          row + block_info['yoff'])

                    is_edge = False
                    for neighbor_index in range(8):
                        neighbor_x = seed.first + neighbor_col[neighbor_index]
                        neighbor_y = seed.second + neighbor_row[neighbor_index]

                        # Is the neighbor off the bounds of the raster?
                        # skip if so.
                        if not 0 <= neighbor_x < flow_dir_n_cols:
                            continue

                        if not 0 <= neighbor_y < flow_dir_n_rows:
                            continue

                        if mr.get(neighbor_x, neighbor_y) != ws_id:
                            is_edge = True
                            break

                    # Insert the seed into the seed_watersheds structure
                    # if the seed is on a boundary of the geometry.
                    if is_edge:
                        if (seed_watersheds.find(seed) == seed_watersheds.end()):
                            seed_watersheds[seed] = cset[int]()
                        seed_watersheds[seed].insert(ws_id)
                        seed_ids[seed] = seed_id
                        seed_id_to_seed[seed_id] = seed
                        seed_id += 1
            mr.close()
            del block
            del block_info

        buffered_working_outlets_layer = None
    buffered_working_outlets_vector = None

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
            (seed.first // flow_dir_block_x_size) +
            ((seed.second // flow_dir_block_y_size) * (flow_dir_n_cols // flow_dir_block_x_size)))
        if block_index > n_blocks:
            print 'block_index %s > %s' % (block_index, n_blocks)
        seeds_in_block[block_index].insert(seed)

    # Phase 5: Iteration.
    #    * For each blockwise seed coordinate cluster:
    #        * Process neighbors.
    #        * If a neighbor is upstream and is a known seed, mark connectivity between seeds.
    cdef cmap[int, cset[int]] upstream_seed_ids  # {int seed_id: int upstream seed_id}
    cdef cset[int] nested_fragment_ids
    cdef int* reverse_flow = [4, 5, 6, 7, 0, 1, 2, 3]
    cdef cset[CoordinatePair] process_queue_set
    cdef queue[CoordinatePair] process_queue

    scratch_raster_path = os.path.join(
            working_dir_path, 'scratch_raster.tif')
    LOGGER.info('Creating new scratch raster at %s' % scratch_raster_path)
    pygeoprocessing.new_raster_from_base(
        d8_flow_dir_raster_path_band[0], scratch_raster_path,
        gdal.GDT_UInt32, [NO_WATERSHED], fill_value_list=[NO_WATERSHED],
        gtiff_creation_options=GTIFF_CREATION_OPTIONS)
    scratch_managed_raster = _ManagedRaster(scratch_raster_path, 1, 1)

    mask_raster_path = os.path.join(working_dir_path, 'mask_raster.tif')
    LOGGER.info('Creating new mask raster at %s' % scratch_raster_path)
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

    # Phase 6.1: Conditional graph traversal and ID consolidation
    #    * This phase walks the fragment graph and identifies fragments that can be
    #      consolidated via reclassification.  The goal of this is to maintain
    #      the minimal set of fragments that will join to the correct output.
    #      This is needed because:
    #        * Polygonization is faster with fewer fragments
    #        * Joining via cascaded_union/union is surprisingly slow, and we can avoid
    #          many union operations by being careful about which fragments need to
    #          remain separate.

    # Reverse the upstream_fragments dictionary.
    LOGGER.info('Building network of downstream fragments')
    cdef cmap[int, int] downstream_fragments  # {fragment_id: int downstream_fragment_id}
    cdef cmap[int, cset[int]].iterator nested_fragments_iterator = nested_fragments.begin()
    cdef cset[int] upstream_fragments
    cdef cset[int].iterator upstream_fragments_iterator
    while nested_fragments_iterator != nested_fragments.end():
        downstream_fragment_id = deref(nested_fragments_iterator).first
        upstream_fragments = deref(nested_fragments_iterator).second

        upstream_fragments_iterator = upstream_fragments.begin()
        while upstream_fragments_iterator != upstream_fragments.end():
            upstream_fragment_id = deref(upstream_fragments_iterator)
            downstream_fragments[upstream_fragment_id] = downstream_fragment_id
            inc(upstream_fragments_iterator)
        inc(nested_fragments_iterator)

    # Locate the seeds that are as far down the seed tree as we can go.
    LOGGER.info('Determining downstream-most fragments')
    cdef queue[CoordinatePair] starter_seeds
    cdef cset[CoordinatePair] starter_seeds_set
    seeds_in_watersheds_iterator = seed_watersheds.begin()  # reusing previous iterator
    while seeds_in_watersheds_iterator != seed_watersheds.end():
        # Only need the seed; don't need the member ws_ids.
        seed = deref(seeds_in_watersheds_iterator).first
        inc(seeds_in_watersheds_iterator)

        seed_id = seed_ids[seed]
        while True:
            if downstream_fragments.find(seed_id) != downstream_fragments.end():
                seed_id = downstream_fragments[seed_id]
            else:
                break
        seed = seed_id_to_seed[seed_id]
        starter_seeds.push(seed)
        starter_seeds_set.insert(seed)

    cdef cmap[int, int] reclassification

    LOGGER.info('Determining active fragments')
    cdef cstack[CoordinatePair] stack
    cdef cset[int] member_watersheds
    cdef cmap[CoordinatePair, int] effective_seed_ids
    cdef cmap[CoordinatePair, cset[int]] effective_watersheds
    cdef cmap[CoordinatePair, cset[int]] downstream_watersheds
    cdef cset[int] watersheds_downstream_of_neighbor
    cdef int starter_id
    cdef CoordinatePair starter_seed, current_seed, upstream_seed, neighbor_seed
    cdef cset[CoordinatePair] visited
    cdef int upstream_seed_id
    while starter_seeds.size() > 0:
        starter_seed = starter_seeds.front()
        starter_seeds_set.erase(starter_seed)
        starter_seeds.pop()

        member_watersheds = seed_watersheds[starter_seed]
        if effective_seed_ids.find(starter_seed) != effective_seed_ids.end():
            starter_id = effective_seed_ids[starter_seed]
        else:
            starter_id = seed_ids[starter_seed]
            effective_seed_ids[starter_seed] = starter_id

        stack.push(starter_seed)
        while stack.size() > 0:
            current_seed = stack.top()
            stack.pop()

            current_seed_id = seed_ids[current_seed]

            reclassification[seed_ids[current_seed]] = starter_id
            visited.insert(current_seed)

            # First, check to see if there are fragments upstream of this one.
            if nested_fragments.find(current_seed_id) != nested_fragments.end():
                upstream_fragments = nested_fragments[current_seed_id]
                upstream_fragments_iterator = upstream_fragments.begin()

                while upstream_fragments_iterator != upstream_fragments.end():
                    upstream_seed_id = deref(upstream_fragments_iterator)
                    inc(upstream_fragments_iterator)

                    upstream_seed = seed_id_to_seed[upstream_seed_id]

                    # check to see if the watershed membership of the upstream seed is a
                    # subset of the watersheds of the starter seed.
                    upstream_watersheds = seed_watersheds[upstream_seed]
                    upstream_watersheds_iterator = upstream_watersheds.begin()
                    is_subset = True
                    while upstream_watersheds_iterator != upstream_watersheds.end():
                        ws_id = deref(upstream_watersheds_iterator)
                        inc(upstream_watersheds_iterator)

                        if member_watersheds.find(ws_id) == member_watersheds.end():
                            is_subset = False
                            break

                    if is_subset:
                        stack.push(upstream_seed)
                        effective_watersheds[upstream_seed] = member_watersheds
                    else:
                        # The upstream seed appears to be the start of a different fragment.
                        # Add it to the starter seeds queue.
                        effective_watersheds[upstream_seed] = upstream_watersheds
                        if starter_seeds_set.find(upstream_seed) == starter_seeds_set.end():
                            starter_seeds_set.insert(upstream_seed)
                            starter_seeds.push(upstream_seed)

                            # Noting which watersheds are downstream of the upstream
                            # pixel is important for visiting neighbors and expanding
                            # into them, below.
                            downstream_watersheds[upstream_seed] = member_watersheds

            # Second, visit the neighbors of this seed and see if there are any
            # that match.
            for neighbor_id in xrange(8):
                neighbor_seed = CoordinatePair(
                    current_seed.first + neighbor_col[neighbor_id],
                    current_seed.second + neighbor_row[neighbor_id])

                if not 0 <= neighbor_seed.first < flow_dir_n_rows:
                    continue
                if not 0 <= neighbor_seed.second < flow_dir_n_cols:
                    continue

                # Is this pixel a seed?  If it isn't, we can ignore it.
                if seed_watersheds.find(neighbor_seed) == seed_watersheds.end():
                    continue

                neighbor_seed_id = seed_ids[neighbor_seed]
                neighbor_watersheds = seed_watersheds[neighbor_seed]

                # Do the member watersheds of this neighbor seed exactly
                # match the member watersheds of the starter seed?
                if seed_watersheds[neighbor_seed] == member_watersheds:
                    if (downstream_watersheds.find(current_seed) != downstream_watersheds.end() and
                            downstream_watersheds.find(neighbor_seed) != downstream_watersheds.end()):
                        if downstream_watersheds[current_seed] == downstream_watersheds[neighbor_seed]:
                            effective_seed_ids[neighbor_seed] = starter_id
                    else:
                        if downstream_fragments.find(neighbor_seed_id) != downstream_fragments.end():
                            downstream_seed_id = downstream_fragments[neighbor_seed_id]
                            downstream_seed = seed_id_to_seed[downstream_seed_id]

                            # test to see if the watersheds of the downstream seed are a
                            # subset of the watersheds of the neighbor seed.
                            watersheds_downstream_of_neighbor = seed_watersheds[downstream_seed]
                            downstream_watersheds_iterator = watersheds_downstream_of_neighbor.begin()
                            is_subset = True
                            while downstream_watersheds_iterator != watersheds_downstream_of_neighbor.end():
                                ws_id = deref(downstream_watersheds_iterator)
                                inc(downstream_watersheds_iterator)

                                if neighbor_watersheds.find(ws_id) == neighbor_watersheds.end():
                                    is_subset = False
                                    break

                            if is_subset:
                                effective_seed_ids[neighbor_seed] = starter_id
                        else:
                            # No known downstream seeds to check, ok to expand into the seed.
                            effective_seed_ids[neighbor_seed] = starter_id

                    if visited.find(neighbor_seed) == visited.end():
                        stack.push(neighbor_seed)

    # Now that we've determined the reclassification, we need to rewrite a few things:
    #  * The fragment ids may need to be remapped
    #  * The nested fragment IDs will need to be recreated for the new subset.
    LOGGER.info('Remapping graph nodes: nested fragments')
    cdef cmap[int, cset[int]] nested_fragments_consolidated
    cdef cmap[int, cset[int]].iterator nf_iterator = nested_fragments.begin()
    cdef cset[int].iterator nested_fragment_id_iterator
    cdef int fragment_id, fragment_id_consolidated
    while nf_iterator != nested_fragments.end():
        fragment_id = deref(nf_iterator).first
        nested_fragment_ids = deref(nf_iterator).second
        inc(nf_iterator)
        fragment_id_consolidated = reclassification[fragment_id]

        # If the fragment has not yet been initialized, create the new set.
        if nested_fragments_consolidated.find(fragment_id_consolidated) == nested_fragments_consolidated.end():
            nested_fragments_consolidated[fragment_id_consolidated] = cset[int]()

        nested_fragment_id_iterator = nested_fragment_ids.begin()
        while nested_fragment_id_iterator != nested_fragment_ids.end():
            nested_fragment_id = deref(nested_fragment_id_iterator)
            inc(nested_fragment_id_iterator)
            nested_fragments_consolidated[fragment_id_consolidated].insert(nested_fragment_id)

    LOGGER.info("Consolidating fragment ids")
    reclassified_scratch_path = os.path.join(working_dir_path, 'scratch_reclassified.tif')
    pygeoprocessing.reclassify_raster(
        (scratch_raster_path, 1), dict(reclassification),
        reclassified_scratch_path, gdal.GDT_UInt32, 0,
        values_required=False)  # setting to False to eliminate expensive check.

    # Phase 6.2: Polygonization
    #    * Polygonize the seeds fragments.
    LOGGER.info('Polygonizing fragments')
    scratch_raster = gdal.OpenEx(reclassified_scratch_path, gdal.OF_RASTER)
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
    target_fragments_scratch_layer = target_fragments_vector.CreateLayer(
        'watershed_fragments_scratch', flow_dir_srs, ogr.wkbMultiPolygon)
    target_fragments_scratch_layer.CreateField(ogr.FieldDefn('fragment_id', ogr.OFTInteger64))

    gdal.Polygonize(
        scratch_band,  # the source band to be analyzed
        mask_band,  # the mask band indicating valid pixels
        target_fragments_scratch_layer,  # polygons are added to this layer
        0,  # field index of 'fragment_id' field.
        ['8CONNECTED=8'],  # use 8-connectedness algorithm.
        _make_polygonize_callback(LOGGER)
    )
    target_fragments_layer = target_fragments_vector.CreateLayer(
        'watershed_fragments', flow_dir_srs, ogr.wkbMultiPolygon)

    # Create the fields in the target vector that already existed in the
    # outflow points vector
    target_fragments_layer.CreateField(ogr.FieldDefn('fragment_id', ogr.OFTInteger64))
    target_fragments_layer.CreateField(ogr.FieldDefn('upstream_fragments', ogr.OFTString))
    target_fragments_layer.CreateField(ogr.FieldDefn('member_ws_ids', ogr.OFTString))

    # The Polygonization algorithm will sometimes identify regions that
    # should be contiguous in a single polygon, but are not.  For this reason,
    # we need an extra consolidation step here to make sure that we only produce
    # 1 feature per fragment ID.
    counts = {}   # map {fragment_id: Feature IDs with this fragment_id}
    fragments_with_duplicates = set([])
    for feature in target_fragments_scratch_layer:
        fid = feature.GetFID()
        fragment_id = feature.GetFieldAsInteger('fragment_id')
        if fragment_id in counts:
            counts[fragment_id].add(fid)
            fragments_with_duplicates.add(fragment_id)
        else:
            counts[fragment_id] = set([fid])
    target_fragments_scratch_layer.ResetReading()

    LOGGER.info('Consolidating %s fragments with identical IDs',
        len(fragments_with_duplicates))

    fids_to_be_skipped = set([])
    cdef CoordinatePair fragment_seed
    target_fragments_layer.StartTransaction()
    for fragment_id_with_duplicates in fragments_with_duplicates:
        new_geometry = ogr.Geometry(ogr.wkbMultiPolygon)
        for duplicate_fid in counts[fragment_id_with_duplicates]:
            duplicate_feature = target_fragments_scratch_layer.GetFeature(duplicate_fid)
            new_geometry.AddGeometry(duplicate_feature.GetGeometryRef())
            fids_to_be_skipped.add(duplicate_fid)

        consolidated_feature = ogr.Feature(target_fragments_layer.GetLayerDefn())
        consolidated_feature.SetField('fragment_id', fragment_id)
        fragment_seed = seed_id_to_seed[fragment_id]
        consolidated_feature.SetField('upstream_fragments',
            ','.join([str(s) for s in sorted(nested_fragments_consolidated[fragment_id])]))
        consolidated_feature.SetField('member_ws_ids',
            ','.join([str(s) for s in sorted(set(seed_watersheds[fragment_seed]))]))
        consolidated_feature.SetGeometry(new_geometry.UnionCascaded())
        target_fragments_layer.CreateFeature(consolidated_feature)
    target_fragments_layer.CommitTransaction()

    LOGGER.info('Copying remaining fragments')
    target_fragments_scratch_layer.ResetReading()
    target_fragments_layer.StartTransaction()
    for feature in target_fragments_scratch_layer:
        if feature.GetFID() in fids_to_be_skipped:
            continue

        fragment_id = feature.GetField('fragment_id')
        consolidated_feature = ogr.Feature(target_fragments_layer.GetLayerDefn())
        consolidated_feature.SetField('fragment_id', fragment_id)
        fragment_seed = seed_id_to_seed[fragment_id]
        consolidated_feature.SetField('upstream_fragments',
            ','.join([str(s) for s in sorted(nested_fragments_consolidated[fragment_id])]))
        consolidated_feature.SetField('member_ws_ids',
            ','.join([str(s) for s in sorted(set(seed_watersheds[fragment_seed]))]))
        new_geometry = ogr.Geometry(ogr.wkbMultiPolygon)
        new_geometry.AddGeometry(feature.GetGeometryRef())
        consolidated_feature.SetGeometry(new_geometry)
        target_fragments_layer.CreateFeature(consolidated_feature)
    target_fragments_layer.CommitTransaction()

    # Close the watershed fragments vector from this disjoint geometry set.
    disjoint_outlets_layer = None
    disjoint_outlets_vector = None
    target_fragments_layer = None
    target_fragments_vector = None

OGR_ERRORS = dict((getattr(ogr, a), a) for a in dir(ogr) if a.startswith('OGRERR'))

def _add_geometries_to_multipolygon(multipolygon, new_geometry):
    if new_geometry.GetGeometryCount() == 0:
        e = multipolygon.AddGeometry(new_geometry)
        if e != 0:
            LOGGER.warn('Error %s in AddGeometry 0', OGR_ERRORS[e])
    else:
        for sub_geometry in new_geometry:
            e = multipolygon.AddGeometry(sub_geometry)
            if e != 0:
                LOGGER.warn('Error %s in AddGeometry n', OGR_ERRORS[e])


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

    working_fragments_layer = watersheds_vector.CreateLayer(
        'working_fragments', fragments_srs, ogr.wkbMultiPolygon)

    for field_defn in outflow_attributes_layer.schema:
        field_type = field_defn.GetType()
        if field_type in (ogr.OFTInteger, ogr.OFTReal):
            field_defn.SetWidth(24)
        watersheds_layer.CreateField(field_defn)

    # Phase 1: Load in all of the fragments
    LOGGER.info('Loading fragments')
    upstream_fragments = collections.defaultdict(list)
    n_fragments = fragments_layer.GetFeatureCount()
    fragment_ids = set([])
    original_fragment_fids = {}  # map fragment ids to original feature IDs (could be multiple FIDs)
    fragment_fids = {}  # map compiled fragment FIDs to their feature IDs.
    fragments_in_watershed = collections.defaultdict(set)  # {ws_id: set(fragment_ids)}
    cdef int fragment_id, starter_fragment_id, upstream_fragment_id
    cdef last_log_time = ctime(NULL)
    working_fragments_layer.StartTransaction()
    for fragment in fragments_layer:
        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            LOGGER.info('%s of %s fragments loaded (%.2f %%)',
                len(original_fragment_fids), n_fragments,
                (float(len(original_fragment_fids))/n_fragments)*100.)

        fragment_id = fragment.GetField('fragment_id')
        if fragment_id in original_fragment_fids:
            original_fragment_fids[fragment_id].add(fragment.GetFID())
        else:
            original_fragment_fids[fragment_id] = set([fragment.GetFID()])
        fragment_ids.add(fragment_id)
        member_ws_ids = set([int(ws_id) for ws_id in
                             fragment.GetField('member_ws_ids').split(',')])
        for ws_id in member_ws_ids:
            fragments_in_watershed[ws_id].add(fragment_id)

        geometry = fragment.GetGeometryRef()

        try:
            upstream_fragments[fragment_id] = [
                int(f) for f in fragment.GetField('upstream_fragments').split(',')]
        except ValueError:
            # If no upstream fragments the string will be '', and ''.split(',')
            # turns into [''], which crashes when you cast it to an int.
            # We're using a defaultdict(list) here, so no need to do anything.
            if fragment_id in fragment_fids:
                # If we've already encountered a feature with this fragment id,
                # get that feature and add this new geometry to it.
                working_fragment_feature = working_fragments_layer.GetFeature(
                    fragment_fids[fragment_id])
                working_fragment_geometry = working_fragment_feature.GetGeometryRef()

                # working_fragment_layer takes multipolygon geometries, so we can
                # just add the new geometry to the existing one.
                _add_geometries_to_multipolygon(working_fragment_geometry, geometry)

                working_fragment_feature.SetGeometry(working_fragment_geometry)
                working_fragments_layer.SetFeature(working_fragment_feature)
            else:
                # If this is the first time we're encountering a geometry with
                # this feature ID, create a new feature for it and add it to the
                # working feature layer.
                working_fragment_feature = ogr.Feature(
                    working_fragments_layer.GetLayerDefn())
                working_fragment_geometry = ogr.Geometry(ogr.wkbMultiPolygon)

                # working_fragment_layer takes multipolygon geometries, so we can
                # just add the new geometry to the existing one.
                _add_geometries_to_multipolygon(working_fragment_geometry, geometry)

                working_fragment_feature.SetGeometry(working_fragment_geometry)
                working_fragments_layer.CreateFeature(working_fragment_feature)
                fragment_fids[fragment_id] = working_fragment_feature.GetFID()
    working_fragments_layer.CommitTransaction()

    n_solo_fragments = working_fragments_layer.GetFeatureCount()
    LOGGER.info('%s fragments have no upstream fragments',
        n_solo_fragments)

    # Construct the base geometries for each watershed.
    LOGGER.info('Joining upstream fragments')
    cdef cstack[int] stack
    cdef cset[int] stack_set
    cdef clist[int] upstream_fragment_ids
    last_log_time = ctime(NULL)
    working_fragments_layer.StartTransaction()
    for starter_fragment_id in fragment_ids:
        # If the geometry has already been compiled, we can skip the stack step.
        if starter_fragment_id in fragment_fids:
            continue

        if starter_fragment_id is None:
            LOGGER.warn('Starter fragment is None')

        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            n_complete = working_fragments_layer.GetFeatureCount()
            LOGGER.info('%s of %s fragments complete (%.3f %%)',
                n_complete, n_fragments, (float(n_complete-n_solo_fragments)/n_solo_fragments)*100)

        for fragment_id in upstream_fragments[starter_fragment_id]:
            stack.push(fragment_id)
            stack_set.insert(fragment_id)

        geometries = ogr.Geometry(ogr.wkbMultiPolygon)
        for fragment_fid in original_fragment_fids[starter_fragment_id]:
            fragment = fragments_layer.GetFeature(fragment_fid)
            if fragment is None:
                LOGGER.warn('No fragment at FID %s', fragment_fid)

            _add_geometries_to_multipolygon(geometries, fragment.GetGeometryRef())

        while not stack.empty():
            fragment_id = stack.top()
            stack.pop()
            stack_set.erase(fragment_id)

            try:
                # Base case: this fragment already has geometry in the working layer.
                fragment_feature = working_fragments_layer.GetFeature(
                    fragment_fids[fragment_id])
                fragment_geometry = fragment_feature.GetGeometryRef()
                _add_geometries_to_multipolygon(geometries, fragment_geometry)
            except KeyError:
                # Fragment geometry has not yet been compiled.
                # Get the fragment's geometry and push it onto the stack.

                # Are all of the geometries upstream of this fragment in the compiled dict?
                # If yes, cascade-union them and save.
                upstream_fragment_ids = upstream_fragments[fragment_id]
                all_upstream_fragments_complete = True
                for upstream_fragment_id in upstream_fragment_ids:
                    if upstream_fragment_id not in fragment_fids:
                        all_upstream_fragments_complete = False
                        break

                if all_upstream_fragments_complete:
                    upstream_fragment_feature = ogr.Feature(
                        working_fragments_layer.GetLayerDefn())

                    local_geometry = ogr.Geometry(ogr.wkbMultiPolygon)
                    for upstream_fragment_id in upstream_fragment_ids:
                        upstream_feature = working_fragments_layer.GetFeature(
                            fragment_fids[upstream_fragment_id])
                        upstream_geom = upstream_feature.GetGeometryRef()
                        _add_geometries_to_multipolygon(local_geometry, upstream_geom)

                    upstream_fragment_feature.SetGeometry(local_geometry)
                    working_fragments_layer.CreateFeature(upstream_fragment_feature)
                    fragment_fids[fragment_id] = upstream_fragment_feature.GetFID()
                # If not, push the current fragment to the stack (so we visit it later)
                else:
                    if stack_set.find(fragment_id) != stack_set.end():
                        stack.push(fragment_id)
                        stack_set.insert(fragment_id)
                    for upstream_fragment_id in upstream_fragment_ids:
                        # Don't push a geometry that's already been calculated
                        if upstream_fragment_id in fragment_fids:
                            continue

                        if stack_set.find(upstream_fragment_id) != stack_set.end():
                            stack.push(upstream_fragment_id)
                            stack_set.insert(upstream_fragment_id)

        fragment_feature = ogr.Feature(working_fragments_layer.GetLayerDefn())
        fragment_feature.SetGeometry(geometries)
        working_fragments_layer.CreateFeature(fragment_feature)
        fragment_fids[starter_fragment_id] = fragment_feature.GetFID()
    working_fragments_layer.CommitTransaction()

    # Load the attributes table into a dict for easier accesses
    watershed_attributes = {}
    for feature in outflow_attributes_layer:
        watershed_attributes[feature.GetField('__ws_id__')] = feature.items()

    LOGGER.info('Copying attributes')
    last_log_time = ctime(NULL)
    for ws_id, fragments_set in fragments_in_watershed.items():
        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            n_compiled = watersheds_layer.GetFeatureCount()
            n_watersheds = len(fragments_in_watershed)
            LOGGER.info('Compiled %s of %s watersheds (%.2f %%)',
                n_compiled, n_watersheds, (float(n_compiled)/n_watersheds)*100.)

        watersheds_layer.StartTransaction()
        target_feature = ogr.Feature(watersheds_layer_defn)

        # Compile the geometry from all member fragments.
        # TODO: try timing the cascaded union against rasterizing, polygonizing and clearing the raster ... maybe that's faster?
        #   - If it is, maybe we can get away with *not* doing any unions at all, and just relying on the rasterization?
        #   - Try this method out on the fragments vector (without delineating watersheds again)
        #   - May need to store extra information in *another* aspatial table in the fragments vector
        #        - I'm thinking that this would need to be raster information so we can create the sample raster easily.
        #   - If rasterization ends up being a good idea, maybe I can do disjoint sets to speed up the process?
        new_geometry = ogr.Geometry(ogr.wkbMultiPolygon)
        for fragment_id in fragments_set:
            fragment_feature = working_fragments_layer.GetFeature(
                fragment_fids[fragment_id])
            fragment_geometry = fragment_feature.GetGeometryRef()
            _add_geometries_to_multipolygon(new_geometry, fragment_geometry)

        target_feature.SetGeometry(new_geometry.Buffer(0))
        #target_feature.SetGeometry(new_geometry.UnionCascaded())

        # Copy field values over to the new feature.
        for field_name, field_value in watershed_attributes[ws_id].items():
            target_feature.SetField(field_name, field_value)

        watersheds_layer.CreateFeature(target_feature)
        watersheds_layer.CommitTransaction()
    LOGGER.info('Compilation 100% complete')


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


def group_seeds_into_fragments_d8(
        d8_flow_dir_raster_path_band, seeds_to_watershed_membership_map):
    """Group seeds into contiguous fragments, represented by a unique ID.

    Fragment membership is determined by walking the flow direction raster to
    determine upstream and downstream linkages between seeds and then analyze
    the resulting (abbreviated) flow graph (as well as the seeds' watershed
    membership) to determine which seeds should be grouped together by IDs.

    The point of this is to reduce the number of unique fragments (represented
    in a raster by a unique ID) needed.  Successfully reducing this has the
    following benefits:

        * Reducing the number of IDs dramatically increases the number of
          fragments that can be uniquely identified.  In pracrice, I've seen an
          order of magnitude increase here.
        * Reducing the number of IDs reduces the time spent polygonizing
          fragments after we've walked the flow direction raster.
        * Reducing the number of polygonized fragments reduces the time spent
          unioning geometries after the fragments have been polygonized.

    Parameters:
        d8_flow_dir_raster_path_band (tuple): A path/band index tuple.
        seeds_to_watershed_membership_map (dict): A dict mapping a seed
            (a tuple representing the (x, y) index of a pixel) to a set of
            integer watersheds.  The set represents the unique watershed IDs
            that this seed belongs to.
    """
    cdef _ManagedRaster flow_dir_managed_raster
    flow_dir_managed_raster = _ManagedRaster(d8_flow_dir_raster_path_band[0],
                                                  d8_flow_dir_raster_path_band[1],
                                                  0)  # read-only



def delineate_watersheds_d8(
        d8_flow_dir_raster_path_band, outflow_vector_path,
        target_fragments_vector_path, working_dir=None):
    if (d8_flow_dir_raster_path_band is not None and not
            _is_raster_path_band_formatted(d8_flow_dir_raster_path_band)):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                d8_flow_dir_raster_path_band))

    cdef int ws_id  # start indexing ws_id at 1
    if starting_ws_id is None:
        ws_id = 1
    else:
        if not isinstance(starting_ws_id, int) or starting_ws_id <= 0:
            raise ValueError(
                'starting_ws_id must be a positive, nonzero integer; %s found'
                % starting_ws_id)
        else:
            ws_id = starting_ws_id

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
    # Optimizations that can be done outside of watershed delineation:
    #  * geometries can be simplified according to the nyquist-shannon sampling theorem
    #  * any geometries outside of the bounding box of the flow direction raster can be
    #    clipped to the bounding box or else excluded entirely.
    #  * Geometries must be valid, and should be repaired outside of watershed delineation.
    #
    #
    # 1. Determine which pixels are seeds and which watersheds are represented by which seed
    #    * This is the disjoint-polygon/rasterization step
    #        * Points and geometries only overlapping one pixel do not need to be rasterized.
    #        * Lines (geometries with no area) and small polygons need to be
    #          buffered to ensure no overlap before rasterization.
    #    * This should also be nodata-aware.  Should not end up with seeds that are over nodata.
    # 2. Use the flow direction raster and the watershed membership data structures to
    #    identify which seeds should have which IDs, grouped as needed into fragments.
    #    * While doing this, we can also track upstream/downstream dependencies and return
    #      them as needed to the parent function.
    #    * Assume UInt32 for return ID.
    #    * Raise an error if more seeds than 2**32-1
    # 3. Group seeds together into blocks to try to take advantage of LRUCache
    # 4. Iterate through seeds, delineating watersheds.
    # 5. Polygonize the fragments.
    #
    # 6? Recursively join fragments.


def delineate_watersheds_trivial_d8(
        d8_flow_dir_raster_path_band, outflow_vector_path,
        target_watersheds_vector_path, working_dir=None):

    try:
        if working_dir is not None:
            os.makedirs(working_dir)
    except OSError:
        pass
    working_dir_path = tempfile.mkdtemp(
        dir=working_dir, prefix='watershed_delineation_%s_' % time.strftime(
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
    flow_dir_bbox = shapely.prepared.prep(
        shapely.geometry.box(*flow_dir_info['bounding_box']))
    flow_dir_managed_raster = _ManagedRaster(d8_flow_dir_raster_path_band[0],
                                             d8_flow_dir_raster_path_band[1], 0)

    gpkg_driver = gdal.GetDriverByName('GPKG')
    flow_dir_srs = osr.SpatialReference()
    flow_dir_srs.ImportFromWkt(flow_dir_info['projection'])

    source_outlets_vector = gdal.OpenEx(outflow_vector_path, gdal.OF_VECTOR)
    if source_outlets_vector is None:
        raise ValueError(u'Could not open outflow vector %s' % outflow_vector_path)

    scratch_raster_path = os.path.join(working_dir_path, 'scratch.tif')
    pygeoprocessing.new_raster_from_base(
        d8_flow_dir_raster_path_band[0], scratch_raster_path, gdal.GDT_Int32,
        [-1], fill_value_list=[0], gtiff_creation_options=GTIFF_CREATION_OPTIONS)

    driver = ogr.GetDriverByName('GPKG')
    watersheds_srs = osr.SpatialReference()
    watersheds_srs.ImportFromWkt(flow_dir_info['projection'])
    watersheds_vector = driver.CreateDataSource(target_watersheds_vector_path)
    watersheds_layer = watersheds_vector.CreateLayer(
        'watersheds', watersheds_srs, ogr.wkbPolygon)
    index_field = ogr.FieldDefn('ws_id', ogr.OFTInteger)
    index_field.SetWidth(24)
    watersheds_layer.CreateField(index_field)

    #polygons_layer = watersheds_vector.CreateLayer(
    #    'polygons', watersheds_srs, ogr.wkbPolygon)

    source_outlets_layer = source_outlets_vector.GetLayer()
    feature_count = source_outlets_layer.GetFeatureCount()
    cdef int* reverse_flow = [4, 5, 6, 7, 0, 1, 2, 3]
    cdef int* neighbor_col = [1, 1, 0, -1, -1, -1, 0, 1]
    cdef int* neighbor_row = [0, -1, -1, -1, 0, 1, 1, 1]
    cdef double minx, miny, maxx, maxy
    cdef double cx_pixel, cy_pixel  # coordinates of the pixel
    cdef long ix_pixel, iy_pixel  # indexes of the pixel
    cdef queue[CoordinatePair] process_queue
    cdef cset[CoordinatePair] process_queue_set
    cdef CoordinatePair pixel_coords, neighbor_pixel
    ws_id_to_fid = {}
    for ws_id, feature in enumerate(source_outlets_layer, 1):
        geometry = feature.GetGeometryRef()
        minx, maxx, miny, maxy = geometry.GetEnvelope()
        geom_bbox = shapely.geometry.box(minx, miny, maxx, maxy)

        # If the geometry's envelope does not intersect with the bounding box
        # of the DEM, skip the geometry entirely.
        if not flow_dir_bbox.intersects(geom_bbox):
            LOGGER.info(
                'Skipping watershed %s of %s; feature does not intersect '
                'flow direction raster', ws_id, feature_count)
            continue

        #bbox_feature = ogr.Feature(polygons_layer.GetLayerDefn())
        #bbox_feature.SetGeometry(ogr.CreateGeometryFromWkb(geom_bbox.wkb))
        #polygons_layer.CreateFeature(bbox_feature)

        # Otherwise:
        # Build a shapely prepared polygon of the feature's geometry.
        geom_prepared = shapely.prepared.prep(shapely.wkb.loads(geometry.ExportToWkb()))

        # Expand the bounding box to align with the nearest pixels.
        minx = min(minx, minx - fmod(minx, flow_dir_pixelsize_x))
        miny = min(miny, miny + fmod(miny, fabs(flow_dir_pixelsize_y)))
        maxx = max(maxx, maxx - fmod(maxx, flow_dir_pixelsize_x) + flow_dir_pixelsize_x)
        maxy = max(maxy, maxy + fmod(maxy, fabs(flow_dir_pixelsize_y)) + fabs(flow_dir_pixelsize_y))

        #bbox_geometry = shapely.geometry.box(minx, miny, maxx, maxy)
        #bbox_feature = ogr.Feature(polygons_layer.GetLayerDefn())
        #bbox_feature.SetGeometry(ogr.CreateGeometryFromWkb(bbox_geometry.wkb))
        #polygons_layer.CreateFeature(bbox_feature)

        scratch_managed_raster = _ManagedRaster(scratch_raster_path, 1, 1)

        # Use the DEM's geotransform to determine the starting coordinates for iterating
        # over the pixels within the area of the envelope.
        #polygons_layer.StartTransaction()

        # this needs to be over the pixel directly, lest the coordinates are off-by-one.
        cx_pixel = minx + (flow_dir_pixelsize_x / 2.)
        while cx_pixel < maxx:
            ix_pixel = <long>((cx_pixel - flow_dir_origin_x) // flow_dir_pixelsize_x)

            cy_pixel = maxy
            while cy_pixel > miny:
                iy_pixel = <long>((cy_pixel - flow_dir_origin_y) // flow_dir_pixelsize_y)

                pixel_geometry = shapely.geometry.box(cx_pixel,
                                                      cy_pixel + flow_dir_pixelsize_y,
                                                      cx_pixel + flow_dir_pixelsize_x,
                                                      cy_pixel)
                cy_pixel += flow_dir_pixelsize_y

                if not geom_prepared.intersects(pixel_geometry):
                    continue

                if not 0 <= ix_pixel < flow_dir_n_cols:
                    continue

                if not 0 <= iy_pixel < flow_dir_n_rows:
                    continue

                if flow_dir_managed_raster.get(ix_pixel, iy_pixel) == flow_dir_nodata:
                    continue

                pixel_coords = CoordinatePair(ix_pixel, iy_pixel)

                process_queue_set.insert(pixel_coords)
                process_queue.push(pixel_coords)

                #pixel_feature = ogr.Feature(polygons_layer.GetLayerDefn())
                #pixel_feature.SetGeometry(ogr.CreateGeometryFromWkb(pixel_geometry.wkb))
                #polygons_layer.CreateFeature(pixel_feature)

            cx_pixel += flow_dir_pixelsize_x

        #polygons_layer.CommitTransaction()

        if process_queue.size() == 0:
            LOGGER.info('Skipping watershed %s of %s; feature has no valid pixels',
                        ws_id, feature_count)
            continue

        LOGGER.info('Delineating watershed %s of %s from %s pixels', ws_id,
                    feature_count, process_queue.size())
        ws_id_to_fid[ws_id] = feature.GetFID()

        while not process_queue.empty():
            current_pixel = process_queue.front()
            process_queue_set.erase(current_pixel)
            process_queue.pop()

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

                # If we've already enqueued the neighbor (either it's
                # upstream of another pixel or it's a watershed seed), we
                # don't need to re-enqueue it.
                if (process_queue_set.find(neighbor_pixel) !=
                        process_queue_set.end()):
                    continue

                # Does the neighbor flow into this pixel?
                # If yes, enqueue it.
                if (reverse_flow[neighbor_index] ==
                        flow_dir_managed_raster.get(
                            neighbor_pixel.first, neighbor_pixel.second)):
                    process_queue.push(neighbor_pixel)
                    process_queue_set.insert(neighbor_pixel)

        scratch_managed_raster.close()

        # Polygonize this new fragment.
        scratch_raster = gdal.OpenEx(scratch_raster_path, gdal.OF_RASTER | gdal.GA_Update)
        scratch_band = scratch_raster.GetRasterBand(1)
        result = gdal.Polygonize(
            scratch_band,  # The source band
            scratch_band,  # The mask indicating valid pixels
            watersheds_layer,
            0,  # ws_id field index
            ['8CONNECTED=8'])

        # TODO: copy all of the fields over from the source vector.
        scratch_band.Fill(0)  # reset the scratch band
        scratch_band.FlushCache()
        scratch_band = None
        scratch_raster = None

    watersheds_layer = None
    watersheds_vector = None

    LOGGER.info('Copying field values to the target vector.')
    # last step: open up the source vector and the target vector, create all
    # fields needed and set the values accordingly.
    source_vector = gdal.OpenEx(outflow_vector_path, gdal.OF_VECTOR)
    source_layer = source_vector.GetLayer()

    watersheds_vector = gdal.OpenEx(target_watersheds_vector_path,
                                    gdal.OF_VECTOR | gdal.GA_Update)
    watersheds_layer = watersheds_vector.GetLayer('watersheds')
    for source_field_defn in source_layer.schema:
        watersheds_layer.CreateField(source_field_defn)

    for watershed_feature in watersheds_layer:
        ws_id = watershed_feature.GetField('ws_id')

        source_feature = source_layer.GetFeature(ws_id_to_fid[ws_id])
        for field_name, field_value in source_feature.items().items():
            watershed_feature.SetField(field_name, field_value)
        watersheds_layer.SetFeature(watershed_feature)

    watersheds_layer = None
    watersheds_vector = None
    source_layer = None
    source_vector = None

    shutil.rmtree(working_dir_path)

