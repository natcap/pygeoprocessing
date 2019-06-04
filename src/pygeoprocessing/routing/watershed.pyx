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
    'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
    'BLOCKYSIZE=%d' % (1 << BLOCK_BITS))

# this is used to calculate the opposite D8 direction interpreting the index
# as a D8 direction
cdef int* D8_REVERSE_DIRECTION = [4, 5, 6, 7, 0, 1, 2, 3]
cdef int* NEIGHBOR_COL = [1, 1, 0, -1, -1, -1, 0, 1]
cdef int* NEIGHBOR_ROW = [0, -1, -1, -1, 0, 1, 1, 1]

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

    def __init__(self, raster_path, band_id, write_mode):
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
            raise OSError('%s is not a file.' % raster_path)
        raster_info = pygeoprocessing.get_raster_info(raster_path)

        if not (1 <= band_id <= raster_info['n_bands']):
            err_msg = (
                "Error: band ID (%s) is not a valid band number. "
                "This exception is happening in Cython, so it will cause a "
                "hard seg-fault, but it's otherwise meant to be a "
                "ValueError." % (band_id))
            LOGGER.error(err_msg)
            raise ValueError(err_msg)

        block_xsize, block_ysize = raster_info['block_size']
        if (block_xsize & (block_xsize - 1) != 0) or (
                block_ysize & (block_ysize - 1) != 0):
            # If inputs are not a power of two, this will at least print
            # an error message. Unfortunately with Cython, the exception will
            # present itself as a hard seg-fault, but I'm leaving the
            # ValueError in here at least for readability.
            err_msg = (
                "Error: Block size is not a power of two: "
                "block_xsize: %d, %d, %s. This exception is happening"
                "in Cython, so it will cause a hard seg-fault, but it's"
                "otherwise meant to be a ValueError." % (
                    block_xsize, block_ysize, raster_path))
            LOGGER.error(err_msg)
            raise ValueError(err_msg)

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
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        self.raster_x_size, self.raster_y_size = raster_info['raster_size']
        self.block_xsize, self.block_ysize = raster_info['block_size']
        self.block_xmod = self.block_xsize-1
        self.block_ymod = self.block_ysize-1

        self.band_id = band_id

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


def split_vector_into_seeds(
        source_vector_path, d8_flow_dir_raster_path_band,
        source_vector_layer=None, working_dir=None, remove=True,
        write_diagnostic_vector=False, start_index=0):
    """Analyze the source vector and break all geometries into seeds.

    For D8 watershed delination, ``seeds`` represent (x, y) pixel coordinates
    on the flow direction raster as determined by a rasterization with GDAL's
    ``ALL_TOUCHED=TRUE`` option enabled.  Seeds can represent multiple
    watersheds, which is the result of having multiple geometries overlapping
    one or more of the same pixels.

    A seed will only be created when the seed is over a valid D8 flow direction
    pixel.  Any geometries over nodata pixels on the flow direction raster will
    not have seeds created where they overlap with nodata.

    For optimal performance, consider preparing your source vector such that:

        * All geometries are valid.  This function makes no attempt to correct
          invalid geometries and an exception will be raised if a geometry is
          found to be invalid.
        * All geometries are simplified to 1/2 the pixel width of your flow
          direction raster according to the Nyquist-Shannon sampling theorem.
          Doing so will result in faster (and no less accurate) rasterizations
          for particularly complex geometries.
        * All geometries intersect the flow direction raster.  Failure to do so
          will result in slower-than-necessary rasterization and additional
          overhead in looping.

    Parameters:
        source_vector_path (string): A path to a GDAL-compatible vector on
            disk containing one or more layers with one or more geometries to
            analyze.  All geometries in the target layer must be valid, but may
            be of any type (point, polygon, etc.) supported by GDAL.
            The projection of this vector must match that of the flow direction
            raster.
        d8_flow_dir_raster_path_band (tuple): A (path, band) tuple where
            ``path`` represents a path to a GDAL-compatible raster on disk and
            ``band`` represents a 1-based band index.  The projection of this
            raster must match that of the source vector.
        source_vector_layer=None (string, int or None): An identifier for
            the layer of the vector at ``source_vector_path`` to use.
            If ``None``, the first layer from the source vector will be used.
        working_dir=None (string or None): The path to a directory on disk
            where intermediate files will be stored.  This directory will be
            created if it does not exist, and intermediate files created will
            be removed.  If ``None``, a new temporary folder will be created
            within the system temp directory.

    Returns:
        seed_watershed_membership (dict): A python dict mapping (x, y) pixel
            index tuples to sets of integer watershed FIDs.  The watershed ID
            represents the FID of the outflow geometry represented by this
            seed.  A seed with multiple watershed FIDs is a part of multiple
            watersheds.
    """
    # TODO: does this function also need to identify which source features map to which WS_IDs?
    #       Can we just refer to FIDs instead of using internal identifiers?

    # Structure:
    #  * Loop through the source geometries
    #      * Any points can be removed from the vector processing and entered directly into the data structure
    #      * Geometries only intersecting a single pixel can be treated like points.
    #      * Geometries with no area (e.g. lines) and small polygons must be buffered.
    #      * Remaining geometries should be copied to a new temporary vector.
    #  * Use this temporary vector to determine sets of disjoint polygons
    #  * For each set of disjoint polygons:
    #      * Rasterize the set of polygons
    #      * Iterblocks through the rasterized raster
    #      * Determine seeds for any pixels that are over valid flow direction pixels.
    #      * Track the seeds and watershed membership in the output dict.
    #  * Remove the working directory
    #  * Return the data structure

    try:
        if working_dir is not None:
            os.makedirs(working_dir)
    except OSError:
        pass
    working_dir_path = tempfile.mkdtemp(
        dir=working_dir, prefix='seed_extraction_%s_' % time.strftime(
            '%Y-%m-%d_%H_%M_%S', time.gmtime()))

    source_vector = gdal.OpenEx(source_vector_path, gdal.OF_VECTOR)
    if source_vector_layer is None:
        source_vector_layer = 0  # default parameter value for GetLayer
    source_layer = source_vector.GetLayer(source_vector_layer)

    flow_dir_info = pygeoprocessing.get_raster_info(
        d8_flow_dir_raster_path_band[0])
    source_gt = flow_dir_info['geotransform']
    flow_dir_bbox = shapely.prepared.prep(
        shapely.geometry.box(*flow_dir_info['bounding_box']))
    flow_dir_srs = osr.SpatialReference()
    flow_dir_srs.ImportFromWkt(flow_dir_info['projection'])
    cdef int flow_dir_nodata = flow_dir_info['nodata'][0]
    cdef float minx, miny, maxx, maxy
    cdef double x_origin = source_gt[0]
    cdef double y_origin = source_gt[3]
    cdef double x_pixelwidth = source_gt[1]
    cdef double y_pixelwidth = source_gt[5]
    cdef double pixel_area = abs(x_pixelwidth * y_pixelwidth)
    cdef double buffer_dist = math.hypot(x_pixelwidth, y_pixelwidth) / 2. * 1.1
    no_watershed = (2**32)-1  # max value for UInt32


    seed_ids = {}  # map {(x, y coordinate pair): ID}

    # watershed IDs are represented by the FID of the outflow geometry.
    seed_watersheds = collections.defaultdict(set)  # map {(x, y coordinate pair): set(watershed IDs)}

    flow_dir_managed_raster = _ManagedRaster(d8_flow_dir_raster_path_band[0],
                                             d8_flow_dir_raster_path_band[1],
                                             0)  # read-only
    gpkg_driver = gdal.GetDriverByName('GPKG')
    temp_polygons_vector_path = os.path.join(working_dir_path, 'temp_polygons.gpkg')
    temp_polygons_vector = gpkg_driver.Create(
        temp_polygons_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    temp_polygons_layer = temp_polygons_vector.CreateLayer(
        'outlet_geometries', flow_dir_srs, source_layer.GetGeomType())
    temp_polygons_layer.CreateField(ogr.FieldDefn('WSID', ogr.OFTInteger))

    seed_id = start_index  # assume Seed IDs can be from (2 - 2**32-1) inclusive in UInt32
    temp_polygons_layer.StartTransaction()
    for feature in source_layer:
        if seed_id > 2**32-1:
            raise ValueError('Too many seeds have been created.')

        # This will raise a GEOS exception if the geometry is invalid.
        geometry = shapely.wkb.loads(feature.GetGeometryRef().ExportToWkb())

        # If the geometry doesn't intersect the flow dir bounding box at all,
        # don't bother with it.
        if not flow_dir_bbox.intersects(shapely.geometry.box(*geometry.bounds)):
            continue

        minx, miny, maxx, maxy = geometry.bounds
        minx_pixelcoord = <int>((minx - x_origin) // x_pixelwidth)
        miny_pixelcoord = <int>((miny - y_origin) // y_pixelwidth)
        maxx_pixelcoord = <int>((maxx - x_origin) // x_pixelwidth)
        maxy_pixelcoord = <int>((maxy - y_origin) // y_pixelwidth)

        # If the geometry only intersects a single pixel, we can treat it
        # as a single point, which means that we can track it directly in our
        # seeds data structure and not have to include it in the disjoint set
        # determination.
        if minx_pixelcoord == maxx_pixelcoord and miny_pixelcoord == maxy_pixelcoord:
            # If the point is over nodata, skip it.
            if (flow_dir_managed_raster.get(minx_pixelcoord, miny_pixelcoord)
                    == flow_dir_nodata):
                continue

            seed = (minx_pixelcoord, miny_pixelcoord)
            seed_ids[seed] = seed_id
            seed_watersheds[seed].add(feature.GetFID())
            seed_id += 1
            continue  # No need to create a feature for this outflow geometry

        else:
            # If we can't fit the geometry into a single pixel, there remain
            # two special cases that warrant buffering:
            #     * It's a line (lines don't have area). We want to avoid a
            #       situation where multiple lines cover the same pixels
            #       but don't intersect.  Such geometries should be treated
            #       as overlapping and handled in disjoint sets.
            #     * It's a polygon that has area smaller than a pixel. This
            #       came up in real-world sample data (the Montana Lakes
            #       example, specifically), where some very small lakes were
            #       disjoint but both overlapped only one pixel, the same pixel.
            #       This lead to a race condition in rasterization where only
            #       one of them would be in the output vector.
            if (geometry.area == 0.0 or geometry.area <= pixel_area):
                geometry = geometry.buffer(buffer_dist)

            new_feature = ogr.Feature(temp_polygons_layer.GetLayerDefn())
            new_feature.SetField('WSID', feature.GetFID())
            new_feature.SetGeometry(ogr.CreateGeometryFromWkb(geometry.wkb))
            temp_polygons_layer.CreateFeature(new_feature)
    temp_polygons_layer.CommitTransaction()

    temp_polygons_schema = temp_polygons_layer.schema
    temp_polygons_n_features = temp_polygons_layer.GetFeatureCount()

    if temp_polygons_n_features > 0:
        LOGGER.info('Determining sets of non-overlapping geometries')

        for set_index, disjoint_polygon_fid_set in enumerate(
                pygeoprocessing.calculate_disjoint_polygon_set(
                    temp_polygons_vector_path,
                    bounding_box = flow_dir_info['bounding_box']),
                start=1):
            LOGGER.info('Creating a vector of %s disjoint geometries',
                        len(disjoint_polygon_fid_set))

            disjoint_vector_path = os.path.join(
                    working_dir_path, 'disjoint_outflow_%s.gpkg' % set_index)
            disjoint_vector = gpkg_driver.Create(disjoint_vector_path, 0, 0, 0,
                                                 gdal.GDT_Unknown)
            disjoint_layer = disjoint_vector.CreateLayer(
                'outflow_geometries', flow_dir_srs, ogr.wkbPolygon)
            disjoint_layer.CreateFields(temp_polygons_schema)

            disjoint_layer.StartTransaction()
            # Though the buffered working layer was used for determining the sets
            # of nonoverlapping polygons, we want to use the *original* outflow
            # geometries for rasterization.  This step gets the appropriate features
            # from the correct vectors so we keep the correct geometries.
            for temp_polygons_fid in disjoint_polygon_fid_set:
                disjoint_feature = temp_polygons_layer.GetFeature(temp_polygons_fid)
                disjoint_wsid = disjoint_feature.GetField('WSID')

                source_feature = source_layer.GetFeature(disjoint_wsid)  # WSID represents FID in this vector

                # The disjoint vector to be rasterized should have the
                # geometries in the disjoint set, but from the source vector.
                new_feature = ogr.Feature(disjoint_layer.GetLayerDefn())
                new_feature.SetGeometry(source_feature.GetGeometryRef())
                new_feature.SetField('WSID', disjoint_wsid)  # preserve the original fid
                disjoint_layer.CreateFeature(new_feature)

            disjoint_layer.CommitTransaction()

            disjoint_layer = None
            disjoint_vector = None

            tmp_seed_raster_path = os.path.join(working_dir_path,
                                                'disjoint_outflow_%s.tif' % set_index)
            pygeoprocessing.new_raster_from_base(
                d8_flow_dir_raster_path_band[0], tmp_seed_raster_path,
                gdal.GDT_UInt32, [no_watershed], fill_value_list=[no_watershed],
                gtiff_creation_options=GTIFF_CREATION_OPTIONS)

            pygeoprocessing.rasterize(
                disjoint_vector_path, tmp_seed_raster_path, None,
                ['ALL_TOUCHED=TRUE', 'ATTRIBUTE=WSID'],
                layer_index='outflow_geometries')

            flow_dir_raster = gdal.OpenEx(d8_flow_dir_raster_path_band[0], gdal.OF_RASTER)
            flow_dir_band = flow_dir_raster.GetRasterBand(d8_flow_dir_raster_path_band[1])

            tmp_seed_raster = gdal.OpenEx(tmp_seed_raster_path, gdal.OF_RASTER)
            tmp_seed_band = tmp_seed_raster.GetRasterBand(1)
            for block_info in pygeoprocessing.iterblocks(
                    (tmp_seed_raster_path, 1), offset_only=True):
                flow_dir_array = flow_dir_band.ReadAsArray(**block_info)
                tmp_seed_array = tmp_seed_band.ReadAsArray(**block_info)

                valid_outflow_geoms_mask = ((flow_dir_array != flow_dir_nodata) &
                                            (tmp_seed_array != no_watershed))
                for (row, col) in zip(*numpy.nonzero(valid_outflow_geoms_mask)):
                    ws_id = tmp_seed_array[row, col]
                    seed = (col + block_info['xoff'], row + block_info['yoff'])
                    seed_watersheds[seed].add(ws_id)
                    seed_ids[seed] = seed_id
                    seed_id += 1

            tmp_seed_band = None
            tmp_seed_raster = None
            flow_dir_band = None
            flow_dir_raster = None


    if write_diagnostic_vector:
        diagnostic_vector = gpkg_driver.Create(
            os.path.join(working_dir_path, 'diagnostic.gpkg'),
            0, 0, 0, gdal.GDT_Unknown)
        seeds_layer = diagnostic_vector.CreateLayer(
            'seeds', flow_dir_srs, ogr.wkbPoint)
        seeds_layer.CreateField(ogr.FieldDefn('watersheds', ogr.OFTString))

        seeds_layer.StartTransaction()
        for seed, watershed_id_set in seed_watersheds.items():
            feature = ogr.Feature(seeds_layer.GetLayerDefn())
            point = shapely.geometry.Point(
                x_origin + seed[0] * x_pixelwidth + (x_pixelwidth / 2.),
                y_origin + seed[1] * y_pixelwidth + (y_pixelwidth / 2.))
            feature.SetGeometry(ogr.CreateGeometryFromWkb(point.wkb))
            feature.SetField('watersheds', ','.join([str(s) for s in sorted(watershed_id_set)]))
            seeds_layer.CreateFeature(feature)

        seeds_layer.CommitTransaction()

    if remove:
        shutil.rmtree(working_dir_path, ignore_errors=True)
    return dict(seed_watersheds)  # remove defaultdict capabilities


def group_seeds_into_fragments_d8(
        d8_flow_dir_raster_path_band, seeds_to_watershed_membership_map,
        diagnostic_vector_path=None):
    """Group seeds into contiguous fragments, represented by a unique ID.

    Fragment membership is determined by walking the flow direction raster to
    determine upstream and downstream linkages between seeds and then analyze
    the resulting abbreviated flow graph (as well as the seeds' watershed
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

    Returns:
        seed_ids (dict): A dict mapping tuples of (x, y) pixel coordinates of
            seeds to their new unique IDs having been grouped together into
            contiguous fragments.
    """
    flow_dir_info = pygeoprocessing.get_raster_info(d8_flow_dir_raster_path_band[0])
    flow_dir_srs = osr.SpatialReference()
    flow_dir_srs.ImportFromWkt(flow_dir_info['projection'])
    gpkg_driver = gdal.GetDriverByName('GPKG')
    cdef long flow_dir_n_cols = flow_dir_info['raster_size'][0]
    cdef long flow_dir_n_rows = flow_dir_info['raster_size'][1]
    source_gt = flow_dir_info['geotransform']
    cdef double x_origin = source_gt[0]
    cdef double y_origin = source_gt[3]
    cdef double x_pixelwidth = source_gt[1]
    cdef double y_pixelwidth = source_gt[5]
    cdef int flow_dir_nodata = flow_dir_info['nodata'][0]
    cdef _ManagedRaster flow_dir_managed_raster
    flow_dir_managed_raster = _ManagedRaster(d8_flow_dir_raster_path_band[0],
                                             d8_flow_dir_raster_path_band[1],
                                             0)  # read-only
    seed_ids = {}
    for seed_id, seed in enumerate(seeds_to_watershed_membership_map.keys(), 1):
        seed_ids[seed] = seed_id

    # Step 1: determine which fragments are downstream of one another.
    downstream_seeds = {}
    for starter_seed in seeds_to_watershed_membership_map.keys():
        seed_flow_dir = flow_dir_managed_raster.get(starter_seed[0], starter_seed[1])
        neighbor_seed = (starter_seed[0] + NEIGHBOR_COL[seed_flow_dir],
                         starter_seed[1] + NEIGHBOR_ROW[seed_flow_dir])

        while True:
            # is the index a seed?
            # If yes, we've found a downstream seed and we're done.
            # D8 can only have one seed downstream of another.
            if neighbor_seed in seeds_to_watershed_membership_map:
                downstream_seeds[starter_seed] = neighbor_seed
                break

            if not 0 <= neighbor_seed[0] < flow_dir_n_cols:
                break
            if not 0 <= neighbor_seed[1] < flow_dir_n_rows:
                break

            current_flow_dir = flow_dir_managed_raster.get(neighbor_seed[0], neighbor_seed[1])
            if current_flow_dir == flow_dir_nodata:
                break

            neighbor_seed = (neighbor_seed[0] + NEIGHBOR_COL[current_flow_dir],
                             neighbor_seed[1] + NEIGHBOR_ROW[current_flow_dir])

    # TODO: make sure we can support multiple upstream fragments in this data structure.
    # now that we know which fragments are downstream of one another, we also
    # need to know which fragments are upstream of one another.
    nested_fragments = collections.defaultdict(list)
    for upstream_seed, downstream_seed in downstream_seeds.items():
        nested_fragments[downstream_seed].append(upstream_seed)
    nested_fragments = dict(nested_fragments)

    # Step 2: find the starter seeds.
    starter_seeds = set([])
    for seed in seeds_to_watershed_membership_map:
        try:
            while True:
                seed = downstream_seeds[seed]
        except KeyError:
            # Continue downstream until we can't.  When we can't any more, this
            # is our starter seed.
            starter_seeds.add(seed)

    starter_seeds = list(starter_seeds)  # can't change size of set during iteration
    effective_watersheds = {}
    visited = set([])
    effective_seed_ids = {}
    downstream_watersheds = {}
    reclassification = {}

    for starter_seed in starter_seeds:
        stack = [starter_seed]

        member_watersheds = seeds_to_watershed_membership_map[starter_seed]
        try:
            starter_id = effective_seed_ids[starter_seed]
        except KeyError:
            starter_id = seed_ids[starter_seed]
            effective_seed_ids[starter_seed] = starter_id

        while len(stack) > 0:
            current_seed = stack.pop()
            reclassification[seed_ids[current_seed]] = starter_id
            visited.add(current_seed)

            # Are there nested fragments?  If yes, see if this seed ID can
            # expand into the nested fragments.
            if current_seed in nested_fragments:
                for upstream_seed in nested_fragments[current_seed]:
                    if seeds_to_watershed_membership_map[upstream_seed].issubset(
                            seeds_to_watershed_membership_map[starter_seed]):
                        if upstream_seed not in stack and upstream_seed not in visited:
                            stack.append(upstream_seed)
                        effective_watersheds[upstream_seed] = seeds_to_watershed_membership_map[starter_seed]
                    else:
                        # The upstream seed appears to be the start of a different fragment.
                        # Add it to the starter seeds queue.
                        effective_watersheds[upstream_seed] = seeds_to_watershed_membership_map[upstream_seed]
                        if upstream_seed not in starter_seeds:
                            starter_seeds.append(upstream_seed)

                        # noting which watersheds are downstream of the
                        # upstream pixel is important for visiting
                        # neighbors and expanding into them, below.
                        downstream_watersheds[upstream_seed] = member_watersheds

            # visit neighbors and see if there are any neighbors that match
            for neighbor_id in xrange(8):
                neighbor_col = current_seed[0] + NEIGHBOR_COL[neighbor_id]
                neighbor_row = current_seed[1] + NEIGHBOR_ROW[neighbor_id]
                if not 0 <= neighbor_row < flow_dir_n_rows:
                    continue
                if not 0 <= neighbor_col < flow_dir_n_cols:
                    continue

                neighbor_seed = (neighbor_col, neighbor_row)

                # Does neighbor belong to current watershed?
                # If it doesn't exist (meaning it's a pixel that isn't a seed),
                # we don't consider it.
                if neighbor_seed not in seeds_to_watershed_membership_map:
                    continue

                if neighbor_seed in stack:
                    continue

                if seeds_to_watershed_membership_map[neighbor_seed] == member_watersheds:
                    # If we can compare the downstream neighbors, do so.
                    try:
                        # We only want to expand into the neighbor IFF the
                        # downstream watersheds match.
                        if downstream_watersheds[current_seed] == downstream_watersheds[neighbor_seed]:
                            effective_seed_ids[neighbor_seed] = starter_id
                    except KeyError:
                        # If we can't compare the downstream watersheds, it's
                        # because we're still too far downstream to have any
                        # meaningful linkages between them, and we can safely
                        # expand into the neighbor.

                        # check to see if there's a known downstream seed of this seed.
                        # if there is, we don't want to expand into it.
                        try:
                            downstream_seed = downstream_seeds[neighbor_seed]
                            if seeds_to_watershed_membership_map[neighbor_seed].issuperset(
                                    seeds_to_watershed_membership_map[downstream_seed]):
                                effective_seed_ids[neighbor_seed] = starter_id
                        except KeyError:
                            # no known downstream seeds to check, we're probably ok?
                            effective_seed_ids[neighbor_seed] = starter_id

                    # Add the seed to the stack if it isn't there already.
                    if neighbor_seed not in visited and neighbor_seed not in stack:
                        stack.append(neighbor_seed)

            try:
                for upstream_seed in nested_fragments[current_seed]:
                    # If the upstream seed is not a neighbor, it's probably some distance away.
                    if upstream_seed not in stack and upstream_seed not in visited:
                        stack.append(upstream_seed)
            except KeyError:
                # If there are no seeds upstream of the current seed, we can safely pass.
                pass


    consolidated_reclassification = {}
    for new_index, reclass_value in enumerate(numpy.unique(reclassification.values()), 1):
        consolidated_reclassification[reclass_value] = new_index

    if diagnostic_vector_path:
        # Write out seed IDs to a diagnostic vector.
        diagnostic_vector = gpkg_driver.Create(
            diagnostic_vector_path,
            0, 0, 0, gdal.GDT_Unknown)
        diagnostic_layer = diagnostic_vector.CreateLayer(
            'grouped_seeds', flow_dir_srs, ogr.wkbPoint)
        diagnostic_layer.CreateField(ogr.FieldDefn('seed_id', ogr.OFTInteger))
        diagnostic_layer.CreateField(ogr.FieldDefn('original_id', ogr.OFTInteger))
        diagnostic_layer.CreateField(ogr.FieldDefn('member_watersheds', ogr.OFTString))
        diagnostic_layer.CreateField(ogr.FieldDefn('seed_coords', ogr.OFTString))
        diagnostic_layer.CreateField(ogr.FieldDefn('upstream_seeds', ogr.OFTString))
        diagnostic_layer.CreateField(ogr.FieldDefn('downstream_seeds', ogr.OFTString))
        diagnostic_layer.StartTransaction()

    final_seed_ids = {}
    for seed, seed_id in seed_ids.items():
        original_reclassification = reclassification[seed_id]
        new_id = consolidated_reclassification[original_reclassification]
        final_seed_ids[seed] = new_id

        if diagnostic_vector_path:
            feature = ogr.Feature(diagnostic_layer.GetLayerDefn())
            point = shapely.geometry.Point(
                x_origin + seed[0] * x_pixelwidth + (x_pixelwidth / 2.),
                y_origin + seed[1] * y_pixelwidth + (y_pixelwidth / 2.))
            feature.SetGeometry(ogr.CreateGeometryFromWkb(point.wkb))
            feature.SetField('seed_id', new_id)
            feature.SetField('original_id', seed_id)
            feature.SetField('member_watersheds', ','.join([str(s) for s in sorted(seeds_to_watershed_membership_map[seed])]))
            feature.SetField('seed_coords', '(%s, %s)' % seed)
            try:
                feature.SetField('upstream_seeds', ','.join([str(s) for s in sorted(nested_fragments[seed])]))
            except KeyError:
                feature.SetField('upstream_seeds', '')

            try:
                feature.SetField('downstream_seeds', ','.join([str(s) for s in sorted(downstream_seeds[seed])]))
            except KeyError:
                feature.SetField('downstream_seeds', '')
            diagnostic_layer.CreateFeature(feature)

    if diagnostic_vector_path:
        diagnostic_layer.CommitTransaction()

    return final_seed_ids


def delineate_watersheds_d8(
        d8_flow_dir_raster_path_band, outflow_vector_path,
        target_fragments_vector_path, working_dir=None):
    if (d8_flow_dir_raster_path_band is not None and not
            _is_raster_path_band_formatted(d8_flow_dir_raster_path_band)):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                d8_flow_dir_raster_path_band))

    cdef int ws_id  # start indexing ws_id at 1

    flow_dir_info = pygeoprocessing.get_raster_info(
        d8_flow_dir_raster_path_band[0])
    source_gt = flow_dir_info['geotransform']
    flow_dir_srs = osr.SpatialReference()
    flow_dir_srs.ImportFromWkt(flow_dir_info['projection'])
    gpkg_driver = gdal.GetDriverByName('GPKG')

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

    seed_watersheds_python = split_vector_into_seeds(
        outflow_vector_path, d8_flow_dir_raster_path_band,
        write_diagnostic_vector=True, remove=False,
        working_dir=working_dir_path)

    seed_ids_python = group_seeds_into_fragments_d8(
        d8_flow_dir_raster_path_band, seed_watersheds_python,
        diagnostic_vector_path=os.path.join(working_dir_path, 'seed_grouping_diagnostics.gpkg'))

    LOGGER.info('Converting seed IDs to C++')
    cdef int seed_id
    cdef CoordinatePair seed
    cdef cmap[CoordinatePair, int] seed_ids
    cdef cmap[int, CoordinatePair] seed_id_to_seed
    for seed_tuple, seed_id in seed_ids_python.items():
        seed = CoordinatePair(seed_tuple[0], seed_tuple[1])
        seed_ids[seed] = seed_id
        seed_id_to_seed[seed_id] = seed

    LOGGER.info('Splitting seeds into their blocks.')
    cdef int block_index
    cdef int n_blocks = (
        ((flow_dir_n_cols // flow_dir_block_x_size) + 1) *
        ((flow_dir_n_rows // flow_dir_block_y_size) + 1))
    cdef cmap[int, cset[CoordinatePair]] seeds_in_block

    # Initialize the seeds_in_block data structure
    for block_index in range(n_blocks):
        seeds_in_block[block_index] = cset[CoordinatePair]()

    for seed_tuple, watersheds in seed_watersheds_python.items():
        # Determine the block index mathematically.  We only need to be able to
        # group pixels together, so the specific number used does not matter.
        seed = CoordinatePair(seed_tuple[0], seed_tuple[1])
        block_index = (
            (seed.first // flow_dir_block_x_size) +
            ((seed.second // flow_dir_block_y_size) * (flow_dir_n_cols // flow_dir_block_x_size)))
        if block_index > n_blocks:
            print 'block_index %s > %s' % (block_index, n_blocks)
        seeds_in_block[block_index].insert(seed)

    # create a scratch raster for writing out the fragments during iteration.
    # Because fragment IDs will start at 1, we can also use this as the mask
    # raster for polygonization.
    scratch_raster_path = os.path.join(
            working_dir_path, 'scratch_raster.tif')
    LOGGER.info('Creating new scratch raster at %s' % scratch_raster_path)

    cdef unsigned int no_watershed = (2**32)-1  # max value for UInt32
    pygeoprocessing.new_raster_from_base(
        d8_flow_dir_raster_path_band[0], scratch_raster_path,
        gdal.GDT_UInt32, [no_watershed], fill_value_list=[no_watershed],
        gtiff_creation_options=GTIFF_CREATION_OPTIONS)
    scratch_managed_raster = _ManagedRaster(scratch_raster_path, 1, 1)

    flow_dir_managed_raster = _ManagedRaster(d8_flow_dir_raster_path_band[0],
                                             d8_flow_dir_raster_path_band[1],
                                             0)  # Read-only

    LOGGER.info('Starting delineation from %s seeds', len(seed_watersheds_python))
    cdef cmap[int, cset[int]] nested_fragments
    cdef cset[int] nested_fragment_ids
    cdef cset[CoordinatePair] process_queue_set
    cdef queue[CoordinatePair] process_queue

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

                scratch_managed_raster.set(current_pixel.first,
                                           current_pixel.second, seed_id)

                for neighbor_index in range(8):
                    neighbor_pixel = CoordinatePair(
                        current_pixel.first + NEIGHBOR_COL[neighbor_index],
                        current_pixel.second + NEIGHBOR_ROW[neighbor_index])

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
                    if (D8_REVERSE_DIRECTION[neighbor_index] ==
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
                        pixel_visited = scratch_managed_raster.get(
                            neighbor_pixel.first, neighbor_pixel.second)
                        if pixel_visited == no_watershed:
                            process_queue.push(neighbor_pixel)
                            process_queue_set.insert(neighbor_pixel)

            nested_fragments[seed_id] = nested_fragment_ids
    scratch_managed_raster.close()  # flush the scratch raster.
    flow_dir_managed_raster.close()  # don't need this any longer.

    seeds_in_block.clear()

    LOGGER.info('Polygonizing fragments')
    scratch_raster = gdal.OpenEx(scratch_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    scratch_band = scratch_raster.GetRasterBand(1)
    scratch_nodata = scratch_band.GetNoDataValue()

    # Replace any nodata pixels with a value of 0
    # For some reason, blocks written out have a value of zero rather than nodata. The
    # rasterization mask considers valid pixels to be those with a nonzero value, so
    # all nodata pixels need to be converted to 0 to make this work.
    for block_info in pygeoprocessing.iterblocks((scratch_raster_path, 1), offset_only=True):
        block = scratch_band.ReadAsArray(**block_info)
        mask = block == scratch_nodata
        valid_pixels = block[mask]

        if valid_pixels.size > 0:  # skip blocks that don't have any nodata
            block[mask] = 0
            scratch_band.WriteArray(block, xoff=block_info['xoff'], yoff=block_info['yoff'])

    scratch_band.FlushCache()

    target_fragments_vector = gpkg_driver.Create(target_fragments_vector_path,
                                                 0, 0, 0, gdal.GDT_Unknown)
    if target_fragments_vector is None:
        raise RuntimeError(  # Because I frequently have this open in QGIS when I shouldn't.
            "Could not open target fragments vector for writing. Do you have "
            "access to this path?  Is the file open in another program?")

    # Create a spatial layer for the fragment geometries.
    # This layer only needs to know which fragments are upstream of a given fragment.
    # BTW, the same number of fragments are polygonized whether the layer is a
    # Polygon layer or a MultiPolygon layer.
    target_fragments_scratch_layer = target_fragments_vector.CreateLayer(
        'watershed_fragments_scratch', flow_dir_srs, ogr.wkbPolygon)
    target_fragments_scratch_layer.CreateField(ogr.FieldDefn('fragment_id', ogr.OFTInteger64))

    gdal.Polygonize(
        scratch_band,  # the source band to be analyzed
        scratch_band,  # the mask band indicating valid pixels
        target_fragments_scratch_layer,  # polygons are added to this layer
        0,  # field index of 'fragment_id' field.
        [],  # 8CONNECTED=8 sometimes creates invalid geometries.
        _make_polygonize_callback(LOGGER)
    )

    # Create a non-spatial database layer for storing all of the watershed
    # attributes without the geometries.  This is important because we need to
    # know the ws_id of each outlet feature.
    # TODO: don't include watersheds that are not represented in the fragments
    outflow_vector = gdal.OpenEx(outflow_vector_path)
    outflow_layer = outflow_vector.GetLayer()  # TODO: user-defined?
    target_fragments_watershed_attrs_layer = target_fragments_vector.CreateLayer(
        'watershed_attributes', geom_type=ogr.wkbNone,
        options=['ASPATIAL_VARIANT=OGR_ASPATIAL'])
    target_fragments_watershed_attrs_layer.CreateFields(outflow_layer.schema)
    target_fragments_watershed_attrs_layer.CreateField(
        ogr.FieldDefn('outflow_feature_id', ogr.OFTInteger))
    target_fragments_watershed_attrs_layer.StartTransaction()
    all_watersheds_in_seeds = set(itertools.chain(*seed_watersheds_python.values()))
    for outflow_feature in outflow_layer:
        # Skip any watesheds (represented by outflow layer FID) not in the
        # known set of watersheds.  This will exclude any outflow features
        # that are over nodata or outside the bounds of the raster.
        if outflow_feature.GetFID() not in all_watersheds_in_seeds:
            continue

        new_feature = ogr.Feature(
            target_fragments_watershed_attrs_layer.GetLayerDefn())
        for attr_name, attr_value in outflow_feature.items().items():
            new_feature.SetField(attr_name, attr_value)
        new_feature.SetField('outflow_feature_id', outflow_feature.GetFID())
        target_fragments_watershed_attrs_layer.CreateFeature(new_feature)
    target_fragments_watershed_attrs_layer.CommitTransaction()

    # Now need to write out the fragment copying.
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
        fragment_id = fragment_id_with_duplicates
        consolidated_feature.SetField('fragment_id', fragment_id)
        fragment_seed = seed_id_to_seed[fragment_id]
        consolidated_feature.SetField('upstream_fragments',
            ','.join([str(s) for s in sorted(nested_fragments[fragment_id])]))
        consolidated_feature.SetField('member_ws_ids',
            ','.join([str(s) for s in sorted(
                set(seed_watersheds_python[tuple(fragment_seed)]))]))
        consolidated_feature.SetGeometry(new_geometry)
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
            ','.join([str(s) for s in sorted(nested_fragments[fragment_id])]))
        consolidated_feature.SetField('member_ws_ids',
            ','.join([str(s) for s in sorted(set(seed_watersheds_python[fragment_seed]))]))
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

    seeds = split_vector_into_seeds(
        outflow_vector_path, d8_flow_dir_raster_path_band,
        working_dir=working_dir_path, remove=False, start_index=1)

    seeds_in_ws_id = collections.defaultdict(set)
    for seed_tuple, ws_id_set in seeds.items():
        for ws_id in ws_id_set:
            seeds_in_ws_id[ws_id].add(seed_tuple)

    seeds_in_ws_id = dict(seeds_in_ws_id)

    feature_count = len(seeds_in_ws_id)

    cdef int* reverse_flow = [4, 5, 6, 7, 0, 1, 2, 3]
    cdef int* neighbor_col = [1, 1, 0, -1, -1, -1, 0, 1]
    cdef int* neighbor_row = [0, -1, -1, -1, 0, 1, 1, 1]
    cdef queue[CoordinatePair] process_queue
    cdef cset[CoordinatePair] process_queue_set
    cdef CoordinatePair pixel_coords, neighbor_pixel, seed
    for ws_id, seeds_in_outlet in sorted(seeds_in_ws_id.items(), key=lambda x: x[0]):
        if len(seeds_in_outlet) == 0:
            LOGGER.info('Skipping watershed %s of %s, no valid seeds found.',
                        ws_id, feature_count)

        for seed_tuple in seeds_in_outlet:
            seed = CoordinatePair(seed_tuple[0], seed_tuple[1])
            process_queue.push(seed)
            process_queue_set.insert(seed)

        LOGGER.info('Delineating watershed %s of %s from %s pixels', ws_id,
                    feature_count, process_queue.size())

        scratch_managed_raster = _ManagedRaster(scratch_raster_path, 1, 1)
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

        # The FID of the source feature is the WS_ID minus 1
        source_feature = source_layer.GetFeature(ws_id-1)
        for field_name, field_value in source_feature.items().items():
            watershed_feature.SetField(field_name, field_value)
        watersheds_layer.SetFeature(watershed_feature)

    watersheds_layer = None
    watersheds_vector = None
    source_layer = None
    source_vector = None

    shutil.rmtree(working_dir_path)

