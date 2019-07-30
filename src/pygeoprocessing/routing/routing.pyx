# coding=UTF-8
# distutils: language=c++
# cython: language_level=3
"""
Provides PyGeprocessing Routing functionality.

Unless otherwise specified, all internal computation of rasters are done in
a float64 space. The only possible loss of precision could occur when an
incoming DEM type is an int64 type and values in that dem exceed 2^52 but GDAL
does not support int64 rasters so no precision loss is possible with a
float64.

D8 float direction conventions follow TauDEM where each flow direction
is encoded as:
     321
     4x0
     567
"""
import time
import os
import logging
import shutil
import tempfile

import numpy
import pygeoprocessing
from osgeo import gdal

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
from libcpp.stack cimport stack
from libcpp.deque cimport deque
from libcpp.set cimport set as cset

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())  # silence logging by default

# This module creates rasters with a memory xy block size of 2**BLOCK_BITS
cdef int BLOCK_BITS = 8

# Number of raster blocks to hold in memory at once per Managed Raster
cdef int MANAGED_RASTER_N_BLOCKS = 2**6

# these are the creation options that'll be used for all the rasters
DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS = ('GTiff', (
    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
    'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
    'BLOCKYSIZE=%d' % (1 << BLOCK_BITS)))

# if nodata is not defined for a float, it's a difficult choice. this number
# probably won't collide with anything ever created by humans
cdef double IMPROBABLE_FLOAT_NODATA = -1.23789789e29

# a pre-computed square root of 2 constant
cdef double SQRT2 = 1.4142135623730951
cdef double SQRT2_INV = 1.0 / 1.4142135623730951

# used to loop over neighbors and offset the x/y values as defined below
#  321
#  4x0
#  567
cdef int* NEIGHBOR_OFFSET_ARRAY = [
    1, 0,  # 0
    1, -1,  # 1
    0, -1,  # 2
    -1, -1,  # 3
    -1, 0,  # 4
    -1, 1,  # 5
    0, 1,  # 6
    1, 1  # 7
    ]

# this is used to calculate the opposite D8 direction interpreting the index
# as a D8 direction
cdef int* D8_REVERSE_DIRECTION = [4, 5, 6, 7, 0, 1, 2, 3]

# exposing stl::priority_queue so we can have all 3 template arguments so
# we can pass a different Compare functor
cdef extern from "<queue>" namespace "std" nogil:
    cdef cppclass priority_queue[T, Container, Compare]:
        priority_queue() except +
        priority_queue(priority_queue&) except +
        priority_queue(Container&)
        bint empty()
        void pop()
        void push(T&)
        size_t size()
        T& top()

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
        void clean(clist[pair[KEY_T,VAL_T]]&, int n_items)
        size_t size()

# this is the class type that'll get stored in the priority queue
cdef struct PixelType:
    double value  # pixel value
    int xi  # pixel x coordinate in the raster
    int yi  # pixel y coordinate in the raster
    int priority # for breaking ties if two `value`s are equal.

# this struct is used to record an intermediate flow pixel's last calculated
# direction and the flow accumulation value so far
cdef struct FlowPixelType:
    int xi
    int yi
    int last_flow_dir
    double value

# used to record x/y locations as needed
cdef struct CoordinateType:
    int xi
    int yi

# this ctype is used to store the block ID and the block buffer as one object
# inside Managed Raster
ctypedef pair[int, double*] BlockBufferPair

# this type is used to create a priority queue on the custom Pixel tpye
ctypedef priority_queue[
    PixelType, deque[PixelType], GreaterPixel] PitPriorityQueueType

# this queue is used to record flow directions
ctypedef queue[int] IntQueueType

# type used to store x/y coordinates and a queue to put them in
ctypedef queue[CoordinateType] CoordinateQueueType

# functor for priority queue of pixels
cdef cppclass GreaterPixel nogil:
    bint get "operator()"(PixelType& lhs, PixelType& rhs):
        # lhs is > than rhs if its value is greater or if it's equal if
        # the priority is >.
        if lhs.value > rhs.value:
            return 1
        if lhs.value == rhs.value:
            if lhs.priority > rhs.priority:
                return 1
        return 0

cdef int is_close(double x, double y):
    return abs(x-y) <= (1e-8+1e-05*abs(y))

# a class to allow fast random per-pixel access to a raster for both setting
# and reading pixels.
cdef class _ManagedRaster:
    cdef LRUCache[int, double*]* lru_cache
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
            self.raster_x_size + (self.block_xsize) - 1) // self.block_xsize
        self.block_ny = (
            self.raster_y_size + (self.block_ysize) - 1) // self.block_ysize

        self.lru_cache = new LRUCache[int, double*](MANAGED_RASTER_N_BLOCKS)
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
        cdef numpy.ndarray[double, ndim=2] block_array = numpy.empty(
            (self.block_ysize, self.block_xsize))
        cdef double *double_buffer
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
            double_buffer = deref(it).second
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

                for xi_copy in range(win_xsize):
                    for yi_copy in range(win_ysize):
                        block_array[yi_copy, xi_copy] = (
                            double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy])
                raster_band.WriteArray(
                    block_array[0:win_ysize, 0:win_xsize],
                    xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            inc(it)
        raster_band.FlushCache()
        raster_band = None
        raster = None

    cdef inline void set(self, int xi, int yi, double value):
        """Set the pixel at `xi,yi` to `value`."""
        if xi < 0 or xi >= self.raster_x_size:
            LOGGER.error("x out of bounds %s" % xi)
        if yi < 0 or yi >= self.raster_y_size:
            LOGGER.error("y out of bounds %s" % yi)
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        self.lru_cache.get(
            block_index)[
                ((yi & (self.block_ymod)) << self.block_xbits) +
                (xi & (self.block_xmod))] = value
        if self.write_mode:
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr == self.dirty_blocks.end():
                self.dirty_blocks.insert(block_index)

    cdef inline double get(self, int xi, int yi):
        """Return the value of the pixel at `xi,yi`."""
        if xi < 0 or xi >= self.raster_x_size:
            LOGGER.error("x out of bounds %s" % xi)
        if yi < 0 or yi >= self.raster_y_size:
            LOGGER.error("y out of bounds %s" % yi)
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        return self.lru_cache.get(
            block_index)[
                ((yi & (self.block_ymod)) << self.block_xbits) +
                (xi & (self.block_xmod))]

    cdef void _load_block(self, int block_index) except *:
        cdef int block_xi = block_index % self.block_nx
        cdef int block_yi = block_index // self.block_nx

        # we need the offsets to subtract from global indexes for cached array
        cdef int xoff = block_xi << self.block_xbits
        cdef int yoff = block_yi << self.block_ybits

        cdef int xi_copy, yi_copy
        cdef numpy.ndarray[double, ndim=2] block_array
        cdef double *double_buffer
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
            win_ysize=win_ysize).astype(numpy.float64)
        raster_band = None
        raster = None
        double_buffer = <double*>PyMem_Malloc(
            (sizeof(double) << self.block_xbits) * win_ysize)
        for xi_copy in range(win_xsize):
            for yi_copy in range(win_ysize):
                double_buffer[(yi_copy << self.block_xbits)+xi_copy] = (
                    block_array[yi_copy, xi_copy])
        self.lru_cache.put(
            <int>block_index, <double*>double_buffer, removed_value_list)

        if self.write_mode:
            raster = gdal.OpenEx(
                self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
            raster_band = raster.GetRasterBand(self.band_id)

        block_array = numpy.empty(
            (self.block_ysize, self.block_xsize), dtype=numpy.double)
        while not removed_value_list.empty():
            # write the changed value back if desired
            double_buffer = removed_value_list.front().second

            if self.write_mode:
                block_index = removed_value_list.front().first

                # write back the block if it's dirty
                dirty_itr = self.dirty_blocks.find(block_index)
                if dirty_itr != self.dirty_blocks.end():
                    self.dirty_blocks.erase(dirty_itr)

                    block_xi = block_index % self.block_nx
                    block_yi = block_index // self.block_nx

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

                    for xi_copy in range(win_xsize):
                        for yi_copy in range(win_ysize):
                            block_array[yi_copy, xi_copy] = double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy]
                    raster_band.WriteArray(
                        block_array[0:win_ysize, 0:win_xsize],
                        xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            removed_value_list.pop_front()

        if self.write_mode:
            raster_band = None
            raster = None

    cdef void flush(self) except *:
        cdef clist[BlockBufferPair] removed_value_list
        cdef double *double_buffer
        cdef cset[int].iterator dirty_itr
        cdef int block_index, block_xi, block_yi
        cdef int xoff, yoff, win_xsize, win_ysize

        self.lru_cache.clean(removed_value_list, self.lru_cache.size())

        raster_band = None
        if self.write_mode:
            raster = gdal.OpenEx(
                self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
            raster_band = raster.GetRasterBand(self.band_id)

        block_array = numpy.empty(
            (self.block_ysize, self.block_xsize), dtype=numpy.double)
        while not removed_value_list.empty():
            # write the changed value back if desired
            double_buffer = removed_value_list.front().second

            if self.write_mode:
                block_index = removed_value_list.front().first

                # write back the block if it's dirty
                dirty_itr = self.dirty_blocks.find(block_index)
                if dirty_itr != self.dirty_blocks.end():
                    self.dirty_blocks.erase(dirty_itr)

                    block_xi = block_index % self.block_nx
                    block_yi = block_index // self.block_nx

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

                    for xi_copy in range(win_xsize):
                        for yi_copy in range(win_ysize):
                            block_array[yi_copy, xi_copy] = double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy]
                    raster_band.WriteArray(
                        block_array[0:win_ysize, 0:win_xsize],
                        xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            removed_value_list.pop_front()

        if self.write_mode:
            raster_band = None
            raster = None


def _generate_read_bounds(offset_dict, raster_x_size, raster_y_size):
    """Helper function to expand GDAL memory block read bound by 1 pixel.

    This function is used in the context of reading a memory block on a GDAL
    raster plus an additional 1 pixel boundary if it fits into an existing
    numpy array of size (2+offset_dict['y_size'], 2+offset_dict['x_size']).

    Parameters:
        offset_dict (dict): dictionary that has values for 'win_xsize',
            'win_ysize', 'xoff', and 'yoff' to describe the bounding box
            to read from the raster.
        raster_x_size, raster_y_size (int): these are the global x/y sizes
            of the raster that's being read.

    Returns:
        (xa, xb, ya, yb) (tuple of int): bounds that can be used to slice a
            numpy array of size
                (2+offset_dict['y_size'], 2+offset_dict['x_size'])
        modified_offset_dict (dict): a copy of `offset_dict` with the
            `win_*size` keys expanded if the modified bounding box will still
            fit on the array.
    """
    xa = 1
    xb = -1
    ya = 1
    yb = -1
    target_offset_dict = offset_dict.copy()
    if offset_dict['xoff'] > 0:
        xa = None
        target_offset_dict['xoff'] -= 1
        target_offset_dict['win_xsize'] += 1
    if offset_dict['yoff'] > 0:
        ya = None
        target_offset_dict['yoff'] -= 1
        target_offset_dict['win_ysize'] += 1
    if (offset_dict['xoff'] + offset_dict['win_xsize'] < raster_x_size):
        xb = None
        target_offset_dict['win_xsize'] += 1
    if (offset_dict['yoff'] + offset_dict['win_ysize'] < raster_y_size):
        yb = None
        target_offset_dict['win_ysize'] += 1
    return (xa, xb, ya, yb), target_offset_dict


def fill_pits(
        dem_raster_path_band, target_filled_dem_raster_path,
        working_dir=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Fill the pits in a DEM.

        This function defines pits as hydrologically connected regions that do
        not drain to the edge of the raster or a nodata pixel. After the call
        pits are filled to the height of the lowest pour point.

    Parameters:
        dem_raster_path_band (tuple): a path, band number tuple indicating the
            DEM calculate flow direction.
        target_filled_dem_raster_path (string): path the pit filled dem,
            that's created by a call to this function. It is functionally a
            copy of `dem_raster_path_band[0]` with the pit pixels raised to
            the pour point. For runtime efficiency, this raster is tiled and
            its blocksize is set to (1<<BLOCK_BITS, 1<<BLOCK_BITS) even if
            `dem_raster_path_band[0]` was not tiled or a different block size.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run. If this directory doesn't exist
            it is created by this call. If None, a temporary directory is
            created by tempdir.mkdtemp which is removed after the function
            call completes successfully.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None.
    """
    # These variables are used to iterate over the DEM using `iterblock`
    # indexes, a numpy.float64 type is used since we need to statically cast
    # and it's the most complex numerical type and will be compatible without
    # data loss for any lower type that might be used in
    # `dem_raster_path_band[0]`.
    cdef numpy.ndarray[numpy.float64_t, ndim=2] dem_buffer_array
    cdef int win_ysize, win_xsize, xoff, yoff

    # the _root variables remembers the pixel index where the plateau/pit
    # region was first detected when iterating over the DEM.
    cdef int xi_root, yi_root

    # these variables are used as pixel or neighbor indexes. where _q
    # represents a value out of a queue, and _n is related to a neighbor pixel
    cdef int i_n, xi, yi, xi_q, yi_q, xi_n, yi_n

    # these are booleans used to remember the condition that caused a loop
    # to terminate, though downhill and nodata are equivalent for draining,
    # i keep them separate for cognitive readability.
    cdef int downhill_neighbor, nodata_neighbor, downhill_drain, nodata_drain

    # `search_queue` is used to grow a flat region searching for a pour point
    # to determine if region is plateau or, in the absence of a pour point,
    # a pit.
    cdef queue[CoordinateType] search_queue

    # `fill_queue` is used after a region is identified as a pit and its pour
    # height is determined to fill the pit up to the pour height
    cdef queue[CoordinateType] fill_queue

    # a pixel pointer is used to push to a priority queue. it remembers its
    # pixel value, x/y index, and an optional priority value to order if
    # heights are equal.
    cdef PixelType pixel

    # this priority queue is used to iterate over pit pixels in increasing
    # height, to search for the lowest pour point.
    cdef PitPriorityQueueType pit_queue

    # properties of the parallel rasters
    cdef int raster_x_size, raster_y_size, n_x_blocks

    # variables to remember heights of DEM
    cdef double center_val, dem_nodata, fill_height

    # used to uniquely identify each flat/pit region encountered in the
    # algorithm, it's written into the mask rasters to indicate which pixels
    # have already been processed
    cdef int feature_id

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    # determine dem nodata in the working type, or set an improbable value
    # if one can't be determined
    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_path_band[0])
    base_nodata = dem_raster_info['nodata'][dem_raster_path_band[1]-1]
    if base_nodata is not None:
        # cast to a float64 since that's our operating array type
        dem_nodata = numpy.float64(base_nodata)
    else:
        # pick some very improbable value since it's hard to deal with NaNs
        dem_nodata = IMPROBABLE_FLOAT_NODATA

    # these are used to determine if a sample is within the raster
    raster_x_size, raster_y_size = dem_raster_info['raster_size']

    # this is the nodata value for all the flat region and pit masks
    mask_nodata = 0

    # set up the working dir for the mask rasters
    try:
        if working_dir is not None:
            os.makedirs(working_dir)
    except OSError:
        pass
    working_dir_path = tempfile.mkdtemp(
        dir=working_dir, prefix='fill_pits_%s_' % time.strftime(
            '%Y-%m-%d_%H_%M_%S', time.gmtime()))

    # this raster is used to keep track of what pixels have been searched for
    # a plateau or pit. if a pixel is set, it means it is part of a locally
    # undrained area
    flat_region_mask_path = os.path.join(
        working_dir_path, 'flat_region_mask.tif')
    n_x_blocks = raster_x_size >> BLOCK_BITS + 1

    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], flat_region_mask_path, gdal.GDT_Byte,
        [mask_nodata], fill_value_list=[mask_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    flat_region_mask_managed_raster = _ManagedRaster(
        flat_region_mask_path, 1, 1)

    # this raster will have the value of 'feature_id' set to it if it has
    # been searched as part of the search for a pour point for pit number
    # `feature_id`
    pit_mask_path = os.path.join(working_dir_path, 'pit_mask.tif')
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], pit_mask_path, gdal.GDT_Int32,
        [mask_nodata], fill_value_list=[mask_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    pit_mask_managed_raster = _ManagedRaster(
        pit_mask_path, 1, 1)

    # copy the base DEM to the target and set up for writing
    raster_driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
    base_dem_raster = gdal.OpenEx(dem_raster_path_band[0], gdal.OF_RASTER)
    raster_driver.CreateCopy(
        target_filled_dem_raster_path, base_dem_raster,
        options=raster_driver_creation_tuple[1])
    filled_dem_managed_raster = _ManagedRaster(
        target_filled_dem_raster_path, dem_raster_path_band[1], 1)

    # feature_id will start at 1 since the mask nodata is 0.
    feature_id = 0

    # this outer loop searches for a pixel that is locally undrained
    for offset_dict in pygeoprocessing.iterblocks(
            dem_raster_path_band, offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('%.1f%% complete', 100.0 * current_pixel / <float>(
                raster_x_size * raster_y_size))

        # make a buffer big enough to capture block and boundaries around it
        dem_buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.float64)
        dem_buffer_array[:] = dem_nodata

        # attempt to expand read block by a pixel boundary
        (xa, xb, ya, yb), modified_offset_dict = _generate_read_bounds(
            offset_dict, raster_x_size, raster_y_size)
        filled_dem_managed_raster.flush()
        target_dem_raster = gdal.OpenEx(target_filled_dem_raster_path)
        target_dem_band = target_dem_raster.GetRasterBand(1)
        dem_buffer_array[ya:yb, xa:xb] = target_dem_band.ReadAsArray(
                **modified_offset_dict).astype(numpy.float64)
        target_dem_band = None
        target_dem_raster = None

        # search block for locally undrained pixels
        for yi in range(1, win_ysize+1):
            for xi in range(1, win_xsize+1):
                center_val = dem_buffer_array[yi, xi]
                if is_close(center_val, dem_nodata):
                    continue

                # this value is set in case it turns out to be the root of a
                # pit, we'll start the fill from this pixel in the last phase
                # of the algorithm
                xi_root = xi-1+xoff
                yi_root = yi-1+yoff

                if flat_region_mask_managed_raster.get(
                        xi_root, yi_root) != mask_nodata:
                    # already been searched
                    continue

                # search neighbors for downhill or nodata
                downhill_neighbor = 0
                nodata_neighbor = 0

                for i_n in range(8):
                    xi_n = xi_root+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                    yi_n = yi_root+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                    if (xi_n < 0 or xi_n >= raster_x_size or
                            yi_n < 0 or yi_n >= raster_y_size):
                        # it'll drain off the edge of the raster
                        nodata_neighbor = 1
                        break
                    n_height = filled_dem_managed_raster.get(xi_n, yi_n)
                    if is_close(n_height, dem_nodata):
                        # it'll drain to nodata
                        nodata_neighbor = 1
                        break
                    if n_height < center_val:
                        # it'll drain downhill
                        downhill_neighbor = 1
                        break

                if downhill_neighbor or nodata_neighbor:
                    # it drains, so skip
                    continue

                # otherwise, this pixel doesn't drain locally, search to see
                # if it's a pit or plateau
                search_queue.push(CoordinateType(xi_root, yi_root))
                flat_region_mask_managed_raster.set(xi_root, yi_root, 1)
                downhill_drain = 0
                nodata_drain = 0

                # this loop does a BFS starting at this pixel to all pixels
                # of the same height. the _drain variables are used to
                # remember if a drain was encountered. it is preferable to
                # search the whole region even if a drain is encountered, so
                # it can be entirely marked as processed and not re-accessed
                # on later iterations
                while not search_queue.empty():
                    xi_q = search_queue.front().xi
                    yi_q = search_queue.front().yi
                    search_queue.pop()

                    for i_n in range(8):
                        xi_n = xi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = yi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            nodata_drain = 1
                            continue
                        n_height = filled_dem_managed_raster.get(
                            xi_n, yi_n)
                        if is_close(n_height, dem_nodata):
                            nodata_drain = 1
                            continue
                        if n_height < center_val:
                            downhill_drain = 1
                            continue
                        if is_close(n_height, center_val) and (
                                flat_region_mask_managed_raster.get(
                                    xi_n, yi_n) == mask_nodata):
                            # only grow if it's at the same level and not
                            # previously visited
                            search_queue.push(
                                CoordinateType(xi_n, yi_n))
                            flat_region_mask_managed_raster.set(
                                xi_n, yi_n, 1)

                if not downhill_drain and not nodata_drain:
                    # entire region was searched with no drain, do a fill
                    pixel = PixelType(
                        center_val, xi_root, yi_root, (
                            n_x_blocks * (yi_root >> BLOCK_BITS) +
                            xi_root >> BLOCK_BITS))
                    feature_id += 1
                    pit_mask_managed_raster.set(
                        xi_root, yi_root, feature_id)
                    pit_queue.push(pixel)

                # this loop visits pixels in increasing height order, so the
                # first non-processed pixel that's < pixel.height or nodata
                # will be the lowest pour point
                pour_point = 0
                fill_height = dem_nodata
                while not pit_queue.empty():
                    pixel = pit_queue.top()
                    pit_queue.pop()
                    xi_q = pixel.xi
                    yi_q = pixel.yi
                    # this is the potential fill height if pixel is pour point
                    fill_height = pixel.value

                    for i_n in range(8):
                        xi_n = xi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = yi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            # drain off the edge of the raster
                            pour_point = 1
                            break

                        if pit_mask_managed_raster.get(
                                xi_n, yi_n) == feature_id:
                            # this cell has already been processed
                            continue

                        # mark as visited in the search for pour point
                        pit_mask_managed_raster.set(
                            xi_n, yi_n, feature_id)

                        n_height = filled_dem_managed_raster.get(xi_n, yi_n)
                        if is_close(n_height, dem_nodata) or (
                                n_height < fill_height):
                            # we encounter a neighbor not processed that is
                            # lower than the current pixel or nodata
                            pour_point = 1
                            break

                        # push onto queue, set the priority to be the block
                        # index
                        pixel = PixelType(
                            n_height, xi_n, yi_n, (
                                n_x_blocks * (yi_n >> BLOCK_BITS) +
                                xi_n >> BLOCK_BITS))
                        pit_queue.push(pixel)

                    if pour_point:
                        # found a pour point, clear the queue
                        pit_queue = PitPriorityQueueType()

                        # start from original pit seed rather than pour point
                        # this way we can stop filling when we reach a height
                        # equal to fill_height instead of potentially
                        # traversing a plateau area and needing to
                        # differentiate the pixels on the inside of the pit
                        # and the outside.
                        fill_queue.push(CoordinateType(xi_root, yi_root))
                        filled_dem_managed_raster.set(
                            xi_root, yi_root, fill_height)

                # this loop does a BFS to set all DEM pixels to `fill_height`
                while not fill_queue.empty():
                    xi_q = fill_queue.front().xi
                    yi_q = fill_queue.front().yi
                    fill_queue.pop()

                    for i_n in range(8):
                        xi_n = xi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = yi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            continue

                        if filled_dem_managed_raster.get(
                                xi_n, yi_n) < fill_height:
                            filled_dem_managed_raster.set(
                                xi_n, yi_n, fill_height)
                            fill_queue.push(CoordinateType(xi_n, yi_n))

    pit_mask_managed_raster.close()
    flat_region_mask_managed_raster.close()
    shutil.rmtree(working_dir_path)
    LOGGER.info('%.1f%% complete', 100.0)


def flow_dir_d8(
        dem_raster_path_band, target_flow_dir_path,
        working_dir=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """D8 flow direction.

    Parameters:
        dem_raster_path_band (tuple): a path, band number tuple indicating the
            DEM calculate flow direction. This DEM must not have hydrological
            pits or else the target flow direction is undefined.
        target_flow_dir_path (string): path to a byte raster created by this
            call of same dimensions as `dem_raster_path_band` that has a value
            indicating the direction of downhill flow. Values are defined as
            pointing to one of the eight neighbors with the following
            convention:

                321
                4x0
                567

        working_dir (string): If not None, indicates where temporary files
            should be created during this run. If this directory doesn't exist
            it is created by this call.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None.
    """
    # These variables are used to iterate over the DEM using `iterblock`
    # indexes, a numpy.float64 type is used since we need to statically cast
    # and it's the most complex numerical type and will be compatible without
    # data loss for any lower type that might be used in
    # `dem_raster_path_band[0]`.
    cdef numpy.ndarray[numpy.float64_t, ndim=2] dem_buffer_array
    cdef int win_ysize, win_xsize, xoff, yoff

    # the _root variables remembers the pixel index where the plateau/pit
    # region was first detected when iterating over the DEM.
    cdef int xi_root, yi_root

    # these variables are used as pixel or neighbor indexes. where _q
    # represents a value out of a queue, and _n is related to a neighbor pixel
    cdef int i_n, xi, yi, xi_q, yi_q, xi_n, yi_n

    # these are used to recall the local and neighbor heights of pixels
    cdef double root_height, n_height, dem_nodata

    # these are used to track the distance to the drain when we encounter a
    # plateau to route to the shortest path to the drain
    cdef double drain_distance, n_drain_distance

    # this remembers is flow was diagonal in case there is a straight
    # flow that could trump it
    cdef int diagonal_nodata

    # `search_queue` is used to grow a flat region searching for a drain
    # of a plateau
    cdef queue[CoordinateType] search_queue

    # `drain_queue` is used after a plateau drain is defined and iterates
    # until the entire plateau is drained, `nodata_drain_queue` are for
    # the case where the plateau is only drained by nodata pixels
    cdef CoordinateQueueType drain_queue, nodata_drain_queue

    # this queue is used to remember the flow directions of nodata pixels in
    # a plateau in case no other valid drain was found
    cdef queue[int] nodata_flow_dir_queue

    # properties of the parallel rasters
    cdef int raster_x_size, raster_y_size

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    # determine dem nodata in the working type, or set an improbable value
    # if one can't be determined
    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_path_band[0])
    base_nodata = dem_raster_info['nodata'][dem_raster_path_band[1]-1]
    if base_nodata is not None:
        # cast to a float64 since that's our operating array type
        dem_nodata = numpy.float64(base_nodata)
    else:
        # pick some very improbable value since it's hard to deal with NaNs
        dem_nodata = IMPROBABLE_FLOAT_NODATA

    # these are used to determine if a sample is within the raster
    raster_x_size, raster_y_size = dem_raster_info['raster_size']

    # this is the nodata value for all the flat region and pit masks
    mask_nodata = 0

    # set up the working dir for the mask rasters
    try:
        if working_dir is not None:
            os.makedirs(working_dir)
    except OSError:
        pass
    working_dir_path = tempfile.mkdtemp(
        dir=working_dir, prefix='flow_dir_d8_%s_' % time.strftime(
            '%Y-%m-%d_%H_%M_%S', time.gmtime()))

    # this raster is used to keep track of what pixels have been searched for
    # a plateau. if a pixel is set, it means it is part of a locally
    # undrained area
    flat_region_mask_path = os.path.join(
        working_dir_path, 'flat_region_mask.tif')
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], flat_region_mask_path, gdal.GDT_Byte,
        [mask_nodata], fill_value_list=[mask_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    flat_region_mask_managed_raster = _ManagedRaster(
        flat_region_mask_path, 1, 1)

    flow_dir_nodata = 128
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], target_flow_dir_path, gdal.GDT_Byte,
        [flow_dir_nodata], fill_value_list=[flow_dir_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    flow_dir_managed_raster = _ManagedRaster(target_flow_dir_path, 1, 1)

    # this creates a raster that's used for a dynamic programming solution to
    # shortest path to the drain for plateaus. the raster is filled with
    # raster_x_size * raster_y_size as a distance that's greater than the
    # longest plateau drain distance possible for this raster.
    plateau_distance_path = os.path.join(
        working_dir_path, 'plateau_distance.tif')
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], plateau_distance_path, gdal.GDT_Float64,
        [-1], fill_value_list=[raster_x_size * raster_y_size],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    plateau_distance_managed_raster = _ManagedRaster(
        plateau_distance_path, 1, 1)

    # this raster is for random access of the DEM

    compatable_dem_raster_path_band = None
    dem_block_xsize, dem_block_ysize = dem_raster_info['block_size']
    if (dem_block_xsize & (dem_block_xsize - 1) != 0) or (
            dem_block_ysize & (dem_block_ysize - 1) != 0):
        LOGGER.warning("dem is not a power of 2, creating a copy that is.")
        compatable_dem_raster_path_band = (
            os.path.join(working_dir_path, 'compatable_dem.tif'),
            dem_raster_path_band[1])
        raster_driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
        dem_raster = gdal.OpenEx(dem_raster_path_band[0], gdal.OF_RASTER)
        raster_driver.CreateCopy(
            compatable_dem_raster_path_band[0], dem_raster,
            options=raster_driver_creation_tuple[1])
        dem_raster = None
        LOGGER.info("compatible dem complete")
    else:
        compatable_dem_raster_path_band = dem_raster_path_band
    dem_managed_raster = _ManagedRaster(
        compatable_dem_raster_path_band[0],
        compatable_dem_raster_path_band[1], 0)

    # and this raster is for efficient block-by-block reading of the dem
    dem_raster = gdal.OpenEx(
        compatable_dem_raster_path_band[0], gdal.OF_RASTER)
    dem_band = dem_raster.GetRasterBand(compatable_dem_raster_path_band[1])

    # this outer loop searches for a pixel that is locally undrained
    for offset_dict in pygeoprocessing.iterblocks(
            compatable_dem_raster_path_band, offset_only=True,
            largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('%.1f%% complete', 100.0 * current_pixel / <float>(
                raster_x_size * raster_y_size))

        # make a buffer big enough to capture block and boundaries around it
        dem_buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.float64)
        dem_buffer_array[:] = dem_nodata

        # attempt to expand read block by a pixel boundary
        (xa, xb, ya, yb), modified_offset_dict = _generate_read_bounds(
            offset_dict, raster_x_size, raster_y_size)
        dem_buffer_array[ya:yb, xa:xb] = dem_band.ReadAsArray(
                **modified_offset_dict).astype(numpy.float64)

        # ensure these are set for the complier
        xi_n = -1
        yi_n = -1

        # search block for to set flow direction
        for yi in range(1, win_ysize+1):
            for xi in range(1, win_xsize+1):
                root_height = dem_buffer_array[yi, xi]
                if is_close(root_height, dem_nodata):
                    continue

                # this value is set in case it turns out to be the root of a
                # pit, we'll start the fill from this pixel in the last phase
                # of the algorithm
                xi_root = xi-1+xoff
                yi_root = yi-1+yoff

                if flow_dir_managed_raster.get(
                        xi_root, yi_root) != flow_dir_nodata:
                    # already been defined
                    continue

                # initialize variables to indicate the largest slope_dir is
                # undefined, the largest slope seen so far is flat, and the
                # largest nodata is at least a diagonal away
                largest_slope_dir = -1
                largest_slope = 0.0

                for i_n in range(8):
                    xi_n = xi+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                    yi_n = yi+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                    n_height = dem_buffer_array[yi_n, xi_n]
                    if is_close(n_height, dem_nodata):
                        continue
                    n_slope = root_height - n_height
                    if i_n & 1:
                        # if diagonal, adjust the slope
                        n_slope *= SQRT2_INV
                    if n_slope > largest_slope:
                        largest_slope_dir = i_n
                        largest_slope = n_slope

                if largest_slope_dir >= 0:
                    # define flow dir and move on
                    flow_dir_managed_raster.set(
                        xi_root, yi_root, largest_slope_dir)
                    continue

                # otherwise, this pixel doesn't drain locally, so it must
                # be a plateau, search for the drains of the plateau
                search_queue.push(CoordinateType(xi_root, yi_root))
                flat_region_mask_managed_raster.set(xi_root, yi_root, 1)

                # this loop does a BFS starting at this pixel to all pixels
                # of the same height. if a drain is encountered, it is pushed
                # on a queue for later processing.

                while not search_queue.empty():
                    xi_q = search_queue.front().xi
                    yi_q = search_queue.front().yi
                    search_queue.pop()

                    largest_slope_dir = -1
                    largest_slope = 0.0
                    diagonal_nodata = 1
                    for i_n in range(8):
                        xi_n = xi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = yi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]

                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            n_height = dem_nodata
                        else:
                            n_height = dem_managed_raster.get(xi_n, yi_n)
                        if is_close(n_height, dem_nodata):
                            if diagonal_nodata and largest_slope == 0.0:
                                largest_slope_dir = i_n
                                diagonal_nodata = i_n & 1
                            continue
                        n_slope = root_height - n_height
                        if n_slope < 0:
                            continue
                        if n_slope == 0.0:
                            if flat_region_mask_managed_raster.get(
                                    xi_n, yi_n) == mask_nodata:
                                # only grow if it's at the same level and not
                                # previously visited
                                search_queue.push(CoordinateType(xi_n, yi_n))
                                flat_region_mask_managed_raster.set(
                                    xi_n, yi_n, 1)
                            continue
                        if i_n & 1:
                            n_slope *= SQRT2_INV
                        if n_slope > largest_slope:
                            largest_slope = n_slope
                            largest_slope_dir = i_n

                    if largest_slope_dir >= 0:
                        if largest_slope > 0.0:
                            # regular downhill pixel
                            flow_dir_managed_raster.set(
                                xi_q, yi_q, largest_slope_dir)
                            plateau_distance_managed_raster.set(
                                xi_q, yi_q, 0.0)
                            drain_queue.push(CoordinateType(xi_q, yi_q))
                        else:
                            # must be a nodata drain, save on queue for later
                            nodata_drain_queue.push(
                                CoordinateType(xi_q, yi_q))
                            nodata_flow_dir_queue.push(largest_slope_dir)

                # if there's no downhill drains, try the nodata drains
                if drain_queue.empty():
                    # push the nodata drain queue over to the drain queue
                    # and set all the flow directions on the nodata drain
                    # pixels
                    while not nodata_drain_queue.empty():
                        xi_q = nodata_drain_queue.front().xi
                        yi_q = nodata_drain_queue.front().yi
                        flow_dir_managed_raster.set(
                            xi_q, yi_q, nodata_flow_dir_queue.front())
                        plateau_distance_managed_raster.set(xi_q, yi_q, 0.0)
                        drain_queue.push(nodata_drain_queue.front())
                        nodata_flow_dir_queue.pop()
                        nodata_drain_queue.pop()
                else:
                    # clear the nodata drain queues
                    nodata_flow_dir_queue = IntQueueType()
                    nodata_drain_queue = CoordinateQueueType()

                # this loop does a BFS from the plateau drain to any other
                # neighboring undefined pixels
                while not drain_queue.empty():
                    xi_q = drain_queue.front().xi
                    yi_q = drain_queue.front().yi
                    drain_queue.pop()

                    drain_distance = plateau_distance_managed_raster.get(
                        xi_q, yi_q)

                    for i_n in range(8):
                        xi_n = xi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = yi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            continue

                        n_drain_distance = drain_distance + (
                            SQRT2 if i_n & 1 else 1.0)

                        if dem_managed_raster.get(
                                xi_n, yi_n) == root_height and (
                                plateau_distance_managed_raster.get(
                                    xi_n, yi_n) > n_drain_distance):
                            # neighbor is at same level and has longer drain
                            # flow path than current
                            flow_dir_managed_raster.set(
                                xi_n, yi_n, D8_REVERSE_DIRECTION[i_n])
                            plateau_distance_managed_raster.set(
                                xi_n, yi_n, n_drain_distance)
                            drain_queue.push(CoordinateType(xi_n, yi_n))
    dem_band = None
    dem_raster = None
    flow_dir_managed_raster.close()
    flat_region_mask_managed_raster.close()
    dem_managed_raster.close()
    plateau_distance_managed_raster.close()
    shutil.rmtree(working_dir_path)
    LOGGER.info('%.1f%% complete', 100.0)


def flow_accumulation_d8(
        flow_dir_raster_path_band, target_flow_accum_raster_path,
        weight_raster_path_band=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """D8 flow accumulation.

    Parameters:
        flow_dir_raster_path_band (tuple): a path, band number tuple
            for a flow accumulation raster whose pixels indicate the flow
            out of a pixel in one of 8 directions in the following
            configuration:
                321
                4x0
                567
        target_flow_accum_raster_path (string): path to flow
            accumulation raster created by this call. After this call, the
            value of each pixel will be 1 plus the number of upstream pixels
            that drain to that pixel. Note the target type of this raster
            is a 64 bit float so there is minimal risk of overflow and the
            possibility of handling a float dtype in
            `weight_raster_path_band`.
        weight_raster_path_band (tuple): optional path and band number to a
            raster that will be used as the per-pixel flow accumulation
            weight. If `None`, 1 is the default flow accumulation weight.
            This raster must be the same dimensions as
            `flow_dir_mfd_raster_path_band`.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None.
    """
    # These variables are used to iterate over the DEM using `iterblock`
    # indexes, a numpy.float64 type is used since we need to statically cast
    # and it's the most complex numerical type and will be compatible without
    # data loss for any lower type that might be used in
    # `dem_raster_path_band[0]`.
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] flow_dir_buffer_array
    cdef int win_ysize, win_xsize, xoff, yoff

    # the _root variables remembers the pixel index where the plateau/pit
    # region was first detected when iterating over the DEM.
    cdef int xi_root, yi_root

    # these variables are used as pixel or neighbor indexes.
    # _n is related to a neighbor pixel
    cdef int i_n, xi, yi, xi_n, yi_n

    # used to hold flow direction values
    cdef int flow_dir, upstream_flow_dir, flow_dir_nodata

    # used as a holder variable to account for upstream flow
    cdef double upstream_flow_accum

    # this value is used to store the current weight which might be 1 or
    # come from a predefined flow accumulation weight raster
    cdef double weight_val
    cdef double weight_nodata = IMPROBABLE_FLOAT_NODATA  # set to something

    # `search_stack` is used to walk upstream to calculate flow accumulation
    # values
    cdef stack[FlowPixelType] search_stack
    cdef FlowPixelType flow_pixel

    # properties of the parallel rasters
    cdef int raster_x_size, raster_y_size

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    if not _is_raster_path_band_formatted(flow_dir_raster_path_band):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                flow_dir_raster_path_band))
    if weight_raster_path_band and not _is_raster_path_band_formatted(
            weight_raster_path_band):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                weight_raster_path_band))

    flow_accum_nodata = IMPROBABLE_FLOAT_NODATA
    pygeoprocessing.new_raster_from_base(
        flow_dir_raster_path_band[0], target_flow_accum_raster_path,
        gdal.GDT_Float64, [flow_accum_nodata],
        fill_value_list=[flow_accum_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    flow_accum_managed_raster = _ManagedRaster(
        target_flow_accum_raster_path, 1, 1)

    flow_dir_managed_raster = _ManagedRaster(
        flow_dir_raster_path_band[0], flow_dir_raster_path_band[1], 0)
    flow_dir_raster = gdal.OpenEx(
        flow_dir_raster_path_band[0], gdal.OF_RASTER)
    flow_dir_band = flow_dir_raster.GetRasterBand(
        flow_dir_raster_path_band[1])

    cdef _ManagedRaster weight_raster = None
    if weight_raster_path_band:
        weight_raster = _ManagedRaster(
            weight_raster_path_band[0], weight_raster_path_band[1], 0)
        raw_weight_nodata = pygeoprocessing.get_raster_info(
            weight_raster_path_band[0])['nodata'][
                weight_raster_path_band[1]-1]
        if raw_weight_nodata is not None:
            weight_nodata = raw_weight_nodata

    flow_dir_raster_info = pygeoprocessing.get_raster_info(
        flow_dir_raster_path_band[0])
    raster_x_size, raster_y_size = flow_dir_raster_info['raster_size']

    tmp_flow_dir_nodata = flow_dir_raster_info['nodata'][
        flow_dir_raster_path_band[1]-1]
    if tmp_flow_dir_nodata is None:
        flow_dir_nodata = 128
    else:
        flow_dir_nodata = tmp_flow_dir_nodata

    # this outer loop searches for a pixel that is locally undrained
    for offset_dict in pygeoprocessing.iterblocks(
            flow_dir_raster_path_band, offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('%.1f%% complete', 100.0 * current_pixel / <float>(
                raster_x_size * raster_y_size))

        # make a buffer big enough to capture block and boundaries around it
        flow_dir_buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.uint8)
        flow_dir_buffer_array[:] = flow_dir_nodata

        # attempt to expand read block by a pixel boundary
        (xa, xb, ya, yb), modified_offset_dict = _generate_read_bounds(
            offset_dict, raster_x_size, raster_y_size)
        flow_dir_buffer_array[ya:yb, xa:xb] = flow_dir_band.ReadAsArray(
                **modified_offset_dict).astype(numpy.uint8)

        # ensure these are set for the complier
        xi_n = -1
        yi_n = -1

        # search block for to set flow direction
        for yi in range(1, win_ysize+1):
            for xi in range(1, win_xsize+1):
                flow_dir = flow_dir_buffer_array[yi, xi]
                if flow_dir == flow_dir_nodata:
                    continue

                xi_n = xi+NEIGHBOR_OFFSET_ARRAY[2*flow_dir]
                yi_n = yi+NEIGHBOR_OFFSET_ARRAY[2*flow_dir+1]

                if flow_dir_buffer_array[yi_n, xi_n] == flow_dir_nodata:
                    xi_root = xi-1+xoff
                    yi_root = yi-1+yoff

                    if weight_raster is not None:
                        weight_val = <double>weight_raster.get(
                            xi_root, yi_root)
                        if is_close(weight_val, weight_nodata):
                            weight_val = 0.0
                    else:
                        weight_val = 1.0
                    search_stack.push(
                        FlowPixelType(xi_root, yi_root, 0, weight_val))

                while not search_stack.empty():
                    flow_pixel = search_stack.top()
                    search_stack.pop()

                    preempted = 0
                    for i_n in range(flow_pixel.last_flow_dir, 8):
                        xi_n = flow_pixel.xi+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = flow_pixel.yi+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            # no upstream here
                            continue
                        upstream_flow_dir = <int>flow_dir_managed_raster.get(
                            xi_n, yi_n)
                        if upstream_flow_dir == flow_dir_nodata or (
                                upstream_flow_dir !=
                                D8_REVERSE_DIRECTION[i_n]):
                            # no upstream here
                            continue
                        upstream_flow_accum = <double>(
                            flow_accum_managed_raster.get(xi_n, yi_n))
                        if is_close(upstream_flow_accum, flow_accum_nodata):
                            # process upstream before this one
                            flow_pixel.last_flow_dir = i_n
                            search_stack.push(flow_pixel)
                            if weight_raster is not None:
                                weight_val = <double>weight_raster.get(
                                    xi_n, yi_n)
                                if is_close(weight_val, weight_nodata):
                                    weight_val = 0.0
                            else:
                                weight_val = 1.0
                            search_stack.push(
                                FlowPixelType(xi_n, yi_n, 0, weight_val))
                            preempted = 1
                            break
                        flow_pixel.value += upstream_flow_accum
                    if not preempted:
                        flow_accum_managed_raster.set(
                            flow_pixel.xi, flow_pixel.yi,
                            flow_pixel.value)
    flow_accum_managed_raster.close()
    flow_dir_managed_raster.close()
    if weight_raster is not None:
        weight_raster.close()
    LOGGER.info('%.1f%% complete', 100.0)


def flow_dir_mfd(
        dem_raster_path_band, target_flow_dir_path, working_dir=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Multiple flow direction.

    Parameters:
        dem_raster_path_band (tuple): a path, band number tuple indicating the
            DEM calculate flow direction. This DEM must not have hydrological
            pits or else the target flow direction will be undefined.
        target_flow_dir_path (string): path to a raster created by this call
            of a 32 bit int raster of the same dimensions and projections as
            `dem_raster_path_band[0]`. The value of the pixel indicates the
            proportion of flow from that pixel to its neighbors given these
            indexes:

                321
                4x0
                567

            The pixel value is formatted as 8 separate 4 bit integers
            compressed into a 32 bit int. To extract the proportion of flow
            from a particular direction given the pixel value 'x' one can
            shift and mask as follows 0xF & (x >> (4*dir)), where `dir` is
            one of the 8 directions indicated above.

        working_dir (string): If not None, indicates where temporary files
            should be created during this run. If this directory doesn't exist
            it is created by this call.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None.
    """
    # These variables are used to iterate over the DEM using `iterblock`
    # indexes, a numpy.float64 type is used since we need to statically cast
    # and it's the most complex numerical type and will be compatible without
    # data loss for any lower type that might be used in
    # `dem_raster_path_band[0]`.
    cdef numpy.ndarray[numpy.float64_t, ndim=2] dem_buffer_array
    cdef int win_ysize, win_xsize, xoff, yoff

    # the _root variables remembers the pixel index where the plateau/pit
    # region was first detected when iterating over the DEM.
    cdef int xi_root, yi_root

    # these variables are used as pixel or neighbor indexes. where _q
    # represents a value out of a queue, and _n is related to a neighbor pixel
    cdef int i_n, xi, yi, xi_q, yi_q, xi_n, yi_n

    # these are used to recall the local and neighbor heights of pixels
    cdef double root_height, n_height, dem_nodata, n_slope

    # these are used to track the distance to the drain when we encounter a
    # plateau to route to the shortest path to the drain
    cdef double drain_distance, n_drain_distance

    # `drain_search_queue` is used to grow a flat region searching for a drain
    # of a plateau
    cdef queue[CoordinateType] drain_search_queue

    # downhill_slope_array array will keep track of the floating point value of
    # the downhill slopes in a pixel
    cdef double downhill_slope_array[8]
    cdef double nodata_downhill_slope_array[8]

    # a pointer reference to whatever kind of slope we're considering
    cdef double *working_downhill_slope_array

    # as the neighbor slopes are calculated, this variable gathers them
    # together to calculate the final contribution of neighbor slopes to
    # fraction of flow
    cdef double sum_of_slope_weights, sum_of_nodata_slope_weights

    # this variable will be used to pack the neighbor slopes into a single
    # value to write to the raster or read from neighbors
    cdef int compressed_integer_slopes

    # `distance_drain_queue` is used after a plateau drain is defined and
    # iterates until the entire plateau is drained,
    # `nodata_distance_drain_queue` is for
    # the case where the plateau is only drained by nodata pixels
    cdef CoordinateQueueType distance_drain_queue, nodata_distance_drain_queue

    # direction drain queue is used in the last phase to set flow direction
    # based on the distance to drain
    cdef CoordinateQueueType direction_drain_queue

    # this queue is used to remember the flow directions of nodata pixels in
    # a plateau in case no other valid drain was found
    cdef queue[int] nodata_flow_dir_queue

    # properties of the parallel rasters
    cdef int raster_x_size, raster_y_size

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    # determine dem nodata in the working type, or set an improbable value
    # if one can't be determined
    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_path_band[0])
    base_nodata = dem_raster_info['nodata'][dem_raster_path_band[1]-1]
    if base_nodata is not None:
        # cast to a float64 since that's our operating array type
        dem_nodata = numpy.float64(base_nodata)
    else:
        # pick some very improbable value since it's hard to deal with NaNs
        dem_nodata = IMPROBABLE_FLOAT_NODATA

    # these are used to determine if a sample is within the raster
    raster_x_size, raster_y_size = dem_raster_info['raster_size']

    # this is the nodata value for all the flat region and pit masks
    mask_nodata = 0

    # set up the working dir for the mask rasters
    try:
        if working_dir is not None:
            os.makedirs(working_dir)
    except OSError:
        pass
    working_dir_path = tempfile.mkdtemp(
        dir=working_dir,
        prefix='flow_dir_multiple_flow_dir_%s_' % time.strftime(
            '%Y-%m-%d_%H_%M_%S', time.gmtime()))

    # this raster is used to keep track of what pixels have been searched for
    # a plateau. if a pixel is set, it means it is part of a locally
    # undrained area
    flat_region_mask_path = os.path.join(
        working_dir_path, 'flat_region_mask.tif')
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], flat_region_mask_path, gdal.GDT_Byte,
        [mask_nodata], fill_value_list=[mask_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    flat_region_mask_managed_raster = _ManagedRaster(
        flat_region_mask_path, 1, 1)

    flow_dir_nodata = 0
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], target_flow_dir_path, gdal.GDT_Int32,
        [flow_dir_nodata], fill_value_list=[flow_dir_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    flow_dir_managed_raster = _ManagedRaster(target_flow_dir_path, 1, 1)

    plateu_drain_mask_path = os.path.join(
        working_dir_path, 'plateu_drain_mask.tif')
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], plateu_drain_mask_path, gdal.GDT_Byte,
        [mask_nodata], fill_value_list=[mask_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    plateau_drain_mask_managed_raster = _ManagedRaster(
        plateu_drain_mask_path, 1, 1)

    # this creates a raster that's used for a dynamic programming solution to
    # shortest path to the drain for plateaus. the raster is filled with
    # raster_x_size * raster_y_size as a distance that's greater than the
    # longest plateau drain distance possible for this raster.
    plateau_distance_path = os.path.join(
        working_dir_path, 'plateau_distance.tif')
    plateau_distance_nodata = raster_x_size * raster_y_size
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], plateau_distance_path, gdal.GDT_Float64,
        [plateau_distance_nodata], fill_value_list=[
            raster_x_size * raster_y_size],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    plateau_distance_managed_raster = _ManagedRaster(
        plateau_distance_path, 1, 1)

    # this raster is for random access of the DEM
    compatable_dem_raster_path_band = None
    dem_block_xsize, dem_block_ysize = dem_raster_info['block_size']
    if (dem_block_xsize & (dem_block_xsize - 1) != 0) or (
            dem_block_ysize & (dem_block_ysize - 1) != 0):
        LOGGER.warning("dem is not a power of 2, creating a copy that is.")
        compatable_dem_raster_path_band = (
            os.path.join(working_dir_path, 'compatable_dem.tif'),
            dem_raster_path_band[1])
        raster_driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
        dem_raster = gdal.OpenEx(dem_raster_path_band[0], gdal.OF_RASTER)
        raster_driver.CreateCopy(
            compatable_dem_raster_path_band[0], dem_raster,
            options=raster_driver_creation_tuple[1])
        dem_raster = None
        LOGGER.info("compatible dem complete")
    else:
        compatable_dem_raster_path_band = dem_raster_path_band
    dem_managed_raster = _ManagedRaster(
        compatable_dem_raster_path_band[0],
        compatable_dem_raster_path_band[1], 0)

    # and this raster is for efficient block-by-block reading of the dem
    dem_raster = gdal.OpenEx(
        compatable_dem_raster_path_band[0], gdal.OF_RASTER)
    dem_band = dem_raster.GetRasterBand(
        compatable_dem_raster_path_band[1])

    # this outer loop searches for a pixel that is locally undrained
    for offset_dict in pygeoprocessing.iterblocks(
            compatable_dem_raster_path_band, offset_only=True,
            largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('%.1f%% complete', 100.0 * current_pixel / <float>(
                raster_x_size * raster_y_size))

        # make a buffer big enough to capture block and boundaries around it
        dem_buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.float64)
        dem_buffer_array[:] = dem_nodata

        # check if we can widen the border to include real data from the
        # raster
        (xa, xb, ya, yb), modified_offset_dict = _generate_read_bounds(
            offset_dict, raster_x_size, raster_y_size)
        dem_buffer_array[ya:yb, xa:xb] = dem_band.ReadAsArray(
                **modified_offset_dict).astype(numpy.float64)

        # ensure these are set for the complier
        xi_n = -1
        yi_n = -1

        # search block for to set flow direction
        for yi in range(1, win_ysize+1):
            for xi in range(1, win_xsize+1):
                root_height = dem_buffer_array[yi, xi]
                if is_close(root_height, dem_nodata):
                    continue

                # this value is set in case it turns out to be the root of a
                # pit, we'll start the fill from this pixel in the last phase
                # of the algorithm
                xi_root = xi-1+xoff
                yi_root = yi-1+yoff

                if flow_dir_managed_raster.get(
                        xi_root, yi_root) != flow_dir_nodata:
                    # already been defined
                    continue

                # PHASE 1 - try to set the direction based on local values
                # initialize variables to indicate the largest slope_dir is
                # undefined, the largest slope seen so far is flat, and the
                # largest nodata is at least a diagonal away
                sum_of_downhill_slopes = 0.0
                for i_n in range(8):
                    # initialize downhill slopes to 0.0
                    downhill_slope_array[i_n] = 0.0
                    xi_n = xi+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                    yi_n = yi+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                    n_height = dem_buffer_array[yi_n, xi_n]
                    if is_close(n_height, dem_nodata):
                        continue
                    n_slope = root_height - n_height
                    if n_slope > 0.0:
                        if i_n & 1:
                            # if diagonal, adjust the slope
                            n_slope *= SQRT2_INV
                        downhill_slope_array[i_n] = n_slope
                        sum_of_downhill_slopes += n_slope

                if sum_of_downhill_slopes > 0.0:
                    compressed_integer_slopes = 0
                    for i_n in range(8):
                        compressed_integer_slopes |= (<int>(
                            0.5 + downhill_slope_array[i_n] /
                            sum_of_downhill_slopes * 0xF)) << (i_n * 4)

                    flow_dir_managed_raster.set(
                        xi_root, yi_root, compressed_integer_slopes)
                    continue

                # PHASE 2 - search for what drains the plateau, prefer
                # downhill drains, but fall back if nodata pixels are the
                # only drain

                # otherwise, this pixel doesn't drain locally, so it must
                # be a plateau, search for the drains of the plateau
                drain_search_queue.push(CoordinateType(xi_root, yi_root))
                flat_region_mask_managed_raster.set(xi_root, yi_root, 1)

                # this loop does a BFS starting at this pixel to all pixels
                # of the same height. if a drain is encountered, it is pushed
                # on a queue for later processing.
                while not drain_search_queue.empty():
                    xi_q = drain_search_queue.front().xi
                    yi_q = drain_search_queue.front().yi
                    drain_search_queue.pop()

                    sum_of_slope_weights = 0.0
                    sum_of_nodata_slope_weights = 0.0
                    for i_n in range(8):
                        # initialize downhill slopes to 0.0
                        downhill_slope_array[i_n] = 0.0
                        nodata_downhill_slope_array[i_n] = 0.0
                        xi_n = xi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = yi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]

                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            n_height = dem_nodata
                        else:
                            n_height = dem_managed_raster.get(xi_n, yi_n)
                        if is_close(n_height, dem_nodata):
                            n_slope = SQRT2_INV if i_n & 1 else 1.0
                            sum_of_nodata_slope_weights += n_slope
                            nodata_downhill_slope_array[i_n] = n_slope
                            continue
                        n_slope = root_height - n_height
                        if n_slope < 0:
                            continue
                        if n_slope == 0.0:
                            if flat_region_mask_managed_raster.get(
                                    xi_n, yi_n) == mask_nodata:
                                # only grow if it's at the same level and not
                                # previously visited
                                drain_search_queue.push(
                                    CoordinateType(xi_n, yi_n))
                                flat_region_mask_managed_raster.set(
                                    xi_n, yi_n, 1)
                            continue
                        if i_n & 1:
                            n_slope *= SQRT2_INV
                        downhill_slope_array[i_n] = n_slope
                        sum_of_slope_weights += downhill_slope_array[i_n]

                    working_downhill_slope_sum = 0.0
                    working_downhill_slope_array = NULL
                    if sum_of_slope_weights > 0.0:
                        working_downhill_slope_array = downhill_slope_array
                        working_downhill_slope_sum = sum_of_slope_weights
                    elif sum_of_nodata_slope_weights > 0.0:
                        working_downhill_slope_array = (
                            nodata_downhill_slope_array)
                        working_downhill_slope_sum = (
                            sum_of_nodata_slope_weights)

                    if working_downhill_slope_sum > 0.0:
                        compressed_integer_slopes = 0
                        for i_n in range(8):
                            compressed_integer_slopes |= (<int>(
                                0.5 + working_downhill_slope_array[i_n] /
                                working_downhill_slope_sum * 0xF)) << (
                                i_n * 4)
                        if sum_of_slope_weights > 0.0:
                            # regular downhill pixel
                            flow_dir_managed_raster.set(
                                xi_q, yi_q, compressed_integer_slopes)
                            plateau_distance_managed_raster.set(
                                xi_q, yi_q, 0.0)
                            plateau_drain_mask_managed_raster.set(
                                xi_q, yi_q, 1)
                            distance_drain_queue.push(
                                CoordinateType(xi_q, yi_q))
                        else:
                            nodata_distance_drain_queue.push(
                                CoordinateType(xi_q, yi_q))
                            nodata_flow_dir_queue.push(
                                compressed_integer_slopes)

                # if there's no downhill drains, try the nodata drains
                if distance_drain_queue.empty():
                    # push the nodata drain queue over to the drain queue
                    # and set all the flow directions on the nodata drain
                    # pixels
                    while not nodata_distance_drain_queue.empty():
                        xi_q = nodata_distance_drain_queue.front().xi
                        yi_q = nodata_distance_drain_queue.front().yi
                        flow_dir_managed_raster.set(
                            xi_q, yi_q, nodata_flow_dir_queue.front())
                        plateau_distance_managed_raster.set(xi_q, yi_q, 0.0)
                        plateau_drain_mask_managed_raster.set(xi_q, yi_q, 1)
                        distance_drain_queue.push(
                            nodata_distance_drain_queue.front())
                        nodata_flow_dir_queue.pop()
                        nodata_distance_drain_queue.pop()
                else:
                    # clear the nodata drain queues
                    nodata_flow_dir_queue = IntQueueType()
                    nodata_distance_drain_queue = CoordinateQueueType()

                # copy the drain queue to another queue
                for _ in range(distance_drain_queue.size()):
                    distance_drain_queue.push(
                        distance_drain_queue.front())
                    direction_drain_queue.push(distance_drain_queue.front())
                    distance_drain_queue.pop()

                # PHASE 3 - build up a distance raster for the plateau such
                # that the pixel value indicates how far it is from the
                # nearest drain

                # this loop does a BFS from the plateau drain to any other
                # neighboring undefined pixels
                while not distance_drain_queue.empty():
                    xi_q = distance_drain_queue.front().xi
                    yi_q = distance_drain_queue.front().yi
                    distance_drain_queue.pop()

                    drain_distance = plateau_distance_managed_raster.get(
                        xi_q, yi_q)

                    for i_n in range(8):
                        xi_n = xi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = yi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            continue

                        n_drain_distance = drain_distance + (
                            SQRT2 if i_n & 1 else 1.0)

                        if is_close(dem_managed_raster.get(
                                xi_n, yi_n), root_height) and (
                                plateau_distance_managed_raster.get(
                                    xi_n, yi_n) > n_drain_distance):
                            # neighbor is at same level and has longer drain
                            # flow path than current
                            plateau_distance_managed_raster.set(
                                xi_n, yi_n, n_drain_distance)
                            distance_drain_queue.push(
                                CoordinateType(xi_n, yi_n))

                # PHASE 4 - set the plateau pixel flow direction based on the
                # distance to the nearest drain
                while not direction_drain_queue.empty():
                    xi_q = direction_drain_queue.front().xi
                    yi_q = direction_drain_queue.front().yi
                    direction_drain_queue.pop()

                    drain_distance = plateau_distance_managed_raster.get(
                        xi_q, yi_q)

                    sum_of_slope_weights = 0.0
                    for i_n in range(8):
                        xi_n = xi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = yi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                        downhill_slope_array[i_n] = 0.0

                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            continue

                        if dem_managed_raster.get(xi_n, yi_n) != root_height:
                            continue

                        n_distance = plateau_distance_managed_raster.get(
                            xi_n, yi_n)
                        if n_distance == plateau_distance_nodata:
                            continue
                        if n_distance < drain_distance:
                            n_slope = SQRT2_INV if i_n & 1 else 1.0
                            downhill_slope_array[i_n] = n_slope
                            sum_of_slope_weights += n_slope
                        elif not plateau_drain_mask_managed_raster.get(
                                xi_n, yi_n):
                            direction_drain_queue.push(
                                CoordinateType(xi_n, yi_n))
                            plateau_drain_mask_managed_raster.set(
                                xi_n, yi_n, 1)

                    if sum_of_slope_weights == 0:
                        continue
                    compressed_integer_slopes = 0
                    for i_n in range(8):
                        compressed_integer_slopes |= (<int>(
                            0.5 + downhill_slope_array[i_n] /
                            sum_of_slope_weights * 0xF)) << (i_n * 4)
                    flow_dir_managed_raster.set(
                        xi_q, yi_q, compressed_integer_slopes)

    dem_band = None
    dem_raster = None
    plateau_drain_mask_managed_raster.close()
    flow_dir_managed_raster.close()
    flat_region_mask_managed_raster.close()
    dem_managed_raster.close()
    plateau_distance_managed_raster.close()
    shutil.rmtree(working_dir_path)
    LOGGER.info('%.1f%% complete', 100.0)


def flow_accumulation_mfd(
        flow_dir_mfd_raster_path_band, target_flow_accum_raster_path,
        weight_raster_path_band=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Multiple flow direction accumulation.

    Parameters:
        flow_dir_mfd_raster_path_band (tuple): a path, band number tuple
            for a multiple flow direction raster generated from a call to
            `flow_dir_mfd`. The format of this raster is described in the
            docstring of that function.
        target_flow_accum_raster_path (string): a path to a raster created by
            a call to this function that is the same dimensions and projection
            as `flow_dir_mfd_raster_path_band[0]`. The value in each pixel is
            1 plus the proportional contribution of all upstream pixels that
            flow into it. The proportion is determined as the value of the
            upstream flow dir pixel in the downslope direction pointing to
            the current pixel divided by the sum of all the flow weights
            exiting that pixel. Note the target type of this raster
            is a 64 bit float so there is minimal risk of overflow and the
            possibility of handling a float dtype in
            `weight_raster_path_band`.
        weight_raster_path_band (tuple): optional path and band number to a
            raster that will be used as the per-pixel flow accumulation
            weight. If `None`, 1 is the default flow accumulation weight.
            This raster must be the same dimensions as
            `flow_dir_mfd_raster_path_band`. If a weight nodata pixel is
            encountered it will be treated as a weight value of 0.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None.

    """
    # These variables are used to iterate over the DEM using `iterblock`
    # indexes, a numpy.float64 type is used since we need to statically cast
    # and it's the most complex numerical type and will be compatible without
    # data loss for any lower type that might be used in
    # `dem_raster_path_band[0]`.
    cdef numpy.ndarray[numpy.int32_t, ndim=2] flow_dir_mfd_buffer_array
    cdef int win_ysize, win_xsize, xoff, yoff

    # the _root variables remembers the pixel index where the plateau/pit
    # region was first detected when iterating over the DEM.
    cdef int xi_root, yi_root

    # these variables are used as pixel or neighbor indexes.
    # _n is related to a neighbor pixel
    cdef int i_n, xi, yi, xi_n, yi_n, i_upstream_flow

    # used to hold flow direction values
    cdef int flow_dir_mfd, upstream_flow_weight

    # used as a holder variable to account for upstream flow
    cdef int compressed_upstream_flow_dir, upstream_flow_dir_sum

    # used to determine if the upstream pixel has been processed, and if not
    # to trigger a recursive uphill walk
    cdef double upstream_flow_accum

    cdef double flow_accum_nodata = IMPROBABLE_FLOAT_NODATA
    cdef double weight_nodata = IMPROBABLE_FLOAT_NODATA

    # this value is used to store the current weight which might be 1 or
    # come from a predefined flow accumulation weight raster
    cdef double weight_val

    # `search_stack` is used to walk upstream to calculate flow accumulation
    # values represented in a flow pixel which stores the x/y position,
    # next direction to check, and running flow accumulation value.
    cdef stack[FlowPixelType] search_stack
    cdef FlowPixelType flow_pixel

    # properties of the parallel rasters
    cdef int raster_x_size, raster_y_size

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    if not _is_raster_path_band_formatted(flow_dir_mfd_raster_path_band):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                flow_dir_mfd_raster_path_band))
    if weight_raster_path_band and not _is_raster_path_band_formatted(
            weight_raster_path_band):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                weight_raster_path_band))

    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_raster_path_band[0], target_flow_accum_raster_path,
        gdal.GDT_Float64, [flow_accum_nodata],
        fill_value_list=[flow_accum_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)

    flow_accum_managed_raster = _ManagedRaster(
        target_flow_accum_raster_path, 1, 1)

    flow_dir_managed_raster = _ManagedRaster(
        flow_dir_mfd_raster_path_band[0], flow_dir_mfd_raster_path_band[1], 0)
    flow_dir_raster = gdal.OpenEx(
        flow_dir_mfd_raster_path_band[0], gdal.OF_RASTER)
    flow_dir_band = flow_dir_raster.GetRasterBand(
        flow_dir_mfd_raster_path_band[1])

    cdef _ManagedRaster weight_raster = None
    if weight_raster_path_band:
        weight_raster = _ManagedRaster(
            weight_raster_path_band[0], weight_raster_path_band[1], 0)
        raw_weight_nodata = pygeoprocessing.get_raster_info(
            weight_raster_path_band[0])['nodata'][
                weight_raster_path_band[1]-1]
        if raw_weight_nodata is not None:
            weight_nodata = raw_weight_nodata

    flow_dir_raster_info = pygeoprocessing.get_raster_info(
        flow_dir_mfd_raster_path_band[0])
    raster_x_size, raster_y_size = flow_dir_raster_info['raster_size']

    # this outer loop searches for a pixel that is locally undrained
    for offset_dict in pygeoprocessing.iterblocks(
            flow_dir_mfd_raster_path_band, offset_only=True,
            largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('%.1f%% complete', 100.0 * current_pixel / <float>(
                raster_x_size * raster_y_size))

        # make a buffer big enough to capture block and boundaries around it
        flow_dir_mfd_buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.int32)
        flow_dir_mfd_buffer_array[:] = 0  # 0 means no flow at all

        # check if we can widen the border to include real data from the
        # raster
        (xa, xb, ya, yb), modified_offset_dict = _generate_read_bounds(
            offset_dict, raster_x_size, raster_y_size)
        flow_dir_mfd_buffer_array[ya:yb, xa:xb] = flow_dir_band.ReadAsArray(
                **modified_offset_dict).astype(numpy.int32)

        # ensure these are set for the complier
        xi_n = -1
        yi_n = -1

        # search block for to set flow accumulation
        for yi in range(1, win_ysize+1):
            for xi in range(1, win_xsize+1):
                flow_dir_mfd = flow_dir_mfd_buffer_array[yi, xi]
                if flow_dir_mfd == 0:
                    # no flow in this pixel, so skip
                    continue

                for i_n in range(8):
                    if ((flow_dir_mfd >> (i_n * 4)) & 0xF) == 0:
                        # no flow in that direction
                        continue
                    xi_n = xi+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                    yi_n = yi+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]

                    if flow_dir_mfd_buffer_array[yi_n, xi_n] == 0:
                        # if the entire value is zero, it flows nowhere
                        # and the root pixel is draining to it, thus the
                        # root must be a drain
                        xi_root = xi-1+xoff
                        yi_root = yi-1+yoff
                        if weight_raster is not None:
                            weight_val = <double>weight_raster.get(
                                xi_root, yi_root)
                            if is_close(weight_val, weight_nodata):
                                weight_val = 0.0
                        else:
                            weight_val = 1.0
                        search_stack.push(
                            FlowPixelType(xi_root, yi_root, 0, weight_val))
                        break

                while not search_stack.empty():
                    flow_pixel = search_stack.top()
                    search_stack.pop()

                    preempted = 0
                    for i_n in range(flow_pixel.last_flow_dir, 8):
                        xi_n = flow_pixel.xi+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = flow_pixel.yi+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            # no upstream here
                            continue
                        compressed_upstream_flow_dir = (
                            <int>flow_dir_managed_raster.get(xi_n, yi_n))
                        upstream_flow_weight = (
                            compressed_upstream_flow_dir >> (
                                D8_REVERSE_DIRECTION[i_n] * 4)) & 0xF
                        if upstream_flow_weight == 0:
                            # no upstream flow to this pixel
                            continue
                        upstream_flow_accum = (
                            flow_accum_managed_raster.get(xi_n, yi_n))
                        if is_close(upstream_flow_accum, flow_accum_nodata):
                            # process upstream before this one
                            flow_pixel.last_flow_dir = i_n
                            search_stack.push(flow_pixel)
                            if weight_raster is not None:
                                weight_val = <double>weight_raster.get(
                                    xi_n, yi_n)
                                if is_close(weight_val, weight_nodata):
                                    weight_val = 0.0
                            else:
                                weight_val = 1.0
                            search_stack.push(
                                FlowPixelType(xi_n, yi_n, 0, weight_val))
                            preempted = 1
                            break
                        upstream_flow_dir_sum = 0
                        for i_upstream_flow in range(8):
                            upstream_flow_dir_sum += (
                                compressed_upstream_flow_dir >> (
                                    i_upstream_flow * 4)) & 0xF

                        flow_pixel.value += (
                            upstream_flow_accum * upstream_flow_weight /
                            <float>upstream_flow_dir_sum)
                    if not preempted:
                        flow_accum_managed_raster.set(
                            flow_pixel.xi, flow_pixel.yi,
                            flow_pixel.value)
    flow_accum_managed_raster.close()
    flow_dir_managed_raster.close()
    if weight_raster is not None:
        weight_raster.close()
    LOGGER.info('%.1f%% complete', 100.0)


def distance_to_channel_d8(
        flow_dir_d8_raster_path_band, channel_raster_path_band,
        target_distance_to_channel_raster_path,
        weight_raster_path_band=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Calculate distance to channel with D8 flow.

    Parameters:
        flow_dir_d8_raster_path_band (tuple): a path/band index tuple
            indicating the raster that defines the D8 flow direction
            raster for this call. The pixel values are integers that
            correspond to outflow in the following configuration:
                321
                4x0
                567
        channel_raster_path_band (tuple): a path/band tuple of the same
            dimensions and projection as `flow_dir_d8_raster_path_band[0]`
            that indicates where the channels in the problem space lie. A
            channel is indicated if the value of the pixel is 1. Other values
            are ignored.
        target_distance_to_channel_raster_path (string): path to a raster
            created by this call that has per-pixel distances from a given
            pixel to the nearest downhill channel.
        weight_raster_path_band (tuple): optional path and band number to a
            raster that will be used as the per-pixel flow distance
            weight. If `None`, 1 is the default distance between neighboring
            pixels. This raster must be the same dimensions as
            `flow_dir_mfd_raster_path_band`.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None.
    """
    # These variables are used to iterate over the DEM using `iterblock`
    # indexes
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] channel_buffer_array
    cdef int win_ysize, win_xsize, xoff, yoff

    # these variables are used as pixel or neighbor indexes.
    # _n is related to a neighbor pixel
    cdef int i_n, xi, yi, xi_q, yi_q, xi_n, yi_n

    # `distance_to_channel_stack` is the datastructure that walks upstream
    # from a defined flow distance pixel
    cdef stack[PixelType] distance_to_channel_stack

    # properties of the parallel rasters
    cdef int raster_x_size, raster_y_size

    # these area used to store custom per-pixel weights and per-pixel values
    # for distance updates
    cdef double weight_val, pixel_val
    cdef double weight_nodata = IMPROBABLE_FLOAT_NODATA

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    for path in (
            flow_dir_d8_raster_path_band, channel_raster_path_band,
            weight_raster_path_band):
        if path is not None and not _is_raster_path_band_formatted(path):
            raise ValueError(
                "%s is supposed to be a raster band tuple but it's not." % (
                    path))

    distance_nodata = -1
    pygeoprocessing.new_raster_from_base(
        flow_dir_d8_raster_path_band[0],
        target_distance_to_channel_raster_path,
        gdal.GDT_Float64, [distance_nodata],
        fill_value_list=[distance_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    distance_to_channel_managed_raster = _ManagedRaster(
        target_distance_to_channel_raster_path, 1, 1)

    cdef _ManagedRaster weight_raster = None
    if weight_raster_path_band:
        weight_raster = _ManagedRaster(
            weight_raster_path_band[0], weight_raster_path_band[1], 0)
        raw_weight_nodata = pygeoprocessing.get_raster_info(
            weight_raster_path_band[0])['nodata'][
                weight_raster_path_band[1]-1]
        if raw_weight_nodata is not None:
            weight_nodata = raw_weight_nodata

    channel_managed_raster = _ManagedRaster(
        channel_raster_path_band[0], channel_raster_path_band[1], 0)

    flow_dir_d8_managed_raster = _ManagedRaster(
        flow_dir_d8_raster_path_band[0], flow_dir_d8_raster_path_band[1], 0)
    channel_raster = gdal.OpenEx(channel_raster_path_band[0], gdal.OF_RASTER)
    channel_band = channel_raster.GetRasterBand(channel_raster_path_band[1])

    flow_dir_raster_info = pygeoprocessing.get_raster_info(
        flow_dir_d8_raster_path_band[0])
    raster_x_size, raster_y_size = flow_dir_raster_info['raster_size']

    # this outer loop searches for undefined channels
    for offset_dict in pygeoprocessing.iterblocks(
            channel_raster_path_band, offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('%.1f%% complete', 100.0 * current_pixel / <float>(
                raster_x_size * raster_y_size))

        # make a buffer big enough to capture block and boundaries around it
        channel_buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.uint8)
        channel_buffer_array[:] = 0  # 0 means no channel

        # check if we can widen the border to include real data from the
        # raster
        (xa, xb, ya, yb), modified_offset_dict = _generate_read_bounds(
            offset_dict, raster_x_size, raster_y_size)
        channel_buffer_array[ya:yb, xa:xb] = channel_band.ReadAsArray(
            **modified_offset_dict).astype(numpy.int8)

        # ensure these are set for the complier
        xi_n = -1
        yi_n = -1

        # search block for to search for a channel seed
        for yi in range(1, win_ysize+1):
            for xi in range(1, win_xsize+1):
                if channel_buffer_array[yi, xi] != 1:
                    # no channel seed
                    continue

                distance_to_channel_stack.push(
                    PixelType(0.0, xi+xoff-1, yi+yoff-1, 0))

                while not distance_to_channel_stack.empty():
                    xi_q = distance_to_channel_stack.top().xi
                    yi_q = distance_to_channel_stack.top().yi
                    pixel_val = distance_to_channel_stack.top().value
                    distance_to_channel_stack.pop()

                    distance_to_channel_managed_raster.set(
                        xi_q, yi_q, pixel_val)

                    for i_n in range(8):
                        xi_n = xi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = yi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]

                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            continue

                        if channel_managed_raster.get(xi_n, yi_n) == 1:
                            # it's a channel, it'll get picked up in the
                            # outer loop
                            continue

                        if (flow_dir_d8_managed_raster.get(xi_n, yi_n) ==
                                D8_REVERSE_DIRECTION[i_n]):
                            # if a weight is passed we use it directly and do
                            # not consider that a diagonal pixel is further
                            # away than an adjacent one. If no weight is used
                            # then "distance" is being calculated and we
                            # account for diagonal distance.
                            if weight_raster is not None:
                                weight_val = weight_raster.get(xi_n, yi_n)
                                if is_close(weight_val, weight_nodata):
                                    weight_val = 0.0
                            else:
                                weight_val = (SQRT2 if i_n % 2 else 1)

                            distance_to_channel_stack.push(
                                PixelType(
                                    weight_val + pixel_val, xi_n, yi_n, 0))

    distance_to_channel_managed_raster.close()
    flow_dir_d8_managed_raster.close()
    channel_managed_raster.close()
    if weight_raster is not None:
        weight_raster.close()


def distance_to_channel_mfd(
        flow_dir_mfd_raster_path_band, channel_raster_path_band,
        target_distance_to_channel_raster_path, weight_raster_path_band=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Calculate distance to channel with multiple flow direction.

    Parameters:
        flow_dir_mfd_raster_path_band (tuple): a path/band index tuple
            indicating the raster that defines the mfd flow accumulation
            raster for this call. This raster should be generated by a call
            to `pygeoprocessing.routing.flow_dir_mfd.

        channel_raster_path_band (tuple): a path/band tuple of the same
            dimensions and projection as `flow_dir_mfd_raster_path_band[0]`
            that indicates where the channels in the problem space lie. A
            channel is indicated if the value of the pixel is 1. Other values
            are ignored.
        target_distance_to_channel_raster_path (string): path to a raster
            created by this call that has per-pixel distances from a given
            pixel to the nearest downhill channel.
        weight_raster_path_band (tuple): optional path and band number to a
            raster that will be used as the per-pixel flow distance
            weight. If `None`, 1 is the default distance between neighboring
            pixels. This raster must be the same dimensions as
            `flow_dir_mfd_raster_path_band`.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None.
    """
    # These variables are used to iterate over the DEM using `iterblock`
    # indexes
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] channel_buffer_array
    cdef numpy.ndarray[numpy.int32_t, ndim=2] flow_dir_buffer_array
    cdef int win_ysize, win_xsize, xoff, yoff

    # these variables are used as pixel or neighbor indexes.
    # _n is related to a neighbor pixel
    cdef int i_n, xi, yi, xi_n, yi_n
    cdef int flow_dir_weight, sum_of_flow_weights, compressed_flow_dir

    # used to remember if the current pixel is a channel for routing
    cdef int is_a_channel

    # `distance_to_channel_queue` is the data structure that walks upstream
    # from a defined flow distance pixel
    cdef stack[FlowPixelType] distance_to_channel_stack

    # properties of the parallel rasters
    cdef int raster_x_size, raster_y_size

    # this value is used to store the current weight which might be 1 or
    # come from a predefined flow accumulation weight raster
    cdef double weight_val
    cdef double weight_nodata = IMPROBABLE_FLOAT_NODATA

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    for path in (
            flow_dir_mfd_raster_path_band, channel_raster_path_band,
            weight_raster_path_band):
        if path is not None and not _is_raster_path_band_formatted(path):
            raise ValueError(
                "%s is supposed to be a raster band tuple but it's not." % (
                    path))

    distance_nodata = IMPROBABLE_FLOAT_NODATA
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_raster_path_band[0],
        target_distance_to_channel_raster_path,
        gdal.GDT_Float64, [distance_nodata],
        fill_value_list=[distance_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    distance_to_channel_managed_raster = _ManagedRaster(
        target_distance_to_channel_raster_path, 1, 1)

    channel_managed_raster = _ManagedRaster(
        channel_raster_path_band[0], channel_raster_path_band[1], 0)

    flow_dir_mfd_managed_raster = _ManagedRaster(
        flow_dir_mfd_raster_path_band[0], flow_dir_mfd_raster_path_band[1], 0)
    channel_raster = gdal.OpenEx(channel_raster_path_band[0], gdal.OF_RASTER)
    channel_band = channel_raster.GetRasterBand(channel_raster_path_band[1])

    flow_dir_mfd_raster = gdal.OpenEx(
        flow_dir_mfd_raster_path_band[0], gdal.OF_RASTER)
    flow_dir_mfd_band = flow_dir_mfd_raster.GetRasterBand(
        flow_dir_mfd_raster_path_band[1])

    cdef _ManagedRaster weight_raster = None
    if weight_raster_path_band:
        weight_raster = _ManagedRaster(
            weight_raster_path_band[0], weight_raster_path_band[1], 0)
        raw_weight_nodata = pygeoprocessing.get_raster_info(
            weight_raster_path_band[0])['nodata'][
                weight_raster_path_band[1]-1]
        if raw_weight_nodata is not None:
            weight_nodata = raw_weight_nodata
        else:
            weight_nodata = IMPROBABLE_FLOAT_NODATA

    flow_dir_raster_info = pygeoprocessing.get_raster_info(
        flow_dir_mfd_raster_path_band[0])
    raster_x_size, raster_y_size = flow_dir_raster_info['raster_size']

    # this outer loop searches for undefined channels
    for offset_dict in pygeoprocessing.iterblocks(
            channel_raster_path_band, offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('%.1f%% complete', 100.0 * current_pixel / <float>(
                raster_x_size * raster_y_size))

        # make a buffer big enough to capture block and boundaries around it
        channel_buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.uint8)
        flow_dir_buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.int32)
        channel_buffer_array[:] = 0  # 0 means no channel
        flow_dir_buffer_array[:] = 0  # 0 means no flow

        # check if we can widen the border to include real data from the
        # raster
        (xa, xb, ya, yb), modified_offset_dict = _generate_read_bounds(
            offset_dict, raster_x_size, raster_y_size)
        channel_buffer_array[ya:yb, xa:xb] = channel_band.ReadAsArray(
            **modified_offset_dict).astype(numpy.int8)

        flow_dir_buffer_array[ya:yb, xa:xb] = flow_dir_mfd_band.ReadAsArray(
            **modified_offset_dict).astype(numpy.int32)

        # ensure these are set for the complier
        xi_n = -1
        yi_n = -1

        # search block for a pixel that has undefined distance to channel
        for yi in range(1, win_ysize+1):
            for xi in range(1, win_xsize+1):
                xi_root = xi+xoff-1
                yi_root = yi+yoff-1

                if channel_buffer_array[yi, xi] == 1:
                    distance_to_channel_managed_raster.set(
                        xi_root, yi_root, 0)
                    continue

                if flow_dir_buffer_array[yi, xi] == 0:
                    # nodata flow, so we skip
                    continue

                if is_close(distance_to_channel_managed_raster.get(
                        xi_root, yi_root), distance_nodata):
                    distance_to_channel_stack.push(
                        FlowPixelType(xi_root, yi_root, 0, 0.0))

                while not distance_to_channel_stack.empty():
                    pixel = distance_to_channel_stack.top()
                    distance_to_channel_stack.pop()
                    is_a_channel = (
                        channel_managed_raster.get(pixel.xi, pixel.yi) == 1)
                    if is_a_channel:
                        distance_to_channel_managed_raster.set(
                            pixel.xi, pixel.yi, 0)
                        continue

                    compressed_flow_dir = (
                        <int>flow_dir_mfd_managed_raster.get(
                            pixel.xi, pixel.yi))

                    preempted = 0
                    for i_n in range(pixel.last_flow_dir, 8):
                        flow_dir_weight = 0xF & (
                            compressed_flow_dir >> (i_n * 4))
                        if flow_dir_weight == 0:
                            continue

                        xi_n = pixel.xi+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = pixel.yi+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]

                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            continue

                        n_distance = distance_to_channel_managed_raster.get(
                            xi_n, yi_n)

                        if is_close(n_distance, distance_nodata):
                            preempted = 1
                            pixel.last_flow_dir = i_n
                            distance_to_channel_stack.push(pixel)
                            distance_to_channel_stack.push(
                                FlowPixelType(xi_n, yi_n, 0, 0.0))
                            break

                        # if a weight is passed we use it directly and do
                        # not consider that a diagonal pixel is further
                        # away than an adjacent one. If no weight is used
                        # then "distance" is being calculated and we account
                        # for diagonal distance.
                        if weight_raster is not None:
                            weight_val = weight_raster.get(xi_n, yi_n)
                            if is_close(weight_val, weight_nodata):
                                weight_val = 0.0
                        else:
                            weight_val = (SQRT2 if i_n % 2 else 1)

                        pixel.value += flow_dir_weight * (
                            weight_val + n_distance)

                    if preempted:
                        continue

                    sum_of_flow_weights = 0
                    for i_n in range(8):
                        sum_of_flow_weights += 0xF & (
                            compressed_flow_dir >> (i_n * 4))

                    if sum_of_flow_weights != 0:
                        pixel.value = pixel.value / sum_of_flow_weights
                    else:
                        pixel.value = 0
                    distance_to_channel_managed_raster.set(
                        pixel.xi, pixel.yi, pixel.value)

    distance_to_channel_managed_raster.close()
    channel_managed_raster.close()
    flow_dir_mfd_managed_raster.close()
    if weight_raster is not None:
        weight_raster.close()
    LOGGER.info('%.1f%% complete', 100.0)


def extract_streams_mfd(
        flow_accum_raster_path_band, flow_dir_mfd_path_band,
        double flow_threshold, target_stream_raster_path,
        double trace_threshold_proportion=1.0,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Classify a stream raster from MFD flow accumulation.

    This function classifies pixels as streams that have a flow accumulation
    value >= `flow_threshold` and can trace further upstream with a fuzzy
    propotion if `trace_threshold_proportion` is set < 1.0

    Parameters:
        flow_accum_raster_path_band (tuple): a string/integer tuple indicating
            the flow accumulation raster to use as a basis for thresholding
            a stream. Values in this raster that are >= flow_threshold will
            be classified as streams. This raster should be derived from
            `dem_raster_path_band` using
            `pygeoprocessing.routing.flow_accumulation_mfd`.
        flow_dir_mfd_path_band (str): path to multiple flow direction
            raster, required to join divergent streams.
        flow_threshold (float): the value in `flow_accum_raster_path_band` to
            indicate where a stream exists.
        target_stream_raster_path (str): path to the target stream raster.
            This raster will be the same dimensions and projection as
            `dem_raster_path_band` and will contain 1s where a stream is
            defined, 0 where the flow accumulation layer is defined but no
            stream exists, and nodata otherwise.
        trace_threshold_proportion (float): this value indicates what
            proportion of the flow_threshold is enough to classify a pixel
            as a stream after the stream has been traced from a
            `flow_threshold` drain. Setting this value < 1.0 is useful for
            classifying streams in regions that have highly divergent flow
            directions.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None.
    """
    if trace_threshold_proportion < 0.or trace_threshold_proportion > 1.0:
        raise ValueError(
            "trace_threshold_proportion should be in the range [0.0, 1.0] "
            "actual value is: %s" % trace_threshold_proportion)

    flow_accum_info = pygeoprocessing.get_raster_info(
        flow_accum_raster_path_band[0])
    cdef double flow_accum_nodata = flow_accum_info['nodata'][
        flow_accum_raster_path_band[1]-1]
    stream_nodata = 255

    cdef int raster_x_size, raster_y_size
    raster_x_size, raster_y_size = flow_accum_info['raster_size']

    pygeoprocessing.new_raster_from_base(
        flow_accum_raster_path_band[0], target_stream_raster_path,
        gdal.GDT_Byte, [stream_nodata], fill_value_list=[stream_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)

    cdef _ManagedRaster flow_accum_mr = _ManagedRaster(
        flow_accum_raster_path_band[0], flow_accum_raster_path_band[1], 0)
    cdef _ManagedRaster stream_mr = _ManagedRaster(
        target_stream_raster_path, 1, 1)
    cdef _ManagedRaster flow_dir_mfd_mr = _ManagedRaster(
        flow_dir_mfd_path_band[0], flow_dir_mfd_path_band[1], 0)

    cdef int xoff, yoff, win_xsize, win_ysize
    cdef int xi, yi, xi_root, yi_root, i_n, xi_n, yi_n, i_sn, xi_sn, yi_sn
    cdef int flow_dir_mfd
    cdef double flow_accum
    cdef double trace_flow_threshold = (
        trace_threshold_proportion * flow_threshold)
    cdef int n_iterations = 0
    cdef int is_outlet, stream_val

    cdef int flow_dir_nodata = pygeoprocessing.get_raster_info(
        flow_dir_mfd_path_band[0])['nodata'][flow_dir_mfd_path_band[1]-1]

    # this queue is used to march the front from the stream pixel or the
    # backwards front for tracing downstream
    cdef CoordinateQueueType open_set, backtrace_set
    cdef int xi_bn, yi_bn # used for backtrace neighbor coordinates

    cdef time_t last_log_time = ctime(NULL)

    for block_offsets in pygeoprocessing.iterblocks(
            (target_stream_raster_path, 1), offset_only=True):
        xoff = block_offsets['xoff']
        yoff = block_offsets['yoff']
        win_xsize = block_offsets['win_xsize']
        win_ysize = block_offsets['win_ysize']
        for yi in range(win_ysize):
            yi_root = yi+yoff
            if ctime(NULL) - last_log_time > 5.0:
                last_log_time = ctime(NULL)
                current_pixel = xoff + yoff * raster_x_size
                LOGGER.info('%.1f%% complete', 100.0 * current_pixel / <float>(
                    raster_x_size * raster_y_size))
            for xi in range(win_xsize):
                xi_root = xi+xoff
                flow_accum = flow_accum_mr.get(xi_root, yi_root)
                if is_close(flow_accum, flow_accum_nodata):
                    continue
                if stream_mr.get(xi_root, yi_root) != stream_nodata:
                    continue
                stream_mr.set(xi_root, yi_root, 0)
                if flow_accum < flow_threshold:
                    continue

                flow_dir_mfd = <int>flow_dir_mfd_mr.get(xi_root, yi_root)
                is_outlet = 0
                for i_n in range(8):
                    if ((flow_dir_mfd >> (i_n * 4)) & 0xF) == 0:
                        # no flow in that direction
                        continue
                    xi_n = xi_root+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                    yi_n = yi_root+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                    if (xi_n < 0 or xi_n >= raster_x_size or
                            yi_n < 0 or yi_n >= raster_y_size):
                        # it'll drain off the edge of the raster
                        is_outlet = 1
                        break
                    if flow_accum_mr.get(xi_n, yi_n) == flow_accum_nodata:
                        is_outlet = 1
                        break
                if is_outlet:
                    open_set.push(CoordinateType(xi_root, yi_root))
                    stream_mr.set(xi_root, yi_root, 1)

                n_iterations = 0
                while open_set.size() > 0:
                    xi_n = open_set.front().xi
                    yi_n = open_set.front().yi
                    open_set.pop()
                    n_iterations += 1
                    for i_sn in range(8):
                        xi_sn = xi_n+NEIGHBOR_OFFSET_ARRAY[2*i_sn]
                        yi_sn = yi_n+NEIGHBOR_OFFSET_ARRAY[2*i_sn+1]
                        if (xi_sn < 0 or xi_sn >= raster_x_size or
                                yi_sn < 0 or yi_sn >= raster_y_size):
                            continue
                        flow_dir_mfd = <int>flow_dir_mfd_mr.get(xi_sn, yi_sn)
                        if flow_dir_mfd == flow_dir_nodata:
                            continue
                        if ((flow_dir_mfd >>
                                (D8_REVERSE_DIRECTION[i_sn] * 4)) & 0xF) > 0:
                            # upstream pixel flows into this one
                            stream_val = <int>stream_mr.get(xi_sn, yi_sn)
                            if stream_val != 1 and stream_val != 2:
                                flow_accum = flow_accum_mr.get(
                                    xi_sn, yi_sn)
                                if flow_accum >= flow_threshold:
                                    stream_mr.set(xi_sn, yi_sn, 1)
                                    open_set.push(
                                        CoordinateType(xi_sn, yi_sn))
                                    # see if we're in a potential stream and
                                    # found a connection
                                    backtrace_set.push(
                                        CoordinateType(xi_sn, yi_sn))
                                    while backtrace_set.size() > 0:
                                        xi_bn = backtrace_set.front().xi
                                        yi_bn = backtrace_set.front().yi
                                        backtrace_set.pop()
                                        flow_dir_mfd = <int>(
                                            flow_dir_mfd_mr.get(xi_bn, yi_bn))
                                        for i_sn in range(8):
                                            if (flow_dir_mfd >> (i_sn*4)) & 0xF > 0:
                                                xi_sn = xi_bn+NEIGHBOR_OFFSET_ARRAY[2*i_sn]
                                                yi_sn = yi_bn+NEIGHBOR_OFFSET_ARRAY[2*i_sn+1]
                                                if (xi_sn < 0 or xi_sn >= raster_x_size or
                                                        yi_sn < 0 or yi_sn >= raster_y_size):
                                                    continue
                                                if stream_mr.get(xi_sn, yi_sn) == 2:
                                                    stream_mr.set(xi_sn, yi_sn, 1)
                                                    backtrace_set.push(
                                                        CoordinateType(xi_sn, yi_sn))
                                elif flow_accum >= trace_flow_threshold:
                                    stream_mr.set(xi_sn, yi_sn, 2)
                                    open_set.push(
                                        CoordinateType(xi_sn, yi_sn))

    stream_mr.close()
    LOGGER.info('filter out incomplete divergent streams')
    block_offsets_list = list(pygeoprocessing.iterblocks(
        (target_stream_raster_path, 1), offset_only=True))
    stream_raster = gdal.OpenEx(
        target_stream_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    stream_band = stream_raster.GetRasterBand(1)
    for block_offsets in block_offsets_list:
        stream_array = stream_band.ReadAsArray(**block_offsets)
        stream_array[stream_array == 2] = 0
        stream_band.WriteArray(
            stream_array, xoff=block_offsets['xoff'],
            yoff=block_offsets['yoff'])
    stream_band = None
    stream_raster = None
    LOGGER.info('100.0% complete')


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
