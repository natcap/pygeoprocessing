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

D8 flow direction conventions encode the flow direction as::

     3 2 1
     4 x 0
     5 6 7

This is slightly different from how TauDEM encodes flow direction, which is as::
    4 3 2
    5 x 1
    6 7 8

To convert a TauDEM flow direction raster to a pygeoprocessing-compatible flow
direction raster, the following ``raster_map`` call may be used::

    taudem_flow_dir_path = 'taudem_d8_flow_dir.tif'
    pygeoprocessing.raster_map(
        lambda d: d+1, [taudem_flow_dir_path],
        'pygeoprocessing_d8_flow_dir.tif')
"""
import collections
import logging
import os
import shutil
import tempfile
import time

cimport cython
cimport numpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.time cimport time as ctime
from libc.time cimport time_t
from libc.math cimport isnan
from libcpp.deque cimport deque
from libcpp.list cimport list as clist
from libcpp.pair cimport pair
from libcpp.queue cimport queue
from libcpp.set cimport set as cset
from libcpp.stack cimport stack
from libcpp.vector cimport vector
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import shapely.wkb
import shapely.ops
import scipy.stats

from ..geoprocessing_core import DEFAULT_OSR_AXIS_MAPPING_STRATEGY
import pygeoprocessing

LOGGER = logging.getLogger(__name__)

cdef float _LOGGING_PERIOD = 10.0

# This module creates rasters with a memory xy block size of 2**BLOCK_BITS
cdef int BLOCK_BITS = 8

# Number of raster blocks to hold in memory at once per Managed Raster
cdef int MANAGED_RASTER_N_BLOCKS = 2**6

# these are the creation options that'll be used for all the rasters
DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS = ('GTiff', (
    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
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
cdef int *D8_XOFFSET = [1, 1, 0, -1, -1, -1, 0, 1]
cdef int *D8_YOFFSET = [0, -1, -1, -1, 0, +1, +1, +1]

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
    unsigned int xi  # pixel x coordinate in the raster
    unsigned int yi  # pixel y coordinate in the raster
    int priority # for breaking ties if two `value`s are equal.

# this struct is used to record an intermediate flow pixel's last calculated
# direction and the flow accumulation value so far
cdef struct FlowPixelType:
    unsigned int xi
    unsigned int yi
    int last_flow_dir
    double value

cdef struct DecayingValue:
    double decayed_value
    double min_value

cdef struct WeightedFlowPixelType:
    int xi
    int yi
    int last_flow_dir
    double value
    queue[DecayingValue] decaying_values


# this struct is used in distance_to_channel_mfd to add up each pixel's
# weighted distances and flow weights
cdef struct MFDFlowPixelType:
    unsigned int xi
    unsigned int yi
    int last_flow_dir
    double sum_of_weighted_distances
    double sum_of_weights

# used when constructing geometric streams, the x/y coordinates represent
# a seed point to walk upstream from, the upstream_d8_dir indicates the
# d8 flow direction to walk and the source_id indicates the source stream it
# spawned from
cdef struct StreamConnectivityPoint:
    unsigned int xi
    unsigned int yi
    int upstream_d8_dir
    unsigned int source_id

# used to record x/y locations as needed
cdef struct CoordinateType:
    unsigned int xi
    unsigned int yi


cdef struct FinishType:
    unsigned int xi
    unsigned int yi
    unsigned int n_pushed

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

cdef int _is_close(double x, double y, double abs_delta, double rel_delta):
    if isnan(x) and isnan(y):
        return 1
    return abs(x-y) <= (abs_delta+rel_delta*abs(y))

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
    cdef unsigned int raster_x_size
    cdef unsigned int raster_y_size
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
            n_attempts = 5
            while True:
                raster = gdal.OpenEx(
                    self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
                if raster is None:
                    if n_attempts == 0:
                        raise RuntimeError(
                            f'could not open {self.raster_path} for writing')
                    LOGGER.warning(
                        f'opening {self.raster_path} resulted in null, '
                        f'trying {n_attempts} more times.')
                    n_attempts -= 1
                    time.sleep(0.5)
                raster_band = raster.GetRasterBand(self.band_id)
                break

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
            max_retries = 5
            while max_retries > 0:
                raster = gdal.OpenEx(
                    self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
                if raster is None:
                    max_retries -= 1
                    LOGGER.error(
                        f'unable to open {self.raster_path}, retrying...')
                    time.sleep(0.2)
                    continue
                break
            if max_retries == 0:
                raise ValueError(
                    f'unable to open {self.raster_path} in '
                    'ManagedRaster.flush')
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
        long long max_pixel_fill_count=-1,
        single_outlet_tuple=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Fill the pits in a DEM.

    This function defines pits as hydrologically connected regions that do
    not drain to the edge of the raster or a nodata pixel. After the call
    pits are filled to the height of the lowest pour point.

    Parameters:
        dem_raster_path_band (tuple): a path, band number tuple indicating the
            DEM calculate flow direction.
        target_filled_dem_raster_path (str): path the pit filled dem,
            that's created by a call to this function. It is functionally a
            single band copy of ``dem_raster_path_band`` with the pit pixels
            raised to the pour point. For runtime efficiency, this raster is
            tiled and its blocksize is set to (``1<<BLOCK_BITS``,
            ``1<<BLOCK_BITS``)
            even if ``dem_raster_path_band[0]`` was not tiled or a different
            block size.
        working_dir (str): If not None, indicates where temporary files
            should be created during this run. If this directory doesn't exist
            it is created by this call. If None, a temporary directory is
            created by tempdir.mkdtemp which is removed after the function
            call completes successfully.
        max_pixel_fill_count (int): maximum number of pixels to fill a pit
            before leaving as a depression. Useful if there are natural
            large depressions. Value of -1 fills the raster with no search
            limit.
        single_outlet_tuple (tuple): If not None, this is an x/y tuple in
            raster coordinates indicating the only pixel that can be
            considered a drain. If None then any pixel that would drain to
            the edge of the raster or a nodata hole will be considered a
            drain.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None.
    """
    # These variables are used to iterate over the DEM using `iterblock`
    # indexes
    cdef unsigned int win_ysize, win_xsize, xoff, yoff

    # the _root variables remembers the pixel index where the plateau/pit
    # region was first detected when iterating over the DEM.
    cdef unsigned int xi_root, yi_root

    # these variables are used as pixel or neighbor indexes. where _q
    # represents a value out of a queue, and _n is related to a neighbor pixel
    cdef unsigned int i_n, xi, yi, xi_q, yi_q, xi_n, yi_n

    # these are booleans used to remember the condition that caused a loop
    # to terminate, though downhill and nodata are equivalent for draining,
    # i keep them separate for cognitive readability.
    cdef int downhill_neighbor, nodata_neighbor, natural_drain_exists

    # keep track of how many steps searched on the pit to test against
    # max_pixel_fill_count
    cdef long long search_steps

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
    cdef unsigned int raster_x_size, raster_y_size, n_x_blocks

    # variables to remember heights of DEM
    cdef double center_val, dem_nodata, fill_height

    # used to uniquely identify each flat/pit region encountered in the
    # algorithm, it's written into the mask rasters to indicate which pixels
    # have already been processed
    cdef int feature_id

    cdef unsigned long current_pixel

    # used to handle the case for single outlet mode
    cdef int single_outlet=0, outlet_x=-1, outlet_y=-1
    if single_outlet_tuple is not None:
        single_outlet = 1
        outlet_x, outlet_y = single_outlet_tuple

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
        [mask_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    flat_region_mask_managed_raster = _ManagedRaster(
        flat_region_mask_path, 1, 1)

    # this raster will have the value of 'feature_id' set to it if it has
    # been searched as part of the search for a pour point for pit number
    # `feature_id`
    pit_mask_path = os.path.join(working_dir_path, 'pit_mask.tif')
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], pit_mask_path, gdal.GDT_Int32,
        [mask_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    pit_mask_managed_raster = _ManagedRaster(
        pit_mask_path, 1, 1)

    # copy the base DEM to the target and set up for writing
    base_datatype = pygeoprocessing.get_raster_info(
        dem_raster_path_band[0])['datatype']
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], target_filled_dem_raster_path,
        base_datatype, [dem_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    filled_dem_raster = gdal.OpenEx(
        target_filled_dem_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    filled_dem_band = filled_dem_raster.GetRasterBand(1)
    for offset_info, block_array in pygeoprocessing.iterblocks(
                dem_raster_path_band):
        filled_dem_band.WriteArray(
            block_array, xoff=offset_info['xoff'], yoff=offset_info['yoff'])
    filled_dem_band.FlushCache()
    filled_dem_raster.FlushCache()
    filled_dem_band = None
    filled_dem_raster = None
    filled_dem_managed_raster = _ManagedRaster(
        target_filled_dem_raster_path, 1, 1)

    # feature_id will start at 1 since the mask nodata is 0.
    feature_id = 0

    # this outer loop searches for a pixel that is locally undrained
    for offset_dict in pygeoprocessing.iterblocks(
            dem_raster_path_band, offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info(
                '(fill pits): '
                f'{current_pixel} of {raster_x_size * raster_y_size} '
                'pixels complete')

        # search block for locally undrained pixels
        for yi in range(win_ysize):
            yi_root = yi+yoff
            for xi in range(win_xsize):
                xi_root = xi+xoff
                # this value is set in case it turns out to be the root of a
                # pit, we'll start the fill from this pixel in the last phase
                # of the algorithm
                center_val = filled_dem_managed_raster.get(xi_root, yi_root)
                if _is_close(center_val, dem_nodata, 1e-8, 1e-5):
                    continue

                if flat_region_mask_managed_raster.get(
                        xi_root, yi_root) != mask_nodata:
                    continue

                # a single outlet trivially drains
                if (single_outlet and
                        xi_root == outlet_x and
                        yi_root == outlet_y):
                    continue

                # search neighbors for downhill or nodata
                downhill_neighbor = 0
                nodata_neighbor = 0
                for i_n in range(8):
                    xi_n = xi_root+D8_XOFFSET[i_n]
                    yi_n = yi_root+D8_YOFFSET[i_n]

                    if (xi_n < 0 or xi_n >= raster_x_size or
                            yi_n < 0 or yi_n >= raster_y_size):
                        if not single_outlet:
                            # it'll drain off the edge of the raster
                            nodata_neighbor = 1
                            break
                        else:
                            # continue so we don't access out of bounds
                            continue
                    n_height = filled_dem_managed_raster.get(xi_n, yi_n)
                    if _is_close(n_height, dem_nodata, 1e-8, 1e-5):
                        if not single_outlet:
                            # it'll drain to nodata
                            nodata_neighbor = 1
                            break
                        else:
                            # skip the rest so it doesn't drain downhill
                            continue
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
                natural_drain_exists = 0

                # this loop does a BFS starting at this pixel to all pixels
                # of the same height. the _drain variables are used to
                # remember if a drain was encountered. it is preferable to
                # search the whole region even if a drain is encountered, so
                # it can be entirely marked as processed and not re-accessed
                # on later iterations
                search_steps = 0
                while not search_queue.empty():
                    xi_q = search_queue.front().xi
                    yi_q = search_queue.front().yi
                    search_queue.pop()
                    if (max_pixel_fill_count > 0 and
                            search_steps > max_pixel_fill_count):
                        # clear the search queue and quit
                        LOGGER.debug(
                            'exceeded max pixel fill count when searching '
                            'for plateau drain')
                        while not search_queue.empty():
                            search_queue.pop()
                        natural_drain_exists = 1
                        break
                    search_steps += 1

                    for i_n in range(8):
                        xi_n = xi_q+D8_XOFFSET[i_n]
                        yi_n = yi_q+D8_YOFFSET[i_n]

                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            if not single_outlet:
                                natural_drain_exists = 1
                            continue

                        n_height = filled_dem_managed_raster.get(
                            xi_n, yi_n)
                        if _is_close(n_height, dem_nodata, 1e-8, 1e-5):
                            if not single_outlet:
                                natural_drain_exists = 1
                            continue
                        if n_height < center_val:
                            natural_drain_exists = 1
                            continue

                        if flat_region_mask_managed_raster.get(
                                xi_n, yi_n) == 1:
                            # been set before on a previous iteration, skip
                            continue
                        if n_height == center_val:
                            # only grow if it's at the same level and not
                            # previously visited
                            search_queue.push(
                                CoordinateType(xi_n, yi_n))
                            flat_region_mask_managed_raster.set(
                                xi_n, yi_n, 1)

                if not natural_drain_exists:
                    # this space does not naturally drain, so fill it
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
                search_steps = 0
                while not pit_queue.empty():
                    pixel = pit_queue.top()
                    pit_queue.pop()
                    xi_q = pixel.xi
                    yi_q = pixel.yi

                    # search to see if the fill has gone on too long
                    if (max_pixel_fill_count > 0 and
                            search_steps > max_pixel_fill_count):
                        # clear pit_queue and quit
                        LOGGER.debug(
                            'exceeded max pixel fill count when searching '
                            'for pour point')
                        pit_queue = PitPriorityQueueType()
                        natural_drain_exists = 1
                        break
                    search_steps += 1

                    # this is the potential fill height if pixel is pour point
                    fill_height = pixel.value

                    for i_n in range(8):
                        xi_n = xi_q+D8_XOFFSET[i_n]
                        yi_n = yi_q+D8_YOFFSET[i_n]

                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            # drain off the edge of the raster
                            if not single_outlet:
                                pour_point = 1
                                break
                            else:
                                continue

                        if pit_mask_managed_raster.get(
                                xi_n, yi_n) == feature_id:
                            # this cell has already been processed
                            continue
                        # mark as visited in the search for pour point
                        pit_mask_managed_raster.set(xi_n, yi_n, feature_id)

                        n_height = filled_dem_managed_raster.get(xi_n, yi_n)
                        if (single_outlet and xi_n == outlet_x
                                and yi_n == outlet_y):
                            fill_height = n_height
                            pour_point = 1
                            break

                        if _is_close(n_height, dem_nodata, 1e-8, 1e-5):
                            # we encounter a neighbor not processed that
                            # is nodata
                            if not single_outlet:
                                # it's only a pour point if we aren't in
                                # single outlet mode
                                pour_point = 1
                            # skip so we don't go negative
                            continue
                        if n_height < fill_height:
                            # we encounter a neighbor not processed that is
                            # lower than the current pixel
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
                        xi_n = xi_q+D8_XOFFSET[i_n]
                        yi_n = yi_q+D8_YOFFSET[i_n]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            continue
                        n_height = filled_dem_managed_raster.get(xi_n, yi_n)
                        if _is_close(n_height, dem_nodata, 1e-8, 1e-5):
                            continue
                        if n_height < fill_height:
                            filled_dem_managed_raster.set(
                                xi_n, yi_n, fill_height)
                            fill_queue.push(CoordinateType(xi_n, yi_n))

    pit_mask_managed_raster.close()
    flat_region_mask_managed_raster.close()
    shutil.rmtree(working_dir_path)
    LOGGER.info('(fill pits): complete')


def flow_dir_d8(
        dem_raster_path_band, target_flow_dir_path,
        working_dir=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """D8 flow direction.

    Parameters:
        dem_raster_path_band (tuple): a path, band number tuple indicating the
            DEM calculate flow direction. This DEM must not have hydrological
            pits or else the target flow direction is undefined.
        target_flow_dir_path (str): path to a byte raster created by this
            call of same dimensions as ``dem_raster_path_band`` that has a value
            indicating the direction of downhill flow. Values are defined as
            pointing to one of the eight neighbors with the following
            convention::

                3 2 1
                4 x 0
                5 6 7

        working_dir (str): If not None, indicates where temporary files
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
    cdef unsigned int win_ysize, win_xsize, xoff, yoff

    # the _root variables remembers the pixel index where the plateau/pit
    # region was first detected when iterating over the DEM.
    cdef unsigned int xi_root, yi_root

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
    cdef unsigned int raster_x_size, raster_y_size

    cdef unsigned long current_pixel

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
        [mask_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    flat_region_mask_managed_raster = _ManagedRaster(
        flat_region_mask_path, 1, 1)

    flow_dir_nodata = 128
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], target_flow_dir_path, gdal.GDT_Byte,
        [flow_dir_nodata],
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

        if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info(
                '(flow dir d8): '
                f'{current_pixel} of {raster_x_size*raster_y_size} '
                f'pixels complete')

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
                if _is_close(root_height, dem_nodata, 1e-8, 1e-5):
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
                    xi_n = xi+D8_XOFFSET[i_n]
                    yi_n = yi+D8_YOFFSET[i_n]
                    n_height = dem_buffer_array[yi_n, xi_n]
                    if _is_close(n_height, dem_nodata, 1e-8, 1e-5):
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
                        xi_n = xi_q+D8_XOFFSET[i_n]
                        yi_n = yi_q+D8_YOFFSET[i_n]

                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            n_height = dem_nodata
                        else:
                            n_height = dem_managed_raster.get(xi_n, yi_n)
                        if _is_close(n_height, dem_nodata, 1e-8, 1e-5):
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
                        xi_n = xi_q+D8_XOFFSET[i_n]
                        yi_n = yi_q+D8_YOFFSET[i_n]
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
    LOGGER.info('(flow dir d8): complete')


def flow_accumulation_d8(
        flow_dir_raster_path_band, target_flow_accum_raster_path,
        weight_raster_path_band=None, custom_decay_factor=None,
        double min_decay_proportion=0.001,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """D8 flow accumulation.

    Parameters:
        flow_dir_raster_path_band (tuple): a path, band number tuple
            for a flow accumulation raster whose pixels indicate the flow
            out of a pixel in one of 8 directions in the following
            configuration::

                3 2 1
                4 x 0
                5 6 7

        target_flow_accum_raster_path (str): path to flow
            accumulation raster created by this call. After this call, the
            value of each pixel will be 1 plus the number of upstream pixels
            that drain to that pixel. Note the target type of this raster
            is a 64 bit float so there is minimal risk of overflow and the
            possibility of handling a float dtype in
            ``weight_raster_path_band``.
        weight_raster_path_band (tuple): optional path and band number to a
            raster that will be used as the per-pixel flow accumulation
            weight. If ``None``, 1 is the default flow accumulation weight.
            This raster must be the same dimensions as
            ``flow_dir_mfd_raster_path_band``.
        custom_decay_factor=None (float or str): a custom decay factor, either
            represented as a float (where the same decay factor is applied to
            all valid pixels) or a raster where pixel values represent
            spatially-explicit decay values.
        min_decay_proportion=0.001 (float): A value representing the minimum
            decayed value that should continue to be tracked along the flow
            path when using a custom decay factor.  If the upstream decayed
            contribution falls below this value, it is not included.
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
    cdef unsigned int win_ysize, win_xsize, xoff, yoff

    # the _root variables remembers the pixel index where the plateau/pit
    # region was first detected when iterating over the DEM.
    cdef unsigned int xi_root, yi_root

    # these variables are used as pixel or neighbor indexes.
    # _n is related to a neighbor pixel
    cdef unsigned int i_n, xi, yi, xi_n, yi_n

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
    cdef stack[WeightedFlowPixelType] search_stack
    cdef WeightedFlowPixelType flow_pixel

    # properties of the parallel rasters
    cdef unsigned int raster_x_size, raster_y_size

    cdef unsigned long current_pixel

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

    cdef short do_decayed_accumulation = False
    cdef short use_const_decay_factor = False
    cdef float decay_factor = 1.0
    cdef double min_allowed_decayed_load, local_decay_factor
    cdef double on_pixel_load
    if custom_decay_factor is not None:
        do_decayed_accumulation = True
        if isinstance(custom_decay_factor, (int, float)):
            decay_factor = custom_decay_factor
            use_const_decay_factor = True
        else:  # assume a path/band tuple
            if not _is_raster_path_band_formatted(custom_decay_factor):
                raise ValueError(
                    "%s is supposed to be a raster band tuple but it's not." % (
                        custom_decay_factor))
            decay_factor_managed_raster = _ManagedRaster(
                custom_decay_factor[0], custom_decay_factor[1], 0)

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
    cdef queue[DecayingValue] decay_from_upstream
    cdef double upstream_transport_factor = 1.0  # proportion of load transported to downstream pixel
    for offset_dict in pygeoprocessing.iterblocks(
            flow_dir_raster_path_band, offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('Flow accumulation D8 %.1f%% complete', 100.0 * current_pixel / <float>(
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

                xi_n = xi+D8_XOFFSET[flow_dir]
                yi_n = yi+D8_YOFFSET[flow_dir]

                if flow_dir_buffer_array[yi_n, xi_n] == flow_dir_nodata:
                    xi_root = xi-1+xoff
                    yi_root = yi-1+yoff

                    if weight_raster is not None:
                        weight_val = <double>weight_raster.get(
                            xi_root, yi_root)
                        if _is_close(weight_val, weight_nodata, 1e-8, 1e-5):
                            weight_val = 0.0
                    else:
                        weight_val = 1.0
                    search_stack.push(
                        WeightedFlowPixelType(xi_root, yi_root, 0, weight_val,
                                              queue[DecayingValue]()))

                # Drain the queue of upstream neighbors since we're starting
                # from a new root pixel.
                while not decay_from_upstream.empty():
                    decay_from_upstream.pop()

                while not search_stack.empty():
                    flow_pixel = search_stack.top()
                    search_stack.pop()

                    # Empty when not doing decaying flow accumulation.
                    while not decay_from_upstream.empty():
                        upstream_decaying_value = decay_from_upstream.front()
                        decay_from_upstream.pop()
                        if upstream_decaying_value.decayed_value > upstream_decaying_value.min_value:
                            flow_pixel.decaying_values.push(upstream_decaying_value)

                    upstream_pixels_remain = 0
                    for i_n in range(flow_pixel.last_flow_dir, 8):
                        xi_n = flow_pixel.xi+D8_XOFFSET[i_n]
                        yi_n = flow_pixel.yi+D8_YOFFSET[i_n]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            # neighbor not upstream: off edges of the raster
                            continue
                        upstream_flow_dir = <int>flow_dir_managed_raster.get(
                            xi_n, yi_n)
                        if upstream_flow_dir == flow_dir_nodata or (
                                upstream_flow_dir != D8_REVERSE_DIRECTION[i_n]):
                            # neighbor not upstream: nodata or doesn't flow in
                            continue

                        if do_decayed_accumulation:
                            if use_const_decay_factor:
                                upstream_transport_factor = decay_factor
                            else:
                                upstream_transport_factor = (
                                    decay_factor_managed_raster.get(xi_n, yi_n))
                                # TODO: how to handle nodata here?  Assume decay factor of 0?

                        upstream_flow_accum = <double>(
                            flow_accum_managed_raster.get(xi_n, yi_n))
                        if _is_close(upstream_flow_accum, flow_accum_nodata, 1e-8, 1e-5):
                            # process upstream before this one
                            # Flow accumulation pixel is nodata until it and everything
                            # upstream of it has been computed.
                            flow_pixel.last_flow_dir = i_n
                            search_stack.push(flow_pixel)
                            if weight_raster is not None:
                                weight_val = <double>weight_raster.get(
                                    xi_n, yi_n)
                                if _is_close(weight_val, weight_nodata, 1e-8, 1e-5):
                                    weight_val = 0.0
                            else:
                                weight_val = 1.0
                            flow_pixel.last_flow_dir = i_n
                            if do_decayed_accumulation:
                                flow_pixel.decaying_values.push(
                                    DecayingValue(weight_val,
                                                  weight_val * min_decay_proportion))
                            search_stack.push(
                                WeightedFlowPixelType(xi_n, yi_n, 0, weight_val,
                                                      queue[DecayingValue]()))
                            upstream_pixels_remain = 1
                            break

                        if not do_decayed_accumulation:
                            flow_pixel.value += upstream_flow_accum

                    if not upstream_pixels_remain:
                        # flow_pixel.value already has the on-pixel load
                        # from upstream, so we just need to add it from the
                        # decaying values queue
                        if do_decayed_accumulation:
                            while not flow_pixel.decaying_values.empty():
                                decayed_value = flow_pixel.decaying_values.front()
                                flow_pixel.decaying_values.pop()
                                decayed_value.decayed_value *= upstream_transport_factor
                                flow_pixel.value += decayed_value.decayed_value
                                decay_from_upstream.push(decayed_value)

                        flow_accum_managed_raster.set(
                            flow_pixel.xi, flow_pixel.yi,
                            flow_pixel.value)
    flow_accum_managed_raster.close()
    flow_dir_managed_raster.close()
    if do_decayed_accumulation:
        if not use_const_decay_factor:
            decay_factor_managed_raster.close()
    if weight_raster is not None:
        weight_raster.close()
    LOGGER.info('Flow accumulation D8 %.1f%% complete', 100.0)


def flow_dir_mfd(
        dem_raster_path_band, target_flow_dir_path, working_dir=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Multiple flow direction.

    Parameters:
        dem_raster_path_band (tuple): a path, band number tuple indicating the
            DEM calculate flow direction. This DEM must not have hydrological
            pits or else the target flow direction will be undefined.
        target_flow_dir_path (str): path to a raster created by this call
            of a 32 bit int raster of the same dimensions and projections as
            ``dem_raster_path_band[0]``. The value of the pixel indicates the
            proportion of flow from that pixel to its neighbors given these
            indexes::

                3 2 1
                4 x 0
                5 6 7

            The pixel value is formatted as 8 separate 4 bit integers
            compressed into a 32 bit int. To extract the proportion of flow
            from a particular direction given the pixel value 'x' one can
            shift and mask as follows ``0xF & (x >> (4*dir))``, where ``dir``
            is one of the 8 directions indicated above.
        working_dir (str): If not None, indicates where temporary files
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
    cdef unsigned int xi_root, yi_root

    # these variables are used as pixel or neighbor indexes. where _q
    # represents a value out of a queue, and _n is related to a neighbor pixel
    cdef unsigned int i_n, xi, yi, xi_q, yi_q, xi_n, yi_n

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
    cdef unsigned long raster_x_size, raster_y_size

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    # This is used in progress logging to represent how many pixels have been
    # visited so far.
    cdef unsigned long current_pixel

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
        [mask_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    flat_region_mask_managed_raster = _ManagedRaster(
        flat_region_mask_path, 1, 1)

    flow_dir_nodata = 0
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], target_flow_dir_path, gdal.GDT_Int32,
        [flow_dir_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    flow_dir_managed_raster = _ManagedRaster(target_flow_dir_path, 1, 1)

    plateu_drain_mask_path = os.path.join(
        working_dir_path, 'plateu_drain_mask.tif')
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], plateu_drain_mask_path, gdal.GDT_Byte,
        [mask_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    plateau_drain_mask_managed_raster = _ManagedRaster(
        plateu_drain_mask_path, 1, 1)

    # this creates a raster that's used for a dynamic programming solution to
    # shortest path to the drain for plateaus. the raster is filled with
    # raster_x_size * raster_y_size as a distance that's greater than the
    # longest plateau drain distance possible for this raster.
    plateau_distance_path = os.path.join(
        working_dir_path, 'plateau_distance.tif')
    cdef unsigned long plateau_distance_nodata = raster_x_size * raster_y_size
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], plateau_distance_path, gdal.GDT_Float64,
        [plateau_distance_nodata],
        fill_value_list=[plateau_distance_nodata],
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

        if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
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
                if _is_close(root_height, dem_nodata, 1e-8, 1e-5):
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
                    xi_n = xi+D8_XOFFSET[i_n]
                    yi_n = yi+D8_YOFFSET[i_n]
                    n_height = dem_buffer_array[yi_n, xi_n]
                    if _is_close(n_height, dem_nodata, 1e-8, 1e-5):
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
                        xi_n = xi_q+D8_XOFFSET[i_n]
                        yi_n = yi_q+D8_YOFFSET[i_n]

                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            n_height = dem_nodata
                        else:
                            n_height = dem_managed_raster.get(xi_n, yi_n)
                        if _is_close(n_height, dem_nodata, 1e-8, 1e-5):
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
                        xi_n = xi_q+D8_XOFFSET[i_n]
                        yi_n = yi_q+D8_YOFFSET[i_n]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            continue

                        n_drain_distance = drain_distance + (
                            SQRT2 if i_n & 1 else 1.0)

                        if (<double>(dem_managed_raster.get(
                                xi_n, yi_n)) == root_height) and (
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
                        xi_n = xi_q+D8_XOFFSET[i_n]
                        yi_n = yi_q+D8_YOFFSET[i_n]
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
    LOGGER.info('Flow dir MFD %.1f%% complete', 100.0)


def flow_accumulation_mfd(
        flow_dir_mfd_raster_path_band, target_flow_accum_raster_path,
        weight_raster_path_band=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Multiple flow direction accumulation.

    Parameters:
        flow_dir_mfd_raster_path_band (tuple): a path, band number tuple
            for a multiple flow direction raster generated from a call to
            ``flow_dir_mfd``. The format of this raster is described in the
            docstring of that function.
        target_flow_accum_raster_path (str): a path to a raster created by
            a call to this function that is the same dimensions and projection
            as ``flow_dir_mfd_raster_path_band[0]``. The value in each pixel is
            1 plus the proportional contribution of all upstream pixels that
            flow into it. The proportion is determined as the value of the
            upstream flow dir pixel in the downslope direction pointing to
            the current pixel divided by the sum of all the flow weights
            exiting that pixel. Note the target type of this raster
            is a 64 bit float so there is minimal risk of overflow and the
            possibility of handling a float dtype in
            ``weight_raster_path_band``.
        weight_raster_path_band (tuple): optional path and band number to a
            raster that will be used as the per-pixel flow accumulation
            weight. If ``None``, 1 is the default flow accumulation weight.
            This raster must be the same dimensions as
            ``flow_dir_mfd_raster_path_band``. If a weight nodata pixel is
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
    cdef unsigned long win_ysize, win_xsize, xoff, yoff

    # These are used to estimate % complete
    cdef unsigned long long visit_count, pixel_count

    # the _root variables remembers the pixel index where the plateau/pit
    # region was first detected when iterating over the DEM.
    cdef unsigned long xi_root, yi_root

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
    cdef unsigned long raster_x_size, raster_y_size

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    cdef unsigned long current_pixel

    if not _is_raster_path_band_formatted(flow_dir_mfd_raster_path_band):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                flow_dir_mfd_raster_path_band))
    if weight_raster_path_band and not _is_raster_path_band_formatted(
            weight_raster_path_band):
        raise ValueError(
            "%s is supposed to be a raster band tuple but it's not." % (
                weight_raster_path_band))

    LOGGER.debug('creating target flow accum raster layer')
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_raster_path_band[0], target_flow_accum_raster_path,
        gdal.GDT_Float64, [flow_accum_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)

    flow_accum_managed_raster = _ManagedRaster(
        target_flow_accum_raster_path, 1, 1)

    # make a temporary raster to mark where we have visisted
    LOGGER.debug('creating visited raster layer')
    tmp_dir_root = os.path.dirname(target_flow_accum_raster_path)
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir_root, prefix='mfd_flow_dir_')
    visited_raster_path = os.path.join(tmp_dir, 'visited.tif')
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_raster_path_band[0], visited_raster_path,
        gdal.GDT_Byte, [0],
        raster_driver_creation_tuple=('GTiff', (
            'SPARSE_OK=TRUE', 'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS))))
    visited_managed_raster = _ManagedRaster(visited_raster_path, 1, 1)

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
    pixel_count = raster_x_size * raster_y_size
    visit_count = 0

    LOGGER.debug('starting search')
    # this outer loop searches for a pixel that is locally undrained
    for offset_dict in pygeoprocessing.iterblocks(
            flow_dir_mfd_raster_path_band, offset_only=True,
            largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('Flow accum MFD %.1f%% complete', 100.0 * current_pixel / <float>(
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
                    xi_n = xi+D8_XOFFSET[i_n]
                    yi_n = yi+D8_YOFFSET[i_n]

                    if flow_dir_mfd_buffer_array[yi_n, xi_n] == 0:
                        # if the entire value is zero, it flows nowhere
                        # and the root pixel is draining to it, thus the
                        # root must be a drain
                        xi_root = xi-1+xoff
                        yi_root = yi-1+yoff
                        if weight_raster is not None:
                            weight_val = <double>weight_raster.get(
                                xi_root, yi_root)
                            if _is_close(weight_val, weight_nodata, 1e-8, 1e-5):
                                weight_val = 0.0
                        else:
                            weight_val = 1.0
                        search_stack.push(
                            FlowPixelType(xi_root, yi_root, 0, weight_val))
                        visited_managed_raster.set(xi_root, yi_root, 1)
                        visit_count += 1
                        break

                while not search_stack.empty():
                    flow_pixel = search_stack.top()
                    search_stack.pop()

                    if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
                        last_log_time = ctime(NULL)
                        LOGGER.info(
                            'Flow accum MFD %.1f%% complete',
                            100.0 * visit_count / float(pixel_count))

                    preempted = 0
                    for i_n in range(flow_pixel.last_flow_dir, 8):
                        xi_n = flow_pixel.xi+D8_XOFFSET[i_n]
                        yi_n = flow_pixel.yi+D8_YOFFSET[i_n]
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
                        if (_is_close(upstream_flow_accum, flow_accum_nodata, 1e-8, 1e-5)
                                and not visited_managed_raster.get(
                                    xi_n, yi_n)):
                            # process upstream before this one
                            flow_pixel.last_flow_dir = i_n
                            search_stack.push(flow_pixel)
                            if weight_raster is not None:
                                weight_val = <double>weight_raster.get(
                                    xi_n, yi_n)
                                if _is_close(weight_val, weight_nodata, 1e-8, 1e-5):
                                    weight_val = 0.0
                            else:
                                weight_val = 1.0
                            search_stack.push(
                                FlowPixelType(xi_n, yi_n, 0, weight_val))
                            visited_managed_raster.set(xi_n, yi_n, 1)
                            visit_count += 1
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
    visited_managed_raster.close()
    try:
        shutil.rmtree(tmp_dir)
    except OSError:
        LOGGER.exception("couldn't remove temp dir")
    LOGGER.info('Flow accum MFD %.1f%% complete', 100.0)


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
            correspond to outflow in the following configuration::

                3 2 1
                4 x 0
                5 6 7

        channel_raster_path_band (tuple): a path/band tuple of the same
            dimensions and projection as ``flow_dir_d8_raster_path_band[0]``
            that indicates where the channels in the problem space lie. A
            channel is indicated if the value of the pixel is 1. Other values
            are ignored.
        target_distance_to_channel_raster_path (str): path to a raster
            created by this call that has per-pixel distances from a given
            pixel to the nearest downhill channel.
        weight_raster_path_band (tuple): optional path and band number to a
            raster that will be used as the per-pixel flow distance
            weight. If ``None``, 1 is the default distance between neighboring
            pixels. This raster must be the same dimensions as
            ``flow_dir_mfd_raster_path_band``.
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
    cdef unsigned int win_ysize, win_xsize, xoff, yoff

    # these variables are used as pixel or neighbor indexes.
    # _n is related to a neighbor pixel
    cdef unsigned int i_n, xi, yi, xi_q, yi_q, xi_n, yi_n

    # `distance_to_channel_stack` is the datastructure that walks upstream
    # from a defined flow distance pixel
    cdef stack[PixelType] distance_to_channel_stack

    # properties of the parallel rasters
    cdef unsigned int raster_x_size, raster_y_size

    # these area used to store custom per-pixel weights and per-pixel values
    # for distance updates
    cdef double weight_val, pixel_val
    cdef double weight_nodata = IMPROBABLE_FLOAT_NODATA

    cdef unsigned long current_pixel

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

        if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('Dist to channel D8 %.1f%% complete', 100.0 * current_pixel / <float>(
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
                        xi_n = xi_q+D8_XOFFSET[i_n]
                        yi_n = yi_q+D8_YOFFSET[i_n]

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
                                if _is_close(weight_val, weight_nodata, 1e-8, 1e-5):
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

    MFD distance to channel for a pixel ``i`` is the average distance that
    water travels from ``i`` until it reaches a stream, defined as::

              ⎧ 0,                                       if i is a stream pixel
              ⎪
              ⎪ undefined,   if i is not a stream and there are 0 elements in N
       d_i =  ⎨
              ⎪
              ⎪   ⎲ ((d_n + l(i, n)) * p(i, n),                       otherwise
              ⎪   ⎳
              ⎩  n ∈ N

        where
        - N is the set of immediate downstream neighbors of i that drain to
          a stream on the map
        - p(i, n) is the proportion of water on pixel i that flows onto n
        - l(i, n) is the distance between i and n. This is 1 for lateral
          neighbors, and √2 for diagonal neighbors.

    Distance is measured in units of pixel lengths. Pixels must be square, and
    both input rasters must have the same spatial reference, geotransform, and
    extent, so that their pixels overlap exactly.

    If a none of a pixel's flow drains to any stream on the map, ``d_i`` is
    undefined, and that pixel has nodata in the output.

    If only some of a pixel's flow reaches a stream on the map, ``d_i`` is
    defined in terms of only that portion of flow that reaches a stream. It is
    the average distance traveled by the water that does reach a stream.

    The algorithm begins from an arbitrary pixel and pushes downstream
    neighbors to the stack depth-first, until reaching a stream (the first case
    in the function for ``d_i`` above) or a raster edge (the second case).
    Non-stream pixels are only calculated once all their downstream neighbors
    have been calculated (the third case).

    Args:
        flow_dir_mfd_raster_path_band (tuple): a path/band index tuple
            indicating the raster that defines the mfd flow accumulation
            raster for this call. This raster should be generated by a call
            to ``pygeoprocessing.routing.flow_dir_mfd``.
        channel_raster_path_band (tuple): a path/band tuple of the same
            dimensions and projection as ``flow_dir_mfd_raster_path_band[0]``
            that indicates where the channels in the problem space lie. A
            channel is indicated if the value of the pixel is 1. Other values
            are ignored.
        target_distance_to_channel_raster_path (str): path to a raster
            created by this call that has per-pixel distances from a given
            pixel to the nearest downhill channel.
        weight_raster_path_band (tuple): optional path and band number to a
            raster that will be used as the per-pixel flow distance
            weight. If ``None``, 1 is the default distance between neighboring
            pixels. This raster must be the same dimensions as
            ``flow_dir_mfd_raster_path_band``.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at ``geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS``.

    Returns:
        None.
    """
    # These variables are used to iterate over the DEM using `iterblock`
    # indexes
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] channel_buffer_array
    cdef numpy.ndarray[numpy.int32_t, ndim=2] flow_dir_buffer_array
    cdef unsigned int win_ysize, win_xsize, xoff, yoff

    # these variables are used as pixel or neighbor indexes.
    # _n is related to a neighbor pixel
    cdef unsigned int i_n, xi, yi, xi_n, yi_n
    cdef int flow_dir_weight, mfd_value

    # used to remember if the current pixel is a channel for routing
    cdef int is_a_channel

    # `distance_to_channel_queue` is the data structure that walks upstream
    # from a defined flow distance pixel
    cdef stack[MFDFlowPixelType] distance_to_channel_stack

    # properties of the parallel rasters
    cdef unsigned int raster_x_size, raster_y_size

    # this value is used to store the current weight which might be 1 or
    # come from a predefined flow accumulation weight raster
    cdef double weight_val
    cdef double weight_nodata = IMPROBABLE_FLOAT_NODATA

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    cdef unsigned long current_pixel

    for path in (
            flow_dir_mfd_raster_path_band, channel_raster_path_band,
            weight_raster_path_band):
        if path is not None and not _is_raster_path_band_formatted(path):
            raise ValueError(
                "%s is supposed to be a raster band tuple but it's not." % (
                    path))

    distance_nodata = -1
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_raster_path_band[0],
        target_distance_to_channel_raster_path,
        gdal.GDT_Float64, [distance_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    distance_to_channel_managed_raster = _ManagedRaster(
        target_distance_to_channel_raster_path, 1, 1)
    channel_managed_raster = _ManagedRaster(
        channel_raster_path_band[0], channel_raster_path_band[1], 0)

    tmp_work_dir = tempfile.mkdtemp(
        suffix=None, prefix='dist_to_channel_mfd_work_dir',
        dir=os.path.dirname(target_distance_to_channel_raster_path))
    visited_raster_path = os.path.join(tmp_work_dir, 'visited.tif')
    pygeoprocessing.new_raster_from_base(
        flow_dir_mfd_raster_path_band[0],
        visited_raster_path,
        gdal.GDT_Byte, [0],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    visited_managed_raster = _ManagedRaster(visited_raster_path, 1, 1)

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

        if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info('Dist to channel MFD %.1f%% complete', 100.0 * current_pixel / <float>(
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

                # d_i is 0 if pixel i is a stream
                if channel_buffer_array[yi, xi] == 1:
                    distance_to_channel_managed_raster.set(
                        xi_root, yi_root, 0)
                    continue

                if flow_dir_buffer_array[yi, xi] == 0:
                    # nodata flow, so we skip
                    continue

                # the algorithm starts from an arbitrary pixel
                # it pushes its downstream neighbors to a stack, working depth-first
                # until reaching either a stream or a raster edge
                if visited_managed_raster.get(xi_root, yi_root) == 0:
                    visited_managed_raster.set(xi_root, yi_root, 1)
                    distance_to_channel_stack.push(
                        MFDFlowPixelType(xi_root, yi_root, 0, 0, 0))

                while not distance_to_channel_stack.empty():
                    pixel = distance_to_channel_stack.top()
                    distance_to_channel_stack.pop()
                    is_a_channel = (
                        channel_managed_raster.get(pixel.xi, pixel.yi) == 1)
                    if is_a_channel:
                        distance_to_channel_managed_raster.set(
                            pixel.xi, pixel.yi, 0)
                        continue

                    mfd_value = (
                        <int>flow_dir_mfd_managed_raster.get(
                            pixel.xi, pixel.yi))

                    # a pixel will be "preempted" if it gets pushed back onto
                    # the stack because not all its downstream neighbors are
                    # visited yet
                    preempted = 0

                    # iterate over neighbors
                    # look for ones that are downstream, that are within the raster bounds,
                    # and that haven't been visited yet
                    # when we find one that hasn't been visited yet, push `pixel` and
                    # the neighbor back onto the stack and exit the loop
                    # if all have been visited, we can calculate d_i
                    for i_n in range(pixel.last_flow_dir, 8):
                        flow_dir_weight = 0xF & (mfd_value >> (i_n * 4))
                        if flow_dir_weight == 0:
                            continue

                        xi_n = pixel.xi+D8_XOFFSET[i_n]
                        yi_n = pixel.yi+D8_YOFFSET[i_n]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            continue

                        # if the pixel has a neighbor that hasn't been visited yet,
                        # push it back onto the stack, and also push its neighbor
                        # onto the stack, and exit the loop.
                        # we won't calculate d_i until d_n is defined for every n
                        if visited_managed_raster.get(xi_n, yi_n) == 0:
                            visited_managed_raster.set(xi_n, yi_n, 1)
                            preempted = 1
                            pixel.last_flow_dir = i_n
                            distance_to_channel_stack.push(pixel)
                            distance_to_channel_stack.push(
                                MFDFlowPixelType(xi_n, yi_n, 0, 0, 0))
                            break

                        # at this point, the pixel's downstream neighbor n
                        # distance to channel (d_n) will already be calculated
                        n_distance = distance_to_channel_managed_raster.get(
                            xi_n, yi_n)

                        # if it's still nodata, that means this neighbor
                        # never reaches a stream. we don't count this towards
                        # the average.
                        if n_distance == distance_nodata:
                            # a channel was never found
                            continue

                        # if a weight is passed we use it directly and do
                        # not consider that a diagonal pixel is further
                        # away than an adjacent one. If no weight is used
                        # then "distance" is being calculated and we account
                        # for diagonal distance.
                        if weight_raster is not None:
                            weight_val = weight_raster.get(xi_n, yi_n)
                            if _is_close(weight_val, weight_nodata, 1e-8, 1e-5):
                                weight_val = 0.0
                        else:
                            # neighbors 0, 2, 4, 6 are lateral
                            # 1, 3, 5, 7 are diagonal
                            weight_val = (SQRT2 if i_n % 2 else 1)

                        # for only those neighbors which do drain to a stream,
                        # sum up the MFD values (the denominator in the average)
                        pixel.sum_of_weights += flow_dir_weight
                        # and sum up the d_n values weighted by MFD (the numerator)
                        pixel.sum_of_weighted_distances += flow_dir_weight * (
                            weight_val + n_distance)

                    # "preempted" means that this pixel got pushed back onto the stack
                    # because at least one of its downstream neighbors isn't done
                    # and so this pixel isn't ready to be calculated yet
                    if preempted:
                        continue

                    # if the sum of flow weights is 0, that means that no
                    # neighbors drain to a stream, and so d_i is undefined.
                    # in that case leave it unset (nodata).
                    if pixel.sum_of_weights != 0:
                        # divide to get the average
                        distance_to_channel_managed_raster.set(
                            pixel.xi, pixel.yi,
                            pixel.sum_of_weighted_distances / pixel.sum_of_weights)

    distance_to_channel_managed_raster.close()
    channel_managed_raster.close()
    flow_dir_mfd_managed_raster.close()
    if weight_raster is not None:
        weight_raster.close()
    visited_managed_raster.close()
    shutil.rmtree(tmp_work_dir)
    LOGGER.info('Dist to channel MFD %.1f%% complete', 100.0)


def extract_streams_mfd(
        flow_accum_raster_path_band, flow_dir_mfd_path_band,
        double flow_threshold, target_stream_raster_path,
        double trace_threshold_proportion=1.0,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Classify a stream raster from MFD flow accumulation.

    This function classifies pixels as streams that have a flow accumulation
    value >= ``flow_threshold`` and can trace further upstream with a fuzzy
    propotion if ``trace_threshold_proportion`` is set < 1.0

    Parameters:
        flow_accum_raster_path_band (tuple): a string/integer tuple indicating
            the flow accumulation raster to use as a basis for thresholding
            a stream. Values in this raster that are >= flow_threshold will
            be classified as streams. This raster should be derived from
            ``dem_raster_path_band`` using
            ``pygeoprocessing.routing.flow_accumulation_mfd``.
        flow_dir_mfd_path_band (str): path to multiple flow direction
            raster, required to join divergent streams.
        flow_threshold (float): the value in ``flow_accum_raster_path_band`` to
            indicate where a stream exists.
        target_stream_raster_path (str): path to the target stream raster.
            This raster will be the same dimensions and projection as
            ``dem_raster_path_band`` and will contain 1s where a stream is
            defined, 0 where the flow accumulation layer is defined but no
            stream exists, and nodata otherwise.
        trace_threshold_proportion (float): this value indicates what
            proportion of the flow_threshold is enough to classify a pixel
            as a stream after the stream has been traced from a
            ``flow_threshold`` drain. Setting this value < 1.0 is useful for
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
        gdal.GDT_Byte, [stream_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)

    cdef _ManagedRaster flow_accum_mr = _ManagedRaster(
        flow_accum_raster_path_band[0], flow_accum_raster_path_band[1], 0)
    cdef _ManagedRaster stream_mr = _ManagedRaster(
        target_stream_raster_path, 1, 1)
    cdef _ManagedRaster flow_dir_mfd_mr = _ManagedRaster(
        flow_dir_mfd_path_band[0], flow_dir_mfd_path_band[1], 0)

    cdef unsigned int xoff, yoff, win_xsize, win_ysize
    cdef unsigned int xi, yi, xi_root, yi_root, i_n, xi_n, yi_n, i_sn, xi_sn, yi_sn
    cdef int flow_dir_mfd
    cdef double flow_accum
    cdef double trace_flow_threshold = (
        trace_threshold_proportion * flow_threshold)
    cdef unsigned int n_iterations = 0
    cdef int is_outlet, stream_val
    cdef unsigned long current_pixel

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
            if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
                last_log_time = ctime(NULL)
                current_pixel = xoff + yoff * raster_x_size
                LOGGER.info('Extract streams MFD %.1f%% complete', 100.0 * current_pixel / <float>(
                    raster_x_size * raster_y_size))
            for xi in range(win_xsize):
                xi_root = xi+xoff
                flow_accum = flow_accum_mr.get(xi_root, yi_root)
                if _is_close(flow_accum, flow_accum_nodata, 1e-8, 1e-5):
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
                    xi_n = xi_root+D8_XOFFSET[i_n]
                    yi_n = yi_root+D8_YOFFSET[i_n]
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
                        xi_sn = xi_n+D8_XOFFSET[i_sn]
                        yi_sn = yi_n+D8_YOFFSET[i_sn]
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
                                                xi_sn = xi_bn+D8_XOFFSET[i_sn]
                                                yi_sn = yi_bn+D8_YOFFSET[i_sn]
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
    LOGGER.info('Extract streams MFD: filter out incomplete divergent streams')
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
    LOGGER.info('Extract streams MFD: 100.0% complete')


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


def extract_strahler_streams_d8(
        flow_dir_d8_raster_path_band, flow_accum_raster_path_band,
        dem_raster_path_band,
        target_stream_vector_path,
        long min_flow_accum_threshold=100,
        int river_order=5,
        float min_p_val=0.05,
        autotune_flow_accumulation=False,
        osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):
    """Extract Strahler order stream geometry from flow accumulation.

    Creates a Strahler ordered stream vector containing line segments
    representing each separate stream fragment. The final vector contains
    at least the fields:

        * "order" (int): an integer representing the stream order
        * "river_id" (int): unique ID used by all stream segments that
            connect to the same outlet.
        * "drop_distance" (float): this is the drop distance in DEM units
            from the upstream to downstream component of this stream
            segment.
        * "outlet" (int): 1 if this segment is an outlet, 0 if not.
        * "us_fa" (int): flow accumulation value at the upstream end of
            the stream segment.
        * "ds_fa" (int): flow accumulation value at the downstream end of
            the stream segment
        * "thresh_fa" (int): the final threshold flow accumulation value
            used to determine the river segments.
        * "upstream_d8_dir" (int): a bookkeeping parameter from stream
            calculations that is left in due to the overhead
            of deleting a field.
        * "ds_x" (int): the raster x coordinate for the outlet.
        * "ds_y" (int): the raster y coordinate for the outlet.
        * "ds_x_1" (int): the x raster space coordinate that is 1 pixel
            upstream from the outlet.
        * "ds_y_1" (int): the y raster space coordinate that is 1 pixel
            upstream from the outlet.
        * "us_x" (int): the raster x coordinate for the upstream inlet.
        * "us_y" (int): the raster y coordinate for the upstream inlet.

    Args:
        flow_dir_d8_raster_path_band (tuple): a path/band representing the D8
            flow direction raster.
        flow_accum_raster_path_band (tuple): a path/band representing the D8
            flow accumulation raster represented by
            ``flow_dir_d8_raster_path_band``.
        dem_raster_path_band (tuple): a path/band representing the DEM used to
            derive flow dir.
        target_stream_vector_path (tuple): a single layer line vector created
            by this function representing the stream segments extracted from
            the above arguments. Contains the fields "order" and "parent" as
            described above.
        min_flow_accum_threshold (int): minimum number of upstream pixels
            required to create a stream. If ``autotune_flow_accumulation``
            is True, then the final value may be adjusted based on
            significant differences in 1st and 2nd order streams.
        river_order (int): what stream order to define as a river in terms of
            automatically determining flow accumulation threshold for that
            stream collection.
        min_p_val (float): minimum p_value test for significance
        autotune_flow_accumulation (bool): If true, uses a t-test to test for
            significant distances in order 1 and order 2 streams. If it is
            significant the flow accumulation parameter is adjusted upwards
            until the drop distances are insignificant.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter
            should not be changed unless you know what you are doing.

    Returns:
        None.
    """
    flow_dir_info = pygeoprocessing.get_raster_info(
        flow_dir_d8_raster_path_band[0])
    if flow_dir_info['projection_wkt']:
        flow_dir_srs = osr.SpatialReference()
        flow_dir_srs.ImportFromWkt(flow_dir_info['projection_wkt'])
        flow_dir_srs.SetAxisMappingStrategy(osr_axis_mapping_strategy)
    else:
        flow_dir_srs = None
    gpkg_driver = gdal.GetDriverByName('GPKG')

    stream_vector = gpkg_driver.Create(
        target_stream_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    stream_basename = os.path.basename(
        os.path.splitext(target_stream_vector_path)[0])
    stream_layer = stream_vector.CreateLayer(
        stream_basename, flow_dir_srs, ogr.wkbLineString)
    stream_layer.CreateField(ogr.FieldDefn('order', ogr.OFTInteger))
    stream_layer.CreateField(ogr.FieldDefn('drop_distance', ogr.OFTReal))
    stream_layer.CreateField(ogr.FieldDefn('outlet', ogr.OFTInteger))
    stream_layer.CreateField(ogr.FieldDefn('river_id', ogr.OFTInteger))
    stream_layer.CreateField(ogr.FieldDefn('us_fa', ogr.OFTInteger64))
    stream_layer.CreateField(ogr.FieldDefn('ds_fa', ogr.OFTInteger64))
    stream_layer.CreateField(ogr.FieldDefn('thresh_fa', ogr.OFTInteger64))
    stream_layer.CreateField(ogr.FieldDefn('upstream_d8_dir', ogr.OFTInteger))
    stream_layer.CreateField(ogr.FieldDefn('ds_x', ogr.OFTInteger))
    stream_layer.CreateField(ogr.FieldDefn('ds_y', ogr.OFTInteger))
    stream_layer.CreateField(ogr.FieldDefn('ds_x_1', ogr.OFTInteger))
    stream_layer.CreateField(ogr.FieldDefn('ds_y_1', ogr.OFTInteger))
    stream_layer.CreateField(ogr.FieldDefn('us_x', ogr.OFTInteger))
    stream_layer.CreateField(ogr.FieldDefn('us_y', ogr.OFTInteger))
    flow_dir_managed_raster = _ManagedRaster(
        flow_dir_d8_raster_path_band[0], flow_dir_d8_raster_path_band[1], 0)

    flow_accum_managed_raster = _ManagedRaster(
        flow_accum_raster_path_band[0], flow_accum_raster_path_band[1], 0)

    dem_managed_raster = _ManagedRaster(
        dem_raster_path_band[0], dem_raster_path_band[1], 0)

    cdef int flow_nodata = pygeoprocessing.get_raster_info(
        flow_dir_d8_raster_path_band[0])['nodata'][
            flow_dir_d8_raster_path_band[1]-1]
    cdef double flow_accum_nodata = pygeoprocessing.get_raster_info(
        flow_accum_raster_path_band[0])['nodata'][
        flow_accum_raster_path_band[1]-1]

    # D8 flow directions encoded as
    # 321
    # 4x0
    # 567
    cdef unsigned int xoff, yoff, i, j, d, d_n, n_cols, n_rows
    cdef unsigned int win_xsize, win_ysize

    n_cols, n_rows = flow_dir_info['raster_size']

    LOGGER.info('(extract_strahler_streams_d8): seed the drains')
    cdef unsigned long n_pixels = n_cols * n_rows
    cdef long n_processed = 0
    cdef time_t last_log_time
    last_log_time = ctime(NULL)
    cdef stack[StreamConnectivityPoint] source_point_stack
    cdef StreamConnectivityPoint source_stream_point

    cdef unsigned int x_l=-1, y_l=-1  # the _l is for "local" aka "current" pixel

    # D8 backflow directions encoded as
    # 765
    # 0x4
    # 123
    cdef unsigned int x_n, y_n  # the _n is for "neighbor"
    cdef unsigned int upstream_count=0, upstream_index
    # this array is filled out as upstream directions are calculated and
    # indexed by `upstream_count`
    cdef int *upstream_dirs = [0, 0, 0, 0, 0, 0, 0, 0]
    cdef double local_flow_accum
    # used to determine if source is a drain and should be tracked
    cdef int is_drain

    # map x/y tuple to list of streams originating from that point
    # 2 tuple -> list of int
    coord_to_stream_ids = collections.defaultdict(list)

    # First pass - search for bifurcating stream points
    #   look at all pixels in the raster. If a pixel has:
    #       * flow accumulation > threshold and
    #       * more than two upstream neighbors with
    #         flow accumulation > threshold or
    #       * drains to edge or nodata pixel
    #   record a seed point for that bifurcation for later processing.
    stream_layer.StartTransaction()
    for offset_dict in pygeoprocessing.iterblocks(
            flow_dir_d8_raster_path_band, offset_only=True):
        if ctime(NULL)-last_log_time > _LOGGING_PERIOD:
            LOGGER.info(
                '(extract_strahler_streams_d8): drain seeding '
                f'{n_processed} of {n_pixels} pixels complete')
            last_log_time = ctime(NULL)
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        n_processed += win_xsize * win_ysize

        for i in range(win_xsize):
            for j in range(win_ysize):
                is_drain = 0
                x_l = xoff + i
                y_l = yoff + j
                local_flow_accum = <double>flow_accum_managed_raster.get(
                    x_l, y_l)
                if (local_flow_accum < min_flow_accum_threshold or
                        _is_close(local_flow_accum,
                                  flow_accum_nodata, 1e-8, 1e-5)):
                    continue
                # check to see if it's a drain
                d_n = <int>flow_dir_managed_raster.get(x_l, y_l)
                x_n = x_l + D8_XOFFSET[d_n]
                y_n = y_l + D8_YOFFSET[d_n]
                if (x_n < 0 or y_n < 0 or x_n >= n_cols or y_n >= n_rows or
                        <int>flow_dir_managed_raster.get(
                            x_n, y_n) == flow_nodata):
                    is_drain = 1

                if not is_drain and (
                        local_flow_accum < 2*min_flow_accum_threshold):
                    # if current pixel is < 2*flow threshold then it can't
                    # bifurcate into two pixels == flow threshold
                    continue

                upstream_count = 0
                for d in range(8):
                    x_n = x_l + D8_XOFFSET[d]
                    y_n = y_l + D8_YOFFSET[d]
                    # check if on border
                    if x_n < 0 or y_n < 0 or x_n >= n_cols or y_n >= n_rows:
                        continue
                    d_n = <int>flow_dir_managed_raster.get(x_n, y_n)
                    if d_n == flow_nodata:
                        continue
                    if (D8_REVERSE_DIRECTION[d] == d_n and
                            <double>flow_accum_managed_raster.get(
                                x_n, y_n) >= min_flow_accum_threshold):
                        upstream_dirs[upstream_count] = d
                        upstream_count += 1
                if upstream_count <= 1 and not is_drain:
                    continue
                for upstream_index in range(upstream_count):
                    # hit a branch!
                    stream_feature = ogr.Feature(
                        stream_layer.GetLayerDefn())
                    stream_feature.SetField('outlet', 0)
                    stream_layer.CreateFeature(stream_feature)
                    stream_fid = stream_feature.GetFID()
                    source_point_stack.push(StreamConnectivityPoint(
                        x_l, y_l, upstream_dirs[upstream_index], stream_fid))
                    coord_to_stream_ids[(x_l, y_l)].append(stream_fid)
    LOGGER.info(
        '(extract_strahler_streams_d8): '
        f'drain seeding complete')
    LOGGER.info('(extract_strahler_streams_d8): starting upstream walk')
    n_points = source_point_stack.size()

    # map downstream ids to list of upstream connected streams
    # id -> list of ids
    downstream_to_upstream_ids = {}

    # map upstream id to downstream connected stream id -> id
    upstream_to_downstream_id = {}

    while not source_point_stack.empty():
        if ctime(NULL)-last_log_time > _LOGGING_PERIOD:
            LOGGER.info(
                '(extract_strahler_streams_d8): '
                'stream segment creation '
                f'{n_points-source_point_stack.size()} of {n_points} '
                'source points complete')
            last_log_time = ctime(NULL)

        # This coordinate is the downstream end of the stream
        source_stream_point = source_point_stack.top()
        source_point_stack.pop()

        payload = _calculate_stream_geometry(
            source_stream_point.xi, source_stream_point.yi,
            source_stream_point.upstream_d8_dir,
            flow_dir_info['geotransform'], n_cols, n_rows,
            flow_accum_managed_raster, flow_dir_managed_raster, flow_nodata,
            min_flow_accum_threshold, coord_to_stream_ids)
        if payload is None:
            LOGGER.debug(
                f'no geometry for source point at '
                f'{source_stream_point.xi}, {source_stream_point.yi} '
                f'upstream direction: {source_stream_point.upstream_d8_dir}')
            continue
        x_u, y_u, ds_x_1, ds_y_1, upstream_id_list, stream_line = payload

        downstream_dem = dem_managed_raster.get(
            source_stream_point.xi, source_stream_point.yi)

        stream_feature = stream_layer.GetFeature(
            source_stream_point.source_id)
        stream_feature.SetField(
            'ds_fa', flow_accum_managed_raster.get(
                source_stream_point.xi, source_stream_point.yi))
        stream_feature.SetField('ds_x', source_stream_point.xi)
        stream_feature.SetField('ds_y', source_stream_point.yi)
        stream_feature.SetField('ds_x_1', ds_x_1)
        stream_feature.SetField('ds_y_1', ds_y_1)
        stream_feature.SetField('us_x', x_u)
        stream_feature.SetField('us_y', y_u)
        stream_feature.SetField(
            'upstream_d8_dir', source_stream_point.upstream_d8_dir)

        # record the downstream connected component for all the upstream
        # connected components
        for upstream_id in upstream_id_list:
            upstream_to_downstream_id[upstream_id] = (
                source_stream_point.source_id)

        # record the upstream connected components for the downstream component
        downstream_to_upstream_ids[source_stream_point.source_id] = (
            upstream_id_list)

        # if no upstream it means it is an order 1 source stream
        if not upstream_id_list:
            stream_feature.SetField('order', 1)
        stream_feature.SetGeometry(stream_line)

        # calculate the drop distance
        upstream_dem = dem_managed_raster.get(x_u, y_u)
        drop_distance = upstream_dem - downstream_dem
        stream_feature.SetField('drop_distance', drop_distance)
        stream_feature.SetField(
            'us_fa', flow_accum_managed_raster.get(x_u, y_u))
        stream_feature.SetField('thresh_fa', min_flow_accum_threshold)
        stream_layer.SetFeature(stream_feature)

    LOGGER.info(
        '(extract_strahler_streams_d8): stream segment creation complete')

    LOGGER.info('(extract_strahler_streams_d8): determining stream order')
    # seed the list with all order 1 streams
    stream_layer.SetAttributeFilter('"order"=1')
    streams_to_process = [stream_feature for stream_feature in stream_layer]
    base_feature_count = len(streams_to_process)
    outlet_fid_list = []
    while streams_to_process:
        if ctime(NULL)-last_log_time > 2.0:
            LOGGER.info(
                '(extract_strahler_streams_d8): '
                'stream order processing: '
                f'{base_feature_count-len(streams_to_process)} of '
                f'{base_feature_count} stream fragments complete')
            last_log_time = ctime(NULL)
        # fetch the downstream and connected upstream ids
        stream_feature = streams_to_process.pop(0)
        stream_fid = stream_feature.GetFID()
        if stream_fid not in upstream_to_downstream_id:
            # it's an outlet so no downstream to process
            stream_feature.SetField('outlet', 1)
            stream_layer.SetFeature(stream_feature)
            outlet_fid_list.append(stream_feature.GetFID())
            stream_feature = None
            continue
        downstream_fid = upstream_to_downstream_id[stream_fid]
        downstream_feature = stream_layer.GetFeature(downstream_fid)
        if downstream_feature.GetField('order') is not None:
            # downstream component already processed
            downstream_feature = None
            continue
        connected_upstream_fids = downstream_to_upstream_ids[downstream_fid]
        # check that all upstream IDs are defined and construct stream order
        # list
        stream_order_list = []
        all_defined = True
        for upstream_fid in connected_upstream_fids:
            upstream_feature = stream_layer.GetFeature(upstream_fid)
            upstream_order = upstream_feature.GetField('order')
            if upstream_order is not None:
                stream_order_list.append(upstream_order)
            else:
                # found an upstream not defined, that means it'll be processed
                # later
                all_defined = False
                break
        if not all_defined:
            # we'll revisit this stream later when the other connected
            # components are processed
            continue
        sorted_stream_order_list = sorted(stream_order_list)
        downstream_order = sorted_stream_order_list[-1]
        if len(sorted_stream_order_list) > 1 and (
                sorted_stream_order_list[-1] ==
                sorted_stream_order_list[-2]):
            # if there are at least two equal order streams feeding in,
            # we go up one order
            downstream_order += 1
        downstream_feature.SetField('order', downstream_order)
        stream_layer.SetFeature(downstream_feature)
        streams_to_process.append(downstream_feature)
        downstream_feature = None
    LOGGER.info(
        '(extract_strahler_streams_d8): stream order processing complete')

    LOGGER.info(
        '(extract_strahler_streams_d8): determine rivers')
    working_river_id = 0
    for outlet_index, outlet_fid in enumerate(outlet_fid_list):
        # walk upstream starting from this outlet to search for rivers
        # defined as stream segments whose order is <= river_order. Note it
        # can be < river_order because we may have some streams that have
        # outlets for shorter rivers that can't get to river_order.
        if ctime(NULL)-last_log_time > _LOGGING_PERIOD:
            LOGGER.info(
                '(extract_strahler_streams_d8): '
                'flow accumulation adjustment '
                f'{outlet_index+1} of {len(outlet_fid_list)} '
                'outlets complete')
            last_log_time = ctime(NULL)
        search_stack = [outlet_fid]
        while search_stack:
            stream_layer.CommitTransaction()
            stream_layer.StartTransaction()
            feature_id = search_stack.pop()
            stream_feature = stream_layer.GetFeature(feature_id)
            stream_order = stream_feature.GetField('order')

            if (stream_order > river_order or
                    stream_feature.GetField('river_id') is not None):
                # keep walking upstream until there's an order <= river_order
                search_stack.extend(
                    downstream_to_upstream_ids[feature_id])
            else:
                # walk up the stream setting every upstream segment's
                # river_id to working_river_id
                stream_layer.SetFeature(stream_feature)
                upstream_stack = [feature_id]

                streams_by_order = collections.defaultdict(list)
                drop_distance_collection = collections.defaultdict(list)
                max_upstream_flow_accum = collections.defaultdict(int)
                while upstream_stack:
                    feature_id = upstream_stack.pop()
                    stream_feature = stream_layer.GetFeature(feature_id)
                    stream_feature.SetField('river_id', working_river_id)
                    stream_layer.SetFeature(stream_feature)
                    order = stream_feature.GetField('order')
                    streams_by_order[order].append(stream_feature)
                    drop_distance_collection[order].append(
                        stream_feature.GetField('drop_distance'))
                    max_upstream_flow_accum[order] = max(
                        max_upstream_flow_accum[order],
                        stream_feature.GetField('us_fa'))
                    stream_feature = None
                    upstream_stack.extend(
                        downstream_to_upstream_ids[feature_id])

                working_flow_accum_threshold = min_flow_accum_threshold
                while drop_distance_collection and autotune_flow_accumulation:
                    stream_layer.CommitTransaction()
                    stream_layer.StartTransaction()
                    # decide how much bigger to make the flow_accum
                    # find a test_order that tests p_val > 0.5 then retest
                    test_order = min(drop_distance_collection)
                    while test_order+1 <= max(drop_distance_collection):
                        if (len(drop_distance_collection[test_order]) < 3 or
                                len(drop_distance_collection[
                                    test_order+1]) < 3):
                            # too small to test so it's not significant
                            break
                        _, p_val = scipy.stats.ttest_ind(
                            drop_distance_collection[test_order],
                            drop_distance_collection[test_order+1],
                            equal_var=True)
                        if p_val > min_p_val or numpy.isnan(p_val):
                            # not too big or just too few elements
                            break
                        test_order += 1
                    if test_order == min(drop_distance_collection):
                        # order 1/2 streams are not statistically different
                        break
                    # try to make a reasonable estimate for flow accum
                    working_flow_accum_threshold *= 1.25
                    # reconstruct stream segments of <= test_order
                    for order in range(1, test_order+1):
                        # This will build up a list of kept or reconstructed
                        # streams. Other streams will be deleted.
                        streams_to_retest = []
                        # The drop distance set will be recalculated
                        # dynamically for the next loop
                        if order in max_upstream_flow_accum:
                            del max_upstream_flow_accum[order]
                        if order in drop_distance_collection:
                            del drop_distance_collection[order]
                        while streams_by_order[order]:
                            stream_feature = streams_by_order[order].pop()
                            if (stream_feature.GetField('ds_fa') <
                                    working_flow_accum_threshold):
                                # this flow accumulation is too small, it's
                                # not relevant anymore
                                # remove from connectivity and delete
                                _delete_feature(
                                    stream_feature, stream_layer,
                                    upstream_to_downstream_id,
                                    downstream_to_upstream_ids)
                                continue
                            if (stream_feature.GetField('us_fa') >=
                                    working_flow_accum_threshold):
                                # this whole stream still fits in the
                                # threshold so keep it
                                # add drop distance to working set
                                drop_distance_collection[order].append(
                                    stream_feature.GetField('drop_distance'))
                                max_upstream_flow_accum[order] = max(
                                    max_upstream_flow_accum[order],
                                    stream_feature.GetField('us_fa'))
                                stream_layer.SetFeature(stream_feature)
                                streams_to_retest.append(stream_feature)
                                continue
                            # recalculate stream geometry
                            ds_x = stream_feature.GetField('ds_x')
                            ds_y = stream_feature.GetField('ds_y')
                            upstream_d8_dir = stream_feature.GetField(
                                'upstream_d8_dir')
                            payload = _calculate_stream_geometry(
                                ds_x, ds_y, upstream_d8_dir,
                                flow_dir_info['geotransform'], n_cols, n_rows,
                                flow_accum_managed_raster,
                                flow_dir_managed_raster, flow_nodata,
                                working_flow_accum_threshold,
                                coord_to_stream_ids)
                            if payload is None:
                                _delete_feature(
                                    stream_feature, stream_layer,
                                    upstream_to_downstream_id,
                                    downstream_to_upstream_ids)
                                continue
                            (x_u, y_u, ds_x_1, ds_y_1, upstream_id_list,
                                stream_line) = payload
                            # recalculate the drop distance set
                            stream_feature.SetGeometry(stream_line)
                            upstream_dem = dem_managed_raster.get(x_u, y_u)
                            downstream_dem = dem_managed_raster.get(
                                ds_x, ds_y)
                            drop_distance = upstream_dem - downstream_dem
                            drop_distance_collection[order].append(
                                drop_distance)
                            stream_feature.SetField(
                                'drop_distance', drop_distance)
                            stream_feature.SetField(
                                'us_fa', flow_accum_managed_raster.get(
                                    x_u, y_u))
                            stream_feature.SetField(
                                'thresh_fa', working_flow_accum_threshold)
                            stream_feature.SetField(
                                'ds_x', ds_x)
                            stream_feature.SetField(
                                'ds_y', ds_y)
                            stream_feature.SetField(
                                'ds_x_1', ds_x_1)
                            stream_feature.SetField(
                                'ds_y_1', ds_y_1)
                            stream_feature.SetField('us_x', x_u)
                            stream_feature.SetField('us_y', y_u)

                            streams_to_retest.append(stream_feature)
                            stream_layer.SetFeature(stream_feature)

                        streams_by_order[order] = streams_to_retest
                working_river_id += 1

    LOGGER.info(
        '(extract_strahler_streams_d8): '
        'flow accumulation adjustment complete')

    stream_layer.DeleteField(
        stream_layer.FindFieldIndex('upstream_d8_dir', 1))
    stream_layer.CommitTransaction()
    stream_layer.StartTransaction()
    LOGGER.info(
        '(extract_strahler_streams_d8): '
        'final pass on stream order and geometry')

    # seed the stack with all the upstream orders
    working_stack = [
        fid for fid in downstream_to_upstream_ids if
        not downstream_to_upstream_ids[fid]]
    fid_to_order = {}
    processed_segments = 0
    segments_to_process = len(downstream_to_upstream_ids)
    deleted_set = set()
    while working_stack:
        if ctime(NULL)-last_log_time > _LOGGING_PERIOD:
            LOGGER.info(
                '(extract_strahler_streams_d8): '
                'final pass on stream order '
                f'{processed_segments} of {segments_to_process} '
                'segments complete')
            last_log_time = ctime(NULL)
        processed_segments += 1

        working_fid = working_stack.pop()
        # invariant: working_fid and all upstream are processed, order not set

        upstream_fid_list = downstream_to_upstream_ids[working_fid]
        if upstream_fid_list:
            order_count = collections.defaultdict(int)
            for upstream_fid in upstream_fid_list:
                order_count[fid_to_order[upstream_fid]] += 1
            working_order = max(order_count)
            if order_count[working_order] > 1:
                working_order += 1
            fid_to_order[working_fid] = working_order
        else:
            fid_to_order[working_fid] = 1

        working_feature = stream_layer.GetFeature(working_fid)
        working_feature.SetField('order', fid_to_order[working_fid])
        stream_layer.SetFeature(working_feature)
        working_feature = None

        if working_fid not in upstream_to_downstream_id:
            # nothing downstream so it's done
            continue

        downstream_fid = upstream_to_downstream_id[working_fid]
        connected_fids = downstream_to_upstream_ids[downstream_fid]
        if len(connected_fids) == 1:
            # There's only one downstream, join it.
            # Downstream order is the same as upstream
            fid_to_order[downstream_fid] = fid_to_order[working_fid]
            del fid_to_order[working_fid]

            # set downstream order to working order
            downstream_to_upstream_ids[downstream_fid] = (
                downstream_to_upstream_ids[working_fid])
            # since we're deleting the upstream segment we need upstream
            # connecting segments to connect to the new downstream
            for upstream_fid in downstream_to_upstream_ids[downstream_fid]:
                upstream_to_downstream_id[upstream_fid] = downstream_fid
            del downstream_to_upstream_ids[working_fid]
            del upstream_to_downstream_id[working_fid]

            # join working line with downstream line
            working_feature = stream_layer.GetFeature(working_fid)
            downstream_feature = stream_layer.GetFeature(downstream_fid)
            downstream_geom = downstream_feature.GetGeometryRef()
            working_geom = working_feature.GetGeometryRef()

            # Union creates a multiline string by default but we know it's
            # connected only at one point, so the next step ensures it's a
            # regular linestring
            multi_line = working_geom.Union(downstream_geom)
            joined_line = ogr.CreateGeometryFromWkb(
                shapely.ops.linemerge(shapely.wkb.loads(
                    bytes(multi_line.ExportToWkb()))).wkb)

            downstream_feature.SetGeometry(joined_line)
            downstream_feature.SetField(
                'us_x', working_feature.GetField('us_x'))
            downstream_feature.SetField(
                'us_y', working_feature.GetField('us_y'))
            stream_layer.SetFeature(downstream_feature)
            working_feature = None
            downstream_feature = None
            multi_line = None
            joined_line = None

            # delete working line
            stream_layer.DeleteFeature(working_fid)
            deleted_set.add(working_fid)

            # push downstream line for processing
            working_stack.append(downstream_fid)
            continue

        # otherwise check if connected streams are all defined and if so
        # set a new downstream order
        upstream_all_defined = True
        for connected_fid in connected_fids:
            if connected_fid == working_fid:
                # skip current
                continue
            if connected_fid not in fid_to_order:
                # upstream not defined so skip it and it will be processed
                # on another iteration
                upstream_all_defined = False
                break

        if not upstream_all_defined:
            # wait for other upstream components to be defined
            continue

        # all upstream components of this fid are calculated so it can be
        # calculated now too
        working_stack.append(downstream_fid)

    LOGGER.info(
        '(extract_strahler_streams_d8): '
        'final pass on stream order complete')
    LOGGER.info(
        '(extract_strahler_streams_d8): '
        'commit transaction due to stream joining')
    stream_layer.CommitTransaction()
    stream_layer = None
    stream_vector = None
    LOGGER.info('(extract_strahler_streams_d8): all done')


def _build_discovery_finish_rasters(
        flow_dir_d8_raster_path_band, target_discovery_raster_path,
        target_finish_raster_path):
    """Generates a discovery and finish time raster for a given d8 flow path.

    Args:
        flow_dir_d8_raster_path_band (tuple): a D8 flow raster path band tuple
        target_discovery_raster_path (str): path to a generated raster that
            creates discovery time (i.e. what count the pixel is visited in)
        target_finish_raster_path (str): path to generated raster that creates
            maximum upstream finish time.

    Returns:
        None
    """
    flow_dir_info = pygeoprocessing.get_raster_info(
        flow_dir_d8_raster_path_band[0])
    cdef int n_cols, n_rows
    n_cols, n_rows = flow_dir_info['raster_size']
    cdef int flow_dir_nodata = (
        flow_dir_info['nodata'][flow_dir_d8_raster_path_band[1]-1])

    flow_dir_managed_raster = _ManagedRaster(
        flow_dir_d8_raster_path_band[0], flow_dir_d8_raster_path_band[1], 0)
    pygeoprocessing.new_raster_from_base(
        flow_dir_d8_raster_path_band[0], target_discovery_raster_path,
        gdal.GDT_Float64, [-1])
    discovery_managed_raster = _ManagedRaster(
        target_discovery_raster_path, 1, 1)
    pygeoprocessing.new_raster_from_base(
        flow_dir_d8_raster_path_band[0], target_finish_raster_path,
        gdal.GDT_Float64, [-1])
    finish_managed_raster = _ManagedRaster(target_finish_raster_path, 1, 1)

    cdef stack[CoordinateType] discovery_stack
    cdef stack[FinishType] finish_stack
    cdef CoordinateType raster_coord
    cdef FinishType finish_coordinate

    cdef long discovery_count = 0
    cdef int n_processed, n_pixels
    n_pixels = n_rows * n_cols
    n_processed = 0
    cdef time_t last_log_time = ctime(NULL)
    cdef int n_pushed

    cdef int i, j, xoff, yoff, win_xsize, win_ysize, x_l, y_l, x_n, y_n
    cdef int n_dir, test_dir

    for offset_dict in pygeoprocessing.iterblocks(
            flow_dir_d8_raster_path_band, offset_only=True):
        # search raster block by raster block
        if ctime(NULL)-last_log_time > _LOGGING_PERIOD:
            LOGGER.info(
                f'(discovery time processing): '
                f'{n_processed/n_pixels*100:.1f}% complete')
            last_log_time = ctime(NULL)
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        n_processed += win_xsize * win_ysize

        for i in range(win_xsize):
            for j in range(win_ysize):
                x_l = xoff + i
                y_l = yoff + j
                # check to see if this pixel is a drain
                d_n = <int>flow_dir_managed_raster.get(x_l, y_l)
                if d_n == flow_dir_nodata:
                    continue

                # check if downstream neighbor runs off raster or is nodata
                x_n = x_l + D8_XOFFSET[d_n]
                y_n = y_l + D8_YOFFSET[d_n]

                if (x_n < 0 or y_n < 0 or x_n >= n_cols or y_n >= n_rows or
                        <int>flow_dir_managed_raster.get(
                            x_n, y_n) == flow_dir_nodata):
                    discovery_stack.push(CoordinateType(x_l, y_l))
                    finish_stack.push(FinishType(x_l, y_l, 1))

                while not discovery_stack.empty():
                    # This coordinate is the downstream end of the stream
                    raster_coord = discovery_stack.top()
                    discovery_stack.pop()

                    discovery_managed_raster.set(
                        raster_coord.xi, raster_coord.yi, discovery_count)
                    discovery_count += 1

                    n_pushed = 0
                    # check each neighbor to see if it drains to this cell
                    # if so, it's on the traversal path
                    for test_dir in range(8):
                        x_n = raster_coord.xi + D8_XOFFSET[test_dir % 8]
                        y_n = raster_coord.yi + D8_YOFFSET[test_dir % 8]
                        if x_n < 0 or y_n < 0 or \
                                x_n >= n_cols or y_n >= n_rows:
                            continue
                        n_dir = <int>flow_dir_managed_raster.get(x_n, y_n)
                        if n_dir == flow_dir_nodata:
                            continue
                        if D8_REVERSE_DIRECTION[test_dir] == n_dir:
                            discovery_stack.push(CoordinateType(x_n, y_n))
                            n_pushed += 1
                    # this reference is for the previous top and represents
                    # how many elements must be processed before finish
                    # time can be defined
                    finish_stack.push(
                        FinishType(
                            raster_coord.xi, raster_coord.yi, n_pushed))

                    # pop the finish stack until n_pushed > 1
                    if n_pushed == 0:
                        while (not finish_stack.empty() and
                               finish_stack.top().n_pushed <= 1):
                            finish_coordinate = finish_stack.top()
                            finish_stack.pop()
                            finish_managed_raster.set(
                                finish_coordinate.xi, finish_coordinate.yi,
                                discovery_count-1)
                        if not finish_stack.empty():
                            # then take one more because one branch is done
                            finish_coordinate = finish_stack.top()
                            finish_stack.pop()
                            finish_coordinate.n_pushed -= 1
                            finish_stack.push(finish_coordinate)


def calculate_subwatershed_boundary(
        d8_flow_dir_raster_path_band,
        strahler_stream_vector_path, target_watershed_boundary_vector_path,
        max_steps_per_watershed=1000000,
        outlet_at_confluence=False):
    """Calculate a linestring boundary around all subwatersheds.

    Subwatersheds start where the ``strahler_stream_vector`` has a junction
    starting at this highest upstream to lowest and ending at the outlet of
    a river.

    Args:
        d8_flow_dir_raster_path_band (tuple): raster/path band for d8 flow dir
            raster
        strahler_stream_vector_path (str): path to stream segment vector
        target_watershed_boundary_vector_path (str): path to created vector
            of linestring for watershed boundaries. Contains the fields:

            * "stream_id": this is the stream ID from the
              ``strahler_stream_vector_path`` that corresponds to this
              subwatershed.
            * "terminated_early": if set to 1 this watershed generation was
              terminated before it could be complete. This value should
              always be 0 unless something is wrong as a software bug
              or some degenerate case of data.
            * "outlet_x", "outlet_y": this is the x/y coordinate in raster
              space of the outlet of the watershed. It can be useful when
              determining other properties about the watershed when indexed
              with underlying raster data that created the streams in
              ``strahler_stream_vector_path``.

        max_steps_per_watershed (int): maximum number of steps to take when
            defining a watershed boundary. Useful if the DEM is large and
            degenerate or some other user known condition to limit long large
            polygons. Defaults to 1000000.
        outlet_at_confluence (bool): If True the outlet of subwatersheds
            starts at the confluence of streams. If False (the default)
            subwatersheds will start one pixel up from the confluence.

    Returns:
        None.
    """
    workspace_dir = tempfile.mkdtemp(
        prefix='calculate_subwatershed_boundary_workspace_',
        dir=os.path.join(
            os.path.dirname(target_watershed_boundary_vector_path)))
    discovery_time_raster_path = os.path.join(workspace_dir, 'discovery.tif')
    finish_time_raster_path = os.path.join(workspace_dir, 'finish.tif')

    # construct the discovery/finish time rasters for fast individual cell
    # watershed detection
    _build_discovery_finish_rasters(
        d8_flow_dir_raster_path_band, discovery_time_raster_path,
        finish_time_raster_path)

    shutil.copyfile(
        discovery_time_raster_path, f'{discovery_time_raster_path}_bak.tif')

    # the discovery raster is filled with nodata around the edges of
    # discovered watersheds, so it is opened for writing
    discovery_managed_raster = _ManagedRaster(
        discovery_time_raster_path, 1, 1)
    finish_managed_raster = _ManagedRaster(finish_time_raster_path, 1, 0)
    d8_flow_dir_managed_raster = _ManagedRaster(
        d8_flow_dir_raster_path_band[0], d8_flow_dir_raster_path_band[1], 0)

    discovery_info = pygeoprocessing.get_raster_info(
        discovery_time_raster_path)
    cdef long discovery_nodata = discovery_info['nodata'][0]

    cdef unsigned int n_cols, n_rows
    n_cols, n_rows = discovery_info['raster_size']

    geotransform = discovery_info['geotransform']
    cdef double g0, g1, g2, g3, g4, g5
    g0, g1, g2, g3, g4, g5 = geotransform

    if discovery_info['projection_wkt']:
        discovery_srs = osr.SpatialReference()
        discovery_srs.ImportFromWkt(discovery_info['projection_wkt'])
    else:
        discovery_srs = None
    gpkg_driver = gdal.GetDriverByName('GPKG')

    if os.path.exists(target_watershed_boundary_vector_path):
        LOGGER.warning(
            f'{target_watershed_boundary_vector_path} exists, removing '
            'before creating a new one.')
        os.remove(target_watershed_boundary_vector_path)
    watershed_vector = gpkg_driver.Create(
        target_watershed_boundary_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    watershed_basename = os.path.basename(os.path.splitext(
        target_watershed_boundary_vector_path)[0])
    watershed_layer = watershed_vector.CreateLayer(
        watershed_basename, discovery_srs, ogr.wkbPolygon)
    watershed_layer.CreateField(ogr.FieldDefn('stream_fid', ogr.OFTInteger))
    watershed_layer.CreateField(
        ogr.FieldDefn('terminated_early', ogr.OFTInteger))
    watershed_layer.CreateField(ogr.FieldDefn('outlet_x', ogr.OFTInteger))
    watershed_layer.CreateField(ogr.FieldDefn('outlet_y', ogr.OFTInteger))
    watershed_layer.StartTransaction()

    cdef unsigned int x_l, y_l
    cdef int outflow_dir
    cdef double x_f, y_f
    cdef double x_p, y_p
    cdef long discovery, finish

    cdef time_t last_log_time = ctime(NULL)

    stream_vector = gdal.OpenEx(strahler_stream_vector_path, gdal.OF_VECTOR)
    stream_layer = stream_vector.GetLayer()

    # construct linkage data structure for upstream streams
    upstream_fid_map = collections.defaultdict(list)
    for stream_feature in stream_layer:
        ds_x = int(stream_feature.GetField('ds_x'))
        ds_y = int(stream_feature.GetField('ds_y'))
        upstream_fid_map[(ds_x, ds_y)].append(
            stream_feature.GetFID())

    stream_layer.ResetReading()
    # construct visit order, this list will have a tuple of (fid, 0/1)
    # this stack will be used to build watersheds from upstream to downstream
    visit_order_stack = []
    # visit the highest order to lowest order in case there's a branching
    # junction of a order 1 and order 5 stream... visit order 5 upstream
    # first
    stream_layer.SetAttributeFilter(f'"outlet"=1')
    # these are done last
    for _, outlet_fid in sorted([
            (x.GetField('order'), x.GetFID()) for x in stream_layer],
            reverse=True):
        working_stack = [outlet_fid]
        processed_nodes = set()
        while working_stack:
            working_fid = working_stack[-1]
            processed_nodes.add(working_fid)
            working_feature = stream_layer.GetFeature(working_fid)

            us_x = int(working_feature.GetField('us_x'))
            us_y = int(working_feature.GetField('us_y'))
            ds_x_1 = int(working_feature.GetField('ds_x_1'))
            ds_y_1 = int(working_feature.GetField('ds_y_1'))

            upstream_coord = (us_x, us_y)
            upstream_fids = [
                fid for fid in upstream_fid_map[upstream_coord]
                if fid not in processed_nodes]
            if upstream_fids:
                working_stack.extend(upstream_fids)
            else:
                working_stack.pop()
                # the `not outlet_at_confluence` bit allows us to seed
                # even if the order is 1, otherwise confluences fill
                # the order 1 streams
                if (working_feature.GetField('order') > 1 or
                        not outlet_at_confluence):
                    if outlet_at_confluence:
                        # seed the upstream point
                        visit_order_stack.append((working_fid, us_x, us_y))
                    else:
                        # seed the downstream but +1 step point
                        visit_order_stack.append(
                            (working_fid, ds_x_1, ds_y_1))
                if working_feature.GetField('outlet') == 1:
                    # an outlet is a special case where the outlet itself
                    # should be a subwatershed done last.
                    ds_x = int(working_feature.GetField('ds_x'))
                    ds_y = int(working_feature.GetField('ds_y'))
                    if not outlet_at_confluence:
                        # undo the previous visit because it will be at
                        # one pixel up and we want the pixel right at
                        # the outlet
                        visit_order_stack.pop()
                    visit_order_stack.append((working_fid, ds_x, ds_y))

    cdef int edge_side, edge_dir, cell_to_test, out_dir_increase=-1
    cdef int left, right, n_steps, terminated_early
    cdef int delta_x, delta_y
    cdef int _int_max_steps_per_watershed = max_steps_per_watershed

    for index, (stream_fid, x_l, y_l) in enumerate(visit_order_stack):
        if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
            LOGGER.info(
                f'(calculate_subwatershed_boundary): watershed building '
                f'{(index/len(visit_order_stack))*100:.1f}% complete')
            last_log_time = ctime(NULL)
        discovery = <long>discovery_managed_raster.get(x_l, y_l)
        if discovery == -1:
            continue
        boundary_list = [(x_l, y_l)]
        finish = <long>finish_managed_raster.get(x_l, y_l)
        outlet_x = x_l
        outlet_y = y_l

        watershed_boundary = ogr.Geometry(ogr.wkbLinearRing)
        outflow_dir = <int>d8_flow_dir_managed_raster.get(x_l, y_l)

        # this is the center point of the pixel that will be offset to
        # make the edge
        x_f = x_l+0.5
        y_f = y_l+0.5

        x_f += D8_XOFFSET[outflow_dir]*0.5
        y_f += D8_YOFFSET[outflow_dir]*0.5
        if outflow_dir % 2 == 0:
            # need to back up the point a bit
            x_f -= D8_YOFFSET[outflow_dir]*0.5
            y_f += D8_XOFFSET[outflow_dir]*0.5

        x_p, y_p = gdal.ApplyGeoTransform(geotransform, x_f, y_f)
        watershed_boundary.AddPoint(x_p, y_p)

        # keep track of how many steps x/y and when we get back to 0 we've
        # made a loop
        delta_x, delta_y = 0, 0

        # determine the first edge
        if outflow_dir % 2 == 0:
            # outflow through a straight side, so trivial edge detection
            edge_side = outflow_dir
            edge_dir = (2+edge_side) % 8
        else:
            # diagonal outflow requires testing neighboring cells to
            # determine first edge
            cell_to_test = (outflow_dir+1) % 8
            edge_side = cell_to_test
            edge_dir = (cell_to_test+2) % 8
            if _in_watershed(
                    x_l, y_l, cell_to_test, discovery, finish,
                    n_cols, n_rows,
                    discovery_managed_raster, discovery_nodata):
                edge_side = (edge_side-2) % 8
                edge_dir = (edge_dir-2) % 8
                x_l += D8_XOFFSET[edge_dir]
                y_l += D8_YOFFSET[edge_dir]
                # note the pixel moved
                boundary_list.append((x_l, y_l))

        n_steps = 0
        terminated_early = 0
        while True:
            # step the edge then determine the projected coordinates
            x_f += D8_XOFFSET[edge_dir]
            y_f += D8_YOFFSET[edge_dir]
            delta_x += D8_XOFFSET[edge_dir]
            delta_y += D8_YOFFSET[edge_dir]
            # equivalent to gdal.ApplyGeoTransform(geotransform, x_f, y_f)
            # to eliminate python function call overhead
            x_p = g0 + g1*x_f + g2*y_f
            y_p = g3 + g4*x_f + g5*y_f
            watershed_boundary.AddPoint(x_p, y_p)
            n_steps += 1
            if n_steps > _int_max_steps_per_watershed:
                LOGGER.warning('quitting, too many steps')
                terminated_early = 1
                break
            if x_l < 0 or y_l < 0 or x_l >= n_cols or y_l >= n_rows:
                # This is unexpected but worth checking since missing this
                # error would be very difficult to debug.
                raise RuntimeError(
                    f'{x_l}, {y_l} out of bounds for '
                    f'{n_cols}x{n_rows} raster.')
            if edge_side - ((edge_dir-2) % 8) == 0:
                # counterclockwise configuration
                left = edge_dir
                right = (left-1) % 8
                out_dir_increase = 2
            else:
                # clockwise configuration (swapping "left" and "right")
                right = edge_dir
                left = (edge_side+1)
                out_dir_increase = -2
            left_in = _in_watershed(
                x_l, y_l, left, discovery, finish, n_cols, n_rows,
                discovery_managed_raster, discovery_nodata)
            right_in = _in_watershed(
                x_l, y_l, right, discovery, finish, n_cols, n_rows,
                discovery_managed_raster, discovery_nodata)
            if right_in:
                # turn right
                out_dir = edge_side
                edge_side = (edge_side-out_dir_increase) % 8
                edge_dir = out_dir
                # pixel moves to be the right cell
                x_l += D8_XOFFSET[right]
                y_l += D8_YOFFSET[right]
                _diagonal_fill_step(
                    x_l, y_l, right,
                    discovery, finish, discovery_managed_raster,
                    discovery_nodata,
                    boundary_list)
            elif left_in:
                # step forward
                x_l += D8_XOFFSET[edge_dir]
                y_l += D8_YOFFSET[edge_dir]
                # the pixel moves forward
                boundary_list.append((x_l, y_l))
            else:
                # turn left
                edge_side = edge_dir
                edge_dir = (edge_side + out_dir_increase) % 8

            if delta_x == 0 and delta_y == 0:
                # met the start point so we completed the watershed loop
                break

        watershed_feature = ogr.Feature(watershed_layer.GetLayerDefn())
        watershed_polygon = ogr.Geometry(ogr.wkbPolygon)
        watershed_polygon.AddGeometry(watershed_boundary)
        watershed_feature.SetGeometry(watershed_polygon)
        watershed_feature.SetField('stream_fid', stream_fid)
        watershed_feature.SetField('terminated_early', terminated_early)
        watershed_feature.SetField('outlet_x', outlet_x)
        watershed_feature.SetField('outlet_y', outlet_y)
        watershed_layer.CreateFeature(watershed_feature)

        # this loop fills in the raster at the boundary, done at end so it
        # doesn't interfere with the loop return to think the cells are no
        # longer in the watershed
        for boundary_x, boundary_y in boundary_list:
            discovery_managed_raster.set(boundary_x, boundary_y, -1)
    watershed_layer.CommitTransaction()
    watershed_layer = None
    watershed_vector = None
    discovery_managed_raster.close()
    finish_managed_raster.close()
    shutil.rmtree(workspace_dir)
    LOGGER.info(
        '(calculate_subwatershed_boundary): watershed building 100% complete')


def detect_lowest_drain_and_sink(dem_raster_path_band):
    """Find the lowest drain and sink pixel in the DEM.

    This function is used to specify conditions to DEMs that are known to
    have one real sink/drain, but may have several numerical sink/drains by
    detecting both the lowest pixel that could drain the raster on an edge
    and the lowest internal pixel that might sink the whole raster.

    Example:
        raster A contains the following
            * pixel at (3, 4) at 10m draining to a nodata  pixel
            * pixel at (15, 19) at 11m draining to a nodata pixel
            * pixel at (19, 21) at 10m draining to a nodata pixel
            * pit pixel at (10, 15) at 5m surrounded by non-draining pixels
            * pit pixel at (25, 15) at 15m surrounded by non-draining pixels
            * pit pixel at (2, 125) at 5m surrounded by non-draining pixels

        The result is two pixels indicating the first lowest edge and first
        lowest sink seen:
            drain_pixel = (3, 4), 10
            sink_pixel = (10, 15), 5

    Args:
        dem_raster_path_band (tuple): a raster/path band tuple to detect
            sinks in.

    Return:
        (drain_pixel, drain_height, sink_pixel, sink_height) -
            two (x, y) tuples with corresponding heights, first
            list is for edge drains, the second is for pit sinks. The x/y
            coordinate is in raster coordinate space and _height is the
            height of the given pixels in edge and pit respectively.
    """
    # this outer loop drives the raster block search
    cdef double lowest_drain_height = numpy.inf
    cdef double lowest_sink_height = numpy.inf

    drain_pixel = None
    sink_pixel = None

    dem_raster_info = pygeoprocessing.get_raster_info(
        dem_raster_path_band[0])
    cdef double dem_nodata
    # guard against undefined nodata by picking a value that's unlikely to
    # be a dem value
    if dem_raster_info['nodata'][0] is not None:
        dem_nodata = dem_raster_info['nodata'][0]
    else:
        dem_nodata = IMPROBABLE_FLOAT_NODATA

    raster_x_size, raster_y_size = dem_raster_info['raster_size']

    cdef _ManagedRaster dem_managed_raster = _ManagedRaster(
        dem_raster_path_band[0], dem_raster_path_band[1], 0)

    cdef time_t last_log_time = ctime(NULL)

    for offset_dict in pygeoprocessing.iterblocks(
            dem_raster_path_band, offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > _LOGGING_PERIOD:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            LOGGER.info(
                '(infer_sinks): '
                f'{current_pixel} of {raster_x_size * raster_y_size} '
                'pixels complete')

        # search block for local sinks
        for yi in range(0, win_ysize):
            yi_root = yi+yoff
            for xi in range(0, win_xsize):
                xi_root = xi+xoff
                center_val = dem_managed_raster.get(xi_root, yi_root)
                if _is_close(center_val, dem_nodata, 1e-8, 1e-5):
                    continue

                if (center_val > lowest_drain_height and
                        center_val > lowest_sink_height):
                    # already found something lower
                    continue

                # search neighbors for downhill or nodata
                pixel_drains = 0
                for i_n in range(8):
                    xi_n = xi_root+D8_XOFFSET[i_n]
                    yi_n = yi_root+D8_YOFFSET[i_n]

                    if (xi_n < 0 or xi_n >= raster_x_size or
                            yi_n < 0 or yi_n >= raster_y_size):
                        # it'll drain off the edge of the raster
                        if center_val < lowest_drain_height:
                            # found a new lower edge height
                            lowest_drain_height = center_val
                            drain_pixel = (xi_root, yi_root)
                        pixel_drains = 1
                        break
                    n_height = dem_managed_raster.get(xi_n, yi_n)
                    if _is_close(n_height, dem_nodata, 1e-8, 1e-5):
                        # it'll drain to nodata
                        if center_val < lowest_drain_height:
                            # found a new lower edge height
                            lowest_drain_height = center_val
                            drain_pixel = (xi_root, yi_root)
                        pixel_drains = 1
                        break
                    if n_height < center_val:
                        # it'll drain downhill
                        pixel_drains = 1
                        break
                if not pixel_drains and center_val < lowest_sink_height:
                    lowest_sink_height = center_val
                    sink_pixel = (xi_root, yi_root)
    return (
        drain_pixel, lowest_drain_height,
        sink_pixel, lowest_sink_height)


def detect_outlets(
        flow_dir_raster_path_band, flow_dir_type, target_outlet_vector_path):
    """Create point vector indicating flow raster outlets.

    If either D8 or MFD rasters have a flow direction to the edge of the
    raster or to a nodata flow direction pixel the originating pixel is
    considered an outlet.

    Args:
        flow_dir_raster_path_band (tuple): raster path/band tuple
            indicating D8 or MFD flow direction created by
            `routing.flow_dir_d8` or `routing.flow_dir_mfd`.
        flow_dir_type (str): one of 'd8' or 'mfd' to indicate the
            ``flow_dir_raster_path_band`` is either a D8 or MFD flow
            direction raster.
        target_outlet_vector_path (str): path to a vector that is created
            by this call that will be in the same projection units as the
            raster and have a point feature in the center of each pixel that
            is a raster outlet. Additional fields include:

                * "i" - the column raster coordinate where the outlet exists
                * "j" - the row raster coordinate where the outlet exists
                * "ID" - unique identification for the outlet.

    Return:
        None.
    """
    flow_dir_type = flow_dir_type.lower()
    if flow_dir_type not in ['d8', 'mfd']:
        raise ValueError(
            f'expected flow dir type of either d8 or mfd but got '
            f'{flow_dir_type}')
    cdef int d8_flow_dir_mode = (flow_dir_type == 'd8')

    cdef unsigned int xoff, yoff, win_xsize, win_ysize, xi, yi
    cdef unsigned int xi_root, yi_root, raster_x_size, raster_y_size
    cdef int flow_dir, flow_dir_n
    cdef int next_id=0, n_dir, is_outlet
    cdef char x_off_border, y_off_border, win_xsize_border, win_ysize_border

    cdef numpy.ndarray[numpy.npy_int32, ndim=2] flow_dir_block

    raster_info = pygeoprocessing.get_raster_info(
        flow_dir_raster_path_band[0])

    cdef int flow_dir_nodata = raster_info['nodata'][
        flow_dir_raster_path_band[1]-1]

    raster_x_size, raster_y_size = raster_info['raster_size']

    flow_dir_raster = gdal.OpenEx(
        flow_dir_raster_path_band[0], gdal.OF_RASTER)
    flow_dir_band = flow_dir_raster.GetRasterBand(
        flow_dir_raster_path_band[1])

    if raster_info['projection_wkt']:
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(raster_info['projection_wkt'])
    else:
        raster_srs = None

    gpkg_driver = gdal.GetDriverByName('GPKG')

    if os.path.exists(target_outlet_vector_path):
        LOGGER.warning(
            f'outlet detection: {target_outlet_vector_path} exists, '
            'removing before creating a new one.')
        os.remove(target_outlet_vector_path)
    outlet_vector = gpkg_driver.Create(
        target_outlet_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    outet_basename = os.path.basename(
        os.path.splitext(target_outlet_vector_path)[0])
    outlet_layer = outlet_vector.CreateLayer(
        outet_basename, raster_srs, ogr.wkbPoint)
    # i and j indicate the coordinates of the point in raster space whereas
    # the geometry is in projected space
    outlet_layer.CreateField(ogr.FieldDefn('i', ogr.OFTInteger))
    outlet_layer.CreateField(ogr.FieldDefn('j', ogr.OFTInteger))
    outlet_layer.CreateField(ogr.FieldDefn('ID', ogr.OFTInteger))
    outlet_layer.StartTransaction()

    cdef time_t last_log_time = ctime(NULL)
    # iterate by iterblocks so ReadAsArray can efficiently cache reads
    # and writes
    LOGGER.info('outlet detection: 0% complete')
    for block_offsets in pygeoprocessing.iterblocks(
            flow_dir_raster_path_band, offset_only=True):
        xoff = block_offsets['xoff']
        yoff = block_offsets['yoff']
        win_xsize = block_offsets['win_xsize']
        win_ysize = block_offsets['win_ysize']

        # Make an array with a 1 pixel border around the iterblocks window
        # That border will be filled in with nodata or data from the raster
        # if the window does not align with a top/bottom/left/right edge
        flow_dir_block = numpy.empty(
            (win_ysize+2, win_xsize+2), dtype=numpy.int32)

        # Test for left border and if so stripe nodata on the left margin
        x_off_border = 0
        if xoff > 0:
            x_off_border = 1
        else:
            flow_dir_block[:, 0] = flow_dir_nodata

        # Test for top border and if so stripe nodata on the top margin
        y_off_border = 0
        if yoff > 0:
            y_off_border = 1
        else:
            flow_dir_block[0, :] = flow_dir_nodata

        # Test for right border and if so stripe nodata on the right margin
        win_xsize_border = 0
        if xoff+win_xsize < raster_x_size-1:
            win_xsize_border += 1
        else:
            flow_dir_block[:, -1] = flow_dir_nodata

        # Test for bottom border and if so stripe nodata on the bottom margin
        win_ysize_border = 0
        if yoff+win_ysize < raster_y_size-1:
            win_ysize_border += 1
        else:
            flow_dir_block[-1, :] = flow_dir_nodata

        # Read iterblock plus a possible margin on top/bottom/left/right side
        # and read as type int32 to handle both d8 or mfd formats
        flow_dir_block[
            1-y_off_border:win_ysize+1+win_ysize_border,
            1-x_off_border:win_xsize+1+win_xsize_border] = \
            flow_dir_band.ReadAsArray(
                xoff=xoff-x_off_border,
                yoff=yoff-y_off_border,
                win_xsize=win_xsize+win_xsize_border+x_off_border,
                win_ysize=win_ysize+win_ysize_border+y_off_border).astype(
                    numpy.int32)

        for yi in range(1, win_ysize+1):
            if ctime(NULL) - last_log_time > 5.0:
                last_log_time = ctime(NULL)
                current_pixel = xoff + yoff * raster_x_size
                LOGGER.info(
                    f'''outlet detection: {
                        100.0 * current_pixel / <float>(
                            raster_x_size * raster_y_size):.1f} complete''')
            for xi in range(1, win_xsize+1):
                flow_dir = flow_dir_block[yi, xi]
                if flow_dir == flow_dir_nodata:
                    continue

                is_outlet = 1
                if d8_flow_dir_mode:
                    # inspect the outflow pixel neighbor
                    flow_dir_n = flow_dir_block[
                        yi+D8_YOFFSET[flow_dir],
                        xi+D8_XOFFSET[flow_dir]]

                    # if the outflow pixel is outside the raster boundaries or
                    # is a nodata pixel it must mean xi,yi is an outlet
                    if flow_dir_n != flow_dir_nodata:
                        is_outlet = 0
                else:
                    # inspect all the outflow pixel neighbors in MFD mode
                    for n_dir in range(8):
                        # shift the 0xF mask to the outflow direction and
                        # test if there's any outflow or not. 0 means nothing
                        # flows out of that pixel in the ``n_dir`` direction.
                        # 0xF is a binary number equaling 1111. The
                        # <<(n_dir*4) will shift these 1111's over 4*`n_dir`
                        # spaces thus aligning them with the section of the
                        # MFD integer that represents the proportional flow
                        # in that direction. The final bitwise & masks the
                        # entire MFD direction to the bit shifted 1111 mask,
                        # if it equals 0 it means there was no proportional
                        # flow in the `n_dir` direction.
                        if flow_dir&(0xF<<(n_dir*4)) == 0:
                            continue
                        flow_dir_n = flow_dir_block[
                            yi+D8_YOFFSET[n_dir],
                            xi+D8_XOFFSET[n_dir]]
                        if flow_dir_n != flow_dir_nodata:
                            is_outlet = 0
                            break

                # if the outflow pixel is outside the raster boundaries or
                # is a nodata pixel it must mean xi,yi is an outlet
                if is_outlet:
                    outlet_point = ogr.Geometry(ogr.wkbPoint)
                    # calculate global x/y raster coordinate, the -1 is for
                    # the left/top border of the test array window
                    xi_root = xi+xoff-1
                    yi_root = yi+yoff-1
                    # created a projected point in the center of the pixel
                    # thus the + 0.5 to x and y
                    proj_x, proj_y = gdal.ApplyGeoTransform(
                        raster_info['geotransform'],
                        xi_root+0.5, yi_root+0.5)
                    outlet_point.AddPoint(proj_x, proj_y)
                    outlet_feature = ogr.Feature(outlet_layer.GetLayerDefn())
                    outlet_feature.SetGeometry(outlet_point)
                    # save the raster coordinates of the outlet pixel as i,j
                    outlet_feature.SetField('i', xi_root)
                    outlet_feature.SetField('j', yi_root)
                    outlet_feature.SetField('ID', next_id)
                    next_id += 1
                    outlet_layer.CreateFeature(outlet_feature)
                    outlet_feature = None
                    outlet_point = None

    flow_dir_raster = None
    flow_dir_band = None
    LOGGER.info('outlet detection: 100% complete -- committing transaction')
    outlet_layer.CommitTransaction()
    outlet_layer = None
    outlet_vector = None
    LOGGER.info('outlet detection: done')


cdef void _diagonal_fill_step(
        int x_l, int y_l, int edge_dir,
        long discovery, long finish,
        _ManagedRaster discovery_managed_raster,
        long discovery_nodata, boundary_list):
    """Fill diagonal that are in the watershed behind the new edge.

    Used as a helper function to mark pixels as part of the watershed
    boundary in one step if they are diagonal and also contained within the
    watershed. Prevents a case like this:

    iii
    ii1
    i1o

    Instead would fill the diagonal like this:

    iii
    i11
    i1o

    Args:
        x_l/y_l (int): leading coordinate of the watershed boundary
            edge.
        edge_dir (int): D8 direction that points which direction the edge
            came from
        discovery/finish (long): the discovery and finish time that defines
            whether a pixel discovery time is inside a watershed or not.
        discovery_managed_raster (_ManagedRaster): discovery time raster
            x/y gives the discovery time for that pixel.
        discovery_nodata (long): nodata value for discovery raster
        boundary_list (list): this list is appended to for new pixels that
            should be neighbors in the fill.

    Return:
        None.
    """
    # always add the current pixel
    boundary_list.append((x_l, y_l))

    # this section determines which back diagonal was in the watershed and
    # fills it. if none are we pick one so there's no degenerate case
    cdef int xdelta = D8_XOFFSET[edge_dir]
    cdef int ydelta = D8_YOFFSET[edge_dir]
    test_list = [
        (x_l - xdelta, y_l),
        (x_l, y_l - ydelta)]
    for x_t, y_t in test_list:
        point_discovery = <long>discovery_managed_raster.get(
            x_t, y_t)
        if (point_discovery != discovery_nodata and
                point_discovery >= discovery and
                point_discovery <= finish):
            boundary_list.append((int(x_t), int(y_t)))
            # there's only one diagonal to fill in so it's done here
            return

    # if there's a degenerate case then just add the xdelta,
    # it doesn't matter
    boundary_list.append(test_list[0])


cdef int _in_watershed(
        int x_l, int y_l, int direction_to_test, int discovery, int finish,
        int n_cols, int n_rows,
        _ManagedRaster discovery_managed_raster,
        long discovery_nodata):
    """Test if pixel in direction is in the watershed.

    Args:
        x_l/y_l (int): leading coordinate of the watershed boundary
            edge.
        direction_to_test (int): D8 direction that points which direction the edge
            came from
        discovery/finish (long): the discovery and finish time that defines
            whether a pixel discovery time is inside a watershed or not.
        n_cols/n_rows (int): number of columns/rows in the discovery raster,
            used to ensure step does not go out of bounds.
        discovery_managed_raster (_ManagedRaster): discovery time raster
            x/y gives the discovery time for that pixel.
        discovery_nodata (long): nodata value for discovery raster

    Return:
        1 if in, 0 if out.
    """
    cdef int x_n = x_l + D8_XOFFSET[direction_to_test]
    cdef int y_n = y_l + D8_YOFFSET[direction_to_test]
    if x_n < 0 or y_n < 0 or x_n >= n_cols or y_n >= n_rows:
        return 0
    cdef long point_discovery = <long>discovery_managed_raster.get(x_n, y_n)
    return (point_discovery != discovery_nodata and
            point_discovery >= discovery and
            point_discovery <= finish)


cdef _calculate_stream_geometry(
        int x_l, int y_l, int upstream_d8_dir, geotransform, int n_cols,
        int n_rows, _ManagedRaster flow_accum_managed_raster,
        _ManagedRaster flow_dir_managed_raster, int flow_dir_nodata,
        int flow_accum_threshold, coord_to_stream_ids):
    """Calculate the upstream geometry from the given point.

    Creates a new georeferenced linestring geometry that maps the source x/y
    to an upstream line such that the upper point stops when flow accum <
    the provided threshold.

    Args:
        x_l/y_l (int): integer x/y downstream coordinates to seed the search.
        upstream_d8_dir (int): upstream D8 direction to search
        geotransform (list): 6 element list representing the geotransform
            used to convert to georeferenced coordinates.
        n_cols/n_rows (int): number of columns and rows in raster.
        flow_accum_managed_raster (ManagedRaster): flow accumulation raster
        flow_dir_managed_raster (ManagedRaster): d8 flow direction raster
        flow_dir_nodata (int): nodata for flow direction
        flow_accum_threshold (int): minimum flow accumulation value to define
            string.
        coord_to_stream_ids (dict): map raster space coordinate tuple to
            a list of stream ids

    Returns:
        A tuple of (x, y, x_1, y_1, l, line) where:

            * x, y raster coordinates of the upstream source of the stream
                segment
            * x_1, y_1 is the first step upstream of the stream segment
                in raster coordinates. Useful when iterating from the
                confluence or one step up.
            * l is the list of upstream stream IDs at the upstream point
            * and `stream_line` is a georeferenced linestring connecting x/y
                to upper point where upper point's threshold is the last
                point where its flow accum value is >=
                ``flow_accum_threshold``.

        Or ``None` if the point at (x_l, y_l) is below flow accum threshold.

    """
    cdef int x_1, y_1, x_n, y_n, d, d_n, stream_end=0, pixel_length

    if flow_accum_managed_raster.get(x_l, y_l) < flow_accum_threshold:
        return None
    upstream_id_list = []
    # anchor the line at the downstream end
    stream_line = ogr.Geometry(ogr.wkbLineString)
    x_p, y_p = gdal.ApplyGeoTransform(geotransform, x_l+0.5, y_l+0.5)
    stream_line.AddPoint(x_p, y_p)

    # initialize next_dir and last_dir so we only drop new points when
    # the line changes direction
    cdef int next_dir = upstream_d8_dir
    cdef int last_dir = next_dir

    stream_end = 0
    pixel_length = 0
    # initialize these for the compiler warniing
    x_1 = -1
    y_1 = -1
    while not stream_end:
        # walk upstream
        x_l += D8_XOFFSET[next_dir]
        y_l += D8_YOFFSET[next_dir]

        stream_end = 1
        pixel_length += 1
        # do <= 1 in case there's a degenerate single point stream
        if pixel_length <= 1:
            x_1 = x_l
            y_1 = y_l

        # check if we reached an upstream junction
        if (x_l, y_l) in coord_to_stream_ids:
            upstream_id_list = coord_to_stream_ids[(x_l, y_l)]
            del coord_to_stream_ids[(x_l, y_l)]
        elif <int>flow_accum_managed_raster.get(x_l, y_l) >= \
                flow_accum_threshold:
            # check to see if we can take a step upstream
            for d in range(8):
                x_n = x_l + D8_XOFFSET[d]
                y_n = y_l + D8_YOFFSET[d]

                # check out of bounds
                if x_n < 0 or y_n < 0 or x_n >= n_cols or y_n >= n_rows:
                    continue

                # check for nodata
                d_n = <int>flow_dir_managed_raster.get(x_n, y_n)
                if d_n == flow_dir_nodata:
                    continue

                # check if there's an upstream inflow pixel with flow accum
                # greater than the threshold
                if D8_REVERSE_DIRECTION[d] == d_n and (
                        <int>flow_accum_managed_raster.get(
                         x_n, y_n) > flow_accum_threshold):
                    stream_end = 0
                    next_dir = d
                    break
        else:
            # terminated because of flow accumulation too small, so back up
            # one pixel
            pixel_length -= 1

        # drop a point on the line if direction changed or last point
        if last_dir != next_dir or stream_end:
            x_p, y_p = gdal.ApplyGeoTransform(
                geotransform, x_l+0.5, y_l+0.5)
            stream_line.AddPoint(x_p, y_p)
            last_dir = next_dir

    if pixel_length == 0:
        return None
    return x_l, y_l, x_1, y_1, upstream_id_list, stream_line


def _delete_feature(
        stream_feature, stream_layer, upstream_to_downstream_id,
        downstream_to_upstream_ids):
    """Helper for Strahler extraction to delete all references to a stream.

    Args:
        stream_feature (ogr.Feature): feature to delete
        stream_layer (ogr.Layer): layer to delete the feature
        upstream_to_downstream_id (dict): can be referenced by FID and should
            remove all instances of stream from this dict
        downstream_to_upstream_ids (dict): stream feature contained in
            the values of this dict and should remove all instances of stream
            from this dict

    Returns:
        None.
    """
    stream_fid = stream_feature.GetFID()
    if stream_fid in upstream_to_downstream_id:
        downstream_fid = upstream_to_downstream_id[
            stream_fid]
        del upstream_to_downstream_id[stream_fid]
        if downstream_fid in downstream_to_upstream_ids:
            downstream_to_upstream_ids[
                downstream_fid].remove(stream_fid)
    if stream_fid in downstream_to_upstream_ids:
        del downstream_to_upstream_ids[
            stream_fid]
    stream_layer.DeleteFeature(stream_fid)
