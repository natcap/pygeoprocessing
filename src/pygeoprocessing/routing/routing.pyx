# xxxxcython: linetrace=True
# xxxxdistutils: define_macros=CYTHON_TRACE_NOGIL=1
# distutils: language=c++
"""
Provides PyGeprocessing Routing functionality.

Unless otherwise specified, all internal computation of rasters are done in
a float64 space. The only possible loss of precision could occur when an
incoming DEM type is an int64 type and values in that dem exceed 2^52 but GDAL
does not support int64 rasters so no precision loss is possible with a
float64.

D8 float direction conventions follow TauDEM where each flow direction
is encoded as:
    # 321
    # 4 0
    # 567
"""
import time
import errno
import os
import logging
import shutil
import tempfile

import numpy
import scipy.signal
import pygeoprocessing
from .. import geoprocessing
from osgeo import gdal

cimport numpy
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.time cimport time_t
from libc.time cimport time as ctime
from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libc.math cimport isnan
from libcpp.list cimport list
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.pair cimport pair
from libcpp.queue cimport queue
from libcpp.stack cimport stack
from libcpp.deque cimport deque
from libcpp.set cimport set

# This module expects rasters with a memory xy block size of 2**BLOCK_BITS
cdef int BLOCK_BITS = 8
cdef int MANAGED_RASTER_N_BLOCKS = 2**6
NODATA = -1
# if nodata is not defined for a float, it's a difficult choice. this number
# probably won't collide
IMPROBABLE_FLOAT_NOATA = -1.23789789e29
cdef int _NODATA = NODATA
GDAL_INTERNAL_RASTER_TYPE = gdal.GDT_Float64

cdef bint isclose(double a, double b):
    return abs(a - b) <= (1e-5 + 1e-7 * abs(b))

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

cdef struct PitSeed:
    int xoff
    int yoff
    int xi
    int yi
    double height

ctypedef (PitSeed*) PitSeedPtr

# this is the class type that'll get stored in the priority queue
cdef struct Pixel:
    double value  # pixel value
    int xi  # pixel x coordinate in the raster
    int yi  # pixel y coordinate in the raster

ctypedef (Pixel*) PixelPtr

cdef struct FlowPixel:
    int n_i  # flow direction of pixel
    int xi # pixel x coordinate in the raster
    int yi # pixel y coordinate in the raster
    double flow_val  # flow accumulation value

cdef extern from "LRUCache.h" nogil:
    cdef cppclass LRUCache[KEY_T, VAL_T]:
        LRUCache(int)
        void put(KEY_T&, VAL_T&, list[pair[KEY_T,VAL_T]]&)
        list[pair[KEY_T,VAL_T]].iterator begin()
        list[pair[KEY_T,VAL_T]].iterator end()
        bint exist(KEY_T &)
        VAL_T get(KEY_T &)

ctypedef pair[int, double*] BlockBufferPair

cdef class ManagedRaster:
    cdef LRUCache[int, double*]* lru_cache
    cdef set[int] dirty_blocks
    cdef int block_xsize
    cdef int block_ysize
    cdef int raster_x_size
    cdef int raster_y_size
    cdef int block_nx
    cdef int block_ny
    cdef int write_mode
    cdef int block_bits
    cdef char* raster_path
    cdef long int cache_misses
    cdef long int gets
    cdef long int sets
    cdef int closed


    def __cinit__(self, char* raster_path, n_blocks, write_mode):
        """Create new instance of Managed Raster.

        Parameters:
            raster_path (char*): path to raster
            n_blocks (int): number of raster memory blocks to store in the
                cache.
            write_mode (boolean): if true, this raster is writable and dirty
                memory blocks will be written back to the raster as blocks
                are swapped out of the cache or when the object deconstructs.

        Returns:
            None.
        """
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        self.raster_x_size, self.raster_y_size = raster_info['raster_size']
        self.block_xsize, self.block_ysize = raster_info['block_size']
        self.block_bits = BLOCK_BITS
        self.block_xsize = 1 << self.block_bits
        self.block_ysize = 1 << self.block_bits
        self.block_nx = (
            self.raster_x_size + (self.block_xsize) - 1) / self.block_xsize
        self.block_ny = (
            self.raster_y_size + (self.block_ysize) - 1) / self.block_ysize

        self.lru_cache = new LRUCache[int, double*](n_blocks)
        self.raster_path = raster_path
        self.write_mode = write_mode
        self.cache_misses = 0
        self.sets = 0
        self.gets = 0
        self.closed = 0

    def __dealloc__(self):
        """Deallocate ManagedRaster.

        This operation manually frees memory from the LRUCache and writes any
        dirty memory blocks back to the raster if `self.write_mode` is True.
        """
        self.close()

    def close(self):
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

        cdef list[BlockBufferPair].iterator it = self.lru_cache.begin()
        cdef list[BlockBufferPair].iterator end = self.lru_cache.end()
        if not self.write_mode:
            while it != end:
                # write the changed value back if desired
                PyMem_Free(deref(it).second)
                inc(it)
            return

        raster = gdal.Open(self.raster_path, gdal.GA_Update)
        raster_band = raster.GetRasterBand(1)

        # if we get here, we're in write_mode
        cdef set[int].iterator dirty_itr
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
                xoff = block_xi * self.block_xsize
                yoff = block_yi * self.block_ysize

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
                            double_buffer[yi_copy * self.block_xsize + xi_copy])
                raster_band.WriteArray(
                    block_array[0:win_ysize, 0:win_xsize],
                    xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            inc(it)
        raster_band.FlushCache()
        raster_band = None
        raster = None

    cdef void set(self, int xi, int yi, double value):
        self.sets += 1
        cdef int block_xi = xi >> self.block_bits
        cdef int block_yi = yi >> self.block_bits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        cdef int xoff = block_xi << self.block_bits
        cdef int yoff = block_yi << self.block_bits
        self.lru_cache.get(
            block_index)[(yi-yoff)*self.block_xsize+xi-xoff] = value
        if self.write_mode:
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr == self.dirty_blocks.end():
                self.dirty_blocks.insert(block_index)

    cdef double get(self, int xi, int yi):
        self.gets += 1
        cdef int block_xi = xi >> self.block_bits
        cdef int block_yi = yi >> self.block_bits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        cdef int xoff = block_xi << self.block_bits
        cdef int yoff = block_yi << self.block_bits
        return self.lru_cache.get(
            block_index)[(yi-yoff)*self.block_xsize+xi-xoff]

    cdef void _load_block(self, int block_index) except *:
        cdef int block_xi = block_index % self.block_nx
        cdef int block_yi = block_index / self.block_nx

        # we need the offsets to subtract from global indexes for cached array
        cdef int xoff = block_xi * self.block_xsize
        cdef int yoff = block_yi * self.block_ysize

        cdef int xi_copy, yi_copy
        cdef numpy.ndarray[double, ndim=2] block_array
        cdef double *double_buffer
        cdef list[BlockBufferPair] removed_value_list

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

        raster = gdal.Open(self.raster_path)
        raster_band = raster.GetRasterBand(1)
        block_array = raster_band.ReadAsArray(
            xoff=xoff, yoff=yoff, win_xsize=win_xsize,
            win_ysize=win_ysize).astype(
            numpy.float64)
        raster_band = None
        raster = None
        double_buffer = <double*>PyMem_Malloc(
            sizeof(double) * self.block_xsize * win_ysize)
        for xi_copy in xrange(win_xsize):
            for yi_copy in xrange(win_ysize):
                double_buffer[yi_copy*self.block_xsize+xi_copy] = (
                    block_array[yi_copy, xi_copy])
        self.lru_cache.put(
            <int>block_index, <double*>double_buffer, removed_value_list)

        if self.write_mode:
            raster = gdal.Open(self.raster_path, gdal.GA_Update)
            raster_band = raster.GetRasterBand(1)

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
                    block_yi = block_index / self.block_nx

                    xoff = block_xi * self.block_xsize
                    yoff = block_yi * self.block_ysize

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
                            block_array[yi_copy, xi_copy] = double_buffer[
                                yi_copy * self.block_xsize + xi_copy]
                    raster_band.WriteArray(
                        block_array[0:win_ysize, 0:win_xsize],
                        xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            removed_value_list.pop_front()

        if self.write_mode:
            raster_band = None
            raster = None
        self.cache_misses += 1

ctypedef pair[int, int] CoordinatePair

# This functor is used to determine order in the priority queue by comparing
# value only.
cdef cppclass GreaterPixel nogil:
    bint get "operator()"(PixelPtr& lhs, PixelPtr& rhs):
        return deref(lhs).value > deref(rhs).value

# This is used to identify pit pixels to prioritize and group them by local
# block
cdef cppclass GreaterPitSeed nogil:
    bint get "operator()"(PitSeedPtr& lhs, PitSeedPtr& rhs):
        if deref(lhs).height > deref(rhs).height:
            return 1
        if deref(lhs).height == deref(rhs).height:
            if deref(lhs).xoff > deref(rhs).xoff:
                return 1
            if deref(lhs).xoff == deref(rhs).xoff and (
                    deref(lhs).yoff > deref(rhs).yoff):
                return 1
        return 0


def fill_pits(
        dem_raster_path_band, target_filled_dem_raster_path,
        working_dir=None):
    """Drain the plateaus from a trivially routed flow direction raster.

    Parameters:
        dem_raster_path_band (tuple): a path, band number tuple indicating the
            DEM calculate flow direction.
        target_filled_dem_raster_path (string): path to pit filled dem.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run.


    Returns:
        None.
    """
    # TODO: review these variable names and make sure they make sense
    # TODO: should working directory exist beforehand?
    cdef numpy.ndarray[numpy.float64_t, ndim=2] dem_buffer_array
    cdef int win_ysize, win_xsize
    cdef int xoff, yoff, i, xi, yi, xi_q, yi_q, xi_n, yi_n
    cdef int xi_root, yi_root
    cdef int raster_x_size, raster_y_size
    cdef double center_val, dem_nodata, fill_height
    cdef int feature_id
    cdef int downhill_neighbor, nodata_neighbor, downhill_drain, nodata_drain
    cdef queue[CoordinatePair] search_queue, fill_queue
    cdef priority_queue[PitSeedPtr, deque[PitSeedPtr], GreaterPitSeed] pit_queue
    cdef PitSeedPtr pitseed

    try:
        os.makedirs(working_dir)
    except OSError:
        pass

    working_dir_path = tempfile.mkdtemp(
        dir=working_dir, prefix='fill_pits_%s_' % time.strftime(
            '%Y-%m-%d_%H_%M_%S', time.gmtime()))

    logger = logging.getLogger(
        'pygeoprocessing.routing.detect_plateus_and_drains')
    logger.addHandler(logging.NullHandler())  # silence logging by default

    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_path_band[0])
    base_nodata = dem_raster_info['nodata'][dem_raster_path_band[1]-1]
    if base_nodata is not None:
        # cast to a float64 since that's our operating array type
        dem_nodata = numpy.float64(base_nodata)
    else:
        # pick some very improbable value since it's hard to deal with NaNs
        dem_nodata = IMPROBABLE_FLOAT_NOATA
    dem_raster = gdal.OpenEx(dem_raster_path_band[0], gdal.OF_RASTER)
    dem_band = dem_raster.GetRasterBand(dem_raster_path_band[1])

    cdef int* OFFSET_ARRAY = [
        1, 0,  # 0
        1, -1,  # 1
        0, -1,  # 2
        -1, -1,  # 3
        -1, 0,  # 4
        -1, 1,  # 5
        0, 1,  # 6
        1, 1  # 7
        ]

    # these are used to determine if a sample is within the raster
    raster_x_size, raster_y_size = dem_raster_info['raster_size']

    mask_path = os.path.join(working_dir_path, 'mask.tif')
    mask_nodata = -1
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], mask_path, gdal.GDT_Int32,
        [mask_nodata], fill_value_list=[mask_nodata],
        gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS)))

    pit_mask_path = os.path.join(working_dir_path, 'pit_mask.tif')

    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], pit_mask_path, gdal.GDT_Int32,
        [mask_nodata], fill_value_list=[mask_nodata],
        gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS)))

    # this raster is used to keep track of what pixels have been searched for
    # a plateau or pit. if a pixel is set, it means it is connected to a
    # plateau or pit whose value is equal to the ID associated with that
    # region
    mask_managed_raster = ManagedRaster(
        mask_path, MANAGED_RASTER_N_BLOCKS, 1)

    # this raster will have the value of 'feature_id' set to it if it has
    # been searched as part of the search for a pour point for feature
    # `feature_id`
    pit_mask_managed_raster = ManagedRaster(
        pit_mask_path, MANAGED_RASTER_N_BLOCKS, 1)

    gdal_driver = gdal.GetDriverByName('GTiff')
    dem_raster = gdal.Open(dem_raster_path_band[0])
    gdal_driver.CreateCopy(target_filled_dem_raster_path, dem_raster)

    dem_raster = gdal.OpenEx(target_filled_dem_raster_path, gdal.OF_RASTER)
    dem_band = dem_raster.GetRasterBand(dem_raster_path_band[1])

    filled_dem_managed_raster = ManagedRaster(
        target_filled_dem_raster_path, MANAGED_RASTER_N_BLOCKS, 1)

    feature_id = -1
    for offset_dict in pygeoprocessing.iterblocks(
            dem_raster_path_band[0], offset_only=True, largest_block=0):
        # statically type these for later
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        # make a buffer big enough to capture block and boundaries around it
        dem_buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.float64)
        dem_buffer_array[:] = dem_nodata

        # default numpy array boundaries
        buffer_off = {
            'xa': 1,
            'xb': -1,
            'ya': 1,
            'yb': -1
        }
        # check if we can widen the border to include real data from the
        # raster
        for a_buffer_id, b_buffer_id, off_id, win_size_id, raster_size in [
                ('xa', 'xb', 'xoff', 'win_xsize', raster_x_size),
                ('ya', 'yb', 'yoff', 'win_ysize', raster_y_size)]:
            if offset_dict[off_id] > 0:
                # in this case we have valid data to the left (or up)
                # grow the window and buffer slice in that direction
                buffer_off[a_buffer_id] = None
                offset_dict[off_id] -= 1
                offset_dict[win_size_id] += 1

            if offset_dict[off_id] + offset_dict[win_size_id] < raster_size:
                # here we have valid data to the right (or bottom)
                # grow the right buffer and add 1 to window
                buffer_off[b_buffer_id] = None
                offset_dict[win_size_id] += 1

        # read in the valid memory block
        dem_buffer_array[
            buffer_off['ya']:buffer_off['yb'],
            buffer_off['xa']:buffer_off['xb']] = dem_band.ReadAsArray(
                **offset_dict).astype(numpy.float64)

        # search block for undrained pixels
        for yi in xrange(1, win_ysize+1):
            for xi in xrange(1, win_xsize+1):
                center_val = dem_buffer_array[yi, xi]
                if isclose(center_val, dem_nodata):
                    continue

                # this value is set in case it turns out to be the root of a
                # pit, the fill will start from this pixel outward
                xi_root = xi-1+xoff
                yi_root = yi-1+yoff

                if not isclose(
                        mask_nodata,
                        mask_managed_raster.get(xi_root, yi_root)):
                    # already been searched
                    continue

                # search neighbors for downhill or nodata
                downhill_neighbor = 0
                nodata_neighbor = 0

                for i in xrange(8):
                    xi_n = xi_root+OFFSET_ARRAY[2*i]
                    yi_n = yi_root+OFFSET_ARRAY[2*i+1]
                    if (xi_n < 0 or xi_n >= raster_x_size or
                            yi_n < 0 or yi_n >= raster_y_size):
                        # it'll drain off the edge of the raster
                        nodata_neighbor = 1
                        break
                    if isclose(dem_nodata, filled_dem_managed_raster.get(
                            xi_n, yi_n)):
                        # it'll drain to nodata
                        nodata_neighbor = 1
                        break
                    n_height = filled_dem_managed_raster.get(xi_n, yi_n)
                    if n_height < center_val:
                        # it'll drain downhill
                        downhill_neighbor = 1
                        break

                if downhill_neighbor or nodata_neighbor:
                    continue

                # otherwise, this pixel doesn't drain locally, search to see
                # if it's a pit or plateau
                feature_id += 1
                search_queue.push(CoordinatePair(xi_root, yi_root))
                mask_managed_raster.set(xi_root, yi_root, feature_id)
                downhill_drain = 0
                nodata_drain = 0

                # this loop does a BFS starting at this pixel to all pixels
                # of the same height. at the end it'll remember if it drained
                # or not
                while not search_queue.empty():
                    xi_q = search_queue.front().first
                    yi_q = search_queue.front().second
                    search_queue.pop()

                    for i in xrange(8):
                        xi_n = xi_q+OFFSET_ARRAY[2*i]
                        yi_n = yi_q+OFFSET_ARRAY[2*i+1]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            nodata_drain = 1
                            continue
                        if isclose(dem_nodata, filled_dem_managed_raster.get(
                                xi_n, yi_n)):
                            nodata_drain = 1
                            continue
                        n_height = filled_dem_managed_raster.get(
                            xi_n, yi_n)
                        if n_height < center_val:
                            downhill_drain = 1
                            continue
                        if n_height == center_val and isclose(
                                mask_nodata, mask_managed_raster.get(
                                    xi_n, yi_n)):
                            # only grow if it's at the same level and not
                            # previously visited
                            search_queue.push(
                                CoordinatePair(xi_n, yi_n))
                            mask_managed_raster.set(
                                xi_n, yi_n, feature_id)

                if not downhill_drain and not nodata_drain:
                    # entire region was searched with no drain, do a fill
                    # and prioritize visit by block defined by xoff/yoff
                    pitseed = <PitSeed*>PyMem_Malloc(sizeof(PitSeed))
                    deref(pitseed).xoff = xoff
                    deref(pitseed).yoff = yoff
                    deref(pitseed).xi = xi_root
                    deref(pitseed).yi = yi_root
                    deref(pitseed).height = center_val
                    pit_mask_managed_raster.set(
                        xi_root, yi_root, feature_id)
                    pit_queue.push(pitseed)

                # this loop visits pixels in increasing height order, so the
                # first non-processed pixel that's < pitseed.height or nodata
                # will be the lowest pour point
                pour_point = 0
                fill_height = dem_nodata
                while not pit_queue.empty():
                    pitseed = pit_queue.top()
                    pit_queue.pop()
                    xi_q = deref(pitseed).xi
                    yi_q = deref(pitseed).yi

                    # this is the potential fill height if pixel is pour point
                    fill_height = deref(pitseed).height

                    logger.debug(
                        'visiting block %d %d',
                        deref(pitseed).xoff, deref(pitseed).yoff)
                    PyMem_Free(pitseed)

                    for i in xrange(8):
                        xi_n = xi_q+OFFSET_ARRAY[2*i]
                        yi_n = yi_q+OFFSET_ARRAY[2*i+1]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            nodata_drain = 1
                            break

                        if pit_mask_managed_raster.get(
                                xi_n, yi_n) == feature_id:
                            # this cell has already been processed
                            continue
                        # mark as visited in the search for pour point
                        pit_mask_managed_raster.set(
                            xi_n, yi_n, feature_id)

                        n_height = filled_dem_managed_raster.get(
                            xi_n, yi_n)
                        if isclose(n_height, dem_nodata) or (
                                n_height < fill_height):
                            # we encounter a pixel not processed that is less
                            # than the neighbor's height or nodata, it is a
                            # pour point
                            pour_point = 1
                            break

                        # push onto queue
                        pitseed = <PitSeed*>PyMem_Malloc(sizeof(PitSeed))
                        # TODO: get correct xoff/yoff or remove entirely
                        deref(pitseed).xoff = xoff
                        deref(pitseed).yoff = yoff
                        deref(pitseed).xi = xi_n
                        deref(pitseed).yi = yi_n
                        deref(pitseed).height = n_height
                        pit_queue.push(pitseed)

                    if pour_point:
                        # clear the queue
                        while not pit_queue.empty():
                            PyMem_Free(pit_queue.top())
                            pit_queue.pop()

                        # start from original pit seed rather than pour point
                        # this way we can stop filling when we reach a height
                        # equal to fill_height
                        fill_queue.push(CoordinatePair(xi_root, yi_root))

                # TODO: assert pour_point == 1?
                # TODO: assert fill-height != nodata?
                while not fill_queue.empty():
                    xi_q = fill_queue.front().first
                    yi_q = fill_queue.front().second
                    fill_queue.pop()
                    filled_dem_managed_raster.set(xi_q, yi_q, fill_height)

                    for i in xrange(8):
                        xi_n = xi_q+OFFSET_ARRAY[2*i]
                        yi_n = yi_q+OFFSET_ARRAY[2*i+1]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            continue

                        if filled_dem_managed_raster.get(
                                xi_n, yi_n) < fill_height:
                            filled_dem_managed_raster.set(
                                xi_n, yi_n, fill_height)
                            fill_queue.push(
                                CoordinatePair(xi_n, yi_n))
    pit_mask_managed_raster.close()
    mask_managed_raster.close()
    shutil.rmtree(working_dir_path)


def simple_d8(
        dem_raster_path_band, target_flow_direction_path):
    """Calculate D8 flow directions on simple grid.

    Parameters:
        dem_raster_path_band (tuple): a path, band number tuple indicating the
            DEM calculate flow direction.
        target_flow_direction_path (string): path to a int8 single band raster
            that will be created as a flow direction output
        temp_dir_path (string): if not None, indicates where algorithm can
            construct intermediate files for bookkeeping during algorithm
            processing.

    Returns:
        None.
    """
    cdef numpy.ndarray[numpy.float64_t, ndim=2] buffer_array
    cdef int win_ysize, win_xsize
    cdef int xoff, yoff, i, xi, yi
    cdef int raster_x_size, raster_y_size
    cdef double c_min, center_val, n_val, dem_nodata
    cdef int c_min_index, nodata_index

    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_path_band[0])
    base_nodata = dem_raster_info['nodata'][dem_raster_path_band[1]-1]
    if base_nodata is not None:
        # cast to a float64 since that's our operating array type
        dem_nodata = numpy.float64(base_nodata)
    else:
        # pick some very improbable value since it's hard to deal with NaNs
        dem_nodata = IMPROBABLE_FLOAT_NOATA
    dem_raster = gdal.OpenEx(dem_raster_path_band[0], gdal.OF_RASTER)
    dem_band = dem_raster.GetRasterBand(dem_raster_path_band[1])

    raster_x_size, raster_y_size = dem_raster_info['raster_size']

    cdef int* OFFSET_ARRAY = [
        1, 0,  # 0
        1, -1,  # 1
        0, -1,  # 2
        -1, -1,  # 3
        -1, 0,  # 4
        -1, 1,  # 5
        0, 1,  # 6
        1, 1  # 7
        ]

    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], target_flow_direction_path, gdal.GDT_Byte,
        [255], fill_value_list=[255], gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS)))

    # these are used to determine if a sample is within the raster
    raster_x_size, raster_y_size = dem_raster_info['raster_size']

    # used to set flow directions
    flow_dir_managed_raster = ManagedRaster(
        target_flow_direction_path, MANAGED_RASTER_N_BLOCKS, 1)

    for offset_dict in pygeoprocessing.iterblocks(
        dem_raster_path_band[0], offset_only=True, largest_block=0):

        # statically type these for later
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        # make a buffer big enough to capture block and boundaries around it
        buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.float64)
        buffer_array[:] = dem_nodata

        # default numpy array boundaries
        buffer_off = {
            'xa': 1,
            'xb': -1,
            'ya': 1,
            'yb': -1
        }
        # check if we can widen the border to include real data from the
        # raster
        for a_buffer_id, b_buffer_id, off_id, win_size_id, raster_size in [
                ('xa', 'xb', 'xoff', 'win_xsize', raster_x_size),
                ('ya', 'yb', 'yoff', 'win_ysize', raster_y_size)]:
            if offset_dict[off_id] > 0:
                # in this case we have valid data to the left (or up)
                # grow the window and buffer slice in that direction
                buffer_off[a_buffer_id] = None
                offset_dict[off_id] -= 1
                offset_dict[win_size_id] += 1

            if offset_dict[off_id] + offset_dict[win_size_id] < raster_size:
                # here we have valid data to the right (or bottom)
                # grow the right buffer and add 1 to window
                buffer_off[b_buffer_id] = None
                offset_dict[win_size_id] += 1

        # read in the valid memory block
        buffer_array[
            buffer_off['ya']:buffer_off['yb'],
            buffer_off['xa']:buffer_off['xb']] = dem_band.ReadAsArray(
                **offset_dict).astype(numpy.float64)

        # irrespective of how we sampled the DEM only look at the block in
        # the middle for valid
        for yi in xrange(1, win_ysize+1):
            for xi in xrange(1, win_xsize+1):
                center_val = buffer_array[yi, xi]
                if isclose(center_val, dem_nodata):
                    continue

                # this uses the offset array to visit the neighbors rather
                # than 8 identical if statements.
                c_min = center_val
                c_min_index = -1
                nodata_index = -1
                for i in xrange(8):
                    n_val = buffer_array[
                        yi+OFFSET_ARRAY[2*i+1], xi+OFFSET_ARRAY[2*i]]
                    if isclose(n_val, dem_nodata):
                        nodata_index = i
                    elif n_val < c_min:
                        c_min = n_val
                        c_min_index = i

                if c_min_index > -1:
                    flow_dir_managed_raster.set(
                        xi-1+xoff, yi-1+yoff, c_min_index)
                elif nodata_index > -1:
                    flow_dir_managed_raster.set(
                        xi-1+xoff, yi-1+yoff, nodata_index)


def drain_plateus_d8(
        dem_raster_path_band, flow_dir_path):
    """Drain the plateaus from a trivially routed flow direction raster.

    Parameters:
        dem_raster_path_band (tuple): a path, band number tuple indicating the
            DEM calculate flow direction.
        flow_dir_path (tuple): a path to a single band
            int8 raster that has flow directions trivially
            defined.

    Returns:
        None.
    """
    cdef numpy.ndarray[numpy.float64_t, ndim=2] dem_buffer_array
    cdef int win_ysize, win_xsize
    cdef int xoff, yoff, i, xi, yi, xi_q, yi_q
    cdef int n_x_off, n_y_off
    cdef int raster_x_size, raster_y_size
    cdef double center_val, dem_nodata
    cdef int flow_dir_nodata
    cdef int blob_nodata, blob_id
    cdef queue[CoordinatePair] fill_queue, drain_queue

    logger = logging.getLogger('pygeoprocessing.routing.drain_plateus_d8')
    logger.addHandler(logging.NullHandler())  # silence logging by default


    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_path_band[0])
    base_nodata = dem_raster_info['nodata'][dem_raster_path_band[1]-1]
    if base_nodata is not None:
        # cast to a float64 since that's our operating array type
        dem_nodata = numpy.float64(base_nodata)
    else:
        # pick some very improbable value since it's hard to deal with NaNs
        dem_nodata = IMPROBABLE_FLOAT_NOATA
    dem_raster = gdal.OpenEx(dem_raster_path_band[0], gdal.OF_RASTER)
    dem_band = dem_raster.GetRasterBand(dem_raster_path_band[1])

    flow_dir_raster_info = pygeoprocessing.get_raster_info(flow_dir_path)
    flow_dir_nodata = flow_dir_raster_info['nodata'][0]

    cdef int* OFFSET_ARRAY = [
        1, 0,  # 0
        1, -1,  # 1
        0, -1,  # 2
        -1, -1,  # 3
        -1, 0,  # 4
        -1, 1,  # 5
        0, 1,  # 6
        1, 1  # 7
        ]

    cdef int* REVERSE_FLOW_DIR = [
        4,  # 0
        5,  # 1
        6,  # 2
        7,  # 3
        0,  # 4
        1,  # 5
        2,  # 6
        3  # 7
    ]

    # these are used to determine if a sample is within the raster
    raster_x_size, raster_y_size = dem_raster_info['raster_size']

    blob_nodata = -1
    blob_raster_path = 'blob_raster.tif'
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], blob_raster_path, gdal.GDT_Int32,
        [blob_nodata], fill_value_list=[blob_nodata], gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS)))

    # used to set flow directions
    flow_dir_managed_raster = ManagedRaster(
        flow_dir_path, MANAGED_RASTER_N_BLOCKS, 1)

    blob_managed_raster = ManagedRaster(
        blob_raster_path, MANAGED_RASTER_N_BLOCKS, 1)

    dem_managed_raster = ManagedRaster(
        dem_raster_path_band[0], MANAGED_RASTER_N_BLOCKS, 0)

    blob_id = -1
    for offset_dict in pygeoprocessing.iterblocks(
            dem_raster_path_band[0], offset_only=True, largest_block=0):
        # statically type these for later
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        # make a buffer big enough to capture block and boundaries around it
        dem_buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.float64)
        dem_buffer_array[:] = dem_nodata

        # default numpy array boundaries
        buffer_off = {
            'xa': 1,
            'xb': -1,
            'ya': 1,
            'yb': -1
        }
        # check if we can widen the border to include real data from the
        # raster
        for a_buffer_id, b_buffer_id, off_id, win_size_id, raster_size in [
                ('xa', 'xb', 'xoff', 'win_xsize', raster_x_size),
                ('ya', 'yb', 'yoff', 'win_ysize', raster_y_size)]:
            if offset_dict[off_id] > 0:
                # in this case we have valid data to the left (or up)
                # grow the window and buffer slice in that direction
                buffer_off[a_buffer_id] = None
                offset_dict[off_id] -= 1
                offset_dict[win_size_id] += 1

            if offset_dict[off_id] + offset_dict[win_size_id] < raster_size:
                # here we have valid data to the right (or bottom)
                # grow the right buffer and add 1 to window
                buffer_off[b_buffer_id] = None
                offset_dict[win_size_id] += 1

        # read in the valid memory block
        dem_buffer_array[
            buffer_off['ya']:buffer_off['yb'],
            buffer_off['xa']:buffer_off['xb']] = dem_band.ReadAsArray(
                **offset_dict).astype(numpy.float64)

        # irrespective of how we sampled the DEM only look at the block in
        # the middle for valid
        for yi in xrange(1, win_ysize+1):
            for xi in xrange(1, win_xsize+1):
                center_val = dem_buffer_array[yi, xi]
                if isclose(center_val, dem_nodata):
                    continue
                if not isclose(
                    flow_dir_nodata,
                    flow_dir_managed_raster.get(xi-1+xoff, yi-1+yoff)):
                    # if the flow direction is defined it won't be a plateau
                    continue

                if not isclose(
                        blob_nodata,
                        blob_managed_raster.get(xi-1+xoff, yi-1+yoff)):
                    # we've visited this pixel before, we'll get to it
                    continue

                # flow direction is undefined and we haven't visited anymore
                # we're visiting a new plateau
                blob_id += 1
                fill_queue.push(CoordinatePair(xi-1+xoff, yi-1+yoff))
                blob_managed_raster.set(xi-1+xoff, yi-1+yoff, blob_id)
                while not fill_queue.empty():
                    xi_q = fill_queue.front().first
                    yi_q = fill_queue.front().second
                    fill_queue.pop()

                    # this is a pixel in a plateau
                    drain_pushed = 0
                    for i in xrange(8):
                        n_x_off = xi_q+OFFSET_ARRAY[2*i]
                        n_y_off = yi_q+OFFSET_ARRAY[2*i+1]
                        if (n_x_off < 0 or n_x_off >= raster_x_size or
                                n_y_off < 0 or n_y_off >= raster_y_size):
                            # don't visit neighbors that run off the edge
                            continue
                        if isclose(dem_nodata, dem_managed_raster.get(
                                n_x_off, n_y_off)):
                            # if neighbor is undefined dem, skip
                            continue
                        neighbor_flow_dir = flow_dir_managed_raster.get(
                            n_x_off, n_y_off)
                        if isclose(flow_dir_nodata, neighbor_flow_dir) and (
                                isclose(
                                    blob_nodata,
                                    blob_managed_raster.get(
                                        n_x_off, n_y_off))) and (
                                dem_managed_raster.get(
                                    n_x_off, n_y_off) == center_val):
                            # if neighbor flow dir is undefined and at the
                            # same height and we haven't visited before,
                            # expand fill to that pixel
                            fill_queue.push(CoordinatePair(n_x_off, n_y_off))
                            blob_managed_raster.set(n_x_off, n_y_off, blob_id)
                            continue
                        if not isclose(
                                flow_dir_nodata, neighbor_flow_dir) and (
                                    dem_managed_raster.get(
                                        n_x_off, n_y_off) <= center_val and
                                    not drain_pushed):
                            # the neighbor is flow dir defined and at correct
                            # height to be a drain, push this pixel to drain
                            # queue
                            flow_dir_managed_raster.set(xi_q, yi_q, i)
                            drain_queue.push(CoordinatePair(xi_q, yi_q))
                            drain_pushed = 1

                while not drain_queue.empty():
                    xi_q = drain_queue.front().first
                    yi_q = drain_queue.front().second
                    drain_queue.pop()

                    # search for defined neighbors
                    for i in xrange(8):
                        n_x_off = xi_q+OFFSET_ARRAY[2*i]
                        n_y_off = yi_q+OFFSET_ARRAY[2*i+1]
                        if (n_x_off < 0 or n_x_off >= raster_x_size or
                                n_y_off < 0 or n_y_off >= raster_y_size):
                            continue
                        n_height = dem_managed_raster.get(n_x_off, n_y_off)
                        if isclose(
                            flow_dir_nodata,
                            flow_dir_managed_raster.get(
                                n_x_off, n_y_off)) and n_height == center_val:
                            # if the neighbor has a defined flow direction and is same height as plateau, it drains the current cell
                            flow_dir_managed_raster.set(
                                n_x_off, n_y_off, REVERSE_FLOW_DIR[i])
                            drain_queue.push(CoordinatePair(n_x_off, n_y_off))

def detect_plateus_d8(
        dem_raster_path_band, flow_dir_path, target_plateau_mask_path):
    """Drain the plateaus from a trivially routed flow direction raster.

    Parameters:
        dem_raster_path_band (tuple): a path, band number tuple indicating the
            DEM calculate flow direction.
        flow_dir_path (tuple): a path to a single band
            int8 raster that has flow directions trivially
            defined.
        target_plateau_mask_path (string): path to output raster that will
            have a non-nodata value when a pixel is part of a plateau.

    Returns:
        None.
    """
    cdef numpy.ndarray[numpy.float64_t, ndim=2] dem_buffer_array
    cdef int win_ysize, win_xsize
    cdef int xoff, yoff, i, xi, yi, xi_q, yi_q
    cdef int n_x_off, n_y_off
    cdef int raster_x_size, raster_y_size
    cdef double center_val, dem_nodata
    cdef int flow_dir_nodata
    cdef int blob_nodata
    cdef int plateau_id
    cdef queue[CoordinatePair] fill_queue, search_queue

    logger = logging.getLogger('pygeoprocessing.routing.detect_plateus_d8')
    logger.addHandler(logging.NullHandler())  # silence logging by default


    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_path_band[0])
    base_nodata = dem_raster_info['nodata'][dem_raster_path_band[1]-1]
    if base_nodata is not None:
        # cast to a float64 since that's our operating array type
        dem_nodata = numpy.float64(base_nodata)
    else:
        # pick some very improbable value since it's hard to deal with NaNs
        dem_nodata = IMPROBABLE_FLOAT_NOATA
    dem_raster = gdal.OpenEx(dem_raster_path_band[0], gdal.OF_RASTER)
    dem_band = dem_raster.GetRasterBand(dem_raster_path_band[1])

    flow_dir_raster_info = pygeoprocessing.get_raster_info(flow_dir_path)
    flow_dir_nodata = flow_dir_raster_info['nodata'][0]

    cdef int* OFFSET_ARRAY = [
        1, 0,  # 0
        1, -1,  # 1
        0, -1,  # 2
        -1, -1,  # 3
        -1, 0,  # 4
        -1, 1,  # 5
        0, 1,  # 6
        1, 1  # 7
        ]

    # these are used to determine if a sample is within the raster
    raster_x_size, raster_y_size = dem_raster_info['raster_size']

    plateau_mask_nodata = -1
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], target_plateau_mask_path, gdal.GDT_Int32,
        [plateau_mask_nodata], fill_value_list=[plateau_mask_nodata],
        gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS)))

    blob_nodata = -1
    blob_raster_path = 'plateau_detection_blob.tif'
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], blob_raster_path, gdal.GDT_Int32,
        [blob_nodata], fill_value_list=[blob_nodata],
        gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS)))

    # used to set flow directions
    flow_dir_managed_raster = ManagedRaster(
        flow_dir_path, MANAGED_RASTER_N_BLOCKS, 0)

    plateau_mask_managed_raster = ManagedRaster(
        target_plateau_mask_path, MANAGED_RASTER_N_BLOCKS, 1)

    blob_managed_raster = ManagedRaster(
        blob_raster_path, MANAGED_RASTER_N_BLOCKS, 1)

    dem_managed_raster = ManagedRaster(
        dem_raster_path_band[0], MANAGED_RASTER_N_BLOCKS, 0)

    plateau_id = -1
    for offset_dict in pygeoprocessing.iterblocks(
            dem_raster_path_band[0], offset_only=True, largest_block=0):
        # statically type these for later
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        # make a buffer big enough to capture block and boundaries around it
        dem_buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.float64)
        dem_buffer_array[:] = dem_nodata

        # default numpy array boundaries
        buffer_off = {
            'xa': 1,
            'xb': -1,
            'ya': 1,
            'yb': -1
        }
        # check if we can widen the border to include real data from the
        # raster
        for a_buffer_id, b_buffer_id, off_id, win_size_id, raster_size in [
                ('xa', 'xb', 'xoff', 'win_xsize', raster_x_size),
                ('ya', 'yb', 'yoff', 'win_ysize', raster_y_size)]:
            if offset_dict[off_id] > 0:
                # in this case we have valid data to the left (or up)
                # grow the window and buffer slice in that direction
                buffer_off[a_buffer_id] = None
                offset_dict[off_id] -= 1
                offset_dict[win_size_id] += 1

            if offset_dict[off_id] + offset_dict[win_size_id] < raster_size:
                # here we have valid data to the right (or bottom)
                # grow the right buffer and add 1 to window
                buffer_off[b_buffer_id] = None
                offset_dict[win_size_id] += 1

        # read in the valid memory block
        dem_buffer_array[
            buffer_off['ya']:buffer_off['yb'],
            buffer_off['xa']:buffer_off['xb']] = dem_band.ReadAsArray(
                **offset_dict).astype(numpy.float64)

        # irrespective of how we sampled the DEM only look at the block in
        # the middle for valid
        for yi in xrange(1, win_ysize+1):
            for xi in xrange(1, win_xsize+1):
                center_val = dem_buffer_array[yi, xi]
                if isclose(center_val, dem_nodata):
                    continue
                if not isclose(
                    flow_dir_nodata,
                    flow_dir_managed_raster.get(xi-1+xoff, yi-1+yoff)):
                    # if the flow direction is defined it won't be a plateau
                    continue

                if not isclose(
                        blob_nodata,
                        blob_managed_raster.get(xi-1+xoff, yi-1+yoff)):
                    # we've visited this pixel before so skip
                    continue

                if not isclose(
                        plateau_mask_nodata,
                        plateau_mask_managed_raster.get(
                            xi-1+xoff, yi-1+yoff)):
                    # already been marked as a plateau
                    continue


                # we're visiting a new plateau
                plateau_id += 1
                search_queue.push(CoordinatePair(xi-1+xoff, yi-1+yoff))
                blob_managed_raster.set(xi-1+xoff, yi-1+yoff, plateau_id)
                while not search_queue.empty():
                    xi_q = search_queue.front().first
                    yi_q = search_queue.front().second
                    search_queue.pop()
                    push_drain = 0

                    for i in xrange(8):
                        n_x_off = xi_q+OFFSET_ARRAY[2*i]
                        n_y_off = yi_q+OFFSET_ARRAY[2*i+1]
                        if (n_x_off < 0 or n_x_off >= raster_x_size or
                                n_y_off < 0 or n_y_off >= raster_y_size):
                            # it'll drain off the edge of the raster
                            push_drain = 1
                            break
                        if isclose(dem_nodata, dem_managed_raster.get(
                                n_x_off, n_y_off)):
                            # it'll drain to nodata
                            push_drain = 1
                            break
                        n_height = dem_managed_raster.get(n_x_off, n_y_off)
                        if n_height < center_val:
                            # it'll drain downhill
                            push_drain = 1
                            break
                        if n_height == center_val and isclose(
                                blob_nodata, blob_managed_raster.get(
                                    n_x_off, n_y_off)):
                            search_queue.push(
                                CoordinatePair(n_x_off, n_y_off))
                            blob_managed_raster.set(
                                n_x_off, n_y_off, plateau_id)

                    if push_drain:
                        while not search_queue.empty():
                            search_queue.pop()
                        plateau_mask_managed_raster.set(xi_q, yi_q, plateau_id)
                        fill_queue.push(CoordinatePair(xi_q, yi_q))

                while not fill_queue.empty():
                    xi_q = fill_queue.front().first
                    yi_q = fill_queue.front().second
                    fill_queue.pop()

                    for i in xrange(8):
                        n_x_off = xi_q+OFFSET_ARRAY[2*i]
                        n_y_off = yi_q+OFFSET_ARRAY[2*i+1]
                        if (n_x_off < 0 or n_x_off >= raster_x_size or
                                n_y_off < 0 or n_y_off >= raster_y_size):
                            continue
                        if isclose(
                            plateau_mask_nodata,
                            plateau_mask_managed_raster.get(
                                n_x_off, n_y_off)) and dem_managed_raster.get(
                                    n_x_off, n_y_off) == center_val:
                            # if neighbor is same height (part of plateau)
                            # mark and push to queue
                            plateau_mask_managed_raster.set(
                                n_x_off, n_y_off, plateau_id)
                            fill_queue.push(CoordinatePair(n_x_off, n_y_off))


def flow_accumulation_d8(
        flow_dir_raster_path_band,
        target_flow_accumulation_raster_path, weight_raster_path_band=None,
        temp_dir_path=None):
    """Calculate flow accumulation given D8 flow direction.

    Parameters:
        flow_dir_raster_path_band (tuple): a path, band number tuple
            indicating the D8 flow direction raster with direction convention:

            # 321
            # 4 0
            # 567

            This raster can be created from a call to
            `pygeoprocessing.routing.fill_pits`,

        target_flow_accmulation_raster_path (string): path to single band
            raster to be created. Each pixel value will indicate the number
            of upstream pixels that feed it including the current pixel.
        weight_raster_path_band (tuple): if not None, path to  a raster that
            is of the same dimensions as `flow_dir_raster_path_band`
            that is to be used in place of "1" for flow accumulation per
            pixel.
        temp_dir_path (string): if not None, path to a directory where we can
            create temporary files

    Returns:
        None.
    """
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] buffer_array
    cdef int raster_x_size, raster_y_size
    cdef int xi_n, yi_n, i
    cdef int xi, yi, win_ysize, win_xsize
    cdef int xoff, yoff
    cdef int flow_direction_nodata, n_dir, flow_dir
    cdef int use_weights
    cdef double weight_val
    cdef double weight_nodata = 0  # fill in with something for compiler
    cdef double flow_accum
    cdef stack[FlowPixel] flow_stack
    cdef FlowPixel fp
    cdef ManagedRaster flow_accumulation_managed_raster
    cdef ManagedRaster flow_dir_managed_raster
    cdef ManagedRaster weight_raster_path_raster = None

    logger = logging.getLogger('pygeoprocessing.routing.flow_accumulation')
    logger.addHandler(logging.NullHandler())  # silence logging by default

    # flow direction scheme is
    # 321
    # 4 0
    # 567
    # each flow direction is encoded as 1 << n, n in [0, 7]

    # use this to have offsets to visit neighbor pixels, pick 2 at a time to
    # add to a (xi, yi) tuple
    cdef int* OFFSET_ARRAY = [
        1, 0,  # 0
        1, -1,  # 1
        0, -1,  # 2
        -1, -1,  # 3
        -1, 0,  # 4
        -1, 1,  # 5
        0, 1,  # 6
        1, 1  # 7
        ]

    # this is used to set flow direction on a neighbor by indicating which
    # neighbor it flows to
    cdef int* REVERSE_FLOW_DIR = [
        4,  # 0
        5,  # 1
        6,  # 2
        7,  # 3
        0,  # 4
        1,  # 5
        2,  # 6
        3  # 7
    ]

    # make an interesting temporary directory that has the time/date and
    # 'flow_accumulation' on it so we can figure out what's going on if we
    # ever run across it in a temp dir.
    temp_dir_path = tempfile.mkdtemp(
        dir=temp_dir_path, prefix='flow_accumulation_', suffix=time.strftime(
            '%Y-%m-%d_%H_%M_%S', time.gmtime()))

    pygeoprocessing.new_raster_from_base(
        flow_dir_raster_path_band[0],
        target_flow_accumulation_raster_path, GDAL_INTERNAL_RASTER_TYPE,
        [_NODATA], fill_value_list=[_NODATA],
        gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS)))

    logger.info(
        'flow accumulation raster created at %s',
        target_flow_accumulation_raster_path)

    # these are used to determine if a sample is within the raster
    flow_direction_raster_info = pygeoprocessing.get_raster_info(
        flow_dir_raster_path_band[0])

    base_flow_direction_nodata = flow_direction_raster_info['nodata'][
        flow_dir_raster_path_band[1]-1]

    if base_flow_direction_nodata is not None:
        flow_direction_nodata = base_flow_direction_nodata
    else:
        # pick some impossible value given our conventions
        flow_direction_nodata = 99

    raster_x_size, raster_y_size = flow_direction_raster_info['raster_size']
    # used to set flow directions
    flow_dir_managed_raster = ManagedRaster(
        flow_dir_raster_path_band[0], MANAGED_RASTER_N_BLOCKS, 0)

    # the flow accumulation result
    flow_accumulation_managed_raster = ManagedRaster(
        target_flow_accumulation_raster_path, MANAGED_RASTER_N_BLOCKS, 1)

    use_weights = 0
    if weight_raster_path_band is not None:
        weight_raster_path_raster = ManagedRaster(
            weight_raster_path_band[0], MANAGED_RASTER_N_BLOCKS, 0)
        base_weight_nodata = pygeoprocessing.get_raster_info(
            weight_raster_path_band[0])['nodata'][
                weight_raster_path_band[1]-1]
        if base_weight_nodata is not None:
            weight_nodata = base_weight_nodata
        else:
            weight_nodata = IMPROBABLE_FLOAT_NOATA

        use_weights = 1

    flow_direction_raster = gdal.Open(flow_dir_raster_path_band[0])
    flow_direction_band = flow_direction_raster.GetRasterBand(
        flow_dir_raster_path_band[1])

    logger.info('finding drains')
    start_drain_time = ctime(NULL)
    for offset_dict in pygeoprocessing.iterblocks(
            flow_dir_raster_path_band[0], offset_only=True,
            largest_block=0):
        # statically type these for later
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        # make a buffer big enough to capture block and boundaries around it
        buffer_array = numpy.empty(
            (win_ysize+2, win_xsize+2), dtype=numpy.uint8)
        buffer_array[:] = flow_direction_nodata

        # default numpy array boundaries
        buffer_off = {
            'xa': 1,
            'xb': -1,
            'ya': 1,
            'yb': -1
        }
        # check if we can widen the border to include real data from the
        # raster
        for a_buffer_id, b_buffer_id, off_id, win_size_id, raster_size in [
                ('xa', 'xb', 'xoff', 'win_xsize', raster_x_size),
                ('ya', 'yb', 'yoff', 'win_ysize', raster_y_size)]:
            if offset_dict[off_id] > 0:
                # in this case we have valid data to the left (or up)
                # grow the window and buffer slice in that direction
                buffer_off[a_buffer_id] = None
                offset_dict[off_id] -= 1
                offset_dict[win_size_id] += 1

            if offset_dict[off_id] + offset_dict[win_size_id] < raster_size:
                # here we have valid data to the right (or bottom)
                # grow the right buffer and add 1 to window
                buffer_off[b_buffer_id] = None
                offset_dict[win_size_id] += 1

        # read in the valid memory block
        buffer_array[
            buffer_off['ya']:buffer_off['yb'],
            buffer_off['xa']:buffer_off['xb']] = (
                flow_direction_band.ReadAsArray(
                    **offset_dict).astype(numpy.int8))

        for yi in xrange(1, win_ysize+1):
            for xi in xrange(1, win_xsize+1):
                flow_dir = (buffer_array[yi, xi])
                if isclose(flow_dir, flow_direction_nodata):
                    continue
                n_dir = buffer_array[
                    yi+OFFSET_ARRAY[2*flow_dir+1],
                    xi+OFFSET_ARRAY[2*flow_dir]]
                if isclose(n_dir, flow_direction_nodata):
                    # it flows to nodata (or edge) so it's a seed
                    if use_weights:
                        weight_val = weight_raster_path_raster.get(
                            xi-1+xoff, yi-1+yoff)
                        if isclose(weight_val, weight_nodata):
                            weight_val = 0
                    else:
                        weight_val = 1
                    flow_stack.push(
                        FlowPixel(0, xi-1+xoff, yi-1+yoff, weight_val))

    logger.info("drains detected in %fs", ctime(NULL)-start_drain_time)
    while not flow_stack.empty():
        fp = flow_stack.top()
        flow_stack.pop()

        if (fp.xi == 0 or fp.xi == (raster_x_size-1) or
                fp.yi == 0 or fp.yi == (raster_y_size-1)):
            check_bounds_top = 1
        else:
            check_bounds_top = 0

        all_checked = 1
        for i in xrange(fp.n_i, 8):
            # neighbor x,y indexes
            xi_n = fp.xi+OFFSET_ARRAY[2*i]
            yi_n = fp.yi+OFFSET_ARRAY[2*i+1]
            if check_bounds_top:
                if (xi_n < 0 or yi_n < 0 or
                        xi_n >= raster_x_size or yi_n >= raster_y_size):
                    continue
            if flow_dir_managed_raster.get(
                    xi_n, yi_n) == REVERSE_FLOW_DIR[i]:
                flow_accum = flow_accumulation_managed_raster.get(xi_n, yi_n)
                if flow_accum == _NODATA:
                    flow_stack.push(FlowPixel(i, fp.xi, fp.yi, fp.flow_val))
                    if use_weights:
                        weight_val = weight_raster_path_raster.get(xi_n, yi_n)
                        if isclose(weight_val, weight_nodata):
                            weight_val = 0
                        flow_stack.push(FlowPixel(0, xi_n, yi_n, weight_val))
                    else:
                        flow_stack.push(FlowPixel(0, xi_n, yi_n, 1))
                    all_checked = 0  # indicate failure
                    break
                else:
                    fp.flow_val += flow_accum
        if all_checked:
            flow_accumulation_managed_raster.set(fp.xi, fp.yi, fp.flow_val)

    shutil.rmtree(temp_dir_path)


def downstream_flow_length_d8(
        flow_dir_raster_path_band, flow_accum_raster_path_band,
        double flow_threshold, target_flow_length_raster_path,
        weight_raster_path_band=None, temp_dir_path=None):
    """Calculates downstream flow distance to the points in the flow
        accumulation raster that are >= `flow_threshold`.

    Parameters:
        flow_dir_raster_path_band (tuple): a path, band tuple to a flow
            direction raster calculated by `routing.fill_pits`.

            indicating the D8 flow direction raster with direction convention:
             321
             4 0
             567

        flow_accum_raster_path_band (tuple): a path/band tuple to a raster
            of same dimensions as the flow_dir_raster indicating values where
            flow flow length should terminate/start.
        flow_threshold (float): Flow accumulation values in
            `flow_accum_raster_` that are >= `flow_threshold` are classified
            as streams for the context of determining the distance to streams
            in this function. i.e. any flow accumulation value >=
            `flow_threshold` will have a distance of 0 to this pixel.
        target_flow_length_raster_path (string): path to output raster for
            flow length to the drains.
        weight_raster_path_band (tuple): if not None, path to a raster/band
            tuple where the raster  is of the same dimensions as
            `flow_dir_raster_path_band` that is to be used in place of
            accumulating downstream distance. Without this raster the value
            of each pixel is "1" (or sqrt(2) for diagonal lengths). If this
            raster is not None, the pixel values are used in lieu of "1". If
            the weight is nodata in an otherwise defined flow path, a value
            of 0 is used for the weight.
        temp_dir_path (string): if not None, a path to a directory where
            temporary files can be constructed. Otherwise uses system tempdir.

    Returns:
        None.
    """
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] buffer_array
    cdef int raster_x_size, raster_y_size
    cdef int xi_n, yi_n, i
    cdef int xi, yi, win_ysize, win_xsize
    cdef int xoff, yoff
    cdef int flow_direction_nodata, n_dir, flow_dir
    cdef int use_weights
    cdef double flow_length
    cdef double weight_val
    cdef double weight_nodata = 0  # fill in with something for compiler
    cdef stack[FlowPixel] flow_stack
    cdef FlowPixel fp
    cdef ManagedRaster flow_accum_managed_raster
    cdef ManagedRaster flow_dir_managed_raster
    cdef ManagedRaster weight_raster_path_raster = None

    logger = logging.getLogger(
        'pygeoprocessing.routing.downstream_flow_length')
    logger.addHandler(logging.NullHandler())  # silence logging by default

    # flow direction scheme is
    # 321
    # 4 0
    # 567
    # each flow direction is encoded as 1 << n, n in [0, 7]

    # use this to have offsets to visit neighbor pixels, pick 2 at a time to
    # add to a (xi, yi) tuple
    cdef int* OFFSET_ARRAY = [
        1, 0,  # 0
        1, -1,  # 1
        0, -1,  # 2
        -1, -1,  # 3
        -1, 0,  # 4
        -1, 1,  # 5
        0, 1,  # 6
        1, 1  # 7
        ]

    # this is used to set flow direction on a neighbor by indicating which
    # neighbor it flows to
    cdef int* REVERSE_FLOW_DIR = [
        4,  # 0
        5,  # 1
        6,  # 2
        7,  # 3
        0,  # 4
        1,  # 5
        2,  # 6
        3  # 7
    ]

    # make an interesting temporary directory that has the time/date and
    # 'flow_accumulation' on it so we can figure out what's going on if we
    # ever run across it in a temp dir.
    temp_dir_path = tempfile.mkdtemp(
        dir=temp_dir_path, prefix='downstream_flow_length_',
        suffix=time.strftime('%Y-%m-%d_%H_%M_%S', time.gmtime()))

    pygeoprocessing.new_raster_from_base(
        flow_dir_raster_path_band[0],
        target_flow_length_raster_path, GDAL_INTERNAL_RASTER_TYPE,
        [_NODATA], fill_value_list=[_NODATA],
        gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS)))

    logger.info(
        'flow accumulation raster created at %s',
        target_flow_length_raster_path)

    # these are used to determine if a sample is within the raster
    flow_direction_raster_info = pygeoprocessing.get_raster_info(
        flow_dir_raster_path_band[0])
    base_flow_direction_nodata = flow_direction_raster_info['nodata'][
        flow_dir_raster_path_band[1]-1]

    if base_flow_direction_nodata is not None:
        flow_direction_nodata = base_flow_direction_nodata
    else:
        # pick some very impossible value given routing conventions
        flow_direction_nodata = 99

    raster_x_size, raster_y_size = flow_direction_raster_info['raster_size']
    # used to set flow directions
    flow_dir_managed_raster = ManagedRaster(
        flow_dir_raster_path_band[0], MANAGED_RASTER_N_BLOCKS, 0)

    # the flow accumulation result
    flow_length_managed_raster = ManagedRaster(
        target_flow_length_raster_path, MANAGED_RASTER_N_BLOCKS, 1)

    flow_accum_managed_raster = ManagedRaster(
        flow_accum_raster_path_band[0], MANAGED_RASTER_N_BLOCKS, 0)

    use_weights = 0
    if weight_raster_path_band is not None:
        weight_raster_path_raster = ManagedRaster(
            weight_raster_path_band[0], MANAGED_RASTER_N_BLOCKS, 0)
        base_weight_nodata = pygeoprocessing.get_raster_info(
            weight_raster_path_band[0])['nodata'][
                weight_raster_path_band[1]-1]
        if base_weight_nodata is not None:
            weight_nodata = base_weight_nodata
        else:
            weight_nodata = IMPROBABLE_FLOAT_NOATA
        use_weights = 1

    flow_direction_raster = gdal.Open(flow_dir_raster_path_band[0])
    flow_direction_band = flow_direction_raster.GetRasterBand(
        flow_dir_raster_path_band[1])

    logger.info('finding drains')
    start_drain_time = ctime(NULL)
    for offset_dict in pygeoprocessing.iterblocks(
            flow_dir_raster_path_band[0], offset_only=True,
            largest_block=0):
        # statically type these for later
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        # make a buffer big enough to capture block and boundaries around it
        buffer_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2),
            dtype=numpy.uint8)
        buffer_array[:] = flow_direction_nodata

        # default numpy array boundaries
        buffer_off = {
            'xa': 1,
            'xb': -1,
            'ya': 1,
            'yb': -1
        }
        # check if we can widen the border to include real data from the
        # raster
        for a_buffer_id, b_buffer_id, off_id, win_size_id, raster_size in [
                ('xa', 'xb', 'xoff', 'win_xsize', raster_x_size),
                ('ya', 'yb', 'yoff', 'win_ysize', raster_y_size)]:
            if offset_dict[off_id] > 0:
                # in thise case we have valid data to the left (or up)
                # grow the window and buffer slice in that direction
                buffer_off[a_buffer_id] = None
                offset_dict[off_id] -= 1
                offset_dict[win_size_id] += 1

            if offset_dict[off_id] + offset_dict[win_size_id] < raster_size:
                # here we have valid data to the right (or bottom)
                # grow the right buffer and add 1 to window
                buffer_off[b_buffer_id] = None
                offset_dict[win_size_id] += 1

        # read in the valid memory block
        buffer_array[
            buffer_off['ya']:buffer_off['yb'],
            buffer_off['xa']:buffer_off['xb']] = (
                flow_direction_band.ReadAsArray(
                    **offset_dict).astype(numpy.int8))

        # irrespective of how we sampled the DEM only look at the block in
        # the middle for valid
        for yi in xrange(1, win_ysize+1):
            for xi in xrange(1, win_xsize+1):
                flow_dir = (buffer_array[yi, xi])
                if isclose(flow_dir, flow_direction_nodata):
                    continue
                n_dir = buffer_array[
                    yi+OFFSET_ARRAY[2*flow_dir+1],
                    xi+OFFSET_ARRAY[2*flow_dir]]
                if (isclose(n_dir, flow_direction_nodata) or
                        flow_accum_managed_raster.get(
                            xi-1+xoff, yi-1+yoff) >= flow_threshold):
                    # it hit a stream threshold, so it's a seed
                    # initial distance on stream is 0
                    flow_stack.push(
                        FlowPixel(0, xi-1+xoff, yi-1+yoff, 0))
                    flow_length_managed_raster.set(
                        xi-1+xoff, yi-1+yoff, 0)

    logger.info("drains detected in %fs", ctime(NULL)-start_drain_time)
    while not flow_stack.empty():
        fp = flow_stack.top()
        flow_stack.pop()

        if (fp.xi == 0 or fp.xi == (raster_x_size-1) or
                fp.yi == 0 or fp.yi == (raster_y_size-1)):
            check_bounds_top = 1
        else:
            check_bounds_top = 0

        all_checked = 1
        for i in xrange(fp.n_i, 8):
            # neighbor x,y indexes
            xi_n = fp.xi+OFFSET_ARRAY[2*i]
            yi_n = fp.yi+OFFSET_ARRAY[2*i+1]
            if check_bounds_top:
                if (xi_n < 0 or yi_n < 0 or
                        xi_n >= raster_x_size or yi_n >= raster_y_size):
                    continue
            if flow_dir_managed_raster.get(
                    xi_n, yi_n) == REVERSE_FLOW_DIR[i]:
                flow_length = flow_length_managed_raster.get(xi_n, yi_n)
                if flow_length == _NODATA:
                    flow_stack.push(FlowPixel(i, fp.xi, fp.yi, fp.flow_val))
                    if use_weights:
                        weight_val = weight_raster_path_raster.get(xi_n, yi_n)
                        if isclose(weight_val, weight_nodata):
                            weight_val = 0
                    else:
                        # add either straight line or diagonal direction
                        weight_val = 1.0 if i % 2 == 0 else 1.4142135
                    flow_stack.push(
                        FlowPixel(0, xi_n, yi_n, fp.flow_val+weight_val))
                    all_checked = 0  # indicates neighbor not calculated
                    break
        if all_checked:
            flow_length_managed_raster.set(fp.xi, fp.yi, fp.flow_val)
    shutil.rmtree(temp_dir_path)
