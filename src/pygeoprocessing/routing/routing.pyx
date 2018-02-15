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
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
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
from libcpp.vector cimport vector

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

# this is the class type that'll get stored in the priority queue
cdef struct Pixel:
    double value # pixel value
    int xi  # pixel x coordinate in the raster
    int yi # pixel y coordinate in the raster

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

    def __dealloc__(self):
        """Deallocate ManagedRaster.

        This operation manually frees memory from the LRUCache and writes any
        dirty memory blocks back to the raster if `self.write_mode` is True.
        """
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
                free(deref(it).second)
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
            free(double_buffer)
            inc(it)

        raster_band.FlushCache()
        raster_band = None
        raster = None

    cdef void set(self, int xi, int yi, double value):
        cdef int block_xi = xi >> self.block_bits
        cdef int block_yi = yi >> self.block_bits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self.load_block(block_index)
        cdef int xoff = block_xi << self.block_bits
        cdef int yoff = block_yi << self.block_bits
        self.lru_cache.get(
            block_index)[(yi-yoff)*self.block_xsize+xi-xoff] = value
        if self.write_mode:
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr == self.dirty_blocks.end():
                self.dirty_blocks.insert(block_index)

    cdef double get(self, int xi, int yi):
        cdef int block_xi = xi >> self.block_bits
        cdef int block_yi = yi >> self.block_bits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self.load_block(block_index)
        cdef int xoff = block_xi << self.block_bits
        cdef int yoff = block_yi << self.block_bits
        return self.lru_cache.get(
            block_index)[(yi-yoff)*self.block_xsize+xi-xoff]

    cdef void load_block(self, int block_index) except *:
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
        double_buffer = <double*>malloc(
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
            free(double_buffer)
            removed_value_list.pop_front()

        if self.write_mode:
            raster_band = None
            raster = None

ctypedef pair[int, int] CoordinatePair

# This functor is used to determine order in the priority queue by comparing
# value only.
cdef cppclass GreaterPixel nogil:
    bint get "operator()"(Pixel& lhs, Pixel& rhs):
        return lhs.value > rhs.value

ctypedef double[:, :] FloatMemView


def fill_pits(
        dem_raster_path_band, target_filled_dem_raster_path,
        target_flow_direction_path, temp_dir_path=None):
    """Fill hydrological pits in input DEM.

    Implementation of the algorithm described in "An efficient variant of the
    Priority-Flood algorithm for filling depressions in raster digital
    elevation models. Zhou, Sun, and Fu."

    Parameters:
        dem_raster_path_band (tuple): a path, band number tuple indicating the
            DEM to be filled.
        target_filled_dem_raster_path (string): path to a single band raster
            that will be created as a copy of `dem_raster_path_band` with any
            hydrological depressions filled.
        target_flow_direction_path (string): path to a int8 single band raster
            that will be created as a flow direction output
        temp_dir_path (string): if not None, indicates where algorithm can
            construct intermediate files for bookkeeping during algorithm
            processing.

    Returns:
        None.
    """
    cdef numpy.ndarray[numpy.float64_t, ndim=2] buffer_array
    cdef numpy.float64_t center_value, s_center_value
    cdef int i, j, yi, xi, xi_q, yi_q, xi_s, yi_s, xi_n, yi_n, xj_n, yj_n
    cdef int raster_x_size, raster_y_size
    cdef int win_ysize, win_xsize
    cdef int xoff, yoff
    cdef long pixels_to_process = 0
    cdef long last_pixels_to_process
    cdef numpy.float64_t dem_nodata, n_value
    cdef priority_queue[Pixel, vector[Pixel], GreaterPixel] p_queue
    cdef Pixel p
    cdef queue[CoordinatePair] q, sq
    cdef ManagedRaster flag_managed_raster, dem_filled_managed_raster
    cdef int check_bounds

    logger = logging.getLogger('pygeoprocessing.routing.fill_pits')
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
    # 'fill_pits' on it so we can figure out what's going on if we ever run
    # across it again.
    temp_dir_path = tempfile.mkdtemp(
        dir=temp_dir_path, prefix='fill_pits_', suffix=time.strftime(
            '%Y-%m-%d_%H_%M_%S', time.gmtime()))
    flag_raster_path = os.path.join(temp_dir_path, 'flag_raster.tif')

    # make a byte flag raster, no need for a nodata value but initialize to 0
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], flag_raster_path, gdal.GDT_Byte,
        [None], fill_value_list=[0], gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS)))

    logger.info('flag raster created at %s', flag_raster_path)

    # this will always make the DEM a 64 bit float, it's the 'safest' choice
    # since we need to statically type a DEM. Is it possible to template
    # this algorithm? maybe?
    logger.info(
        'copying %s dem to %s', dem_raster_path_band,
        target_filled_dem_raster_path)
    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_path_band[0])
    base_nodata = dem_raster_info['nodata'][dem_raster_path_band[1]-1]
    if base_nodata is not None:
        # cast to a float64 since that's our operating array type
        dem_nodata = numpy.float64(base_nodata)
    else:
        # pick some very improbable value since it's hard to deal with NaNs
        dem_nodata = IMPROBABLE_FLOAT_NOATA
    gtiff_driver = gdal.GetDriverByName('GTiff')
    dem_raster = gdal.OpenEx(dem_raster_path_band[0], gdal.OF_RASTER)
    gtiff_driver.CreateCopy(
        target_filled_dem_raster_path, dem_raster,
        options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS)))

    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], target_flow_direction_path, gdal.GDT_Byte,
        [255], fill_value_list=[255], gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1 << BLOCK_BITS)))

    # these are used to determine if a sample is within the raster
    raster_x_size, raster_y_size = dem_raster_info['raster_size']
    pixels_to_process = raster_x_size * raster_y_size

    # used to set flow directions
    flow_dir_managed_raster = ManagedRaster(
        target_flow_direction_path, MANAGED_RASTER_N_BLOCKS, 1)

    # used to set and read flags
    flag_managed_raster = ManagedRaster(
        flag_raster_path, MANAGED_RASTER_N_BLOCKS, 1)
    # used to set filled DEM and read current DEM.
    dem_filled_managed_raster = ManagedRaster(
        target_filled_dem_raster_path, MANAGED_RASTER_N_BLOCKS, 1)

    dem_raster = gdal.Open(dem_raster_path_band[0])
    dem_band = dem_raster.GetRasterBand(dem_raster_path_band[1])

    logger.info('detecting building edges')
    start_edge_time = ctime(NULL)
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
            buffer_off['xa']:buffer_off['xb']] = dem_band.ReadAsArray(
                **offset_dict).astype(numpy.float64)

        # irrespective of how we sampled the DEM only look at the block in
        # the middle for valid
        for yi in xrange(1, win_ysize+1):
            for xi in xrange(1, win_xsize+1):
                center_value = buffer_array[yi, xi]
                if isclose(center_value, dem_nodata):
                    # if nodata, mark done
                    flag_managed_raster.set(xi-1+xoff, yi-1+yoff, 1)
                    pixels_to_process -= 1
                    continue

                # this uses the offset array to visit the neighbors rather
                # than 8 identical if statements.
                for i in xrange(8):
                    if isclose(buffer_array[
                            yi+OFFSET_ARRAY[2*i+1],
                            xi+OFFSET_ARRAY[2*i]], dem_nodata):
                        p_queue.push(
                            Pixel(
                                center_value, xi-1+xoff, yi-1+yoff))
                        flag_managed_raster.set(xi-1+xoff, yi-1+yoff, 1)
                        # set it to flow off the edge, this might get changed
                        # later if it's caught in a flow
                        flow_dir_managed_raster.set(xi-1+xoff, yi-1+yoff, i)
                        pixels_to_process -= 1
                        break
    logger.info("edges detected in %fs", ctime(NULL)-start_edge_time)
    start_pit_time = ctime(NULL)
    logger.info("filling pits, pixels to process: %d", pixels_to_process)
    last_pixels_to_process = pixels_to_process
    while not p_queue.empty():
        if ctime(NULL) - start_pit_time > 5.0:
            logger.info(
                "pixels to process: %d; pixels processed since last log %d",
                pixels_to_process, last_pixels_to_process - pixels_to_process)
            last_pixels_to_process = pixels_to_process
            start_pit_time = ctime(NULL)
        p = p_queue.top()
        xi = p.xi
        yi = p.yi
        center_value = p.value
        # loop invariant, center_value != nodata
        p_queue.pop()

        if (xi == 0 or xi == (raster_x_size-1) or
                yi == 0 or yi == (raster_y_size-1)):
            check_bounds_top = 1
        else:
            check_bounds_top = 0

        for i in xrange(8):
            # neighbor x,y indexes
            xi_n = xi+OFFSET_ARRAY[2*i]
            yi_n = yi+OFFSET_ARRAY[2*i+1]
            if check_bounds_top:
                if (xi_n < 0 or yi_n < 0 or
                        xi_n >= raster_x_size or yi_n >= raster_y_size):
                    continue

            if flag_managed_raster.get(xi_n, yi_n):
                # if flag is set, cell is processed, so skip
                continue
            # we're about to process, so set its flag
            flag_managed_raster.set(xi_n, yi_n, 1)
            n_value = dem_filled_managed_raster.get(xi_n, yi_n)
            # loop invariant, n_value != nodata because flag is not set
            flow_dir_managed_raster.set(xi_n, yi_n, REVERSE_FLOW_DIR[i])
            pixels_to_process -= 1
            if n_value <= center_value:
                # neighbor is less than current cell so we grow the region
                dem_filled_managed_raster.set(xi_n, yi_n, center_value)
                q.push(CoordinatePair(xi_n, yi_n))
                while not q.empty():
                    xi_q = q.front().first
                    yi_q = q.front().second
                    if (xi_q == 0 or yi_q == 0 or xi_q == (raster_x_size-1) or
                            yi_q == (raster_y_size-1)):
                        check_bounds = 1
                    else:
                        check_bounds = 0
                    q.pop()
                    for i in xrange(8):
                        # neighbor x,y indexes
                        xi_n = xi_q+OFFSET_ARRAY[2*i]
                        yi_n = yi_q+OFFSET_ARRAY[2*i+1]
                        if check_bounds:
                            if (xi_n < 0 or yi_n < 0 or
                                    xi_n >= raster_x_size or
                                    yi_n >= raster_y_size):
                                continue
                        # if flag is set, then skip
                        if flag_managed_raster.get(xi_n, yi_n):
                            continue
                        # it's about to be filled or pushed to slope queue
                        # so okay to set flag here

                        # li: n_value is not nodata because flag was not set
                        n_value = dem_filled_managed_raster.get(xi_n, yi_n)
                        flow_dir_managed_raster.set(
                            xi_n, yi_n, REVERSE_FLOW_DIR[i])
                        pixels_to_process -= 1

                        # check for <= center value
                        if n_value <= center_value:
                            # filled as neighbor
                            flag_managed_raster.set(xi_n, yi_n, 1)
                            q.push(CoordinatePair(xi_n, yi_n))
                            # raise neighbor dem to center value
                            if n_value < center_value:
                                dem_filled_managed_raster.set(
                                    xi_n, yi_n, center_value)
                        else:
                            # not flat so must be a slope pixel,
                            # push to slope queue
                            # filled as upslope
                            flag_managed_raster.set(xi_n, yi_n, 1)
                            sq.push(CoordinatePair(xi_n, yi_n))
            else:
                # otherwise it's a slope pixel, push to slope queue
                sq.push(CoordinatePair(xi_n, yi_n))

            # grow up the slopes
            while not sq.empty():
                isProcessed = 0
                xi_s = sq.front().first
                yi_s = sq.front().second
                if (xi_s <= 1 or xi_s >= raster_x_size-2 or
                        yi_s <= 1 or yi_s >= raster_y_size-2):
                    check_bounds = 1
                else:
                    check_bounds = 0
                sq.pop()
                s_center_value = dem_filled_managed_raster.get(xi_s, yi_s)
                for i in xrange(8):
                    xi_n = xi_s+OFFSET_ARRAY[2*i]
                    yi_n = yi_s+OFFSET_ARRAY[2*i+1]
                    if check_bounds:
                        if (xi_n < 0 or yi_n < 0 or
                                xi_n >= raster_x_size or
                                yi_n >= raster_y_size):
                            continue
                    if flag_managed_raster.get(xi_n, yi_n):
                        continue
                    n_value = dem_filled_managed_raster.get(xi_n, yi_n)
                    # loop invariant: n_value not nodata because flag not set
                    # if neighbor is higher than center, grow slope
                    if n_value > s_center_value:
                        flow_dir_managed_raster.set(
                            xi_n, yi_n, REVERSE_FLOW_DIR[i])
                        pixels_to_process -= 1
                        sq.push(CoordinatePair(xi_n, yi_n))
                        flag_managed_raster.set(xi_n, yi_n, 1)
                    elif not isProcessed:
                        isProcessed = 1
                        # nonRegionCell call from pseudocode
                        isBoundary = 1
                        for j in xrange(8):
                            # check neighbors of neighbor
                            xj_n = xi_n+OFFSET_ARRAY[2*j]
                            yj_n = yi_n+OFFSET_ARRAY[2*j+1]
                            if check_bounds:
                                if (xj_n < 0 or yj_n < 0 or
                                        xj_n >= raster_x_size or
                                        yj_n > raster_y_size):
                                    continue
                            j_value = dem_filled_managed_raster.get(
                                xj_n, yj_n)
                            # check for nodata
                            if isclose(j_value, dem_nodata):
                                continue
                            if ((j_value < n_value) and
                                    flag_managed_raster.get(xj_n, yj_n)):
                                # if flag(j) && DEM(j) < DEM(n) it's not a
                                # boundary because downhill neighbor has been
                                # processed
                                isBoundary = 0
                                break
                        if isBoundary:
                            p_queue.push(Pixel(s_center_value, xi_s, yi_s))
                        else:
                            # USE i_n in MFD for i_s
                            isProcessed = 0
    logger.info("pits filled in %fs", ctime(NULL)-start_pit_time)
    logger.info("pixels left to process: %d", pixels_to_process)
    # clean up flag managed raster before it is deleted
    del flag_managed_raster
    shutil.rmtree(temp_dir_path)


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
