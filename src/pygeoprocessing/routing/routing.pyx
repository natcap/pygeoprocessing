# distutils: language=c++
"""Pitfilling module."""
import errno
import os
import logging
import shutil
import tempfile
import time

import numpy
import scipy.signal
import pygeoprocessing
from osgeo import gdal

cimport numpy
cimport cython
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libcpp.vector cimport vector
from libcpp.list cimport list
from libcpp.queue cimport queue
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from libcpp.list cimport list
from libcpp.pair cimport pair

# This module expects rasters with a memory xy block size of 2**BLOCK_BITS
cdef int BLOCK_BITS = 8

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
    double value
    int xi
    int yi

cdef extern from "LRUCache.h" nogil:
    cdef cppclass LRUCache[KEY_T, VAL_T]:
        LRUCache(int)
        void put(KEY_T&, VAL_T&, list[pair[KEY_T,VAL_T]]&)
        list[pair[KEY_T,VAL_T]].iterator begin()
        list[pair[KEY_T,VAL_T]].iterator end()
        bint exist(KEY_T &)
        VAL_T get(KEY_T &)

ctypedef pair[int, double*] BlockBufferPair

# TODO: make managed raster band aware
cdef class ManagedRaster:
    cdef LRUCache[int, double*]* lru_cache
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
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        self.raster_x_size, self.raster_y_size = raster_info['raster_size']
        self.block_xsize, self.block_ysize = raster_info['block_size']
        self.block_bits = BLOCK_BITS
        self.block_xsize = 1<<self.block_bits
        self.block_ysize = 1<<self.block_bits
        """if (self.block_xsize != (1 << self.block_bits) or
                self.block_ysize != (1 << self.block_bits)):
            error_string = (
                "Expected block size that was %d bits wide, got %s" % (
                    self.block_bits, raster_info['block_size']))
            raise ValueError(error_string)
        """
        self.block_nx = (
            self.raster_x_size + (self.block_xsize) - 1) / self.block_xsize
        self.block_ny = (
            self.raster_y_size + (self.block_ysize) - 1) / self.block_ysize

        self.lru_cache = new LRUCache[int, double*](n_blocks)
        self.raster_path = raster_path
        self.write_mode = write_mode

    def __dealloc__(self):
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

        while it != end:
            # write the changed value back if desired
            double_buffer = deref(it).second

            if self.write_mode:
                block_index = deref(it).first
                block_xi = block_index % self.block_nx
                block_yi = block_index / self.block_nx

                # we need the offsets to subtract from global indexes for cached array
                xoff = block_xi * self.block_xsize
                yoff = block_yi * self.block_ysize

                win_xsize = self.block_xsize
                win_ysize = self.block_ysize

                # load a new block
                if xoff+win_xsize > self.raster_x_size:
                    win_xsize = win_xsize - (xoff+win_xsize - self.raster_x_size)
                if yoff+win_ysize > self.raster_y_size:
                    win_ysize = win_ysize - (yoff+win_ysize - self.raster_y_size)

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

    cdef void get_block(self, int xc, int yc, double* block, int border) except *:
        """Load a block from the raster into `block`.

        Parameters:
            xc (int): x coordinate at center of block
            yy (int): y coordinate at center of block
            block (double*): pre-allocated block of size `(1+2*border)**2`
                after the call this array will contain either random pixels
                if the block lies outside of the raster, or the pixel values
                that overlap the block.
            border (int): number of pixels around `xc`, `yc` to load into
                `block`.

        Returns:
            None
        """
        cdef int block_xi
        cdef int block_yi
        cdef int block_index
        cdef int xi, yi
        cdef int xoff
        cdef int yoff

        for xi in xrange(xc-border, xc+border+1):
            if xi < 0 or xi >= self.raster_x_size:
                continue
            for yi in xrange(yc-border, yc+border+1):
                if yi < 0 or yi >= self.raster_y_size:
                    continue

                block_xi = xi >> self.block_bits
                block_yi = yi >> self.block_bits
                block_index = block_yi * self.block_nx + block_xi
                # this is the flat index for the block
                if not self.lru_cache.exist(block_index):
                    self.load_block(block_index)
                xoff = block_xi << self.block_bits
                yoff = block_yi << self.block_bits
                block[(yi-(yc-border))*(1+border*2)+xi-(xc-border)] = (
                    self.lru_cache.get(
                        block_index)[(yi-yoff)*self.block_xsize+xi-xoff])

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
                block_xi = block_index % self.block_nx
                block_yi = block_index / self.block_nx

                # we need the offsets to subtract from global indexes for cached array
                xoff = block_xi * self.block_xsize
                yoff = block_yi * self.block_ysize

                win_xsize = self.block_xsize
                win_ysize = self.block_ysize

                # load a new block
                if xoff+win_xsize > self.raster_x_size:
                    win_xsize = win_xsize - (xoff+win_xsize - self.raster_x_size)
                if yoff+win_ysize > self.raster_y_size:
                    win_ysize = win_ysize - (yoff+win_ysize - self.raster_y_size)

                for xi_copy in xrange(win_xsize):
                    for yi_copy in xrange(win_ysize):
                        block_array[yi_copy, xi_copy] = (
                            double_buffer[yi_copy * self.block_xsize + xi_copy])
                raster_band.WriteArray(
                    block_array[0:win_ysize, 0:win_xsize],
                    xoff=xoff, yoff=yoff)
            free(double_buffer)
            removed_value_list.pop_front()

        if self.write_mode:
            raster_band.FlushCache()
            raster_band = None
            raster = None

ctypedef pair[int, int] CoordinatePair

# This functor is used to determine order in the priority queue by comparing
# value only.
cdef cppclass GreaterPixel nogil:
    bint get "operator()"(Pixel& lhs, Pixel& rhs):
        return lhs.value > rhs.value

ctypedef double[:, :] FloatMemView


#@cython.boundscheck(False)
def fill_pits(
        dem_raster_band_path, target_filled_dem_raster_path,
        target_flow_direction_path, temp_dir_path=None):
    """Fill hydrological pits in input DEM.

    Implementation of the algorithm described in "An efficient variant of the
    Priority-Flood alogirhtm for filling depressions in raster digital
    elevation models. Zhou, Sun, and Fu.

    Parameters:
        dem_raster_band_path (tuple): a path, band number tuple indicating the
            DEM to be filled.
        target_filled_dem_raster_path (string): path to a single band raster
            that will be created as a copy of `dem_raster_band_path` with any
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
    cdef numpy.float64_t dem_nodata
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
        1 << 4, # 0
        1 << 5, # 1
        1 << 6, # 2
        1 << 7, # 3
        1 << 0, # 4
        1 << 1, # 5
        1 << 2, # 6
        1 << 3 # 7
    ]

    # make an interesting temporary directory that has the time/date and
    # 'fill_pits' on it so we can figure out what's going on if we ever run
    # across it again.
    temp_dir_path = tempfile.mkdtemp(
        dir=temp_dir_path, prefix='fill_pits', suffix=time.strftime(
        '%Y-%m-%d_%H_%M_%S', time.gmtime()))
    flag_raster_path = os.path.join(temp_dir_path, 'flag_raster.tif')

    # make a byte flag raster, no need for a nodata value but initialize to 0
    pygeoprocessing.new_raster_from_base(
        dem_raster_band_path[0], flag_raster_path, gdal.GDT_Byte,
        [None], fill_value_list=[0], gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=IF_SAFER', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1<<BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1<<BLOCK_BITS)))

    logger.info('flag raster created at %s', flag_raster_path)

    # this will always make the DEM a 64 bit float, it's the 'safest' choice
    # since we need to statically type a DEM. Is it possible to template
    # this algorithm? maybe?
    logger.info(
        'copying %s dem to %s', dem_raster_band_path,
        target_filled_dem_raster_path)
    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_band_path[0])
    dem_nodata = numpy.float64(
        dem_raster_info['nodata'][dem_raster_band_path[1]-1])
    pygeoprocessing.new_raster_from_base(
        dem_raster_band_path[0], target_filled_dem_raster_path,
        gdal.GDT_Float64, [dem_nodata], fill_value_list=[dem_nodata],
        gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=IF_SAFER', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1<<BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1<<BLOCK_BITS)))
    target_filled_dem_raster = gdal.Open(
        target_filled_dem_raster_path, gdal.GA_Update)
    target_filled_dem_band = target_filled_dem_raster.GetRasterBand(1)
    for block_offset, block_data in pygeoprocessing.iterblocks(
            dem_raster_band_path[0], astype=[numpy.float64]):
        target_filled_dem_band.WriteArray(
            block_data, xoff=block_offset['xoff'], yoff=block_offset['yoff'])
    target_filled_dem_band = None
    target_filled_dem_raster = None

    pygeoprocessing.new_raster_from_base(
        dem_raster_band_path[0], target_flow_direction_path, gdal.GDT_Byte,
        [255], fill_value_list=[255], gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=IF_SAFER', 'COMPRESS=LZW',
            'BLOCKXSIZE=%d' % (1<<BLOCK_BITS),
            'BLOCKYSIZE=%d' % (1<<BLOCK_BITS)))

    # these are used to determine if a sample is within the raster
    raster_x_size, raster_y_size = dem_raster_info['raster_size']

    # used to set flow directions
    flow_direction_managed_raster = ManagedRaster(
        target_flow_direction_path, 2**10, 1)

    # used to set and read flags
    flag_managed_raster = ManagedRaster(flag_raster_path, 2**10, 1)
    # used to set filled DEM and read current DEM.
    dem_filled_managed_raster = ManagedRaster(
        target_filled_dem_raster_path, 2**10, 1)

    dem_raster = gdal.Open(dem_raster_band_path[0])
    dem_band = dem_raster.GetRasterBand(dem_raster_band_path[1])

    logger.info('detecting building edges')
    start_edge_time = time.time()
    for offset_dict in pygeoprocessing.iterblocks(
            dem_raster_band_path[0], offset_only=True, largest_block=0):

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
                        break
    logger.info("edges detected in %fs", time.time()-start_edge_time)
    start_pit_time = time.time()
    logger.info('filling pits')
    while not p_queue.empty():
        p = p_queue.top()
        xi = p.xi
        yi = p.yi
        center_value = p.value
        # loop invariant, center_value != nodata because it wouldn't have been pushed
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
            if n_value <= center_value:
                # neighbor is less than current cell so we grow the region
                dem_filled_managed_raster.set(xi_n, yi_n, center_value)
                flow_direction_managed_raster.set(
                    xi_n, yi_n, REVERSE_FLOW_DIR[i])
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
                        flow_direction_managed_raster.set(
                            xi_n, yi_n, REVERSE_FLOW_DIR[i])

                        # check for <= center value
                        if n_value <= center_value:
                            flag_managed_raster.set(xi_n, yi_n, 1) # filled as neighbor
                            q.push(CoordinatePair(xi_n, yi_n))
                            # raise neighbor dem to center value
                            if n_value < center_value:
                                dem_filled_managed_raster.set(
                                    xi_n, yi_n, center_value)
                        else:
                            # not flat so must be a slope pixel,
                            # push to slope queue
                            flag_managed_raster.set(xi_n, yi_n, 1) # filled as upslope
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
                        flow_direction_managed_raster.set(
                            xi_n, yi_n, REVERSE_FLOW_DIR[i])
                        sq.push(CoordinatePair(xi_n, yi_n))
                        flag_managed_raster.set(xi_n, yi_n, 1)
                    elif not isProcessed:
                        isProcessed = 1
                        # nonRegionCell call
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
    logger.info("pits filled in %fs", time.time()-start_pit_time)

    del flag_managed_raster

    shutil.rmtree(temp_dir_path)

def flow_accmulation(
        flow_direction_raster_band_path,
        target_flow_accumulation_raster_path):
    """Calculate flow accumulation given flow direction.

    Parameters:
        flow_direction_raster_band_path (tuple): a path, band number tuple
            indicating the D8 flow direction raster with direction convention:

            # 321
            # 4 0
            # 567
            # each flow direction is encoded as 1 << n, n in [0, 7]

        target_flow_accmulation_raster_path (string): path to single band
            raster to be created. Each pixel value will indicate the number
            of upstream pixels that feed it including the current pixel.

    Returns:
        None.
    """
    pass
