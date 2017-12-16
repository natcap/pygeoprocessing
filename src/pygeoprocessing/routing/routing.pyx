# distutils: language=c++
"""Pitfilling module."""
import tempfile
import errno
import os

import numpy
import scipy.signal
import pygeoprocessing
import gdal

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
cdef cppclass Pixel nogil:
    void Pixel(double value, int xi, int yi):
        this.xi = xi
        this.yi = yi
        this.value = value
    int xi
    int yi
    double value

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
    cdef char* raster_path

    def __cinit__(self, char* raster_path, n_blocks, write_mode):
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        self.raster_x_size, self.raster_y_size = raster_info['raster_size']
        self.block_xsize, self.block_ysize = raster_info['block_size']

        self.block_nx = (self.raster_x_size + self.block_xsize - 1) / self.block_xsize
        self.block_ny = (self.raster_y_size + self.block_ysize - 1) / self.block_ysize

        self.lru_cache = new LRUCache[int, double*](n_blocks)
        self.raster_path = raster_path
        self.write_mode = write_mode

    def __dealloc__(self):
        print 'IMPLEMENT SAVING DIRTY CACHE'

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

        print 'saving', self.raster_path

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


    cdef void set(self, int xi, int yi, double value) except *:
        cdef int block_xi = xi / self.block_xsize
        cdef int block_yi = yi / self.block_ysize
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self.load_block(block_index)
        cdef int xoff = block_xi * self.block_xsize
        cdef int yoff = block_yi * self.block_ysize
        self.lru_cache.get(
            block_index)[(yi-yoff)*self.block_xsize+xi-xoff] = value

    cdef double get(self, int xi, int yi) except *:
        cdef int block_xi = xi / self.block_xsize
        cdef int block_yi = yi / self.block_ysize
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self.load_block(block_index)
        cdef int xoff = block_xi * self.block_xsize
        cdef int yoff = block_yi * self.block_ysize
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

# made this type def because cython sometimes has trouble recognizing the
# "*" pointer type notation without some extra syntax
ctypedef Pixel* PixelPtr

# This functor is used to determine order in the priority queue by comparing
# value only.
cdef cppclass GreaterPixel nogil:
    bint get "operator()"(PixelPtr& lhs, PixelPtr& rhs):
        return lhs[0].value > rhs[0].value


def priority_queue_tracer():
    cdef priority_queue[PixelPtr, vector[PixelPtr], GreaterPixel] q
    cdef PixelPtr x = new Pixel(1, 1, 1)
    q.push(x)
    q.push(new Pixel(1, 1, 19))
    q.push(new Pixel(2, 1, 11))
    q.push(new Pixel(-1, 15, 1))
    q.push(new Pixel(5, 1, 11))
    q.push(new Pixel(0, 12, 1))

    cdef PixelPtr p
    while not q.empty():
        p = q.top()
        q.pop()
        print p[0].value
        del p

ctypedef double[:, :] FloatMemView


#@cython.boundscheck(False)
def fill_pits(dem_raster_band_path, workspace_dir):
    """Identify drainage DEM pixels that abut nodata or raster edge."""
    cdef numpy.ndarray[numpy.float64_t, ndim=2] buffer_array
    cdef numpy.float64_t center_value, s_center_value
    cdef int i, j, yi, xi, xi_q, yi_q, xi_s, yi_s, xi_n, yi_n, xj_n, yj_n
    cdef int watershed_id
    cdef int raster_x_size, raster_y_size
    cdef int win_ysize, win_xsize
    cdef int xoff, yoff
    cdef numpy.float64_t dem_nodata
    cdef numpy.int32_t FLAG_NODATA = -1
    cdef priority_queue[PixelPtr, vector[PixelPtr], GreaterPixel] p_queue
    cdef PixelPtr p
    cdef queue[CoordinatePair] q, sq
    cdef ManagedRaster flag_managed_raster
    cdef ManagedRaster dem_managed_raster

    # use this to have offsets to visit neighbor pixels, pick 2 at a time to
    # add to a (xi, yi) tuple
    cdef int* OFFSET_ARRAY = [
        -1, -1,
        -1, 0,
        -1, 1,
        0, -1,
        0, 1,
        1, -1,
        1, 0,
        1, 1]
    try:
        os.mkdir(workspace_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    _, flag_raster_path = tempfile.mkstemp(
        suffix='.tif', prefix='flag', dir=workspace_dir)

    pygeoprocessing.new_raster_from_base(
        dem_raster_band_path[0], flag_raster_path, gdal.GDT_Int32,
        [FLAG_NODATA], fill_value_list=[0],
        gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=IF_SAFER', 'COMPRESS=NONE'))

    _, dem_filled_raster_path = tempfile.mkstemp(
        suffix='.tif', prefix='dem', dir=workspace_dir)

    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_band_path[0])
    dem_nodata = numpy.array(dem_raster_info['nodata']).astype(
        numpy.float64)[dem_raster_band_path[1]-1]
    raster_x_size, raster_y_size = dem_raster_info['raster_size']
    print dem_nodata, raster_x_size, raster_y_size

    # this raster is to fill in as we go so we can also see what the algorithm is touching
    pygeoprocessing.new_raster_from_base(
        dem_raster_band_path[0], dem_filled_raster_path, gdal.GDT_Float64,
        [dem_nodata], fill_value_list=[dem_nodata],
        gtiff_creation_options=(
            'TILED=YES', 'BIGTIFF=IF_SAFER', 'COMPRESS=NONE'))

    flag_managed_raster = ManagedRaster(flag_raster_path, 2**11, 1)

    dem_managed_raster = ManagedRaster(dem_raster_band_path[0], 2**11, 0)
    dem_filled_managed_raster = ManagedRaster(
        dem_filled_raster_path, 2**11, 1)

    dem_raster = gdal.Open(dem_raster_band_path[0])
    dem_band = dem_raster.GetRasterBand(dem_raster_band_path[1])

    print 'building edges'
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
                            yi+OFFSET_ARRAY[2*i],
                            xi+OFFSET_ARRAY[2*i+1]], dem_nodata):
                        p_queue.push(
                            new Pixel(
                                center_value, xi-1+xoff, yi-1+yoff))
                        flag_managed_raster.set(xi-1+xoff, yi-1+yoff, 2)
                        break
    print 'printing result'
    while not p_queue.empty():
        p = p_queue.top()
        p_queue.pop()
        xi = p.xi
        yi = p.yi
        center_value = p.value
        # loop invariant, center_value != nodata because it wouldn't have been pushed
        del p

        for i in xrange(8):
            # neighbor x,y indexes
            xi_n = xi+OFFSET_ARRAY[2*i+1]
            yi_n = yi+OFFSET_ARRAY[2*i]
            if (xi_n < 0 or yi_n < 0 or
                    xi_n >= raster_x_size or yi_n >= raster_y_size):
                continue

            if flag_managed_raster.get(xi_n, yi_n):
                # if flag is set, cell is processed, so skip
                continue
            # we're about to process, so set its flag
            flag_managed_raster.set(xi_n, yi_n, 3) # 1 means start of upstream
            n_value = dem_managed_raster.get(xi_n, yi_n)
            # loop invariant, n_value != nodata because flag is not set
            if n_value <= center_value:
                # neighbor is less than current cell so we grow the region

                q.push(CoordinatePair(xi_n, yi_n))
                while not q.empty():
                    xi_q = q.front().first
                    yi_q = q.front().second
                    q.pop()
                    for i in xrange(8):
                        # neighbor x,y indexes
                        xi_n = xi_q+OFFSET_ARRAY[2*i+1]
                        yi_n = yi_q+OFFSET_ARRAY[2*i]
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
                        n_value = dem_managed_raster.get(xi_n, yi_n)

                        # check for <= center value
                        if n_value <= center_value:
                            flag_managed_raster.set(xi_n, yi_n, 4) # filled as neighbor
                            q.push(CoordinatePair(xi_n, yi_n))
                            # raise neighbor dem to center value
                            if n_value < center_value:
                                dem_filled_managed_raster.set(
                                    xi_n, yi_n, center_value)
                        else:
                            # not flat so must be a slope pixel,
                            # push to slope queue
                            flag_managed_raster.set(xi_n, yi_n, 5) # filled as upslope
                            sq.push(CoordinatePair(xi_n, yi_n))
            else:
                # otherwise it's a slope pixel, push to slope queue
                sq.push(CoordinatePair(xi_n, yi_n))

            # grow up the slopes
            while not sq.empty():
                isProcessed = 0
                xi_s = sq.front().first
                yi_s = sq.front().second
                sq.pop()
                s_center_value = dem_managed_raster.get(xi_s, yi_s)
                for i in xrange(8):
                    xi_n = xi_s+OFFSET_ARRAY[2*i+1]
                    yi_n = yi_s+OFFSET_ARRAY[2*i]
                    if (xi_n < 0 or yi_n < 0 or
                            xi_n >= raster_x_size or
                            yi_n >= raster_y_size):
                        continue
                    if flag_managed_raster.get(xi_n, yi_n):
                        continue
                    n_value = dem_managed_raster.get(xi_n, yi_n)
                    # loop invariant: n_value not nodata because flag not set
                    # if neighbor is higher than center, grow slope
                    if n_value > s_center_value:
                        sq.push(CoordinatePair(xi_n, yi_n))
                        flag_managed_raster.set(xi_n, yi_n, 6)
                    elif not isProcessed:
                        isProcessed = 1
                        # nonRegionCell call
                        isBoundary = 1
                        for j in xrange(8):
                            # check neighbors of neighbor
                            xj_n = xi_n+OFFSET_ARRAY[2*j+1]
                            yj_n = yi_n+OFFSET_ARRAY[2*j]
                            if (xj_n < 0 or yj_n < 0 or
                                    xj_n >= raster_x_size or
                                    yj_n > raster_y_size):
                                continue
                            j_value = dem_managed_raster.get(xj_n, yj_n)
                            # check for nodata
                            if isclose(j_value, dem_nodata):
                                continue
                            if (flag_managed_raster.get(xj_n, yj_n) and
                                    (j_value < n_value)):
                                # if flag(j) && DEM(j) < DEM(n) it's not a
                                # boundary because downhill neighbor has been
                                # processed
                                isBoundary = 0
                                break
                        if isBoundary:
                            p_queue.push(
                                new Pixel(s_center_value, xi_s, yi_s))
                        else:
                            isProcessed = 0

    print 'done printing result'
    print 'deleting flag'
    del flag_managed_raster
    print 'deleting dem'
    del dem_managed_raster
