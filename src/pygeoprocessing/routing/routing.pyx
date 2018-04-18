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

# This module creates rasters with a memory xy block size of 2**BLOCK_BITS
cdef int BLOCK_BITS = 8

# Number of raster blocks to hold in memory at once per Managed Raster
cdef int MANAGED_RASTER_N_BLOCKS = 2**6

# these are the creation options that'll be used for all the rasters
GTIFF_CREATION_OPTIONS = (
    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
    'BLOCKXSIZE=%d' % (1 << BLOCK_BITS),
    'BLOCKYSIZE=%d' % (1 << BLOCK_BITS))

# if nodata is not defined for a float, it's a difficult choice. this number
# probably won't collide with anything ever created by humans
IMPROBABLE_FLOAT_NOATA = -1.23789789e29

# this is a fast C function to determine if two doubles are almost equal
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
ctypedef pair[int, double*] BlockBufferPair

# a class to allow fast random per-pixel access to a raster for both setting
# and reading pixels.
cdef class _ManagedRaster:
    cdef LRUCache[int, double*]* lru_cache
    cdef cset[int] dirty_blocks
    cdef int block_xsize
    cdef int block_ysize
    cdef int block_xbits
    cdef int block_ybits
    cdef int raster_x_size
    cdef int raster_y_size
    cdef int block_nx
    cdef int block_ny
    cdef int write_mode
    cdef char* raster_path
    cdef int band_id
    cdef int closed

    def __cinit__(self, char* raster_path, int band_id, write_mode):
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

        if (self.block_xsize & (self.block_xsize - 1) != 0) or (
                self.block_ysize & (self.block_ysize - 1) != 0):
            # If inputs are not a power of two, this will at least print
            # an error message. Unfortunately with Cython, the exception will
            # present itself as a hard seg-fault, but I'm leaving the
            # ValueError in here at least for readability.
            err_msg = (
                "Error: Block size is not a power of two: "
                "block_xsize: %d, %d, %s" % (
                    self.block_xsize, self.block_ysize, raster_path))
            print err_msg
            raise ValueError(err_msg)

        self.block_xbits = numpy.log2(self.block_xsize)
        self.block_ybits = numpy.log2(self.block_ysize)
        self.block_nx = (
            self.raster_x_size + (self.block_xsize) - 1) / self.block_xsize
        self.block_ny = (
            self.raster_y_size + (self.block_ysize) - 1) / self.block_ysize

        self.lru_cache = new LRUCache[int, double*](MANAGED_RASTER_N_BLOCKS)
        self.raster_path = raster_path
        self.band_id = band_id
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

        raster = gdal.Open(self.raster_path, gdal.GA_Update)
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

                for xi_copy in xrange(win_xsize):
                    for yi_copy in xrange(win_ysize):
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

    cdef void set(self, int xi, int yi, double value):
        """Set the pixel at `xi,yi` to `value`."""
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        cdef int xoff = block_xi << self.block_xbits
        cdef int yoff = block_yi << self.block_ybits
        self.lru_cache.get(
            block_index)[((yi-yoff)<<self.block_xbits)+xi-xoff] = value
        if self.write_mode:
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr == self.dirty_blocks.end():
                self.dirty_blocks.insert(block_index)

    cdef double get(self, int xi, int yi):
        """Return the value of the pixel at `xi,yi`."""
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        cdef int xoff = block_xi << self.block_xbits
        cdef int yoff = block_yi << self.block_ybits
        return self.lru_cache.get(
            block_index)[((yi-yoff)<<self.block_xbits)+xi-xoff]

    cdef void _load_block(self, int block_index) except *:
        cdef int block_xi = block_index % self.block_nx
        cdef int block_yi = block_index / self.block_nx

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

        raster = gdal.Open(self.raster_path)
        raster_band = raster.GetRasterBand(self.band_id)
        block_array = raster_band.ReadAsArray(
            xoff=xoff, yoff=yoff, win_xsize=win_xsize,
            win_ysize=win_ysize).astype(
            numpy.float64)
        raster_band = None
        raster = None
        double_buffer = <double*>PyMem_Malloc(
            (sizeof(double) << self.block_xbits) * win_ysize)
        for xi_copy in xrange(win_xsize):
            for yi_copy in xrange(win_ysize):
                double_buffer[(yi_copy<<self.block_xbits)+xi_copy] = (
                    block_array[yi_copy, xi_copy])
        self.lru_cache.put(
            <int>block_index, <double*>double_buffer, removed_value_list)

        if self.write_mode:
            raster = gdal.Open(self.raster_path, gdal.GA_Update)
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

# type used to store x/y coordinates
ctypedef pair[int, int] CoordinatePair

# this is the class type that'll get stored in the priority queue
cdef struct Pixel:
    double value  # pixel value
    int xi  # pixel x coordinate in the raster
    int yi  # pixel y coordinate in the raster
    int priority # for breaking ties if two `value`s are equal.
ctypedef (Pixel*) PixelPtr

# functor for priority queue of pixels
cdef cppclass GreaterPixel nogil:
    bint get "operator()"(PixelPtr& lhs, PixelPtr& rhs):
        # lhs is > than rhs if its value is greater or if it's equal if
        # the priority is >.
        if deref(lhs).value > deref(rhs).value:
            return 1
        if deref(lhs).value == deref(rhs).value:
            if deref(lhs).priority > deref(rhs).priority:
                return 1
        return 0


def fill_pits(
        dem_raster_path_band, target_filled_dem_raster_path,
        working_dir=None):
    """Fill the pits in a DEM.

        This function defines pits as hydrologically connected regions that do
        not drain to the edge of the raster or a nodata pixel. After the call
        pits are filled to the height of the lowest pour point.

    Parameters:
        dem_raster_path_band (tuple): a path, band number tuple indicating the
            DEM calculate flow direction.
        target_filled_dem_raster_path (string): path to pit filled dem, that's
            functionally a copy of `dem_raster_path_band[0]` with the pit
            pixels raised to the pour point. For runtime efficiency, this
            raster is tiled and its blocksize is set to
            (1<<BLOCK_BITS, 1<<BLOCK_BITS) even if `dem_raster_path_band[0]`
            was not tiled or a different block size.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run. If this directory doesn't exist
            it is created by this call. If None, a temporary directory is
            created by tempdir.mkdtemp which is removed after the function
            call completes successfully.

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
    cdef queue[CoordinatePair] search_queue

    # `fill_queue` is used after a region is identified as a pit and its pour
    # height is determined to fill the pit up to the pour height
    cdef queue[CoordinatePair] fill_queue

    # a pixel pointer is used to push to a priority queue. it remembers its
    # pixel value, x/y index, and an optional priority value to order if
    # heights are equal.
    cdef PixelPtr pixel

    # this priority queue is used to iterate over pit pixels in increasing
    # height, to search for the lowest pour point.
    cdef priority_queue[PixelPtr, deque[PixelPtr], GreaterPixel] pit_queue

    # properties of the parallel rasters
    cdef int raster_x_size, raster_y_size, n_x_blocks

    # variables to remember heights of DEM
    cdef double center_val, dem_nodata, fill_height

    # used to uniquely identify each flat/pit region encountered in the
    # algorithm, it's written into the mask rasters to indicate which pixels
    # have already been processed
    cdef int feature_id

    # used to loop over neighbors and offset the x/y values as defined below
    # 321
    # 4x0
    # 567
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

    # used for time-delayed logging
    cdef time_t last_log_time
    last_log_time = ctime(NULL)

    logger = logging.getLogger('pygeoprocessing.routing.fill_pits')
    logger.addHandler(logging.NullHandler())  # silence logging by default

    # determine dem nodata in the working type, or set an improbable value
    # if one can't be determined
    dem_raster_info = pygeoprocessing.get_raster_info(dem_raster_path_band[0])
    base_nodata = dem_raster_info['nodata'][dem_raster_path_band[1]-1]
    if base_nodata is not None:
        # cast to a float64 since that's our operating array type
        dem_nodata = numpy.float64(base_nodata)
    else:
        # pick some very improbable value since it's hard to deal with NaNs
        dem_nodata = IMPROBABLE_FLOAT_NOATA

    # these are used to determine if a sample is within the raster
    raster_x_size, raster_y_size = dem_raster_info['raster_size']

    # this is the nodata value for all the flat region and pit masks
    mask_nodata = 0

    # set up the working dir for the mask rasters
    try:
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
        gtiff_creation_options=GTIFF_CREATION_OPTIONS)
    flat_region_mask_managed_raster = _ManagedRaster(
        flat_region_mask_path, 1, 1)

    # this raster will have the value of 'feature_id' set to it if it has
    # been searched as part of the search for a pour point for pit number
    # `feature_id`
    pit_mask_path = os.path.join(working_dir_path, 'pit_mask.tif')
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], pit_mask_path, gdal.GDT_Int32,
        [mask_nodata], fill_value_list=[mask_nodata],
        gtiff_creation_options=GTIFF_CREATION_OPTIONS)
    pit_mask_managed_raster = _ManagedRaster(
        pit_mask_path, 1, 1)

    # copy the base DEM to the target and set up for writing
    gdal_driver = gdal.GetDriverByName('GTiff')
    base_dem_raster = gdal.Open(dem_raster_path_band[0])
    gdal_driver.CreateCopy(
        target_filled_dem_raster_path, base_dem_raster,
        options=GTIFF_CREATION_OPTIONS)
    target_dem_raster = gdal.OpenEx(
        target_filled_dem_raster_path, gdal.OF_RASTER)
    target_dem_band = target_dem_raster.GetRasterBand(dem_raster_path_band[1])
    filled_dem_managed_raster = _ManagedRaster(
        target_filled_dem_raster_path, dem_raster_path_band[1], 1)

    # feature_id will start at 1 since the mask nodata is 0.
    feature_id = 0

    # this outer loop searches for a pixel that is locally undrained
    for offset_dict in pygeoprocessing.iterblocks(
            dem_raster_path_band[0], offset_only=True, largest_block=0):
        win_xsize = offset_dict['win_xsize']
        win_ysize = offset_dict['win_ysize']
        xoff = offset_dict['xoff']
        yoff = offset_dict['yoff']

        if ctime(NULL) - last_log_time > 5.0:
            last_log_time = ctime(NULL)
            current_pixel = xoff + yoff * raster_x_size
            logger.info('%.2f%% complete', 100.0 * current_pixel / <float>(
                raster_x_size * raster_y_size))

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
            buffer_off['xa']:buffer_off['xb']] = target_dem_band.ReadAsArray(
                **offset_dict).astype(numpy.float64)

        # search block for locally undrained pixels
        for yi in xrange(1, win_ysize+1):
            for xi in xrange(1, win_xsize+1):
                center_val = dem_buffer_array[yi, xi]
                if isclose(center_val, dem_nodata):
                    continue

                # this value is set in case it turns out to be the root of a
                # pit, we'll start the fill from this pixel in the last phase
                # of the algorithm
                xi_root = xi-1+xoff
                yi_root = yi-1+yoff

                if not isclose(
                        mask_nodata,
                        flat_region_mask_managed_raster.get(
                            xi_root, yi_root)):
                    # already been searched
                    continue

                # search neighbors for downhill or nodata
                downhill_neighbor = 0
                nodata_neighbor = 0

                for i_n in xrange(8):
                    xi_n = xi_root+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                    yi_n = yi_root+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
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
                    # it drains, so skip
                    continue

                # otherwise, this pixel doesn't drain locally, search to see
                # if it's a pit or plateau
                search_queue.push(CoordinatePair(xi_root, yi_root))
                flat_region_mask_managed_raster.set(
                    xi_root, yi_root, 1)
                downhill_drain = 0
                nodata_drain = 0

                # this loop does a BFS starting at this pixel to all pixels
                # of the same height. the _drain variables are used to
                # remember if a drain was encountered. it is preferable to
                # search the whole region even if a drain is encountered, so
                # it can be entirely marked as processed and not re-accessed
                # on later iterations
                while not search_queue.empty():
                    xi_q = search_queue.front().first
                    yi_q = search_queue.front().second
                    search_queue.pop()

                    for i_n in xrange(8):
                        xi_n = xi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = yi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
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
                                mask_nodata,
                                flat_region_mask_managed_raster.get(
                                    xi_n, yi_n)):
                            # only grow if it's at the same level and not
                            # previously visited
                            search_queue.push(
                                CoordinatePair(xi_n, yi_n))
                            flat_region_mask_managed_raster.set(
                                xi_n, yi_n, 1)

                if not downhill_drain and not nodata_drain:
                    # entire region was searched with no drain, do a fill
                    pixel = <Pixel*>PyMem_Malloc(sizeof(Pixel))
                    deref(pixel).xi = xi_root
                    deref(pixel).yi = yi_root
                    deref(pixel).value = center_val
                    # set the priority to be the block index
                    deref(pixel).priority = (
                        n_x_blocks * (yi_root >> BLOCK_BITS) +
                        xi_root >> BLOCK_BITS)
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
                    xi_q = deref(pixel).xi
                    yi_q = deref(pixel).yi
                    # this is the potential fill height if pixel is pour point
                    fill_height = deref(pixel).value
                    PyMem_Free(pixel)

                    for i_n in xrange(8):
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
                        if isclose(n_height, dem_nodata) or (
                                n_height < fill_height):
                            # we encounter a neighbor not processed that is
                            # lower than the current pixel or nodata
                            pour_point = 1
                            break

                        # push onto queue
                        pixel = <Pixel*>PyMem_Malloc(sizeof(Pixel))
                        deref(pixel).xi = xi_n
                        deref(pixel).yi = yi_n
                        deref(pixel).value = n_height
                        # set the priority to be the block index
                        deref(pixel).priority = (
                            n_x_blocks * (yi_n >> BLOCK_BITS) +
                            xi_n >> BLOCK_BITS)
                        pit_queue.push(pixel)

                    if pour_point:
                        # found a pour point, clear the queue
                        while not pit_queue.empty():
                            PyMem_Free(pit_queue.top())
                            pit_queue.pop()

                        # start from original pit seed rather than pour point
                        # this way we can stop filling when we reach a height
                        # equal to fill_height instead of potentially
                        # traversing a plateau area and needing to
                        # differentiate the pixels on the inside of the pit
                        # and the outside.
                        fill_queue.push(CoordinatePair(xi_root, yi_root))
                        filled_dem_managed_raster.set(
                            xi_root, yi_root, fill_height)

                # this loop does a BFS to set all DEM pixels to `fill_height`
                while not fill_queue.empty():
                    xi_q = fill_queue.front().first
                    yi_q = fill_queue.front().second
                    fill_queue.pop()

                    for i_n in xrange(8):
                        xi_n = xi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n]
                        yi_n = yi_q+NEIGHBOR_OFFSET_ARRAY[2*i_n+1]
                        if (xi_n < 0 or xi_n >= raster_x_size or
                                yi_n < 0 or yi_n >= raster_y_size):
                            continue

                        if filled_dem_managed_raster.get(
                                xi_n, yi_n) < fill_height:
                            filled_dem_managed_raster.set(
                                xi_n, yi_n, fill_height)
                            fill_queue.push(CoordinatePair(xi_n, yi_n))

    pit_mask_managed_raster.close()
    flat_region_mask_managed_raster.close()
    shutil.rmtree(working_dir_path)
    logger.info('%.2f%% complete', 100.0)
