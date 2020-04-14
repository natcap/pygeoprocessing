# coding=UTF-8
# distutils: language=c++
# cython: language_level=3
import os
import tempfile
import logging
import time
import sys
import traceback
import shutil


from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport numpy
import numpy
cimport cython
from libcpp.list cimport list as clist
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp.vector cimport vector
cimport libcpp.algorithm
from libc.stdio cimport FILE
from libc.stdio cimport fopen
from libc.stdio cimport fwrite
from libc.stdio cimport fread
from libc.stdio cimport fclose
from osgeo import gdal
import pygeoprocessing
import taskgraph


DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS = ('GTIFF', (
    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
    'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
LOGGER = logging.getLogger('pygeoprocessing.geoprocessing_core')

cdef float _NODATA = -1.0

cdef extern from "FastFileIterator.h" nogil:
    cdef cppclass FastFileIterator[DATA_T]:
        FastFileIterator(const char*, size_t)
        DATA_T next()
        size_t size()
    int FastFileIteratorCompare[DATA_T](
        FastFileIterator[DATA_T]*, FastFileIterator[DATA_T]*)

cdef extern from "FastFileIteratorIndex.h" nogil:
    cdef cppclass FastFileIteratorIndex[DATA_T]:
        FastFileIteratorIndex(const char*, const char*, size_t)
        long long next()
        DATA_T get_last_val()
        size_t size()
    int FastFileIteratorIndexCompare[DATA_T](
        FastFileIteratorIndex[DATA_T]*, FastFileIteratorIndex[DATA_T]*)

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


ctypedef pair[int, double*] BlockBufferPair
cdef int BLOCK_BITS = 8
cdef int MANAGED_RASTER_N_BLOCKS = 2**6

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


# This resolves an issue on Mac OS X Catalina where cimporting ``push_heap``
# and ``pop_heap`` from the Standard Library would cause compilation to fail
# with an error message about the candidate function template not being
# viable.  The SO answer to a related question
# (https://stackoverflow.com/a/57586789/299084) suggests a workaround: don't
# tell Cython that we have a template function.  Using ``...`` here allows
# us to not have to specify all of the types for which we need a working
# ``push_heap`` and ``pop_heap``.
cdef extern from "<algorithm>" namespace "std":
    void push_heap(...)
    void pop_heap(...)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _distance_transform_edt(
        region_raster_path, g_raster_path, float sample_d_x,
        float sample_d_y, target_distance_raster_path,
        raster_driver_creation_tuple):
    """Calculate the euclidean distance transform on base raster.

    Calculates the euclidean distance transform on the base raster in units of
    pixels multiplied by an optional scalar constant. The implementation is
    based off the algorithm described in:  Meijster, Arnold, Jos BTM Roerdink,
    and Wim H. Hesselink. "A general algorithm for computing distance
    transforms in linear time." Mathematical Morphology and its applications
    to image and signal processing. Springer, Boston, MA, 2002. 331-340.

    The base mask raster represents the area to distance transform from as
    any pixel that is not 0 or nodata. It is computationally convenient to
    calculate the distance transform on the entire raster irrespective of
    nodata placement and thus produces a raster that will have distance
    transform values even in pixels that are nodata in the base.

    Parameters:
        region_raster_path (string): path to a byte raster where region pixels
            are indicated by a 1 and 0 otherwise.
        g_raster_path (string): path to a raster created by this call that
            is used as the intermediate "g" variable described in Meijster
            et. al.
        sample_d_x (float):
        sample_d_y (float):
            These parameters scale the pixel distances when calculating the
            distance transform. `d_x` is the x direction when changing a
            column index, and `d_y` when changing a row index. Both values
            must be > 0.
        target_distance_raster_path (string): path to the target raster
            created by this call that is the exact euclidean distance
            transform from any pixel in the base raster that is not nodata and
            not 0. The units are in (pixel distance * sampling_distance).
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None

    """
    cdef int yoff, row_index, block_ysize, win_ysize, n_rows
    cdef int xoff, block_xsize, win_xsize, n_cols
    cdef int q_index, local_x_index, local_y_index, u_index
    cdef int tq, sq
    cdef float gu, gsq, w
    cdef numpy.ndarray[numpy.float32_t, ndim=2] g_block
    cdef numpy.ndarray[numpy.int32_t, ndim=1] s_array
    cdef numpy.ndarray[numpy.int32_t, ndim=1] t_array
    cdef numpy.ndarray[numpy.float32_t, ndim=2] dt
    cdef numpy.ndarray[numpy.int8_t, ndim=2] mask_block

    mask_raster = gdal.OpenEx(region_raster_path, gdal.OF_RASTER)
    mask_band = mask_raster.GetRasterBand(1)

    n_cols = mask_raster.RasterXSize
    n_rows = mask_raster.RasterYSize

    raster_info = pygeoprocessing.get_raster_info(region_raster_path)
    pygeoprocessing.new_raster_from_base(
        region_raster_path, g_raster_path, gdal.GDT_Float32, [_NODATA],
        fill_value_list=None,
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    g_raster = gdal.OpenEx(g_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    g_band = g_raster.GetRasterBand(1)
    g_band_blocksize = g_band.GetBlockSize()

    # normalize the sample distances so we don't get a strange numerical
    # overflow
    max_sample = max(sample_d_x, sample_d_y)
    sample_d_x /= max_sample
    sample_d_y /= max_sample

    # distances can't be larger than half the perimeter of the raster.
    cdef float numerical_inf = max(sample_d_x, 1.0) * max(sample_d_y, 1.0) * (
        raster_info['raster_size'][0] + raster_info['raster_size'][1])
    # scan 1
    done = False
    block_xsize = raster_info['block_size'][0]
    mask_block = numpy.empty((n_rows, block_xsize), dtype=numpy.int8)
    g_block = numpy.empty((n_rows, block_xsize), dtype=numpy.float32)
    for xoff in numpy.arange(0, n_cols, block_xsize):
        win_xsize = block_xsize
        if xoff + win_xsize > n_cols:
            win_xsize = n_cols - xoff
            mask_block = numpy.empty((n_rows, win_xsize), dtype=numpy.int8)
            g_block = numpy.empty((n_rows, win_xsize), dtype=numpy.float32)
            done = True
        mask_band.ReadAsArray(
            xoff=xoff, yoff=0, win_xsize=win_xsize, win_ysize=n_rows,
            buf_obj=mask_block)
        # base case
        g_block[0, :] = (mask_block[0, :] == 0) * numerical_inf
        for row_index in range(1, n_rows):
            for local_x_index in range(win_xsize):
                if mask_block[row_index, local_x_index] == 1:
                    g_block[row_index, local_x_index] = 0
                else:
                    g_block[row_index, local_x_index] = (
                        g_block[row_index-1, local_x_index] + sample_d_y)
        for row_index in range(n_rows-2, -1, -1):
            for local_x_index in range(win_xsize):
                if (g_block[row_index+1, local_x_index] <
                        g_block[row_index, local_x_index]):
                    g_block[row_index, local_x_index] = (
                        sample_d_y + g_block[row_index+1, local_x_index])
        g_band.WriteArray(g_block, xoff=xoff, yoff=0)
        if done:
            break
    g_band.FlushCache()

    cdef float distance_nodata = -1.0

    pygeoprocessing.new_raster_from_base(
        region_raster_path, target_distance_raster_path.encode('utf-8'),
        gdal.GDT_Float32, [distance_nodata], fill_value_list=None,
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    target_distance_raster = gdal.OpenEx(
        target_distance_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    target_distance_band = target_distance_raster.GetRasterBand(1)

    LOGGER.info('Distance Transform Phase 2')
    s_array = numpy.empty(n_cols, dtype=numpy.int32)
    t_array = numpy.empty(n_cols, dtype=numpy.int32)

    done = False
    block_ysize = g_band_blocksize[1]
    g_block = numpy.empty((block_ysize, n_cols), dtype=numpy.float32)
    dt = numpy.empty((block_ysize, n_cols), dtype=numpy.float32)
    mask_block = numpy.empty((block_ysize, n_cols), dtype=numpy.int8)
    sq = 0  # initialize so compiler doesn't complain
    gsq = 0
    for yoff in numpy.arange(0, n_rows, block_ysize):
        win_ysize = block_ysize
        if yoff + win_ysize >= n_rows:
            win_ysize = n_rows - yoff
            g_block = numpy.empty((win_ysize, n_cols), dtype=numpy.float32)
            mask_block = numpy.empty((win_ysize, n_cols), dtype=numpy.int8)
            dt = numpy.empty((win_ysize, n_cols), dtype=numpy.float32)
            done = True
        g_band.ReadAsArray(
            xoff=0, yoff=yoff, win_xsize=n_cols, win_ysize=win_ysize,
            buf_obj=g_block)
        mask_band.ReadAsArray(
            xoff=0, yoff=yoff, win_xsize=n_cols, win_ysize=win_ysize,
            buf_obj=mask_block)
        for local_y_index in range(win_ysize):
            q_index = 0
            s_array[0] = 0
            t_array[0] = 0
            for u_index in range(1, n_cols):
                gu = g_block[local_y_index, u_index]**2
                while (q_index >= 0):
                    tq = t_array[q_index]
                    sq = s_array[q_index]
                    gsq = g_block[local_y_index, sq]**2
                    if ((sample_d_x*(tq-sq))**2 + gsq <= (
                            sample_d_x*(tq-u_index))**2 + gu):
                        break
                    q_index -= 1
                if q_index < 0:
                    q_index = 0
                    s_array[0] = u_index
                    sq = u_index
                    gsq = g_block[local_y_index, sq]**2
                else:
                    w = (float)(sample_d_x + ((
                        (sample_d_x*u_index)**2 - (sample_d_x*sq)**2 +
                        gu - gsq) / (2*sample_d_x*(u_index-sq))))
                    if w < n_cols*sample_d_x:
                        q_index += 1
                        s_array[q_index] = u_index
                        t_array[q_index] = <int>(w / sample_d_x)

            sq = s_array[q_index]
            gsq = g_block[local_y_index, sq]**2
            tq = t_array[q_index]
            for u_index in range(n_cols-1, -1, -1):
                if mask_block[local_y_index, u_index] != 1:
                    dt[local_y_index, u_index] = (
                        sample_d_x*(u_index-sq))**2+gsq
                else:
                    dt[local_y_index, u_index] = 0
                if u_index <= tq:
                    q_index -= 1
                    if q_index >= 0:
                        sq = s_array[q_index]
                        gsq = g_block[local_y_index, sq]**2
                        tq = t_array[q_index]

        valid_mask = g_block != _NODATA
        # "unnormalize" distances along with square root
        dt[valid_mask] = numpy.sqrt(dt[valid_mask]) * max_sample
        dt[~valid_mask] = _NODATA
        target_distance_band.WriteArray(dt, xoff=0, yoff=yoff)

        # we do this in the case where the blocksize is many times larger than
        # the raster size so we don't re-loop through the only block
        if done:
            break

    target_distance_band.ComputeStatistics(0)
    target_distance_band.FlushCache()
    target_distance_band = None
    mask_band = None
    g_band = None
    target_distance_raster = None
    mask_raster = None
    g_raster = None

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def calculate_slope(
        base_elevation_raster_path_band, target_slope_path,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Create a percent slope raster from DEM raster.

    Base algorithm is from Zevenbergen & Thorne "Quantitative Analysis of Land
    Surface Topography" 1987 although it has been modified to include the
    diagonal pixels by classic finite difference analysis.

    For the following notation, we define each pixel's DEM value by a letter
    with this spatial scheme:

        abc
        def
        ghi

    Then the slope at e is defined at ([dz/dx]^2 + [dz/dy]^2)^0.5

    Where

    [dz/dx] = ((c+2f+i)-(a+2d+g)/(8*x_cell_size)
    [dz/dy] = ((g+2h+i)-(a+2b+c))/(8*y_cell_size)

    In cases where a cell is nodata, we attempt to use the middle cell inline
    with the direction of differentiation (either in x or y direction).  If
    no inline pixel is defined, we use `e` and multiply the difference by
    2^0.5 to account for the diagonal projection.

    Parameters:
        base_elevation_raster_path_band (string): a path/band tuple to a
            raster of height values. (path_to_raster, band_index)
        target_slope_path (string): path to target slope raster; will be a
            32 bit float GeoTIFF of same size/projection as calculate slope
            with units of percent slope.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to
            geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None
    """
    cdef numpy.npy_float64 a, b, c, d, e, f, g, h, i, dem_nodata
    cdef numpy.npy_float64 x_cell_size, y_cell_size,
    cdef numpy.npy_float64 dzdx_accumulator, dzdy_accumulator
    cdef int row_index, col_index, n_rows, n_cols,
    cdef int x_denom_factor, y_denom_factor, win_xsize, win_ysize
    cdef numpy.ndarray[numpy.npy_float64, ndim=2] dem_array
    cdef numpy.ndarray[numpy.npy_float64, ndim=2] slope_array
    cdef numpy.ndarray[numpy.npy_float64, ndim=2] dzdx_array
    cdef numpy.ndarray[numpy.npy_float64, ndim=2] dzdy_array

    dem_raster = gdal.OpenEx(base_elevation_raster_path_band[0])
    dem_band = dem_raster.GetRasterBand(base_elevation_raster_path_band[1])
    dem_info = pygeoprocessing.get_raster_info(
        base_elevation_raster_path_band[0])
    raw_nodata = dem_info['nodata'][0]
    if raw_nodata is None:
        # if nodata is undefined, choose most negative 32 bit float
        raw_nodata = numpy.finfo(numpy.float32).min
    dem_nodata = raw_nodata
    x_cell_size, y_cell_size = dem_info['pixel_size']
    n_cols, n_rows = dem_info['raster_size']
    cdef numpy.npy_float64 slope_nodata = numpy.finfo(numpy.float32).min
    pygeoprocessing.new_raster_from_base(
        base_elevation_raster_path_band[0], target_slope_path,
        gdal.GDT_Float32, [slope_nodata],
        fill_value_list=[float(slope_nodata)],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    target_slope_raster = gdal.OpenEx(target_slope_path, gdal.GA_Update)
    target_slope_band = target_slope_raster.GetRasterBand(1)

    for block_offset in pygeoprocessing.iterblocks(
            base_elevation_raster_path_band, offset_only=True):
        block_offset_copy = block_offset.copy()
        # try to expand the block around the edges if it fits
        x_start = 1
        win_xsize = block_offset['win_xsize']
        x_end = win_xsize+1
        y_start = 1
        win_ysize = block_offset['win_ysize']
        y_end = win_ysize+1

        if block_offset['xoff'] > 0:
            block_offset_copy['xoff'] -= 1
            block_offset_copy['win_xsize'] += 1
            x_start -= 1
        if block_offset['xoff']+win_xsize < n_cols:
            block_offset_copy['win_xsize'] += 1
            x_end += 1
        if block_offset['yoff'] > 0:
            block_offset_copy['yoff'] -= 1
            block_offset_copy['win_ysize'] += 1
            y_start -= 1
        if block_offset['yoff']+win_ysize < n_rows:
            block_offset_copy['win_ysize'] += 1
            y_end += 1

        dem_array = numpy.empty(
            (win_ysize+2, win_xsize+2),
            dtype=numpy.float64)
        dem_array[:] = dem_nodata
        slope_array = numpy.empty(
            (win_ysize, win_xsize),
            dtype=numpy.float64)
        dzdx_array = numpy.empty(
            (win_ysize, win_xsize),
            dtype=numpy.float64)
        dzdy_array = numpy.empty(
            (win_ysize, win_xsize),
            dtype=numpy.float64)

        dem_band.ReadAsArray(
            buf_obj=dem_array[y_start:y_end, x_start:x_end],
            **block_offset_copy)

        for row_index in range(1, win_ysize+1):
            for col_index in range(1, win_xsize+1):
                # Notation of the cell below comes from the algorithm
                # description, cells are arraged as follows:
                # abc
                # def
                # ghi
                e = dem_array[row_index, col_index]
                if e == dem_nodata:
                    # we use dzdx as a guard below, no need to set dzdy
                    dzdx_array[row_index-1, col_index-1] = slope_nodata
                    continue
                dzdx_accumulator = 0.0
                dzdy_accumulator = 0.0
                x_denom_factor = 0
                y_denom_factor = 0
                a = dem_array[row_index-1, col_index-1]
                b = dem_array[row_index-1, col_index]
                c = dem_array[row_index-1, col_index+1]
                d = dem_array[row_index, col_index-1]
                f = dem_array[row_index, col_index+1]
                g = dem_array[row_index+1, col_index-1]
                h = dem_array[row_index+1, col_index]
                i = dem_array[row_index+1, col_index+1]

                # a - c direction
                if a != dem_nodata and c != dem_nodata:
                    dzdx_accumulator += a - c
                    x_denom_factor += 2
                elif a != dem_nodata and b != dem_nodata:
                    dzdx_accumulator += a - b
                    x_denom_factor += 1
                elif b != dem_nodata and c != dem_nodata:
                    dzdx_accumulator += b - c
                    x_denom_factor += 1
                elif a != dem_nodata:
                    dzdx_accumulator += (a - e) * 2**0.5
                    x_denom_factor += 1
                elif c != dem_nodata:
                    dzdx_accumulator += (e - c) * 2**0.5
                    x_denom_factor += 1

                # d - f direction
                if d != dem_nodata and f != dem_nodata:
                    dzdx_accumulator += 2 * (d - f)
                    x_denom_factor += 4
                elif d != dem_nodata:
                    dzdx_accumulator += 2 * (d - e)
                    x_denom_factor += 2
                elif f != dem_nodata:
                    dzdx_accumulator += 2 * (e - f)
                    x_denom_factor += 2

                # g - i direction
                if g != dem_nodata and i != dem_nodata:
                    dzdx_accumulator += g - i
                    x_denom_factor += 2
                elif g != dem_nodata and h != dem_nodata:
                    dzdx_accumulator += g - h
                    x_denom_factor += 1
                elif h != dem_nodata and i != dem_nodata:
                    dzdx_accumulator += h - i
                    x_denom_factor += 1
                elif g != dem_nodata:
                    dzdx_accumulator += (g - e) * 2**0.5
                    x_denom_factor += 1
                elif i != dem_nodata:
                    dzdx_accumulator += (e - i) * 2**0.5
                    x_denom_factor += 1

                # a - g direction
                if a != dem_nodata and g != dem_nodata:
                    dzdy_accumulator += a - g
                    y_denom_factor += 2
                elif a != dem_nodata and d != dem_nodata:
                    dzdy_accumulator += a - d
                    y_denom_factor += 1
                elif d != dem_nodata and g != dem_nodata:
                    dzdy_accumulator += d - g
                    y_denom_factor += 1
                elif a != dem_nodata:
                    dzdy_accumulator += (a - e) * 2**0.5
                    y_denom_factor += 1
                elif g != dem_nodata:
                    dzdy_accumulator += (e - g) * 2**0.5
                    y_denom_factor += 1

                # b - h direction
                if b != dem_nodata and h != dem_nodata:
                    dzdy_accumulator += 2 * (b - h)
                    y_denom_factor += 4
                elif b != dem_nodata:
                    dzdy_accumulator += 2 * (b - e)
                    y_denom_factor += 2
                elif h != dem_nodata:
                    dzdy_accumulator += 2 * (e - h)
                    y_denom_factor += 2

                # c - i direction
                if c != dem_nodata and i != dem_nodata:
                    dzdy_accumulator += c - i
                    y_denom_factor += 2
                elif c != dem_nodata and f != dem_nodata:
                    dzdy_accumulator += c - f
                    y_denom_factor += 1
                elif f != dem_nodata and i != dem_nodata:
                    dzdy_accumulator += f - i
                    y_denom_factor += 1
                elif c != dem_nodata:
                    dzdy_accumulator += (c - e) * 2**0.5
                    y_denom_factor += 1
                elif i != dem_nodata:
                    dzdy_accumulator += (e - i) * 2**0.5
                    y_denom_factor += 1

                if x_denom_factor != 0:
                    dzdx_array[row_index-1, col_index-1] = (
                        dzdx_accumulator / (x_denom_factor * x_cell_size))
                else:
                    dzdx_array[row_index-1, col_index-1] = 0.0
                if y_denom_factor != 0:
                    dzdy_array[row_index-1, col_index-1] = (
                        dzdy_accumulator / (y_denom_factor * y_cell_size))
                else:
                    dzdy_array[row_index-1, col_index-1] = 0.0
        valid_mask = dzdx_array != slope_nodata
        slope_array[:] = slope_nodata
        # multiply by 100 for percent output
        slope_array[valid_mask] = 100.0 * numpy.sqrt(
            dzdx_array[valid_mask]**2 + dzdy_array[valid_mask]**2)
        target_slope_band.WriteArray(
            slope_array, xoff=block_offset['xoff'],
            yoff=block_offset['yoff'])

    dem_band = None
    target_slope_band = None
    gdal.Dataset.__swig_destroy__(dem_raster)
    gdal.Dataset.__swig_destroy__(target_slope_raster)
    dem_raster = None
    target_slope_raster = None


@cython.boundscheck(False)
@cython.cdivision(True)
def stats_worker(stats_work_queue, exception_queue):
    """Worker to calculate continuous min, max, mean and standard deviation.

    Parameters:
        stats_work_queue (Queue): a queue of 1D numpy arrays or None. If
            None, function puts a (min, max, mean, stddev) tuple to the
            queue and quits.

    Returns:
        None

    """
    cdef numpy.ndarray[numpy.float64_t, ndim=1] block
    cdef double M_local = 0.0
    cdef double S_local = 0.0
    cdef double min_value = 0.0
    cdef double max_value = 0.0
    cdef double x = 0.0
    cdef int i, n_elements
    cdef long long n = 0L
    payload = None

    try:
        while True:
            payload = stats_work_queue.get()
            if payload is None:
                LOGGER.debug('payload is None, terminating')
                break
            block = payload.astype(numpy.float64)
            n_elements = block.size
            with nogil:
                for i in range(n_elements):
                    n = n + 1
                    x = block[i]
                    if n <= 0:
                        with gil:
                            LOGGER.error('invalid value for n %s' % n)
                    if n == 1:
                        M_local = x
                        S_local = 0.0
                        min_value = x
                        max_value = x
                    else:
                        M_last = M_local
                        M_local = M_local+(x - M_local)/<double>(n)
                        S_local = S_local+(x-M_last)*(x-M_local)
                        if x < min_value:
                            min_value = x
                        elif x > max_value:
                            max_value = x

        if n > 0:
            stats_work_queue.put(
                (min_value, max_value, M_local, (S_local / <double>n) ** 0.5))
        else:
            LOGGER.warning(
                "No valid pixels were received, sending None.")
            stats_work_queue.put(None)
    except Exception as e:
        LOGGER.exception(
            "exception %s %s %s %s %s", x, M_local, S_local, n, payload)
        exception_queue.put(e)
        while not stats_work_queue.empty():
            stats_work_queue.get()
        raise


ctypedef long long int64t
ctypedef FastFileIterator[long long]* FastFileIteratorLongLongIntPtr
ctypedef FastFileIterator[double]* FastFileIteratorDoublePtr


def raster_band_percentile(
        base_raster_path_band, working_sort_directory, percentile_list,
        heap_buffer_size=2**28, ffi_buffer_size=2**10):
    """Calculate percentiles of a raster band.

    Parameters:
        base_raster_path_band (tuple): raster path band tuple to a raster
            that is of any integer or real type.
        working_sort_directory (str): path to a directory that does not
            exist or is empty. This directory will be used to create heapfiles
            with sizes no larger than ``heap_buffer_size`` which are written in the
            of the pattern N.dat where N is in the numbering 0, 1, 2, ... up
            to the number of files necessary to handle the raster.
        percentile_list (list): sorted list of percentiles to report must
            contain values in the range [0, 100].
        heap_buffer_size (int): defines approximately how many elements to hold in
            a single heap file. This is proportional to the amount of maximum
            memory to use when storing elements before a sort and write to
            disk.
        ffi_buffer_size (int): defines how many elements will be stored per
            heap file buffer for iteration.

    Returns:
        A list of len(percentile_list) elements long containing the
        percentile values (ranging from [0, 100]) in ``base_raster_path_band``
        where the interpolation scheme is "higher" (i.e. any percentile splits
        will select the next element higher than the percentile cutoff).

    """
    raster_type = pygeoprocessing.get_raster_info(
        base_raster_path_band[0])['datatype']
    if raster_type in (
            gdal.GDT_Byte, gdal.GDT_Int16, gdal.GDT_UInt16, gdal.GDT_Int32,
            gdal.GDT_UInt32):
        return _raster_band_percentile_int(
            base_raster_path_band, working_sort_directory, percentile_list,
            heap_buffer_size, ffi_buffer_size)
    elif raster_type in (gdal.GDT_Float32, gdal.GDT_Float64):
        return _raster_band_percentile_double(
            base_raster_path_band, working_sort_directory, percentile_list,
            heap_buffer_size, ffi_buffer_size)
    else:
        raise ValueError(
            'Cannot process raster type %s (not a known integer nor float '
            'type)', raster_type)


def _raster_band_percentile_int(
        base_raster_path_band, working_sort_directory, percentile_list,
        heap_buffer_size, ffi_buffer_size):
    """Calculate percentiles of a raster band of an integer type.

    Parameters:
        base_raster_path_band (tuple): raster path band tuple to a raster that
            is of an integer type.
        working_sort_directory (str): path to a directory that does not
            exist or is empty. This directory will be used to create heapfiles
            with sizes no larger than ``heap_buffer_size`` which are written in the
            of the pattern N.dat where N is in the numbering 0, 1, 2, ... up
            to the number of files necessary to handle the raster.
        percentile_list (list): sorted list of percentiles to report must
            contain values in the range [0, 100].
        heap_buffer_size (int): defines approximately how many elements to hold in
            a single heap file. This is proportional to the amount of maximum
            memory to use when storing elements before a sort and write to
            disk.
        ffi_buffer_size (int): defines how many elements to store in a file
            buffer at any time.

    Returns:
        A list of len(percentile_list) elements long containing the
        percentile values (ranging from [0, 100]) in ``base_raster_path_band``
        where the interpolation scheme is "higher" (i.e. any percentile splits
        will select the next element higher than the percentile cutoff).

    """
    cdef FILE *fptr
    cdef FastFileIteratorLongLongIntPtr fast_file_iterator
    cdef vector[FastFileIteratorLongLongIntPtr] fast_file_iterator_vector
    cdef vector[FastFileIteratorLongLongIntPtr].iterator ffiv_iter
    cdef int percentile_index = 0
    cdef long long i, n_elements = 0
    cdef int64t next_val = 0L
    cdef double step_size, current_percentile
    cdef double current_step = 0.0
    result_list = []
    rm_dir_when_done = False
    try:
        os.makedirs(working_sort_directory)
        rm_dir_when_done = True
    except OSError:
        pass

    cdef int64t[:] buffer_data

    heapfile_list = []
    file_index = 0
    raster_info = pygeoprocessing.get_raster_info(
        base_raster_path_band[0])
    nodata = raster_info['nodata'][base_raster_path_band[1]-1]
    cdef long long n_pixels = raster_info['raster_size'][0] * raster_info['raster_size'][1]

    LOGGER.debug('total number of pixels %s (%s)', n_pixels, raster_info['raster_size'])
    cdef long long pixels_processed = 0
    LOGGER.debug('sorting data to heap')
    last_update = time.time()
    for _, block_data in pygeoprocessing.iterblocks(
            base_raster_path_band, largest_block=heap_buffer_size):
        pixels_processed += block_data.size
        if time.time() - last_update > 5.0:
            LOGGER.debug(
                'data sort to heap %.2f%% complete',
                (100.*pixels_processed)/n_pixels)
            last_update = time.time()
        buffer_data = numpy.sort(
            block_data[~numpy.isclose(block_data, nodata)]).astype(
            numpy.int64)
        if buffer_data.size == 0:
            continue
        n_elements += buffer_data.size
        file_path = os.path.join(
            working_sort_directory, '%d.dat' % file_index)
        heapfile_list.append(file_path)
        fptr = fopen(bytes(file_path.encode()), "wb")
        fwrite(
            <int64t*>&buffer_data[0], sizeof(int64t), buffer_data.size,
            fptr)
        fclose(fptr)
        file_index += 1

        fast_file_iterator = new FastFileIterator[int64t](
            (bytes(file_path.encode())), ffi_buffer_size)
        fast_file_iterator_vector.push_back(fast_file_iterator)
        push_heap(
            fast_file_iterator_vector.begin(),
            fast_file_iterator_vector.end(),
            FastFileIteratorCompare[int64t])
    LOGGER.debug('calculating percentiles')
    current_percentile = percentile_list[percentile_index]
    step_size = 0
    if n_elements > 0:
        step_size = 100.0 / n_elements

    for i in range(n_elements):
        if time.time() - last_update > 5.0:
            LOGGER.debug(
                'calculating percentiles %.2f%% complete',
                100.0 * i / float(n_elements))
            last_update = time.time()
        current_step = step_size * i
        next_val = fast_file_iterator_vector.front().next()
        if current_step >= current_percentile:
            result_list.append(next_val)
            percentile_index += 1
            if percentile_index >= len(percentile_list):
                break
            current_percentile = percentile_list[percentile_index]
        pop_heap(
            fast_file_iterator_vector.begin(),
            fast_file_iterator_vector.end(),
            FastFileIteratorCompare[int64t])
        if fast_file_iterator_vector.back().size() > 0:
            push_heap(
                fast_file_iterator_vector.begin(),
                fast_file_iterator_vector.end(),
                FastFileIteratorCompare[int64t])
        else:
            fast_file_iterator = fast_file_iterator_vector.back()
            del fast_file_iterator
            fast_file_iterator_vector.pop_back()
    if percentile_index < len(percentile_list):
        result_list.append(next_val)

    # free all the iterator memory
    ffiv_iter = fast_file_iterator_vector.begin()
    while ffiv_iter != fast_file_iterator_vector.end():
        fast_file_iterator = deref(ffiv_iter)
        del fast_file_iterator
        inc(ffiv_iter)
    fast_file_iterator_vector.clear()
    # delete all the heap files
    for file_path in heapfile_list:
        try:
            os.remove(file_path)
        except OSError:
            # you never know if this might fail!
            LOGGER.warning('unable to remove %s', file_path)
    if rm_dir_when_done:
        shutil.rmtree(working_sort_directory)
    return result_list


def _raster_band_percentile_double(
        base_raster_path_band, working_sort_directory, percentile_list,
        heap_buffer_size, ffi_buffer_size):
    """Calculate percentiles of a raster band of a real type.

    Parameters:
        base_raster_path_band (tuple): raster path band tuple to raster that
            is a real/float type.
        working_sort_directory (str): path to a directory that does not
            exist or is empty. This directory will be used to create heapfiles
            with sizes no larger than ``heap_buffer_size`` which are written in the
            of the pattern N.dat where N is in the numbering 0, 1, 2, ... up
            to the number of files necessary to handle the raster.
        percentile_list (list): sorted list of percentiles to report must
            contain values in the range [0, 100].
        heap_buffer_size (int): defines approximately how many elements to hold in
            a single heap file. This is proportional to the amount of maximum
            memory to use when storing elements before a sort and write to
            disk.
        ffi_buffer_size (int): defines how many elements to store in a file
            buffer at any time.

    Returns:
        A list of len(percentile_list) elements long containing the
        percentile values (ranging from [0, 100]) in ``base_raster_path_band``
        where the interpolation scheme is "higher" (i.e. any percentile splits
        will select the next element higher than the percentile cutoff).

    """
    cdef FILE *fptr
    cdef double[:] buffer_data
    cdef FastFileIteratorDoublePtr fast_file_iterator
    cdef vector[FastFileIteratorDoublePtr] fast_file_iterator_vector
    cdef int percentile_index = 0
    cdef long long i, n_elements = 0
    cdef double next_val = 0.0
    cdef double current_step = 0.0
    cdef double step_size, current_percentile
    result_list = []
    rm_dir_when_done = False
    try:
        os.makedirs(working_sort_directory)
        rm_dir_when_done = True
    except OSError as e:
        LOGGER.warning("couldn't make working_sort_directory: %s", str(e))
    file_index = 0
    nodata = pygeoprocessing.get_raster_info(
        base_raster_path_band[0])['nodata'][base_raster_path_band[1]-1]
    heapfile_list = []

    raster_info = pygeoprocessing.get_raster_info(
        base_raster_path_band[0])
    nodata = raster_info['nodata'][base_raster_path_band[1]-1]
    n_pixels = numpy.prod(raster_info['raster_size'])
    pixels_processed = 0

    last_update = time.time()
    LOGGER.debug('sorting data to heap')
    for _, block_data in pygeoprocessing.iterblocks(
            base_raster_path_band, largest_block=heap_buffer_size):
        pixels_processed += block_data.size
        if time.time() - last_update > 5.0:
            LOGGER.debug(
                'data sort to heap %.2f%% complete',
                (100.*pixels_processed)/n_pixels)
            last_update = time.time()
        buffer_data = numpy.sort(
            block_data[~numpy.isclose(block_data, nodata)]).astype(
            numpy.double)
        if buffer_data.size == 0:
            continue
        n_elements += buffer_data.size
        file_path = os.path.join(
            working_sort_directory, '%d.dat' % file_index)
        heapfile_list.append(file_path)
        fptr = fopen(bytes(file_path.encode()), "wb")
        fwrite(
            <double*>&buffer_data[0], sizeof(double), buffer_data.size, fptr)
        fclose(fptr)
        file_index += 1

        fast_file_iterator = new FastFileIterator[double](
            (bytes(file_path.encode())), ffi_buffer_size)
        fast_file_iterator_vector.push_back(fast_file_iterator)
        push_heap(
            fast_file_iterator_vector.begin(),
            fast_file_iterator_vector.end(),
            FastFileIteratorCompare[double])

    current_percentile = percentile_list[percentile_index]
    step_size = 0
    if n_elements > 0:
        step_size = 100.0 / n_elements

    LOGGER.debug('calculating percentiles')
    for i in range(n_elements):
        if time.time() - last_update > 5.0:
            LOGGER.debug(
                'calculating percentiles %.2f%% complete',
                100.0 * i / float(n_elements))
            last_update = time.time()
        current_step = step_size * i
        next_val = fast_file_iterator_vector.front().next()
        if current_step >= current_percentile:
            result_list.append(next_val)
            percentile_index += 1
            if percentile_index >= len(percentile_list):
                break
            current_percentile = percentile_list[percentile_index]
        pop_heap(
            fast_file_iterator_vector.begin(),
            fast_file_iterator_vector.end(),
            FastFileIteratorCompare[double])
        if fast_file_iterator_vector.back().size() > 0:
            push_heap(
                fast_file_iterator_vector.begin(),
                fast_file_iterator_vector.end(),
                FastFileIteratorCompare[double])
        else:
            fast_file_iterator_vector.pop_back()
    if percentile_index < len(percentile_list):
        result_list.append(next_val)
    # free all the iterator memory
    ffiv_iter = fast_file_iterator_vector.begin()
    while ffiv_iter != fast_file_iterator_vector.end():
        fast_file_iterator = deref(ffiv_iter)
        del fast_file_iterator
        inc(ffiv_iter)
    fast_file_iterator_vector.clear()
    # delete all the heap files
    for file_path in heapfile_list:
        try:
            os.remove(file_path)
        except OSError:
            # you never know if this might fail!
            LOGGER.warning('unable to remove %s', file_path)
    if rm_dir_when_done:
        shutil.rmtree(working_sort_directory)
    LOGGER.debug('here is percentile_list: %s', str(result_list))
    return result_list


ctypedef FastFileIteratorIndex[double]* FastFileIteratorIndexDoublePtr
ctypedef vector[FastFileIteratorIndexDoublePtr]* FastFileIteratorIndexVectorPtr


def normalize_op(base_array, total_sum, base_nodata, target_nodata):
    """Divide base by total and guard against nodata."""
    result = numpy.empty(base_array.shape, dtype=numpy.float64)
    result[:] = target_nodata
    valid_mask = ~numpy.isclose(base_array, base_nodata)
    result[valid_mask] = (
        base_array[valid_mask].astype(numpy.float64) / total_sum)
    return result


def sum_rasters_op(*array_nodata_list):
    """Sum all non-nodata pixels in array_list.

    Args:
        array_nodata_list (list): 2*n+1 length list where the first n elements
            are arrays, the second n elements are nodata values for those
            arrays and the last element is the target nodata

    Returns:
        sum of valid pixels and target nodata where no coverage.

    """
    result = numpy.zeros(array_nodata_list[0].shape, dtype=numpy.float64)
    total_valid_mask = numpy.zeros(
        array_nodata_list[0].shape, dtype=numpy.bool)
    n = len(array_nodata_list)
    for array, nodata in zip(
            array_nodata_list[0:n//2], array_nodata_list[n//2::]):
        valid_mask = ~numpy.isclose(array, nodata)
        total_valid_mask |= valid_mask
        result[valid_mask] += array[valid_mask]
    result[~total_valid_mask] = array_nodata_list[-1]
    return result


def sum_raster(raster_path_band):
    """Sum the raster and return the result."""
    nodata = pygeoprocessing.get_raster_info(
        raster_path_band[0])['nodata'][raster_path_band[1]-1]

    raster_sum = 0.0
    for _, array in pygeoprocessing.iterblocks(raster_path_band):
        valid_mask = ~numpy.isclose(array, nodata)
        raster_sum += numpy.sum(array[valid_mask])

    return raster_sum


def count_valid(raster_path_band):
    """Sum the raster and return the result."""
    nodata = pygeoprocessing.get_raster_info(
        raster_path_band[0])['nodata'][raster_path_band[1]-1]

    valid_count = 0
    for _, array in pygeoprocessing.iterblocks(raster_path_band):
        valid_mask = ~numpy.isclose(array, nodata) & (array > 0)
        valid_count += numpy.count_nonzero(valid_mask)

    return valid_count


def raster_optimization(
        raster_path_band_list,
        target_sum_list, target_working_directory, target_suffix=None,
        heap_buffer_size=2**28, ffi_buffer_size=2**10,
        preconditioner_weight=0.8):
    """Create a optimized raster selection given the target sum list.

    Args:
        raster_band_list (list): list of (raster_path, band_id) tuples.
        target_sum_list (list): list of floating point values for the target
            sums in each of the raster path bands.
        target_working_directory (str): path to a directory this function can
            use to build up optimization caches and results. This directory
            will have a "./churn" subdirectory containing intermediate files
            that can be resused as part of tweaks to the optimization. The
            root of the directory will contain output rasters of the form
            `basename_band-id_target-suffix.tif`. If `target_suffix` is None
            it is not added to the target raster.
        target_suffix (str): if not none, this suffix is added with a preceding
            '_' to the target filenames generated by this function.
        preconditioner_weight (float): number between 0 and 1, this proportion
            determines how much of the optimization is determined by the
            preconditioner.

    Returns:
        None

    """
    cdef double[:] buffer_data
    cdef int64t[:] index_data
    cdef FastFileIteratorIndexDoublePtr fast_file_iterator
    cdef FastFileIteratorIndexVectorPtr fast_file_iterator_vector_ptr
    cdef vector[FastFileIteratorIndexVectorPtr] fast_file_iterator_vector_ptr_vector
    cdef int n_cols = 0, okay_to_fill = 0
    cdef int n_rasters = len(raster_path_band_list)
    churn_dir = os.path.join(target_working_directory)
    try:
        os.makedirs(churn_dir)
    except OSError:
        pass

    heapfile_list = []
    cdef double[:] max_sum_array = numpy.zeros(n_rasters)
    cdef double[:] max_proportion_list = numpy.zeros(n_rasters)
    cdef int raster_index

    dim_set = set()
    for raster_path_band in raster_path_band_list:
        n_cols, n_rows = pygeoprocessing.get_raster_info(
            raster_path_band[0])['raster_size']
        dim_set.add((n_cols, n_rows))

    if len(dim_set) > 1:
        error_message = f"dimensions don't match: {str(dim_set)}"
        LOGGER.error(error_message)
        raise RuntimeError(error_message)

    sum_list = []
    for raster_path_band in raster_path_band_list:
        sum_list.append(sum_raster(raster_path_band))

    raster_nodata_list = []
    normalized_raster_band_path_list = []
    normalized_nodata_list = []
    prop_nodata = -1
    # calculate normalized rasters of their total
    for (path, band_id), sum_val in zip(raster_path_band_list, sum_list):
        if sum_val > 0:
            raster_path_band_list.append((path, 1))
            raster_nodata_list.append(
                (pygeoprocessing.get_raster_info(path)[
                    'nodata'][band_id-1], 'raw'))
            proportional_path = os.path.join(
                churn_dir, f'prop_{os.path.basename(path)}')
            pygeoprocessing.raster_calculator([
                (path, 1), (sum_val, 'raw'),
                (pygeoprocessing.get_raster_info(path)['nodata'][band_id-1],
                 'raw'),
                (prop_nodata, 'raw')], normalize_op, proportional_path,
                gdal.GDT_Float64, prop_nodata)
            normalized_raster_band_path_list.append((proportional_path, 1))
            normalized_nodata_list.append((prop_nodata, 'raw'))
        else:
            normalized_raster_band_path_list.append((path, band_id))
            normalized_nodata_list.append(
                (pygeoprocessing.get_raster_info(path)['nodata'][0], 'raw'))

    normalized_sum_raster_path = os.path.join(churn_dir, 'prop_sum.tif')
    pygeoprocessing.raster_calculator(
        [*normalized_raster_band_path_list, *normalized_nodata_list,
         (prop_nodata, 'raw')],
        sum_rasters_op, normalized_sum_raster_path, gdal.GDT_Float64,
        prop_nodata)

    cdef long long valid_pixel_count = count_valid(
        (normalized_sum_raster_path, 1))

    # sort proportional and base rasters
    heapfile_directory_list = []
    for raster_index, raster_path_band in enumerate(
            [*raster_path_band_list, (normalized_sum_raster_path, 1)]):
        # sort the raster from high to low including pixel loc
        # calculate the sum along with it
        # calculate the target proportion based on that sum
        pixels_processed = 0
        last_update = time.time()

        raster_id = os.path.splitext(os.path.basename(raster_path_band[0]))[0]
        working_sort_directory = os.path.join(
            target_working_directory, raster_id)
        LOGGER.debug(working_sort_directory)
        heapfile_directory_list.append(working_sort_directory)
        try:
            os.makedirs(working_sort_directory)
        except OSError:
            pass

        raster_info = pygeoprocessing.get_raster_info(
            raster_path_band[0])
        nodata = raster_info['nodata'][raster_path_band[1]-1]
        n_pixels = numpy.prod(raster_info['raster_size'])
        sum_val = 0.0
        n_elements = 0
        file_index = 0
        n_cols = raster_info['raster_size'][0]

        fast_file_iterator_vector_ptr = \
            new vector[FastFileIteratorIndexDoublePtr]()
        for offset_data, block_data in pygeoprocessing.iterblocks(
                raster_path_band, largest_block=heap_buffer_size):
            pixels_processed += block_data.size
            if time.time() - last_update > 5.0:
                LOGGER.debug(
                    'data sort to %s heap %.2f%% complete',
                    raster_id, (100.*pixels_processed)/n_pixels)
                last_update = time.time()
            xx, yy = numpy.meshgrid(
                range(block_data.shape[1]), range(block_data.shape[0]))
            # Don't keep data that's nodata or zero (should never be negative
            # but don't do those either)
            valid_mask = (
                ~numpy.isclose(block_data, nodata) &
                ~numpy.isclose(block_data, 0) &
                (block_data > 0))
            # -1 to reverse sort from large to small
            base_data = -block_data[valid_mask]
            if base_data.size == 0:
                # skip if nothing valid
                continue

            flat_indexes = (
                (yy.astype(numpy.int64)+offset_data['yoff'])*n_cols +
                (xx.astype(numpy.int64)+offset_data['xoff']))[
                    valid_mask].flatten()
            sorted_data = numpy.diff(numpy.sort(base_data))
            min_eps = 1e-4 * numpy.min(sorted_data[sorted_data > 0])
            # add enough randomness that same values are not the same
            sort_args = numpy.argsort(
                base_data + min_eps*numpy.random.random(base_data.shape),
                axis=None)
            buffer_data = (base_data.flatten()[sort_args]).astype(numpy.double)
            index_data = flat_indexes[sort_args].astype(numpy.int64)
            sum_val += numpy.sum(buffer_data)
            n_elements += buffer_data.size
            file_path = os.path.join(
                working_sort_directory, '%d.dat' % file_index)
            index_path = os.path.join(
                working_sort_directory, 'index_%d.dat' % file_index)
            heapfile_list.append(file_path)

            fptr = fopen(bytes(file_path.encode()), "wb")
            fwrite(
                <double*>&buffer_data[0],
                sizeof(double), buffer_data.size, fptr)
            fclose(fptr)

            index_fptr = fopen(bytes(index_path.encode()), "wb")
            fwrite(
                <int64t*>&index_data[0],
                sizeof(int64t), index_data.size, index_fptr)
            fclose(index_fptr)

            file_index += 1

            fast_file_iterator = new FastFileIteratorIndex[double](
                (bytes(file_path.encode())),
                (bytes(index_path.encode())),
                ffi_buffer_size)

            # put on back and heapify
            deref(fast_file_iterator_vector_ptr).push_back(fast_file_iterator)
            push_heap(
                deref(fast_file_iterator_vector_ptr).begin(),
                deref(fast_file_iterator_vector_ptr).end(),
                FastFileIteratorIndexCompare[double])

        if raster_index < n_rasters:
            max_sum_array[raster_index] = sum_val
            max_proportion_list[raster_index] = (
                target_sum_list[raster_index] / sum_val)
            fast_file_iterator_vector_ptr_vector.push_back(
                fast_file_iterator_vector_ptr)

    # core algorithm, visit each pixel
    if target_suffix is not None:
        target_suffix = '_%s' % target_suffix
    else:
        target_suffix = ''
    mask_raster_path = os.path.join(
        target_working_directory, 'optimal_mask%s.tif' % target_suffix)
    cdef int mask_nodata = 0
    pygeoprocessing.new_raster_from_base(
        raster_path_band_list[0][0], mask_raster_path, gdal.GDT_Byte,
        [mask_nodata])
    cdef _ManagedRaster mask_managed_raster = _ManagedRaster(
        mask_raster_path, 1, 1)

    # n_rasters-1 because the first raster is a proportional raster that's
    # used as an optimization preconditioner
    cdef double[:] running_goal_sum_array = numpy.zeros(n_rasters)
    cdef double[:] active_val_array = numpy.zeros(n_rasters)
    cdef double[:] prop_to_meet_vals = max_proportion_list.copy()
    cdef double[:] target_sum_array = numpy.array(target_sum_list)
    cdef int i, max_prop_index, x, y
    cdef int64t active_index
    cdef double active_prop_to_meet
    cdef double threshold_prop

    cdef _ManagedRaster[:] managed_raster_array

    managed_raster_array = numpy.array([
        _ManagedRaster(
            raster_path_band_list[i][0],
            raster_path_band_list[i][1], 0) for i in range(n_rasters)])

    cdef long long count = 0
    # this sets the lower bound on the "highest temperature"/lowest iteration
    # of the simulated annealing
    # 0 index because that is the proportion sum index
    fast_file_iterator_vector_ptr = fast_file_iterator_vector_ptr_vector.back()
    fast_file_iterator_vector_ptr_vector.pop_back()
    # iterate through proportional sum using deterministic simulated annealing

    cdef long long pixel_set_in_preconditioner = 0

    # all but 1-preconditioner_weight will be determined by preconditioner
    cdef double precondition_threshold = 1.0 - preconditioner_weight
    while True:
        count += 1
        active_index = (
            deref(fast_file_iterator_vector_ptr).front().next())
        # update the heap
        pop_heap(
            deref(fast_file_iterator_vector_ptr).begin(),
            deref(fast_file_iterator_vector_ptr).end(),
            FastFileIteratorIndexCompare[double])
        if deref(fast_file_iterator_vector_ptr).back().size() > 0:
            push_heap(
                deref(fast_file_iterator_vector_ptr).begin(),
                deref(fast_file_iterator_vector_ptr).end(),
                FastFileIteratorIndexCompare[double])
        else:
            fast_file_iterator = deref(
                fast_file_iterator_vector_ptr).back()
            del fast_file_iterator
            deref(fast_file_iterator_vector_ptr).pop_back()

        # i don't think we should ever have this but check anyway
        if active_index == -1:
            LOGGER.error('got active index -1!!!')
            break

        x = active_index % n_cols
        y = active_index // n_cols
        # check that the pixel hasn't already been selected
        if mask_managed_raster.get(x, y) == mask_nodata:
            # check if any of the pools that are already full would be
            # additionally filled by selecting this pixel, if so, skip it
            okay_to_fill = 1
            for i in range(n_rasters):
                active_val = (<_ManagedRaster>managed_raster_array[i]).get(
                    x, y)
                # we could get a garbage area so check first
                if active_val > 0:
                    threshold_prop = precondition_threshold * (
                        <double>(count+1)/<double>(valid_pixel_count))
                    LOGGER.debug('threshold/prop to meet: %f %f', threshold_prop, prop_to_meet_vals[i])
                    if prop_to_meet_vals[i] <= threshold_prop:
                        okay_to_fill = 0
                        break
                    active_val_array[i] = active_val
                else:
                    active_val_array[i] = 0

            if okay_to_fill:
                for i in range(n_rasters):
                    running_goal_sum_array[i] += active_val_array[i]
                    prop_to_meet_vals[i] = (
                        target_sum_array[i] -
                        running_goal_sum_array[i]) / (
                            max_sum_array[i+1])
                mask_managed_raster.set(x, y, 1)
                pixel_set_in_preconditioner += 1
        else:
            # it's already set
            continue
        break

    del fast_file_iterator_vector_ptr

    # iterate remaining props to meet through individual targets
    count = 0
    cdef long long pixel_set_in_general = 0
    while True:
        max_prop_index = -1
        active_prop_to_meet = 0.0
        if count % 1000000 == 0:
            LOGGER.debug(count)
        for i in range(n_rasters):
            if (prop_to_meet_vals[i] > 0 and
                    (prop_to_meet_vals[i] > active_prop_to_meet)):
                max_prop_index = i
                active_prop_to_meet = prop_to_meet_vals[i]
        if max_prop_index == -1:
            LOGGER.debug('all targets met')
            break

        fast_file_iterator_vector_ptr = \
            fast_file_iterator_vector_ptr_vector[max_prop_index]
        while True:
            count += 1
            active_index = (
                deref(fast_file_iterator_vector_ptr).front().next())
            # update the heap
            pop_heap(
                deref(fast_file_iterator_vector_ptr).begin(),
                deref(fast_file_iterator_vector_ptr).end(),
                FastFileIteratorIndexCompare[double])
            if deref(fast_file_iterator_vector_ptr).back().size() > 0:
                push_heap(
                    deref(fast_file_iterator_vector_ptr).begin(),
                    deref(fast_file_iterator_vector_ptr).end(),
                    FastFileIteratorIndexCompare[double])
            else:
                fast_file_iterator = deref(
                    fast_file_iterator_vector_ptr).back()
                del fast_file_iterator
                deref(fast_file_iterator_vector_ptr).pop_back()

            # i don't think we should ever have this but check anyway
            if active_index == -1:
                LOGGER.debug('got active index -1!!!')
                break

            x = active_index % n_cols
            y = active_index // n_cols
            # check that the pixel hasn't already been selected
            if mask_managed_raster.get(x, y) == mask_nodata:
                mask_managed_raster.set(x, y, 1)
                pixel_set_in_general += 1
                # update all the pools
                for i in range(n_rasters):
                    active_val = (<_ManagedRaster>managed_raster_array[i]).get(
                        x, y)
                    # we could get a garbage area so check first
                    if active_val > 0:
                        running_goal_sum_array[i] += active_val
                        prop_to_meet_vals[i] = (
                            target_sum_array[i] -
                            running_goal_sum_array[i]) / (
                                max_sum_array[i])

                        # LOGGER.debug(
                        #     '%d, %f, %f, %s', active_index, active_val,
                        #     prop_to_meet_vals[i],
                        #     raster_path_band_list[i][0])
            else:
                # it's already set
                # LOGGER.debug('already set')
                continue
            break

    with open(os.path.join(
            target_working_directory, f'results{target_suffix}.csv'), 'w') as \
            results_file:
        results_file.write(
                'base raster,target sum,final sum,proportion beyond optimal\n')
        for i in range(n_rasters):
            results_file.write(
                '%s,%f,%f,%f\n' % (
                    os.path.basename(raster_path_band_list[i][0]),
                    target_sum_list[i],
                    running_goal_sum_array[i],
                    prop_to_meet_vals[i]))

    # free all the iterator memory
    while fast_file_iterator_vector_ptr_vector.size() > 0:
        fast_file_iterator_vector_ptr = \
            fast_file_iterator_vector_ptr_vector.back()
        fast_file_iterator_vector_ptr_vector.pop_back()

        while deref(fast_file_iterator_vector_ptr).size() > 0:
            fast_file_iterator = deref(fast_file_iterator_vector_ptr).back()
            del fast_file_iterator
            deref(fast_file_iterator_vector_ptr).pop_back()

    LOGGER.debug(
        'pixels set in preconditioner vs general: %d %d',
        pixel_set_in_preconditioner, pixel_set_in_general)

    # delete all the heap files
    for heapfile_dir in heapfile_directory_list:
        try:
            shutil.rmtree(heapfile_dir)
        except OSError:
            # you never know if this might fail!
            LOGGER.warning('unable to remove %s', heapfile_dir)
