# cython: profile=True
# cython: linetrace=True
import os
import tempfile
import logging
import time
import sys
import traceback

cimport numpy
import numpy
cimport cython
from libcpp.map cimport map

from libc.math cimport sqrt
from libc.math cimport exp
from libc.math cimport ceil

from osgeo import gdal
import pygeoprocessing

_DEFAULT_GTIFF_CREATION_OPTIONS = ('TILED=YES', 'BIGTIFF=IF_SAFER')

LOGGER = logging.getLogger('geoprocessing_core')


cdef long long _f(long long x, long long i, long long gi):
    return (x-i)*(x-i)+ gi*gi


@cython.cdivision(True)
cdef long long _sep(long long i, long long u, long long gu, long long gi):
    return (u*u - i*i + gu*gu - gi*gi) / (2*(u-i))


#@cython.boundscheck(False)
def distance_transform_edt(input_mask_uri, output_distance_uri):
    """Calculate the Euclidean distance transform on input_mask_uri and output
        the result into an output raster

        input_mask_uri - a gdal raster to calculate distance from the 0 value
            pixels

        output_distance_uri - will make a float raster w/ same dimensions and
            projection as input_mask_uri where all non-zero values of
            input_mask_uri are equal to the euclidean distance to the closest
            0 pixel.

        returns nothing"""

    input_mask_ds = gdal.Open(input_mask_uri)
    input_mask_band = input_mask_ds.GetRasterBand(1)
    cdef int n_cols = input_mask_ds.RasterXSize
    cdef int n_rows = input_mask_ds.RasterYSize
    cdef int block_size = input_mask_band.GetBlockSize()[0]
    cdef int input_nodata = input_mask_band.GetNoDataValue()

    #create a transposed g function
    file_handle, g_dataset_uri = tempfile.mkstemp()
    os.close(file_handle)
    cdef int g_nodata = -1

    input_projection = input_mask_ds.GetProjection()
    input_geotransform = input_mask_ds.GetGeoTransform()
    driver = gdal.GetDriverByName('GTiff')
    #invert the rows and columns since it's a transpose
    g_dataset = driver.Create(
        g_dataset_uri.encode('utf-8'), n_cols, n_rows, 1, gdal.GDT_Int32,
        options=['TILED=YES', 'BLOCKXSIZE=%d' % block_size, 'BLOCKYSIZE=%d' % block_size])

    g_dataset.SetProjection(input_projection)
    g_dataset.SetGeoTransform(input_geotransform)
    g_band = g_dataset.GetRasterBand(1)
    g_band.SetNoDataValue(g_nodata)

    cdef float output_nodata = -1.0
    output_dataset = driver.Create(
        output_distance_uri.encode('utf-8'), n_cols, n_rows, 1,
        gdal.GDT_Float64, options=['TILED=YES', 'BLOCKXSIZE=%d' % block_size,
        'BLOCKYSIZE=%d' % block_size])
    output_dataset.SetProjection(input_projection)
    output_dataset.SetGeoTransform(input_geotransform)
    output_band = output_dataset.GetRasterBand(1)
    output_band.SetNoDataValue(output_nodata)

    #the euclidan distance will be less than this
    cdef int numerical_inf = n_cols + n_rows

    LOGGER.info('Distance Transform Phase 1')
    output_blocksize = output_band.GetBlockSize()
    if output_blocksize[0] != block_size or output_blocksize[1] != block_size:
        raise Exception(
            "Output blocksize should be %d,%d, instead it's %d,%d" % (
                block_size, block_size, output_blocksize[0], output_blocksize[1]))

    #phase one, calculate column G(x,y)

    cdef numpy.ndarray[numpy.int32_t, ndim=2] g_array
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] b_array

    cdef int col_index, row_index, q_index, u_index
    cdef long long w
    cdef int n_col_blocks = int(numpy.ceil(n_cols/float(block_size)))
    cdef int col_block_index, local_col_index, win_xsize
    cdef double current_time, last_time
    last_time = time.time()
    for col_block_index in xrange(n_col_blocks):
        current_time = time.time()
        if current_time - last_time > 5.0:
            LOGGER.info(
                'Distance transform phase 1 %.2f%% complete' %
                (col_block_index/float(n_col_blocks)*100.0))
            last_time = current_time
        local_col_index = col_block_index * block_size
        if n_cols - local_col_index < block_size:
            win_xsize = n_cols - local_col_index
        else:
            win_xsize = block_size
        b_array = input_mask_band.ReadAsArray(
            xoff=local_col_index, yoff=0, win_xsize=win_xsize,
            win_ysize=n_rows)
        g_array = numpy.empty((n_rows, win_xsize), dtype=numpy.int32)

        #initalize the first element to either be infinate distance, or zero if it's a blob
        for col_index in xrange(win_xsize):
            if b_array[0, col_index] and b_array[0, col_index] != input_nodata:
                g_array[0, col_index] = 0
            else:
                g_array[0, col_index] = numerical_inf

            #pass 1 go down
            for row_index in xrange(1, n_rows):
                if b_array[row_index, col_index] and b_array[row_index, col_index] != input_nodata:
                    g_array[row_index, col_index] = 0
                else:
                    g_array[row_index, col_index] = (
                        1 + g_array[row_index - 1, col_index])

            #pass 2 come back up
            for row_index in xrange(n_rows-2, -1, -1):
                if (g_array[row_index + 1, col_index] <
                    g_array[row_index, col_index]):
                    g_array[row_index, col_index] = (
                        1 + g_array[row_index + 1, col_index])
        g_band.WriteArray(
            g_array, xoff=local_col_index, yoff=0)

    g_band.FlushCache()
    LOGGER.info('Distance Transform Phase 2')
    cdef numpy.ndarray[numpy.int64_t, ndim=2] s_array
    cdef numpy.ndarray[numpy.int64_t, ndim=2] t_array
    cdef numpy.ndarray[numpy.float64_t, ndim=2] dt


    cdef int n_row_blocks = int(numpy.ceil(n_rows/float(block_size)))
    cdef int row_block_index, local_row_index, win_ysize

    for row_block_index in xrange(n_row_blocks):
        current_time = time.time()
        if current_time - last_time > 5.0:
            LOGGER.info(
                'Distance transform phase 2 %.2f%% complete' %
                (row_block_index/float(n_row_blocks)*100.0))
            last_time = current_time

        local_row_index = row_block_index * block_size
        if n_rows - local_row_index < block_size:
            win_ysize = n_rows - local_row_index
        else:
            win_ysize = block_size

        g_array = g_band.ReadAsArray(
            xoff=0, yoff=local_row_index, win_xsize=n_cols,
            win_ysize=win_ysize)

        s_array = numpy.zeros((win_ysize, n_cols), dtype=numpy.int64)
        t_array = numpy.zeros((win_ysize, n_cols), dtype=numpy.int64)
        dt = numpy.empty((win_ysize, n_cols), dtype=numpy.float64)

        for row_index in xrange(win_ysize):
            q_index = 0
            s_array[row_index, 0] = 0
            t_array[row_index, 0] = 0
            for u_index in xrange(1, n_cols):
                while (q_index >= 0 and
                    _f(t_array[row_index, q_index], s_array[row_index, q_index],
                        g_array[row_index, s_array[row_index, q_index]]) >
                    _f(t_array[row_index, q_index], u_index, g_array[row_index, u_index])):
                    q_index -= 1
                if q_index < 0:
                   q_index = 0
                   s_array[row_index, 0] = u_index
                else:
                    w = 1 + _sep(
                        s_array[row_index, q_index], u_index, g_array[row_index, u_index],
                        g_array[row_index, s_array[row_index, q_index]])
                    if w < n_cols:
                        q_index += 1
                        s_array[row_index, q_index] = u_index
                        t_array[row_index, q_index] = w

            for u_index in xrange(n_cols-1, -1, -1):
                dt[row_index, u_index] = _f(
                    u_index, s_array[row_index, q_index],
                    g_array[row_index, s_array[row_index, q_index]])
                if u_index == t_array[row_index, q_index]:
                    q_index -= 1

        b_array = input_mask_band.ReadAsArray(
            xoff=0, yoff=local_row_index, win_xsize=n_cols,
            win_ysize=win_ysize)

        dt = numpy.sqrt(dt)
        dt[b_array == input_nodata] = output_nodata
        output_band.WriteArray(dt, xoff=0, yoff=local_row_index)

    output_band.FlushCache()
    output_band = None
    gdal.Dataset.__swig_destroy__(output_dataset)
    output_dataset = None
    input_mask_band = None
    gdal.Dataset.__swig_destroy__(input_mask_ds)
    input_mask_ds = None
    g_band = None
    gdal.Dataset.__swig_destroy__(g_dataset)
    g_dataset = None
    try:
        os.remove(g_dataset_uri)
    except OSError:
        LOGGER.warn("couldn't remove file %s" % g_dataset_uri)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def calculate_slope(
        dem_raster_path_band, target_slope_path,
        gtiff_creation_options=_DEFAULT_GTIFF_CREATION_OPTIONS):
    """Create a percent slope raster from DEM raster.

    Base algorithm is from Zevenbergen & Thorne "Quantiative Analysis of Land
    Surface Topgraphy" 1987 although it has been modified to include the
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
        dem_raster_path_band (string): a path/band tuple to a raster of height
            values. (path_to_raster, band_index)
        target_slope_path (string): path to target slope raster; will be a
            32 bit float GeoTIFF of same size/projection as calculate slope
            with units of percent slope.
        gtiff_creation_options (list or tuple): list of strings that will be
            passed as GDAL "dataset" creation options to the GTIFF driver.

    Returns:
        None
    """
    cdef numpy.npy_float64 a, b, c, d, e, f, g, h, i, dem_nodata, z
    cdef numpy.npy_float64 x_cell_size, y_cell_size,
    cdef numpy.npy_float64 dzdx_accumulator, dzdy_accumulator
    cdef int row_index, col_index, n_rows, n_cols,
    cdef int x_denom_factor, y_denom_factor, win_xsize, win_ysize
    cdef numpy.ndarray[numpy.npy_float64, ndim=2] dem_array
    cdef numpy.ndarray[numpy.npy_float64, ndim=2] slope_array
    cdef numpy.ndarray[numpy.npy_float64, ndim=2] dzdx_array
    cdef numpy.ndarray[numpy.npy_float64, ndim=2] dzdy_array

    dem_raster = gdal.Open(dem_raster_path_band[0])
    dem_band = dem_raster.GetRasterBand(dem_raster_path_band[1])
    dem_info = pygeoprocessing.get_raster_info(dem_raster_path_band[0])
    dem_nodata = dem_info['nodata'][0]
    x_cell_size, y_cell_size = dem_info['pixel_size']
    n_cols, n_rows = dem_info['raster_size']
    cdef numpy.npy_float64 slope_nodata = numpy.finfo(numpy.float32).min
    pygeoprocessing.new_raster_from_base(
        dem_raster_path_band[0], target_slope_path, gdal.GDT_Float32,
        [slope_nodata], fill_value_list=[float(slope_nodata)],
        gtiff_creation_options=gtiff_creation_options)
    target_slope_raster = gdal.Open(target_slope_path, gdal.GA_Update)
    target_slope_band = target_slope_raster.GetRasterBand(1)

    for block_offset in pygeoprocessing.iterblocks(
            dem_raster_path_band[0], offset_only=True):
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

        for row_index in xrange(1, win_ysize+1):
            for col_index in xrange(1, win_xsize+1):
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
