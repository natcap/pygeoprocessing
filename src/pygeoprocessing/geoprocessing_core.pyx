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
@cython.binding(True)
def distance_transform_edt(base_mask_raster_path_band, target_distance_path):
    """Calculate the Euclidean distance transform.

    Parameters:
        base_mask_raster_path_band (tuple): a (path, band index) tuple to
            calculate value from non-zero valued pixels.

        target_distance_path (string): a path to a raster created by this
        function with same dimensions and projection as base_mask_path where
        all non-zero values of base_mask_path are equal to the euclidean
        distance to the closest 0 pixel.

    Returns:
        None."""
    file_handle, base_mask_path = tempfile.mkstemp()
    os.close(file_handle)
    nodata_base_mask = 255
    base_raster_info = pygeoprocessing.get_raster_info(
        base_mask_raster_path_band[0])
    base_nodata = base_raster_info['nodata'][
        base_mask_raster_path_band[1]-1]

    def _mask_op(base_array):
        """Convert base_array to 1 if >0, 0 if == 0 or nodata."""
        result = numpy.empty(base_array.shape, dtype=numpy.int8)
        result[:] = nodata_base_mask
        valid_mask = base_array != base_nodata
        result[valid_mask] = base_array[valid_mask] != 0
        return result

    pygeoprocessing.raster_calculator(
        [base_mask_raster_path_band], _mask_op, base_mask_path,
        gdal.GDT_Byte, nodata_base_mask, calc_raster_stats=False)

    base_mask_raster = gdal.Open(base_mask_path)
    base_mask_band = base_mask_raster.GetRasterBand(1)

    cdef int n_cols = base_mask_raster.RasterXSize
    cdef int n_rows = base_mask_raster.RasterYSize

    # create a transposed g function
    file_handle, g_path = tempfile.mkstemp()
    os.close(file_handle)
    g_path = 'g.tif'
    raster_info = pygeoprocessing.get_raster_info(
        base_mask_raster_path_band[0])
    nodata = raster_info['nodata'][base_mask_raster_path_band[1]-1]
    nodata_out = 255
    pygeoprocessing.new_raster_from_base(
        base_mask_raster_path_band[0], g_path, gdal.GDT_Int32, [-1],
        fill_value_list=None)
    g_raster = gdal.Open(g_path, gdal.GA_Update)
    g_band = g_raster.GetRasterBand(1)

    cdef int g_nodata = -1
    numerical_inf = (
        raster_info['raster_size'][0] + raster_info['raster_size'][1])
    block_index = {}
    last_g_row = {}
    # scan 1
    for block_offset, mask_block in pygeoprocessing.iterblocks(
            base_mask_raster_path_band[0]):
        block_index[(block_offset['yoff'], block_offset['xoff'])] = (
            block_offset)
        g_block = numpy.empty(mask_block.shape, dtype=numpy.int32)
        if block_offset['yoff'] == 0:
            # base case
            g_block[0, :] = (mask_block[0, :] == 0) * numerical_inf
        else:
            index = (block_offset['xoff'], block_offset['yoff'])
            g_prev_row = last_g_row[index]
            del last_g_row[index]

            active_mask = mask_block[0, :] == 1
            g_block[0, active_mask] = 0
            g_block[0, ~active_mask] = (
                g_prev_row[0, ~active_mask] + 1)
            del g_prev_row

        for row_index in xrange(1, block_offset['win_ysize']):
            active_mask = mask_block[row_index, :] == 1
            g_block[row_index, active_mask] = 0
            g_block[row_index, ~active_mask] = (
                g_block[row_index-1, ~active_mask] + 1)

        g_band.WriteArray(
            g_block, xoff=block_offset['xoff'], yoff=block_offset['yoff'])
        last_g_row[(
            block_offset['xoff'],
            block_offset['yoff'] + block_offset['win_ysize'])] = (
                g_block[-1, :])
    del last_g_row

    # scan 2
    base_mask_raster = gdal.Open(g_path)
    base_mask_band = base_mask_raster.GetRasterBand(1)
    # go in lowest blocks to highest
    last_bottom_g_row = {}
    for index in reversed(sorted(block_index)):
        block_offset = block_index[index]
        mask_block = base_mask_band.ReadAsArray(**block_offset)
        g_block = g_band.ReadAsArray(**block_offset)

        # if this is the bottom block, have a special case, otherwise load
        # the previous row
        if ((block_offset['yoff'] + block_offset['win_ysize']) !=
                raster_info['raster_size'][1]):
            index = (block_offset['xoff'], block_offset['yoff'])
            g_prev_row = last_bottom_g_row[index]
            del last_bottom_g_row[index]
            active_mask = (
                g_prev_row[0, :] < g_block[block_offset['win_ysize']-1, :])
            g_block[block_offset['win_ysize']-1, active_mask] = (
                1 + g_prev_row[0, active_mask])

        for row_index in reversed(xrange(
                block_offset['win_ysize'] - 1)):
            active_mask = g_block[row_index+1, :] < g_block[row_index, :]
            g_block[row_index, active_mask] = (
                1 + g_block[row_index+1, active_mask])
        g_band.WriteArray(
            g_block, xoff=block_offset['xoff'], yoff=block_offset['yoff'])
        last_bottom_g_row[(
            block_offset['xoff'],
            block_offset['yoff']-block_offset['win_ysize'])] = g_block[0, :]
    del last_bottom_g_row
    g_band.FlushCache()
    driver = gdal.GetDriverByName('GTiff')

    cdef float output_nodata = -1.0
    pygeoprocessing.new_raster_from_base(
        base_mask_raster_path_band[0], target_distance_path.encode('utf-8'),
        gdal.GDT_Float64, [output_nodata], fill_value_list=None)
    target_distance_raster = gdal.Open(target_distance_path, gdal.GA_Update)
    target_distance_band = target_distance_raster.GetRasterBand(1)

    LOGGER.info('Distance Transform Phase 2')
    cdef numpy.ndarray[numpy.int64_t, ndim=2] s_array
    cdef numpy.ndarray[numpy.int64_t, ndim=2] t_array
    cdef numpy.ndarray[numpy.float64_t, ndim=2] dt

    cdef int win_ysize

    cdef double current_time, last_time
    last_time = time.time()

    s_array = numpy.zeros((1, n_cols), dtype=numpy.int64)
    t_array = numpy.zeros((1, n_cols), dtype=numpy.int64)
    dt = numpy.empty((1, n_cols), dtype=numpy.float64)
    for row_index in xrange(n_rows):
        g_array = g_band.ReadAsArray(
            xoff=0, yoff=row_index, win_xsize=n_cols,
            win_ysize=1)
        q_index = 0
        s_array[0, 0] = 0
        t_array[0, 0] = 0
        for u_index in xrange(1, n_cols):
            while (q_index >= 0 and
                _f(t_array[0, q_index], s_array[0, q_index],
                    g_array[0, s_array[0, q_index]]) >
                _f(t_array[0, q_index], u_index, g_array[0, u_index])):
                q_index -= 1
            if q_index < 0:
               q_index = 0
               s_array[0, 0] = u_index
            else:
                w = 1 + _sep(
                    s_array[0, q_index], u_index, g_array[0, u_index],
                    g_array[0, s_array[0, q_index]])
                if w < n_cols:
                    q_index += 1
                    s_array[0, q_index] = u_index
                    t_array[0, q_index] = w

        for u_index in xrange(n_cols-1, -1, -1):
            dt[0, u_index] = _f(
                u_index, s_array[0, q_index],
                g_array[0, s_array[0, q_index]])
            if u_index == t_array[0, q_index]:
                q_index -= 1

        dt = numpy.sqrt(dt)
        dt[g_array == base_nodata] = output_nodata
        target_distance_band.WriteArray(dt, xoff=0, yoff=row_index)

    target_distance_band.FlushCache()
    gdal.Dataset.__swig_destroy__(target_distance_raster)
    gdal.Dataset.__swig_destroy__(base_mask_raster)
    gdal.Dataset.__swig_destroy__(g_raster)
    try:
        pass #os.remove(g_path)
    except OSError:
        LOGGER.warn("couldn't remove file %s" % g_path)


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
