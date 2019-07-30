# coding=UTF-8
"""A collection of GDAL dataset and raster utilities."""
import logging

from osgeo import gdal
import sympy
import numpy
import numpy.ma
from . import geoprocessing

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())  # silence logging by default


def evaluate_raster_calculator_expression(
        expression_str, symbol_to_path_band_map, target_nodata,
        target_raster_path, default_nan=None, default_inf=None):
    """Evaluate the arithmetic expression of rasters.

    Evaluate the symbolic arithmetic expression in `expression_str` where the
    symbols represent equally sized GIS rasters. With the following rules:

        * any nodata pixels in a raster will cause the entire pixel stack
          to be `target_nodata`. If `target_nodata` is None, this will be 0.
        * any calculations the result in NaN or inf values will be replaced
          by the corresponding values in `default_nan` and `default_inf`.
          If either of these are not defined an NaN or inf result will cause
          a ValueError exception to be raised.
        * valid arithmetic expressions are those available in the `sympy`
          library and include: +, -, *, /, <, <=, >, >=, !=, &, and |.

    Parameters:
        expression_str (str): a valid arithmetic expression whose variables
            are defined in `symbol_to_path_band_map`.
        symbol_to_path_band_map (dict): a dict of symbol/(path, band) pairs to
            indicate which symbol maps to which raster and corresponding
            band. All symbol names correspond to
            symbols in `expression_str`. Ex:
                expression_str = '2*x+b'
                symbol_to_path_band_mapband_ = {
                    'x': (path_to_x_raster, 1),
                    'b': (path_to_b_raster, 1)
                }
            All rasters represented in this structure must have the same
            raster size.
        target_nodata (numeric): desired nodata value for
            `target_raster_path`.
        target_raster_path (str): path to the raster that is created by
            `expression_str`.
        default_nan (numeric): if a calculation results in an NaN that
            value is replaces with this value. A ValueError exception is
            raised if this case occurs and `default_nan` is None.
        default_inf (numeric): if a calculation results in an +/- inf
            that value is replaced with this value. A ValueError exception is
            raised if this case occurs and `default_nan` is None.

    Returns:
        None.

    """
    LOGGER.debug('evaluating: %s', expression_str)
    symbol_list, raster_path_band_list = zip(*symbol_to_path_band_map.items())
    raster_op = sympy.lambdify(symbol_list, expression_str, 'numpy')

    geoprocessing.raster_calculator(
        [path_band for path_band in raster_path_band_list] +
        [(geoprocessing.get_raster_info(
            path_band[0])['nodata'][path_band[1]-1], 'raw')
         for path_band in raster_path_band_list] + [
            (raster_op, 'raw'), (target_nodata, 'raw'), (default_nan, 'raw'),
            (default_inf, 'raw')],
        _generic_raster_op, target_raster_path, gdal.GDT_Float32,
        target_nodata)


def _generic_raster_op(*arg_list):
    """General raster array operation with well conditioned args.

    Parameters:
        arg_list (list): a list of length 2*n+4 defined as:
            [array_0, ... array_n, nodata_0, ... nodata_n,
             func, target_nodata, default_nan, default_inf]

            Where `func` is a function that takes 2*n elements. The first
            `n` elements are `numpy.ndarrays` and the second set of `n`
            elements are the corresponding nodata for those arrays. T

            `target_noata` is the result of an element in `func` if any of the
            array values that would produce the result contain a nodata value.

            `default_nan` and `default_inf` is the value that should be
            replaced if the result of applying `func` to its arguments results
            in an `numpy.nan` or `numpy.inf` value. A ValueError exception is
            raised if a `numpy.nan` or `numpy.inf` is produced by `func` but
            the corresponding `default_*` argument is `None`.

    Returns:
        func applied to a masked version of array_0, ... array_n where only
        valid nodata values in the raster stack are used. Otherwise the target
        pixels are set to target_nodata.

    """
    n = int((len(arg_list)-4) // 2)
    array_list = arg_list[0:n]
    target_dtype = numpy.result_type(*[x.dtype for x in array_list])
    result = numpy.empty(arg_list[0].shape, dtype=target_dtype)
    nodata_list = arg_list[n:2*n]
    func = arg_list[2*n]
    target_nodata = arg_list[2*n+1]
    default_nan = arg_list[2*n+2]
    default_inf = arg_list[2*n+3]
    if target_nodata is not None:
        result[:] = target_nodata
    nodata_present = any([x is not None for x in nodata_list])
    valid_mask = None
    if nodata_present:
        valid_mask = ~numpy.logical_or.reduce(
            [numpy.isclose(array, nodata)
             for array, nodata in zip(array_list, nodata_list)
             if nodata is not None])
        func_result = func(*[array[valid_mask] for array in array_list])
    else:
        # there's no nodata values to mask so operate directly
        func_result = func(*array_list)

    if nodata_present:
        result[valid_mask] = func_result
    else:
        result[:] = func_result

    is_nan_mask = numpy.isnan(result)
    if is_nan_mask.any():
        if default_nan:
            result[is_nan_mask] = default_nan
        else:
            raise ValueError(
                'Encountered NaN in calculation but `default_nan` is None.')

    is_inf_mask = numpy.isinf(result)
    if is_inf_mask.any():
        if default_inf:
            result[is_inf_mask] = default_inf
        else:
            raise ValueError(
                'Encountered inf in calculation but `default_inf` is None.')

    return result
