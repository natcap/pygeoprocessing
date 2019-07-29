# coding=UTF-8
"""A collection of GDAL dataset and raster utilities."""
import logging
import tempfile
import shutil

from osgeo import gdal
import sympy
import numpy
import numpy.ma
from . import geoprocessing

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())  # silence logging by default


def evaluate_raster_calculator_expression(
        expression_str, symbol_to_path_band_map, target_nodata,
        target_raster_path, churn_dir=None, target_sr_wkt=None,
        target_pixel_size=None, resample_method=None,
        default_nan=None, default_inf=None):
    """Evaluate the arithmetic expression of rasters.

    Evaluate the symbolic arithmetic expression in `expression_str

    Any nodata pixels in the path list are cause the corresponding pixel to
    be a nodata value.

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
        churn_dir (str): path to a temporary "churn" directory. If not
            specified uses a tempfile.mkdtemp.
        default_nan (numeric): if a calculation results in an NaN that
            value is replaces with this value.

    Returns:
        None.

    """
    LOGGER.debug('evaluating: %s', expression_str)
    delete_churn_dir = False
    if churn_dir is None:
        churn_dir = tempfile.mkdtemp()
        delete_churn_dir = True
    symbol_list, raster_path_band_list = zip(*symbol_to_path_band_map.items())
    raster_op = sympy.lambdify(symbol_list, expression_str, 'numpy')

    geoprocessing.raster_calculator(
        [path_band for path_band in raster_path_band_list] +
        [(geoprocessing.get_raster_info(
            path_band[0])['nodata'][path_band[1]-1], 'raw')
         for path_band in raster_path_band_list] + [
            (raster_op, 'raw'), (target_nodata, 'raw'), (default_nan, 'raw')],
        _general_raster_calculator_op, target_raster_path, gdal.GDT_Float32,
        target_nodata)
    if delete_churn_dir:
        shutil.rmtree(churn_dir)


def _general_raster_calculator_op(*arg_list):
    """General raster operation with well conditioned args.

    Parameters:
        arg_list (list): list is 2*n+3 length long laid out as:
            array_0, ... array_n, nodata_0, ... nodata_n,
            op, target_nodata, invalid_value_replacement

            The first element `op` is an operation that takes n elements which
            are numpy.ndarrays. The second n elements are the nodata values
            for the corresponding ndarrays. The second to value is the target
            nodata value and the final value is the value to set if an NaN
            occurs in the calculations.

    Returns:
        op applied to a masked version of array_0, ... array_n where only
        valid nodata values in the raster stack are used. Otherwise the target
        pixels are set to target_nodata.

    """
    n = int((len(arg_list)-3) // 2)
    result = numpy.empty(arg_list[0].shape, dtype=numpy.float32)
    array_list = arg_list[0:n]
    nodata_list = arg_list[n:2*n]
    op = arg_list[2*n]
    target_nodata = arg_list[2*n+1]
    invalid_value_replacement = arg_list[2*n+2]
    result[:] = target_nodata
    if any([x is not None for x in nodata_list]):
        valid_mask = ~numpy.logical_or.reduce(
            [numpy.isclose(array, nodata)
             for array, nodata in zip(array_list, nodata_list)
             if nodata is not None])
        result[valid_mask] = op(*[array[valid_mask] for array in array_list])
    else:
        # there's no nodata values to mask so operate directly
        result[:] = op(*array_list)

    result[numpy.isnan(result) | numpy.isinf(result)] = (
        invalid_value_replacement)
    return result
