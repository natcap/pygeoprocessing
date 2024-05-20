# coding=UTF-8
"""Module to hold symbolic PyGeoprocessing utilities."""
import ast
import logging

from . import geoprocessing
from .geoprocessing import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
import numpy

LOGGER = logging.getLogger(__name__)


def evaluate_raster_calculator_expression(
        expression, symbol_to_path_band_map, target_nodata,
        target_raster_path, default_nan=None, default_inf=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Evaluate the arithmetic expression of rasters.

    Evaluate the symbolic arithmetic expression in ``expression`` where the
    symbols represent equally sized GIS rasters. With the following rules:

    * any nodata pixels in a raster will cause the entire pixel stack to be
      ``target_nodata``. If ``target_nodata`` is None, this will be 0.
    * any calculations the result in NaN or inf values will be replaced by the
      corresponding values in ``default_nan`` and ``default_inf``.  If either of
      these are not defined an NaN or inf result will cause a ValueError
      exception to be raised.
    * the following arithmetic operators are available:
      ``+``, ``-``, ``*``, ``/``, ``<``, ``<=``, ``>``, ``>=``, ``!=``, ``&``,
      and ``|``.

    Args:
        expression (str): a valid arithmetic expression whose variables
            are defined in ``symbol_to_path_band_map``.
        symbol_to_path_band_map (dict): a dict of symbol/(path, band) pairs to
            indicate which symbol maps to which raster and corresponding
            band. All symbol names correspond to
            symbols in ``expression``. Example::

                expression = '2*x+b'
                symbol_to_path_band_map = {
                    'x': (path_to_x_raster, 1),
                    'b': (path_to_b_raster, 1)
                }

            All rasters represented in this structure must have the same
            raster size.
        target_nodata (float): desired nodata value for
            ``target_raster_path``.
        target_raster_path (str): path to the raster that is created by
            ``expression``.
        default_nan (float): if a calculation results in an NaN that
            value is replaces with this value. A ValueError exception is
            raised if this case occurs and ``default_nan`` is None.
        default_inf (float): if a calculation results in an +/- inf
            that value is replaced with this value. A ValueError exception is
            raised if this case occurs and ``default_nan`` is None.

    Returns:
        None

    """
    # its a common error to pass something other than a string for
    # ``expression`` but the resulting error is obscure, so test for that and
    # make a helpful error
    if not isinstance(expression, str):
        raise ValueError(
            "Expected type `str` for `expression` but instead got %s", str(
                type(expression)))

    # remove any raster bands that don't have corresponding symbols in the
    # expression
    active_symbols = set()
    for tree_node in ast.walk(ast.parse(expression)):
        if isinstance(tree_node, ast.Name):
            active_symbols.add(tree_node.id)

    LOGGER.debug(
        'evaluating: %s\nactive symbols: %s\n',
        expression, sorted(active_symbols))
    symbol_list, raster_path_band_list = zip(*[
        (symbol, raster_path_band) for symbol, raster_path_band in
        sorted(symbol_to_path_band_map.items()) if symbol in active_symbols])

    missing_symbols = set(active_symbols) - set(symbol_list)
    if missing_symbols:
        raise ValueError(
            'The variables %s are defined in the expression but are not in '
            'symbol_to_path_band_map' % ', '.join(sorted(missing_symbols)))

    raster_path_band_const_list = (
        [path_band for path_band in raster_path_band_list] +
        [(geoprocessing.get_raster_info(
            path_band[0])['nodata'][path_band[1]-1], 'raw')
         for path_band in raster_path_band_list] + [
            (expression, 'raw'), (target_nodata, 'raw'), (default_nan, 'raw'),
            (default_inf, 'raw'), (symbol_list, 'raw')])

    # Determine the target gdal type by gathering all the numpy types to
    # determine what the result type would be if they were all applied in
    # an operation.
    target_numpy_type = numpy.result_type(*[
        geoprocessing.get_raster_info(path)['numpy_type']
        for path, band_id in raster_path_band_const_list
        if isinstance(band_id, int)])

    target_gdal_type, type_creation_options = (
        geoprocessing._numpy_to_gdal_type(target_numpy_type))

    driver, creation_options = raster_driver_creation_tuple
    creation_options = list(creation_options) + type_creation_options

    geoprocessing.raster_calculator(
        raster_path_band_const_list, _generic_raster_op, target_raster_path,
        target_gdal_type, target_nodata,
        raster_driver_creation_tuple=(driver, creation_options))


def _generic_raster_op(*arg_list):
    """General raster array operation with well conditioned args.

    Parameters:
        arg_list (list): a list of length 2*n+5 defined as:
            [array_0, ... array_n, nodata_0, ... nodata_n,
             expression, target_nodata, default_nan, default_inf, kwarg_names]

            Where ``expression`` is a string expression to be evaluated by
            ``eval`` that takes ``n`` ``numpy.ndarray``elements.

            ``target_nodata`` is the result of an element in ``expression`` if
            any of the array values that would produce the result contain a
            nodata value.

            ``default_nan`` and ``default_inf`` is the value that should be
            replaced if the result of applying ``expression`` to its arguments
            results in an ``numpy.nan`` or ``numpy.inf`` value. A ValueError
            exception is raised if a ``numpy.nan`` or ``numpy.inf`` is
            produced by ``func`` but the corresponding ``default_*`` argument
            is ``None``.

            ``kwarg_names`` is a list of the variable names present in
            ``expression`` in the same order as the incoming numpy arrays.

    Returns:
        func applied to a masked version of array_0, ... array_n where only
        valid non-nodata values in the raster stack are used. Otherwise the
        target pixels are set to target_nodata.

    """
    n = int((len(arg_list)-4) // 2)
    array_list = arg_list[0:n]
    target_dtype = numpy.result_type(*[x.dtype for x in array_list])
    result = numpy.empty(arg_list[0].shape, dtype=target_dtype)
    nodata_list = arg_list[n:2*n]
    expression = arg_list[2*n]
    target_nodata = arg_list[2*n+1]
    default_nan = arg_list[2*n+2]
    default_inf = arg_list[2*n+3]
    kwarg_names = arg_list[2*n+4]
    nodata_present = any([x is not None for x in nodata_list])
    if target_nodata is not None:
        result[:] = target_nodata

    valid_mask = None
    if nodata_present:
        valid_mask = ~numpy.logical_or.reduce(
            [numpy.isclose(array, nodata)
             for array, nodata in zip(array_list, nodata_list)
             if nodata is not None])
        if not valid_mask.all() and target_nodata is None:
            raise ValueError(
                "`target_nodata` is undefined (None) but there are nodata "
                "values present in the input rasters.")
        user_symbols = {symbol: array[valid_mask] for (symbol, array) in
                        zip(kwarg_names, array_list)}
    else:
        # there's no nodata values to mask so operate directly
        user_symbols = dict(zip(kwarg_names, array_list))

    # They say ``eval`` is dangerous, and it honestly probably is.
    # As far as we can tell, the benefits of being able to evaluate these sorts
    # of expressions will outweight the risks and, as always, folks shouldn't
    # be running code they don't trust.
    func_result = eval(expression, {}, user_symbols)

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
