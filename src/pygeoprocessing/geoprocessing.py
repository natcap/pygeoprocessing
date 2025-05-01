# coding=UTF-8
"""A collection of raster and vector algorithms and utilities."""
import collections
import functools
import logging
import math
import os
import pprint
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import warnings

import numpy
import numpy.ma
import rtree
import scipy.interpolate
import scipy.ndimage
import scipy.signal
import scipy.signal.signaltools
import scipy.sparse
import shapely.ops
import shapely.prepared
import shapely.wkb
from osgeo import gdal
from osgeo import gdal_array
from osgeo import gdalconst
from osgeo import ogr
from osgeo import osr

from . import geoprocessing_core
from .geoprocessing_core import DEFAULT_CREATION_OPTIONS
from .geoprocessing_core import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
from .geoprocessing_core import DEFAULT_OSR_AXIS_MAPPING_STRATEGY
from .geoprocessing_core import gdal_use_exceptions
from .geoprocessing_core import GDALUseExceptions
from .geoprocessing_core import INT8_CREATION_OPTIONS

# This is used to efficiently pass data to the raster stats worker if available
if sys.version_info >= (3, 8):
    import multiprocessing.shared_memory

GDAL_VERSION = tuple(int(_) for _ in gdal.__version__.split('.'))


class ReclassificationMissingValuesError(Exception):
    """Raised when a raster value is not a valid key to a dictionary.

    Attributes:
        msg (str) - error message
        missing_values (list) - a list of the missing values from the raster
            that are not keys in the dictionary

    """

    def __init__(self, missing_values, raster_path, value_map):
        """See Attributes for args docstring."""
        self.msg = (
            f'The following {missing_values.size} raster values '
            f'{missing_values} from "{raster_path}" do not have corresponding '
            f'entries in the value map: {value_map}.')
        self.missing_values = missing_values
        super().__init__(self.msg)


LOGGER = logging.getLogger(__name__)

# Used in joining finished TaskGraph Tasks.
_MAX_TIMEOUT = 60.0

_VALID_GDAL_TYPES = (
    set([getattr(gdal, x) for x in dir(gdal.gdalconst) if 'GDT_' in x]))

_LOGGING_PERIOD = 5.0  # min 5.0 seconds per update log message for the module
_LARGEST_ITERBLOCK = 2**16  # largest block for iterblocks to read in cells

class TimedLoggingAdapter(logging.LoggerAdapter):
    """A logging adapter to restrict logging based on a timer.

    The objective is to have a ``logging.LOGGER``-like object that can be
    called multiple times in rapid successtion, but with log messages only
    propagating every X seconds.

    This object is helpful for creating consistency in logging callbacks and is
    derived from the python stdlib ``logging.LoggerAdapter``.
    """

    def __init__(self, interval_s=_LOGGING_PERIOD):
        """Initialize the timed logging adapter.

        Args:
            interval_s (float): The logging interval, in seconds.  Defaults to
                ``_LOGGING_PERIOD``.
        """
        logging.LoggerAdapter.__init__(self, LOGGER, extra=None)
        self.interval = interval_s
        self.last_time = time.time()

    def log(self, level, msg, *args, **kwargs):
        """Log a ``LogRecord``.

        Args:
            level (int): The logging level.
            msg (str): The log message.
            args (list): The user-defined positional arguments for the log
                message.
            kwargs (dict): The user-defined keyword arguments for the log
                message.

        Returns:
            ``None``.
        """
        # The stacklevel arg to logging was introduced to logging in python
        # 3.8 and there isn't a clear way to spoof this in python 3.7, so
        # ignore.
        if sys.version_info >= (3, 8):
            # Don't override user-defined stacklevel if present.
            if 'stacklevel' not in kwargs:
                # Based on logging internals, 3 is the expected stack depth.
                kwargs['stacklevel'] = 3

            # Python 3.11 modified the stacklevel argument to be more
            # consistent with the behavior in the warnings module.
            # https://github.com/python/cpython/issues/89334
            if sys.version_info >= (3, 11):
                kwargs['stacklevel'] -= 1

        now = time.time()
        if now >= self.last_time + self.interval:
            self.last_time = now
            self.logger.log(level, msg, *args, **kwargs)


@gdal_use_exceptions
def raster_calculator(
        base_raster_path_band_const_list, local_op, target_raster_path,
        datatype_target, nodata_target,
        calc_raster_stats=True, use_shared_memory=False,
        largest_block=_LARGEST_ITERBLOCK, max_timeout=_MAX_TIMEOUT,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Apply local a raster operation on a stack of rasters.

    This function applies a user defined function across a stack of
    rasters' pixel stack. The rasters in ``base_raster_path_band_list`` must
    be spatially aligned and have the same cell sizes.

    Args:
        base_raster_path_band_const_list (sequence): a sequence containing:

            * ``(str, int)`` tuples, referring to a raster path/band index pair
              to use as an input.
            * ``numpy.ndarray`` s of up to two dimensions.  These inputs must
              all be broadcastable to each other AND the size of the raster
              inputs.
            * ``(object, 'raw')`` tuples, where ``object`` will be passed
              directly into the ``local_op``.

            All rasters must have the same raster size. If only arrays are
            input, numpy arrays must be broadcastable to each other and the
            final raster size will be the final broadcast array shape. A value
            error is raised if only "raw" inputs are passed.
        local_op (function): a function that must take in as many parameters as
            there are elements in ``base_raster_path_band_const_list``. The
            parameters in ``local_op`` will map 1-to-1 in order with the values
            in ``base_raster_path_band_const_list``. ``raster_calculator`` will
            call ``local_op`` to generate the pixel values in ``target_raster``
            along memory block aligned processing windows. Note any
            particular call to ``local_op`` will have the arguments from
            ``raster_path_band_const_list`` sliced to overlap that window.
            If an argument from ``raster_path_band_const_list`` is a
            raster/path band tuple, it will be passed to ``local_op`` as a 2D
            numpy array of pixel values that align with the processing window
            that ``local_op`` is targeting. A 2D or 1D array will be sliced to
            match the processing window and in the case of a 1D array tiled in
            whatever dimension is flat. If an argument is a scalar it is
            passed as as scalar.
            The return value must be a 2D array of the same size as any of the
            input parameter 2D arrays and contain the desired pixel values
            for the target raster.
        target_raster_path (string): the path of the output raster.  The
            projection, size, and cell size will be the same as the rasters
            in ``base_raster_path_const_band_list`` or the final broadcast
            size of the constant/ndarray values in the list.
        datatype_target (gdal datatype; int): the desired GDAL output type of
            the target raster.
        nodata_target (numerical value): the desired nodata value of the
            target raster.
        calc_raster_stats (boolean): If True, calculates and sets raster
            statistics (min, max, mean, and stdev) for target raster.
        use_shared_memory (boolean): If True, uses Python Multiprocessing
            shared memory to calculate raster stats for faster performance.
            This feature is available for Python >= 3.8 and will otherwise
            be ignored for earlier versions of Python.
        largest_block (int): Attempts to internally iterate over raster blocks
            with this many elements.  Useful in cases where the blocksize is
            relatively small, memory is available, and the function call
            overhead dominates the iteration.  Defaults to 2**20.  A value of
            anything less than the original blocksize of the raster will
            result in blocksizes equal to the original size.
        max_timeout (float): amount of time in seconds to wait for stats
            worker thread to join. Default is _MAX_TIMEOUT.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to
            geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None

    Raises:
        ValueError: invalid input provided

    """
    if not base_raster_path_band_const_list:
        raise ValueError(
            "`base_raster_path_band_const_list` is empty and "
            "should have at least one value.")

    # It's a common error to not pass in path/band tuples, so check for that
    # and report error if so
    bad_raster_path_list = False
    if not isinstance(base_raster_path_band_const_list, (list, tuple)):
        bad_raster_path_list = True
    else:
        for value in base_raster_path_band_const_list:
            if (not _is_raster_path_band_formatted(value) and
                not isinstance(value, numpy.ndarray) and
                not (isinstance(value, tuple) and len(value) == 2 and
                     value[1] == 'raw')):
                bad_raster_path_list = True
                break
    if bad_raster_path_list:
        raise ValueError(
            "Expected a sequence of path / integer band tuples, "
            "ndarrays, or (value, 'raw') pairs for "
            "`base_raster_path_band_const_list`, instead got: "
            "%s" % pprint.pformat(base_raster_path_band_const_list))

    base_raster_path_band_list = [
        path_band for path_band in base_raster_path_band_const_list
        if _is_raster_path_band_formatted(path_band)]

    # check that the target raster is not also an input raster
    if target_raster_path in [x[0] for x in base_raster_path_band_list]:
        raise ValueError(
            "%s is used as a target path, but it is also in the base input "
            "path list %s" % (
                target_raster_path, str(base_raster_path_band_const_list)))

    # check that raster inputs are all the same dimensions
    raster_info_list = [
        get_raster_info(path_band[0])
        for path_band in base_raster_path_band_list]
    geospatial_info = [
        raster_info['raster_size'] for raster_info in raster_info_list]
    if len(set(geospatial_info)) > 1:
        raise ValueError(
            "Input Rasters are not the same dimensions. The "
            "following raster are not identical: %s" % pprint.pformat(
                [(path_band[0], dimensions) for (path_band, dimensions) in
                zip(base_raster_path_band_list, geospatial_info)]))

    numpy_broadcast_list = [
        x for x in base_raster_path_band_const_list
        if isinstance(x, numpy.ndarray)]
    stats_worker_thread = None
    try:
        # numpy.broadcast can only take up to 32 arguments, this loop works
        # around that restriction:
        while len(numpy_broadcast_list) > 1:
            numpy_broadcast_list = (
                [numpy.broadcast(*numpy_broadcast_list[:32])] +
                numpy_broadcast_list[32:])
        if numpy_broadcast_list:
            numpy_broadcast_size = numpy_broadcast_list[0].shape
    except ValueError:
        # this gets raised if numpy.broadcast fails
        raise ValueError(
            "Numpy array inputs cannot be broadcast into a single shape %s" %
            numpy_broadcast_list)

    if numpy_broadcast_list and len(numpy_broadcast_list[0].shape) > 2:
        raise ValueError(
            "Numpy array inputs must be 2 dimensions or less %s" %
            numpy_broadcast_list)

    # if there are both rasters and arrays, check the numpy shape will
    # be broadcastable with raster shape
    if raster_info_list and numpy_broadcast_list:
        # geospatial lists x/y order and numpy does y/x so reverse size list
        raster_shape = tuple(reversed(raster_info_list[0]['raster_size']))
        invalid_broadcast_size = False
        if len(numpy_broadcast_size) == 1:
            # if there's only one dimension it should match the last
            # dimension first, in the raster case this is the columns
            # because of the row/column order of numpy. No problem if
            # that value is ``1`` because it will be broadcast, otherwise
            # it should be the same as the raster.
            if (numpy_broadcast_size[0] != raster_shape[1] and
                    numpy_broadcast_size[0] != 1):
                invalid_broadcast_size = True
        else:
            for dim_index in range(2):
                # no problem if 1 because it'll broadcast, otherwise must
                # be the same value
                if (numpy_broadcast_size[dim_index] !=
                        raster_shape[dim_index] and
                        numpy_broadcast_size[dim_index] != 1):
                    invalid_broadcast_size = True
        if invalid_broadcast_size:
            raise ValueError(
                "Raster size %s cannot be broadcast to numpy shape %s" % (
                    raster_shape, numpy_broadcast_size))

    with GDALUseExceptions():
        # create a "canonical" argument list that's bands, 2d numpy arrays, or
        # raw values only
        base_canonical_arg_list = []
        base_raster_list = []
        base_band_list = []
        for value in base_raster_path_band_const_list:
            # the input has been tested and value is either a raster/path band
            # tuple, 1d ndarray, 2d ndarray, or (value, 'raw') tuple.
            if _is_raster_path_band_formatted(value):
                # it's a raster/path band, keep track of open raster and band
                # for later so we can `None` them.
                base_raster_list.append(gdal.OpenEx(value[0], gdal.OF_RASTER))
                base_band_list.append(
                    base_raster_list[-1].GetRasterBand(value[1]))
                base_canonical_arg_list.append(base_band_list[-1])
            elif isinstance(value, numpy.ndarray):
                if value.ndim == 1:
                    # easier to process as a 2d array for writing to band
                    base_canonical_arg_list.append(
                        value.reshape((1, value.shape[0])))
                else:  # dimensions are two because we checked earlier.
                    base_canonical_arg_list.append(value)
            elif isinstance(value, tuple):
                base_canonical_arg_list.append(value)
            else:
                raise ValueError(
                    "An unexpected ``value`` occurred. This should never happen. "
                    "Value: %r" % value)

        # create target raster
        if raster_info_list:
            # if rasters are passed, the target is the same size as the raster
            n_cols, n_rows = raster_info_list[0]['raster_size']
        elif numpy_broadcast_list:
            # numpy arrays in args and no raster result is broadcast shape
            # expanded to two dimensions if necessary
            if len(numpy_broadcast_size) == 1:
                n_rows, n_cols = 1, numpy_broadcast_size[0]
            else:
                n_rows, n_cols = numpy_broadcast_size
        else:
            raise ValueError(
                "Only (object, 'raw') values have been passed. Raster "
                "calculator requires at least a raster or numpy array as a "
                "parameter. This is the input list: %s" % pprint.pformat(
                    base_raster_path_band_const_list))

        if datatype_target not in _VALID_GDAL_TYPES:
            raise ValueError(
                'Invalid target type, should be a gdal.GDT_* type, received '
                '"%s"' % datatype_target)

        # create target raster
        raster_driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
        try:
            os.makedirs(os.path.dirname(target_raster_path))
        except OSError:
            pass
        target_raster = raster_driver.Create(
            target_raster_path, n_cols, n_rows, 1, datatype_target,
            options=raster_driver_creation_tuple[1])

        target_band = target_raster.GetRasterBand(1)
        if nodata_target is not None:
            target_band.SetNoDataValue(nodata_target)
        if base_raster_list:
            # use the first raster in the list for the projection and geotransform
            target_raster.SetProjection(base_raster_list[0].GetProjection())
            target_raster.SetGeoTransform(base_raster_list[0].GetGeoTransform())
        target_band.FlushCache()
        target_raster.FlushCache()

        try:
            timed_logger = TimedLoggingAdapter(_LOGGING_PERIOD)

            block_offset_list = list(iterblocks(
                (target_raster_path, 1), offset_only=True,
                largest_block=largest_block))

            if calc_raster_stats:
                # if this queue is used to send computed valid blocks of
                # the raster to an incremental statistics calculator worker
                stats_worker_queue = queue.Queue()
                exception_queue = queue.Queue()

                if sys.version_info >= (3, 8) and use_shared_memory:
                    # The stats worker keeps running variables as a float64, so
                    # all input rasters are dtype float64 -- make the shared memory
                    # size equivalent.
                    block_size_bytes = (
                        numpy.dtype(numpy.float64).itemsize *
                        block_offset_list[0]['win_xsize'] *
                        block_offset_list[0]['win_ysize'])

                    shared_memory = multiprocessing.shared_memory.SharedMemory(
                        create=True, size=block_size_bytes)
            else:
                stats_worker_queue = None

            if calc_raster_stats:
                # To avoid doing two passes on the raster to calculate standard
                # deviation, we implement a continuous statistics calculation
                # as the raster is computed. This computational effort is high
                # and benefits from running in parallel. This queue and worker
                # takes a valid block of a raster and incrementally calculates
                # the raster's statistics. When ``None`` is pushed to the queue
                # the worker will finish and return a (min, max, mean, std)
                # tuple.
                LOGGER.info('starting stats_worker')
                stats_worker_thread = threading.Thread(
                    target=geoprocessing_core.stats_worker,
                    args=(stats_worker_queue, len(block_offset_list)))
                stats_worker_thread.daemon = True
                stats_worker_thread.start()
                LOGGER.info('started stats_worker %s', stats_worker_thread)

            pixels_processed = 0
            n_pixels = n_cols * n_rows

            # iterate over each block and calculate local_op
            for block_offset in block_offset_list:
                # read input blocks
                offset_list = (block_offset['yoff'], block_offset['xoff'])
                blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])
                data_blocks = []
                for value in base_canonical_arg_list:
                    if isinstance(value, gdal.Band):
                        data_blocks.append(value.ReadAsArray(**block_offset))
                        # I've encountered the following error when a gdal raster
                        # is corrupt, often from multiple threads writing to the
                        # same file. This helps to catch the error early rather
                        # than lead to confusing values of ``data_blocks`` later.
                        if not isinstance(data_blocks[-1], numpy.ndarray):
                            raise ValueError(
                                f"got a {data_blocks[-1]} when trying to read "
                                f"{value.GetDataset().GetFileList()} at "
                                f"{block_offset}, expected numpy.ndarray.")
                    elif isinstance(value, numpy.ndarray):
                        # must be numpy array and all have been conditioned to be
                        # 2d, so start with 0:1 slices and expand if possible
                        slice_list = [slice(0, 1)] * 2
                        tile_dims = list(blocksize)
                        for dim_index in [0, 1]:
                            if value.shape[dim_index] > 1:
                                slice_list[dim_index] = slice(
                                    offset_list[dim_index],
                                    offset_list[dim_index] +
                                    blocksize[dim_index],)
                                tile_dims[dim_index] = 1
                        data_blocks.append(
                            numpy.tile(value[tuple(slice_list)], tile_dims))
                    else:
                        # must be a raw tuple
                        data_blocks.append(value[0])

                target_block = local_op(*data_blocks)

                if (not isinstance(target_block, numpy.ndarray) or
                        target_block.shape != blocksize):
                    raise ValueError(
                        "Expected `local_op` to return a numpy.ndarray of "
                        "shape %s but got this instead: %s" % (
                            blocksize, target_block))

                target_band.WriteArray(
                    target_block, yoff=block_offset['yoff'],
                    xoff=block_offset['xoff'])

                # send result to stats calculator
                if stats_worker_queue:
                    # guard against an undefined nodata target
                    if nodata_target is not None:
                        target_block = target_block[target_block != nodata_target]
                    target_block = target_block.astype(numpy.float64).flatten()

                    if sys.version_info >= (3, 8) and use_shared_memory:
                        shared_memory_array = numpy.ndarray(
                            target_block.shape, dtype=target_block.dtype,
                            buffer=shared_memory.buf)
                        shared_memory_array[:] = target_block[:]

                        stats_worker_queue.put((
                            shared_memory_array.shape, shared_memory_array.dtype,
                            shared_memory))
                    else:
                        stats_worker_queue.put(target_block)

                pixels_processed += blocksize[0] * blocksize[1]
                timed_logger.info(
                    '%s %.1f%% complete',
                    os.path.basename(target_raster_path),
                    float(pixels_processed) / n_pixels * 100.0)

            LOGGER.info('100.0% complete')

            if calc_raster_stats:
                LOGGER.info("Waiting for raster stats worker result.")
                stats_worker_thread.join(max_timeout)
                if stats_worker_thread.is_alive():
                    LOGGER.error("stats_worker_thread.join() timed out")
                    raise RuntimeError("stats_worker_thread.join() timed out")
                payload = stats_worker_queue.get(True, max_timeout)
                if payload is not None:
                    target_min, target_max, target_mean, target_stddev = payload
                    # In Cython 3.0.0+, taking a square root may return a complex.
                    # Using only the real component of the complex value mimics the
                    # C behavior that we expect from our stats worker.
                    if isinstance(target_stddev, complex):
                        target_stddev = target_stddev.real
                    target_band.SetStatistics(
                        float(target_min), float(target_max), float(target_mean),
                        float(target_stddev))
                    target_band.FlushCache()
        except Exception:
            LOGGER.exception('exception encountered in raster_calculator')
            raise
        finally:
            # This block ensures that rasters are destroyed even if there's an
            # exception raised.
            base_band_list[:] = []
            base_raster_list[:] = []
            target_band.FlushCache()
            target_band = None
            target_raster.FlushCache()
            target_raster = None

            if calc_raster_stats and stats_worker_thread:
                if stats_worker_thread.is_alive():
                    stats_worker_queue.put(None, True, max_timeout)
                    LOGGER.info("Waiting for raster stats worker result.")
                    stats_worker_thread.join(max_timeout)
                    if stats_worker_thread.is_alive():
                        LOGGER.error("stats_worker_thread.join() timed out")
                        raise RuntimeError(
                            "stats_worker_thread.join() timed out")

                if sys.version_info >= (3, 8) and use_shared_memory:
                    LOGGER.debug(
                        f'unlink shared memory for process {os.getpid()}')
                    shared_memory.close()
                    shared_memory.unlink()
                    LOGGER.debug(
                        f'unlinked shared memory for process {os.getpid()}')

                # check for an exception in the workers, otherwise get result
                # and pass to writer
                try:
                    exception = exception_queue.get_nowait()
                    LOGGER.error("Exception encountered at termination.")
                    raise exception
                except queue.Empty:
                    pass


def array_equals_nodata(array, nodata):
    """Check for the presence of ``nodata`` values in ``array``.

    The comparison supports ``numpy.nan`` and unset (``None``) nodata values.

    Args:
        array (numpy array): the array to mask for nodata values.
        nodata (number): the nodata value to check for. Supports ``numpy.nan``.

    Returns:
        A boolean numpy array with values of 1 where ``array`` is equal to
        ``nodata`` and 0 otherwise.
    """
    # If nodata is undefined, nothing matches nodata.
    if nodata is None:
        return numpy.zeros(array.shape, dtype=bool)

    # comparing an integer array against numpy.nan works correctly and is
    # faster than using numpy.isclose().
    if numpy.issubdtype(array.dtype, numpy.integer):
        return array == nodata
    return numpy.isclose(array, nodata, equal_nan=True)


def choose_dtype(*raster_paths):
    """
    Choose an appropriate dtype for an output derived from the given inputs.

    Returns the dtype with the greatest size/precision among the inputs, so
    that information will not be lost.

    Args:
        *raster_paths: series of raster path strings

    Returns:
        numpy dtype
    """
    dtypes = [get_raster_info(path)['numpy_type'] for path in raster_paths]
    return numpy.result_type(*dtypes)


def choose_nodata(dtype):
    """
    Choose an appropriate nodata value for data of a given dtype.

    Args:
        dtype (numpy.dtype): data type for which to choose nodata

    Returns:
        number to use as nodata value
    """
    try:
        return float(numpy.finfo(dtype).max)
    except ValueError:
        return int(numpy.iinfo(dtype).max)


def raster_map(op, rasters, target_path, target_nodata=None, target_dtype=None,
               raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Apply a pixelwise function to a series of raster inputs.

    The output raster will have nodata where any input raster has nodata.
    Raster inputs are split into aligned blocks, and the function is
    applied individually to each stack of blocks (as numpy arrays).

    Args:
        op (function): Function to apply to the inputs. It should accept a
            number of arguments equal to the length of ``*inputs``. It should
            return a numpy array with the same shape as its array input(s).
        rasters (list[str]): Paths to rasters to input to ``op``, in the order
            that they will be passed to ``op``. All rasters should be aligned
            and have the same dimensions.
        target_path (str): path to write out the output raster.
        target_nodata (number): Nodata value to use for the output raster.
            Optional. If not provided, a suitable nodata value will be chosen.
        target_dtype (numpy.dtype): dtype to use for the output. Optional. If
            not provided, a suitable dtype will be chosen.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to
            geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS. If the
            ``target_dtype`` is int8, the ``PIXELTYPE=SIGNEDBYTE`` option will
            be added to the creation options tuple if it is not already there.

    Returns:
        ``None``
    """
    nodatas = []
    for raster in rasters:
        raster_info = get_raster_info(raster)
        if raster_info['n_bands'] > 1:
            LOGGER.warning(f'{raster} has more than one band. Only the first '
                           'band will be used.')
        nodatas.append(raster_info['nodata'][0])

    # choose an appropriate dtype if none was given
    if target_dtype is None:
        target_dtype = choose_dtype(*rasters)

    # choose an appropriate nodata value if none was given
    # if the user provides a target nodata,
    # check that it can fit in the target dtype
    if target_nodata is None:
        target_nodata = choose_nodata(target_dtype)
    else:
        if not numpy.can_cast(numpy.min_scalar_type(target_nodata),
                              target_dtype, casting='same_kind'):
            raise ValueError(
                f'Target nodata value {target_nodata} is incompatible with '
                f'the target dtype {target_dtype}')

    driver, options = raster_driver_creation_tuple
    gdal_type, type_creation_options = _numpy_to_gdal_type(target_dtype)
    options = list(options) + type_creation_options

    def apply_op(*arrays):
        """Apply the function ``op`` to the input arrays.

        Args:
            *arrays: numpy arrays with the same shape.

        Returns:
            numpy array
        """
        result = numpy.full(arrays[0].shape, target_nodata, dtype=target_dtype)

        # make a mask that is True where all input arrays are valid,
        # and False where any input array is invalid.
        valid_mask = numpy.full(arrays[0].shape, True)
        for array, nodata in zip(arrays, nodatas):
            valid_mask &= ~array_equals_nodata(array, nodata)

        # mask all arrays to the area where they all are valid
        masked_arrays = [array[valid_mask] for array in arrays]
        # apply op to the masked arrays in order
        result[valid_mask] = op(*masked_arrays)
        return result

    raster_calculator(
        [(path, 1) for path in rasters],  # assume the first band
        apply_op,
        target_path,
        gdal_type,
        target_nodata,
        raster_driver_creation_tuple=(driver, options))


def raster_reduce(function, raster_path_band, initializer, mask_nodata=True,
                  largest_block=_LARGEST_ITERBLOCK):
    """Cumulatively apply a reducing function to each block of a raster.

    This effectively reduces the entire raster to a single value, but it works
    by blocks to be memory-efficient.

    The ``function`` signature should be ``function(aggregator, block)``, where
    ``aggregator`` is the aggregated value so far, and ``block`` is a flattened
    numpy array containing the data from the block to reduce next.

    ``function`` is called once on each block. On the first ``function`` call,
    ``aggregator`` is initialized with ``initializer``. The return value from
    each ``function`` call is passed in as the ``aggregator`` argument to the
    subsequent ``function`` call. When all blocks have been reduced, the return
    value of the final ``function`` call is returned.

    Example:
        Calculate the sum of all values in a raster::

            raster_reduce(lambda total, block: total + numpy.sum(block),
                          (raster_path, 1), 0)

        Calculate a histogram of all values in a raster::

            def add_to_histogram(histogram, block):
                return histogram + numpy.histogram(block, bins=10)[0]

            raster_reduce(add_to_histogram, (raster_path, 1), numpy.zeros(10))

        Calculate the sum of all values in a raster, excluding nodata::

            nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
            def sum_excluding_nodata(total, block):
                return total + numpy.sum(block[block != nodata])

            raster_reduce(sum_excluding_nodata, (raster_path, 1), 0)

    Args:
        function (func): function to apply to each raster block
        raster_path_band (tuple): (path, band) tuple of the raster to reduce
        initializer (obj): value to initialize the aggregator for the
            first function call
        mask_nodata (bool): if True, mask out nodata before aggregating. A
            flattened array of non-nodata pixels from each block is passed to
            the ``function``. if False, each block is passed to the
            ``function`` without masking.
        largest_block (int): largest block parameter to pass to ``iterblocks``

    Returns:
        aggregate value, the final value returned from ``function``
    """
    aggregator = initializer
    timed_logger = TimedLoggingAdapter(_LOGGING_PERIOD)
    pixels_processed = 0
    raster_info = get_raster_info(raster_path_band[0])
    x_size, y_size = raster_info['raster_size']
    n_pixels = x_size * y_size
    for (_, block) in iterblocks(raster_path_band,
                                 largest_block=largest_block):
        if mask_nodata:
            data = block[~array_equals_nodata(
                block, raster_info['nodata'][raster_path_band[1] - 1])]
        else:
            data = block.flatten()
        aggregator = function(aggregator, data)
        pixels_processed += block.size
        timed_logger.info(
            f'{raster_path_band[0]} reduce '
            f'{pixels_processed / n_pixels * 100:.1f}%% complete'
        )

    LOGGER.info('100.0%% complete')
    return aggregator


@gdal_use_exceptions
def align_and_resize_raster_stack(
        base_raster_path_list, target_raster_path_list, resample_method_list,
        target_pixel_size, bounding_box_mode, base_vector_path_list=None,
        raster_align_index=None, base_projection_wkt_list=None,
        target_projection_wkt=None, mask_options=None,
        vector_mask_options=None, gdal_warp_options=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
        osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY,
        working_dir=None):
    """Generate rasters from a base such that they align geospatially.

    This function resizes base rasters that are in the same geospatial
    projection such that the result is an aligned stack of rasters that have
    the same cell size, dimensions, and bounding box. This is achieved by
    clipping or resizing the rasters to intersected, unioned, or equivocated
    bounding boxes of all the raster and vector input.

    Args:
        base_raster_path_list (sequence): a sequence of base raster paths that
            will be transformed and will be used to determine the target
            bounding box.
        target_raster_path_list (sequence): a sequence of raster paths that
            will be created to one-to-one map with ``base_raster_path_list``
            as aligned versions of those original rasters. If there are
            duplicate paths in this list, the function will raise a ValueError.
        resample_method_list (sequence): a sequence of resampling methods
            which one to one map each path in ``base_raster_path_list`` during
            resizing.  Each element must be one of
            "near|bilinear|cubic|cubicspline|lanczos|mode".
        target_pixel_size (list/tuple): the target raster's x and y pixel size
            example: (30, -30).
        bounding_box_mode (string): one of "union", "intersection", or
            a sequence of floats of the form [minx, miny, maxx, maxy] in the
            target projection coordinate system.  Depending
            on the value, output extents are defined as the union,
            intersection, or the explicit bounding box.
        base_vector_path_list (sequence): a sequence of base vector paths
            whose bounding boxes will be used to determine the final bounding
            box of the raster stack if mode is 'union' or 'intersection'.  If
            mode is 'bb=[...]' then these vectors are not used in any
            calculation.
        raster_align_index (int): indicates the index of a
            raster in ``base_raster_path_list`` that the target rasters'
            bounding boxes pixels should align with.  This feature allows
            rasters whose raster dimensions are the same, but bounding boxes
            slightly shifted less than a pixel size to align with a desired
            grid layout.  If ``None`` then the bounding box of the target
            rasters is calculated as the precise intersection, union, or
            bounding box.
        base_projection_wkt_list (sequence): if not ``None``, this is a
            sequence of base projections of the rasters in
            ``base_raster_path_list``. If a value is ``None``, the projection
            is read directly from the raster. Use this argument if there are
            rasters with no projection defined, but the projections are known.
        target_projection_wkt (string): if not ``None``, this is the desired
            projection of all target rasters in Well Known Text format, and
            target rasters will be warped to this projection. If ``None``,
            the base SRS will be passed to the target.
        mask_options (dict): optional, if not None, this is a
            dictionary of options to use an existing vector's geometry to
            mask out pixels in the target raster that do not overlap the
            vector's geometry. Keys to this dictionary are:

            * ``'mask_vector_path'`` (str): path to the mask vector file.
              This vector will be automatically projected to the target
              projection if its base coordinate system does not match the
              target.
            * ``'mask_layer_name'`` (str): the layer name to use for masking.
              If this key is not in the dictionary the default is to use
              the layer at index 0.
            * ``'mask_vector_where_filter'`` (str): an SQL WHERE string.
              This will be used to filter the geometry in the mask. Ex: ``'id
              > 10'`` would use all features whose field value of 'id' is >
              10.
            * ``'mask_raster_path'`` (str): Optional. the string path to where
              the mask raster should be written on disk.  If not provided, a
              temporary file will be created within ``working_dir``.

        vector_mask_options (dict): optional.  Alias for ``mask_options``.
            This parameter is deprecated and will be removed in a future
            version of ``pygeoprocessing``.
        gdal_warp_options (sequence): if present, the contents of this list
            are passed to the ``warpOptions`` parameter of ``gdal.Warp``. See
            the `GDAL Warp documentation
            <https://gdal.org/api/gdalwarp_cpp.html#_CPPv415GDALWarpOptions>`_
            for valid options.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter
            should not be changed unless you know what you are doing.
        working_dir=None (str): if present, the path to a directory within
           which a temporary directory will be created.  If not provided, the
           new directory will be created within the system's temporary
           directory.

    Returns:
        None

    Raises:
        ValueError
            If any combination of the raw bounding boxes, raster
            bounding boxes, vector bounding boxes, and/or vector_mask
            bounding box does not overlap to produce a valid target.
        ValueError
            If any of the input or target lists are of different
            lengths.
        ValueError
            If there are duplicate paths on the target list which would
            risk corrupted output.
        ValueError
            If some combination of base, target, and embedded source
            reference systems results in an ambiguous target coordinate
            system.
        ValueError
            If ``mask_options`` is not None but the
            ``mask_vector_path`` is undefined or doesn't point to a valid
            file.
        ValueError
            If ``pixel_size`` is not a 2 element sequence of numbers.
    """
    if vector_mask_options is not None:
        warnings.warn('The vector_mask_options parameter is deprecated and '
                      'will be removed in a future release of '
                      'pygeoprocessing. Please use mask_options instead.',
                      DeprecationWarning)
        mask_options = vector_mask_options

    # make sure that the input lists are of the same length
    list_lengths = [
        len(base_raster_path_list), len(target_raster_path_list),
        len(resample_method_list)]
    if len(set(list_lengths)) != 1:
        raise ValueError(
            "base_raster_path_list, target_raster_path_list, and "
            "resample_method_list must be the same length "
            " current lengths are %s" % (str(list_lengths)))

    unique_targets = set(target_raster_path_list)
    if len(unique_targets) != len(target_raster_path_list):
        seen = set()
        duplicate_list = []
        for path in target_raster_path_list:
            if path not in seen:
                seen.add(path)
            else:
                duplicate_list.append(path)
        raise ValueError(
            "There are duplicated paths on the target list. This is an "
            "invalid state of ``target_path_list``. Duplicates: %s" % (
                duplicate_list))

    # we can accept 'union', 'intersection', or a 4 element list/tuple
    if bounding_box_mode not in ["union", "intersection"] and (
            not isinstance(bounding_box_mode, (list, tuple)) or
            len(bounding_box_mode) != 4):
        raise ValueError("Unknown bounding_box_mode %s" % (
            str(bounding_box_mode)))

    n_rasters = len(base_raster_path_list)
    if ((raster_align_index is not None) and
            ((raster_align_index < 0) or (raster_align_index >= n_rasters))):
        raise ValueError(
            "Alignment index is out of bounds of the datasets index: %s"
            " n_elements %s" % (raster_align_index, n_rasters))

    _assert_is_valid_pixel_size(target_pixel_size)

    # used to get bounding box, projection, and possible alignment info
    raster_info_list = [
        get_raster_info(path) for path in base_raster_path_list]

    # get the literal or intersecting/unioned bounding box
    if isinstance(bounding_box_mode, (list, tuple)):
        # if it's a sequence or tuple, it must be a manual bounding box
        LOGGER.debug(
            "assuming manual bounding box mode of %s", bounding_box_mode)
        target_bounding_box = bounding_box_mode
    else:
        # either intersection or union, get list of bounding boxes, reproject
        # if necessary, and reduce to a single box
        if base_vector_path_list is not None:
            # vectors are only interesting for their bounding boxes, that's
            # this construction is inside an else.
            vector_info_list = [
                get_vector_info(path) for path in base_vector_path_list]
        else:
            vector_info_list = []

        raster_bounding_box_list = []
        for raster_index, raster_info in enumerate(raster_info_list):
            # this block calculates the base projection of ``raster_info`` if
            # ``target_projection_wkt`` is defined, thus implying a
            # reprojection will be necessary.
            if target_projection_wkt:
                if base_projection_wkt_list and \
                        base_projection_wkt_list[raster_index]:
                    # a base is defined, use that
                    base_raster_projection_wkt = \
                        base_projection_wkt_list[raster_index]
                else:
                    # otherwise use the raster's projection and there must
                    # be one since we're reprojecting
                    base_raster_projection_wkt = raster_info['projection_wkt']
                    if not base_raster_projection_wkt:
                        raise ValueError(
                            "no projection for raster %s" %
                            base_raster_path_list[raster_index])
                # since the base spatial reference is potentially different
                # than the target, we need to transform the base bounding
                # box into target coordinates so later we can calculate
                # accurate bounding box overlaps in the target coordinate
                # system
                raster_bounding_box_list.append(
                    transform_bounding_box(
                        raster_info['bounding_box'],
                        base_raster_projection_wkt, target_projection_wkt))
            else:
                raster_bounding_box_list.append(raster_info['bounding_box'])

        # include the vector bounding box information to make a global list
        # of target bounding boxes
        bounding_box_list = [
            vector_info['bounding_box'] if target_projection_wkt is None else
            transform_bounding_box(
                vector_info['bounding_box'],
                vector_info['projection_wkt'], target_projection_wkt)
            for vector_info in vector_info_list] + raster_bounding_box_list

        target_bounding_box = merge_bounding_box_list(
            bounding_box_list, bounding_box_mode)

    if mask_options:
        # ensure the mask exists and intersects with the target bounding box
        if 'mask_vector_path' not in mask_options:
            raise ValueError(
                'mask_options passed, but no value for '
                '"mask_vector_path": %s', mask_options)

        mask_vector_info = get_vector_info(
            mask_options['mask_vector_path'])

        if 'mask_vector_where_filter' in mask_options:
            # the bounding box only exists for the filtered features
            mask_vector = gdal.OpenEx(
                mask_options['mask_vector_path'], gdal.OF_VECTOR)
            mask_layer = mask_vector.GetLayer()
            mask_layer.SetAttributeFilter(
                mask_options['mask_vector_where_filter'])
            mask_bounding_box = merge_bounding_box_list(
                [[feature.GetGeometryRef().GetEnvelope()[i]
                  for i in [0, 2, 1, 3]] for feature in mask_layer],
                'union')
            mask_layer = None
            mask_vector = None
        else:
            # if no where filter then use the raw vector bounding box
            mask_bounding_box = mask_vector_info['bounding_box']

        mask_vector_projection_wkt = mask_vector_info['projection_wkt']
        if mask_vector_projection_wkt is not None and \
                target_projection_wkt is not None:
            mask_vector_bb = transform_bounding_box(
                mask_bounding_box, mask_vector_info['projection_wkt'],
                target_projection_wkt)
        else:
            mask_vector_bb = mask_vector_info['bounding_box']
        # Calling `merge_bounding_box_list` will raise an ValueError if the
        # bounding box of the mask and the target do not intersect. The
        # result is otherwise not used.
        _ = merge_bounding_box_list(
            [target_bounding_box, mask_vector_bb], 'intersection')

    if raster_align_index is not None and raster_align_index >= 0:
        # bounding box needs alignment
        align_bounding_box = (
            raster_info_list[raster_align_index]['bounding_box'])
        align_pixel_size = (
            raster_info_list[raster_align_index]['pixel_size'])
        # adjust bounding box so lower left corner aligns with a pixel in
        # raster[raster_align_index]
        for index in [0, 1]:
            n_pixels = int(
                (target_bounding_box[index] - align_bounding_box[index]) /
                float(align_pixel_size[index]))
            target_bounding_box[index] = (
                n_pixels * align_pixel_size[index] +
                align_bounding_box[index])

    if mask_options:
        # Create a warped VRT.
        # This allows us to cheaply figure out the dimensions, projection, etc.
        # of the target raster without actually warping the entire raster to a
        # GTiff.
        temp_working_dir = tempfile.mkdtemp(dir=working_dir, prefix='mask-')
        warped_vrt = os.path.join(temp_working_dir, 'warped.vrt')
        warp_raster(
            base_raster_path=base_raster_path_list[0],
            target_pixel_size=target_pixel_size,
            target_raster_path=warped_vrt,
            resample_method='near',
            target_bb=target_bounding_box,
            raster_driver_creation_tuple=('VRT', []),
            target_projection_wkt=target_projection_wkt,
            base_projection_wkt=(
                None if not base_projection_wkt_list else
                base_projection_wkt_list[0]),
            gdal_warp_options=gdal_warp_options)

        # Convert the warped VRT to a GTiff for rasterization.
        if 'mask_raster_path' in mask_options:
            mask_raster_path = mask_options['mask_raster_path']
        else:
            # Add the mask raster path ot the vector mask options to force
            # warp_raster to use the existing raster mask.
            mask_raster_path = os.path.join(temp_working_dir, 'mask.tif')
            mask_options['mask_raster_path'] = mask_raster_path
        new_raster_from_base(
            warped_vrt, mask_raster_path, gdal.GDT_Byte, [0], [0])

        # Rasterize the vector onto the new GTiff.
        rasterize(mask_options['mask_vector_path'],
                  mask_raster_path, burn_values=[1],
                  where_clause=(
                      mask_options['mask_vector_where_filter']
                      if 'mask_vector_where_filter' in mask_options
                      else None))

    for index, (base_path, target_path, resample_method) in enumerate(zip(
            base_raster_path_list, target_raster_path_list,
            resample_method_list)):
        warp_raster(
            base_path, target_pixel_size, target_path, resample_method,
            target_bb=target_bounding_box,
            raster_driver_creation_tuple=(raster_driver_creation_tuple),
            target_projection_wkt=target_projection_wkt,
            base_projection_wkt=(
                None if not base_projection_wkt_list else
                base_projection_wkt_list[index]),
            mask_options=mask_options,
            gdal_warp_options=gdal_warp_options)
        LOGGER.info(
            '%d of %d aligned: %s', index+1, n_rasters,
            os.path.basename(target_path))

    LOGGER.info("aligned all %d rasters.", n_rasters)

    if mask_options:
        shutil.rmtree(temp_working_dir, ignore_errors=True)


@gdal_use_exceptions
def new_raster_from_base(
        base_path, target_path, datatype, band_nodata_list,
        fill_value_list=None, n_rows=None, n_cols=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Create new raster by coping spatial reference/geotransform of base.

    A convenience function to simplify the creation of a new raster from the
    basis of an existing one.  Depending on the input mode, one can create
    a new raster of the same dimensions, geotransform, and georeference as
    the base.  Other options are provided to change the raster dimensions,
    number of bands, nodata values, data type, and core raster creation
    options.

    Args:
        base_path (string): path to existing raster.
        target_path (string): path to desired target raster.
        datatype: the pixel datatype of the output raster, for example
            gdal.GDT_Float32.  See the following header file for supported
            pixel types:
            http://www.gdal.org/gdal_8h.html#22e22ce0a55036a96f652765793fb7a4
        band_nodata_list (sequence): list of nodata values, one for each band,
            to set on target raster.  If value is 'None' the nodata value is
            not set for that band.  The number of target bands is inferred
            from the length of this list.
        fill_value_list (sequence): list of values to fill each band with. If
            None, no filling is done.
        n_rows (int): if not None, defines the number of target raster rows.
        n_cols (int): if not None, defines the number of target raster
            columns.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None
    """
    base_raster = gdal.OpenEx(base_path, gdal.OF_RASTER)
    if n_rows is None:
        n_rows = base_raster.RasterYSize
    if n_cols is None:
        n_cols = base_raster.RasterXSize
    driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])

    local_raster_creation_options = list(raster_driver_creation_tuple[1])
    numpy_dtype = _gdal_to_numpy_type(datatype, local_raster_creation_options)
    base_band = base_raster.GetRasterBand(1)
    block_size = base_band.GetBlockSize()
    # It's not clear how or IF we can determine if the output should be
    # striped or tiled.  Here we leave it up to the default inputs or if its
    # obviously not striped we tile.
    if not any(
            ['TILED' in option for option in local_raster_creation_options]):
        # TILED not set, so lets try to set it to a reasonable value
        if block_size[0] != n_cols:
            # if x block is not the width of the raster it *must* be tiled
            # otherwise okay if it's striped or tiled, I can't construct a
            # test case to cover this, but there is nothing in the spec that
            # restricts this so I have it just in case.
            local_raster_creation_options.append('TILED=YES')

    if not any(
            ['BLOCK' in option for option in local_raster_creation_options]):
        # not defined, so lets copy what we know from the current raster
        local_raster_creation_options.extend([
            'BLOCKXSIZE=%d' % block_size[0],
            'BLOCKYSIZE=%d' % block_size[1]])

    # make target directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(target_path))
    except OSError:
        pass

    base_band = None
    n_bands = len(band_nodata_list)
    target_raster = driver.Create(
        target_path, n_cols, n_rows, n_bands, datatype,
        options=local_raster_creation_options)
    target_raster.SetProjection(base_raster.GetProjection())
    target_raster.SetGeoTransform(base_raster.GetGeoTransform())
    base_raster = None

    for index, nodata_value in enumerate(band_nodata_list):
        if nodata_value is None:
            continue
        target_band = target_raster.GetRasterBand(index + 1)
        try:
            target_band.SetNoDataValue(nodata_value.item())
        except AttributeError:
            target_band.SetNoDataValue(nodata_value)

    target_raster.FlushCache()
    timed_logger = TimedLoggingAdapter(_LOGGING_PERIOD)
    pixels_processed = 0
    n_pixels = n_cols * n_rows
    if fill_value_list is not None:
        for index, fill_value in enumerate(fill_value_list):
            if fill_value is None:
                continue
            target_band = target_raster.GetRasterBand(index + 1)
            # some rasters are very large and a fill can appear to cause
            # computation to hang. This block, though possibly slightly less
            # efficient than ``band.Fill`` will give real-time feedback about
            # how the fill is progressing.
            for offsets in iterblocks((target_path, 1), offset_only=True):
                shape = (offsets['win_ysize'], offsets['win_xsize'])
                fill_array = numpy.full(shape, fill_value, dtype=numpy_dtype)
                pixels_processed += (
                    offsets['win_ysize'] * offsets['win_xsize'])
                target_band.WriteArray(
                    fill_array, offsets['xoff'], offsets['yoff'])
                timed_logger.info(
                    f'filling new raster {target_path} with {fill_value} '
                    f'-- {float(pixels_processed)/n_pixels*100.0:.2f}% '
                    f'complete')
            target_band = None
    target_band = None
    target_raster = None


@gdal_use_exceptions
def create_raster_from_vector_extents(
        base_vector_path, target_raster_path, target_pixel_size,
        target_pixel_type, target_nodata, fill_value=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Create a blank raster based on a vector file extent.

    Args:
        base_vector_path (string): path to vector shapefile to base the
            bounding box for the target raster.
        target_raster_path (string): path to location of generated geotiff;
            the upper left hand corner of this raster will be aligned with the
            bounding box of the source vector and the extent will be exactly
            equal or contained the source vector's bounding box depending on
            whether the pixel size divides evenly into the source bounding
            box; if not coordinates will be rounded up to contain the original
            extent.
        target_pixel_size (list/tuple): the x/y pixel size as a sequence
            Example::

                [30.0, -30.0]

        target_pixel_type (int): gdal GDT pixel type of target raster
        target_nodata (numeric): target nodata value. Can be None if no nodata
            value is needed.
        fill_value (int/float): value to fill in the target raster; no fill if
            value is None
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None
    """
    # Determine the width and height of the tiff in pixels based on the
    # maximum size of the combined envelope of all the features
    vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    shp_extent = None
    for layer_index in range(vector.GetLayerCount()):
        layer = vector.GetLayer(layer_index)
        for feature in layer:
            try:
                # envelope is [xmin, xmax, ymin, ymax]
                feature_extent = feature.GetGeometryRef().GetEnvelope()
                if shp_extent is None:
                    shp_extent = list(feature_extent)
                else:
                    # expand bounds of current bounding box to include that
                    # of the newest feature
                    shp_extent = [
                        f(shp_extent[index], feature_extent[index])
                        for index, f in enumerate([min, max, min, max])]
            except AttributeError as error:
                # For some valid OGR objects the geometry can be undefined
                # since it's valid to have a NULL entry in the attribute table
                # this is expressed as a None value in the geometry reference
                # this feature won't contribute
                LOGGER.warning(error)
        layer = None

    target_srs_wkt = vector.GetLayer(0).GetSpatialRef().ExportToWkt()
    vector = None

    if shp_extent is None:
        raise ValueError(
            f'the vector at {base_vector_path} has no geometry, cannot '
            f'create a raster from these extents')

    create_raster_from_bounding_box(
        target_bounding_box=[
            shp_extent[0], shp_extent[2], shp_extent[1], shp_extent[3]
        ],
        target_raster_path=target_raster_path,
        target_pixel_size=target_pixel_size,
        target_pixel_type=target_pixel_type,
        target_srs_wkt=target_srs_wkt,
        target_nodata=target_nodata,
        fill_value=fill_value,
        raster_driver_creation_tuple=raster_driver_creation_tuple
    )

@gdal_use_exceptions
def create_raster_from_bounding_box(
        target_bounding_box, target_raster_path, target_pixel_size,
        target_pixel_type, target_srs_wkt, target_nodata, fill_value=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Create a raster from a given bounding box.

    Args:
        target_bounding_box (tuple): a 4-element iterable of (minx, miny,
            maxx, maxy) in projected units matching the SRS of
            ``target_srs_wkt``.
        target_raster_path (string): The path to where the new raster should be
            created on disk.
        target_pixel_size (tuple): A 2-element tuple of the (x, y) pixel size
            of the target raster.  Elements are in units of the target SRS.
        target_pixel_type (int): The GDAL GDT_* type of the target raster.
        target_srs_wkt (string): The SRS of the target raster, in Well-Known
            Text format.
        target_nodata (float): The nodata value of the target raster, or
            ``None`` if no nodata value is to be set.
        fill_value=None (number): If provided, the value that the target raster
            should be filled with.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        ``None``
    """
    bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = target_bounding_box

    driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
    n_bands = 1

    # determine the raster size that bounds the input bounding box and then
    # adjust the bounding box to be that size
    target_x_size = int(abs(
        float(bbox_maxx - bbox_minx) / target_pixel_size[0]))
    target_y_size = int(abs(
        float(bbox_maxy - bbox_miny) / target_pixel_size[1]))
    x_residual = (
        abs(target_x_size * target_pixel_size[0]) -
        (bbox_maxx - bbox_minx))
    if not numpy.isclose(x_residual, 0.0):
        target_x_size += 1
    y_residual = (
        abs(target_y_size * target_pixel_size[1]) -
        (bbox_maxy - bbox_miny))
    if not numpy.isclose(y_residual, 0.0):
        target_y_size += 1

    if target_x_size == 0:
        LOGGER.warning(
            "bounding_box is so small that x dimension rounds to 0; "
            "clamping to 1.")
        target_x_size = 1
    if target_y_size == 0:
        LOGGER.warning(
            "bounding_box is so small that y dimension rounds to 0; "
            "clamping to 1.")
        target_y_size = 1

    raster = driver.Create(
        target_raster_path, target_x_size, target_y_size, n_bands,
        target_pixel_type, options=raster_driver_creation_tuple[1])
    raster.SetProjection(target_srs_wkt)

    # Set the transform based on the upper left corner and given pixel
    # dimensions.
    x_source = bbox_maxx if target_pixel_size[0] < 0 else bbox_minx
    y_source = bbox_maxy if target_pixel_size[1] < 0 else bbox_miny
    raster_transform = [
        x_source, target_pixel_size[0], 0,
        y_source, 0, target_pixel_size[1]]
    raster.SetGeoTransform(raster_transform)

    # Fill the band if requested.
    band = raster.GetRasterBand(1)
    if fill_value is not None:
        band.Fill(fill_value)

    # Set the nodata value.
    if target_nodata is not None:
        band.SetNoDataValue(float(target_nodata))

    band = None
    raster = None


@gdal_use_exceptions
def interpolate_points(
        base_vector_path, vector_attribute_field, target_raster_path_band,
        interpolation_mode):
    """Interpolate point values onto an existing raster.

    Args:
        base_vector_path (string): path to a shapefile that contains point
            vector layers.
        vector_attribute_field (field): a string in the vector referenced at
            ``base_vector_path`` that refers to a numeric value in the
            vector's attribute table.  This is the value that will be
            interpolated across the raster.
        target_raster_path_band (tuple): a path/band number tuple to an
            existing raster which likely intersects or is nearby the source
            vector. The band in this raster will take on the interpolated
            numerical values  provided at each point.
        interpolation_mode (string): the interpolation method to use for
            scipy.interpolate.griddata, one of 'linear', near', or 'cubic'.

    Return:
        None
    """
    source_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    point_list = []
    value_list = []
    for layer_index in range(source_vector.GetLayerCount()):
        layer = source_vector.GetLayer(layer_index)
        for point_feature in layer:
            value = point_feature.GetField(vector_attribute_field)
            # Add in the numpy notation which is row, col
            # Here the point geometry is in the form x, y (col, row)
            geometry = point_feature.GetGeometryRef()
            point = geometry.GetPoint()
            point_list.append([point[1], point[0]])
            value_list.append(value)

    point_array = numpy.array(point_list)
    value_array = numpy.array(value_list)

    # getting the offsets first before the raster is opened in update mode
    offset_list = list(
        iterblocks(target_raster_path_band, offset_only=True))
    target_raster = gdal.OpenEx(
        target_raster_path_band[0], gdal.OF_RASTER | gdal.GA_Update)
    band = target_raster.GetRasterBand(target_raster_path_band[1])
    nodata = band.GetNoDataValue()
    geotransform = target_raster.GetGeoTransform()
    for offset in offset_list:
        grid_y, grid_x = numpy.mgrid[
            offset['yoff']:offset['yoff']+offset['win_ysize'],
            offset['xoff']:offset['xoff']+offset['win_xsize']]
        grid_y = grid_y * geotransform[5] + geotransform[3]
        grid_x = grid_x * geotransform[1] + geotransform[0]

        # this is to be consistent with GDAL 2.0's change of 'nearest' to
        # 'near' for an interpolation scheme that SciPy did not change.
        if interpolation_mode == 'near':
            interpolation_mode = 'nearest'
        raster_out_array = scipy.interpolate.griddata(
            point_array, value_array, (grid_y, grid_x), interpolation_mode,
            nodata)
        band.WriteArray(raster_out_array, offset['xoff'], offset['yoff'])


def align_bbox(geotransform, bbox):
    """Pad a bounding box so that it aligns with the grid of a given geotransform.

    Ignores row and column rotation.

    Args:
        geotransform (list): GDAL raster geotransform to align to
        bbox (list): bounding box in the form [minx, miny, maxx, maxy]

    Returns:
        padded bounding box in the form [minx, miny, maxx, maxy]

    Raises:
        ValueError if invalid geotransform or bounding box are provided
    """
    # NOTE: x_origin and y_origin do not necessarily equal minx and miny.
    # If pixel_width is positive, x_origin = minx
    # If pixel_width is negative, x_origin = maxx
    # If pixel height is positive, y_origin = miny
    # If pixel height is negative, y_origin = maxy
    x_origin, pixel_width, r_x, y_origin, r_y, pixel_height = geotransform
    if r_x != 0 or r_y != 0:
        LOGGER.warning('Row and/or column rotation supplied to align_bbox '
                       'will be ignored')
    if pixel_width == 0 or pixel_height == 0:
        raise ValueError('Pixel width and height must not be 0')
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        raise ValueError('bbox must be in order [minX, minY, maxX, maxY]')
    if ((pixel_width > 0 and bbox[0] < x_origin) or
        (pixel_height > 0 and bbox[1] < y_origin) or
        (pixel_width < 0 and bbox[0] > x_origin) or
        (pixel_height < 0 and bbox[1] > y_origin)):
        raise ValueError('bbox must fall within geotransform grid')
    return [
        x_origin + abs(pixel_width) * math.floor(
            (bbox[0] - x_origin) / pixel_width * numpy.sign(pixel_width)),
        y_origin + abs(pixel_height) * math.floor(
            (bbox[1] - y_origin) / pixel_height * numpy.sign(pixel_height)),
        x_origin + abs(pixel_width) * math.ceil(
            (bbox[2] - x_origin) / pixel_width * numpy.sign(pixel_width)),
        y_origin + abs(pixel_height) * math.ceil(
            (bbox[3] - y_origin) / pixel_height * numpy.sign(pixel_height))]


@gdal_use_exceptions
def zonal_statistics(
        base_raster_path_band, aggregate_vector_path,
        aggregate_layer_name=None, ignore_nodata=True,
        polygons_might_overlap=True, include_value_counts=False,
        working_dir=None):
    """Collect stats on pixel values which lie within polygons or multipolygons.

    This function summarizes raster statistics including min, max,
    mean, and pixel count over the regions on the raster that are
    overlapped by the polygons in the vector layer. Statistics are calculated
    in two passes, where first polygons aggregate over pixels in the raster
    whose centers intersect with the polygon. In the second pass, any polygons
    that are not aggregated use their bounding box to intersect with the
    raster for overlap statistics.

    Statistics are calculated on the set of pixels that fall within each
    feature polygon. If ``ignore_nodata`` is false, nodata pixels are considered
    valid when calculating the statistics:

    - 'min': minimum valid pixel value
    - 'max': maximum valid pixel value
    - 'sum': sum of valid pixel values
    - 'count': number of valid pixels
    - 'nodata_count': number of nodata pixels
    - 'value_counts': number of pixels having each unique value

    Note:
        There may be some degenerate cases where the bounding box vs. actual
        geometry intersection would be incorrect, but these are so unlikely as
        to be manually constructed. If you encounter one of these please create
        an issue at https://github.com/natcap/pygeoprocessing/issues with the
        datasets used.

    Args:
        base_raster_path_band (tuple or list[tuple]): base raster
            (path, band) tuple(s) to analyze. If a list of multiple raster
            bands is provided, they must be aligned (all having the same
            bounding box, geotransform, and projection).
        aggregate_vector_path (string): a path to a polygon vector whose
            geometric features indicate the areas in
            ``base_raster_path_band`` to calculate zonal statistics.
        aggregate_layer_name (string): name of vector layer that will be
            used to aggregate results over.  If set to None, the first layer
            in the DataSource will be used as retrieved by ``.GetLayer()``.
            Note: it is normal and expected to set this field at None if the
            aggregating vector dataset has a single layer as many do,
            including the common 'ESRI Shapefile'.
        ignore_nodata: if true, then nodata pixels are not accounted for when
            calculating min, max, count, or mean.  However, the value of
            ``nodata_count`` will always be the number of nodata pixels
            aggregated under the polygon.
        polygons_might_overlap (boolean): if True the function calculates
            aggregation coverage close to optimally by rasterizing sets of
            polygons that don't overlap.  However, this step can be
            computationally expensive for cases where there are many polygons.
            Setting this flag to ``False`` directs the function rasterize in
            one step.
        include_value_counts (boolean): If True, the function tallies the
            number of pixels of each value under the polygon.  This is useful
            for classified rasters but could exhaust available memory when run
            on a continuous (floating-point) raster.  Defaults to False.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run.

    Return:
        If `base_raster_path_band` is a tuple, the return value is a nested
        dictionary of stats for that raster band. Top-level keys are the
        aggregate feature FIDs. Each nested FID dictionary then contains
        statistics about that feature: 'min', 'max', 'sum', 'count',
        'nodata_count', and optionally 'value_counts'. Example::
            {
                0: {
                    'min': 0,
                    'max': 14,
                    'sum': 42,
                    'count': 8,
                    'nodata_count': 1,
                    'value_counts': {
                        2: 5,
                        4: 1,
                        14: 2
                    }
                }
            }

        If `base_raster_path_band` is a list of tuples, the return value is
        a list of the nested dictionaries described above. Each dictionary in
        the list contains the stats calculated for the corresponding raster
        band in the `base_raster_path_band` list.


    Raises:
        ValueError
            if ``base_raster_path_band`` is incorrectly formatted, or if
            not all of the input raster bands are aligned with each other

        ValueError
            if ``aggregate_vector_path`` has a geometry type other than
            Polygon or MultiPolygon

        RuntimeError
            if the aggregate vector or layer cannot be opened

    """
    # Check that the raster path/band input is formatted correctly
    multi_raster_mode = isinstance(base_raster_path_band, list)
    if not multi_raster_mode:
        base_raster_path_band = [base_raster_path_band]
    for path_band_tuple in base_raster_path_band:
        if not _is_raster_path_band_formatted(path_band_tuple):
            raise ValueError(
                '`base_raster_path_band` not formatted as expected.  Expected '
                f'(path, band_index), received {repr(path_band_tuple)}')

    # Check that all input rasters are aligned. This should hold true if they
    # have the same projection, geotransform, and bounding box.
    for attr in ['geotransform', 'bounding_box', 'projection_wkt']:
        vals = set()
        for path, _ in base_raster_path_band:
            raster_info = get_raster_info(path)
            vals.add(str(raster_info[attr]))
        if len(vals) > 1:
            raise ValueError(
                'All input rasters must be aligned. Multiple values of '
                f'"{attr}" were found among the input rasters: {vals}')

    # Check that the aggregate vector and layer can be opened
    aggregate_vector = gdal.OpenEx(aggregate_vector_path, gdal.OF_VECTOR)
    if aggregate_vector is None:
        raise RuntimeError(
            f"Could not open aggregate vector at {aggregate_vector_path}")
    if aggregate_layer_name is None:
        aggregate_layer = aggregate_vector.GetLayer()
    else:
        aggregate_layer = aggregate_vector.GetLayerByName(aggregate_layer_name)
    if aggregate_layer is None:
        raise RuntimeError(
            f"Could not open layer {aggregate_layer_name} of {aggregate_vector_path}")

    # Check that the vector geometry type is polygon or multipolygon
    if aggregate_layer.GetGeomType() not in [ogr.wkbPolygon, ogr.wkbMultiPolygon]:
        raise ValueError('Vector geometry type must be Polygon or MultiPolygon')

    # Define the default/empty statistics values
    # These values will be returned for features that have no geometry or
    # don't overlap any valid pixels.
    def default_aggregate_dict():
        default_aggregate_dict = {
            'min': None, 'max': None, 'count': 0, 'nodata_count': 0, 'sum': 0}
        if include_value_counts:
            default_aggregate_dict['value_counts'] = collections.Counter()
        return default_aggregate_dict

    # Create a copy of the aggregate vector with the FID copied into a
    # persistent attribute. Do this before reprojecting because converting to
    # GPKG can change the FIDs.
    # If the aggregate vector already contains a field called 'original_fid',
    # this will break and GDAL will emit a warning.
    temp_working_dir = tempfile.mkdtemp(dir=working_dir)
    copy_path = os.path.join(temp_working_dir, 'vector_copy.gpkg')
    target_vector = ogr.GetDriverByName('GPKG').CreateDataSource(copy_path)
    target_layer_id = 'disjoint_vector'
    target_layer = target_vector.CreateLayer(
        name=target_layer_id,
        srs=aggregate_layer.GetSpatialRef(),
        geom_type=aggregate_layer.GetGeomType())
    fid_field_name = 'original_fid'
    target_layer.CreateField(ogr.FieldDefn(fid_field_name, ogr.OFTInteger))
    valid_fid_set = set()
    aggregate_stats_list = [{} for _ in base_raster_path_band]
    original_to_new_fid_map = {}
    for feature in aggregate_layer:
        fid = feature.GetFID()
        # Initialize the output data structure:
        # a list of zonal_stats dicts, one for each raster input
        for stats_dict in aggregate_stats_list:
            stats_dict[fid] = default_aggregate_dict()
        geom_ref = feature.GetGeometryRef()
        if geom_ref is None:
            LOGGER.warning(
                f'Skipping feature with FID {fid} because it has no geometry')
            continue
        valid_fid_set.add(fid)
        feature_copy = ogr.Feature(target_layer.GetLayerDefn())
        feature_copy.SetGeometry(geom_ref.Clone())
        feature_copy.SetField(fid_field_name, fid)
        target_layer.CreateFeature(feature_copy)
        original_to_new_fid_map[fid] = feature_copy.GetFID()
    target_layer, target_vector, feature, feature_copy = None, None, None, None
    geom_ref, aggregate_layer, aggregate_vector = None, None, None

    # Reproject the vector to match the raster projection
    target_vector_path = os.path.join(temp_working_dir, 'reprojected.gpkg')
    reproject_vector(
        base_vector_path=copy_path,
        target_projection_wkt=raster_info['projection_wkt'],
        target_path=target_vector_path,
        layer_id=target_layer_id,
        driver_name='GPKG',
        copy_fields=True)
    target_vector = gdal.OpenEx(target_vector_path, gdal.OF_VECTOR)

    try:
        bbox_intersection = merge_bounding_box_list(
            [raster_info['bounding_box'],
             get_vector_info(target_vector_path)['bounding_box']
            ], 'intersection')
    except ValueError as err:
        if 'Bounding boxes do not intersect' in repr(err):
            LOGGER.error(
                f'Aggregate vector {aggregate_vector_path} does not intersect '
                'the input raster(s)')
            if multi_raster_mode:
                return aggregate_stats_list
            else:
                return aggregate_stats_list[0]
        else:
            # this would be very unexpected to get here, but if it happened
            # and we didn't raise an exception, execution could get weird.
            raise

    # Expand the intersection bounding box to align with the nearest pixels
    # in the original raster
    aligned_bbox = align_bbox(raster_info['geotransform'], bbox_intersection)

    # Clip base rasters to their intersection with the aggregate vector
    LOGGER.info('Clipping rasters to their intersection with the vector')
    target_raster_path_band_list = []
    for i, (base_path, band) in enumerate(base_raster_path_band):
        raster_info = get_raster_info(path)
        if (raster_info['datatype'] in {gdal.GDT_Float32, gdal.GDT_Float64}
                and include_value_counts):
            LOGGER.warning(
                f'Value counts requested on a floating-point raster: {path}. '
                'This can cause excessive memory usage if the raster has '
                'continuous values.')
        target_path = os.path.join(temp_working_dir, f'{i}.tif')
        target_raster_path_band_list.append((target_path, band))
        gdal.Warp(
            destNameOrDestDS=target_path,
            srcDSOrSrcDSTab=base_path,
            format='GTIFF',
            # specify the original pixel size because warp doesn't necessarily
            # preserve it by default. resolution should always be positive
            xRes=abs(raster_info['pixel_size'][0]),
            yRes=abs(raster_info['pixel_size'][1]),
            outputBounds=aligned_bbox,
            callback=_make_logger_callback("Warp %.1f%% complete %s"))

    # Calculate disjoint polygon sets
    if polygons_might_overlap:
        LOGGER.info('calculating disjoint polygon sets')
        # Only consider polygons that overlap the rasters
        # Use the original vector to be sure that the correct FIDs are returned
        disjoint_fid_sets = calculate_disjoint_polygon_set(
            aggregate_vector_path, bounding_box=bbox_intersection)
    else:
        disjoint_fid_sets = [valid_fid_set]

    # Rasterize each disjoint polygon set onto its own raster layer
    fid_nodata = numpy.iinfo(numpy.uint16).max
    fid_raster_paths = []
    for i, disjoint_fid_set in enumerate(disjoint_fid_sets):
        fid_raster_path = os.path.join(temp_working_dir, f'fid_set_{i}.tif')
        fid_set_str = ", ".join(str(fid) for fid in disjoint_fid_set)
        gdal.Rasterize(
            destNameOrDestDS=fid_raster_path,
            srcDS=target_vector,
            allTouched=False,
            attribute=fid_field_name,
            noData=fid_nodata,
            outputBounds=aligned_bbox,
            xRes=abs(raster_info['pixel_size'][0]),  # resolution must be > 0
            yRes=abs(raster_info['pixel_size'][1]),
            format='GTIFF',
            outputType=gdal.GDT_UInt16,
            creationOptions=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1],
            callback=_make_logger_callback(
                f'rasterizing disjoint polygon set {i + 1} of '
                f'{len(disjoint_fid_sets)} set %.1f%% complete (%s)'),
            where=(f'{fid_field_name} IN ({fid_set_str})'))
        fid_raster_paths.append(fid_raster_path)

    timed_logger = TimedLoggingAdapter(_LOGGING_PERIOD)
    # Calculate statistics for each raster and each feature
    # working block-wise through the rasters
    for i, (raster_path, band) in enumerate(target_raster_path_band_list):
        LOGGER.info('calculating stats on raster '
                    f'{i} of {len(target_raster_path_band_list)}')
        # fetch the block offsets before the raster is opened for writing
        offset_list = list(iterblocks((raster_path, band), offset_only=True))
        nodata = get_raster_info(raster_path)['nodata'][band - 1]
        data_source = gdal.OpenEx(raster_path, gdal.OF_RASTER)
        data_band = data_source.GetRasterBand(band)
        found_fids = set()  # track FIDs found on at least one pixel

        for set_index, fid_raster_path in enumerate(fid_raster_paths):
            LOGGER.info(
                f'disjoint polygon set {set_index + 1} of '
                f'{len(fid_raster_paths)}')
            fid_raster = gdal.OpenEx(fid_raster_path, gdal.OF_RASTER)
            fid_band = fid_raster.GetRasterBand(1)

            for offset_index, offset in enumerate(offset_list):
                timed_logger.info(
                    "%.1f%% done calculating stats for polygon set %s on raster %s",
                    offset_index / len(offset_list) * 100, set_index, i)
                fid_block = fid_band.ReadAsArray(**offset)
                data_block = data_band.ReadAsArray(**offset)

                # Update stats for each FID found in this block of data
                for fid in numpy.unique(fid_block):
                    if fid == fid_nodata:
                        continue
                    found_fids.add(fid)
                    # get the pixels that fall within the feature
                    feature_data = data_block[fid_block == fid]
                    nodata_mask = array_equals_nodata(feature_data, nodata)
                    aggregate_stats_list[i][fid]['nodata_count'] += nodata_mask.sum()
                    if ignore_nodata:
                        feature_data = feature_data[~nodata_mask]
                    if feature_data.size == 0:
                        continue

                    # compute stats
                    if aggregate_stats_list[i][fid]['min'] is None:
                        # initialize min/max to an arbitrary data value
                        aggregate_stats_list[i][fid]['min'] = feature_data[0]
                        aggregate_stats_list[i][fid]['max'] = feature_data[0]
                    aggregate_stats_list[i][fid]['min'] = min(
                        feature_data.min(), aggregate_stats_list[i][fid]['min'])
                    aggregate_stats_list[i][fid]['max'] = max(
                        feature_data.max(), aggregate_stats_list[i][fid]['max'])
                    aggregate_stats_list[i][fid]['sum'] += feature_data.sum()
                    aggregate_stats_list[i][fid]['count'] += feature_data.size
                    if include_value_counts:
                        # .update() here is operating on a Counter, so values are
                        # ADDED, not replaced.
                        aggregate_stats_list[i][fid]['value_counts'].update(
                            dict(zip(*numpy.unique(
                                feature_data, return_counts=True))))
        fid_band, fid_raster = None, None
        # Handle edge cases: features that have a geometry but do not
        # overlap the center point of any pixel will not be captured by the
        # method above.
        unset_fids = valid_fid_set.difference(found_fids)
        x_origin, pixel_width, _, y_origin, _, pixel_height = data_source.GetGeoTransform()
        # subtract 1 because bands are 1-indexed
        raster_nodata = get_raster_info(raster_path)['nodata'][band - 1]
        target_layer = target_vector.GetLayerByName(target_layer_id)
        for unset_fid in unset_fids:
            # Look up by the new FID
            # FIDs in target_layer may not be the same as in the input layer
            unset_feat = target_layer.GetFeature(original_to_new_fid_map[unset_fid])
            unset_geom_ref = unset_feat.GetGeometryRef()

            geom_x_min, geom_x_max, geom_y_min, geom_y_max = unset_geom_ref.GetEnvelope()
            unset_geom_ref, unset_feat = None, None
            if pixel_width < 0:
                geom_x_min, geom_x_max = geom_x_max, geom_x_min
            if pixel_height < 0:
                geom_y_min, geom_y_max = geom_y_max, geom_y_min

            xoff = int((geom_x_min - x_origin) / pixel_width)
            yoff = int((geom_y_min - y_origin) / pixel_height)
            win_xsize = int(numpy.ceil(
                (geom_x_max - x_origin) / pixel_width)) - xoff
            win_ysize = int(numpy.ceil(
                (geom_y_max - y_origin) / pixel_height)) - yoff

            # clamp offset to the side of the raster if it's negative
            if xoff < 0:
                win_xsize += xoff
                xoff = 0
            if yoff < 0:
                win_ysize += yoff
                yoff = 0

            # clamp the window to the side of the raster if too big
            if xoff + win_xsize > data_band.XSize:
                win_xsize = data_band.XSize - xoff
            if yoff + win_ysize > data_band.YSize:
                win_ysize = data_band.YSize - yoff

            if win_xsize <= 0 or win_ysize <= 0:
                continue

            # here we consider the pixels that intersect with the geometry's
            # bounding box as being the proxy for the intersection with the
            # polygon itself. This is not a bad approximation since the case
            # that caused the polygon to be skipped in the first phase is that it
            # is as small as a pixel. There could be some degenerate cases that
            # make this estimation very wrong, but we do not know of any that
            # would come from natural data. If you do encounter such a dataset
            # please email the description and datset to jdouglass@stanford.edu.
            unset_fid_block = data_band.ReadAsArray(
                xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize)
            unset_fid_nodata_mask = array_equals_nodata(
                unset_fid_block, raster_nodata)
            if ignore_nodata:
                unset_fid_block = unset_fid_block[~unset_fid_nodata_mask]
            if unset_fid_block.size == 0:
                aggregate_stats_list[i][unset_fid]['min'] = 0
                aggregate_stats_list[i][unset_fid]['max'] = 0
                aggregate_stats_list[i][unset_fid]['sum'] = 0
            else:
                aggregate_stats_list[i][unset_fid]['min'] = numpy.min(
                    unset_fid_block)
                aggregate_stats_list[i][unset_fid]['max'] = numpy.max(
                    unset_fid_block)
                aggregate_stats_list[i][unset_fid]['sum'] = numpy.sum(
                    unset_fid_block)
            aggregate_stats_list[i][unset_fid]['count'] = unset_fid_block.size
            aggregate_stats_list[i][unset_fid]['nodata_count'] = numpy.sum(
                unset_fid_nodata_mask)
            if include_value_counts:
                # .update() here is operating on a Counter, so values are ADDED,
                # not replaced.
                aggregate_stats_list[i][unset_fid]['value_counts'].update(dict(
                    zip(*numpy.unique(unset_fid_block, return_counts=True))))

        # Convert counter object to dictionary
        if include_value_counts:
            for key, value in aggregate_stats_list[i].items():
                aggregate_stats_list[i][key]['value_counts'] = dict(
                    aggregate_stats_list[i][key]['value_counts'])
        LOGGER.info(
            f'all done processing polygon sets for {os.path.basename(aggregate_vector_path)}')

    # dereference gdal objects
    data_band, data_source = None, None
    disjoint_layer, target_layer, target_vector = None, None, None

    shutil.rmtree(temp_working_dir)
    if multi_raster_mode:
        return aggregate_stats_list
    else:
        return aggregate_stats_list[0]


@gdal_use_exceptions
def get_vector_info(vector_path, layer_id=0):
    """Get information about an GDAL vector.

    Args:
        vector_path (str): a path to a GDAL vector.
        layer_id (str/int): name or index of underlying layer to analyze.
            Defaults to 0.

    Raises:
        ValueError if ``vector_path`` does not exist on disk or cannot be
        opened as a gdal.OF_VECTOR.

    Return:
        raster_properties (dictionary):
            a dictionary with the following key-value pairs:

            * ``'projection_wkt'`` (string): projection of the vector in Well
              Known Text.
            * ``'bounding_box'`` (sequence): sequence of floats representing
              the bounding box in projected coordinates in the order
              [minx, miny, maxx, maxy].
            * ``'file_list'`` (sequence): sequence of string paths to the files
              that make up this vector.

    """
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    vector_properties = {}
    vector_properties['file_list'] = vector.GetFileList()
    layer = vector.GetLayer(iLayer=layer_id)
    # projection is same for all layers, so just use the first one
    spatial_ref = layer.GetSpatialRef()
    if spatial_ref:
        vector_projection_wkt = spatial_ref.ExportToWkt()
    else:
        vector_projection_wkt = None
    vector_properties['projection_wkt'] = vector_projection_wkt
    layer_bb = layer.GetExtent()
    layer = None
    vector = None
    # convert form [minx,maxx,miny,maxy] to [minx,miny,maxx,maxy]
    vector_properties['bounding_box'] = [layer_bb[i] for i in [0, 2, 1, 3]]
    return vector_properties


@gdal_use_exceptions
def get_raster_info(raster_path):
    """Get information about a GDAL raster (dataset).

    Args:
       raster_path (String): a path to a GDAL raster.

    Raises:
        ValueError
            if ``raster_path`` is not a file or cannot be opened as a
            ``gdal.OF_RASTER``.

    Return:
        raster_properties (dictionary):
            a dictionary with the properties stored under relevant keys.

        * ``'pixel_size'`` (tuple): (pixel x-size, pixel y-size)
          from geotransform.
        * ``'raster_size'`` (tuple):  number of raster pixels in (x, y)
          direction.
        * ``'nodata'`` (sequence): a sequence of the nodata values in the bands
          of the raster in the same order as increasing band index.
        * ``'n_bands'`` (int): number of bands in the raster.
        * ``'geotransform'`` (tuple): a 6-tuple representing the geotransform
          of (x orign, x-increase, xy-increase, y origin, yx-increase,
          y-increase).
        * ``'datatype'`` (int): An instance of an enumerated gdal.GDT_* int
          that represents the datatype of the raster.
        * ``'projection_wkt'`` (string): projection of the raster in Well Known
          Text.
        * ``'bounding_box'`` (sequence): sequence of floats representing the
          bounding box in projected coordinates in the order
          [minx, miny, maxx, maxy]
        * ``'block_size'`` (tuple): underlying x/y raster block size for
          efficient reading.
        * ``'numpy_type'`` (numpy type): this is the equivalent numpy datatype
          for the raster bands including signed bytes.
        * ``'overviews'`` (sequence): A list of (x, y) tuples for the
          number of pixels in the width and height of each overview level of
          the raster.
        * ``'file_list'`` (sequence): A list of files that make up this raster.

    """
    raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    raster_properties = {}
    raster_properties['file_list'] = raster.GetFileList()
    projection_wkt = raster.GetProjection()
    if not projection_wkt:
        projection_wkt = None
    raster_properties['projection_wkt'] = projection_wkt
    geo_transform = raster.GetGeoTransform()
    raster_properties['geotransform'] = geo_transform
    raster_properties['pixel_size'] = (geo_transform[1], geo_transform[5])
    raster_properties['raster_size'] = (
        raster.GetRasterBand(1).XSize,
        raster.GetRasterBand(1).YSize)
    raster_properties['n_bands'] = raster.RasterCount
    raster_properties['nodata'] = [
        raster.GetRasterBand(index).GetNoDataValue() for index in range(
            1, raster_properties['n_bands']+1)]

    # GDAL creates overviews for the whole raster but has overviews accessed
    # per band.  We assume that all bands have the same overviews.
    raster_properties['overviews'] = []
    for overview_index in range(raster.GetRasterBand(1).GetOverviewCount()):
        overview_band = raster.GetRasterBand(1).GetOverview(overview_index)
        raster_properties['overviews'].append((
            overview_band.XSize, overview_band.YSize))

    # blocksize is the same for all bands, so we can just get the first
    raster_properties['block_size'] = raster.GetRasterBand(1).GetBlockSize()

    # we dont' really know how the geotransform is laid out, all we can do is
    # calculate the x and y bounds, then take the appropriate min/max
    x_bounds = [
        geo_transform[0], geo_transform[0] +
        raster_properties['raster_size'][0] * geo_transform[1] +
        raster_properties['raster_size'][1] * geo_transform[2]]
    y_bounds = [
        geo_transform[3], geo_transform[3] +
        raster_properties['raster_size'][0] * geo_transform[4] +
        raster_properties['raster_size'][1] * geo_transform[5]]

    raster_properties['bounding_box'] = [
        numpy.min(x_bounds), numpy.min(y_bounds),
        numpy.max(x_bounds), numpy.max(y_bounds)]

    # datatype is the same for the whole raster, but is associated with band
    band = raster.GetRasterBand(1)
    raster_properties['datatype'] = band.DataType
    raster_properties['numpy_type'] = _gdal_to_numpy_type(
        band.DataType, band.GetMetadata('IMAGE_STRUCTURE'))
    band = None
    raster = None
    return raster_properties


@gdal_use_exceptions
def reproject_vector(
        base_vector_path, target_projection_wkt, target_path, layer_id=0,
        driver_name='ESRI Shapefile', copy_fields=True,
        target_layer_name=None,
        osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):
    """Reproject OGR DataSource (vector).

    Transforms the features of the base vector to the desired output
    projection in a new vector.

    Note:
        If the ESRI Shapefile driver is used, the ``target_layer_name``
        optional parameter is ignored. ESRI Shapefiles by definition use the
        filename to define the layer name.

    Args:
        base_vector_path (string): Path to the base shapefile to transform.
        target_projection_wkt (string): the desired output projection in Well
            Known Text (by layer.GetSpatialRef().ExportToWkt())
        target_path (string): the filepath to the transformed shapefile
        layer_id (str/int): name or index of layer in ``base_vector_path`` to
            reproject. Defaults to 0.
        driver_name (string): String to pass to ogr.GetDriverByName, defaults
            to 'ESRI Shapefile'.
        copy_fields (bool or iterable): If True, all the fields in
            ``base_vector_path`` will be copied to ``target_path`` during the
            reprojection step. If it is an iterable, it will contain the
            field names to exclusively copy. An unmatched fieldname will be
            ignored. If ``False`` no fields are copied into the new vector.
        target_layer_name=None (str): The name to use for the target layer in
            the new vector.  If ``None`` (the default), the layer name from the
            source layer will be used.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter
            should not be changed unless you know what you are doing.

    Returns:
        None
    """
    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)

    # if this file already exists, then remove it
    if os.path.isfile(target_path):
        LOGGER.warning(
            "%s already exists, removing and overwriting", target_path)
        os.remove(target_path)

    target_sr = osr.SpatialReference(target_projection_wkt)

    # create a new vector from the orginal_datasource
    target_driver = ogr.GetDriverByName(driver_name)
    target_vector = target_driver.CreateDataSource(target_path)

    layer = base_vector.GetLayer(layer_id)
    layer_dfn = layer.GetLayerDefn()

    # Create new layer for target_vector using same name and
    # geometry type from base vector but new projection
    layer_name = layer_dfn.GetName()
    if target_layer_name is not None:
        layer_name = target_layer_name
        if driver_name == 'ESRI Shapefile':
            target_file_basename = os.path.splitext(
                os.path.basename(target_path)[0])
            if layer_name != target_file_basename:
                LOGGER.warning(
                    f'Ignoring user-defined layer name {layer_name}. '
                    f'Defining a layer name is incompatible with the ESRI '
                    'Shapefile vector format.  Use the filename instead or '
                    'use a different vector format.')
    target_layer = target_vector.CreateLayer(
        layer_name, target_sr, layer_dfn.GetGeomType())

    # this will map the target field index to the base index it came from
    # in case we don't need to copy all the fields
    target_to_base_field_id_map = {}
    if copy_fields:
        # Get the number of fields in original_layer
        original_field_count = layer_dfn.GetFieldCount()
        # For every field that's copying, create a duplicate field in the
        # new layer

        for fld_index in range(original_field_count):
            original_field = layer_dfn.GetFieldDefn(fld_index)
            field_name = original_field.GetName()
            if copy_fields is True or field_name in copy_fields:
                target_field = ogr.FieldDefn(
                    field_name, original_field.GetType())
                target_layer.CreateField(target_field)
                target_to_base_field_id_map[fld_index] = len(
                    target_to_base_field_id_map)

    # Get the SR of the original_layer to use in transforming
    base_sr = layer.GetSpatialRef()

    base_sr.SetAxisMappingStrategy(osr_axis_mapping_strategy)
    target_sr.SetAxisMappingStrategy(osr_axis_mapping_strategy)

    # Create a coordinate transformation
    coord_trans = osr.CreateCoordinateTransformation(base_sr, target_sr)

    # Copy all of the features in layer to the new shapefile
    target_layer.StartTransaction()
    error_count = 0
    timed_logger = TimedLoggingAdapter(_LOGGING_PERIOD)
    LOGGER.info("starting reprojection")
    for feature_index, base_feature in enumerate(layer):
        timed_logger.info(
            "reprojection approximately %.1f%% complete on %s",
            100.0 * float(feature_index+1) / (layer.GetFeatureCount()),
            os.path.basename(target_path))

        try:
            geom = base_feature.GetGeometryRef()
            if geom is None:
                # we encountered this error occasionally when transforming
                # clipped global polygons.  Not clear what is happening but
                # perhaps a feature was retained that otherwise wouldn't have
                # been included in the clip
                error_count += 1
                continue

            # Transform geometry into format desired for the new projection
            error_code = geom.Transform(coord_trans)
            if error_code != 0:  # error
                # this could be caused by an out of range transformation
                # whatever the case, don't put the transformed poly into the
                # output set
                error_count += 1
                continue
        except RuntimeError as error:
            # RuntimeError: GDAL's base error when geometries cannot be
            # returned or transformed.
            LOGGER.debug("Skipping feature %s due to %s",
                         feature_index, str(error))
            error_count += 1
            continue

        # Copy original_datasource's feature and set as new shapes feature
        target_feature = ogr.Feature(target_layer.GetLayerDefn())
        target_feature.SetGeometry(geom)

        # For all the fields in the feature set the field values from the
        # source field
        for target_index, base_index in (
                target_to_base_field_id_map.items()):
            try:
                target_feature.SetField(
                    target_index, base_feature.GetField(base_index))
            except RuntimeError:
                try:
                    target_feature.SetFieldBinary(
                        target_index,
                        base_feature.GetFieldAsBinary(base_index))
                except RuntimeError as runtime_error:
                    LOGGER.debug(
                        f"Skipping copy field value for feature {feature_index}, "
                        f"field {layer_dfn.GetFieldDefn(base_index).GetName()} "
                        f"due to: {runtime_error}")

        target_layer.CreateFeature(target_feature)
        target_feature = None
        base_feature = None
    target_layer.CommitTransaction()
    LOGGER.info(
        "reprojection 100.0%% complete on %s", os.path.basename(target_path))
    if error_count > 0:
        LOGGER.warning(
            '%d features out of %d were unable to be transformed and are'
            ' not in the output vector at %s', error_count,
            layer.GetFeatureCount(), target_path)
    layer = None
    base_vector = None


@gdal_use_exceptions
def reclassify_raster(
        base_raster_path_band, value_map, target_raster_path, target_datatype,
        target_nodata, values_required=True,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Reclassify pixel values in a raster.

    A function to reclassify values in raster to any output type. By default
    the values except for nodata must be in ``value_map``. Include the
    base raster nodata value in ``value_map`` to reclassify nodata pixels
    to a valid value.

    Args:
        base_raster_path_band (tuple): a tuple including file path to a raster
            and the band index to operate over. ex: (path, band_index)
        value_map (dictionary): a dictionary of values of
            {source_value: dest_value, ...} where source_value's type is the
            same as the values in ``base_raster_path`` at band ``band_index``.
            Cannot be empty and must contain a mapping for all raster values
            if ``values_required=True``. If nodata is mapped, nodata will be
            reclassified, otherwise nodata will be set to ``target_nodata``.
        target_raster_path (string): target raster output path; overwritten if
            it exists
        target_datatype (gdal type): the numerical type for the target raster
        target_nodata (numerical type): the nodata value for the target raster.
            Must be the same type as target_datatype. All nodata pixels in
            the base raster will be reclassified to this value, unless the
            base raster notata values are present in ``value_map``.
        values_required (bool): If True, raise a ValueError if there is a
            value in the raster that is not found in ``value_map``.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None

    Raises:
        ReclassificationMissingValuesError
            if ``values_required`` is ``True``
            and a pixel value from ``base_raster_path_band`` is not a key in
            ``value_map``.
        ValueError
            - if ``value_map`` is empty
            - if ``base_raster_path_band`` is formatted incorrectly
            - if nodata value not set
        TypeError
            if there are non-numeric keys in ``value_map``

    """
    if len(value_map) == 0:
        raise ValueError("value_map must contain at least one value")
    if not _is_raster_path_band_formatted(base_raster_path_band):
        raise ValueError(
            "Expected a (path, band_id) tuple, instead got '%s'" %
            base_raster_path_band)
    # raise error if there are any non-numeric keys in value_map
    nonnumeric = [key for key in value_map
                  if not isinstance(key, (int, float, numpy.number))]
    if nonnumeric:
        raise TypeError(f"Non-numeric key(s) in value map: {nonnumeric}")

    raster_info = get_raster_info(base_raster_path_band[0])
    nodata = raster_info['nodata'][base_raster_path_band[1]-1]
    # If nodata was included in the value_map pop it from our lists
    # and handle it separately. Doing this on the off chance nodata is
    # a max or min float which can cause floating decimal discrepencies.
    value_map_copy = value_map.copy()
    nodata_dest_value = target_nodata
    if nodata is not None:
        for key, val in value_map.items():
            if numpy.isclose(key, nodata):
                nodata_dest_value = val
                del value_map_copy[key]
                break

    if target_nodata is None and nodata_dest_value is None:
        raise ValueError(
            "target_nodata was set to None and the base raster nodata"
            " value was not represented in the value_map. reclassify_raster"
            " does not assume the base raster nodata value should be used"
            " as the target_nodata value. Set target_nodata to a valid number"
            " and/or add a base raster nodata value mapping to value_map.")

    keys = sorted(numpy.array(list(value_map_copy.keys())))
    values = numpy.array([value_map_copy[x] for x in keys])

    numpy_dtype = _gdal_to_numpy_type(
        target_datatype, raster_driver_creation_tuple[1])

    def _map_dataset_to_value_op(original_values):
        """Convert a block of original values to the lookup values."""
        out_array = numpy.full(
            original_values.shape, target_nodata,
            dtype=numpy_dtype)
        if nodata is None:
            valid_mask = numpy.full(original_values.shape, True)
        else:
            valid_mask = ~numpy.isclose(original_values, nodata)
            out_array[~valid_mask] = nodata_dest_value

        if values_required:
            unique = numpy.unique(original_values[valid_mask])
            has_map = numpy.isin(unique, keys)
            if not all(has_map):
                missing_values = unique[~has_map]
                raise ReclassificationMissingValuesError(
                    missing_values, base_raster_path_band[0], value_map
                )
        index = numpy.digitize(original_values[valid_mask], keys, right=True)
        out_array[valid_mask] = values[index]
        return out_array

    raster_calculator(
        [base_raster_path_band], _map_dataset_to_value_op,
        target_raster_path, target_datatype, target_nodata,
        raster_driver_creation_tuple=raster_driver_creation_tuple)


@gdal_use_exceptions
def warp_raster(
        base_raster_path, target_pixel_size, target_raster_path,
        resample_method, target_bb=None, base_projection_wkt=None,
        target_projection_wkt=None, n_threads=None, mask_options=None,
        vector_mask_options=None, gdal_warp_options=None, working_dir=None,
        use_overview_level=-1,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
        osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):
    """Resize/resample raster to desired pixel size, bbox and projection.

    Args:
        base_raster_path (string): path to base raster.
        target_pixel_size (list/tuple): a two element sequence indicating
            the x and y pixel size in projected units.
        target_raster_path (string): the location of the resized and
            resampled raster.
        resample_method (string): the resampling algorithm. Must be a valid
            resampling algorithm for `gdal.WarpRaster`, one of:
            'rms | mode | sum | q1 | near | q3 | average | cubicspline |
            bilinear | max | med | min | cubic | lanczos'
        target_bb (sequence): if None, target bounding box is the same as the
            source bounding box.  Otherwise it's a sequence of float
            describing target bounding box in target coordinate system as
            [minx, miny, maxx, maxy].
        base_projection_wkt (string): if not None, interpret the projection of
            ``base_raster_path`` as this.
        target_projection_wkt (string): if not None, desired target projection
            in Well Known Text format.
        n_threads (int): optional, if not None this sets the ``N_THREADS``
            option for ``gdal.Warp``.
        mask_options (dict or None): optional. If None, no masking will
            be done.  If a dict, it is a dictionary of options relating to the
            dataset mask. Keys to this dictionary are:

            * ``'mask_vector_path'``: (str) path to the mask vector file. This
              vector will be automatically projected to the target
              projection if its base coordinate system does not match
              the target.  Where there are geometries in this vector, pixels in
              ``base_raster_path`` will propagate to ``target_raster_path``.
            * ``'mask_layer_id'``: (int/str) the layer index or name to use
              for masking, if this key is not in the dictionary the default
              is to use the layer at index 0.
            * ``'mask_vector_where_filter'``: (str) an SQL ``WHERE`` string
              that can be used to filter the geometry in the mask.
              Ex: 'id > 10' would use all features whose field value of 'id' is
              > 10.
            * ``'mask_raster_path'``: (str).  If present in the dict, all other
              keys in ``mask_options`` are ignored.  This string must be
              a path to a raster representing a validity mask, where pixel
              values of 1 indicate validity.  This raster must be in the same
              projection and have the same dimensions as the target warped
              raster.  The general (and easiest) use case for ``warp_raster``
              is to use ``'mask_vector_path'`` instead.

        vector_mask_options=None (dict): Alias for ``mask_options``.
            This option is deprecated and will be removed in a future release
            of ``pygeoprocessing``.
        gdal_warp_options (sequence): if present, the contents of this list
            are passed to the ``warpOptions`` parameter of ``gdal.Warp``. See
            the GDAL Warp documentation for valid options.
        working_dir (string): if defined uses this directory to make
            temporary working files for calculation. Otherwise uses system's
            temp directory.
        use_overview_level=-1 (int/str): The overview level to use for warping.
            A value of ``-1`` (the default) indicates that the base raster
            should be used for the source pixels. A value of ``'AUTO'``
            will make GDAL select the overview with the resolution that is
            closest to the target pixel size and warp using that overview's
            pixel values.  Any other integer indicates that that overview index
            should be used.  For example, suppose the raster has overviews at
            levels 2, 4 and 8.  To use level 2, set ``use_overview_level=0``.
            To use level 8, set ``use_overview_level=2``.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter
            should not be changed unless you know what you are doing.

    Return:
        None

    Raises:
        ValueError
            if ``pixel_size`` is not a 2 element sequence of numbers.
        ValueError
            if ``mask_options`` is not None but the
            ``mask_vector_path`` is undefined or doesn't point to a valid
            file.
        ValueError
            if either ``base_raster_path`` or ``target_raster_path`` are
            not strings.
    """
    for path_key in ['base_raster_path', 'target_raster_path']:
        if not isinstance(locals()[path_key], str):
            raise ValueError('%s must be a string', path_key)

    _assert_is_valid_pixel_size(target_pixel_size)

    base_raster_info = get_raster_info(base_raster_path)
    if target_projection_wkt is None:
        target_projection_wkt = base_raster_info['projection_wkt']

    if vector_mask_options is not None:
        warnings.warn('The vector_mask_options parameter is deprecated and '
                      'will be removed in a future release of '
                      'pygeoprocessing. Please use mask_options instead.',
                      DeprecationWarning)
        mask_options = vector_mask_options

    if target_bb is None:
        # ensure it's a sequence so we can modify it
        working_bb = list(get_raster_info(base_raster_path)['bounding_box'])
        # transform the working_bb if target_projection_wkt is not None
        if target_projection_wkt is not None:
            LOGGER.debug(
                "transforming bounding box from %s ", working_bb)
            working_bb = transform_bounding_box(
                base_raster_info['bounding_box'],
                base_raster_info['projection_wkt'], target_projection_wkt)
            LOGGER.debug(
                "transforming bounding to %s ", working_bb)
    else:
        # ensure it's a sequence so we can modify it
        working_bb = list(target_bb)

    # determine the raster size that bounds the input bounding box and then
    # adjust the bounding box to be that size
    target_x_size = int(abs(
        float(working_bb[2] - working_bb[0]) / target_pixel_size[0]))
    target_y_size = int(abs(
        float(working_bb[3] - working_bb[1]) / target_pixel_size[1]))

    # sometimes bounding boxes are numerically perfect, this checks for that
    x_residual = (
        abs(target_x_size * target_pixel_size[0]) -
        (working_bb[2] - working_bb[0]))
    if not numpy.isclose(x_residual, 0.0):
        target_x_size += 1
    y_residual = (
        abs(target_y_size * target_pixel_size[1]) -
        (working_bb[3] - working_bb[1]))
    if not numpy.isclose(y_residual, 0.0):
        target_y_size += 1

    if target_x_size == 0:
        LOGGER.warning(
            "bounding_box is so small that x dimension rounds to 0; "
            "clamping to 1.")
        target_x_size = 1
    if target_y_size == 0:
        LOGGER.warning(
            "bounding_box is so small that y dimension rounds to 0; "
            "clamping to 1.")
        target_y_size = 1

    # this ensures the bounding boxes perfectly fit a multiple of the target
    # pixel size
    working_bb[2] = working_bb[0] + abs(target_pixel_size[0] * target_x_size)
    working_bb[3] = working_bb[1] + abs(target_pixel_size[1] * target_y_size)

    reproject_callback = _make_logger_callback(
        "Warp %.1f%% complete %s")

    warp_options = []
    if n_threads:
        warp_options.append('NUM_THREADS=%d' % n_threads)
    if gdal_warp_options:
        warp_options.extend(gdal_warp_options)

    mask_vector_path = None
    mask_layer_id = 0
    mask_vector_where_filter = None
    if mask_options:
        if 'mask_raster_path' not in mask_options:
            # translate pygeoprocessing terminology into GDAL warp options.
            if 'mask_vector_path' not in mask_options:
                raise ValueError(
                    'mask_options passed, but no value for '
                    '"mask_vector_path": %s', mask_options)
            mask_vector_path = mask_options['mask_vector_path']
            if not os.path.exists(mask_vector_path):
                raise ValueError(
                    'The mask vector at %s was not found.', mask_vector_path)
            if 'mask_layer_id' in mask_options:
                mask_layer_id = mask_options['mask_layer_id']
            if 'mask_vector_where_filter' in mask_options:
                mask_vector_where_filter = (
                    mask_options['mask_vector_where_filter'])

    if mask_options:
        temp_working_dir = tempfile.mkdtemp(dir=working_dir)
        warped_raster_path = os.path.join(
            temp_working_dir, os.path.basename(target_raster_path).replace(
                '.tif', '_nonmasked.tif'))
    else:
        # if there is no vector path the result is the warp
        warped_raster_path = target_raster_path
    base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)

    raster_creation_options = list(raster_driver_creation_tuple[1])
    _, type_creation_options = _numpy_to_gdal_type(
        base_raster_info['numpy_type'])
    raster_creation_options += type_creation_options

    gdal.Warp(
        warped_raster_path, base_raster,
        format=raster_driver_creation_tuple[0],
        outputBounds=working_bb,
        xRes=abs(target_pixel_size[0]),
        yRes=abs(target_pixel_size[1]),
        resampleAlg=resample_method,
        outputBoundsSRS=target_projection_wkt,
        srcSRS=base_projection_wkt,
        dstSRS=target_projection_wkt,
        multithread=True if warp_options else False,
        warpOptions=warp_options,
        overviewLevel=use_overview_level,
        creationOptions=raster_creation_options,
        callback=reproject_callback,
        callback_data=[target_raster_path])

    if mask_options:
        if 'mask_raster_path' in mask_options:
            # If the user provided a mask raster, use that directly; assume
            # it's been rasterized correctly.
            source_raster_info = get_raster_info(warped_raster_path)
            source_nodata = source_raster_info['nodata'][0]

            def _mask_values(array, mask):
                output = numpy.full(array.shape, source_nodata)
                valid_mask = (
                    mask == 1 & ~array_equals_nodata(array, source_nodata))
                output[valid_mask] = array[valid_mask]
                return output

            raster_calculator(
                [(warped_raster_path, 1),
                 (mask_options['mask_raster_path'], 1)],
                _mask_values, target_raster_path,
                source_raster_info['datatype'], source_nodata)
        else:
            # If the user did not provide a mask in raster form, we can just
            # call down to ``mask_raster``, which will rasterize the vector and
            # then mask out pixels in ``warped_raster_path`` for us.
            updated_raster_driver_creation_tuple = (
                raster_driver_creation_tuple[0],
                tuple(raster_creation_options))
            mask_raster(
                (warped_raster_path, 1),
                mask_options['mask_vector_path'],
                target_raster_path,
                mask_layer_id=mask_layer_id,
                where_clause=mask_vector_where_filter,
                target_mask_value=None, working_dir=temp_working_dir,
                all_touched=False,
                raster_driver_creation_tuple=(
                    updated_raster_driver_creation_tuple))

        shutil.rmtree(temp_working_dir)


@gdal_use_exceptions
def rasterize(
        vector_path, target_raster_path, burn_values=None, option_list=None,
        layer_id=0, where_clause=None):
    """Project a vector onto an existing raster.

    Burn the layer at ``layer_id`` in ``vector_path`` to an existing
    raster at ``target_raster_path_band``.

    Args:
        vector_path (string): filepath to vector to rasterize.
        target_raster_path (string): path to an existing raster to burn vector
            into.  Can have multiple bands.
        burn_values (list/tuple): optional sequence of values to burn into
            each band of the raster.  If used, should have the same length as
            number of bands at the ``target_raster_path`` raster.  If ``None``
            then ``option_list`` must have a valid value.
        option_list (list/tuple): optional a sequence of burn options, if None
            then a valid value for ``burn_values`` must exist. Otherwise, each
            element is a string of the form:

            * ``"ATTRIBUTE=?"``: Identifies an attribute field on the features
              to be used for a burn in value. The value will be burned into all
              output bands. If specified, ``burn_values`` will not be used and
              can be None.
            * ``"CHUNKYSIZE=?"``: The height in lines of the chunk to operate
              on. The larger the chunk size the less times we need to make a
              pass through all the shapes. If it is not set or set to zero the
              default chunk size will be used. Default size will be estimated
              based on the GDAL cache buffer size using formula:
              ``cache_size_bytes/scanline_size_bytes``, so the chunk will not
              exceed the cache.
            * ``"ALL_TOUCHED=TRUE/FALSE"``: May be set to ``TRUE`` to set all
              pixels touched by the line or polygons, not just those whose
              center is within the polygon or that are selected by Brezenhams
              line algorithm. Defaults to ``FALSE``.
            * ``"BURN_VALUE_FROM"``: May be set to "Z" to use the Z values of
              the geometries. The value from burn_values or the
              attribute field value is added to this before burning. In
              default case dfBurnValue is burned as it is (richpsharp:
              note, I'm not sure what this means, but copied from formal
              docs). This is implemented properly only for points and
              lines for now. Polygons will be burned using the Z value
              from the first point.
            * ``"MERGE_ALG=REPLACE/ADD"``: REPLACE results in overwriting of
              value, while ADD adds the new value to the existing
              raster, suitable for heatmaps for instance.

            Example::

                ["ATTRIBUTE=npv", "ALL_TOUCHED=TRUE"]

        layer_id (str/int): name or index of the layer to rasterize. Defaults
            to 0.
        where_clause (str): If not None, is an SQL query-like string to filter
            which features are used to rasterize, (e.x. where="value=1").

    Return:
        None
    """
    raster = gdal.OpenEx(target_raster_path, gdal.GA_Update | gdal.OF_RASTER)
    if raster is None:
        raise ValueError(
            "%s doesn't exist, but needed to rasterize." % target_raster_path)

    rasterize_callback = _make_logger_callback(
        "pygeoprocessing.rasterize RasterizeLayer %.1f%% complete %s")

    if burn_values is None:
        burn_values = []
    if option_list is None:
        option_list = []

    if not burn_values and not option_list:
        raise ValueError(
            "Neither `burn_values` nor `option_list` is set. At least "
            "one must have a value.")

    if not isinstance(burn_values, (list, tuple)):
        raise ValueError(
            "`burn_values` is not a list/tuple, the value passed is '%s'",
            repr(burn_values))

    if not isinstance(option_list, (list, tuple)):
        raise ValueError(
            "`option_list` is not a list/tuple, the value passed is '%s'",
            repr(option_list))

    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer(layer_id)
    if where_clause:
        layer.SetAttributeFilter(where_clause)

    try:
        result = gdal.RasterizeLayer(
            raster, [1], layer, burn_values=burn_values,
            options=option_list, callback=rasterize_callback)
    except Exception:
        # something bad happened, but still clean up
        # this case came out of a flaky test condition where the raster
        # would still be in use by the rasterize layer function
        LOGGER.exception('bad error on rasterizelayer')
        result = -1

    layer = None
    vector = None

    if result != 0:
        # need this __swig_destroy__ because we sometimes encounter a flaky
        # test where the path to the raster cannot be cleaned up because
        # it is still in use somewhere, likely a bug in gdal.RasterizeLayer
        # note it is only invoked if there is a serious error
        gdal.Dataset.__swig_destroy__(raster)
        raise RuntimeError('Rasterize returned a nonzero exit code.')
    raster = None


@gdal_use_exceptions
def calculate_disjoint_polygon_set(
        vector_path, layer_id=0, bounding_box=None,
        geometries_may_touch=False):
    """Create a sequence of sets of polygons that don't overlap.

    Determining the minimal number of those sets is an np-complete problem so
    this is an approximation that builds up sets of maximal subsets.

    Args:
        vector_path (string): a path to an OGR vector.
        layer_id (str/int): name or index of underlying layer in
            ``vector_path`` to calculate disjoint set. Defaults to 0.
        bounding_box (sequence): sequence of floats representing a bounding
            box to filter any polygons by. If a feature in ``vector_path``
            does not intersect this bounding box it will not be considered
            in the disjoint calculation. Coordinates are in the order
            [minx, miny, maxx, maxy].
        geometries_may_touch=False(bool): If ``True``, geometries in a subset
            are allowed to have touching boundaries, but are not allowed to
            have intersecting interiors.  If ``False`` (the default), no
            geometries in a subset may intersect in any way.

    Return:
        subset_list (sequence): sequence of sets of FIDs from vector_path

    """
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    vector_layer = vector.GetLayer(layer_id)
    feature_count = vector_layer.GetFeatureCount()

    if feature_count == 0:
        raise RuntimeError('Vector must have geometries but does not: %s'
                           % vector_path)

    LOGGER.info("build shapely polygon list")
    timed_logger = TimedLoggingAdapter(_LOGGING_PERIOD)

    if bounding_box is None:
        bounding_box = get_vector_info(vector_path)['bounding_box']
    bounding_box = shapely.prepared.prep(shapely.geometry.box(*bounding_box))

    shapely_polygon_lookup = {}
    for poly_feat in vector_layer:
        poly_geom_ref = poly_feat.GetGeometryRef()
        if poly_geom_ref is None:
            LOGGER.warning(
                f'no geometry in {vector_path} FID: {poly_feat.GetFID()}, '
                'skipping...')
            continue
        if poly_geom_ref.IsEmpty():
            LOGGER.warning(
                f'empty geometry in {vector_path} FID: {poly_feat.GetFID()}, '
                'skipping...')
            continue
        # with GDAL>=3.3.0 ExportToWkb returns a bytearray instead of bytes
        shapely_polygon_lookup[poly_feat.GetFID()] = (
            shapely.wkb.loads(bytes(poly_geom_ref.ExportToWkb())))
        poly_geom_ref = None
    poly_feat = None

    LOGGER.info("build shapely rtree index")
    r_tree_index_stream = [
        (poly_fid, poly.bounds, None)
        for poly_fid, poly in shapely_polygon_lookup.items()
        if bounding_box.intersects(poly)]
    if r_tree_index_stream:
        poly_rtree_index = rtree.index.Index(r_tree_index_stream)
    else:
        LOGGER.warning("no polygons intersected the bounding box")
        return []

    vector_layer = None
    vector = None
    LOGGER.info(
        'poly feature lookup 100.0%% complete on %s',
        os.path.basename(vector_path))

    LOGGER.info('build poly intersection lookup')
    poly_intersect_lookup = collections.defaultdict(set)
    for poly_index, (poly_fid, poly_geom) in enumerate(
            shapely_polygon_lookup.items()):
        timed_logger.info(
            "poly intersection lookup approximately %.1f%% complete "
            "on %s", 100.0 * float(poly_index+1) / len(
                shapely_polygon_lookup), os.path.basename(vector_path))
        possible_intersection_set = list(poly_rtree_index.intersection(
            poly_geom.bounds))
        # no reason to prep the polygon to intersect itself
        if len(possible_intersection_set) > 1:
            polygon = shapely.prepared.prep(poly_geom)
        else:
            polygon = poly_geom
        for intersect_poly_fid in possible_intersection_set:
            # If geometries touch (share 1+ boundary point), then do not count
            # it as an intersection.
            if geometries_may_touch and polygon.touches(
                    shapely_polygon_lookup[intersect_poly_fid]):
                continue

            if intersect_poly_fid == poly_fid or polygon.intersects(
                    shapely_polygon_lookup[intersect_poly_fid]):
                poly_intersect_lookup[poly_fid].add(intersect_poly_fid)
        polygon = None
    LOGGER.info(
        'poly intersection feature lookup 100.0%% complete on %s',
        os.path.basename(vector_path))

    # Build maximal subsets
    subset_list = []
    while len(poly_intersect_lookup) > 0:
        # sort polygons by increasing number of intersections
        intersections_list = [
            (len(poly_intersect_set), poly_fid, poly_intersect_set)
            for poly_fid, poly_intersect_set in
            poly_intersect_lookup.items()]
        intersections_list.sort()

        # build maximal subset
        maximal_set = set()
        for _, poly_fid, poly_intersect_set in intersections_list:
            timed_logger.info(
                "maximal subset build approximately %.1f%% complete "
                "on %s", 100.0 * float(
                    feature_count - len(poly_intersect_lookup)) /
                feature_count, os.path.basename(vector_path))
            if not poly_intersect_set.intersection(maximal_set):
                # no intersection, add poly_fid to the maximal set and remove
                # the polygon from the lookup
                maximal_set.add(poly_fid)
                del poly_intersect_lookup[poly_fid]
        # remove all the polygons from intersections once they're computed
        for poly_fid, poly_intersect_set in poly_intersect_lookup.items():
            poly_intersect_lookup[poly_fid] = (
                poly_intersect_set.difference(maximal_set))
        subset_list.append(maximal_set)
    LOGGER.info(
        'maximal subset build 100.0%% complete on %s',
        os.path.basename(vector_path))
    return subset_list


def distance_transform_edt(
        base_region_raster_path_band, target_distance_raster_path,
        sampling_distance=(1., 1.), working_dir=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
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

    Args:
        base_region_raster_path_band (tuple): a tuple including file path to a
            raster and the band index to define the base region pixels. Any
            pixel  that is not 0 and nodata are considered to be part of the
            region.
        target_distance_raster_path (string): path to the target raster that
            is the exact euclidean distance transform from any pixel in the
            base raster that is not nodata and not 0. The units are in
            ``(pixel distance * sampling_distance)``.
        sampling_distance (tuple/list): an optional parameter used to scale
            the pixel distances when calculating the distance transform.
            Defaults to (1.0, 1.0). First element indicates the distance
            traveled in the x direction when changing a column index, and the
            second element in y when changing a row index. Both values must
            be > 0.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None
    """
    working_raster_paths = {}
    for raster_prefix in ['region_mask_raster', 'g_raster']:
        with tempfile.NamedTemporaryFile(
                prefix=raster_prefix, suffix='.tif', delete=False,
                dir=working_dir) as tmp_file:
            working_raster_paths[raster_prefix] = tmp_file.name
    nodata = (get_raster_info(base_region_raster_path_band[0])['nodata'])[
        base_region_raster_path_band[1]-1]
    nodata_out = 255

    def mask_op(base_array):
        """Convert base_array to 1 if not 0 and nodata, 0 otherwise."""
        if nodata is not None:
            return ~numpy.isclose(base_array, nodata) & (base_array != 0)
        else:
            return base_array != 0

    if not isinstance(sampling_distance, (tuple, list)):
        raise ValueError(
            "`sampling_distance` should be a tuple/list, instead it's %s" % (
                type(sampling_distance)))

    sample_d_x, sample_d_y = sampling_distance
    if sample_d_x <= 0. or sample_d_y <= 0.:
        raise ValueError(
            "Sample distances must be > 0.0, instead got %s",
            sampling_distance)

    raster_calculator(
        [base_region_raster_path_band], mask_op,
        working_raster_paths['region_mask_raster'], gdal.GDT_Byte, nodata_out,
        calc_raster_stats=False,
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    geoprocessing_core._distance_transform_edt(
        working_raster_paths['region_mask_raster'],
        working_raster_paths['g_raster'], sampling_distance[0],
        sampling_distance[1], target_distance_raster_path,
        raster_driver_creation_tuple)

    for path in working_raster_paths.values():
        try:
            os.remove(path)
        except OSError:
            LOGGER.warning("couldn't remove file %s", path)


def _next_regular(base):
    """Find the next regular number greater than or equal to base.

    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    This source was taken directly from scipy.signaltools and saves us from
    having to access a protected member in a library that could change in
    future releases:

    https://github.com/scipy/scipy/blob/v0.17.1/scipy/signal/signaltools.py#L211

    Args:
        base (int): a positive integer to start to find the next Hamming
            number.

    Return:
        The next regular number greater than or equal to ``base``.

    """
    if base <= 6:
        return base

    # Quickly check if it's already a power of 2
    if not (base & (base-1)):
        return base

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < base:
        p35 = p5
        while p35 < base:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(base / p35))
            quotient = -(-base // p35)

            # Quickly find next power of 2 >= quotient
            p2 = 2**((quotient - 1).bit_length())

            N = p2 * p35
            if N == base:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == base:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == base:
            return p5
    if p5 < match:
        match = p5
    return match


@gdal_use_exceptions
def convolve_2d(
        signal_path_band, kernel_path_band, target_path,
        ignore_nodata_and_edges=False, mask_nodata=True,
        normalize_kernel=False, target_datatype=gdal.GDT_Float64,
        target_nodata=None, working_dir=None, set_tol_to_zero=1e-8,
        max_timeout=_MAX_TIMEOUT,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Convolve 2D kernel over 2D signal.

    Convolves the raster in ``kernel_path_band`` over ``signal_path_band``.
    Nodata values are treated as 0.0 during the convolution and masked to
    nodata for the output result where ``signal_path`` has nodata.

    Note with default values, boundary effects can be seen in the result where
    the kernel would hang off the edge of the raster or in regions with
    nodata pixels. The function would treat these areas as values with "0.0"
    by default thus pulling the total convolution down in these areas. This
    is similar to setting ``mode='same'`` in Numpy's ``convolve`` function:
    https://numpy.org/doc/stable/reference/generated/numpy.convolve.html

    This boundary effect can be avoided by setting
    ``ignore_nodata_and_edges=True`` which normalizes the target result by
    dynamically accounting for the number of valid signal pixels the kernel
    overlapped during the convolution step.

    Args:
        signal_path_band (tuple): a 2 tuple of the form
            (filepath to signal raster, band index).
        kernel_path_band (tuple): a 2 tuple of the form
            (filepath to kernel raster, band index), all pixel values should
            be valid -- output is not well defined if the kernel raster has
            nodata values.  To create a kernel raster, see the documentation
            and helper functions available in the
            :doc:`pygeoprocessing kernels module <pygeoprocessing.kernels>`.
        target_path (string): filepath to target raster that's the convolution
            of signal with kernel.  Output will be a single band raster of
            same size and projection as ``signal_path_band``. Any nodata pixels
            that align with ``signal_path_band`` will be set to nodata.
        ignore_nodata_and_edges (boolean): If true, any pixels that are equal
            to ``signal_path_band``'s nodata value or signal pixels where the
            kernel extends beyond the edge of the raster are not included when
            averaging the convolution filter. This has the effect of
            "spreading" the result as though nodata and edges beyond the
            bounds of the raster are 0s. If set to false this tends to "pull"
            the signal away from nodata holes or raster edges. Set this value
            to ``True`` to avoid distortions signal values near edges for
            large integrating kernels. It can be useful to set this value to
            ``True`` to fill nodata holes through distance weighted averaging.
            In this case ``mask_nodata`` must be set to ``False`` so the result
            does not mask out these areas which are filled in. When using this
            technique be careful of cases where the kernel does not extend
            over any areas except nodata holes, in this case the resulting
            values in these areas will be nonsensical numbers, perhaps
            numerical infinity or NaNs.
        normalize_kernel (boolean): If true, the result is divided by the
            sum of the kernel.
        mask_nodata (boolean): If true, ``target_path`` raster's output is
            nodata where ``signal_path_band``'s pixels were nodata. Note that
            setting ``ignore_nodata_and_edges`` to ``True`` while setting
            ``mask_nodata`` to ``False`` can allow for a technique involving
            distance weighted averaging to define areas that would otherwise
            be nodata. Be careful in cases where the kernel does not
            extend over any valid non-nodata area since the result can be
            numerical infinity or NaNs.
        target_datatype (GDAL type): a GDAL raster type to set the output
            raster type to, as well as the type to calculate the convolution
            in.  Defaults to GDT_Float64.  Note signed byte is not
            supported.
        target_nodata (int/float): nodata value to set on output raster.
            If ``target_datatype`` is not gdal.GDT_Float64, this value must
            be set.  Otherwise defaults to the minimum value of a float32.
        raster_creation_options (sequence): an argument list that will be
            passed to the GTiff driver for creating ``target_path``.  Useful
            for blocksizes, compression, and more.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run.
        set_tol_to_zero (float): any value within +- this from 0.0 will get
            set to 0.0. This is to handle numerical roundoff errors that
            sometimes result in "numerical zero", such as -1.782e-18 that
            cannot be tolerated by users of this function. If `None` no
            adjustment will be done to output values.
        max_timeout (float): maximum amount of time to wait for worker thread
            to terminate.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        ``None``

    Raises:
        ValueError:
            if ``ignore_nodata_and_edges`` is ``True`` and ``mask_nodata``
            is ``False``.
        ValueError:
            if ``signal_path_band`` or ``kernel_path_band`` is a row based
            blocksize which would result in slow runtimes due to gdal
            cache thrashing.
    """
    if target_datatype is not gdal.GDT_Float64 and target_nodata is None:
        raise ValueError(
            "`target_datatype` is set, but `target_nodata` is None. "
            "`target_nodata` must be set if `target_datatype` is not "
            "`gdal.GDT_Float64`.  `target_nodata` is set to None.")
    if target_nodata is None:
        target_nodata = float(numpy.finfo(numpy.float32).min)

    if ignore_nodata_and_edges and not mask_nodata:
        LOGGER.debug(
            'ignore_nodata_and_edges is True while mask_nodata is False -- '
            'this can yield a nonsensical result in areas where the kernel '
            'touches only nodata values.')

    bad_raster_path_list = []
    for raster_id, raster_path_band in [
            ('signal', signal_path_band), ('kernel', kernel_path_band)]:
        if (not _is_raster_path_band_formatted(raster_path_band)):
            bad_raster_path_list.append((raster_id, raster_path_band))
    if bad_raster_path_list:
        raise ValueError(
            "Expected raster path band sequences for the following arguments "
            f"but instead got: {bad_raster_path_list}")

    signal_raster_info = get_raster_info(signal_path_band[0])
    kernel_raster_info = get_raster_info(kernel_path_band[0])

    for info_dict, raster_path_band in zip(
            [signal_raster_info, kernel_raster_info],
            [signal_path_band, kernel_path_band]):
        if 1 in info_dict['block_size']:
            raise ValueError(
                f'{raster_path_band} has a row blocksize which can make this '
                f'function run very slow, create a square blocksize using '
                f'`warp_raster` or `align_and_resize_raster_stack` which '
                f'creates square blocksizes by default')

    # The nodata value is reset to a different value at the end of this
    # function. Here 0 is chosen as a default value since data are
    # incrementally added to the raster
    new_raster_from_base(
        signal_path_band[0], target_path, target_datatype, [0],
        raster_driver_creation_tuple=raster_driver_creation_tuple)

    n_cols_signal, n_rows_signal = signal_raster_info['raster_size']
    n_cols_kernel, n_rows_kernel = kernel_raster_info['raster_size']
    s_path_band = signal_path_band
    k_path_band = kernel_path_band
    s_nodata = signal_raster_info['nodata'][0]

    # we need the original signal raster info because we want the output to
    # be clipped and NODATA masked to it
    signal_raster = gdal.OpenEx(signal_path_band[0], gdal.OF_RASTER)
    signal_band = signal_raster.GetRasterBand(signal_path_band[1])
    # getting the offset list before it's opened for updating
    target_offset_list = list(iterblocks((target_path, 1), offset_only=True))
    target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER | gdal.GA_Update)
    target_band = target_raster.GetRasterBand(1)

    # if we're ignoring nodata, we need to make a parallel convolved signal
    # of the nodata mask
    if ignore_nodata_and_edges:
        raster_file, mask_raster_path = tempfile.mkstemp(
            suffix='.tif', prefix='convolved_mask',
            dir=os.path.dirname(target_path))
        os.close(raster_file)
        new_raster_from_base(
            signal_path_band[0], mask_raster_path, gdal.GDT_Float64,
            [0.0], raster_driver_creation_tuple=raster_driver_creation_tuple)
        mask_raster = gdal.OpenEx(
            mask_raster_path, gdal.GA_Update | gdal.OF_RASTER)
        mask_band = mask_raster.GetRasterBand(1)

    LOGGER.info('starting convolve')
    timed_logger = TimedLoggingAdapter(_LOGGING_PERIOD)

    # calculate the kernel sum for normalization
    kernel_nodata = kernel_raster_info['nodata'][0]
    kernel_sum = 0.0
    for _, kernel_block in iterblocks(kernel_path_band):
        if kernel_nodata is not None and ignore_nodata_and_edges:
            kernel_block[numpy.isclose(kernel_block, kernel_nodata)] = 0.0
        kernel_sum += numpy.sum(kernel_block)

    # limit the size of the work queue since a large kernel / signal with small
    # block size can have a large memory impact when queuing offset lists.
    work_queue = queue.Queue(10)
    signal_offset_list = list(iterblocks(s_path_band, offset_only=True))
    kernel_offset_list = list(iterblocks(k_path_band, offset_only=True))
    n_blocks = len(signal_offset_list) * len(kernel_offset_list)

    LOGGER.debug('start fill work queue thread')

    def _fill_work_queue():
        """Asynchronously fill the work queue."""
        LOGGER.debug('fill work queue')
        for signal_offset in signal_offset_list:
            for kernel_offset in kernel_offset_list:
                work_queue.put((signal_offset, kernel_offset))
        work_queue.put(None)
        LOGGER.debug('work queue full')

    fill_work_queue_worker = threading.Thread(
        target=_fill_work_queue)
    fill_work_queue_worker.daemon = True
    fill_work_queue_worker.start()

    # limit the size of the write queue so we don't accidentally load a whole
    # array into memory
    LOGGER.debug('start worker thread')
    write_queue = queue.Queue(10)
    worker = threading.Thread(
        target=_convolve_2d_worker,
        args=(
            signal_path_band, kernel_path_band,
            ignore_nodata_and_edges, normalize_kernel,
            set_tol_to_zero, work_queue, write_queue))
    worker.daemon = True
    worker.start()

    n_blocks_processed = 0
    LOGGER.info(f'{n_blocks} sent to workers, wait for worker results')
    while True:
        # the timeout guards against a worst case scenario where the
        # ``_convolve_2d_worker`` has crashed.
        try:
            write_payload = write_queue.get(timeout=max_timeout)
            if write_payload:
                (index_dict, result, mask_result,
                 left_index_raster, right_index_raster,
                 top_index_raster, bottom_index_raster,
                 left_index_result, right_index_result,
                 top_index_result, bottom_index_result) = write_payload
            else:
                worker.join(max_timeout)
                break
        except queue.Empty:
            # Shut down the worker thread.
            # The work queue only has 10 items in it at a time, so it's pretty
            # likely that we can preemptively shut it down by adding a ``None``
            # here and then have the queue not take too much longer to quit.
            work_queue.put(None)

            # Close thread-local raster objects
            signal_raster = signal_band = None
            target_raster = target_band = None
            mask_raster = mask_band = None
            LOGGER.exception("Worker timeout")
            raise RuntimeError(
                f"The convolution worker timed out after {max_timeout} "
                "seconds. Either the timeout is too low for the "
                "size of your data, or the worker has crashed")

        output_array = numpy.empty(
            (index_dict['win_ysize'], index_dict['win_xsize']),
            dtype=numpy.float32)

        # the inital data value in target_band is 0 because that is the
        # temporary nodata selected so that manual resetting of initial
        # data values weren't necessary. at the end of this function the
        # target nodata value is set to `target_nodata`.
        current_output = target_band.ReadAsArray(**index_dict)

        # read the signal block so we know where the nodata are
        potential_nodata_signal_array = signal_band.ReadAsArray(**index_dict)

        valid_mask = numpy.ones(
            potential_nodata_signal_array.shape, dtype=bool)

        # guard against a None nodata value
        if s_nodata is not None and mask_nodata:
            valid_mask[:] = (
                ~numpy.isclose(potential_nodata_signal_array, s_nodata))
        output_array[:] = target_nodata
        output_array[valid_mask] = (
            (result[top_index_result:bottom_index_result,
                    left_index_result:right_index_result])[valid_mask] +
            current_output[valid_mask])
        target_band.WriteArray(
            output_array, xoff=index_dict['xoff'],
            yoff=index_dict['yoff'])

        if ignore_nodata_and_edges:
            # we'll need to save off the mask convolution so we can divide
            # it in total later
            current_mask = mask_band.ReadAsArray(**index_dict)

            output_array[valid_mask] = (
                (mask_result[
                    top_index_result:bottom_index_result,
                    left_index_result:right_index_result])[valid_mask] +
                current_mask[valid_mask])
            mask_band.WriteArray(
                output_array, xoff=index_dict['xoff'],
                yoff=index_dict['yoff'])

        n_blocks_processed += 1
        timed_logger.info(
            "convolution worker approximately %.1f%% complete on %s",
            100.0 * float(n_blocks_processed) / (n_blocks),
            os.path.basename(target_path))

    LOGGER.info(
        f"convolution worker 100.0% complete on "
        f"{os.path.basename(target_path)}")

    target_band.FlushCache()
    if ignore_nodata_and_edges:
        signal_nodata = get_raster_info(signal_path_band[0])['nodata'][
            signal_path_band[1]-1]
        LOGGER.info(
            "need to normalize result so nodata values are not included")
        mask_pixels_processed = 0
        mask_band.FlushCache()
        for target_offset_data in target_offset_list:
            target_block = target_band.ReadAsArray(
                **target_offset_data).astype(numpy.float64)
            signal_block = signal_band.ReadAsArray(**target_offset_data)
            mask_block = mask_band.ReadAsArray(**target_offset_data)
            if mask_nodata and signal_nodata is not None:
                valid_mask = ~numpy.isclose(signal_block, signal_nodata)
            else:
                valid_mask = numpy.ones(target_block.shape, dtype=bool)
            valid_mask &= (mask_block > 0)
            # divide the target_band by the mask_band
            target_block[valid_mask] /= mask_block[valid_mask].astype(
                numpy.float64)

            # scale by kernel sum if necessary since mask division will
            # automatically normalize kernel
            if not normalize_kernel:
                target_block[valid_mask] *= kernel_sum

            target_band.WriteArray(
                target_block, xoff=target_offset_data['xoff'],
                yoff=target_offset_data['yoff'])

            mask_pixels_processed += target_block.size
            timed_logger.info(
                f"""convolution nodata normalizer approximately {
                100 * mask_pixels_processed / (n_cols_signal * n_rows_signal)
                :.1f}% complete on {os.path.basename(target_path)}""")

        mask_raster = None
        mask_band = None
        os.remove(mask_raster_path)
        LOGGER.info(
            f"convolution nodata normalize 100.0% complete on "
            f"{os.path.basename(target_path)}")

    # set the nodata value from 0 to a reasonable value for the result
    target_band.SetNoDataValue(target_nodata)

    target_band = None
    target_raster = None


def iterblocks(
        raster_path_band, largest_block=_LARGEST_ITERBLOCK,
        offset_only=False):
    """Iterate across all the memory blocks in the input raster.

    Result is a generator of block location information and numpy arrays.

    This is especially useful when a single value needs to be derived from the
    pixel values in a raster, such as the sum total of all pixel values, or
    a sequence of unique raster values.  In such cases, ``raster_local_op``
    is overkill, since it writes out a raster.

    As a generator, this can be combined multiple times with itertools.izip()
    to iterate 'simultaneously' over multiple rasters, though the user should
    be careful to do so only with prealigned rasters.

    Args:
        raster_path_band (tuple): a path/band index tuple to indicate
            which raster band iterblocks should iterate over.
        largest_block (int): Attempts to iterate over raster blocks with
            this many elements.  Useful in cases where the blocksize is
            relatively small, memory is available, and the function call
            overhead dominates the iteration.  Defaults to 2**20.  A value of
            anything less than the original blocksize of the raster will
            result in blocksizes equal to the original size.
        offset_only (boolean): defaults to False, if True ``iterblocks`` only
            returns offset dictionary and doesn't read any binary data from
            the raster.  This can be useful when iterating over writing to
            an output.

    Yields:
        If ``offset_only`` is false, on each iteration, a tuple containing a
        dict of block data and a 2-dimensional numpy array are
        yielded. The dict of block data has these attributes:

        * ``data['xoff']`` - The X offset of the upper-left-hand corner of the
          block.
        * ``data['yoff']`` - The Y offset of the upper-left-hand corner of the
          block.
        * ``data['win_xsize']`` - The width of the block.
        * ``data['win_ysize']`` - The height of the block.

        If ``offset_only`` is True, the function returns only the block offset
        data and does not attempt to read binary data from the raster.

    """
    # need to use context manager rather than decorator here because
    # the decorator doesn't work on generators
    with GDALUseExceptions():
        if not _is_raster_path_band_formatted(raster_path_band):
            raise ValueError(
                "`raster_path_band` not formatted as expected.  Expects "
                "(path, band_index), received %s" % repr(raster_path_band))
        raster = gdal.OpenEx(raster_path_band[0], gdal.OF_RASTER)
        band = raster.GetRasterBand(raster_path_band[1])
        block = band.GetBlockSize()
        cols_per_block = block[0]
        rows_per_block = block[1]

        n_cols = raster.RasterXSize
        n_rows = raster.RasterYSize

        block_area = cols_per_block * rows_per_block
        # try to make block wider
        if int(largest_block / block_area) > 0:
            width_factor = int(largest_block / block_area)
            cols_per_block *= width_factor
            if cols_per_block > n_cols:
                cols_per_block = n_cols
            block_area = cols_per_block * rows_per_block
        # try to make block taller
        if int(largest_block / block_area) > 0:
            height_factor = int(largest_block / block_area)
            rows_per_block *= height_factor
            if rows_per_block > n_rows:
                rows_per_block = n_rows

        n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
        n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

        for row_block_index in range(n_row_blocks):
            row_offset = row_block_index * rows_per_block
            row_block_width = n_rows - row_offset
            if row_block_width > rows_per_block:
                row_block_width = rows_per_block
            for col_block_index in range(n_col_blocks):
                col_offset = col_block_index * cols_per_block
                col_block_width = n_cols - col_offset
                if col_block_width > cols_per_block:
                    col_block_width = cols_per_block

                offset_dict = {
                    'xoff': col_offset,
                    'yoff': row_offset,
                    'win_xsize': col_block_width,
                    'win_ysize': row_block_width,
                }
                if offset_only:
                    yield offset_dict
                else:
                    yield (offset_dict, band.ReadAsArray(**offset_dict))

        band = None
        raster = None


@gdal_use_exceptions
def transform_bounding_box(
        bounding_box, base_projection_wkt, target_projection_wkt,
        edge_samples=11,
        osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):
    """Transform input bounding box to output projection.

    This transform accounts for the fact that the reprojected square bounding
    box might be warped in the new coordinate system.  To account for this,
    the function samples points along the original bounding box edges and
    attempts to make the largest bounding box around any transformed point
    on the edge whether corners or warped edges.

    Args:
        bounding_box (sequence): a sequence of 4 coordinates in ``base_epsg``
            coordinate system describing the bound in the order
            [xmin, ymin, xmax, ymax].
        base_projection_wkt (string): the spatial reference of the input
            coordinate system in Well Known Text.
        target_projection_wkt (string): the spatial reference of the desired
            output coordinate system in Well Known Text.
        edge_samples (int): the number of interpolated points along each
            bounding box edge to sample along. A value of 2 will sample just
            the corners while a value of 3 will also sample the corners and
            the midpoint.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This
            parameter should not be changed unless you know what you are
            doing.

    Return:
        A list of the form [xmin, ymin, xmax, ymax] that describes the largest
        fitting bounding box around the original warped bounding box in
        ``new_epsg`` coordinate system.

    Raises:
        ``ValueError`` if resulting transform yields non-finite coordinates.
        This would indicate an ill posed transform region that the user
        should address.

    """
    base_ref = osr.SpatialReference()
    base_ref.ImportFromWkt(base_projection_wkt)

    target_ref = osr.SpatialReference()
    target_ref.ImportFromWkt(target_projection_wkt)

    base_ref.SetAxisMappingStrategy(osr_axis_mapping_strategy)
    target_ref.SetAxisMappingStrategy(osr_axis_mapping_strategy)

    # Create a coordinate transformation
    transformer = osr.CreateCoordinateTransformation(base_ref, target_ref)

    def _transform_point(point):
        """Transform an (x,y) point tuple from base_ref to target_ref."""
        trans_x, trans_y, _ = (transformer.TransformPoint(*point))
        return (trans_x, trans_y)

    # The following list comprehension iterates over each edge of the bounding
    # box, divides each edge into ``edge_samples`` number of points, then
    # reduces that list to an appropriate ``bounding_fn`` given the edge.
    # For example the left edge needs to be the minimum x coordinate so
    # we generate ``edge_samples` number of points between the upper left and
    # lower left point, transform them all to the new coordinate system
    # then get the minimum x coordinate "min(p[0] ...)" of the batch.
    # points are numbered from 0 starting upper right as follows:
    # 0--3
    # |  |
    # 1--2
    p_0 = numpy.array((bounding_box[0], bounding_box[3]))
    p_1 = numpy.array((bounding_box[0], bounding_box[1]))
    p_2 = numpy.array((bounding_box[2], bounding_box[1]))
    p_3 = numpy.array((bounding_box[2], bounding_box[3]))
    raw_bounding_box = [
        bounding_fn(
            [_transform_point(
                p_a * v + p_b * (1 - v)) for v in numpy.linspace(
                    0, 1, edge_samples)])
        for p_a, p_b, bounding_fn in [
            (p_0, p_1, lambda p_list: min([p[0] for p in p_list])),
            (p_1, p_2, lambda p_list: min([p[1] for p in p_list])),
            (p_2, p_3, lambda p_list: max([p[0] for p in p_list])),
            (p_3, p_0, lambda p_list: max([p[1] for p in p_list]))]]

    # sometimes a transform will be so tight that a sampling around it may
    # flip the coordinate system. This flips it back. I found this when
    # transforming the bounding box of Gibraltar in a utm coordinate system
    # to lat/lng.
    minx, maxx = sorted([raw_bounding_box[0], raw_bounding_box[2]])
    miny, maxy = sorted([raw_bounding_box[1], raw_bounding_box[3]])
    transformed_bounding_box = [minx, miny, maxx, maxy]
    if not all(numpy.isfinite(numpy.array(transformed_bounding_box))):
        raise ValueError(
            f'Could not transform bounding box from base to target projection. '
            f'Some transformed coordinates are not finite: '
            f'{transformed_bounding_box}, base bounding box may not fit into '
            f'target coordinate projection system.\n'
            f'Original bounding box: {bounding_box}\n'
            f'Base projection: {base_projection_wkt}\n'
            f'Target projection: {target_projection_wkt}\n')
    return transformed_bounding_box


@gdal_use_exceptions
def mask_raster(
        base_raster_path_band, mask_vector_path, target_mask_raster_path,
        mask_layer_id=0, target_mask_value=None, working_dir=None,
        all_touched=False, where_clause=None,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Mask a raster band with a given vector.

    Args:
        base_raster_path_band (tuple): a (path, band number) tuple indicating
            the data to mask.
        mask_vector_path (path): path to a vector that will be used to mask
            anything outside of the polygon that overlaps with
            ``base_raster_path_band`` to ``target_mask_value`` if defined or
            else ``base_raster_path_band``'s nodata value.
        target_mask_raster_path (str): path to desired target raster that
            is a copy of ``base_raster_path_band`` except any pixels that do
            not intersect with ``mask_vector_path`` are set to
            ``target_mask_value`` or ``base_raster_path_band``'s nodata value
            if ``target_mask_value`` is None.
        mask_layer_id (str/int): an index or name to identify the mask
            geometry layer in ``mask_vector_path``, default is 0.
        target_mask_value (numeric): If not None, this value is written to
            any pixel in ``base_raster_path_band`` that does not intersect
            with ``mask_vector_path``. Otherwise the nodata value of
            ``base_raster_path_band`` is used.
        working_dir (str): this is a path to a directory that can be used to
            hold temporary files required to complete this operation.
        all_touched (bool): if False, a pixel is only masked if its centroid
            intersects with the mask. If True a pixel is masked if any point
            of the pixel intersects the polygon mask.
        where_clause (str): (optional) if not None, it is an SQL compatible
            where clause that can be used to filter the features that are used
            to mask the base raster.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None
    """
    with tempfile.NamedTemporaryFile(
            prefix='mask_raster', delete=False, suffix='.tif',
            dir=working_dir) as mask_raster_file:
        mask_raster_path = mask_raster_file.name

    new_raster_from_base(
        base_raster_path_band[0], mask_raster_path, gdal.GDT_Byte, [255],
        fill_value_list=[0],
        raster_driver_creation_tuple=raster_driver_creation_tuple)

    base_raster_info = get_raster_info(base_raster_path_band[0])

    rasterize(
        mask_vector_path, mask_raster_path, burn_values=[1],
        layer_id=mask_layer_id,
        option_list=[('ALL_TOUCHED=%s' % all_touched).upper()],
        where_clause=where_clause)

    base_nodata = base_raster_info['nodata'][base_raster_path_band[1]-1]

    if target_mask_value is None:
        mask_value = base_nodata
        if mask_value is None:
            LOGGER.warning(
                "No mask value was passed and target nodata is undefined, "
                "defaulting to 0 as the target mask value.")
            mask_value = 0
    else:
        mask_value = target_mask_value

    def mask_op(base_array, mask_array):
        result = numpy.copy(base_array)
        result[mask_array == 0] = mask_value
        return result

    raster_calculator(
        [base_raster_path_band, (mask_raster_path, 1)], mask_op,
        target_mask_raster_path, base_raster_info['datatype'], base_nodata,
        raster_driver_creation_tuple=raster_driver_creation_tuple)

    os.remove(mask_raster_path)


@gdal_use_exceptions
def _gdal_to_numpy_type(gdal_type, metadata):
    """Calculate the equivalent numpy datatype from a GDAL type and metadata.

    Args:
        gdal_type: GDAL.GDT_* data type code
        metadata: mapping or list of strings to check for the existence of
            the 'PIXELTYPE=SIGNEDBYTE' flag

    Returns:
        numpy.dtype that is the equivalent of the input gdal type

    Raises:
        ValueError if an unsupported data type is entered
    """
    if (GDAL_VERSION < (3, 7, 0) and gdal_type == gdal.GDT_Byte and
        (('PIXELTYPE=SIGNEDBYTE' in metadata) or
         ('PIXELTYPE' in metadata and metadata['PIXELTYPE'] == 'SIGNEDBYTE'))):
            return numpy.int8

    numpy_type = gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type)
    if numpy_type is None:
        raise ValueError(f"Unsupported DataType: {gdal_type}")
    return numpy_type


@gdal_use_exceptions
def _numpy_to_gdal_type(numpy_type):
    """Calculate the equivalent GDAL type and metadata from a numpy type.

    Args:
        numpy_type: numpy data type

    Returns:
        (gdal type, metadata) tuple. gdal type is a gdal.GDT_* type code.
        metadata is an empty list in most cases, or ['PIXELTYPE=SIGNEDBYTE']
        if needed to indicate a signed byte type.

    Raises:
        ValueError if an unsupported data type is entered
    """
    numpy_dtype = numpy.dtype(numpy_type)

    if GDAL_VERSION < (3, 7, 0) and numpy_dtype == numpy.dtype(numpy.int8):
        return gdal.GDT_Byte, ['PIXELTYPE=SIGNEDBYTE']

    gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(numpy_dtype)
    if gdal_type is None:
        raise ValueError(f"Unsupported DataType: {numpy_type}")
    return gdal_type, []


def merge_bounding_box_list(bounding_box_list, bounding_box_mode):
    """Create a single bounding box by union or intersection of the list.

    Args:
        bounding_box_list (sequence): a sequence of bounding box coordinates
            in the order [minx, miny, maxx, maxy].
        mode (string): either ``'union'`` or ``'intersection'`` for the
            corresponding reduction mode.

    Return:
        A four tuple bounding box that is the union or intersection of the
        input bounding boxes.

    Raises:
        ValueError
            if the bounding boxes in ``bounding_box_list`` do not
            intersect if the ``bounding_box_mode`` is 'intersection'.

    """
    def _merge_bounding_boxes(bb1, bb2, mode):
        """Merge two bounding boxes through union or intersection.

        Args:
            bb1, bb2 (sequence): sequence of float representing bounding box
                in the form bb=[minx,miny,maxx,maxy]
            mode (string); one of 'union' or 'intersection'

        Return:
            Reduced bounding box of bb1/bb2 depending on mode.

        """
        def _less_than_or_equal(x_val, y_val):
            return x_val if x_val <= y_val else y_val

        def _greater_than(x_val, y_val):
            return x_val if x_val > y_val else y_val

        if mode == "union":
            comparison_ops = [
                _less_than_or_equal, _less_than_or_equal,
                _greater_than, _greater_than]
        if mode == "intersection":
            comparison_ops = [
                _greater_than, _greater_than,
                _less_than_or_equal, _less_than_or_equal]

        bb_out = [op(x, y) for op, x, y in zip(comparison_ops, bb1, bb2)]
        return bb_out

    result_bb = functools.reduce(
        functools.partial(_merge_bounding_boxes, mode=bounding_box_mode),
        bounding_box_list)
    if result_bb[0] > result_bb[2] or result_bb[1] > result_bb[3]:
        raise ValueError(
            "Bounding boxes do not intersect. Base list: %s mode: %s "
            " result: %s" % (bounding_box_list, bounding_box_mode, result_bb))
    return result_bb


@gdal_use_exceptions
def get_gis_type(path):
    """Calculate the GIS type of the file located at ``path``.

    Args:
        path (str): path to a file on disk or network.

    Raises:
        ValueError
            if ``path`` is not a file or cannot be opened as a
            ``gdal.OF_RASTER`` or ``gdal.OF_VECTOR``.

    Return:
        A bitwise OR of all GIS types that PyGeoprocessing models, currently
        this is
        ``pygeoprocessing.RASTER_TYPE``, or ``pygeoprocessing.VECTOR_TYPE``.

    """
    from pygeoprocessing import RASTER_TYPE
    from pygeoprocessing import UNKNOWN_TYPE
    from pygeoprocessing import VECTOR_TYPE
    gis_type = UNKNOWN_TYPE
    try:
        gis_raster = gdal.OpenEx(path, gdal.OF_RASTER)
        gis_type |= RASTER_TYPE
        gis_raster = None
    except RuntimeError:
        pass
    try:
        gis_vector = gdal.OpenEx(path, gdal.OF_VECTOR)
        gis_type |= VECTOR_TYPE
        gis_vector = None
    except RuntimeError:
        pass
    if gis_type == UNKNOWN_TYPE:
        raise ValueError(
            f"Could not open {path} as a gdal.OF_RASTER or gdal.OF_VECTOR.")
    return gis_type


def _make_logger_callback(message):
    """Build a timed logger callback that prints ``message`` replaced.

    Args:
        message (string): a string that expects 2 placement %% variables,
            first for % complete from ``df_complete``, second from
            ``p_progress_arg[0]``.

    Return:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)

    """
    def logger_callback(df_complete, _, p_progress_arg):
        """Argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - logger_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and
                     logger_callback.total_time >= 5.0)):
                # In some multiprocess applications I was encountering a
                # ``p_progress_arg`` of None. This is unexpected and I suspect
                # was an issue for some kind of GDAL race condition. So I'm
                # guarding against it here and reporting an appropriate log
                # if it occurs.
                if p_progress_arg:
                    LOGGER.info(message, df_complete * 100, p_progress_arg[0])
                else:
                    LOGGER.info(message, df_complete * 100, '')
                logger_callback.last_time = current_time
                logger_callback.total_time += current_time
        except AttributeError:
            logger_callback.last_time = time.time()
            logger_callback.total_time = 0.0
        except Exception:
            LOGGER.exception("Unhandled error occurred while logging "
                             "progress.  df_complete: %s, p_progress_arg: %s",
                             df_complete, p_progress_arg)

    return logger_callback


def _is_raster_path_band_formatted(raster_path_band):
    """Return true if raster path band is a (str, int) tuple/list."""
    if not isinstance(raster_path_band, (list, tuple)):
        return False
    elif len(raster_path_band) != 2:
        return False
    elif not isinstance(raster_path_band[0], str):
        return False
    elif not isinstance(raster_path_band[1], int):
        return False
    else:
        return True


@gdal_use_exceptions
def _convolve_2d_worker(
        signal_path_band, kernel_path_band,
        ignore_nodata, normalize_kernel, set_tol_to_zero,
        work_queue, write_queue):
    """Worker function to be used by ``convolve_2d``.

    Args:
        signal_path_band (tuple): a 2 tuple of the form
            (filepath to signal raster, band index).
        kernel_path_band (tuple): a 2 tuple of the form
            (filepath to kernel raster, band index).
        ignore_nodata (boolean): If true, any pixels that are equal to
            ``signal_path_band``'s nodata value are not included when
            averaging the convolution filter.
        normalize_kernel (boolean): If true, the result is divided by the
            sum of the kernel.
        set_tol_to_zero (float): Value to test close to to determine if values
            are zero, and if so, set to zero.
        work_queue (Queue): will contain (signal_offset, kernel_offset)
            tuples that can be used to read raster blocks directly using
            GDAL ReadAsArray(**offset). Indicates the block to operate on.
        write_queue (Queue): mechanism to pass result back to the writer
            contains a (index_dict, result, mask_result,
                 left_index_raster, right_index_raster,
                 top_index_raster, bottom_index_raster,
                 left_index_result, right_index_result,
                 top_index_result, bottom_index_result) tuple that's used
            for writing and masking.

    Return:
        None
    """
    signal_raster = gdal.OpenEx(signal_path_band[0], gdal.OF_RASTER)
    kernel_raster = gdal.OpenEx(kernel_path_band[0], gdal.OF_RASTER)
    signal_band = signal_raster.GetRasterBand(signal_path_band[1])
    kernel_band = kernel_raster.GetRasterBand(kernel_path_band[1])

    signal_raster_info = get_raster_info(signal_path_band[0])
    kernel_raster_info = get_raster_info(kernel_path_band[0])

    n_cols_signal, n_rows_signal = signal_raster_info['raster_size']
    n_cols_kernel, n_rows_kernel = kernel_raster_info['raster_size']
    signal_nodata = signal_raster_info['nodata'][0]
    kernel_nodata = kernel_raster_info['nodata'][0]

    mask_result = None  # in case no mask is needed, variable is still defined

    # calculate the kernel sum for normalization
    kernel_sum = 0.0
    for _, kernel_block in iterblocks(kernel_path_band):
        if kernel_nodata is not None and ignore_nodata:
            kernel_block[numpy.isclose(kernel_block, kernel_nodata)] = 0.0
        kernel_sum += numpy.sum(kernel_block)

    while True:
        payload = work_queue.get()
        if payload is None:
            break

        signal_offset, kernel_offset = payload

        # ensure signal and kernel are internally float64 precision
        # irrespective of their base type
        signal_block = signal_band.ReadAsArray(**signal_offset).astype(
            numpy.float64)
        kernel_block = kernel_band.ReadAsArray(**kernel_offset).astype(
            numpy.float64)

        # don't ever convolve the nodata value
        if signal_nodata is not None:
            signal_nodata_mask = numpy.isclose(signal_block, signal_nodata)
            signal_block[signal_nodata_mask] = 0.0
            if not ignore_nodata:
                signal_nodata_mask[:] = 0
        else:
            signal_nodata_mask = numpy.zeros(
                signal_block.shape, dtype=bool)

        left_index_raster = (
            signal_offset['xoff'] - n_cols_kernel // 2 +
            kernel_offset['xoff'])
        right_index_raster = (
            signal_offset['xoff'] - n_cols_kernel // 2 +
            kernel_offset['xoff'] + signal_offset['win_xsize'] +
            kernel_offset['win_xsize'] - 1)
        top_index_raster = (
            signal_offset['yoff'] - n_rows_kernel // 2 +
            kernel_offset['yoff'])
        bottom_index_raster = (
            signal_offset['yoff'] - n_rows_kernel // 2 +
            kernel_offset['yoff'] + signal_offset['win_ysize'] +
            kernel_offset['win_ysize'] - 1)

        # it's possible that the piece of the integrating kernel
        # doesn't affect the final result, if so we should skip
        if (right_index_raster < 0 or
                bottom_index_raster < 0 or
                left_index_raster > n_cols_signal or
                top_index_raster > n_rows_signal):
            continue

        if kernel_nodata is not None:
            kernel_block[numpy.isclose(kernel_block, kernel_nodata)] = 0.0

        if normalize_kernel:
            kernel_block /= kernel_sum

        # determine the output convolve shape
        shape = (
            numpy.array(signal_block.shape) +
            numpy.array(kernel_block.shape) - 1)

        # add zero padding so FFT is fast
        fshape = [_next_regular(int(d)) for d in shape]
        f_axes_seq = range(len(fshape))

        signal_fft = numpy.fft.rfftn(signal_block, fshape, f_axes_seq)
        kernel_fft = numpy.fft.rfftn(kernel_block, fshape, f_axes_seq)

        # this variable determines the output slice that doesn't include
        # the padded array region made for fast FFTs.
        fslice = tuple([slice(0, int(sz)) for sz in shape])
        # classic FFT convolution
        result = numpy.fft.irfftn(
            signal_fft * kernel_fft, fshape, f_axes_seq)[fslice]
        # nix any roundoff error
        if set_tol_to_zero is not None:
            result[numpy.isclose(result, set_tol_to_zero)] = 0.0

        # if we're ignoring nodata, we need to make a convolution of the
        # nodata mask too
        if ignore_nodata:
            mask_fft = numpy.fft.rfftn(
                numpy.where(signal_nodata_mask, 0.0, 1.0), fshape, f_axes_seq)
            mask_result = numpy.fft.irfftn(
                mask_fft * kernel_fft, fshape, f_axes_seq)[fslice]

        left_index_result = 0
        right_index_result = result.shape[1]
        top_index_result = 0
        bottom_index_result = result.shape[0]

        # we might abut the edge of the raster, clip if so
        if left_index_raster < 0:
            left_index_result = -left_index_raster
            left_index_raster = 0
        if top_index_raster < 0:
            top_index_result = -top_index_raster
            top_index_raster = 0
        if right_index_raster > n_cols_signal:
            right_index_result -= right_index_raster - n_cols_signal
            right_index_raster = n_cols_signal
        if bottom_index_raster > n_rows_signal:
            bottom_index_result -= (
                bottom_index_raster - n_rows_signal)
            bottom_index_raster = n_rows_signal

        # Add result to current output to account for overlapping edges
        index_dict = {
            'xoff': left_index_raster,
            'yoff': top_index_raster,
            'win_xsize': right_index_raster-left_index_raster,
            'win_ysize': bottom_index_raster-top_index_raster
        }

        write_queue.put(
            (index_dict, result, mask_result,
             left_index_raster, right_index_raster,
             top_index_raster, bottom_index_raster,
             left_index_result, right_index_result,
             top_index_result, bottom_index_result))

    # Indicates worker has terminated
    write_queue.put(None)


def _assert_is_valid_pixel_size(target_pixel_size):
    """Return true if ``target_pixel_size`` is a valid 2 element sequence.

    Raises ValueError if not a two element list/tuple and/or the values in
        the sequence are not numerical.

    """
    def _is_number(x):
        """Return true if x is a number."""
        try:
            if isinstance(x, str):
                return False
            float(x)
            return True
        except (ValueError, TypeError):
            return False

    if not isinstance(target_pixel_size, (list, tuple)):
        raise ValueError(
            "target_pixel_size is not a tuple, its value was '%s'",
            repr(target_pixel_size))

    if (len(target_pixel_size) != 2 or
            not all([_is_number(x) for x in target_pixel_size])):
        raise ValueError(
            "Invalid value for `target_pixel_size`, expected two numerical "
            "elements, got: %s", repr(target_pixel_size))
    return True


@gdal_use_exceptions
def shapely_geometry_to_vector(
        shapely_geometry_list, target_vector_path, projection_wkt,
        vector_format, fields=None, attribute_list=None,
        ogr_geom_type=ogr.wkbPolygon):
    """Convert list of geometry to vector on disk.

    Args:
        shapely_geometry_list (list): a list of Shapely objects.
        target_vector_path (str): path to target vector.
        projection_wkt (str): WKT for target vector.
        vector_format (str): GDAL driver name for target vector.
        fields (dict): a python dictionary mapping string fieldname
            to OGR Fieldtypes, if None no fields are added
        attribute_list (list of dicts): a list of python dictionary mapping
            fieldname to field value for each geometry in
            `shapely_geometry_list`, if None, no attributes are created.
        ogr_geom_type (ogr geometry enumerated type): sets the target layer
            geometry type. Defaults to wkbPolygon.

    Return:
        None
    """
    if fields is None:
        fields = {}

    if attribute_list is None:
        attribute_list = [{} for _ in range(len(shapely_geometry_list))]

    num_geoms = len(shapely_geometry_list)
    num_attrs = len(attribute_list)
    if num_geoms != num_attrs:
        raise ValueError(
            f"Geometry count ({num_geoms}) and attribute count "
            f"({num_attrs}) do not match.")

    vector_driver = ogr.GetDriverByName(vector_format)
    target_vector = vector_driver.CreateDataSource(target_vector_path)
    layer_name = os.path.basename(os.path.splitext(target_vector_path)[0])
    projection = osr.SpatialReference()
    projection.ImportFromWkt(projection_wkt)
    target_layer = target_vector.CreateLayer(
        layer_name, srs=projection, geom_type=ogr_geom_type)

    for field_name, field_type in fields.items():
        target_layer.CreateField(ogr.FieldDefn(field_name, field_type))
    layer_defn = target_layer.GetLayerDefn()

    for field_index, (shapely_feature, feature_attributes) in enumerate(
            zip(shapely_geometry_list, attribute_list)):
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)

        if set(fields.keys()) != set(feature_attributes.keys()):
            raise ValueError(
                f"The fields and attributes for feature {field_index} "
                f"do not match.  Field definitions={fields}. "
                f"Feature attributes={feature_attributes}.")

        for field_name, field_value in feature_attributes.items():
            new_feature.SetField(field_name, field_value)
        target_layer.CreateFeature(new_feature)

    target_layer = None
    target_vector = None


@gdal_use_exceptions
def numpy_array_to_raster(
        base_array, target_nodata, pixel_size, origin, projection_wkt,
        target_path,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Create a single band raster of size ``base_array.shape``.

    The GDAL datatype of the target raster is determined by the numpy dtype of
    ``base_array``.

    Note:
        The ``origin`` and ``pixel_size`` parameters must both be defined
        properly as 2-tuples of floats, or else must both be set to ``None``.
        A ``ValueError`` will be raised otherwise.

    Args:
        base_array (numpy.array): a 2d numpy array.
        target_nodata (numeric): nodata value of target array, can be None.
        pixel_size (tuple): square dimensions (in ``(x, y)``) of pixel. Can be
            None to indicate no stated pixel size.
        origin (tuple/list): x/y coordinate of the raster origin. Can be None
            to indicate no stated origin.
        projection_wkt (str): target projection in wkt.  Can be None to
            indicate no projection/SRS.
        target_path (str): path to raster to create that will be of the
            same type of base_array with contents of base_array.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to
            geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None
    """
    driver_name, creation_options = raster_driver_creation_tuple
    raster_driver = gdal.GetDriverByName(driver_name)
    ny, nx = base_array.shape
    gdal_type, type_creation_options = _numpy_to_gdal_type(base_array.dtype)
    new_raster = raster_driver.Create(
        target_path, nx, ny, 1, gdal_type,
        options=list(creation_options) + type_creation_options)
    if projection_wkt is not None:
        new_raster.SetProjection(projection_wkt)
    if origin is not None and pixel_size is not None:
        new_raster.SetGeoTransform(
            [origin[0], pixel_size[0], 0, origin[1], 0, pixel_size[1]])
    elif origin is not None or pixel_size is not None:
        raise ValueError(
            "Origin and pixel size must both be defined or both be None")
    new_band = new_raster.GetRasterBand(1)
    if target_nodata is not None:
        if numpy.issubdtype(type(target_nodata), numpy.floating):
            target_nodata = float(target_nodata)
        elif numpy.issubdtype(type(target_nodata), numpy.integer):
            target_nodata = int(target_nodata)
        # Explicitly leaving off an else clause in case there's an edge case we
        # don't know about.  If so, we should wait for GDAL to raise an error.
        new_band.SetNoDataValue(target_nodata)
    new_band.WriteArray(base_array)
    new_band = None
    new_raster = None


@gdal_use_exceptions
def raster_to_numpy_array(raster_path, band_id=1):
    """Read the entire contents of the raster band to a numpy array.

    Args:
        raster_path (str): path to raster.
        band_id (int): band in the raster to read.

    Return:
        numpy array contents of `band_id` in raster.

    """
    raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(band_id)
    array = band.ReadAsArray()
    band = None
    raster = None
    return array


@gdal_use_exceptions
def stitch_rasters(
        base_raster_path_band_list,
        resample_method_list,
        target_stitch_raster_path_band,
        overlap_algorithm='etch',
        area_weight_m2_to_wgs84=False,
        osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):
    """Stitch the raster in the base list into the existing target.

    Args:
        base_raster_path_band_list (sequence): sequence of raster path/band
            tuples to stitch into target.
        resample_method_list (sequence): a sequence of resampling methods
            which one to one map each path in ``base_raster_path_band_list``
            during resizing.  Each element must be one of,
            'rms | mode | sum | q1 | near | q3 | average | cubicspline |
            bilinear | max | med | min | cubic | lanczos'
        target_stitch_raster_path_band (tuple): raster path/band tuple to an
            existing raster, values in ``base_raster_path_band_list`` will
            be stitched into this raster/band in the order they are in the
            list. The nodata value for the target band must be defined and
            will be written over with values from the base raster. Nodata
            values in the base rasters will not be written into the target.
            If the pixel size or projection are different between base and
            target the base is warped to the target's cell size and target
            with the interpolation method provided. If any part of the
            base raster lies outside of the target, that part of the base
            is ignored. A warning is logged if the entire base raster is
            outside of the target bounds.
        overlap_algorithm (str): this value indicates which algorithm to use
            when a raster is stitched on non-nodata values in the target
            stitch raster. It can be one of the following:

            - 'etch': write a value to the target raster only if the target
              raster pixel is nodata. If the target pixel is non-nodata
              ignore any additional values to write on that pixel.
            - 'replace': write a value to the target raster irrespective
              of the value of the target raster
            - 'add': add the value to be written to the target raster to
              any existing value that is there. If the existing value
              is nodata, treat it as 0.0.
        area_weight_m2_to_wgs84 (bool): If ``True`` the stitched raster will
            be converted to a per-area value before reprojection to wgs84,
            then multiplied by the m^2 area per pixel in the wgs84 coordinate
            space. This is useful when the quantity being stitched is a total
            quantity per pixel rather than a per unit area density. Note
            this assumes input rasters are in a projected space of meters,
            if they are not the stitched output will be nonsensical.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This
            parameter should not be changed unless you know what you are
            doing.

    Return:
        None.
    """
    valid_overlap_algorithms = ['etch', 'replace', 'add']
    if overlap_algorithm not in valid_overlap_algorithms:
        raise ValueError(
            f'overlap algorithm {overlap_algorithm} is not one of '
            f'{valid_overlap_algorithms}')

    if not _is_raster_path_band_formatted(target_stitch_raster_path_band):
        raise ValueError(
            f'Expected raster path/band tuple for '
            f'target_stitch_raster_path_band but got '
            f'"{target_stitch_raster_path_band}"')

    if len(base_raster_path_band_list) != len(resample_method_list):
        raise ValueError(
            f'Expected same number of elements in '
            f'`base_raster_path_band_list` as `resample_method_list` but '
            f'got {len(base_raster_path_band_list)} != '
            f'{len(resample_method_list)} respectively')

    gis_type = get_gis_type(target_stitch_raster_path_band[0])
    from pygeoprocessing import RASTER_TYPE
    if gis_type != RASTER_TYPE:
        raise ValueError(
            f'Target stitch raster is not a raster. '
            f'Location: "{target_stitch_raster_path_band[0]}" '
            f'GIS type: {gis_type}')
    target_raster_info = get_raster_info(target_stitch_raster_path_band[0])
    if target_stitch_raster_path_band[1] > len(target_raster_info['nodata']):
        raise ValueError(
            f'target_stitch_raster_path_band refers to a band that exceeds '
            f'the number of bands in the raster:\n'
            f'target_stitch_raster_path_band[1]: '
            f'{target_stitch_raster_path_band[1]} '
            f'n bands: {len(target_raster_info["nodata"])}')
    target_nodata = target_raster_info['nodata'][
        target_stitch_raster_path_band[1]-1]
    if target_nodata is None:
        raise ValueError(
            f'target stitch raster at "{target_stitch_raster_path_band[0]} "'
            f'nodata value is `None`, expected non-`None` value')

    target_raster = gdal.OpenEx(
        target_stitch_raster_path_band[0], gdal.OF_RASTER | gdal.GA_Update)
    target_band = target_raster.GetRasterBand(
        target_stitch_raster_path_band[1])
    target_inv_gt = gdal.InvGeoTransform(target_raster_info['geotransform'])
    target_raster_x_size, target_raster_y_size = target_raster_info[
        'raster_size']
    for (raster_path, raster_band_id), resample_method in zip(
            base_raster_path_band_list, resample_method_list):
        LOGGER.info(
            f'stitching {(raster_path, raster_band_id)} into '
            f'{target_stitch_raster_path_band}')
        raster_info = get_raster_info(raster_path)

        projected_raster_bounding_box = transform_bounding_box(
            raster_info['bounding_box'],
            raster_info['projection_wkt'],
            target_raster_info['projection_wkt'])

        try:
            # merge the bounding boxes only to see if they don't intersect
            _ = merge_bounding_box_list(
                [projected_raster_bounding_box,
                 target_raster_info['bounding_box']], 'intersection')
        except ValueError:
            LOGGER.warning(
                f'the raster at "{raster_path}"" does not intersect the '
                f'stitch raster at "{target_stitch_raster_path_band[0]}", '
                f'skipping...')
            continue

        # use this to determine if we need to warp and delete if we did at
        # the end
        if (raster_info['projection_wkt'] ==
            target_raster_info['projection_wkt'] and
            raster_info['pixel_size'] ==
                target_raster_info['pixel_size']):
            warped_raster = False
            base_stitch_raster_path = raster_path
        else:
            workspace_dir = tempfile.mkdtemp(
                dir=os.path.dirname(target_stitch_raster_path_band[0]),
                prefix='stitch_rasters_workspace')
            base_stitch_raster_path = os.path.join(
                workspace_dir, os.path.basename(raster_path))
            warp_raster(
                raster_path, target_raster_info['pixel_size'],
                base_stitch_raster_path, resample_method,
                target_projection_wkt=target_raster_info['projection_wkt'],
                working_dir=workspace_dir,
                osr_axis_mapping_strategy=osr_axis_mapping_strategy)
            warped_raster = True

        if warped_raster and area_weight_m2_to_wgs84:
            # determine base area per pixel currently and area per pixel
            # once it is projected to wgs84 pixel sizes
            base_pixel_area_m2 = abs(numpy.prod(raster_info['pixel_size']))
            base_stitch_raster_info = get_raster_info(
                base_stitch_raster_path)
            _, lat_min, _, lat_max = base_stitch_raster_info['bounding_box']
            n_rows = base_stitch_raster_info['raster_size'][1]
            # this column is a longitude invariant latitude variant pixel
            # area for scaling area dependent values
            m2_area_per_lat = _create_latitude_m2_area_column(
                lat_min, lat_max, n_rows)

            def _mult_op(base_array, base_nodata, scale, datatype):
                """Scale non-nodata by scale."""
                result = base_array.astype(datatype)
                if base_nodata is not None:
                    valid_mask = ~numpy.isclose(base_array, base_nodata)
                else:
                    valid_mask = numpy.ones(
                        base_array.shape, dtype=bool)
                result[valid_mask] = result[valid_mask] * scale[valid_mask]
                return result

            base_stitch_nodata = base_stitch_raster_info['nodata'][0]
            scaled_raster_path = os.path.join(
                workspace_dir,
                f'scaled_{os.path.basename(base_stitch_raster_path)}')
            gdal_type = _gdal_to_numpy_type(
                target_band.DataType,
                target_band.GetMetadata('IMAGE_STRUCTURE'))
            # multiply the pixels in the resampled raster by the ratio of
            # the pixel area in the wgs84 units divided by the area of the
            # original pixel
            raster_calculator(
                [(base_stitch_raster_path, 1), (base_stitch_nodata, 'raw'),
                 m2_area_per_lat/base_pixel_area_m2,
                 (gdal_type, 'raw')], _mult_op,
                scaled_raster_path,
                target_raster_info['datatype'], base_stitch_nodata)

            # swap the result to base stitch so the rest of the function
            # operates on the area scaled raster
            os.remove(base_stitch_raster_path)
            base_stitch_raster_path = scaled_raster_path

        base_raster = gdal.OpenEx(base_stitch_raster_path, gdal.OF_RASTER)
        base_gt = base_raster.GetGeoTransform()
        base_band = base_raster.GetRasterBand(raster_band_id)
        base_nodata = base_band.GetNoDataValue()
        # Get the target upper left xoff/yoff w/r/t the stitch raster 0,0
        # coordinates
        target_to_base_xoff, target_to_base_yoff = [
            int(_) for _ in gdal.ApplyGeoTransform(
                target_inv_gt, *gdal.ApplyGeoTransform(base_gt, 0, 0))]
        for offset_dict in iterblocks(
                (base_stitch_raster_path, raster_band_id), offset_only=True):
            _offset_vars = {}
            overlap = True
            for (target_to_base_off, off_val,
                 target_off_id, off_clip_id, win_size_id, raster_size) in [
                (target_to_base_xoff, offset_dict['xoff'],
                 'target_xoff', 'xoff_clip', 'win_xsize',
                 target_raster_x_size),
                (target_to_base_yoff, offset_dict['yoff'],
                 'target_yoff', 'yoff_clip', 'win_ysize',
                 target_raster_y_size)]:
                _offset_vars[target_off_id] = (target_to_base_off+off_val)
                if _offset_vars[target_off_id] >= raster_size:
                    overlap = False
                    break
                # how far to move right to get in the target raster
                _offset_vars[off_clip_id] = 0
                if _offset_vars[target_off_id] < 0:
                    _offset_vars[off_clip_id] = -_offset_vars[target_off_id]
                _offset_vars[win_size_id] = offset_dict[win_size_id]
                if _offset_vars[off_clip_id] >= _offset_vars[win_size_id]:
                    # its too far left for the whole window
                    overlap = False
                    break
                # make the _offset_vars[win_size_id] smaller if it shifts
                # off the target window
                if (_offset_vars[off_clip_id] + _offset_vars[target_off_id] +
                        _offset_vars[win_size_id] >= raster_size):
                    _offset_vars[win_size_id] -= (
                        _offset_vars[off_clip_id] +
                        _offset_vars[target_off_id] +
                        _offset_vars[win_size_id] - raster_size)

            if not overlap:
                continue

            target_array = target_band.ReadAsArray(
                xoff=_offset_vars['target_xoff']+_offset_vars['xoff_clip'],
                yoff=_offset_vars['target_yoff']+_offset_vars['yoff_clip'],
                win_xsize=_offset_vars['win_xsize'],
                win_ysize=_offset_vars['win_ysize'])
            target_nodata_mask = numpy.isclose(target_array, target_nodata)
            base_array = base_band.ReadAsArray(
                xoff=offset_dict['xoff']+_offset_vars['xoff_clip'],
                yoff=offset_dict['yoff']+_offset_vars['yoff_clip'],
                win_xsize=_offset_vars['win_xsize'],
                win_ysize=_offset_vars['win_ysize'])

            if base_nodata is not None:
                base_nodata_mask = numpy.isclose(base_array, base_nodata)
            else:
                base_nodata_mask = numpy.zeros(
                    base_array.shape, dtype=bool)

            if overlap_algorithm == 'etch':
                # place values only where target is nodata
                valid_mask = ~base_nodata_mask & target_nodata_mask
                target_array[valid_mask] = base_array[valid_mask]
            elif overlap_algorithm == 'replace':
                # write valid values into the target -- disregard any
                # existing values in the target
                valid_mask = ~base_nodata_mask
                target_array[valid_mask] = base_array[valid_mask]
            elif overlap_algorithm == 'add':
                # add values to the target and treat target nodata as 0.
                valid_mask = ~base_nodata_mask
                masked_target_array = target_array[valid_mask]
                target_array_nodata_mask = numpy.isclose(
                    masked_target_array, target_nodata)
                target_array[valid_mask] = (
                    base_array[valid_mask] +
                    numpy.where(
                        target_array_nodata_mask, 0, masked_target_array))
            else:
                raise RuntimeError(
                    f'overlap_algorithm {overlap_algorithm} was not defined '
                    f'but also not detected earlier -- this should never '
                    f'happen')

            target_band.WriteArray(
                target_array,
                xoff=_offset_vars['target_xoff']+_offset_vars['xoff_clip'],
                yoff=_offset_vars['target_yoff']+_offset_vars['yoff_clip'])

        base_raster = None
        base_band = None
        if warped_raster:
            shutil.rmtree(workspace_dir)

    target_raster = None
    target_band = None


@gdal_use_exceptions
def build_overviews(
        raster_path, internal=False, resample_method='near',
        overwrite=False, levels='auto'):
    """Build overviews for a raster dataset.

    Args:
        raster_path (str): A path to a raster on disk for which overviews
            should be built.
        internal=False (bool): Whether to modify the raster when building
            overviews. In GeoTiffs, this builds internal overviews when
            ``internal=True``, and external overviews when ``internal=False``.
        resample_method='near' (str): The resample method to use when
            building overviews.  Must be a valid resampling method for
            ``gdal.GDALDataset.BuildOverviews``, one of
            'rms | mode | sum | q1 | near | q3 | average | cubicspline |
            bilinear | max | med | min | cubic | lanczos'.
        overwrite=False (bool): Whether to overwrite existing overviews, if
            any exist.
        levels='auto' (sequence): A sequence of integer overview levels. If
            ``'auto'``, overview levels will be determined by using factors of
            2 until the overview's x and y dimensions are both less than 256.

    Example:
        Generate overviews, regardless of whether overviews already exist
        for the raster, letting the function determine the levels of overviews
        to generate::

            build_overviews(raster_path)

        Generate overviews for 4 levels, at 1/2, 1/4, 1/8 and 1/16 the
        resolution::

            build_overviews(raster_path, levels=[2, 4, 8, 16])

    Returns:
        ``None``
    """
    def overviews_progress(*args, **kwargs):
        pct_complete, name, other = args
        percent = round(pct_complete * 100, 2)
        if time.time() - overviews_progress.last_progress_report > 5.0:
            LOGGER.info(f"Overviews progress: {percent}%")
            overviews_progress.last_progress_report = time.time()
    overviews_progress.last_progress_report = time.time()

    open_flags = gdal.OF_RASTER
    if internal:
        open_flags |= gdal.GA_Update
        LOGGER.info(f"Building internal overviews on {raster_path}")
    else:
        LOGGER.info("Building external overviews.")
    raster = gdal.OpenEx(raster_path, open_flags)
    overview_count = 0
    for band_index in range(1, raster.RasterCount + 1):
        band = raster.GetRasterBand(band_index)
        overview_count += band.GetOverviewCount()

    if overview_count > 0:
        if overwrite:
            LOGGER.info(f"Clearing existing overviews from {raster_path}")
            result = raster.BuildOverviews(
                resampling=resample_method,
                overviewlist=[],
                callback=overviews_progress
            )
            LOGGER.info(f"Overviews cleared from {raster_path}")
        else:
            raise ValueError(
                f"Raster already has overviews.  Use "
                "overwrite=True to override this and regenerate overviews on "
                f"{raster_path}")

    # This loop and limiting factor borrowed from gdaladdo.cpp.
    # Create overviews so long as the overviews are at least 256 pixels in
    # either x or y dimensions.
    if levels == 'auto':
        overview_scales = []
        factor = 2
        limiting_factor = 256
        while (math.ceil(raster.RasterXSize / factor) > limiting_factor or
               math.ceil(raster.RasterYSize / factor) > limiting_factor):
            overview_scales.append(factor)
            factor *= 2
    else:
        overview_scales = [int(level) for level in levels]

    LOGGER.debug(f"Using overviews {overview_scales}")
    result = raster.BuildOverviews(
        resampling=resample_method,
        overviewlist=overview_scales,
        callback=overviews_progress
    )
    LOGGER.info(f"Overviews completed for {raster_path}")
    if result:  # Result will be nonzero on error.
        raise RuntimeError(
            f"Building overviews failed or was interrupted for {raster_path}")


def _m2_area_of_wg84_pixel(pixel_size, center_lat):
    """Calculate m^2 area of a square wgs84 pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Args:
        pixel_size (float): length of side of a square pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in m^2.

    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = math.sqrt(1 - (b/a)**2)
    area_list = []
    for f in [center_lat+pixel_size/2, center_lat-pixel_size/2]:
        zm = 1 - e*math.sin(math.radians(f))
        zp = 1 + e*math.sin(math.radians(f))
        area_list.append(
            math.pi * b**2 * (
                math.log(zp/zm) / (2*e) +
                math.sin(math.radians(f)) / (zp*zm)))
    return abs(pixel_size / 360. * (area_list[0] - area_list[1]))


def _create_latitude_m2_area_column(lat_min, lat_max, n_pixels):
    """Create a (n, 1) sized numpy array with m^2 areas in each element.

    Creates a per pixel m^2 area array that varies with changes in latitude.
    This array can be used to scale values by area when converting to or
    from a WGS84 projection to a projected one.

    Args:
        lat_max (float): maximum latitude in the bound
        lat_min (float): minimum latitude in the bound
        n_pixels (int): number of pixels to create for the column. The
            size of the target square pixels are (lat_max-lat_min)/n_pixels
            degrees per side.

    Return:
        A (n, 1) sized numpy array whose elements are the m^2 areas in each
        element estimated by the latitude value at the center of each pixel.
    """
    pixel_size = (lat_max - lat_min) / n_pixels
    center_lat_array = numpy.linspace(
        lat_min+pixel_size/2, lat_max-pixel_size/2, n_pixels)
    area_array = numpy.array([
        _m2_area_of_wg84_pixel(pixel_size, lat)
        for lat in reversed(center_lat_array)]).reshape((n_pixels, 1))
    return area_array
