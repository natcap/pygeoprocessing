# coding=UTF-8
"""Multiprocessing implementation of raster_calculator."""
import collections
import errno
import multiprocessing
import os
import pprint
import signal
import sys
import time

import numpy
from osgeo import gdal

from ..geoprocessing import _is_raster_path_band_formatted
from ..geoprocessing import _LARGEST_ITERBLOCK
from ..geoprocessing import _LOGGING_PERIOD
from ..geoprocessing import _MAX_TIMEOUT
from ..geoprocessing import _VALID_GDAL_TYPES
from ..geoprocessing import geoprocessing_core
from ..geoprocessing import get_raster_info
from ..geoprocessing import iterblocks
from ..geoprocessing import LOGGER
from ..geoprocessing_core import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
from ..geoprocessing_core import GDALUseExceptions

if sys.version_info >= (3, 8):
    import multiprocessing.shared_memory


def _block_success_handler(callback_state):
    """Used to update callback state after a successful block is complete.

    Updates the blocks complete and last time, if last_time has been >
    _LOGGING_PERIOD then dump a log.

    Args:
        callback_state (dict): contains the following keys
            'blocks_complete' -- number of raster calculator blocks processed
            'total_blocks' -- total number to process
            'last_time' -- last time.time() when a log was printed

    Returns:
        None
    """
    callback_state['blocks_complete'] += 1
    if time.time() - callback_state['last_time'] > _LOGGING_PERIOD:
        LOGGER.info(
            f"""raster_calculator {
                callback_state['blocks_complete']/
                callback_state['total_blocks']*100:.2f}% complete""")
        callback_state['last_time'] = time.time()


RasterPathBand = collections.namedtuple(
    'RasterPathBand', ['path', 'band_id'])


def _build_raster_calc_error_handler(pool):
    def _raster_calc_error_handler(exception):
        """Error handler for raster_calculator."""
        pool.terminate()
        raise Exception(
            f"error on raster calculator worker '{exception}'").with_traceback(
                exception.__traceback__)
    return _raster_calc_error_handler


def _raster_calculator_worker(
        block_offset_queue, base_canonical_arg_list, local_op,
        stats_worker_queue, nodata_target, target_raster_path, write_lock,
        processing_state, result_array_shared_memory):
    """Process a single block of an array for raster_calculation.

    Args:
        block_offset (dict): contains 'xoff', 'yoff', 'xwin_size', 'ywin_size'
            and can be used to pass directly to Band.ReadAsArray.
        base_canonical_arg_list (list): list of RasterPathBand, numpy arrays,
            or 'raw' objects to pass to the ``local_op``.
        local_op (function): callable that a function that must take in as
            many parameters as there are elements in
            ``base_canonical_arg_list``. Full description can be found in the
            public facing ``raster_calculator`` operation.
        stats_worker_queue (queue): pass a shared memory object ``local_op``
            result to queue if stats are being calculated. None otherwise.
        nodata_target (numeric or None): desired target raster nodata
        target_raster_path (str): path to target raster.
        write_lock (multiprocessing.Lock): Lock object used to coordinate
            writes to raster_path.
        processing_state (multiprocessing.Manager.dict): a global object to
            pass to ``__block_success_handler`` for this execution context.
        result_array_shared_memory (multiprocessing.shared_memory): If
            Python version >= 3.8, this is a shared
            memory object used to pass data to the stats worker process if
            required. Should be pre-allocated with enough data to hold the
            largest result from ``local_op`` given any ``block_offset`` from
            ``block_offset_queue``. None otherwise.

    Returns:
        None.

    """
    # read input blocks
    while True:
        block_offset = block_offset_queue.get()
        if block_offset is None:
            # indicates this worker should terminate
            return

        offset_list = (block_offset['yoff'], block_offset['xoff'])
        blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])
        data_blocks = []
        for value in base_canonical_arg_list:
            if isinstance(value, RasterPathBand):
                with GDALUseExceptions():
                    raster = gdal.OpenEx(value.path, gdal.OF_RASTER)
                    band = raster.GetRasterBand(value.band_id)
                    data_blocks.append(band.ReadAsArray(**block_offset))
                    # I've encountered the following error when a gdal raster
                    # is corrupt, often from multiple threads writing to the
                    # same file. This helps to catch the error early rather
                    # than lead to confusing values of ``data_blocks`` later.
                    if not isinstance(data_blocks[-1], numpy.ndarray):
                        raise ValueError(
                            f"got a {data_blocks[-1]} when trying to read "
                            f"{band.GetDataset().GetFileList()} at "
                            f"{block_offset}, expected numpy.ndarray.")
                    raster = None
                    band = None
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

        with write_lock:
            with GDALUseExceptions():
                target_raster = gdal.OpenEx(
                    target_raster_path, gdal.OF_RASTER | gdal.GA_Update)
                target_band = target_raster.GetRasterBand(1)
                target_band.WriteArray(
                    target_block, yoff=block_offset['yoff'],
                    xoff=block_offset['xoff'])
                _block_success_handler(processing_state)
                target_band = None
                target_raster = None

        # send result to stats calculator
        if not stats_worker_queue:
            continue

        # Construct shared memory object to pass to stats worker
        if nodata_target is not None:
            target_block = target_block[target_block != nodata_target]
        target_block = target_block.astype(numpy.float64).flatten()

        if result_array_shared_memory:
            shared_memory_array = numpy.ndarray(
                target_block.shape, dtype=target_block.dtype,
                buffer=result_array_shared_memory.buf)
            shared_memory_array[:] = target_block[:]

            stats_worker_queue.put((
                shared_memory_array.shape, shared_memory_array.dtype,
                result_array_shared_memory))
        else:
            stats_worker_queue.put(target_block)


def _calculate_target_raster_size(
        raster_info_list, base_raster_path_band_const_list):
    """Determine the target raster size.

    Args:
        raster_info_list (list): list of raster info from
            ``base_raster_path_band_const_list``.
        base_raster_path_band_const_list (list/tuple): argument from
            raster_calculator.

    Returns:
        count of number of valid numpy array elements in
            ``base_raster_path_band_const_list``.

    Raises:
        ``ValueError`` if numpy array types in
            ``base_raster_path_band_const_list`` do not have sizes which can
            be broadcast against each other.
        ``ValueError`` if calculated broadcast size is incompatable with the
            raster sizes in ``base_raster_path_band_const_list``.
        ``ValueError`` if only ``'raw'`` objects have been passed as arguments.

    """
    numpy_broadcast_list = [
        x for x in base_raster_path_band_const_list
        if isinstance(x, numpy.ndarray)]
    numpy_broadcast_size = None

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
    if raster_info_list and numpy_broadcast_size:
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

    # create target raster
    if raster_info_list:
        # if rasters are passed, the target is the same size as the raster
        n_cols, n_rows = raster_info_list[0]['raster_size']
    elif numpy_broadcast_size:
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

    return n_cols, n_rows


def _validate_raster_input(
        base_raster_path_band_const_list, raster_info_list,
        target_raster_path):
    """Check for valid raster/arg inputs and output.

    Args:
        base_raster_path_band_const_list (list/tuple): the same object passed
            to .raster_calculator indicating the datastack to process.
        target_raster_path (str): desired target raster path from
            raster_calculator, used to ensure it is not also an input parameter

    Returns:
        None

    Raises:
        ValueError if any input parameter would cause an error when passing to
            .raster_calculator
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

    with GDALUseExceptions():
        # check that any rasters exist on disk and have enough bands
        not_found_paths = []
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        base_raster_path_band_list = [
            path_band for path_band in base_raster_path_band_const_list
            if _is_raster_path_band_formatted(path_band)]
        for value in base_raster_path_band_list:
            if gdal.OpenEx(value[0], gdal.OF_RASTER) is None:
                not_found_paths.append(value[0])
        gdal.PopErrorHandler()
        if not_found_paths:
            raise ValueError(
                "The following files were expected but do not exist on the "
                "filesystem: " + str(not_found_paths))

        # check that band index exists in raster
        invalid_band_index_list = []
        for value in base_raster_path_band_list:
            raster = gdal.OpenEx(value[0], gdal.OF_RASTER)
            if not (1 <= value[1] <= raster.RasterCount):
                invalid_band_index_list.append(value)
            raster = None
        if invalid_band_index_list:
            raise ValueError(
                "The following rasters do not contain requested band "
                "indexes: %s" % invalid_band_index_list)

    # check that the target raster is not also an input raster
    if target_raster_path in [x[0] for x in base_raster_path_band_list]:
        raise ValueError(
            "%s is used as a target path, but it is also in the base input "
            "path list %s" % (
                target_raster_path, str(base_raster_path_band_const_list)))

    # check that raster inputs are all the same dimensions
    geospatial_info_set = set()
    for raster_info in raster_info_list:
        geospatial_info_set.add(raster_info['raster_size'])
    if len(geospatial_info_set) > 1:
        raise ValueError(
            "Input Rasters are not the same dimensions. The "
            "following raster are not identical %s" % str(
                geospatial_info_set))


def raster_calculator(
        base_raster_path_band_const_list, local_op, target_raster_path,
        datatype_target, nodata_target,
        n_workers=max(1, multiprocessing.cpu_count()),
        calc_raster_stats=True, use_shared_memory=False,
        largest_block=_LARGEST_ITERBLOCK,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Apply local a raster operation on a stack of rasters.

    This function applies a user defined function across a stack of
    rasters' pixel stack. The rasters in ``base_raster_path_band_list`` must
    be spatially aligned and have the same cell sizes.

    Args:
        base_raster_path_band_const_list (sequence): a sequence containing
            either (str, int) tuples, ``numpy.ndarray`` s of up to two
            dimensions, or an (object, 'raw') tuple.  A ``(str, int)``
            tuple refers to a raster path band index pair to use as an input.
            The ``numpy.ndarray`` s must be broadcastable to each other AND the
            size of the raster inputs. Values passed by  ``(object, 'raw')``
            tuples pass ``object`` directly into the ``local_op``. All rasters
            must have the same raster size. If only arrays are input, numpy
            arrays must be broadcastable to each other and the final raster
            size will be the final broadcast array shape. A value error is
            raised if only "raw" inputs are passed.
        local_op (function) a function that must take in as many parameters as
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
        n_workers (int): number of Processes to launch for parallel processing,
            defaults to ``multiprocessing.cpu_count()``.
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
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to
            geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None

    Raises:
        ValueError: invalid input provided

    """
    raster_info_list = [
        get_raster_info(path_band[0])
        for path_band in base_raster_path_band_const_list
        if _is_raster_path_band_formatted(path_band)]

    target_raster_path = os.path.abspath(target_raster_path)
    _validate_raster_input(
        base_raster_path_band_const_list, raster_info_list, target_raster_path)

    n_cols, n_rows = _calculate_target_raster_size(
        raster_info_list, base_raster_path_band_const_list)

    # create a "canonical" argument list that contains only
    # (file paths, band id) tuples, 2d numpy arrays, or raw values
    base_canonical_arg_list = []
    for value in base_raster_path_band_const_list:
        # the input has been tested and value is either a raster/path band
        # tuple, 1d ndarray, 2d ndarray, or (value, 'raw') tuple.
        if _is_raster_path_band_formatted(value):
            # it's a raster/path band, keep track of open raster and band
            # for later so we can `None` them.
            base_canonical_arg_list.append(
                RasterPathBand(value[0], value[1]))
        elif isinstance(value, numpy.ndarray):
            if value.ndim == 1:
                # easier to process as a 2d array for writing to band
                base_canonical_arg_list.append(
                    value.reshape((1, value.shape[0])))
            else:  # dimensions are two because we checked earlier.
                base_canonical_arg_list.append(value)
        else:
            # it's a regular tuple
            base_canonical_arg_list.append(value)

    if datatype_target not in _VALID_GDAL_TYPES:
        raise ValueError(
            'Invalid target type, should be a gdal.GDT_* type, received '
            '"%s"' % datatype_target)

    with GDALUseExceptions():
        # create target raster
        raster_driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
        try:
            os.makedirs(os.path.dirname(target_raster_path))
        except OSError as exception:
            # it's fine if the directory already exists, otherwise there's a big
            # error!
            if exception.errno != errno.EEXIST:
                raise

        target_raster = raster_driver.Create(
            target_raster_path, n_cols, n_rows, 1, datatype_target,
            options=raster_driver_creation_tuple[1])

        target_band = target_raster.GetRasterBand(1)
        if nodata_target is not None:
            target_band.SetNoDataValue(nodata_target)
        if raster_info_list:
            # use the first raster in the list for the projection and geotransform
            target_raster.SetProjection(raster_info_list[0]['projection_wkt'])
            target_raster.SetGeoTransform(raster_info_list[0]['geotransform'])
        target_band = None
        target_raster = None

    manager = multiprocessing.Manager()
    stats_worker_queue = None
    if calc_raster_stats:
        # if this queue is used to send computed valid blocks of
        # the raster to an incremental statistics calculator worker
        stats_worker_queue = manager.Queue()

    # iterate over each block and calculate local_op
    block_offset_list = list(iterblocks(
        (target_raster_path, 1), offset_only=True,
        largest_block=largest_block))

    if calc_raster_stats:
        LOGGER.debug('start stats worker')
        stats_worker = multiprocessing.Process(
            target=geoprocessing_core.stats_worker,
            args=(stats_worker_queue, len(block_offset_list)))
        stats_worker.start()

    LOGGER.debug('start workers')
    processing_state = manager.dict()
    processing_state['blocks_complete'] = 0
    processing_state['total_blocks'] = len(block_offset_list)
    processing_state['last_time'] = time.time()
    block_size_bytes = (
        numpy.dtype(numpy.float64).itemsize *
        block_offset_list[0]['win_xsize'] * block_offset_list[0]['win_ysize'])
    target_write_lock = manager.Lock()
    block_offset_queue = multiprocessing.Queue(n_workers)
    process_list = []
    for _ in range(n_workers):
        shared_memory = None
        if calc_raster_stats:
            if sys.version_info >= (3, 8) and use_shared_memory:
                shared_memory = multiprocessing.shared_memory.SharedMemory(
                    create=True, size=block_size_bytes)
        worker = multiprocessing.Process(
            target=_raster_calculator_worker,
            args=(
                block_offset_queue, base_canonical_arg_list, local_op,
                stats_worker_queue, nodata_target, target_raster_path,
                target_write_lock, processing_state, shared_memory))
        worker.start()
        process_list.append((worker, shared_memory))

    # Fill the work queue
    for block_offset in block_offset_list:
        block_offset_queue.put(block_offset)

    for _ in range(n_workers):
        block_offset_queue.put(None)

    LOGGER.info('wait for stats worker to complete')
    stats_worker.join(_MAX_TIMEOUT)
    if stats_worker.is_alive():
        LOGGER.error(
            f'stats worker {stats_worker.pid} '
            'didn\'t terminate, sending kill signal.')
        try:
            os.kill(stats_worker.pid, signal.SIGTERM)
        except Exception:
            LOGGER.exception(f'unable to kill {stats_worker.pid}')

    # wait for the workers to join
    LOGGER.info('all work sent, waiting for workers to finish')
    for worker, shared_memory in process_list:
        worker.join(_MAX_TIMEOUT)
        if worker.is_alive():
            LOGGER.error(
                f'worker {worker.pid} didn\'t terminate, sending kill signal.')
            try:
                os.kill(stats_worker.pid, signal.SIGTERM)
            except Exception:
                LOGGER.exception(f'unable to kill {worker.pid}')
        if shared_memory is not None:
            LOGGER.debug(f'unlink {shared_memory.name}')
            shared_memory.unlink()

    if calc_raster_stats:
        payload = stats_worker_queue.get(True, _MAX_TIMEOUT)
        if payload is not None:
            target_min, target_max, target_mean, target_stddev = payload
            with GDALUseExceptions():
                target_raster = gdal.OpenEx(
                    target_raster_path, gdal.OF_RASTER | gdal.GA_Update)
                target_band = target_raster.GetRasterBand(1)
                target_band.SetStatistics(
                    float(target_min), float(target_max), float(target_mean),
                    float(target_stddev))
                target_band = None
                target_raster = None
    LOGGER.info('raster_calculator 100.0%% complete')
