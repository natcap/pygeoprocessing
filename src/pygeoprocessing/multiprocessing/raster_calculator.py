"""Multiprocessing implementation of raster_calculator."""
import os
import pprint
import multiprocessing
import time
import threading
import queue

from ..geoprocessing import _invoke_timed_callback
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
from osgeo import gdal
import numpy


def count_complete_handler(total_steps, start_time):
    def success_handler(val):
        success_handler.steps_so_far += 1
        if time.time() - success_handler.last_time > 5.0:
            LOGGER.info(
                'raster_calculator '
                f'{success_handler.steps_so_far/total_steps*100:.2f}% '
                'complete')
            success_handler.last_time = time.time()

    success_handler.steps_so_far = 0
    success_handler.last_time = start_time

    return success_handler


class RasterPathBand():
    def __init__(self, path, band_id):
        self.path = path
        self.band_id = band_id


def error_handler(exception):
    raise Exception("error on raster calculator worker").with_traceback(exception.__traceback__ )


def raster_calculator_worker(
        block_offset, base_canonical_arg_list, local_op, stats_worker_queue,
        nodata_target, target_raster_path, write_lock):
    # read input blocks
    offset_list = (block_offset['yoff'], block_offset['xoff'])
    blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])
    data_blocks = []
    for value in base_canonical_arg_list:
        if isinstance(value, RasterPathBand):
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

    # send result to stats calculator
    if stats_worker_queue:
        # guard against an undefined nodata target
        if nodata_target is not None:
            valid_block = target_block[target_block != nodata_target]
            if valid_block.size > 0:
                stats_worker_queue.put(valid_block)
        else:
            stats_worker_queue.put(target_block.flatten())

    with write_lock:
        target_raster = gdal.OpenEx(
            target_raster_path, gdal.OF_RASTER | gdal.GA_Update)
        target_band = target_raster.GetRasterBand(1)
        target_band.WriteArray(
            target_block, yoff=block_offset['yoff'],
            xoff=block_offset['xoff'])


def raster_calculator(
        base_raster_path_band_const_list, local_op, target_raster_path,
        datatype_target, nodata_target, n_workers,
        calc_raster_stats=True, largest_block=_LARGEST_ITERBLOCK,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Apply local a raster operation on a stack of rasters.

    This function applies a user defined function across a stack of
    rasters' pixel stack. The rasters in ``base_raster_path_band_list`` must
    be spatially aligned and have the same cell sizes.

    Args:
        base_raster_path_band_const_list (sequence): a sequence containing
            either (str, int) tuples, ``numpy.ndarray``s of up to two
            dimensions, or an (object, 'raw') tuple.  A ``(str, int)``
            tuple refers to a raster path band index pair to use as an input.
            The ``numpy.ndarray``s must be broadcastable to each other AND the
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
        n_workers (int): number of Processes to launch for parallel processing.
        calc_raster_stats (boolean): If True, calculates and sets raster
            statistics (min, max, mean, and stdev) for target raster.
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
    raster_info_list = [
        get_raster_info(path_band[0])
        for path_band in base_raster_path_band_const_list
        if _is_raster_path_band_formatted(path_band)]
    geospatial_info_set = set()
    for raster_info in raster_info_list:
        geospatial_info_set.add(raster_info['raster_size'])
    if len(geospatial_info_set) > 1:
        raise ValueError(
            "Input Rasters are not the same dimensions. The "
            "following raster are not identical %s" % str(
                geospatial_info_set))

    numpy_broadcast_list = [
        x for x in base_raster_path_band_const_list
        if isinstance(x, numpy.ndarray)]
    stats_worker = None
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
            # for later so we can __swig_destroy__ them.
            base_canonical_arg_list.append(
                RasterPathBand(value[0], value[1]))
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
        manager = multiprocessing.Manager()
        last_time = time.time()

        if calc_raster_stats:
            # if this queue is used to send computed valid blocks of
            # the raster to an incremental statistics calculator worker
            stats_worker_queue = manager.Queue()
            exception_queue = manager.Queue()
        else:
            stats_worker_queue = None
            exception_queue = None

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
            stats_worker = multiprocessing.Process(
                target=geoprocessing_core.stats_worker,
                args=(stats_worker_queue, exception_queue))
            stats_worker.daemon = True
            stats_worker.start()
            LOGGER.info('started stats_worker %s', stats_worker)

        write_lock = manager.Lock()

        # iterate over each block and calculate local_op
        block_offset_list = list(iterblocks(
            (target_raster_path, 1), offset_only=True,
            largest_block=largest_block))

        callback_handler = count_complete_handler(
            len(block_offset_list), time.time())

        with multiprocessing.Pool(processes=n_workers) as pool:
            for block_offset in block_offset_list:
                pool.apply_async(
                    raster_calculator_worker,
                    (block_offset, base_canonical_arg_list, local_op,
                     stats_worker_queue, nodata_target, target_raster_path,
                     write_lock),
                    callback=callback_handler,
                    error_callback=error_handler)
            pool.close()
            pool.join()

        LOGGER.info('100.0%% complete')

        if calc_raster_stats:
            LOGGER.info("signaling stats worker to terminate")
            stats_worker_queue.put(None)
            LOGGER.info("Waiting for raster stats worker result.")
            stats_worker.join(_MAX_TIMEOUT)
            if stats_worker.is_alive():
                raise RuntimeError("stats_worker.join() timed out")
            payload = stats_worker_queue.get(True, _MAX_TIMEOUT)
            if payload is not None:
                target_min, target_max, target_mean, target_stddev = payload
                target_band.SetStatistics(
                    float(target_min), float(target_max), float(target_mean),
                    float(target_stddev))
                target_band.FlushCache()
    finally:
        # This block ensures that rasters are destroyed even if there's an
        # exception raised.
        base_band_list[:] = []
        for raster in base_raster_list:
            gdal.Dataset.__swig_destroy__(raster)
        base_raster_list[:] = []
        target_band.FlushCache()
        target_band = None
        target_raster.FlushCache()
        gdal.Dataset.__swig_destroy__(target_raster)
        target_raster = None

        if calc_raster_stats and stats_worker:
            if stats_worker.is_alive():
                stats_worker_queue.put(None, True, _MAX_TIMEOUT)
                LOGGER.info("Waiting for raster stats worker result.")
                stats_worker.join(_MAX_TIMEOUT)
                if stats_worker.is_alive():
                    raise RuntimeError("stats_worker.join() timed out")

            # check for an exception in the workers, otherwise get result
            # and pass to writer
            try:
                exception = exception_queue.get_nowait()
                LOGGER.error("Exception encountered at termination.")
                raise exception
            except queue.Empty:
                pass
