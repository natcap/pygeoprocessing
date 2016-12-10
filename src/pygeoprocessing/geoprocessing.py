"""A collection of GDAL dataset and raster utilities."""
import threading
import Queue
import logging
import os
import shutil
import functools
import math
import collections
import exceptions
import heapq
import time
import re

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import numpy
import numpy.ma
import scipy.interpolate
import scipy.sparse
import scipy.signal
import scipy.ndimage
import scipy.signal.signaltools
import shapely.wkt
import shapely.ops
from shapely import speedups
import shapely.prepared

import geoprocessing_core

AggregatedValues = collections.namedtuple(
    'AggregatedValues',
    'total pixel_mean hectare_mean n_pixels pixel_min pixel_max')

LOGGER = logging.getLogger('pygeoprocessing.geoprocessing')
_LOGGING_PERIOD = 5.0  # min 5.0 seconds per update log message for the module
_DEFAULT_GTIFF_CREATION_OPTIONS = ('TILED=YES', 'BIGTIFF=IF_SAFER')

# map gdal types to numpy equivalent
_GDAL_TYPE_TO_NUMPY_LOOKUP = {
    gdal.GDT_Int16: numpy.int16,
    gdal.GDT_Int32: numpy.int32,
    gdal.GDT_UInt16: numpy.uint16,
    gdal.GDT_UInt32: numpy.uint32,
    gdal.GDT_Float32: numpy.float32,
    gdal.GDT_Float64: numpy.float64,
}

# A dictionary to map the resampling method input string to the gdal type
_RESAMPLE_DICT = {
    "nearest": gdal.GRA_NearestNeighbour,
    "bilinear": gdal.GRA_Bilinear,
    "cubic": gdal.GRA_Cubic,
    "cubic_spline": gdal.GRA_CubicSpline,
    "lanczos": gdal.GRA_Lanczos,
    'mode': gdal.GRA_Mode,
    }


def raster_calculator(
        base_raster_path_band_list, local_op, target_raster_path,
        datatype_target, nodata_target, dataset_options=None,
        calc_raster_stats=True):
    """Apply local a raster operation on a stack of rasters.

    This function applies a user defined function across a stack of
    rasters' pixel stack. The rasters in `base_raster_path_band_list` must be
    spatially aligned and have the same cell sizes.

    Parameters:
        base_raster_path_band_list (list): a list of (str, int) tuples where
            the strings are raster paths, and ints are band indexes.
        local_op (function) a function that must take in as many arguments as
            there are elements in `base_raster_path_band_list`.  The will be
            in the same order as the rasters in arguments
            can be treated as parallel memory blocks from the original
            rasters though the function is a parallel
            paradigm and does not express the spatial position of the pixels
            in question at the time of the call.
        target_raster_path (string): the path of the output raster.  The
            projection, size, and cell size will be the same as the rasters
            in `base_raster_path_band_list`.
        datatype_target (gdal datatype; int): the desired GDAL output type of
            the target raster.
        nodata_target (numerical value): the desired nodata value of the
            target raster.
        dataset_options (list): this is an argument list that will be
            passed to the GTiff driver.  Useful for blocksizes, compression,
            etc.
        calculate_raster_stats (boolean): If True, calculates and sets raster
            statistics (min, max, mean, and stdev) for target raster.

    Returns:
        None

    Raises:
        ValueError: invalid input provided
    """
    not_found_paths = []
    for path, _ in base_raster_path_band_list:
        if not os.path.exists(path):
            not_found_paths.append(path)

    if len(not_found_paths) != 0:
        raise exceptions.ValueError(
            "The following files were expected but do not exist on the "
            "filesystem: " + str(not_found_paths))

    if target_raster_path in [x[0] for x in base_raster_path_band_list]:
        raise ValueError(
            "%s is used as a target path, but it is also in the base input "
            "path list %s" % (
                target_raster_path, str(base_raster_path_band_list)))

    raster_info_list = [
        get_raster_info(path_band[0])
        for path_band in base_raster_path_band_list]
    geospatial_info_set = set()
    for raster_info in raster_info_list:
        geospatial_info_set.add(
            (raster_info['pixel_size'],
             raster_info['raster_size'],
             raster_info['geotransform'],
             raster_info['projection']))
    if len(geospatial_info_set) > 1:
        raise ValueError(
            "Input Rasters are not geospatially aligned.  For example the "
            "following geospatial stats are not identical %s" % str(
                geospatial_info_set))

    base_raster_list = [
        gdal.Open(path_band[0]) for path_band in base_raster_path_band_list]
    base_band_list = [
        raster.GetRasterBand(index) for raster, (_, index) in zip(
            base_raster_list, base_raster_path_band_list)]

    base_raster_info = get_raster_info(base_raster_path_band_list[0][0])

    new_raster_from_base(
        base_raster_path_band_list[0][0], target_raster_path, 'GTiff',
        nodata_target, datatype_target, dataset_options=dataset_options)
    target_raster = gdal.Open(target_raster_path, gdal.GA_Update)
    target_band = target_raster.GetRasterBand(1)

    n_cols, n_rows = base_raster_info['raster_size']
    xoff = None
    yoff = None
    last_time = time.time()
    raster_blocks = None
    last_blocksize = None
    target_min = None
    target_max = None
    target_sum = 0.0
    target_n = None
    target_mean = None
    target_stddev = None
    for block_offset in iterblocks(
            base_raster_path_band_list[0][0], offset_only=True):
        xoff, yoff = block_offset['xoff'], block_offset['yoff']
        last_time = _invoke_timed_callback(
            last_time, lambda: LOGGER.info(
                'raster stack calculation approx. %.2f%% complete',
                100.0 * ((n_rows - yoff) * n_cols - xoff) /
                (n_rows * n_cols)), _LOGGING_PERIOD)
        blocksize = (block_offset['win_ysize'], block_offset['win_xsize'])

        if last_blocksize != blocksize:
            raster_blocks = [
                numpy.zeros(blocksize, dtype=_gdal_to_numpy_type(band))
                for band in base_band_list]
            last_blocksize = blocksize

        for dataset_index in xrange(len(base_band_list)):
            band_data = block_offset.copy()
            band_data['buf_obj'] = raster_blocks[dataset_index]
            base_band_list[dataset_index].ReadAsArray(**band_data)

        target_block = local_op(*raster_blocks)

        target_band.WriteArray(
            target_block, xoff=block_offset['xoff'],
            yoff=block_offset['yoff'])

        if calc_raster_stats:
            valid_mask = target_block != nodata_target
            valid_block = target_block[valid_mask]
            target_sum += numpy.sum(valid_block)
            if target_min is None:
                target_min = numpy.min(valid_block)
            else:
                target_min = min(numpy.min(valid_block), target_min)
            if target_max is None:
                target_max = numpy.max(valid_block)
            else:
                target_max = max(numpy.max(valid_block), target_max)
            if target_n is None:
                target_n = numpy.sum(valid_block.size)
            else:
                target_n += numpy.sum(valid_block.size)

    # Making sure the band and dataset is flushed and not in memory before
    # adding stats
    target_band.FlushCache()

    if calc_raster_stats and target_min is not None:
        target_mean = target_sum / float(target_n)
        stdev_sum = 0.0
        for block_offset, target_block in iterblocks(target_raster_path):
            valid_mask = target_block != nodata_target
            valid_block = target_block[valid_mask]
            stdev_sum += numpy.sum((valid_block - target_mean) ** 2)
        target_stddev = (stdev_sum / float(target_n)) ** 0.5

        target_band.SetStatistics(
            float(target_min), float(target_max), float(target_mean),
            float(target_stddev))

    target_band = None
    target_raster = None


def align_and_resize_raster_stack(
        base_raster_path_list, target_raster_path_list, resample_method_list,
        target_pixel_size, bounding_box_mode, base_vector_path_list=None,
        raster_align_index=None,
        gtiff_creation_options=_DEFAULT_GTIFF_CREATION_OPTIONS):
    """Generate rasters from a base such that they align geospatially.

    This function resizes base rasters that are in the same geospatial
    projection such that the result is an aligned stack of rasters that have
    the same cell size, dimensions, and bounding box. This is achieved by
    clipping or resizing the rasters to intersected, unioned, or equivocated
    bounding boxes of all the raster and vector input.

    Parameters:
        base_raster_path_list (list): a list of base raster paths that will
            be transformed and will be used to determine the target bounding
            box.
        base_vector_path_list (list): a list of base vector paths that will
            be used to determine the target bounding box depending on
            the bounding box mode.
        target_raster_path_list (list): a list of raster paths that will be
            created to one-to-one map with `base_raster_path_list` as aligned
            versions of those original rasters.
        resample_method_list (list): a list of resampling methods which
            one to one map each path in `base_raster_path_list` during
            resizing.  Each element must be one of
            "nearest|bilinear|cubic|cubic_spline|lanczos".
        target_pixel_size (tuple): the target raster's x and y pixel size.
        bounding_box_mode (string): one of "union", "intersection", or
            "bb=[minx,miny,maxx,maxy]" which defines how the output output
            extents are defined as the union or intersection of the base
            raster and vectors' bounding boxes, or to have a user defined
            boudning box.
        raster_align_index (int): if not None, then refers to the index of a
            raster in `base_raster_path_list` that the target rasters'
            bounding boxes should perfectly align to.  This feature allows
            rasters whose raster dimensions are the same, but bounding boxes
            slightly shifted less than a pixel size to align with.  If `None`
            then the bounding box of the target rasters is calculated as the
            precise intersection, union, or bounding box.
        gtiff_creation_options (list): list of strings that will be passed
            as GDAL "dataset" creation options to the GTIFF driver, or ignored
            if None.

    Returns:
        None
    """
    last_time = time.time()

    # make sure that the input lists are of the same length
    list_lengths = [
        len(base_raster_path_list), len(target_raster_path_list),
        len(resample_method_list)]
    if len(set(list_lengths)) != 1:
        raise ValueError(
            "base_raster_path_list, target_raster_path_list, and "
            "resample_method_list must be the same length "
            " current lengths are %s" % (str(list_lengths)))

    LOGGER.debug(bounding_box_mode)
    LOGGER.debug(type(bounding_box_mode))
    float_re = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
    LOGGER.debug(
        re.match(r'bb=\[%s,%s,%s,%s\]' % ((float_re,)*4), bounding_box_mode))
    # regular expression to match a float
    if bounding_box_mode not in ["union", "intersection"] and not re.match(
            r'bb=\[%s,%s,%s,%s\]' % ((float_re,)*4), bounding_box_mode):
        raise ValueError("Unknown bounding_box_mode %s" % (
            str(bounding_box_mode)))

    if ((raster_align_index is not None) and
            (0 > raster_align_index >= len(base_raster_path_list))):
        raise ValueError(
            "Alignment index is out of bounds of the datasets index: %s"
            "n_elements %s" % (
                raster_align_index, len(base_raster_path_list)))

    raster_info_list = [
        get_raster_info(path) for path in base_raster_path_list]
    if base_vector_path_list is not None:
        vector_info_list = [
            get_vector_info(path) for path in base_vector_path_list]
    else:
        vector_info_list = []

    # get the literal or intersecting/unioned bounding box
    bb_match = re.match(
        r'bb=\[(%s),(%s),(%s),(%s)\]' % ((float_re,)*4), bounding_box_mode)
    if bb_match:
        target_bounding_box = [float(x) for x in bb_match.groups()]
    else:
        # either intersection or union
        target_bounding_box = reduce(
            functools.partial(_merge_bounding_boxes, mode=bounding_box_mode),
            [info['bounding_box'] for info in
             (raster_info_list + vector_info_list)])

    if bounding_box_mode == "intersection" and (
            target_bounding_box[0] >= target_bounding_box[2] or
            target_bounding_box[1] <= target_bounding_box[3]):
        raise ValueError("The rasters' and vectors' intersection is empty "
                         "(not all rasters and vectors touch each other).")

    if raster_align_index >= 0:
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

    for index, (base_path, target_path, resample_method) in enumerate(zip(
            base_raster_path_list, target_raster_path_list,
            resample_method_list)):
        last_time = _invoke_timed_callback(
            last_time, lambda: LOGGER.info(
                "align_dataset_list aligning dataset %d of %d",
                index, len(base_raster_path_list)), _LOGGING_PERIOD)
        warp_raster(
            base_path, target_pixel_size,
            target_path, resample_method,
            target_bb=target_bounding_box,
            gtiff_creation_options=gtiff_creation_options)


def calculate_raster_stats(dataset_path):
    """Calculate min, max, stdev, and mean for all bands in dataset.

    Args:
        dataset_path (string): a path to a GDAL raster dataset that will be
            modified by having its band statistics set

    Returns:
        None
    """

    dataset = gdal.Open(dataset_path, gdal.GA_Update)

    for band_number in range(dataset.RasterCount):
        band = dataset.GetRasterBand(band_number + 1)
        band.ComputeStatistics(False)

    # Close and clean up dataset
    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None


def new_raster_from_base(
        base_path, target_path, datatype, nodata_list, n_bands=1,
        fill_value_list=None, n_rows=None, n_cols=None,
        gtiff_creation_options=()):
    """Create new geotiff by coping spatial reference/geotransform of base.

    A wrapper for the function new_raster_from_base that opens up
    the base_path before passing it to new_raster_from_base.

    Parameters:
        base_path (string): path to existing raster.
        target_path (string): path to desired target raster.
        datatype: the pixel datatype of the output raster, for example
            gdal.GDT_Float32.  See the following header file for supported
            pixel types:
            http://www.gdal.org/gdal_8h.html#22e22ce0a55036a96f652765793fb7a4
        nodata_list (list): list of nodata values, one for each band, to set
            on target raster; okay to have 'None' values.
        n_bands (int): number of bands for the target raster.
        fill_value_list (list): list of values to fill each band with. If None,
            no filling is done.
        n_rows (int): if not None, defines the number of target raster rows.
        n_cols (int): if not None, defines the number of target raster
            columns.
        gtiff_creation_options: a list of dataset options that gets
            passed to the gdal creation driver, overrides defaults

    Returns:
        nothing
    """
    # nodata might be a numpy type coming in, set it to native python type
    base_raster = gdal.Open(base_path)
    if n_rows is None:
        n_rows = base_raster.RasterYSize
    if n_cols is None:
        n_cols = base_raster.RasterXSize
    driver = gdal.GetDriverByName('GTiff')

    base_band = base_raster.GetRasterBand(1)
    block_size = base_band.GetBlockSize()
    metadata = base_band.GetMetadata('IMAGE_STRUCTURE')

    local_gtiff_creation_options = list(gtiff_creation_options)
    # PIXELTYPE is sometimes used to define signed vs. unsigned bytes and
    # the only place that is stored is in the IMAGE_STRUCTURE metadata
    # copy it over if it exists; get this info from the first band since
    # all bands have the same datatype
    metadata = base_band.GetMetadata('IMAGE_STRUCTURE')
    base_band = None
    if 'PIXELTYPE' in metadata:
        local_gtiff_creation_options.append(
            'PIXELTYPE=' + metadata['PIXELTYPE'])

    # first, should it be tiled?  yes if it's not striped
    if block_size[0] != n_cols:
        local_gtiff_creation_options.extend([
            'TILED=YES',
            'BLOCKXSIZE=%d' % block_size[0],
            'BLOCKYSIZE=%d' % block_size[1],
            'BIGTIFF=IF_SAFER'])

    target_raster = driver.Create(
        target_path.encode('utf-8'), n_cols, n_rows, n_bands, datatype,
        options=gtiff_creation_options)
    target_raster.SetProjection(base_raster.GetProjection())
    target_raster.SetGeoTransform(base_raster.GetGeoTransform())
    base_raster = None

    for index, nodata_value in enumerate(nodata_list):
        if nodata_value is None:
            continue
        target_band = target_raster.GetRasterBand(index + 1)
        try:
            target_band.SetNoDataValue(nodata_value.item())
        except AttributeError:
            target_band.SetNoDataValue(nodata_value)

    if fill_value_list is not None:
        for index, fill_value in enumerate(fill_value_list):
            if fill_value is None:
                continue
            target_band = target_raster.GetRasterBand(index + 1)
            target_band.Fill(fill_value)
            target_band = None

    target_raster = None


def create_raster_from_vector_extents_uri(
        shapefile_uri, pixel_size, gdal_format, nodata_out_value, output_uri):
    """Create a blank raster based on a vector file extent.

    A wrapper for create_raster_from_vector_extents

    Args:
        shapefile_uri (string): uri to an OGR datasource to use as the extents
            of the raster
        pixel_size: size of output pixels in the projected units of
            shapefile_uri
        gdal_format: the raster pixel format, something like
            gdal.GDT_Float32
        nodata_out_value: the output nodata value
        output_uri (string): the URI to write the gdal dataset

    Returns:
        dataset (gdal.Dataset): gdal dataset
    """
    datasource = ogr.Open(shapefile_uri)
    create_raster_from_vector_extents(
        pixel_size, pixel_size, gdal_format, nodata_out_value, output_uri,
        datasource)


def create_raster_from_vector_extents(
        xRes, yRes, format, nodata, rasterFile, shp):
    """Create a blank raster based on a vector file extent.

    This code is adapted from http://trac.osgeo.org/gdal/wiki/FAQRaster#HowcanIcreateablankrasterbasedonavectorfilesextentsforusewithgdal_rasterizeGDAL1.8.0

    Args:
        xRes: the x size of a pixel in the output dataset must be a
            positive value
        yRes: the y size of a pixel in the output dataset must be a
            positive value
        format: gdal GDT pixel type
        nodata: the output nodata value
        rasterFile (string): URI to file location for raster
        shp: vector shapefile to base extent of output raster on

    Returns:
        raster: blank raster whose bounds fit within `shp`s bounding box
            and features are equivalent to the passed in data
    """

    # Determine the width and height of the tiff in pixels based on the
    # maximum size of the combined envelope of all the features
    shp_extent = None
    for layer_index in range(shp.GetLayerCount()):
        shp_layer = shp.GetLayer(layer_index)
        for feature_index in range(shp_layer.GetFeatureCount()):
            try:
                feature = shp_layer.GetFeature(feature_index)
                geometry = feature.GetGeometryRef()

                # feature_extent = [xmin, xmax, ymin, ymax]
                feature_extent = geometry.GetEnvelope()
                # This is an array based way of mapping the right function
                # to the right index.
                functions = [min, max, min, max]
                for i in range(len(functions)):
                    try:
                        shp_extent[i] = functions[i](
                            shp_extent[i], feature_extent[i])
                    except TypeError:
                        # need to cast to list because returned as a tuple
                        # and we can't assign to a tuple's index, also need to
                        # define this as the initial state
                        shp_extent = list(feature_extent)
            except AttributeError as e:
                # For some valid OGR objects the geometry can be undefined
                # since it's valid to have a NULL entry in the attribute table
                # this is expressed as a None value in the geometry reference
                # this feature won't contribute
                LOGGER.warn(e)

    tiff_width = int(numpy.ceil(abs(shp_extent[1] - shp_extent[0]) / xRes))
    tiff_height = int(numpy.ceil(abs(shp_extent[3] - shp_extent[2]) / yRes))

    if rasterFile is not None:
        driver = gdal.GetDriverByName('GTiff')
    else:
        rasterFile = ''
        driver = gdal.GetDriverByName('MEM')
    # 1 means only create 1 band
    raster = driver.Create(
        rasterFile, tiff_width, tiff_height, 1, format,
        options=['BIGTIFF=IF_SAFER'])
    raster.GetRasterBand(1).SetNoDataValue(nodata)

    # Set the transform based on the upper left corner and given pixel
    # dimensions
    raster_transform = [shp_extent[0], xRes, 0.0, shp_extent[3], 0.0, -yRes]
    raster.SetGeoTransform(raster_transform)

    # Use the same projection on the raster as the shapefile
    srs = osr.SpatialReference()
    srs.ImportFromWkt(shp.GetLayer(0).GetSpatialRef().__str__())
    raster.SetProjection(srs.ExportToWkt())

    # Initialize everything to nodata
    raster.GetRasterBand(1).Fill(nodata)
    raster.GetRasterBand(1).FlushCache()

    return raster


def vectorize_points_uri(
        shapefile_uri, field, output_uri, interpolation='nearest'):
    """Interpolate values in shapefile onto given raster.

    A wrapper function for pygeoprocessing.vectorize_points, that allows for
    uri passing.

    Args:
        shapefile_uri (string): a uri path to an ogr shapefile
        field (string): a string for the field name
        output_uri (string): a uri path for the output raster
        interpolation (string): interpolation method to use on points, default
            is 'nearest'

    Returns:
        None
    """

    datasource = ogr.Open(shapefile_uri)
    output_raster = gdal.Open(output_uri, 1)
    vectorize_points(
        datasource, field, output_raster, interpolation=interpolation)


def vectorize_points(
        shapefile, datasource_field, dataset, randomize_points=False,
        mask_convex_hull=False, interpolation='nearest'):
    """Interpolate values in shapefile onto given raster.

    Takes a shapefile of points and a field defined in that shapefile
    and interpolate the values in the points onto the given raster

    Args:
        shapefile: ogr datasource of points
        datasource_field: a field in shapefile
        dataset: a gdal dataset must be in the same projection as shapefile

    Keyword Args:
        randomize_points (boolean): (description)
        mask_convex_hull (boolean): (description)
        interpolation (string): the interpolation method to use for
            scipy.interpolate.griddata(). Default is 'nearest'

    Returns:
       None
    """

    # Define the initial bounding box
    gt = dataset.GetGeoTransform()
    # order is left, top, right, bottom of rasterbounds
    bounding_box = [gt[0], gt[3], gt[0] + gt[1] * dataset.RasterXSize,
                    gt[3] + gt[5] * dataset.RasterYSize]

    def in_bounds(point):
        return point[0] <= bounding_box[2] and point[0] >= bounding_box[0] \
            and point[1] <= bounding_box[1] and point[1] >= bounding_box[3]

    layer = shapefile.GetLayer(0)
    point_list = []
    value_list = []

    # Calculate a small amount to perturb points by so that we don't
    # get a linear Delauney triangle, the 1e-6 is larger than eps for
    # floating point, but large enough not to cause errors in interpolation.
    delta_difference = 1e-6 * min(abs(gt[1]), abs(gt[5]))
    if randomize_points:
        random_array = numpy.random.randn(layer.GetFeatureCount(), 2)
        random_offsets = random_array*delta_difference
    else:
        random_offsets = numpy.zeros((layer.GetFeatureCount(), 2))

    for feature_id in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(feature_id)
        geometry = feature.GetGeometryRef()
        # Here the point geometry is in the form x, y (col, row)
        point = geometry.GetPoint()
        if in_bounds(point):
            value = feature.GetField(datasource_field)
            # Add in the numpy notation which is row, col
            point_list.append([point[1]+random_offsets[feature_id, 1],
                               point[0]+random_offsets[feature_id, 0]])
            value_list.append(value)

    point_array = numpy.array(point_list)
    value_array = numpy.array(value_list)

    band = dataset.GetRasterBand(1)

    # Create grid points for interpolation outputs later
    # top-bottom:y_stepsize, left-right:x_stepsize

    # Make as an integer grid then divide subtract by bounding box parts
    # so we don't get a roundoff error and get off by one pixel one way or
    # the other
    grid_y, grid_x = numpy.mgrid[0:band.YSize, 0:band.XSize]
    grid_y = grid_y * gt[5] + bounding_box[1]
    grid_x = grid_x * gt[1] + bounding_box[0]
    nodata = band.GetNoDataValue()

    raster_out_array = scipy.interpolate.griddata(
        point_array, value_array, (grid_y, grid_x), interpolation, nodata)
    band.WriteArray(raster_out_array, 0, 0)

    # Make sure the dataset is closed and cleaned up
    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None


def aggregate_raster_values_uri(
        raster_path_band, shapefile_uri, shapefile_field=None, ignore_nodata=True,
        all_touched=False, polygons_might_overlap=True):
    """Collect stats on pixel values which lie within shapefile polygons.

    Args:
        raster_path_band (tuple): a str, int tuple where the string indicates
            the path on disk to a raster and the int is the band index.  In
            order for hectare mean values to be accurate, this raster must be
            projected in meter units.
        shapefile_uri (string): a uri to a OGR datasource that should overlap
            raster; raises an exception if not.

    Keyword Args:
        shapefile_field (string): a string indicating which key in shapefile to
            associate the output dictionary values with whose values are
            associated with ints; if None dictionary returns a value over
            the entire shapefile region that intersects the raster.
        ignore_nodata: if operation == 'mean' then it does not
            account for nodata pixels when determining the pixel_mean,
            otherwise all pixels in the AOI are used for calculation of the
            mean.  This does not affect hectare_mean which is calculated from
            the geometrical area of the feature.
        all_touched (boolean): if true will account for any pixel whose
            geometry passes through the pixel, not just the center point
        polygons_might_overlap (boolean): if True the function calculates
            aggregation coverage close to optimally by rasterizing sets of
            polygons that don't overlap.  However, this step can be
            computationally expensive for cases where there are many polygons.
            Setting this flag to False directs the function rasterize in one
            step.

    Returns:
        result_tuple (tuple): named tuple of the form
           ('aggregate_values', 'total pixel_mean hectare_mean n_pixels
            pixel_min pixel_max')
           Each of [sum pixel_mean hectare_mean] contains a dictionary that
           maps shapefile_field value to the total, pixel mean, hecatare mean,
           pixel max, and pixel min of the values under that feature.
           'n_pixels' contains the total number of valid pixels used in that
           calculation.  hectare_mean is None if raster_uri is unprojected.

    Raises:
        AttributeError
        TypeError
        OSError
    """
    raster_info = get_raster_info(raster_path_band[0])
    raster_nodata = raster_info['nodata'][raster_path_band[1]]
    out_pixel_size = raster_info['mean_pixel_size']
    clipped_raster_uri = temporary_filename(suffix='.tif')
    vectorize_datasets(
        [raster_path_band], lambda x: x, clipped_raster_uri, gdal.GDT_Float64,
        raster_nodata, out_pixel_size, "union",
        dataset_to_align_index=0, aoi_path=shapefile_uri,
        assert_datasets_projected=False, vectorize_op=False)
    clipped_raster = gdal.Open(clipped_raster_uri)

    # This should be a value that's not in shapefile[shapefile_field]
    mask_nodata = -1
    mask_uri = temporary_filename(suffix='.tif')
    new_raster_from_base_uri(
        clipped_raster_uri, mask_uri, 'GTiff', mask_nodata,
        gdal.GDT_Int32, fill_value=mask_nodata)

    mask_dataset = gdal.Open(mask_uri, gdal.GA_Update)
    shapefile = ogr.Open(shapefile_uri)
    shapefile_layer = shapefile.GetLayer()
    rasterize_layer_args = {
        'options': ['ALL_TOUCHED=%s' % str(all_touched).upper()],
    }

    if shapefile_field is not None:
        # Make sure that the layer name refers to an integer
        layer_d = shapefile_layer.GetLayerDefn()
        field_index = layer_d.GetFieldIndex(shapefile_field)
        if field_index == -1:  # -1 returned when field does not exist.
            # Raise exception if user provided a field that's not in vector
            raise AttributeError(
                'Vector %s must have a field named %s' %
                (shapefile_uri, shapefile_field))

        field_def = layer_d.GetFieldDefn(field_index)
        if field_def.GetTypeName() != 'Integer':
            raise TypeError(
                'Can only aggregate by integer based fields, requested '
                'field is of type  %s' % field_def.GetTypeName())
        # Adding the rasterize by attribute option
        rasterize_layer_args['options'].append(
            'ATTRIBUTE=%s' % shapefile_field)
    else:
        # 9999 is a classic unknown value
        global_id_value = 9999
        rasterize_layer_args['burn_values'] = [global_id_value]

    # loop over the subset of feature layers and rasterize/aggregate each one
    aggregate_dict_values = {}
    aggregate_dict_counts = {}
    result_tuple = AggregatedValues(
        total={},
        pixel_mean={},
        hectare_mean={},
        n_pixels={},
        pixel_min={},
        pixel_max={})

    # make a shapefile that non-overlapping layers can be added to
    driver = ogr.GetDriverByName('ESRI Shapefile')
    layer_dir = temporary_folder()
    subset_layer_datasouce = driver.CreateDataSource(
        os.path.join(layer_dir, 'subset_layer.shp'))
    spat_ref = shapefile_layer.GetSpatialRef()
    subset_layer = subset_layer_datasouce.CreateLayer(
        'subset_layer', spat_ref, ogr.wkbPolygon)
    defn = shapefile_layer.GetLayerDefn()

    # For every field, create a duplicate field and add it to the new
    # subset_layer layer
    defn.GetFieldCount()
    for fld_index in range(defn.GetFieldCount()):
        original_field = defn.GetFieldDefn(fld_index)
        output_field = ogr.FieldDefn(
            original_field.GetName(), original_field.GetType())
        subset_layer.CreateField(output_field)

    # Initialize these dictionaries to have the shapefile fields in the
    # original datasource even if we don't pick up a value later

    # This will store the sum/count with index of shapefile attribute
    if shapefile_field is not None:
        shapefile_table = extract_datasource_table_by_key(
            shapefile_uri, shapefile_field)
    else:
        shapefile_table = {global_id_value: 0.0}

    current_iteration_shapefiles = dict([
        (shapefile_id, 0.0) for shapefile_id in shapefile_table.iterkeys()])
    aggregate_dict_values = current_iteration_shapefiles.copy()
    aggregate_dict_counts = current_iteration_shapefiles.copy()
    # Populate the means and totals with something in case the underlying
    # raster doesn't exist for those features.  we use -9999 as a recognizable
    # nodata value.
    for shapefile_id in shapefile_table:
        result_tuple.pixel_mean[shapefile_id] = -9999
        result_tuple.total[shapefile_id] = -9999
        result_tuple.hectare_mean[shapefile_id] = -9999

    pixel_min_dict = dict(
        [(shapefile_id, None) for shapefile_id in shapefile_table.iterkeys()])
    pixel_max_dict = pixel_min_dict.copy()

    # Loop over each polygon and aggregate
    if polygons_might_overlap:
        minimal_polygon_sets = calculate_disjoint_polygon_set(
            shapefile_uri)
    else:
        minimal_polygon_sets = [
            set([feat.GetFID() for feat in shapefile_layer])]

    clipped_band = clipped_raster.GetRasterBand(1)

    for polygon_set in minimal_polygon_sets:
        # add polygons to subset_layer
        for poly_fid in polygon_set:
            poly_feat = shapefile_layer.GetFeature(poly_fid)
            subset_layer.CreateFeature(poly_feat)
        subset_layer_datasouce.SyncToDisk()

        # nodata out the mask
        mask_band = mask_dataset.GetRasterBand(1)
        mask_band.Fill(mask_nodata)
        mask_band = None

        gdal.RasterizeLayer(
            mask_dataset, [1], subset_layer, **rasterize_layer_args)
        mask_dataset.FlushCache()

        # get feature areas
        feature_areas = collections.defaultdict(int)
        for feature in subset_layer:
            # feature = subset_layer.GetFeature(index)
            geom = feature.GetGeometryRef()
            if shapefile_field is not None:
                feature_id = feature.GetField(shapefile_field)
                feature_areas[feature_id] = geom.GetArea()
            else:
                feature_areas[global_id_value] += geom.GetArea()
        subset_layer.ResetReading()
        geom = None

        # Need a complicated step to see what the FIDs are in the subset_layer
        # then need to loop through and delete them
        fid_to_delete = set()
        for feature in subset_layer:
            fid_to_delete.add(feature.GetFID())
        subset_layer.ResetReading()
        for fid in fid_to_delete:
            subset_layer.DeleteFeature(fid)
        subset_layer_datasouce.SyncToDisk()

        current_iteration_attribute_ids = set()

        for mask_offsets, mask_block in iterblocks(mask_uri):
            clipped_block = clipped_band.ReadAsArray(**mask_offsets)

            unique_ids = numpy.unique(mask_block)
            current_iteration_attribute_ids = (
                current_iteration_attribute_ids.union(unique_ids))
            for attribute_id in unique_ids:
                # ignore masked values
                if attribute_id == mask_nodata:
                    continue

                # Consider values which lie in the polygon for attribute_id
                masked_values = clipped_block[
                    (mask_block == attribute_id) &
                    (~numpy.isnan(clipped_block))]
                # Remove the nodata and ignore values for later processing
                masked_values_nodata_removed = (
                    masked_values[~numpy.in1d(
                        masked_values, [raster_nodata]).
                                  reshape(masked_values.shape)])

                # Find the min and max which might not yet be calculated
                if masked_values_nodata_removed.size > 0:
                    if pixel_min_dict[attribute_id] is None:
                        pixel_min_dict[attribute_id] = numpy.min(
                            masked_values_nodata_removed)
                        pixel_max_dict[attribute_id] = numpy.max(
                            masked_values_nodata_removed)
                    else:
                        pixel_min_dict[attribute_id] = min(
                            pixel_min_dict[attribute_id],
                            numpy.min(masked_values_nodata_removed))
                        pixel_max_dict[attribute_id] = max(
                            pixel_max_dict[attribute_id],
                            numpy.max(masked_values_nodata_removed))

                if ignore_nodata:
                    # Only consider values which are not nodata values
                    aggregate_dict_counts[attribute_id] += (
                        masked_values_nodata_removed.size)
                else:
                    aggregate_dict_counts[attribute_id] += masked_values.size

                aggregate_dict_values[attribute_id] += numpy.sum(
                    masked_values_nodata_removed)

        # Initialize the dictionary to have an n_pixels field that contains the
        # counts of all the pixels used in the calculation.
        result_tuple.n_pixels.update(aggregate_dict_counts.copy())
        result_tuple.pixel_min.update(pixel_min_dict.copy())
        result_tuple.pixel_max.update(pixel_max_dict.copy())
        # Don't want to calculate stats for the nodata
        current_iteration_attribute_ids.discard(mask_nodata)
        for attribute_id in current_iteration_attribute_ids:
            result_tuple.total[attribute_id] = (
                aggregate_dict_values[attribute_id])

            # intitalize to 0
            result_tuple.pixel_mean[attribute_id] = 0.0
            result_tuple.hectare_mean[attribute_id] = 0.0

            if aggregate_dict_counts[attribute_id] != 0.0:
                n_pixels = aggregate_dict_counts[attribute_id]
                result_tuple.pixel_mean[attribute_id] = (
                    aggregate_dict_values[attribute_id] / n_pixels)

                # To get the total area multiply n pixels by their area then
                # divide by 10000 to get Ha.  Notice that's in the denominator
                # so the * 10000 goes on the top
                if feature_areas[attribute_id] != 0:
                    result_tuple.hectare_mean[attribute_id] = 10000.0 * (
                        aggregate_dict_values[attribute_id] /
                        feature_areas[attribute_id])

    # Make sure the dataset is closed and cleaned up
    mask_band = None
    gdal.Dataset.__swig_destroy__(mask_dataset)
    mask_dataset = None

    clipped_band = None
    gdal.Dataset.__swig_destroy__(clipped_raster)
    clipped_raster = None

    for filename in [mask_uri, clipped_raster_uri]:
        try:
            os.remove(filename)
        except OSError as error:
            LOGGER.warn(
                "couldn't remove file %s. Exception %s", filename, str(error))

    subset_layer = None
    ogr.DataSource.__swig_destroy__(subset_layer_datasouce)
    subset_layer_datasouce = None
    try:
        shutil.rmtree(layer_dir)
    except OSError as error:
        LOGGER.warn(
            "couldn't remove directory %s.  Exception %s", layer_dir,
            str(error))

    return result_tuple


def calculate_slope(
        dem_dataset_uri, slope_uri):
    """Create slope raster from DEM raster.

    Follows the algorithm described here:
    http://webhelp.esri.com/arcgiSDEsktop/9.3/index.cfm?TopicName=How%20Slope%20works

    Args:
        dem_dataset_uri (string): a URI to a  single band raster of z values.
        slope_uri (string): a path to the output slope uri in percent.

    Returns:
        None
    """
    slope_nodata = numpy.finfo(numpy.float32).min
    new_raster_from_base_uri(
        dem_dataset_uri, slope_uri, 'GTiff', slope_nodata, gdal.GDT_Float32)
    geoprocessing_core._cython_calculate_slope(dem_dataset_uri, slope_uri)
    calculate_raster_stats_uri(slope_uri)


def get_vector_info(vector_path):
    """Get information about an OGR vector (datasource).

    Parameters:
       vector_path (String): a path to a OGR vector.

    Returns:
        raster_properties (dictionary): a dictionary with the properties
            stored under relevant keys.

            'projection' (string): projection of the vector in Well Known
                Text.
            'bounding_box' (list): list of floats representing the bounding
                box in projected coordinates as "bb=[minx,miny,maxx,maxy]".
    """
    vector = ogr.Open(vector_path)
    vector_properties = {}
    first_layer = vector.GetLayer()
    # projection is same for all layers, so just use the first one
    vector['projection'] = first_layer.GetSpatialRef().ExportToWkt()
    for layer in vector:
        layer_bb = layer.GetExtent()
        # convert form [minx,maxx,miny,maxy] to [minx,miny,maxx,maxy]
        local_bb = [layer_bb[i] for i in [0, 2, 1, 3]]
        if 'bounding_box' not in vector_properties:
            vector_properties['bounding_box'] = local_bb
        else:
            vector_properties['bounding_box'] = _merge_bounding_boxes(
                vector_properties['bounding_box'], local_bb, 'union')


def get_raster_info(raster_path):
    """Get information about a GDAL raster (dataset).

    Parameters:
       raster_path (String): a path to a GDAL raster.

    Returns:
        raster_properties (dictionary): a dictionary with the properties
            stored under relevant keys.

            'pixel_size' (tuple): (pixel x-size, pixel y-size) from
                geotransform.
            'mean_pixel_size' (float): the average size of the absolute value
                of each pixel size element.
            'raster_size' (tuple):  number of raster pixels in (x, y)
                direction.
            'nodata' (list): a list of the nodata values in the bands of the
                raster in the same order as increasing band index.
            'n_bands' (int): number of bands in the raster.
            'geotransform' (tuple): a 6-tuple representing the geotransform of
                (x orign, x-increase, xy-increase,
                 y origin, yx-increase, y-increase).
            'datatype' (int): An instance of an enumerated gdal.GDT_* int
                that represents the datatype of the raster.
            'projection' (string): projection of the raster in Well Known
                Text.
            'bounding_box' (list): list of floats representing the bounding
                box in projected coordinates as "bb=[minx,miny,maxx,maxy]".
    """
    raster_properties = {}
    raster = gdal.Open(raster_path)
    raster_properties['projection'] = raster.GetProjection()
    geo_transform = raster.GetGeoTransform()
    raster_properties['geotransform'] = geo_transform
    raster_properties['pixel_size'] = (geo_transform[1], geo_transform[5])
    raster_properties['mean_pixel_size'] = (
        (abs(geo_transform[1]) + abs(geo_transform[5])) / 2.0)
    raster_properties['raster_size'] = (
        raster.GetRasterBand(1).XSize,
        raster.GetRasterBand(1).YSize)
    raster_properties['n_bands'] = raster.RasterCount
    raster_properties['nodata'] = [
        raster.GetRasterBand(index).GetNoDataValue() for index in xrange(
            1, raster_properties['n_bands']+1)]

    raster_properties['bounding_box'] = [
        geo_transform[0], geo_transform[3],
        (geo_transform[0] +
         raster_properties['raster_size'][0] * geo_transform[1]),
        (geo_transform[3] +
         raster_properties['raster_size'][1] * geo_transform[5])]

    # datatype is the same for the whole raster, but is associated with band
    raster_properties['datatype'] = raster.GetRasterBand(1).DataType
    raster = None
    return raster_properties


def reproject_vector(base_vector_path, target_wkt, target_path):
    """Reproject OGR DataSource (vector).

    Transforms the features of the base vector to the desired output
    projection in a new file.

    Parameters:
        base_vector_path (string): Path to the base shapefile to transform.
        target_wkt (string): the desired output projection in Well Known Text
            (by layer.GetSpatialRef().ExportToWkt())
        target_path (string): the filepath to the transformed shapefile

    Returns:
        None
    """
    base_vector = ogr.Open(base_vector_path)

    # if this file already exists, then remove it
    if os.path.isfile(target_path):
        LOGGER.warn(
            "reproject_vector: %s already exists, removing and overwriting",
            target_path)
        os.remove(target_path)

    target_sr = osr.SpatialReference()
    target_sr.ImportFromWkt(target_wkt)

    # create a new shapefile from the orginal_datasource
    target_driver = ogr.GetDriverByName('ESRI Shapefile')
    target_vector = target_driver.CreateDataSource(target_path)

    # loop through all the layers in the orginal_datasource
    for layer in base_vector:
        layer_dfn = layer.GetLayerDefn()

        # Create new layer for target_vector using same name and
        # geometry type from base vector but new projection
        target_layer = target_vector.CreateLayer(
            layer_dfn.GetName(), target_sr, layer_dfn.GetGeomType())

        # Get the number of fields in original_layer
        original_field_count = layer_dfn.GetFieldCount()

        # For every field, create a duplicate field in the new layer
        for fld_index in xrange(original_field_count):
            original_field = layer_dfn.GetFieldDefn(fld_index)
            target_field = ogr.FieldDefn(
                original_field.GetName(), original_field.GetType())
            target_layer.CreateField(target_field)

        # Get the SR of the original_layer to use in transforming
        base_sr = layer.GetSpatialRef()

        # Create a coordinate transformation
        coord_trans = osr.CoordinateTransformation(base_sr, target_sr)

        # Copy all of the features in layer to the new shapefile
        error_count = 0
        for base_feature in layer:
            geom = base_feature.GetGeometryRef()

            # Transform geometry into format desired for the new projection
            error_code = geom.Transform(coord_trans)
            if error_code != 0:  # error
                # this could be caused by an out of range transformation
                # whatever the case, don't put the transformed poly into the
                # output set
                error_count += 1
                continue

            # Copy original_datasource's feature and set as new shapes feature
            target_feature = ogr.Feature(target_layer.GetLayerDefn())
            target_feature.SetGeometry(geom)

            # For all the fields in the feature set the field values from the
            # source field
            for fld_index in xrange(target_feature.GetFieldCount()):
                target_feature.SetField(
                    fld_index, base_feature.GetField(fld_index))

            target_layer.CreateFeature(target_feature)
            target_feature = None
            base_feature = None
        if error_count > 0:
            LOGGER.warn(
                '%d features out of %d were unable to be transformed and are'
                ' not in the output vector at %s', error_count,
                layer.GetFeatureCount(), target_path)
        layer = None


def reclassify_dataset_uri(
        dataset_uri, value_map, raster_out_uri, out_datatype, out_nodata,
        exception_flag='values_required', assert_dataset_projected=True,
        band_index=0):
    """Reclassify values in a dataset.

    A function to reclassify values in dataset to any output type. By default
    the values except for nodata must be in value_map.

    Args:
        dataset_uri (string): a uri to a gdal dataset
        value_map (dictionary): a dictionary of values of
            {source_value: dest_value, ...}
            where source_value's type is a postive integer type and dest_value
            is of type out_datatype.
        raster_out_uri (string): the uri for the output raster
        out_datatype (gdal type): the type for the output dataset
        out_nodata (numerical type): the nodata value for the output raster.
            Must be the same type as out_datatype
        band_index (int): Indicates which band in `dataset_uri` the
            reclassification should operate on.  Defaults to 0.

    Keyword Args:
        exception_flag (string): either 'none' or 'values_required'.
            If 'values_required' raise an exception if there is a value in the
            raster that is not found in value_map
        assert_dataset_projected (boolean): if True this operation will
            test if the input dataset is not projected and raise an exception
            if so.

    Returns:
        nothing

    Raises:
        Exception: if exception_flag == 'values_required' and the value from
           'key_raster' is not a key in 'attr_dict'
    """
    if exception_flag not in ['none', 'values_required']:
        raise ValueError('unknown exception_flag %s', exception_flag)
    values_required = exception_flag == 'values_required'

    raster_info = get_raster_info(dataset_uri)
    nodata = raster_info['nodata'][band_index]
    value_map_copy = value_map.copy()
    # possible that nodata value is not defined, so test for None first
    # otherwise if nodata not predefined, remap it into the dictionary
    if nodata is not None and nodata not in value_map_copy:
        value_map_copy[nodata] = out_nodata
    keys = sorted(numpy.array(value_map_copy.keys()))
    values = numpy.array([value_map_copy[x] for x in keys])

    def map_dataset_to_value(original_values):
        """Convert a block of original values to the lookup values."""
        if values_required:
            unique = numpy.unique(original_values)
            has_map = numpy.in1d(unique, keys)
            if not all(has_map):
                raise ValueError(
                    'There was not a value for at least the following codes '
                    '%s for this file %s.\nNodata value is: %s' % (
                        str(unique[~has_map]), dataset_uri, str(nodata)))
        index = numpy.digitize(original_values.ravel(), keys, right=True)
        return values[index].reshape(original_values.shape)

    out_pixel_size = raster_info['mean_pixel_size']
    vectorize_datasets(
        [dataset_uri], map_dataset_to_value,
        raster_out_uri, out_datatype, out_nodata, out_pixel_size,
        "intersection", dataset_to_align_index=0,
        vectorize_op=False, assert_datasets_projected=assert_dataset_projected,
        datasets_are_pre_aligned=True)


class DatasetUnprojected(Exception):
    """An exception in case a dataset is unprojected"""
    pass


class DifferentProjections(Exception):
    """An exception in case a set of datasets are not in the same projection"""
    pass


def get_datasource_bounding_box(datasource_uri):
    """Get datasource bounding box where coordinates are in projected units.

    Args:
        dataset_uri (string): a uri to a GDAL dataset

    Returns:
        bounding_box (list):
            [upper_left_x, upper_left_y, lower_right_x, lower_right_y] in
            projected coordinates
    """
    datasource = ogr.Open(datasource_uri)
    layer = datasource.GetLayer(0)
    extent = layer.GetExtent()
    # Reindex datasource extents into the upper left/lower right coordinates
    bounding_box = [extent[0],
                    extent[3],
                    extent[1],
                    extent[2]]
    return bounding_box


def warp_raster(
        base_raster_path, target_pixel_size, target_raster_path,
        resample_method, target_bb=None, target_sr_wkt=None,
        gtiff_creation_options=_DEFAULT_GTIFF_CREATION_OPTIONS):
    """Resize/resample raster to desired pixel size, bbox and projection.

    Parameters:
        base_raster_path (string): path to base raster.
        target_pixel_size (list): a two element list or tuple indicating the
            x and y pixel size in projected units.
        target_raster_path (string): the location of the resized and
            resampled raster.
        resample_method (string): the resampling technique, one of
            "nearest|bilinear|cubic|cubic_spline|lanczos"
        target_bb (list): if None, target bounding box is the same as the
            source bounding box.  Otherwise it's a list of float describing
            target bounding box in target coordinate system as
            [minx, miny, maxx, maxy].
        target_sr_wkt (string): if not None, desired target projection in Well
            Known Text format.
        gtiff_creation_options (list or tuple): list of strings that will be
            passed as GDAL "dataset" creation options to the GTIFF driver.

    Returns:
        None
    """
    resample_dict = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubic_spline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos
        }

    base_raster = gdal.Open(base_raster_path)
    base_sr = osr.SpatialReference()
    base_sr.ImportFromWkt(base_raster.GetProjection())

    if target_bb is None:
        target_bb = get_raster_info(base_raster_path)['bounding_box']

    target_geotransform = [
        target_bb[0], target_pixel_size[0], 0.0, target_bb[1], 0.0,
        target_pixel_size[1]]
    target_x_size = abs(int(numpy.round(
        (target_bb[2] - target_bb[0]) / target_pixel_size[0])))
    target_y_size = abs(int(numpy.round(
        (target_bb[3] - target_bb[1]) / target_pixel_size[1])))

    if target_x_size == 0:
        LOGGER.warn(
            "bounding_box is so small that x dimension rounds to 0; "
            "clamping to 1.")
        target_x_size = 1
    if target_y_size == 0:
        LOGGER.warn(
            "bounding_box is so small that y dimension rounds to 0; "
            "clamping to 1.")
        target_y_size = 1

    local_gtiff_creation_options = list(gtiff_creation_options)
    # PIXELTYPE is sometimes used to define signed vs. unsigned bytes and
    # the only place that is stored is in the IMAGE_STRUCTURE metadata
    # copy it over if it exists; get this info from the first band since
    # all bands have the same datatype
    base_band = base_raster.GetRasterBand(1)
    metadata = base_band.GetMetadata('IMAGE_STRUCTURE')
    if 'PIXELTYPE' in metadata:
        local_gtiff_creation_options.append(
            'PIXELTYPE=' + metadata['PIXELTYPE'])

    # make directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(target_raster_path))
    except OSError:
        pass
    gdal_driver = gdal.GetDriverByName('GTiff')
    target_raster = gdal_driver.Create(
        target_raster_path, target_x_size, target_y_size,
        base_raster.RasterCount, base_band.DataType,
        options=local_gtiff_creation_options)
    base_band = None

    for index in xrange(target_raster.RasterCount):
        base_nodata = base_raster.GetRasterBand(1+index).GetNoDataValue()
        if base_nodata is not None:
            target_band = target_raster.GetRasterBand(1+index)
            target_band.SetNoDataValue(base_nodata)
            target_band = None

    # Set the geotransform
    target_raster.SetGeoTransform(target_geotransform)
    if target_sr_wkt is None:
        target_sr_wkt = base_sr.ExportToWkt()
    target_raster.SetProjection(target_sr_wkt)

    # need to make this a closure so we get the current time and we can affect
    # state
    def _reproject_callback(df_complete, psz_message, p_progress_arg):
        """The argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - _reproject_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and
                     _reproject_callback.total_time >= 5.0)):
                LOGGER.info(
                    "ReprojectImage %.1f%% complete %s, psz_message %s",
                    df_complete * 100, p_progress_arg[0], psz_message)
                _reproject_callback.last_time = current_time
                _reproject_callback.total_time += current_time
        except AttributeError:
            _reproject_callback.last_time = time.time()
            _reproject_callback.total_time = 0.0

    # Perform the projection/resampling
    gdal.ReprojectImage(
        base_raster, target_raster, base_sr.ExportToWkt(),
        target_sr_wkt, resample_dict[resample_method], 0, 0,
        _reproject_callback, [target_raster_path])

    target_raster = None
    base_raster = None
    calculate_raster_stats(target_raster_path)


def extract_datasource_table_by_key(datasource_uri, key_field):
    """Return vector attribute table of first layer as dictionary.

    Create a dictionary lookup table of the features in the attribute table
    of the datasource referenced by datasource_uri.

    Args:
        datasource_uri (string): a uri to an OGR datasource
        key_field: a field in datasource_uri that refers to a key value
            for each row such as a polygon id.

    Returns:
        attribute_dictionary (dict): returns a dictionary of the
            form {key_field_0: {field_0: value0, field_1: value1}...}
    """
    # Pull apart the datasource
    datasource = ogr.Open(datasource_uri)
    layer = datasource.GetLayer()
    layer_def = layer.GetLayerDefn()

    # Build up a list of field names for the datasource table
    field_names = []
    for field_id in xrange(layer_def.GetFieldCount()):
        field_def = layer_def.GetFieldDefn(field_id)
        field_names.append(field_def.GetName())

    # Loop through each feature and build up the dictionary representing the
    # attribute table
    attribute_dictionary = {}
    for feature in layer:
        feature_fields = {}
        for field_name in field_names:
            feature_fields[field_name] = feature.GetField(field_name)
        key_value = feature.GetField(key_field)
        attribute_dictionary[key_value] = feature_fields

    layer.ResetReading()
    # Explictly clean up the layers so the files close
    layer = None
    datasource = None
    return attribute_dictionary


def copy_vector(base_vector_path, copy_vector_path):
    """Create a copy of an ESRI Shapefile.

    Args:
        base_vector_path (string): path to ESRI Shapefile that is to  be
            copied
        copy_vector_path (string): output path for the copy of
            `base_vector_path`

    Returns:
        None
    """
    if os.path.isfile(copy_vector_path):
        os.remove(copy_vector_path)

    shape = ogr.Open(base_vector_path)
    drv = ogr.GetDriverByName('ESRI Shapefile')
    drv.CopyDataSource(shape, copy_vector_path)


def rasterize_layer_uri(
        raster_uri, shapefile_uri, burn_values=[], option_list=[]):
    """Rasterize datasource layer.

    Burn the layer from 'shapefile_uri' onto the raster from 'raster_uri'.
    Will burn 'burn_value' onto the raster unless 'field' is not None,
    in which case it will burn the value from shapefiles field.

    Args:
        raster_uri (string): a URI to a gdal dataset
        shapefile_uri (string): a URI to an ogr datasource

    Keyword Args:
        burn_values (list): the equivalent value for burning
            into a polygon.  If empty uses the Z values.
        option_list (list): a Python list of options for the operation.
            Example: ["ATTRIBUTE=NPV", "ALL_TOUCHED=TRUE"]

    Returns:
        None
    """
    dataset = gdal.Open(raster_uri, gdal.GA_Update)
    shapefile = ogr.Open(shapefile_uri)
    layer = shapefile.GetLayer()

    gdal.RasterizeLayer(
        dataset, [1], layer, burn_values=burn_values, options=option_list)

    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None
    shapefile = None


def calculate_disjoint_polygon_set(shapefile_uri):
    """Create a list of sets of polygons that don't overlap.

    Determining the minimal number of those sets is an np-complete problem so
    this is an approximation that builds up sets of maximal subsets.

    Args:
        shapefile_uri (string): a uri to an OGR shapefile to process

    Returns:
        subset_list (list): list of sets of FIDs from shapefile_uri
    """
    shapefile = ogr.Open(shapefile_uri)
    shapefile_layer = shapefile.GetLayer()

    poly_intersect_lookup = {}
    for poly_feat in shapefile_layer:
        poly_wkt = poly_feat.GetGeometryRef().ExportToWkt()
        shapely_polygon = shapely.wkt.loads(poly_wkt)
        poly_fid = poly_feat.GetFID()
        poly_intersect_lookup[poly_fid] = {
            'poly': shapely_polygon,
            'prepared': shapely.prepared.prep(shapely_polygon),
            'intersects': set(),
        }
    shapefile_layer.ResetReading()

    for poly_fid in poly_intersect_lookup:
        for intersect_poly_fid in poly_intersect_lookup:
            polygon = poly_intersect_lookup[poly_fid]['prepared']
            if polygon.intersects(
                    poly_intersect_lookup[intersect_poly_fid]['poly']):
                poly_intersect_lookup[poly_fid]['intersects'].add(
                    intersect_poly_fid)

    # Build maximal subsets
    subset_list = []
    while len(poly_intersect_lookup) > 0:
        # sort polygons by increasing number of intersections
        heap = []
        for poly_fid, poly_dict in poly_intersect_lookup.iteritems():
            heapq.heappush(
                heap, (len(poly_dict['intersects']), poly_fid, poly_dict))

        # build maximal subset
        maximal_set = set()
        while len(heap) > 0:
            _, poly_fid, poly_dict = heapq.heappop(heap)
            for maxset_fid in maximal_set:
                if maxset_fid in poly_intersect_lookup[poly_fid]['intersects']:
                    # it intersects and can't be part of the maximal subset
                    break
            else:
                # we made it through without an intersection, add poly_fid to
                # the maximal set
                maximal_set.add(poly_fid)
                # remove that polygon and update the intersections
                del poly_intersect_lookup[poly_fid]
        # remove all the polygons from intersections once they're compuated
        for maxset_fid in maximal_set:
            for poly_dict in poly_intersect_lookup.itervalues():
                poly_dict['intersects'].discard(maxset_fid)
        subset_list.append(maximal_set)
    return subset_list


def distance_transform_edt(
        input_mask_uri, output_distance_uri, process_pool=None):
    """Find the Euclidean distance transform on input_mask_uri and output
    the result as raster.

    Args:
        input_mask_uri (string): a gdal raster to calculate distance from
            the non 0 value pixels
        output_distance_uri (string): will make a float raster w/ same
            dimensions and projection as input_mask_uri where all zero values
            of input_mask_uri are equal to the euclidean distance to the
            closest non-zero pixel.

    Keyword Args:
        process_pool: (description)

    Returns:
        None
    """
    mask_as_byte_uri = temporary_filename(suffix='.tif')
    raster_info = get_raster_info(input_mask_uri)
    nodata = raster_info['nodata']
    out_pixel_size = raster_info['mean_pixel_size']
    nodata_out = 255

    def to_byte(input_vector):
        """Convert vector to 1, 0, or nodata value to fit in a byte raster."""
        return numpy.where(
            input_vector == nodata, nodata_out, input_vector != 0)

    # 64 seems like a reasonable blocksize
    blocksize = 64
    vectorize_datasets(
        [input_mask_uri], to_byte, mask_as_byte_uri, gdal.GDT_Byte,
        nodata_out, out_pixel_size, "union",
        dataset_to_align_index=0, assert_datasets_projected=False,
        process_pool=process_pool, vectorize_op=False,
        datasets_are_pre_aligned=True,
        dataset_options=[
            'TILED=YES', 'BLOCKXSIZE=%d' % blocksize,
            'BLOCKYSIZE=%d' % blocksize])

    geoprocessing_core.distance_transform_edt(
        mask_as_byte_uri, output_distance_uri)
    try:
        os.remove(mask_as_byte_uri)
    except OSError:
        LOGGER.warn("couldn't remove file %s", mask_as_byte_uri)


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.

    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    This source was taken directly from scipy.signaltools and saves us from
    having to access a protected member in a library that could change in
    future releases:

    https://github.com/scipy/scipy/blob/v0.17.1/scipy/signal/signaltools.py#L211

    Parameters:
        target (int): a positive integer to start to find the next Hamming
            number.

    Returns:
        The next regular number greater than or equal to `target`.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            p2 = 2**((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def convolve_2d_uri(
        signal_path, kernel_path, output_path, output_type=gdal.GDT_Float64):
    """Convolve 2D kernel over 2D signal.

    Convolves the raster in `kernel_path` over `signal_path`.  Nodata values
    are treated as 0.0 during the convolution and masked to nodata for
    the output result where `signal_path` has nodata.

    Parameters:
        signal_path (string): a filepath to a gdal dataset that's the
            source input.
        kernel_path (string): a filepath to a gdal dataset that's the
            source input.
        output_path (string): a filepath to the gdal dataset
            that's the convolution output of signal and kernel
            that is the same size and projection of signal_path. Any nodata
            pixels that align with `signal_path` will be set to nodata.
        output_type (GDAL type): a GDAL raster type to set the output
            raster type to, as well as the type to calculate the convolution
            in.  Defaults to GDT_Float64

    Returns:
        None
    """
    output_nodata = numpy.finfo(numpy.float32).min
    new_raster_from_base_uri(
        signal_path, output_path, 'GTiff', output_nodata, output_type,
        fill_value=0)

    signal_raster_info = get_raster_info(signal_path)
    kernel_raster_info = get_raster_info(kernel_path)
    signal_raster_size = signal_raster_info['raster_size']
    kernel_raster_size = kernel_raster_info['raster_size']

    n_signal_pixels = signal_raster_size[0] * signal_raster_size[1]
    n_kernel_pixels = kernel_raster_size[0] * kernel_raster_size[1]

    # by experimentation i found having the smaller raster to be cached
    # gives the best performance
    if n_signal_pixels < n_kernel_pixels:
        s_path = signal_path
        k_path = kernel_path
        n_cols_signal, n_rows_signal = signal_raster_info['raster_size']
        n_cols_kernel, n_rows_kernel = kernel_raster_info['raster_size']
        s_nodata = signal_raster_info['nodata']
        k_nodata = kernel_raster_info['nodata']
    else:
        k_path = signal_path
        s_path = kernel_path
        n_cols_signal, n_rows_signal = kernel_raster_info['raster_size']
        n_cols_kernel, n_rows_kernel = signal_raster_info['raster_size']
        k_nodata = signal_raster_info['nodata']
        s_nodata = kernel_raster_info['nodata']

    # we need the original signal raster info because we want the output to
    # be clipped and NODATA masked to it
    signal_nodata = signal_raster_info['nodata']
    signal_ds = gdal.Open(signal_path)
    signal_band = signal_ds.GetRasterBand(1)
    output_ds = gdal.Open(output_path, gdal.GA_Update)
    output_band = output_ds.GetRasterBand(1)

    def _fft_cache(fshape, xoff, yoff, data_block):
        """Helper function to remember the last computed fft.

        Parameters:
            fshape (numpy.ndarray): shape of fft
            xoff,yoff (int): offsets of the data block
            data_block (numpy.ndarray): the 2D array to calculate the FFT
                on if not already calculated.

        Returns:
            fft transformed data_block of fshape size.
        """
        cache_key = (fshape[0], fshape[1], xoff, yoff)
        if cache_key != _fft_cache.key:
            _fft_cache.cache = numpy.fft.rfftn(data_block, fshape)
            _fft_cache.key = cache_key
        return _fft_cache.cache

    _fft_cache.cache = None
    _fft_cache.key = None

    LOGGER.info('starting convolve')
    last_time = time.time()
    signal_data = {}
    for signal_data, signal_block in iterblocks(
            s_path, astype=_GDAL_TYPE_TO_NUMPY_LOOKUP[output_type]):
        last_time = _invoke_timed_callback(
            last_time, lambda: LOGGER.info(
                "convolution operating on signal pixel (%d, %d)",
                signal_data['xoff'], signal_data['yoff']),
            _LOGGING_PERIOD)

        signal_nodata_mask = signal_block == s_nodata
        signal_block[signal_nodata_mask] = 0.0

        for kernel_data, kernel_block in iterblocks(
                k_path, astype=_GDAL_TYPE_TO_NUMPY_LOOKUP[output_type]):
            left_index_raster = (
                signal_data['xoff'] - n_cols_kernel / 2 + kernel_data['xoff'])
            right_index_raster = (
                signal_data['xoff'] - n_cols_kernel / 2 +
                kernel_data['xoff'] + signal_data['win_xsize'] +
                kernel_data['win_xsize'] - 1)
            top_index_raster = (
                signal_data['yoff'] - n_rows_kernel / 2 + kernel_data['yoff'])
            bottom_index_raster = (
                signal_data['yoff'] - n_rows_kernel / 2 +
                kernel_data['yoff'] + signal_data['win_ysize'] +
                kernel_data['win_ysize'] - 1)

            # it's possible that the piece of the integrating kernel
            # doesn't even affect the final result, we can just skip
            if (right_index_raster < 0 or
                    bottom_index_raster < 0 or
                    left_index_raster > n_cols_signal or
                    top_index_raster > n_rows_signal):
                continue

            kernel_nodata_mask = (kernel_block == k_nodata)
            kernel_block[kernel_nodata_mask] = 0.0

            # determine the output convolve shape
            shape = (
                numpy.array(signal_block.shape) +
                numpy.array(kernel_block.shape) - 1)

            # add zero padding so FFT is fast
            fshape = [_next_regular(int(d)) for d in shape]

            kernel_fft = numpy.fft.rfftn(kernel_block, fshape)
            signal_fft = _fft_cache(
                fshape, signal_data['xoff'], signal_data['yoff'],
                signal_block)

            # this variable determines the output slice that doesn't include
            # the padded array region made for fast FFTs.
            fslice = tuple([slice(0, int(sz)) for sz in shape])
            # classic FFT convolution
            result = numpy.fft.irfftn(signal_fft * kernel_fft, fshape)[fslice]

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

            current_output = output_band.ReadAsArray(**index_dict)
            potential_nodata_signal_array = signal_band.ReadAsArray(
                **index_dict)
            output_array = numpy.empty(
                current_output.shape, dtype=numpy.float32)

            # read the signal block so we know where the nodata are
            valid_mask = potential_nodata_signal_array != signal_nodata
            output_array[:] = output_nodata
            output_array[valid_mask] = (
                (result[top_index_result:bottom_index_result,
                        left_index_result:right_index_result])[valid_mask] +
                current_output[valid_mask])

            output_band.WriteArray(
                output_array, xoff=index_dict['xoff'],
                yoff=index_dict['yoff'])
    output_band.FlushCache()


def iterblocks(
        raster_uri, band_list=None, largest_block=2**14, astype=None,
        offset_only=False):
    """Iterate across all the memory blocks in the input raster.

    Result is a generator of block location information and numpy arrays.

    This is especially useful when a single value needs to be derived from the
    pixel values in a raster, such as the sum total of all pixel values, or
    a sequence of unique raster values.  In such cases, `raster_local_op`
    is overkill, since it writes out a raster.

    As a generator, this can be combined multiple times with itertools.izip()
    to iterate 'simultaneously' over multiple rasters, though the user should
    be careful to do so only with prealigned rasters.

    Parameters:
        raster_uri (string): The string filepath to the raster to iterate over.
        band_list=None (list of ints or None): A list of the bands for which
            the matrices should be returned. The band number to operate on.
            Defaults to None, which will return all bands.  Bands may be
            specified in any order, and band indexes may be specified multiple
            times.  The blocks returned on each iteration will be in the order
            specified in this list.
        largest_block (int): Attempts to iterate over raster blocks with
            this many elements.  Useful in cases where the blocksize is
            relatively small, memory is available, and the function call
            overhead dominates the iteration.  Defaults to 2**20.  A value of
            anything less than the original blocksize of the raster will
            result in blocksizes equal to the original size.
        astype (list of numpy types): If none, output blocks are in the native
            type of the raster bands.  Otherwise this parameter is a list
            of len(band_list) length that contains the desired output types
            that iterblock generates for each band.
        offset_only (boolean): defaults to False, if True `iterblocks` only
            returns offset dictionary and doesn't read any binary data from
            the raster.  This can be useful when iterating over writing to
            an output.

    Returns:
        If `offset_only` is false, on each iteration, a tuple containing a dict
        of block data and `n` 2-dimensional numpy arrays are returned, where
        `n` is the number of bands requested via `band_list`. The dict of
        block data has these attributes:

            data['xoff'] - The X offset of the upper-left-hand corner of the
                block.
            data['yoff'] - The Y offset of the upper-left-hand corner of the
                block.
            data['win_xsize'] - The width of the block.
            data['win_ysize'] - The height of the block.

        If `offset_only` is True, the function returns only the block data and
            does not attempt to read binary data from the raster.
    """
    dataset = gdal.Open(raster_uri)

    if band_list is None:
        band_list = range(1, dataset.RasterCount + 1)

    ds_bands = [dataset.GetRasterBand(index) for index in band_list]

    block = ds_bands[0].GetBlockSize()
    cols_per_block = block[0]
    rows_per_block = block[1]

    n_cols = dataset.RasterXSize
    n_rows = dataset.RasterYSize

    block_area = cols_per_block * rows_per_block
    # try to make block wider
    if largest_block / block_area > 0:
        width_factor = largest_block / block_area
        cols_per_block *= width_factor
        if cols_per_block > n_cols:
            cols_per_block = n_cols
        block_area = cols_per_block * rows_per_block
    # try to make block taller
    if largest_block / block_area > 0:
        height_factor = largest_block / block_area
        rows_per_block *= height_factor
        if rows_per_block > n_rows:
            rows_per_block = n_rows

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    # Initialize to None so a block array is created on the first iteration
    last_row_block_width = None
    last_col_block_width = None

    if astype is not None:
        block_type_list = [astype] * len(ds_bands)
    else:
        block_type_list = [
            _gdal_to_numpy_type(ds_band) for ds_band in ds_bands]

    def _block_gen(queue):
        """Load the next memory block via generator paradigm.

        Parameters:
            queue (Queue.Queue): thread safe queue to return offset_dict and
                results

        Returns:
            None
        """
        for row_block_index in xrange(n_row_blocks):
            row_offset = row_block_index * rows_per_block
            row_block_width = n_rows - row_offset
            if row_block_width > rows_per_block:
                row_block_width = rows_per_block

            for col_block_index in xrange(n_col_blocks):
                col_offset = col_block_index * cols_per_block
                col_block_width = n_cols - col_offset
                if col_block_width > cols_per_block:
                    col_block_width = cols_per_block

                # resize the dataset block cache if necessary
                if (last_row_block_width != row_block_width or
                        last_col_block_width != col_block_width):
                    dataset_blocks = [
                        numpy.zeros(
                            (row_block_width, col_block_width),
                            dtype=block_type) for block_type in
                        block_type_list]

                offset_dict = {
                    'xoff': col_offset,
                    'yoff': row_offset,
                    'win_xsize': col_block_width,
                    'win_ysize': row_block_width,
                }
                result = offset_dict
                if not offset_only:
                    for ds_band, block in zip(ds_bands, dataset_blocks):
                        ds_band.ReadAsArray(buf_obj=block, **offset_dict)
                    result = (result,) + tuple(dataset_blocks)
                queue.put(result)
        queue.put('STOP')  # sentinel indicating end of iteration

    # Make the queue only one element deep so it attempts to load the next
    # block while waiting for the next .next() call.
    block_queue = Queue.Queue(1)
    threading.Thread(target=_block_gen, args=(block_queue,)).start()
    for result in iter(block_queue.get, 'STOP'):
        yield result


def transform_bounding_box(
        bounding_box, base_ref_wkt, new_ref_wkt, edge_samples=11):
    """Transform input bounding box to output projection.

    This transform accounts for the fact that the reprojected square bounding
    box might be warped in the new coordinate system.  To account for this,
    the function samples points along the original bounding box edges and
    attempts to make the largest bounding box around any transformed point
    on the edge whether corners or warped edges.

    Parameters:
        bounding_box (list): a list of 4 coordinates in `base_epsg` coordinate
            system describing the bound in the order [xmin, ymin, xmax, ymax]
        base_ref_wkt (string): the spatial reference of the input coordinate
            system in Well Known Text.
        new_ref_wkt (string): the EPSG code of the desired output coordinate
            system in Well Known Text.
        edge_samples (int): the number of interpolated points along each
            bounding box edge to sample along. A value of 2 will sample just
            the corners while a value of 3 will also sample the corners and
            the midpoint.

    Returns:
        A list of the form [xmin, ymin, xmax, ymax] that describes the largest
        fitting bounding box around the original warped bounding box in
        `new_epsg` coordinate system.
    """
    base_ref = osr.SpatialReference()
    base_ref.ImportFromWkt(base_ref_wkt)

    new_ref = osr.SpatialReference()
    new_ref.ImportFromWkt(new_ref_wkt)

    transformer = osr.CoordinateTransformation(base_ref, new_ref)

    def _transform_point(point):
        """Transform an (x,y) point tuple from base_ref to new_ref."""
        trans_x, trans_y, _ = (transformer.TransformPoint(*point))
        return (trans_x, trans_y)

    # The following list comprehension iterates over each edge of the bounding
    # box, divides each edge into `edge_samples` number of points, then
    # reduces that list to an appropriate `bounding_fn` given the edge.
    # For example the left edge needs to be the minimum x coordinate so
    # we generate `edge_samples` number of points between the upper left and
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
    transformed_bounding_box = [
        bounding_fn(
            [_transform_point(
                p_a * v + p_b * (1 - v)) for v in numpy.linspace(
                    0, 1, edge_samples)])
        for p_a, p_b, bounding_fn in [
            (p_0, p_1, lambda p_list: min([p[0] for p in p_list])),
            (p_1, p_2, lambda p_list: min([p[1] for p in p_list])),
            (p_2, p_3, lambda p_list: max([p[0] for p in p_list])),
            (p_3, p_0, lambda p_list: max([p[1] for p in p_list]))]]
    return transformed_bounding_box


def _invoke_timed_callback(
        reference_time, callback_lambda, callback_period):
    """Invoke callback if a certain amount of time has passed.

    This is a convenience function to standardize update callbacks from the
    module.

    Parameters:
        reference_time (float): time to base `callback_period` length from.
        callback_lambda (lambda): function to invoke if difference between
            current time and `reference_time` has exceeded `callback_period`.
        callback_period (float): time in seconds to pass until
            `callback_lambda` is invoked.

    Return:
        `reference_time` if `callback_lambda` not invoked, otherwise the time
        when `callback_lambda` was invoked.
    """
    current_time = time.time()
    if current_time - reference_time > callback_period:
        callback_lambda()
        return current_time
    return reference_time

def _gdal_to_numpy_type(band):
    """Calculate the equivalent numpy datatype from a GDAL raster band type.

    This function doesn't handle complex or unknown types.  If they are
    passed in, this function will rase a ValueERror.

    Parameters:
        band (gdal.Band): GDAL Band

    Returns:
        numpy_datatype (numpy.dtype): equivalent of band.DataType
    """
    if band.DataType in _GDAL_TYPE_TO_NUMPY_LOOKUP:
        return _GDAL_TYPE_TO_NUMPY_LOOKUP[band.DataType]

    # only class not in the lookup is a Byte but double check.
    if band.DataType != gdal.GDT_Byte:
        raise ValueError("Unsupported DataType: %s" % str(band.DataType))

    metadata = band.GetMetadata('IMAGE_STRUCTURE')
    if 'PIXELTYPE' in metadata and metadata['PIXELTYPE'] == 'SIGNEDBYTE':
        return numpy.int8
    return numpy.uint8


def _merge_bounding_boxes(bb1, bb2, mode):
    """Merge two bounding boxes through union or intersection.

    Parameters:
        bb1, bb2 (list): list of float representing bounding box in the
            form bb=[minx,miny,maxx,maxy]
        mode (string); one of 'union' or 'intersection'

    Returns:
        Reduced bounding box of bb1/bb2 depending on mode.
    """
    def _less_than_or_equal(x_val, y_val):
        return x_val if x_val <= y_val else y_val

    def _greater_than(x_val, y_val):
        return x_val if x_val > y_val else y_val

    if mode == "union":
        comparison_ops = [
            _less_than_or_equal, _greater_than, _greater_than,
            _less_than_or_equal]
    if mode == "intersection":
        comparison_ops = [
            _greater_than, _less_than_or_equal, _less_than_or_equal,
            _greater_than]

    bb_out = [op(x, y) for op, x, y in zip(comparison_ops, bb1, bb2)]
    return bb_out
