"""A collection of GDAL dataset and raster utilities"""

import logging
import os
import tempfile
import shutil
import atexit
import functools
import csv
import math
import errno
import collections
import exceptions
import heapq
import time
from types import StringType

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import numpy
import numpy.ma
import scipy.interpolate
import scipy.sparse
import scipy.signal
import scipy.ndimage
import shapely.wkt
import shapely.ops
from shapely import speedups
import shapely.prepared

import pygeoprocessing.geoprocessing_core
from pygeoprocessing import fileio


def _gdal_to_numpy_type(band):
    """Calculates the equivalent numpy datatype from a GDAL raster band type

        band - GDAL band

        returns numpy equivalent of band.DataType"""

    gdal_type_to_numpy_lookup = {
        gdal.GDT_Int16: numpy.int16,
        gdal.GDT_Int32: numpy.int32,
        gdal.GDT_UInt16: numpy.uint16,
        gdal.GDT_UInt32: numpy.uint32,
        gdal.GDT_Float32: numpy.float32,
        gdal.GDT_Float64: numpy.float64
    }

    if band.DataType in gdal_type_to_numpy_lookup:
        return gdal_type_to_numpy_lookup[band.DataType]

    #only class not in the lookup is a Byte but double check.
    if band.DataType != gdal.GDT_Byte:
        raise ValueError("Unknown DataType: %s" % str(band.DataType))

    metadata = band.GetMetadata('IMAGE_STRUCTURE')
    if 'PIXELTYPE' in metadata and metadata['PIXELTYPE'] == 'SIGNEDBYTE':
        return numpy.int8
    return numpy.uint8


#Used to raise an exception if rasters, shapefiles, or both don't overlap
#in regions that should
class SpatialExtentOverlapException(Exception):
    """An exeception class for cases when datasets or datasources don't overlap
        in space"""
    pass

logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('pygeoprocessing.geoprocessing')


def get_nodata_from_uri(dataset_uri):
    """
    Returns the nodata value for the first band from a gdal dataset cast to its
        correct numpy type.

    Args:
        dataset_uri (string): a uri to a gdal dataset

    Returns:
        nodata_cast (?): nodata value for dataset band 1

    """

    dataset = gdal.Open(dataset_uri)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        nodata = _gdal_to_numpy_type(band)(nodata)
    else:
        LOGGER.warn(
            "Warning the nodata value in %s is not set", dataset_uri)

    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None
    return nodata


def get_datatype_from_uri(dataset_uri):
    """
    Returns the datatype for the first band from a gdal dataset

    Args:
        dataset_uri (string): a uri to a gdal dataset

    Returns:
        datatype (?): datatype for dataset band 1"""

    dataset = gdal.Open(dataset_uri)
    band = dataset.GetRasterBand(1)
    datatype = band.DataType

    #Make sure the dataset is closed and cleaned up
    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return datatype


def get_row_col_from_uri(dataset_uri):
    """
    Returns a tuple of number of rows and columns of that dataset uri.

    Args:
        dataset_uri (string): a uri to a gdal dataset

    Returns:
        tuple (tuple): 2-tuple (n_row, n_col) from dataset_uri"""

    dataset = gdal.Open(dataset_uri)
    n_rows = dataset.RasterYSize
    n_cols = dataset.RasterXSize

    #Make sure the dataset is closed and cleaned up
    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return (n_rows, n_cols)


def calculate_raster_stats_uri(dataset_uri):
    """
    Calculates and sets the min, max, stdev, and mean for the bands in
    the raster.

    Args:
        dataset_uri (string): a uri to a GDAL raster dataset that will be
            modified by having its band statistics set

    Returns:
        nothing
    """

    dataset = gdal.Open(dataset_uri, gdal.GA_Update)

    for band_number in range(dataset.RasterCount):
        band = dataset.GetRasterBand(band_number + 1)
        band.ComputeStatistics(0)

    #Make sure the dataset is closed and cleaned up
    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None


def get_statistics_from_uri(dataset_uri):
    """
    Retrieves the min, max, mean, stdev from a GDAL Dataset

    Args:
        dataset_uri (string): a uri to a gdal dataset

    Returns:
        statistics (?): min, max, mean, stddev

    """

    dataset = gdal.Open(dataset_uri)
    band = dataset.GetRasterBand(1)
    statistics = band.GetStatistics(0, 1)

    #Make sure the dataset is closed and cleaned up
    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return statistics


def get_cell_size_from_uri(dataset_uri):
    """
    Returns the cell size of the dataset in meters.  Raises an exception if the
    raster is not square since this'll break most of the raster_utils
    algorithms.

    Args:
        dataset_uri (string): uri to a gdal dataset

    Returns:
        size_meters (?): cell size of the dataset in meters"""

    srs = osr.SpatialReference()
    dataset = gdal.Open(dataset_uri)
    if dataset == None:
        raise IOError(
            'File not found or not valid dataset type at: %s' % dataset_uri)
    srs.SetProjection(dataset.GetProjection())
    linear_units = srs.GetLinearUnits()
    geotransform = dataset.GetGeoTransform()
    #take absolute value since sometimes negative widths/heights
    try:
        numpy.testing.assert_approx_equal(
            abs(geotransform[1]), abs(geotransform[5]))
        size_meters = abs(geotransform[1]) * linear_units
    except AssertionError as e:
        LOGGER.warn(e)
        size_meters = (
            abs(geotransform[1]) + abs(geotransform[5])) / 2.0 * linear_units

    #Make sure the dataset is closed and cleaned up
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return size_meters


def pixel_size_based_on_coordinate_transform_uri(dataset_uri, *args, **kwargs):
    """
    A wrapper for pixel_size_based_on_coordinate_transform that takes a dataset
    uri as an input and opens it before sending it along

    Args:
        dataset_uri (string): a URI to a gdal dataset

        All other parameters pass along

    Returns:
        result (tuple): a tuple containing (pixel width in meters, pixel height
            in meters)

    """
    dataset = gdal.Open(dataset_uri)
    result = pixel_size_based_on_coordinate_transform(dataset, *args, **kwargs)

    #Make sure the dataset is closed and cleaned up
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return result


def pixel_size_based_on_coordinate_transform(dataset, coord_trans, point):
    """
    Calculates the pixel width and height in meters given a coordinate
    transform and reference point on the dataset that's close to the
    transform's projected coordinate sytem.  This is only necessary
    if dataset is not already in a meter coordinate system, for example
    dataset may be in lat/long (WGS84).

    Args:
        dataset (?): a projected GDAL dataset in the form of lat/long decimal
            degrees
        coord_trans (?): an OSR coordinate transformation from dataset
            coordinate system to meters
        point (?): a reference point close to the coordinate transform
            coordinate system.  must be in the same coordinate system as
            dataset.

    Returns:
        pixel_diff (tuple): a 2-tuple containing (pixel width in meters, pixel
            height in meters)
    """
    #Get the first points (x, y) from geoTransform
    geo_tran = dataset.GetGeoTransform()
    pixel_size_x = geo_tran[1]
    pixel_size_y = geo_tran[5]
    top_left_x = point[0]
    top_left_y = point[1]
    #Create the second point by adding the pixel width/height
    new_x = top_left_x + pixel_size_x
    new_y = top_left_y + pixel_size_y
    #Transform two points into meters
    point_1 = coord_trans.TransformPoint(top_left_x, top_left_y)
    point_2 = coord_trans.TransformPoint(new_x, new_y)
    #Calculate the x/y difference between two points
    #taking the absolue value because the direction doesn't matter for pixel
    #size in the case of most coordinate systems where y increases up and x
    #increases to the right (right handed coordinate system).
    pixel_diff_x = abs(point_2[0] - point_1[0])
    pixel_diff_y = abs(point_2[1] - point_1[1])
    return (pixel_diff_x, pixel_diff_y)


def new_raster_from_base_uri(
        base_uri, output_uri, gdal_format, nodata, datatype, fill_value=None,
        n_rows=None, n_cols=None, dataset_options=None):
    """
    A wrapper for the function new_raster_from_base that opens up
    the base_uri before passing it to new_raster_from_base.

    Args:
        base_uri (string): a URI to a GDAL dataset on disk.

        output_uri (string): a string URI to the new output raster dataset.
        gdal_format (string): a string representing the GDAL file format of the
            output raster.  See http://gdal.org/formats_list.html for a list
            of available formats.  This parameter expects the format code, such
            as 'GTiff' or 'MEM'
        nodata (?): a value that will be set as the nodata value for the
            output raster.  Should be the same type as 'datatype'
        datatype (?): the pixel datatype of the output raster, for example
            gdal.GDT_Float32.  See the following header file for supported
            pixel types:
            http://www.gdal.org/gdal_8h.html#22e22ce0a55036a96f652765793fb7a4

    Keyword Args:
        fill_value (?): the value to fill in the raster on creation
        n_rows (?): if set makes the resulting raster have n_rows in it
            if not, the number of rows of the outgoing dataset are equal to
            the base.
        n_cols (?): similar to n_rows, but for the columns.
        dataset_options (?): a list of dataset options that gets
            passed to the gdal creation driver, overrides defaults

    Returns:
        nothing

    """

    pygeoprocessing.geoprocessing_core.new_raster_from_base_uri(
        base_uri,output_uri, gdal_format, nodata, datatype,
        fill_value=fill_value, n_rows=n_rows, n_cols=n_rows,
        dataset_options=dataset_options)


def new_raster_from_base(
        base, output_uri, gdal_format, nodata, datatype, fill_value=None,
        n_rows=None, n_cols=None, dataset_options=None):

    """
    Create a new, empty GDAL raster dataset with the spatial references,
    geotranforms of the base GDAL raster dataset.

    Args:
        base (?): a the GDAL raster dataset to base output size, and transforms
            on
        output_uri (string): a string URI to the new output raster dataset.
        gdal_format (string): a string representing the GDAL file format of the
            output raster.  See http://gdal.org/formats_list.html for a list
            of available formats.  This parameter expects the format code, such
            as 'GTiff' or 'MEM'
        nodata (?): a value that will be set as the nodata value for the
            output raster.  Should be the same type as 'datatype'
        datatype (?): the pixel datatype of the output raster, for example
            gdal.GDT_Float32.  See the following header file for supported
            pixel types:
            http://www.gdal.org/gdal_8h.html#22e22ce0a55036a96f652765793fb7a4

    Keyword Args:
        fill_value (?): the value to fill in the raster on creation
        n_rows (?): if set makes the resulting raster have n_rows in it
            if not, the number of rows of the outgoing dataset are equal to
            the base.
        n_cols (?): similar to n_rows, but for the columns.
        dataset_options (?): a list of dataset options that gets
            passed to the gdal creation driver, overrides defaults

    Returns:
        dataset (?): a new GDAL raster dataset.

    """

    return pygeoprocessing.geoprocessing_core.new_raster_from_base(
        base, output_uri, gdal_format, nodata, datatype, fill_value,
        n_rows, n_cols, dataset_options)


def new_raster(
        cols, rows, projection, geotransform, format, nodata, datatype,
        bands, outputURI):

    """
    Create a new raster with the given properties.

    Args:
        cols (int): number of pixel columns
        rows (int): number of pixel rows
        projection (?): the datum
        geotransform (?): the coordinate system
        format (string): a string representing the GDAL file format of the
            output raster.  See http://gdal.org/formats_list.html for a list
            of available formats.  This parameter expects the format code, such
            as 'GTiff' or 'MEM'
        nodata (?): a value that will be set as the nodata value for the
            output raster.  Should be the same type as 'datatype'
        datatype (?): the pixel datatype of the output raster, for example
            gdal.GDT_Float32.  See the following header file for supported
            pixel types:
            http://www.gdal.org/gdal_8h.html#22e22ce0a55036a96f652765793fb7a4
        bands (int): the number of bands in the raster
        outputURI (string): the file location for the outputed raster.  If
            format is 'MEM' this can be an empty string

    Returns:
        dataset (?): a new GDAL raster with the parameters as described above

    """

    driver = gdal.GetDriverByName(format)
    new_raster = driver.Create(
        outputURI.encode('utf-8'), cols, rows, bands, datatype,
        options=['BIGTIFF=IF_SAFER'])
    new_raster.SetProjection(projection)
    new_raster.SetGeoTransform(geotransform)
    for i in range(bands):
        new_raster.GetRasterBand(i + 1).SetNoDataValue(nodata)
        new_raster.GetRasterBand(i + 1).Fill(nodata)

    return new_raster


def calculate_intersection_rectangle(dataset_list, aoi=None):
    """
    Return a bounding box of the intersections of all the rasters in the
    list.

    Args:
        dataset_list (list): a list of GDAL datasets in the same projection and
            coordinate system

    Keyword Args:
        aoi (?): an OGR polygon datasource which may optionally also restrict
            the extents of the intersection rectangle based on its own
            extents.

    Returns:
        bounding_box (list): a 4 element list that bounds the intersection of
            all the rasters in dataset_list.  [left, top, right, bottom]

    Raises:
        SpatialExtentOverlapException: in cases where the dataset list and aoi
            don't overlap.

    """

    def valid_bounding_box(bb):
        """
        Check to make sure bounding box doesn't collapse on itself

        Args:
            bb (list): a bounding box of the form [left, top, right, bottom]

        Returns:
            is_valid (boolean): True if bb is valid, false otherwise"""

        return bb[0] <= bb[2] and bb[3] <= bb[1]

    #Define the initial bounding box
    gt = dataset_list[0].GetGeoTransform()
    #order is left, top, right, bottom of rasterbounds
    bounding_box = [
        gt[0], gt[3], gt[0] + gt[1] * dataset_list[0].RasterXSize,
        gt[3] + gt[5] * dataset_list[0].RasterYSize]

    dataset_files = []
    for dataset in dataset_list:
        dataset_files.append(dataset.GetDescription())
        #intersect the current bounding box with the one just read
        gt = dataset.GetGeoTransform()
        rec = [
            gt[0], gt[3], gt[0] + gt[1] * dataset.RasterXSize,
            gt[3] + gt[5] * dataset.RasterYSize]
        #This intersects rec with the current bounding box
        bounding_box = [
            max(rec[0], bounding_box[0]),
            min(rec[1], bounding_box[1]),
            min(rec[2], bounding_box[2]),
            max(rec[3], bounding_box[3])]

        #Left can't be greater than right or bottom greater than top
        if not valid_bounding_box(bounding_box):
            raise SpatialExtentOverlapException(
                "These rasters %s don't overlap with this one %s" %
                (unicode(dataset_files[0:-1]), dataset_files[-1]))

    if aoi != None:
        aoi_layer = aoi.GetLayer(0)
        aoi_extent = aoi_layer.GetExtent()
        bounding_box = [
            max(aoi_extent[0], bounding_box[0]),
            min(aoi_extent[3], bounding_box[1]),
            min(aoi_extent[1], bounding_box[2]),
            max(aoi_extent[2], bounding_box[3])]
        if not valid_bounding_box(bounding_box):
            raise SpatialExtentOverlapException(
                "The aoi layer %s doesn't overlap with %s" %
                (aoi, unicode(dataset_files)))

    return bounding_box


def create_raster_from_vector_extents_uri(
        shapefile_uri, pixel_size, gdal_format, nodata_out_value, output_uri):
    """
    A wrapper for create_raster_from_vector_extents

    Args:
        shapefile_uri (string): uri to an OGR datasource to use as the extents
            of the raster
        pixel_size (?): size of output pixels in the projected units of
            shapefile_uri
        gdal_format (?): the raster pixel format, something like
            gdal.GDT_Float32
        nodata_out_value (?): the output nodata value
        output_uri (string): the URI to write the gdal dataset

    Returns:


    """

    datasource = ogr.Open(shapefile_uri)
    create_raster_from_vector_extents(
        pixel_size, pixel_size, gdal_format, nodata_out_value, output_uri,
        datasource)


def create_raster_from_vector_extents(
        xRes, yRes, format, nodata, rasterFile, shp):
    """
    Create a blank raster based on a vector file extent.  This code is
    adapted from http://trac.osgeo.org/gdal/wiki/FAQRaster#HowcanIcreateablankrasterbasedonavectorfilesextentsforusewithgdal_rasterizeGDAL1.8.0

    Args:
        xRes (?): the x size of a pixel in the output dataset must be a
            positive value
        yRes (?): the y size of a pixel in the output dataset must be a
            positive value
        format (?): gdal GDT pixel type
        nodata (?): the output nodata value
        rasterFile (string): URI to file location for raster
        shp (?): vector shapefile to base extent of output raster on

    Returns:
        raster (?): blank raster whose bounds fit within `shp`s bounding box
            and features are equivalent to the passed in data

    """

    #Determine the width and height of the tiff in pixels based on the
    #maximum size of the combined envelope of all the features
    shp_extent = None
    for layer_index in range(shp.GetLayerCount()):
        shp_layer = shp.GetLayer(layer_index)
        for feature_index in range(shp_layer.GetFeatureCount()):
            try:
                feature = shp_layer.GetFeature(feature_index)
                geometry = feature.GetGeometryRef()

                #feature_extent = [xmin, xmax, ymin, ymax]
                feature_extent = geometry.GetEnvelope()
                #This is an array based way of mapping the right function
                #to the right index.
                functions = [min, max, min, max]
                for i in range(len(functions)):
                    try:
                        shp_extent[i] = functions[i](
                            shp_extent[i], feature_extent[i])
                    except TypeError:
                        #need to cast to list because returned as a tuple
                        #and we can't assign to a tuple's index, also need to
                        #define this as the initial state
                        shp_extent = list(feature_extent)
            except AttributeError as e:
                #For some valid OGR objects the geometry can be undefined since
                #it's valid to have a NULL entry in the attribute table
                #this is expressed as a None value in the geometry reference
                #this feature won't contribute
                LOGGER.warn(e)

    #shp_extent = [xmin, xmax, ymin, ymax]
    tiff_width = int(numpy.ceil(abs(shp_extent[1] - shp_extent[0]) / xRes))
    tiff_height = int(numpy.ceil(abs(shp_extent[3] - shp_extent[2]) / yRes))

    if rasterFile != None:
        driver = gdal.GetDriverByName('GTiff')
    else:
        rasterFile = ''
        driver = gdal.GetDriverByName('MEM')
    #1 means only create 1 band
    raster = driver.Create(
        rasterFile, tiff_width, tiff_height, 1, format,
        options=['BIGTIFF=IF_SAFER'])
    raster.GetRasterBand(1).SetNoDataValue(nodata)

    #Set the transform based on the upper left corner and given pixel
    #dimensions
    raster_transform = [shp_extent[0], xRes, 0.0, shp_extent[3], 0.0, -yRes]
    raster.SetGeoTransform(raster_transform)

    #Use the same projection on the raster as the shapefile
    srs = osr.SpatialReference()
    srs.ImportFromWkt(shp.GetLayer(0).GetSpatialRef().__str__())
    raster.SetProjection(srs.ExportToWkt())

    #Initialize everything to nodata
    raster.GetRasterBand(1).Fill(nodata)
    raster.GetRasterBand(1).FlushCache()

    return raster


def vectorize_points(
        shapefile, datasource_field, dataset, randomize_points=False,
        mask_convex_hull=False, interpolation='nearest'):
    """
    Takes a shapefile of points and a field defined in that shapefile
    and interpolates the values in the points onto the given raster

    Args:
        shapefile (?): ogr datasource of points
        datasource_field (?): a field in shapefile
        dataset (?): a gdal dataset must be in the same projection as shapefile

    Keyword Args:
        randomize_points (boolean): (description)
        mask_convex_hull (boolean): (description)
        interpolation (string): the interpolation method to use for
            scipy.interpolate.griddata(). Default is 'nearest'

    Returns:
       nothing

    """

    #Define the initial bounding box
    gt = dataset.GetGeoTransform()
    #order is left, top, right, bottom of rasterbounds
    bounding_box = [gt[0], gt[3], gt[0] + gt[1] * dataset.RasterXSize,
                    gt[3] + gt[5] * dataset.RasterYSize]

    def in_bounds(point):
        return point[0] <= bounding_box[2] and point[0] >= bounding_box[0] \
            and point[1] <= bounding_box[1] and point[1] >= bounding_box[3]

    layer = shapefile.GetLayer(0)
    point_list = []
    value_list = []

    #Calculate a small amount to perturb points by so that we don't
    #get a linear Delauney triangle, the 1e-6 is larger than eps for
    #floating point, but large enough not to cause errors in interpolation.
    delta_difference = 1e-6 * min(abs(gt[1]), abs(gt[5]))
    if randomize_points:
        random_array = numpy.random.randn(layer.GetFeatureCount(), 2)
        random_offsets = random_array*delta_difference
    else:
        random_offsets = numpy.zeros((layer.GetFeatureCount(), 2))

    for feature_id in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(feature_id)
        geometry = feature.GetGeometryRef()
        #Here the point geometry is in the form x, y (col, row)
        point = geometry.GetPoint()
        if in_bounds(point):
            value = feature.GetField(datasource_field)
            #Add in the numpy notation which is row, col
            point_list.append([point[1]+random_offsets[feature_id, 1],
                               point[0]+random_offsets[feature_id, 0]])
            value_list.append(value)

    point_array = numpy.array(point_list)
    value_array = numpy.array(value_list)

    band = dataset.GetRasterBand(1)

    #Create grid points for interpolation outputs later
    #top-bottom:y_stepsize, left-right:x_stepsize

    #Make as an integer grid then divide subtract by bounding box parts
    #so we don't get a roundoff error and get off by one pixel one way or
    #the other
    grid_y, grid_x = numpy.mgrid[0:band.YSize, 0:band.XSize]
    grid_y = grid_y * gt[5] + bounding_box[1]
    grid_x = grid_x * gt[1] + bounding_box[0]

    nodata = band.GetNoDataValue()

    raster_out_array = scipy.interpolate.griddata(
        point_array, value_array, (grid_y, grid_x), interpolation, nodata)
    band.WriteArray(raster_out_array, 0, 0)

    #Make sure the dataset is closed and cleaned up
    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None


def aggregate_raster_values_uri(
        raster_uri, shapefile_uri, shapefile_field=None, ignore_nodata=True,
        threshold_amount_lookup=None, ignore_value_list=[], process_pool=None,
        all_touched=False):

    """
    Collect all the raster values that lie in shapefile depending on the
    value of operation

    Args:
        raster_uri (string): a uri to a GDAL dataset
        shapefile_uri (string): a uri to a OGR datasource that should overlap
            raster; raises an exception if not.

    Keyword Args:
        shapefile_field (string): a string indicating which key in shapefile to
            associate the output dictionary values with whose values are
            associated with ints; if None dictionary returns a value over
            the entire shapefile region that intersects the raster.
        ignore_nodata (?): if operation == 'mean' then it does not
            account for nodata pixels when determining the pixel_mean,
            otherwise all pixels in the AOI are used for calculation of the
            mean.  This does not affect hectare_mean which is calculated from
            the geometrical area of the feature.
        threshold_amount_lookup (dict): a dictionary indexing the
            shapefile_field's to threshold amounts to subtract from the
            aggregate value.  The result will be clamped to zero.
        ignore_value_list (list): a list of values to ignore when
            calculating the stats
        process_pool (?): a process pool for multiprocessing
        all_touched (boolean): if true will account for any pixel whose
            geometry passes through the pixel, not just the center point

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

    raster_nodata = get_nodata_from_uri(raster_uri)
    out_pixel_size = get_cell_size_from_uri(raster_uri)
    clipped_raster_uri = temporary_filename(suffix='.tif')
    vectorize_datasets(
        [raster_uri], lambda x: x, clipped_raster_uri, gdal.GDT_Float64,
        raster_nodata, out_pixel_size, "union",
        dataset_to_align_index=0, aoi_uri=shapefile_uri,
        assert_datasets_projected=False, process_pool=process_pool,
        vectorize_op=False)
    clipped_raster = gdal.Open(clipped_raster_uri)

    #This should be a value that's not in shapefile[shapefile_field]
    mask_nodata = -1
    mask_uri = temporary_filename(suffix='.tif')
    new_raster_from_base_uri(
        clipped_raster_uri, mask_uri, 'GTiff', mask_nodata,
        gdal.GDT_Int32, fill_value=mask_nodata)

    mask_dataset = gdal.Open(mask_uri, gdal.GA_Update)
    shapefile = ogr.Open(shapefile_uri)
    shapefile_layer = shapefile.GetLayer()
    rasterize_layer_args = {
        'options': [],
    }

    if all_touched:
        rasterize_layer_args['options'].append('ALL_TOUCHED=TRUE')

    if shapefile_field is not None:
        #Make sure that the layer name refers to an integer
        layer_d = shapefile_layer.GetLayerDefn()
        fd = layer_d.GetFieldDefn(layer_d.GetFieldIndex(shapefile_field))
        if fd == -1 or fd is None:  # -1 returned when field does not exist.
            # Raise exception if user provided a field that's not in vector
            raise AttributeError(
                'Vector %s must have a field named %s' %
                (shapefile_uri, shapefile_field))
        if fd.GetTypeName() != 'Integer':
            raise TypeError(
                'Can only aggreggate by integer based fields, requested '
                'field is of type  %s' % fd.GetTypeName())
        #Adding the rasterize by attribute option
        rasterize_layer_args['options'].append(
            'ATTRIBUTE=%s' % shapefile_field)
    else:
        #9999 is a classic unknown value
        global_id_value = 9999
        rasterize_layer_args['burn_values'] = [global_id_value]

    #loop over the subset of feature layers and rasterize/aggregate each one
    aggregate_dict_values = {}
    aggregate_dict_counts = {}
    AggregatedValues = collections.namedtuple(
        'AggregatedValues',
        'total pixel_mean hectare_mean n_pixels pixel_min pixel_max')
    result_tuple = AggregatedValues(
        total={},
        pixel_mean={},
        hectare_mean={},
        n_pixels={},
        pixel_min={},
        pixel_max={})

    #make a shapefile that non-overlapping layers can be added to
    driver = ogr.GetDriverByName('ESRI Shapefile')
    layer_dir = temporary_folder()
    subset_layer_datasouce = driver.CreateDataSource(
        os.path.join(layer_dir, 'subset_layer.shp'))
    spat_ref = get_spatial_ref_uri(shapefile_uri)
    subset_layer = subset_layer_datasouce.CreateLayer(
        'subset_layer', spat_ref, ogr.wkbPolygon)
    defn = shapefile_layer.GetLayerDefn()

    #For every field, create a duplicate field and add it to the new
    #subset_layer layer
    defn.GetFieldCount()
    for fld_index in range(defn.GetFieldCount()):
        original_field = defn.GetFieldDefn(fld_index)
        output_field = ogr.FieldDefn(
            original_field.GetName(), original_field.GetType())
        subset_layer.CreateField(output_field)

    #Initialize these dictionaries to have the shapefile fields in the original
    #datasource even if we don't pick up a value later

    #This will store the sum/count with index of shapefile attribute
    if shapefile_field is not None:
        shapefile_table = extract_datasource_table_by_key(
            shapefile_uri, shapefile_field)
    else:
        shapefile_table = {global_id_value: 0.0}

    current_iteration_shapefiles = dict([
        (shapefile_id, 0.0) for shapefile_id in shapefile_table.iterkeys()])
    aggregate_dict_values = current_iteration_shapefiles.copy()
    aggregate_dict_counts = current_iteration_shapefiles.copy()
    #Populate the means and totals with something in case the underlying raster
    #doesn't exist for those features.  we use -9999 as a recognizable nodata
    #value.
    for shapefile_id in shapefile_table:
        result_tuple.pixel_mean[shapefile_id] = -9999
        result_tuple.total[shapefile_id] = -9999
        result_tuple.hectare_mean[shapefile_id] = -9999

    pixel_min_dict = dict(
        [(shapefile_id, None) for shapefile_id in shapefile_table.iterkeys()])
    pixel_max_dict = pixel_min_dict.copy()

    #Loop over each polygon and aggregate
    minimal_polygon_sets = calculate_disjoint_polygon_set(
        shapefile_uri)

    clipped_band = clipped_raster.GetRasterBand(1)
    n_rows = clipped_band.YSize
    n_cols = clipped_band.XSize
    block_col_size, block_row_size = clipped_band.GetBlockSize()

    for polygon_set in minimal_polygon_sets:
        #add polygons to subset_layer
        for poly_fid in polygon_set:
            poly_feat = shapefile_layer.GetFeature(poly_fid)
            subset_layer.CreateFeature(poly_feat)
        subset_layer_datasouce.SyncToDisk()

        #nodata out the mask
        mask_band = mask_dataset.GetRasterBand(1)
        mask_band.Fill(mask_nodata)
        mask_band = None

        gdal.RasterizeLayer(
            mask_dataset, [1], subset_layer, **rasterize_layer_args)

        #get feature areas
        num_features = subset_layer.GetFeatureCount()
        feature_areas = collections.defaultdict(int)
        for feature in subset_layer:
            #feature = subset_layer.GetFeature(index)
            geom = feature.GetGeometryRef()
            if shapefile_field is not None:
                feature_id = feature.GetField(shapefile_field)
                feature_areas[feature_id] = geom.GetArea()
            else:
                feature_areas[global_id_value] += geom.GetArea()
        subset_layer.ResetReading()
        geom = None

        #Need a complicated step to see what the FIDs are in the subset_layer
        #then need to loop through and delete them
        fid_to_delete = set()
        for feature in subset_layer:
            fid_to_delete.add(feature.GetFID())
        subset_layer.ResetReading()
        for fid in fid_to_delete:
            subset_layer.DeleteFeature(fid)
        subset_layer_datasouce.SyncToDisk()

        mask_dataset.FlushCache()
        mask_band = mask_dataset.GetRasterBand(1)
        current_iteration_attribute_ids = set()

        for global_block_row in xrange(int(numpy.ceil(float(n_rows) / block_row_size))):
            for global_block_col in xrange(int(numpy.ceil(float(n_cols) / block_col_size))):
                global_col = global_block_col*block_col_size
                global_row = global_block_row*block_row_size
                global_col_size = min((global_block_col+1)*block_col_size, n_cols) - global_col
                global_row_size = min((global_block_row+1)*block_row_size, n_rows) - global_row
                mask_array = mask_band.ReadAsArray(
                    global_col, global_row, global_col_size, global_row_size)
                clipped_array = clipped_band.ReadAsArray(
                    global_col, global_row, global_col_size, global_row_size)

                unique_ids = numpy.unique(mask_array)
                current_iteration_attribute_ids = (
                    current_iteration_attribute_ids.union(unique_ids))
                for attribute_id in unique_ids:
                    #ignore masked values
                    if attribute_id == mask_nodata:
                        continue

                    #Only consider values which lie in the polygon for attribute_id
                    masked_values = clipped_array[
                        (mask_array == attribute_id) &
                        (~numpy.isnan(clipped_array))]
                    #Remove the nodata and ignore values for later processing
                    masked_values_nodata_removed = (
                        masked_values[~numpy.in1d(
                            masked_values, [raster_nodata] + ignore_value_list).
                            reshape(masked_values.shape)])

                    #Find the min and max which might not yet be calculated
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
                        #Only consider values which are not nodata values
                        aggregate_dict_counts[attribute_id] += (
                            masked_values_nodata_removed.size)
                    else:
                        aggregate_dict_counts[attribute_id] += masked_values.size

                    aggregate_dict_values[attribute_id] += numpy.sum(
                        masked_values_nodata_removed)

        #Initialize the dictionary to have an n_pixels field that contains the
        #counts of all the pixels used in the calculation.
        result_tuple.n_pixels.update(aggregate_dict_counts.copy())
        result_tuple.pixel_min.update(pixel_min_dict.copy())
        result_tuple.pixel_max.update(pixel_max_dict.copy())
        #Don't want to calculate stats for the nodata
        current_iteration_attribute_ids.discard(mask_nodata)
        for attribute_id in current_iteration_attribute_ids:
            if threshold_amount_lookup != None:
                adjusted_amount = max(
                    aggregate_dict_values[attribute_id] -
                    threshold_amount_lookup[attribute_id], 0.0)
            else:
                adjusted_amount = aggregate_dict_values[attribute_id]

            result_tuple.total[attribute_id] = adjusted_amount

            if aggregate_dict_counts[attribute_id] != 0.0:
                n_pixels = aggregate_dict_counts[attribute_id]
                result_tuple.pixel_mean[attribute_id] = (
                    adjusted_amount / n_pixels)

                #To get the total area multiply n pixels by their area then
                #divide by 10000 to get Ha.  Notice that's in the denominator
                #so the * 10000 goes on the top
                if feature_areas[attribute_id] == 0:
                    LOGGER.warn('feature_areas[%d]=0' % (attribute_id))
                    result_tuple.hectare_mean[attribute_id] = 0.0
                else:
                    result_tuple.hectare_mean[attribute_id] = (
                        adjusted_amount / feature_areas[attribute_id] * 10000)
            else:
                result_tuple.pixel_mean[attribute_id] = 0.0
                result_tuple.hectare_mean[attribute_id] = 0.0

        try:
            assert_datasets_in_same_projection([raster_uri])
        except DatasetUnprojected:
            #doesn't make sense to calculate the hectare mean
            LOGGER.warn(
                'raster %s is not projected setting hectare_mean to {}'
                % raster_uri)
            result_tuple.hectare_mean.clear()

    #Make sure the dataset is closed and cleaned up
    mask_band = None
    gdal.Dataset.__swig_destroy__(mask_dataset)
    mask_dataset = None

    #Make sure the dataset is closed and cleaned up
    clipped_band = None
    gdal.Dataset.__swig_destroy__(clipped_raster)
    clipped_raster = None

    for filename in [mask_uri, clipped_raster_uri]:
        try:
            os.remove(filename)
        except OSError:
            LOGGER.warn("couldn't remove file %s" % filename)

    subset_layer = None
    ogr.DataSource.__swig_destroy__(subset_layer_datasouce)
    subset_layer_datasouce = None
    try:
        shutil.rmtree(layer_dir)
    except OSError:
        LOGGER.warn("couldn't remove directory %s" % layer_dir)

    return result_tuple


def reclassify_by_dictionary(
        dataset, rules, output_uri, format, nodata, datatype,
        default_value=None):
    """
    Convert all the non-nodata values in dataset to the values mapped to
    by rules.  If there is no rule for an input value it is replaced by
    default_value.  If default_value is None, nodata is used.

    If default_value is None, the default_value will only be used if a pixel
    is not nodata and the pixel does not have a rule defined for its value.
    If the user wishes to have nodata values also mapped to the default
    value, this can be achieved by defining a rule such as:

        rules[dataset_nodata] = default_value

    Doing so will override the default nodata behaviour.

    Args:
        dataset (?): GDAL raster dataset
        rules (dict): a dictionary of the form:
            {'dataset_value1' : 'output_value1', ...
             'dataset_valuen' : 'output_valuen'}
             used to map dataset input types to output
        output_uri (string): the location to hold the output raster on disk
        format (string): either 'MEM' or 'GTiff'
        nodata (?): output raster dataset nodata value
        datatype (?): a GDAL output type

    Keyword Args:
        default_value (?): the value to be used if a reclass rule is not
            defined.

    Returns:
        return the mapped raster as a GDAL dataset"""

    # If a default value is not set, assume that the default value is the
    # used-defined nodata value.
    # If a default value is defined by the user, assume that nodata values
    # should remain nodata.  This check is sensitive to different nodata values
    # between input and output rasters.  This modification is made on a copy of
    # the rules dictionary.
    if default_value == None:
        default_value = nodata
    else:
        rules = rules.copy()
        if nodata not in rules:
            in_nodata = dataset.GetRasterBand(1).GetNoDataValue()
            rules[in_nodata] = nodata

    output_dataset = new_raster_from_base(
        dataset, output_uri, format, nodata, datatype)
    pygeoprocessing.geoprocessing_core.reclassify_by_dictionary(
        dataset, rules, output_uri, format, default_value, datatype,
        output_dataset,)
    calculate_raster_stats_uri(output_uri)
    return output_dataset


def calculate_slope(
        dem_dataset_uri, slope_uri, aoi_uri=None, process_pool=None):
    """
    Generates raster maps of slope.  Follows the algorithm described here:
    http://webhelp.esri.com/arcgiSDEsktop/9.3/index.cfm?TopicName=How%20Slope%20works

    Args:
        dem_dataset_uri (string): a URI to a  single band raster of z values.
        slope_uri (string): a path to the output slope uri in percent.

    Keyword Args:
        aoi_uri (string): a uri to an AOI input
        process_pool (?): a process pool for multiprocessing

    Returns:
        nothing

    """

    out_pixel_size = get_cell_size_from_uri(dem_dataset_uri)
    dem_nodata = get_nodata_from_uri(dem_dataset_uri)

    dem_small_uri = temporary_filename(suffix='.tif')
    #cast the dem to a floating point one if it's not already
    dem_float_nodata = float(dem_nodata)

    vectorize_datasets(
        [dem_dataset_uri], lambda x: x.astype(numpy.float32), dem_small_uri,
        gdal.GDT_Float32, dem_float_nodata, out_pixel_size, "intersection",
        dataset_to_align_index=0, aoi_uri=aoi_uri, process_pool=process_pool,
        vectorize_op=False)

    slope_nodata = -9999.0
    new_raster_from_base_uri(
        dem_small_uri, slope_uri, 'GTiff', slope_nodata, gdal.GDT_Float32)
    pygeoprocessing.geoprocessing_core._cython_calculate_slope(dem_small_uri, slope_uri)
    calculate_raster_stats_uri(slope_uri)

    os.remove(dem_small_uri)


def clip_dataset_uri(
        source_dataset_uri, aoi_datasource_uri, out_dataset_uri,
        assert_projections=True, process_pool=None):
    """
    This function will clip source_dataset to the bounding box of the
    polygons in aoi_datasource and mask out the values in source_dataset
    outside of the AOI with the nodata values in source_dataset.

    Args:
        source_dataset_uri (string): uri to single band GDAL dataset to clip
        aoi_datasource_uri (string): uri to ogr datasource
        out_dataset_uri (string): path to disk for the clipped datset

    Keyword Args:
        assert_projections (boolean): a boolean value for whether the dataset
            needs to be projected
        process_pool (?): a process pool for multiprocessing

    Returns:
        nothing

    """
    source_dataset = gdal.Open(source_dataset_uri)

    band = source_dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    datatype = band.DataType

    if nodata is None:
        nodata = -9999

    gdal.Dataset.__swig_destroy__(source_dataset)
    source_dataset = None

    pixel_size = get_cell_size_from_uri(source_dataset_uri)
    vectorize_datasets(
        [source_dataset_uri], lambda x: x, out_dataset_uri, datatype, nodata,
        pixel_size, 'intersection', aoi_uri=aoi_datasource_uri,
        assert_datasets_projected=assert_projections,
        process_pool=process_pool, vectorize_op=False)


def create_rat_uri(dataset_uri, attr_dict, column_name):
    """
    URI wrapper for create_rat

    Args:
        dataset_uri (string): a GDAL raster dataset to create the RAT for (...)
        attr_dict (dict): a dictionary with keys that point to a primitive type
           {integer_id_1: value_1, ... integer_id_n: value_n}
        column_name (string): a string for the column name that maps the values

    """
    dataset = gdal.Open(dataset_uri, gdal.GA_Update)
    create_rat(dataset, attr_dict, column_name)

    #Make sure the dataset is closed and cleaned up
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None


def create_rat(dataset, attr_dict, column_name):
    """
    Create a raster attribute table

    Args:
        dataset (?): a GDAL raster dataset to create the RAT for (...)
        attr_dict (dict): a dictionary with keys that point to a primitive type
           {integer_id_1: value_1, ... integer_id_n: value_n}
        column_name (string): a string for the column name that maps the values

    Returns:
        dataset (?): a GDAL raster dataset with an updated RAT

    """

    band = dataset.GetRasterBand(1)

    # If there was already a RAT associated with this dataset it will be blown
    # away and replaced by a new one
    LOGGER.warn('Blowing away any current raster attribute table')
    rat = gdal.RasterAttributeTable()

    rat.SetRowCount(len(attr_dict))

    # create columns
    rat.CreateColumn('Value', gdal.GFT_Integer, gdal.GFU_MinMax)
    rat.CreateColumn(column_name, gdal.GFT_String, gdal.GFU_Name)

    row_count = 0
    for key in sorted(attr_dict.keys()):
        rat.SetValueAsInt(row_count, 0, int(key))
        rat.SetValueAsString(row_count, 1, attr_dict[key])
        row_count += 1

    band.SetDefaultRAT(rat)
    return dataset


def get_raster_properties_uri(dataset_uri):
    """
    Wrapper function for get_raster_properties() that passes in the dataset
    URI instead of the datasets itself

    Args:
        dataset_uri (string): a URI to a GDAL raster dataset

    Returns:
        value (dictionary): a dictionary with the properties stored under
            relevant keys. The current list of things returned is:
            width (w-e pixel resolution), height (n-s pixel resolution),
            XSize, YSize
    """
    dataset = gdal.Open(dataset_uri)
    value = get_raster_properties(dataset)

    #Make sure the dataset is closed and cleaned up
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return value


def get_raster_properties(dataset):
    """
    Get the width, height, X size, and Y size of the dataset and return the
    values in a dictionary.
    *This function can be expanded to return more properties if needed*

    Args:
       dataset (?): a GDAL raster dataset to get the properties from

    Returns:
        dataset_dict (dictionary): a dictionary with the properties stored
            under relevant keys. The current list of things returned is:
            width (w-e pixel resolution), height (n-s pixel resolution),
            XSize, YSize
    """
    dataset_dict = {}
    geo_transform = dataset.GetGeoTransform()
    dataset_dict['width'] = float(geo_transform[1])
    dataset_dict['height'] = float(geo_transform[5])
    dataset_dict['x_size'] = dataset.GetRasterBand(1).XSize
    dataset_dict['y_size'] = dataset.GetRasterBand(1).YSize
    return dataset_dict


def reproject_dataset_uri(
        original_dataset_uri, pixel_spacing, output_wkt, resampling_method,
        output_uri):
    """
    A function to reproject and resample a GDAL dataset given an output
    pixel size and output reference. Will use the datatype and nodata value
    from the original dataset.

    Args:
        original_dataset_uri (string): a URI to a gdal Dataset to written to
            disk
        pixel_spacing (?): output dataset pixel size in projected linear units
        output_wkt (?): output project in Well Known Text
        resampling_method (string): a string representing the one of the
            following resampling methods:
            "nearest|bilinear|cubic|cubic_spline|lanczos"
        output_uri (string): location on disk to dump the reprojected dataset

    Returns:
        projected_dataset (?): reprojected dataset
            (Note from Will: I possibly mislabeled this: looks like it's
                saved to file, with nothing returned)
    """

    # A dictionary to map the resampling method input string to the gdal type
    resample_dict = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubic_spline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos
        }

    # Get the nodata value and datatype from the original dataset
    output_type = get_datatype_from_uri(original_dataset_uri)
    out_nodata = get_nodata_from_uri(original_dataset_uri)

    original_dataset = gdal.Open(original_dataset_uri)

    original_wkt = original_dataset.GetProjection()

    # Create a virtual raster that is projected based on the output WKT. This
    # vrt does not save to disk and is used to get the proper projected bounding
    # box and size.
    vrt = gdal.AutoCreateWarpedVRT(
        original_dataset, None, output_wkt, gdal.GRA_Bilinear)

    geo_t = vrt.GetGeoTransform()
    x_size = vrt.RasterXSize # Raster xsize
    y_size = vrt.RasterYSize # Raster ysize

    # Calculate the extents of the projected dataset. These values will be used
    # to properly set the resampled size for the output dataset
    (ulx, uly) = (geo_t[0], geo_t[3])
    (lrx, lry) = (geo_t[0] + geo_t[1] * x_size, geo_t[3] + geo_t[5] * y_size)

    gdal_driver = gdal.GetDriverByName('GTiff')

    # Create the output dataset to receive the projected output, with the proper
    # resampled arrangement.
    output_dataset = gdal_driver.Create(
        output_uri, int((lrx - ulx)/pixel_spacing),
        int((uly - lry)/pixel_spacing), 1, output_type,
        options=['BIGTIFF=IF_SAFER'])

    # Set the nodata value for the output dataset
    output_dataset.GetRasterBand(1).SetNoDataValue(float(out_nodata))

    # Calculate the new geotransform
    output_geo = (ulx, pixel_spacing, geo_t[2], uly, geo_t[4], -pixel_spacing)

    # Set the geotransform
    output_dataset.SetGeoTransform(output_geo)
    output_dataset.SetProjection(output_wkt)

    # Perform the projection/resampling
    gdal.ReprojectImage(
        original_dataset, output_dataset, original_wkt, output_wkt,
        resample_dict[resampling_method])

    output_dataset.FlushCache()

    #Make sure the dataset is closed and cleaned up
    gdal.Dataset.__swig_destroy__(output_dataset)
    output_dataset = None

    calculate_raster_stats_uri(output_uri)


def reproject_datasource_uri(original_dataset_uri, output_wkt, output_uri):
    """
    URI wrapper for reproject_datasource that takes in the uri for the
    datasource that is to be projected instead of the datasource itself.
    This function directly calls reproject_datasource.

    Args:
        original_dataset_uri (string): a uri to an ogr datasource
        output_wkt (?): the desired projection as Well Known Text
            (by layer.GetSpatialRef().ExportToWkt())
        output_uri (string): the path to where the new shapefile should be
            written to disk.

    Return:
        nothing

    """

    original_dataset = ogr.Open(original_dataset_uri)
    _ = reproject_datasource(original_dataset, output_wkt, output_uri)


def reproject_datasource(original_datasource, output_wkt, output_uri):
    """
    Changes the projection of an ogr datasource by creating a new
    shapefile based on the output_wkt passed in.  The new shapefile
    then copies all the features and fields of the original_datasource
    as its own.

    Args:
        original_datasource (?): an ogr datasource
        output_wkt (?): the desired projection as Well Known Text
            (by layer.GetSpatialRef().ExportToWkt())
        output_uri (string): the filepath to the output shapefile

    Returns:
        output_datasource (?): the reprojected shapefile.
    """
    # if this file already exists, then remove it
    if os.path.isfile(output_uri):
        os.remove(output_uri)

    output_sr = osr.SpatialReference()
    output_sr.ImportFromWkt(output_wkt)

    # create a new shapefile from the orginal_datasource
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(output_uri)

    # loop through all the layers in the orginal_datasource
    for original_layer in original_datasource:

        #Get the original_layer definition which holds needed attribute values
        original_layer_dfn = original_layer.GetLayerDefn()

        #Create the new layer for output_datasource using same name and geometry
        #type from original_datasource, but different projection
        output_layer = output_datasource.CreateLayer(
            original_layer_dfn.GetName(), output_sr,
            original_layer_dfn.GetGeomType())

        #Get the number of fields in original_layer
        original_field_count = original_layer_dfn.GetFieldCount()

        #For every field, create a duplicate field and add it to the new
        #shapefiles layer
        for fld_index in range(original_field_count):
            original_field = original_layer_dfn.GetFieldDefn(fld_index)
            output_field = ogr.FieldDefn(
                original_field.GetName(), original_field.GetType())
            output_layer.CreateField(output_field)

        original_layer.ResetReading()

        #Get the spatial reference of the original_layer to use in transforming
        original_sr = original_layer.GetSpatialRef()

        #Create a coordinate transformation
        coord_trans = osr.CoordinateTransformation(original_sr, output_sr)

        #Copy all of the features in original_layer to the new shapefile
        for original_feature in original_layer:
            geom = original_feature.GetGeometryRef()

            #Transform the geometry into a format desired for the new projection
            geom.Transform(coord_trans)

            #Copy original_datasource's feature and set as new shapes feature
            output_feature = ogr.Feature(
                feature_def=output_layer.GetLayerDefn())
            output_feature.SetFrom(original_feature)
            output_feature.SetGeometry(geom)

            #For all the fields in the feature set the field values from the
            #source field
            for fld_index2 in range(output_feature.GetFieldCount()):
                original_field_value = original_feature.GetField(fld_index2)
                output_feature.SetField(fld_index2, original_field_value)

            output_layer.CreateFeature(output_feature)
            output_feature = None

            original_feature = None

        original_layer = None

    return output_datasource


def unique_raster_values_uri(dataset_uri):
    """
    Returns a list of the unique integer values on the given dataset

    Args:
        dataset_uri (string): a uri to a gdal dataset of some integer type

    Returns:
        value (list): a list of dataset's unique non-nodata values
    """

    dataset = gdal.Open(dataset_uri)
    value = unique_raster_values(dataset)

    #Make sure the dataset is closed and cleaned up
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return value


def unique_raster_values(dataset):
    """
    Returns a list of the unique integer values on the given dataset

    Args:
        dataset (?): a gdal dataset of some integer type

    Returns:
        unique_list (list): a list of dataset's unique non-nodata values

    """

    band, nodata = extract_band_and_nodata(dataset)
    n_rows = band.YSize
    unique_values = numpy.array([])
    for row_index in xrange(n_rows):
        array = band.ReadAsArray(0, row_index, band.XSize, 1)[0]
        array = numpy.append(array, unique_values)
        unique_values = numpy.unique(array)

    unique_list = list(unique_values)
    if nodata in unique_list:
        unique_list.remove(nodata)
    return unique_list


def get_rat_as_dictionary_uri(dataset_uri):
    """
    Returns the RAT of the first band of dataset as a dictionary.

    Args:
        dataset (?): a GDAL dataset that has a RAT associated with the first
            band

    Returns:
        value (dictionary): a 2D dictionary where the first key is the column
            name and second is the row number

    """

    dataset = gdal.Open(dataset_uri)
    value = get_rat_as_dictionary(dataset)

    #Make sure the dataset is closed and cleaned up
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return value


def get_rat_as_dictionary(dataset):
    """
    Returns the RAT of the first band of dataset as a dictionary.

    Args:
        dataset (?): a GDAL dataset that has a RAT associated with the first
            band

    Returns:
        rat_dictionary (dictionary): a 2D dictionary where the first key is the
            column name and second is the row number

    """

    band = dataset.GetRasterBand(1)
    rat = band.GetDefaultRAT()
    n_columns = rat.GetColumnCount()
    n_rows = rat.GetRowCount()
    rat_dictionary = {}

    for col_index in xrange(n_columns):
        #Initialize an empty list to store row data and figure out the type
        #of data stored in that column.
        col_type = rat.GetTypeOfCol(col_index)
        col_name = rat.GetNameOfCol(col_index)
        rat_dictionary[col_name] = []

        #Now burn through all the rows to populate the column
        for row_index in xrange(n_rows):
            #This bit of python ugliness handles the known 3 types of gdal
            #RAT fields.
            if col_type == gdal.GFT_Integer:
                value = rat.GetValueAsInt(row_index, col_index)
            elif col_type == gdal.GFT_Real:
                value = rat.GetValueAsDouble(row_index, col_index)
            else:
                #If the type is not int or real, default to a string,
                #I think this is better than testing for a string and raising
                #an exception if not
                value = rat.GetValueAsString(row_index, col_index)

            rat_dictionary[col_name].append(value)

    return rat_dictionary


def reclassify_dataset_uri(
        dataset_uri, value_map, raster_out_uri, out_datatype, out_nodata,
        exception_flag='none', assert_dataset_projected=True):
    """
    A function to reclassify values in dataset to any output type. If there are
    values in the dataset that are not in value map, they will be mapped to
    out_nodata.

    Args:
        dataset_uri (string): a uri to a gdal dataset
        value_map (dictionary): a dictionary of values of
            {source_value: dest_value, ...}
            where source_value's type is a postive integer type and dest_value
            is of type out_datatype.
        raster_out_uri (string): the uri for the output raster
        out_datatype (?): the type for the output dataset
        out_nodata (?): the nodata value for the output raster.  Must be the
            same type as out_datatype

    Keyword Args:
        exception_flag (string): either 'none' or 'values_required'.
            If 'values_required' raise an exception if there is a value in the
            raster that is not found in value_map
        assert_dataset_projected (boolean): if True this operation will
            test if the input dataset is not projected and raise an exception
            if so.

    Returns:
        reclassified_dataset (?): the new reclassified dataset GDAL raster

    Raises:
        Exception: if exception_flag == 'values_required' and the value from
           'key_raster' is not a key in 'attr_dict'

    """

    nodata = get_nodata_from_uri(dataset_uri)

    def map_dataset_to_value(original_values):
        """Converts a block of original values to the lookup values"""
        all_mapped = numpy.empty(original_values.shape, dtype=numpy.bool)
        out_array = numpy.empty(original_values.shape, dtype=numpy.float)
        nodata_mask = numpy.zeros(original_values.shape, dtype=numpy.bool)
        for key, value in value_map.iteritems():
            mask = original_values == key
            all_mapped = all_mapped | mask
            out_array[mask] = value
        if nodata != None:
            nodata_mask = original_values == nodata
            all_mapped = all_mapped | nodata_mask
        out_array[nodata_mask] = out_nodata
        if not all_mapped.all() and exception_flag == 'values_required':
            raise Exception(
                'There was not a value for at least the following codes '
                'codes %s for this file %s.\nNodata value is: %s' % (
                    str(numpy.unique(original_values[~all_mapped])),
                    dataset_uri, str(nodata)))
        return out_array

    out_pixel_size = get_cell_size_from_uri(dataset_uri)
    vectorize_datasets(
        [dataset_uri], map_dataset_to_value,
        raster_out_uri, out_datatype, out_nodata, out_pixel_size,
        "intersection", dataset_to_align_index=0,
        vectorize_op=False, assert_datasets_projected=assert_dataset_projected)


def reclassify_dataset(
        dataset, value_map, raster_out_uri, out_datatype, out_nodata,
        exception_flag='none'):

    """
    An efficient function to reclassify values in a positive int dataset type
    to any output type.  If there are values in the dataset that are not in
    value map, they will be mapped to out_nodata.

    Args:
        dataset (?): a gdal dataset of some int type
        value_map (dict): a dictionary of values of
            {source_value: dest_value, ...}
            where source_value's type is a postive integer type and dest_value
            is of type out_datatype.
        raster_out_uri (string): the uri for the output raster
        out_datatype (?): the type for the output dataset
        out_nodata (?): the nodata value for the output raster.  Must be the
            same type as out_datatype

    Keyword Args:
        exception_flag (string): either 'none' or 'values_required'.
            If 'values_required' raise an exception if there is a value in the
            raster that is not found in value_map

    Returns:
        reclassified_dataset (?): the new reclassified dataset GDAL raster

    Raises:
        Exception: if exception_flag == 'values_required' and the value from
           'key_raster' is not a key in 'attr_dict'
    """

    out_dataset = new_raster_from_base(
        dataset, raster_out_uri, 'GTiff', out_nodata, out_datatype)
    out_band = out_dataset.GetRasterBand(1)

    in_band, in_nodata = extract_band_and_nodata(dataset)
    in_band.ComputeStatistics(0)

    dataset_max = in_band.GetMaximum()

    #Make an array the same size as the max entry in the dictionary of the same
    #type as the output type.  The +2 adds an extra entry for the nodata values
    #The dataset max ensures that there are enough values in the array
    valid_set = set(value_map.keys())
    map_array_size = max(dataset_max, max(valid_set)) + 2
    valid_set.add(map_array_size - 1) #add the index for nodata
    map_array = numpy.empty(
        (1, map_array_size), dtype=type(value_map.values()[0]))
    map_array[:] = out_nodata

    for key, value in value_map.iteritems():
        map_array[0, key] = value

    for row_index in xrange(in_band.YSize):
        row_array = in_band.ReadAsArray(0, row_index, in_band.XSize, 1)

        #It's possible for in_nodata to be None if the original raster
        #is none, we need to skip the index trick in that case.
        if in_nodata != None:
            #Remaps pesky nodata values the last index in map_array
            row_array[row_array == in_nodata] = map_array_size - 1

        if exception_flag == 'values_required':
            unique_set = set(row_array[0])
            if not unique_set.issubset(valid_set):
                undefined_set = unique_set.difference(valid_set)
                raise ValueError(
                    "The following values were in the raster but not in the "
                    "value_map %s" % (list(undefined_set)))

        row_array = map_array[numpy.ix_([0], row_array[0])]
        out_band.WriteArray(row_array, 0, row_index)

    out_dataset.FlushCache()
    return out_dataset


def load_memory_mapped_array(dataset_uri, memory_file, array_type=None):
    """
    This function loads the first band of a dataset into a memory mapped
    array.

    Args:
        dataset_uri (string): the GDAL dataset to load into a memory mapped
            array
        memory_uri (string): a path to a file OR a file-like object that will
            be used to hold the memory map. It is up to the caller to create
            and delete this file.

    Keyword Args:
        array_type (?): the type of the resulting array, if None defaults
            to the type of the raster band in the dataset

    Returns:
        memory_array (memmap numpy array): a memmap numpy array of the data
            contained in the first band of dataset_uri

    """

    dataset = gdal.Open(dataset_uri)
    band = dataset.GetRasterBand(1)
    n_rows = dataset.RasterYSize
    n_cols = dataset.RasterXSize

    if array_type == None:
        try:
            dtype = _gdal_to_numpy_type(band)
        except KeyError:
            raise TypeError('Unknown GDAL type %s' % band.DataType)
    else:
        dtype = array_type

    memory_array = numpy.memmap(
        memory_file, dtype=dtype, mode='w+', shape=(n_rows, n_cols))

    band.ReadAsArray(buf_obj=memory_array)

    #Make sure the dataset is closed and cleaned up
    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return memory_array


def temporary_filename(suffix=''):
    """
    Returns a temporary filename using mkstemp. The file is deleted
    on exit using the atexit register.

    Keyword Args:
        suffix (string): the suffix to be appended to the temporary file

    Returns:
        fname (?): a unique temporary filename

    """

    file_handle, path = tempfile.mkstemp(suffix=suffix)
    os.close(file_handle)

    def remove_file(path):
        """Function to remove a file and handle exceptions to register
            in atexit"""
        try:
            os.remove(path)
        except OSError:
            #This happens if the file didn't exist, which is okay because maybe
            #we deleted it in a method
            pass

    atexit.register(remove_file, path)
    return path


def temporary_folder():
    """
    Returns a temporary folder using mkdtemp.  The folder is deleted on exit
    using the atexit register.

    Returns:
        path (string): an absolute, unique and temporary folder path.

    """

    path = tempfile.mkdtemp()

    def remove_folder(path):
        """Function to remove a folder and handle exceptions encountered.  This
        function will be registered in atexit."""
        shutil.rmtree(path, ignore_errors=True)

    atexit.register(remove_folder, path)
    return path


class DatasetUnprojected(Exception):
    """An exception in case a dataset is unprojected"""
    pass


class DifferentProjections(Exception):
    """An exception in case a set of datasets are not in the same projection"""
    pass


def assert_datasets_in_same_projection(dataset_uri_list):
    """
    Tests if datasets represented by their uris are projected and in
    the same projection and raises an exception if not.

    Args:
        dataset_uri_list (list): (description)

    Returns:
        is_true (boolean): True (otherwise exception raised)

    Raises:
        DatasetUnprojected: if one of the datasets is unprojected.
        DifferentProjections: if at least one of the datasets is in
            a different projection

    """

    dataset_list = [gdal.Open(dataset_uri) for dataset_uri in dataset_uri_list]
    dataset_projections = []

    unprojected_datasets = set()

    for dataset in dataset_list:
        projection_as_str = dataset.GetProjection()
        dataset_sr = osr.SpatialReference()
        dataset_sr.ImportFromWkt(projection_as_str)
        if not dataset_sr.IsProjected():
            unprojected_datasets.add(dataset.GetFileList()[0])
        dataset_projections.append((dataset_sr, dataset.GetFileList()[0]))

    if len(unprojected_datasets) > 0:
        raise DatasetUnprojected(
            "These datasets are unprojected %s" % (unprojected_datasets))

    for index in range(len(dataset_projections)-1):
        if not dataset_projections[index][0].IsSame(
                dataset_projections[index+1][0]):
            LOGGER.warn(
                "These two datasets might not be in the same projection."
                " The different projections are:\n\n'filename: %s'\n%s\n\n"
                "and:\n\n'filename:%s'\n%s\n\n",
                dataset_projections[index][1],
                dataset_projections[index][0].ExportToPrettyWkt(),
                dataset_projections[index+1][1],
                dataset_projections[index+1][0].ExportToPrettyWkt())

    for dataset in dataset_list:
        #Make sure the dataset is closed and cleaned up
        gdal.Dataset.__swig_destroy__(dataset)
    dataset_list = None
    return True


def get_bounding_box(dataset_uri):
    """
    Returns a bounding box where coordinates are in projected units.

    Args:
        dataset_uri (string): a uri to a GDAL dataset

    Returns:
        bounding_box (list):
            [upper_left_x, upper_left_y, lower_right_x, lower_right_y] in
            projected coordinates

    """

    dataset = gdal.Open(dataset_uri)

    geotransform = dataset.GetGeoTransform()
    n_cols = dataset.RasterXSize
    n_rows = dataset.RasterYSize

    bounding_box = [geotransform[0],
                    geotransform[3],
                    geotransform[0] + n_cols * geotransform[1],
                    geotransform[3] + n_rows * geotransform[5]]

    #Make sure the dataset is closed and cleaned up
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return bounding_box


def get_datasource_bounding_box(datasource_uri):
    """
    Returns a bounding box where coordinates are in projected units.

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
    #Reindex datasource extents into the upper left/lower right coordinates
    bounding_box = [extent[0],
                    extent[3],
                    extent[1],
                    extent[2]]
    return bounding_box


def resize_and_resample_dataset_uri(
        original_dataset_uri, bounding_box, out_pixel_size, output_uri,
        resample_method):
    """
    A function to  a datsaet to larger or smaller pixel sizes

    Args:
        original_dataset_uri (string): a GDAL dataset
        bounding_box (list): [upper_left_x, upper_left_y, lower_right_x,
            lower_right_y]
        out_pixel_size (?): the pixel size in projected linear units
        output_uri (string): the location of the new resampled GDAL dataset
        resample_method (string): the resampling technique, one of
            "nearest|bilinear|cubic|cubic_spline|lanczos"

    Returns:
        nothing

    """

    resample_dict = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubic_spline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos
        }

    original_dataset = gdal.Open(original_dataset_uri)
    original_band = original_dataset.GetRasterBand(1)
    original_nodata = original_band.GetNoDataValue()

    if original_nodata is None:
        original_nodata = -9999

    #gdal python doesn't handle unsigned nodata values well and sometime returns
    #negative numbers.  this guards against that
    if original_band.DataType == gdal.GDT_Byte:
        original_nodata %= 2**8
    if original_band.DataType == gdal.GDT_UInt16:
        original_nodata %= 2**16
    if original_band.DataType == gdal.GDT_UInt32:
        original_nodata %= 2**32

    original_sr = osr.SpatialReference()
    original_sr.ImportFromWkt(original_dataset.GetProjection())

    output_geo_transform = [
        bounding_box[0], out_pixel_size, 0.0, bounding_box[1], 0.0,
        -out_pixel_size]
    new_x_size = abs(
        int(numpy.round((bounding_box[2] - bounding_box[0]) / out_pixel_size)))
    new_y_size = abs(
        int(numpy.round((bounding_box[3] - bounding_box[1]) / out_pixel_size)))

    #create the new x and y size
    block_size = original_band.GetBlockSize()
    #If the original band is tiled, then its x blocksize will be different than
    #the number of columns
    if block_size[0] != original_band.XSize and original_band.XSize > 256 and original_band.YSize > 256:
        #it makes sense for many functions to have 256x256 blocks
        block_size[0] = 256
        block_size[1] = 256
        gtiff_creation_options = [
            'TILED=YES', 'BIGTIFF=IF_SAFER', 'BLOCKXSIZE=%d' % block_size[0],
            'BLOCKYSIZE=%d' % block_size[1]]
    else:
        #it is so small or strangely aligned, use the default creation options
        gtiff_creation_options = []

    create_directories([os.path.dirname(output_uri)])
    gdal_driver = gdal.GetDriverByName('GTiff')
    output_dataset = gdal_driver.Create(
        output_uri, new_x_size, new_y_size, 1, original_band.DataType,
        options=gtiff_creation_options)
    output_band = output_dataset.GetRasterBand(1)

    output_band.SetNoDataValue(original_nodata)

    # Set the geotransform
    output_dataset.SetGeoTransform(output_geo_transform)
    output_dataset.SetProjection(original_sr.ExportToWkt())

    #need to make this a closure so we get the current time and we can affect
    #state
    def reproject_callback(df_complete, psz_message, p_progress_arg):
        """The argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - reproject_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and reproject_callback.total_time >= 5.0)):
                LOGGER.info(
                    "ReprojectImage %.1f%% complete %s, psz_message %s",
                    df_complete * 100, p_progress_arg[0], psz_message)
                reproject_callback.last_time = current_time
                reproject_callback.total_time += current_time
        except AttributeError:
            reproject_callback.last_time = time.time()
            reproject_callback.total_time = 0.0

    # Perform the projection/resampling
    gdal.ReprojectImage(
        original_dataset, output_dataset, original_sr.ExportToWkt(),
        original_sr.ExportToWkt(), resample_dict[resample_method], 0, 0,
        reproject_callback, [output_uri])

    #Make sure the dataset is closed and cleaned up
    original_band = None
    gdal.Dataset.__swig_destroy__(original_dataset)
    original_dataset = None

    output_dataset.FlushCache()
    gdal.Dataset.__swig_destroy__(output_dataset)
    output_dataset = None
    calculate_raster_stats_uri(output_uri)


def align_dataset_list(
        dataset_uri_list, dataset_out_uri_list, resample_method_list,
        out_pixel_size, mode, dataset_to_align_index,
        dataset_to_bound_index=None, aoi_uri=None,
        assert_datasets_projected=True):
    """
    Take a list of dataset uris and generates a new set that is completely
    aligned with identical projections and pixel sizes.

    Args:
        dataset_uri_list (list): a list of input dataset uris
        dataset_out_uri_list (list): a parallel dataset uri list whose
            positions correspond to entries in dataset_uri_list
        resample_method_list (list): a list of resampling methods for each
            output uri in dataset_out_uri list.  Each element must be one of
            "nearest|bilinear|cubic|cubic_spline|lanczos"
        out_pixel_size (?): the output pixel size
        mode (string): one of "union", "intersection", or "dataset" which
            defines how the output output extents are defined as either the
            union or intersection of the input datasets or to have the same
            bounds as an existing raster.  If mode is "dataset" then
            dataset_to_bound_index must be defined
        dataset_to_align_index (int): an int that corresponds to the position
            in one of the dataset_uri_lists that, if positive aligns the output
            rasters to fix on the upper left hand corner of the output
            datasets.  If negative, the bounding box aligns the intersection/
            union without adjustment.

    Keyword Args:
        dataset_to_bound_index (?): if mode is "dataset" then this index is
            used to indicate which dataset to define the output bounds of the
            dataset_out_uri_list
        aoi_uri (string): a URI to an OGR datasource to be used for the
            aoi.  Irrespective of the `mode` input, the aoi will be used
            to intersect the final bounding box.

    Returns:
        nothing

    """

    #This seems a reasonable precursor for some very common issues, numpy gives
    #me a precedent for this.

    #make sure that the input lists are of the same length
    list_lengths = [
        len(dataset_uri_list), len(dataset_out_uri_list),
        len(resample_method_list)]
    if not reduce(lambda x, y: x if x == y else False, list_lengths):
        raise Exception(
            "dataset_uri_list, dataset_out_uri_list, and "
            "resample_method_list must be the same length "
            " current lengths are %s" % (str(list_lengths)))

    if assert_datasets_projected:
        assert_datasets_in_same_projection(dataset_uri_list)
    if mode not in ["union", "intersection", "dataset"]:
        raise Exception("Unknown mode %s" % (str(mode)))

    if dataset_to_align_index >= len(dataset_uri_list):
        raise Exception(
            "Alignment index is out of bounds of the datasets index: %s"
            "n_elements %s" % (dataset_to_align_index, len(dataset_uri_list)))
    if mode == "dataset" and dataset_to_bound_index is None:
        raise Exception(
            "Mode is 'dataset' but dataset_to_bound_index is not defined")
    if mode == "dataset" and (dataset_to_bound_index < 0 or
                              dataset_to_bound_index >= len(dataset_uri_list)):
        raise Exception(
            "dataset_to_bound_index is out of bounds of the datasets index: %s"
            "n_elements %s" % (dataset_to_bound_index, len(dataset_uri_list)))

    def merge_bounding_boxes(bb1, bb2, mode):
        """Helper function to merge two bounding boxes through union or
            intersection"""
        less_than_or_equal = lambda x, y: x if x <= y else y
        greater_than = lambda x, y: x if x > y else y

        if mode == "union":
            comparison_ops = [
                less_than_or_equal, greater_than, greater_than,
                less_than_or_equal]
        if mode == "intersection":
            comparison_ops = [
                greater_than, less_than_or_equal, less_than_or_equal,
                greater_than]

        bb_out = [op(x, y) for op, x, y in zip(comparison_ops, bb1, bb2)]
        return bb_out

    #get the intersecting or unioned bounding box
    if mode == "dataset":
        bounding_box = get_bounding_box(
            dataset_uri_list[dataset_to_bound_index])
    else:
        bounding_box = reduce(
            functools.partial(merge_bounding_boxes, mode=mode),
            [get_bounding_box(dataset_uri) for dataset_uri in dataset_uri_list])

    if aoi_uri != None:
        bounding_box = merge_bounding_boxes(
            bounding_box, get_datasource_bounding_box(aoi_uri), "intersection")


    if (bounding_box[0] >= bounding_box[2] or
            bounding_box[1] <= bounding_box[3]) and mode == "intersection":
        raise Exception("The datasets' intersection is empty "
                        "(i.e., not all the datasets touch each other).")

    if dataset_to_align_index >= 0:
        #bounding box needs alignment
        align_bounding_box = get_bounding_box(
            dataset_uri_list[dataset_to_align_index])
        align_pixel_size = get_cell_size_from_uri(
            dataset_uri_list[dataset_to_align_index])

        for index in [0, 1]:
            n_pixels = int(
                (bounding_box[index] - align_bounding_box[index]) /
                float(align_pixel_size))
            bounding_box[index] = \
                n_pixels * align_pixel_size + align_bounding_box[index]

    for original_dataset_uri, out_dataset_uri, resample_method in zip(
            dataset_uri_list, dataset_out_uri_list, resample_method_list):
        resize_and_resample_dataset_uri(
            original_dataset_uri, bounding_box, out_pixel_size,
            out_dataset_uri, resample_method)

    #If there's an AOI, mask it out
    if aoi_uri != None:
        first_dataset = gdal.Open(dataset_out_uri_list[0])
        n_rows = first_dataset.RasterYSize
        n_cols = first_dataset.RasterXSize
        gdal.Dataset.__swig_destroy__(first_dataset)
        first_dataset = None

        mask_uri = temporary_filename(suffix='.tif')
        new_raster_from_base_uri(
            dataset_out_uri_list[0], mask_uri, 'GTiff', 255, gdal.GDT_Byte,
            fill_value=0)

        mask_dataset = gdal.Open(mask_uri, gdal.GA_Update)
        mask_band = mask_dataset.GetRasterBand(1)
        aoi_datasource = ogr.Open(aoi_uri)
        aoi_layer = aoi_datasource.GetLayer()
        gdal.RasterizeLayer(mask_dataset, [1], aoi_layer, burn_values=[1])
        mask_row = numpy.zeros((1, n_cols), dtype=numpy.int8)

        out_dataset_list = [
            gdal.Open(uri, gdal.GA_Update) for uri in dataset_out_uri_list]
        out_band_list = [
            dataset.GetRasterBand(1) for dataset in out_dataset_list]
        nodata_out_list = [
            get_nodata_from_uri(uri) for uri in dataset_out_uri_list]

        for row_index in range(n_rows):
            mask_row = (mask_band.ReadAsArray(
                0, row_index, n_cols, 1) == 0)
            for out_band, nodata_out in zip(out_band_list, nodata_out_list):
                dataset_row = out_band.ReadAsArray(
                    0, row_index, n_cols, 1)
                out_band.WriteArray(
                    numpy.where(mask_row, nodata_out, dataset_row),
                    xoff=0, yoff=row_index)

        #Remove the mask aoi if necessary
        mask_band = None
        gdal.Dataset.__swig_destroy__(mask_dataset)
        mask_dataset = None
        #Make sure the dataset is closed and cleaned up
        os.remove(mask_uri)

        aoi_layer = None
        ogr.DataSource.__swig_destroy__(aoi_datasource)
        aoi_datasource = None

        #Clean up datasets
        out_band_list = None
        for dataset in out_dataset_list:
            dataset.FlushCache()
            gdal.Dataset.__swig_destroy__(dataset)
        out_dataset_list = None


def assert_file_existance(dataset_uri_list):
    """
    Verify that the uris passed in the argument exist on the filesystem
    if not, raise an exeception indicating which files do not exist

    Args:
        dataset_uri_list (list): a list of relative or absolute file paths to
            validate

    Returns:
        nothing

    Raises:
        IOError: if any files are not found

    """

    not_found_uris = []
    for uri in dataset_uri_list:
        if not os.path.exists(uri):
            not_found_uris.append(uri)

    if len(not_found_uris) != 0:
        error_message = (
            "The following files do not exist on the filesystem: " +
            str(not_found_uris))
        raise exceptions.IOError(error_message)


def vectorize_datasets(
        dataset_uri_list, dataset_pixel_op, dataset_out_uri, datatype_out,
        nodata_out, pixel_size_out, bounding_box_mode,
        resample_method_list=None, dataset_to_align_index=None,
        dataset_to_bound_index=None, aoi_uri=None,
        assert_datasets_projected=True, process_pool=None, vectorize_op=True,
        datasets_are_pre_aligned=False, dataset_options=None):

    """
    This function applies a user defined function across a stack of
    datasets.  It has functionality align the output dataset grid
    with one of the input datasets, output a dataset that is the union
    or intersection of the input dataset bounding boxes, and control
    over the interpolation techniques of the input datasets, if
    necessary.  The datasets in dataset_uri_list must be in the same
    projection; the function will raise an exception if not.

    Args:
        dataset_uri_list (list): a list of file uris that point to files that
            can be opened with gdal.Open.
        dataset_pixel_op (function) a function that must take in as many
            arguments as there are elements in dataset_uri_list.  The arguments
            can be treated as interpolated or actual pixel values from the
            input datasets and the function should calculate the output
            value for that pixel stack.  The function is a parallel
            paradigmn and does not know the spatial position of the
            pixels in question at the time of the call.  If the
            `bounding_box_mode` parameter is "union" then the values
            of input dataset pixels that may be outside their original
            range will be the nodata values of those datasets.  Known
            bug: if dataset_pixel_op does not return a value in some cases
            the output dataset values are undefined even if the function
            does not crash or raise an exception.
        dataset_out_uri (string): the uri of the output dataset.  The
            projection will be the same as the datasets in dataset_uri_list.
        datatype_out (?): the GDAL output type of the output dataset
        nodata_out (?): the nodata value of the output dataset.
        pixel_size_out (?): the pixel size of the output dataset in
            projected coordinates.
        bounding_box_mode (string): one of "union" or "intersection",
            "dataset". If union the output dataset bounding box will be the
            union of the input datasets.  Will be the intersection otherwise.
            An exception is raised if the mode is "intersection" and the
            input datasets have an empty intersection. If dataset it will make
            a bounding box as large as the given dataset, if given
            dataset_to_bound_index must be defined.

    Keyword Args:
        resample_method_list (list): a list of resampling methods
            for each output uri in dataset_out_uri list.  Each element
            must be one of "nearest|bilinear|cubic|cubic_spline|lanczos".
            If None, the default is "nearest" for all input datasets.
        dataset_to_align_index (int): an int that corresponds to the position
            in one of the dataset_uri_lists that, if positive aligns the output
            rasters to fix on the upper left hand corner of the output
            datasets.  If negative, the bounding box aligns the intersection/
            union without adjustment.
        dataset_to_bound_index (?): if mode is "dataset" this indicates which
            dataset should be the output size.
        aoi_uri (string): a URI to an OGR datasource to be used for the
            aoi.  Irrespective of the `mode` input, the aoi will be used
            to intersect the final bounding box.
        assert_datasets_projected (boolean): if True this operation will
            test if any datasets are not projected and raise an exception
            if so.
        process_pool (?): a process pool for multiprocessing
        vectorize_op (boolean): if true the model will try to numpy.vectorize
            dataset_pixel_op.  If dataset_pixel_op is designed to use maximize
            array broadcasting, set this parameter to False, else it may
            inefficiently invoke the function on individual elements.
        datasets_are_pre_aligned (boolean): If this value is set to False
            this operation will first align and interpolate the input datasets
            based on the rules provided in bounding_box_mode,
            resample_method_list, dataset_to_align_index, and
            dataset_to_bound_index, if set to True the input dataset list must
            be aligned, probably by raster_utils.align_dataset_list
        dataset_options (?): this is an argument list that will be
            passed to the GTiff driver.  Useful for blocksizes, compression,
            etc.

    Returns:
        nothing?

    Raises:
        ValueError: invalid input provided

    """
    if type(dataset_uri_list) != list:
        raise ValueError(
            "dataset_uri_list was not passed in as a list, maybe a single "
            "file was passed in?  Here is its value: %s" %
            (str(dataset_uri_list)))

    if aoi_uri == None:
        assert_file_existance(dataset_uri_list)
    else:
        assert_file_existance(dataset_uri_list + [aoi_uri])

    if dataset_out_uri in dataset_uri_list:
        raise ValueError(
            "%s is used as an output file, but it is also an input file "
            "in the input list %s" % (dataset_out_uri, str(dataset_uri_list)))

    #Create a temporary list of filenames whose files delete on the python
    #interpreter exit
    if not datasets_are_pre_aligned:
        #Handle the cases where optional arguments are passed in
        if resample_method_list == None:
            resample_method_list = ["nearest"] * len(dataset_uri_list)
        if dataset_to_align_index == None:
            dataset_to_align_index = -1
        dataset_out_uri_list = [
            temporary_filename(suffix='.tif') for _ in dataset_uri_list]
        #Align and resample the datasets, then load datasets into a list
        align_dataset_list(
            dataset_uri_list, dataset_out_uri_list, resample_method_list,
            pixel_size_out, bounding_box_mode, dataset_to_align_index,
            dataset_to_bound_index=dataset_to_bound_index,
            aoi_uri=aoi_uri,
            assert_datasets_projected=assert_datasets_projected)
        aligned_datasets = [
            gdal.Open(filename, gdal.GA_ReadOnly) for filename in
            dataset_out_uri_list]
    else:
        #otherwise the input datasets are already aligned
        aligned_datasets = [
            gdal.Open(filename, gdal.GA_ReadOnly) for filename in
            dataset_uri_list]

    aligned_bands = [dataset.GetRasterBand(1) for dataset in aligned_datasets]

    n_rows = aligned_datasets[0].RasterYSize
    n_cols = aligned_datasets[0].RasterXSize

    output_dataset = new_raster_from_base(
        aligned_datasets[0], dataset_out_uri, 'GTiff', nodata_out, datatype_out,
        dataset_options=dataset_options)
    output_band = output_dataset.GetRasterBand(1)
    block_size = output_band.GetBlockSize()
    #makes sense to get the largest block size possible to reduce the number
    #of expensive readasarray calls
    for current_block_size in [band.GetBlockSize() for band in aligned_bands]:
        if (current_block_size[0] * current_block_size[1] >
                block_size[0] * block_size[1]):
            block_size = current_block_size

    cols_per_block, rows_per_block = block_size[0], block_size[1]
    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    dataset_blocks = [
        numpy.zeros(
            (rows_per_block, cols_per_block),
            dtype=_gdal_to_numpy_type(band)) for band in aligned_bands]

    #If there's an AOI, mask it out
    if aoi_uri != None:
        mask_uri = temporary_filename(suffix='.tif')
        mask_dataset = new_raster_from_base(
            aligned_datasets[0], mask_uri, 'GTiff', 255, gdal.GDT_Byte,
            fill_value=0, dataset_options=dataset_options)
        mask_band = mask_dataset.GetRasterBand(1)
        aoi_datasource = ogr.Open(aoi_uri)
        aoi_layer = aoi_datasource.GetLayer()
        gdal.RasterizeLayer(mask_dataset, [1], aoi_layer, burn_values=[1])
        mask_array = numpy.zeros(
            (rows_per_block, cols_per_block), dtype=numpy.int8)
        aoi_layer = None
        aoi_datasource = None

    #We only want to do this if requested, otherwise we might have a more
    #efficient call if we don't vectorize.
    if vectorize_op:
        LOGGER.warn("this call is vectorizing which is deprecated and slow")
        dataset_pixel_op = numpy.vectorize(dataset_pixel_op)

    dataset_blocks = [
        numpy.zeros(
            (rows_per_block, cols_per_block),
            dtype=_gdal_to_numpy_type(band)) for band in aligned_bands]

    last_time = time.time()

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

            current_time = time.time()
            if current_time - last_time > 5.0:
                LOGGER.info(
                    'raster stack calculation approx. %.2f%% complete',
                    ((row_block_index * n_col_blocks + col_block_index) /
                     float(n_row_blocks * n_col_blocks) * 100.0))
                last_time = current_time

            for dataset_index in xrange(len(aligned_bands)):
                aligned_bands[dataset_index].ReadAsArray(
                    xoff=col_offset, yoff=row_offset, win_xsize=col_block_width,
                    win_ysize=row_block_width,
                    buf_obj=dataset_blocks[dataset_index][0:row_block_width,0:col_block_width])

            out_block = dataset_pixel_op(*dataset_blocks)

            #Mask out the row if there is a mask
            if aoi_uri != None:
                mask_band.ReadAsArray(
                    xoff=col_offset, yoff=row_offset, win_xsize=col_block_width,
                    win_ysize=row_block_width,
                    buf_obj=mask_array[0:row_block_width,0:col_block_width])
                out_block[mask_array == 0] = nodata_out

            output_band.WriteArray(
                out_block[0:row_block_width, 0:col_block_width],
                xoff=col_offset, yoff=row_offset)

    #Making sure the band and dataset is flushed and not in memory before
    #adding stats
    output_band.FlushCache()
    output_band = None
    output_dataset.FlushCache()
    gdal.Dataset.__swig_destroy__(output_dataset)
    output_dataset = None

    #Clean up the files made by temporary file because we had an issue once
    #where I was running the water yield model over 2000 times and it made
    #so many temporary files I ran out of disk space.
    if aoi_uri != None:
        mask_band = None
        gdal.Dataset.__swig_destroy__(mask_dataset)
        mask_dataset = None
        os.remove(mask_uri)
    aligned_bands = None
    for dataset in aligned_datasets:
        gdal.Dataset.__swig_destroy__(dataset)
    aligned_datasets = None
    if not datasets_are_pre_aligned:
        #if they weren't pre-aligned then we have temporary files to remove
        for temp_dataset_uri in dataset_out_uri_list:
            try:
                os.remove(temp_dataset_uri)
            except OSError:
                LOGGER.warn("couldn't delete file %s", temp_dataset_uri)
    calculate_raster_stats_uri(dataset_out_uri)


def get_lookup_from_table(table_uri, key_field):
    """
    Creates a python dictionary to look up the rest of the fields in a
    table file table indexed by the given key_field

    Args:
        table_uri (string): a URI to a dbf or csv file containing at
            least the header key_field
        key_field (?): (description)

    Returns:
        lookup_dict (dict): a dictionary of the form {key_field_0:
            {header_1: val_1_0, header_2: val_2_0, etc.}
            depending on the values of those fields

    """

    table_object = fileio.TableHandler(table_uri)
    raw_table_dictionary = table_object.get_table_dictionary(key_field.lower())

    lookup_dict = {}
    for key, sub_dict in raw_table_dictionary.iteritems():
        key_value = _smart_cast(key)
        #Map an entire row to its lookup values
        lookup_dict[key_value] = (dict(
            [(sub_key, _smart_cast(value)) for sub_key, value in
             sub_dict.iteritems()]))
    return lookup_dict


def get_lookup_from_csv(csv_table_uri, key_field):
    """
    Creates a python dictionary to look up the rest of the fields in a
    csv table indexed by the given key_field

    Args:
        csv_table_uri (string): a URI to a csv file containing at
            least the header key_field
        key_field (?): (description)

    Returns:
        lookup_dict (dict): returns a dictionary of the form {key_field_0:
            {header_1: val_1_0, header_2: val_2_0, etc.}
            depending on the values of those fields

    """

    def u(string):
        if type(string) is StringType:
            return unicode(string, 'utf-8')
        return string

    with open(csv_table_uri, 'rU') as csv_file:
        csv_reader = csv.reader(csv_file)
        header_row = map(lambda s: u(s), csv_reader.next())
        key_index = header_row.index(key_field)
        #This makes a dictionary that maps the headers to the indexes they
        #represent in the soon to be read lines
        index_to_field = dict(zip(range(len(header_row)), header_row))

        lookup_dict = {}
        for line_num, line in enumerate(csv_reader):
            try:
                key_value = _smart_cast(line[key_index])
            except IndexError as error:
                LOGGER.error('CSV line %s (%s) should have index %s', line_num,
                    line, key_index)
                raise error
            #Map an entire row to its lookup values
            lookup_dict[key_value] = (
                dict([(index_to_field[index], _smart_cast(value))
                      for index, value in zip(range(len(line)), line)]))
        return lookup_dict


def extract_datasource_table_by_key(datasource_uri, key_field):
    """
    Create a dictionary lookup table of the features in the attribute table
    of the datasource referenced by datasource_uri.

    Args:
        datasource_uri (string): a uri to an OGR datasource
        key_field (?): a field in datasource_uri that refers to a key value
            for each row such as a polygon id.

    Returns:
        attribute_dictionary (dict): returns a dictionary of the
            form {key_field_0: {field_0: value0, field_1: value1}...}

    """

    #Pull apart the datasource
    datasource = ogr.Open(datasource_uri)
    layer = datasource.GetLayer()
    layer_def = layer.GetLayerDefn()

    #Build up a list of field names for the datasource table
    field_names = []
    for field_id in xrange(layer_def.GetFieldCount()):
        field_def = layer_def.GetFieldDefn(field_id)
        field_names.append(field_def.GetName())

    #Loop through each feature and build up the dictionary representing the
    #attribute table
    attribute_dictionary = {}
    for feature_index in xrange(layer.GetFeatureCount()):
        feature = layer.GetFeature(feature_index)
        feature_fields = {}
        for field_name in field_names:
            feature_fields[field_name] = feature.GetField(field_name)
        key_value = feature.GetField(key_field)
        attribute_dictionary[key_value] = feature_fields

    #Explictly clean up the layers so the files close
    layer = None
    datasource = None
    return attribute_dictionary


def get_geotransform_uri(dataset_uri):
    """
    Get the geotransform from a gdal dataset

    Args:
        dataset_uri (string): a URI for the dataset

    Returns:
        geotransform (?): a dataset geotransform list

    """

    dataset = gdal.Open(dataset_uri)
    geotransform = dataset.GetGeoTransform()
    gdal.Dataset.__swig_destroy__(dataset)
    return geotransform


def get_spatial_ref_uri(datasource_uri):
    """
    Get the spatial reference of an OGR datasource

    Args:
        datasource_uri (string): a URI to an ogr datasource

    Returns:
        spat_ref (?): a spatial reference

    """

    shape_datasource = ogr.Open(datasource_uri)
    layer = shape_datasource.GetLayer()
    spat_ref = layer.GetSpatialRef()
    return spat_ref


def copy_datasource_uri(shape_uri, copy_uri):
    """
    Create a copy of an ogr shapefile

    Args:
        shape_uri (string): a uri path to the ogr shapefile that is to be
            copied
        copy_uri (string): a uri path for the destination of the copied
            shapefile

    Returns:
        nothing

    """
    if os.path.isfile(copy_uri):
        os.remove(copy_uri)

    shape = ogr.Open(shape_uri)
    drv = ogr.GetDriverByName('ESRI Shapefile')
    drv.CopyDataSource(shape, copy_uri)


def vectorize_points_uri(
        shapefile_uri, field, output_uri, interpolation='nearest'):
    """
    A wrapper function for raster_utils.vectorize_points, that allows for uri
    passing

    Args:
        shapefile_uri (string): a uri path to an ogr shapefile
        field (string): a string for the field name
        output_uri (string): a uri path for the output raster
        interpolation (string): interpolation method to use on points, default
            is 'nearest'

    Returns:
        nothing

    """

    datasource = ogr.Open(shapefile_uri)
    output_raster = gdal.Open(output_uri, 1)
    vectorize_points(
        datasource, field, output_raster, interpolation=interpolation)


def create_directories(directory_list):
    """
    This function will create any of the directories in the directory list
    if possible and raise exceptions if something exception other than
    the directory previously existing occurs.

    Args:
        directory_list (list): a list of string uri paths

    Returns:
        nothing

    """

    for dir_name in directory_list:
        try:
            os.makedirs(dir_name)
        except OSError as exception:
            #It's okay if the directory already exists, if it fails for
            #some other reason, raise that exception
            if exception.errno != errno.EEXIST:
                raise


def dictionary_to_point_shapefile(dict_data, layer_name, output_uri):
    """
    Creates a point shapefile from a dictionary. The point shapefile created
    is not projected and uses latitude and longitude for its geometry.

    Args:
        dict_data (dict): a python dictionary with keys being unique id's that
            point to sub-dictionarys that have key-value pairs. These inner
            key-value pairs will represent the field-value pair for the point
            features. At least two fields are required in the sub-dictionaries,
            All the keys in the sub dictionary should have the same name and
            order. All the values in the sub dictionary should have the same
            type 'lati' and 'long'. These fields determine the geometry of the
            point
            0 : {'lati':97, 'long':43, 'field_a':6.3, 'field_b':'Forest',...},
            1 : {'lati':55, 'long':51, 'field_a':6.2, 'field_b':'Crop',...},
            2 : {'lati':73, 'long':47, 'field_a':6.5, 'field_b':'Swamp',...}
        layer_name (string): a python string for the name of the layer
        output_uri (string): a uri for the output path of the point shapefile

    Returns:
        nothing

    """

    # If the output_uri exists delete it
    if os.path.isfile(output_uri):
        os.remove(output_uri)
    elif os.path.isdir(output_uri):
        shutil.rmtree(output_uri)

    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(output_uri)

    # Set the spatial reference to WGS84 (lat/long)
    source_sr = osr.SpatialReference()
    source_sr.SetWellKnownGeogCS("WGS84")

    output_layer = output_datasource.CreateLayer(
        layer_name, source_sr, ogr.wkbPoint)

    # Outer unique keys
    outer_keys = dict_data.keys()

    # Construct a list of fields to add from the keys of the inner dictionary
    field_list = dict_data[outer_keys[0]].keys()

    # Create a dictionary to store what variable types the fields are
    type_dict = {}
    for field in field_list:
        # Get a value from the field
        val = dict_data[outer_keys[0]][field]
        # Check to see if the value is a String of characters or a number. This
        # will determine the type of field created in the shapefile
        if isinstance(val, str):
            type_dict[field] = 'str'
        else:
            type_dict[field] = 'number'

    for field in field_list:
        field_type = None
        # Distinguish if the field type is of type String or other. If Other, we
        # are assuming it to be a float
        if type_dict[field] == 'str':
            field_type = ogr.OFTString
        else:
            field_type = ogr.OFTReal

        output_field = ogr.FieldDefn(field, field_type)
        output_layer.CreateField(output_field)

    # For each inner dictionary (for each point) create a point and set its
    # fields
    for point_dict in dict_data.itervalues():
        latitude = float(point_dict['lati'])
        longitude = float(point_dict['long'])

        geom = ogr.Geometry(ogr.wkbPoint)
        geom.AddPoint_2D(longitude, latitude)

        output_feature = ogr.Feature(output_layer.GetLayerDefn())

        for field_name in point_dict:
            field_index = output_feature.GetFieldIndex(field_name)
            output_feature.SetField(field_index, point_dict[field_name])

        output_feature.SetGeometryDirectly(geom)
        output_layer.CreateFeature(output_feature)
        output_feature = None

    output_layer.SyncToDisk()


def get_dataset_projection_wkt_uri(dataset_uri):
    """
    Get the projection of a GDAL dataset as well known text (WKT)

    Args:
        dataset_uri (string): a URI for the GDAL dataset

    Returns:
        proj_wkt (string): WKT describing the GDAL dataset project

    """

    dataset = gdal.Open(dataset_uri)
    proj_wkt = dataset.GetProjection()
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None
    return proj_wkt


def unique_raster_values_count(dataset_uri, ignore_nodata=True):
    """
    Return a dict from unique int values in the dataset to their frequency.

    Args:
        dataset_uri (string): uri to a gdal dataset of some integer type

    Keyword Args:
        ignore_nodata (boolean): if set to false, the nodata count is also
            included in the result

    Returns:
        itemfreq (dict): values to count.
    """

    dataset = gdal.Open(dataset_uri)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    itemfreq = collections.defaultdict(int)
    for row_index in range(band.YSize):
        cur_array = band.ReadAsArray(0, row_index, band.XSize, 1)[0]
        for val in numpy.unique(cur_array):
            if ignore_nodata and val == nodata:
                continue
            itemfreq[val] += numpy.count_nonzero(cur_array == val)

    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None
    return itemfreq


def rasterize_layer_uri(
        raster_uri, shapefile_uri, burn_values=[], option_list=[]):
    """
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
        nothing

    """

    dataset = gdal.Open(raster_uri, gdal.GA_Update)
    shapefile = ogr.Open(shapefile_uri)
    layer = shapefile.GetLayer()

    gdal.RasterizeLayer(
        dataset, [1], layer, burn_values=burn_values, options=option_list)

    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None
    shapefile = None


def make_constant_raster_from_base_uri(
        base_dataset_uri, constant_value, out_uri, nodata_value=None,
        dataset_type=gdal.GDT_Float32):
    """
    A helper function that creates a new gdal raster from base, and fills
    it with the constant value provided.

    Args:
        base_dataset_uri (string): the gdal base raster
        constant_value (?): the value to set the new base raster to
        out_uri (string): the uri of the output raster

    Keyword Args:
        nodata_value (?): the value to set the constant raster's nodata
            value to.  If not specified, it will be set to constant_value - 1.0
        dataset_type (?): the datatype to set the dataset to, default
            will be a float 32 value.

    Returns:
        nothing

    """

    if nodata_value == None:
        nodata_value = constant_value - 1.0
    new_raster_from_base_uri(
        base_dataset_uri, out_uri, 'GTiff', nodata_value,
        dataset_type)
    base_dataset = gdal.Open(out_uri, gdal.GA_Update)
    base_band = base_dataset.GetRasterBand(1)
    base_band.Fill(constant_value)

    base_band = None
    gdal.Dataset.__swig_destroy__(base_dataset)
    base_dataset = None


def calculate_disjoint_polygon_set(shapefile_uri):
    """
    Calculates a list of sets of polygons that don't overlap.  Determining
    the minimal number of those sets is an np-complete problem so this is
    an approximation that builds up sets of maximal subsets.

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

    #Build maximal subsets
    subset_list = []
    while len(poly_intersect_lookup) > 0:
        #sort polygons by increasing number of intersections
        heap = []
        for poly_fid, poly_dict in poly_intersect_lookup.iteritems():
            heapq.heappush(
                heap, (len(poly_dict['intersects']), poly_fid, poly_dict))

        #build maximal subset
        maximal_set = set()
        while len(heap) > 0:
            _, poly_fid, poly_dict = heapq.heappop(heap)
            for maxset_fid in maximal_set:
                if maxset_fid in poly_intersect_lookup[poly_fid]['intersects']:
                    #it intersects and can't be part of the maximal subset
                    break
            else:
                #we made it through without an intersection, add poly_fid to
                #the maximal set
                maximal_set.add(poly_fid)
                #remove that polygon and update the intersections
                del poly_intersect_lookup[poly_fid]
            #remove all the polygons from intersections once they're compuated
        for maxset_fid in maximal_set:
            for poly_dict in poly_intersect_lookup.itervalues():
                poly_dict['intersects'].discard(maxset_fid)
        subset_list.append(maximal_set)
    return subset_list


def distance_transform_edt(
        input_mask_uri, output_distance_uri, process_pool=None):
    """
    Calculate the Euclidean distance transform on input_mask_uri and output
    the result into an output raster

    Args:
        input_mask_uri (string): a gdal raster to calculate distance from
            the non 0 value pixels
        output_distance_uri (string): will make a float raster w/ same
            dimensions and projection as input_mask_uri where all zero values
            of input_mask_uri are equal to the euclidean distance to the
            closest non-zero pixel.

    Keyword Args:
        process_pool (?): (description)

    Returns:
        nothing

    """

    mask_as_byte_uri = temporary_filename(suffix='.tif')
    nodata_mask = get_nodata_from_uri(input_mask_uri)
    out_pixel_size = get_cell_size_from_uri(input_mask_uri)
    nodata_out = 255

    def to_byte(input_vector):
        """converts vector to 1, 0, or nodata value to fit in a byte raster"""
        return numpy.where(
            input_vector == nodata_mask, nodata_out, input_vector != 0)

    #64 seems like a reasonable blocksize
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

    pygeoprocessing.geoprocessing_core.distance_transform_edt(
        mask_as_byte_uri, output_distance_uri)
    try:
        os.remove(mask_as_byte_uri)
    except OSError:
        LOGGER.warn("couldn't remove file %s", mask_as_byte_uri)


def convolve_2d_uri(signal_uri, kernel_uri, output_uri, ignore_nodata=True):
    """
    Does a direct convolution on a predefined kernel with the values in
    signal_uri

    Args:
        signal_uri (string): a filepath to a gdal dataset that's the
            souce input.
        kernel_uri (string): a filepath to a gdal dataset that's the
            souce input.
        output_uri (string): a filepath to the gdal dataset
            that's the convolution output of signal and kernel
            that is the same size and projection of signal_uri.
        ignore_nodata (Boolean): If set to true, the nodata values in
            signal_uri and kernel_uri are treated as 0 in the convolution,
            otherwise the raw nodata values are used.  Default True.

    Returns:
        nothing

    """

    output_nodata = -9999

    tmp_signal_uri = temporary_filename()
    tile_dataset_uri(signal_uri, tmp_signal_uri, 256)

    tmp_kernel_uri = temporary_filename()
    tile_dataset_uri(kernel_uri, tmp_kernel_uri, 256)

    new_raster_from_base_uri(
        signal_uri, output_uri, 'GTiff', output_nodata, gdal.GDT_Float32,
        fill_value=0)

    signal_ds = gdal.Open(tmp_signal_uri)
    signal_band = signal_ds.GetRasterBand(1)
    signal_block_col_size, signal_block_row_size = signal_band.GetBlockSize()
    signal_nodata = get_nodata_from_uri(tmp_signal_uri)

    kernel_ds = gdal.Open(tmp_kernel_uri)
    kernel_band = kernel_ds.GetRasterBand(1)
    kernel_block_col_size, kernel_block_row_size = kernel_band.GetBlockSize()
    #make kernel block size a little larger if possible
    kernel_block_col_size *= 3
    kernel_block_row_size *= 3

    kernel_nodata = get_nodata_from_uri(tmp_kernel_uri)

    output_ds = gdal.Open(output_uri, gdal.GA_Update)
    output_band = output_ds.GetRasterBand(1)

    n_rows_signal = signal_band.YSize
    n_cols_signal = signal_band.XSize

    n_rows_kernel = kernel_band.YSize
    n_cols_kernel = kernel_band.XSize

    n_global_block_rows_signal = (
        int(math.ceil(float(n_rows_signal) / signal_block_row_size)))
    n_global_block_cols_signal = (
        int(math.ceil(float(n_cols_signal) / signal_block_col_size)))

    n_global_block_rows_kernel = (
        int(math.ceil(float(n_rows_kernel) / kernel_block_row_size)))
    n_global_block_cols_kernel = (
        int(math.ceil(float(n_cols_kernel) / kernel_block_col_size)))

    last_time = time.time()
    for global_block_row in xrange(n_global_block_rows_signal):
        current_time = time.time()
        if current_time - last_time > 5.0:
            LOGGER.info(
                "convolution %.1f%% complete", global_block_row /
                float(n_global_block_rows_signal) * 100)
            last_time = current_time
        for global_block_col in xrange(n_global_block_cols_signal):
            signal_xoff = global_block_col * signal_block_col_size
            signal_yoff = global_block_row * signal_block_row_size
            win_xsize = min(signal_block_col_size, n_cols_signal - signal_xoff)
            win_ysize = min(signal_block_row_size, n_rows_signal - signal_yoff)

            signal_block = signal_band.ReadAsArray(
                xoff=signal_xoff, yoff=signal_yoff,
                win_xsize=win_xsize, win_ysize=win_ysize)

            if ignore_nodata:
                signal_nodata_mask = signal_block == signal_nodata
                signal_block[signal_nodata_mask] = 0.0

            for kernel_block_row_index in xrange(n_global_block_rows_kernel):
                for kernel_block_col_index in xrange(
                        n_global_block_cols_kernel):
                    kernel_yoff = kernel_block_row_index*kernel_block_row_size
                    kernel_xoff = kernel_block_col_index*kernel_block_col_size

                    kernel_win_ysize = min(
                        kernel_block_row_size, n_rows_kernel - kernel_yoff)
                    kernel_win_xsize = min(
                        kernel_block_col_size, n_cols_kernel - kernel_xoff)

                    left_index_raster = (
                        signal_xoff - n_cols_kernel / 2 + kernel_xoff)
                    right_index_raster = (
                        signal_xoff - n_cols_kernel / 2 + kernel_xoff +
                        win_xsize + kernel_win_xsize - 1)
                    top_index_raster = (
                        signal_yoff - n_rows_kernel / 2 + kernel_yoff)
                    bottom_index_raster = (
                        signal_yoff - n_rows_kernel / 2 + kernel_yoff +
                        win_ysize + kernel_win_ysize - 1)

                    #it's possible that the piece of the integrating kernel
                    #doesn't even affect the final result, we can just skip
                    if (right_index_raster < 0 or
                            bottom_index_raster < 0 or
                            left_index_raster > n_cols_signal or
                            top_index_raster > n_rows_signal):
                        continue

                    kernel_block = kernel_band.ReadAsArray(
                        xoff=kernel_xoff, yoff=kernel_yoff,
                        win_xsize=kernel_win_xsize, win_ysize=kernel_win_ysize)

                    if ignore_nodata:
                        kernel_nodata_mask = (kernel_block == kernel_nodata)
                        kernel_block[kernel_nodata_mask] = 0.0

                    result = scipy.signal.fftconvolve(
                        signal_block, kernel_block, 'full')

                    left_index_result = 0
                    right_index_result = result.shape[1]
                    top_index_result = 0
                    bottom_index_result = result.shape[0]

                    #we might abut the edge of the raster, clip if so
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

                    current_output = output_band.ReadAsArray(
                        xoff=left_index_raster, yoff=top_index_raster,
                        win_xsize=right_index_raster-left_index_raster,
                        win_ysize=bottom_index_raster-top_index_raster)

                    potential_nodata_signal_array = signal_band.ReadAsArray(
                        xoff=left_index_raster, yoff=top_index_raster,
                        win_xsize=right_index_raster-left_index_raster,
                        win_ysize=bottom_index_raster-top_index_raster)
                    nodata_mask = potential_nodata_signal_array == signal_nodata

                    output_array = result[
                        top_index_result:bottom_index_result,
                        left_index_result:right_index_result] + current_output
                    output_array[nodata_mask] = output_nodata

                    output_band.WriteArray(
                        output_array, xoff=left_index_raster,
                        yoff=top_index_raster)

    signal_band = None
    gdal.Dataset.__swig_destroy__(signal_ds)
    signal_ds = None
    os.remove(tmp_signal_uri)


def _smart_cast(value):
    """
    Attempts to cast value to a float, int, or leave it as string

    Args:
        value (?): (description)

    Returns:
        value (?): (description)
    """
    #If it's not a string, don't try to cast it because i got a bug
    #where all my floats were happily cast to ints
    if type(value) != str:
        return value
    for cast_function in [int, float]:
        try:
            return cast_function(value)
        except ValueError:
            pass
    for unicode_type in ['ascii', 'utf-8', 'latin-1']:
        try:
            return value.decode(unicode_type)
        except UnicodeDecodeError:
            pass
    LOGGER.warn("unknown encoding type encountered in _smart_cast: %s" % value)
    return value


def tile_dataset_uri(in_uri, out_uri, blocksize):
    """
    Takes an existing gdal dataset and resamples it into a tiled raster with
    blocks of blocksize X blocksize.

    Args:
        in_uri (string): dataset to base data from
        out_uri (string): output dataset
        blocksize (int): defines the side of the square for the raster, this
            seems to have a lower limit of 16, but is untested

    Returns:
        nothing

    """

    dataset = gdal.Open(in_uri)
    band = dataset.GetRasterBand(1)
    datatype_out = band.DataType
    nodata_out = get_nodata_from_uri(in_uri)
    pixel_size_out = get_cell_size_from_uri(in_uri)
    dataset_options=[
        'TILED=YES', 'BLOCKXSIZE=%d' % blocksize, 'BLOCKYSIZE=%d' % blocksize,
        'BIGTIFF=IF_SAFER']
    vectorize_datasets(
        [in_uri], lambda x: x, out_uri, datatype_out,
        nodata_out, pixel_size_out, 'intersection',
        resample_method_list=None, dataset_to_align_index=None,
        dataset_to_bound_index=None, aoi_uri=None,
        assert_datasets_projected=False, process_pool=None, vectorize_op=False,
        datasets_are_pre_aligned=False, dataset_options=dataset_options)
