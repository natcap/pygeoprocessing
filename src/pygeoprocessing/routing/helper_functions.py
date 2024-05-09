import numpy
from osgeo import gdal

from ..geoprocessing import get_raster_info
from ..geoprocessing import raster_calculator
from ..geoprocessing_core import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
from ..geoprocessing_core import gdal_use_exceptions


@gdal_use_exceptions
def extract_streams_d8(
        flow_accum_raster_path_band, flow_threshold, target_stream_raster_path,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Extract D8 streams based on a flow accumulation threshold.

    Creates an unsigned byte raster where pixel values of 1 indicate the
    presence of a stream and pixel values of 0 indicate the absence of a
    stream. Any flow accumulation pixels greater than ``flow_threshold`` are
    considered stream pixels. Nodata values found in the input flow
    accumulation raster propagate through to the target stream raster.

    Args:
        flow_accum_raster_path_band (tuple): A (path, band) tuple indicating
            the path to a D8 flow accumulation raster and the band index to
            use.
        flow_threshold (number): The flow threshold. Flow accumulation values
            greater than this threshold are considered stream pixels, values
            less than this threshold are non-stream pixels.
        target_stream_raster_path (string): Where the target streams raster
            should be written.
        raster_driver_creation_tuple (tuple): A tuple where the first element
            is the GDAL driver name of the target raster and the second element
            is an iterable of raster creation options for the selected driver.

    Returns:
        ``None``
    """
    flow_accum_nodata = get_raster_info(
        flow_accum_raster_path_band[0])['nodata'][
            flow_accum_raster_path_band[1]-1]
    target_nodata = 255

    def _threshold_streams(flow_accum):
        out_matrix = numpy.full(flow_accum.shape, target_nodata,
                                dtype=numpy.uint8)

        valid_pixels = numpy.ones(flow_accum.shape, dtype=bool)
        if flow_accum_nodata is not None:
            valid_pixels = ~numpy.isclose(flow_accum, flow_accum_nodata)

        over_threshold = flow_accum > flow_threshold
        out_matrix[valid_pixels & over_threshold] = 1
        out_matrix[valid_pixels & ~over_threshold] = 0
        return out_matrix

    raster_calculator(
        [flow_accum_raster_path_band], _threshold_streams,
        target_stream_raster_path, gdal.GDT_Byte, target_nodata)
