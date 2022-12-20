import numpy
from osgeo import gdal

from ..geoprocessing import get_raster_info
from ..geoprocessing import raster_calculator
from ..geoprocessing_core import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS


def extract_streams_d8(
        flow_accum_raster_path_band, flow_threshold, target_stream_raster_path,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
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
