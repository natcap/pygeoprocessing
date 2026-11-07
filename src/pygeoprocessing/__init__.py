"""pygeoprocessing: geoprocessing routines for GIS.

__init__ module imports all the geoprocessing functions into this namespace.
"""
import logging
import os
import platform
from importlib.metadata import PackageNotFoundError, version

import lazy_loader

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())  # silence logging by default

# these are bit masks for the known PyGeoprocessing types
UNKNOWN_TYPE = 0
RASTER_TYPE = 1
VECTOR_TYPE = 2

try:
    __version__ = version('pygeoprocessing')
except PackageNotFoundError:
    # package is not installed
    pass


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules=['geoprocessing'],
    submod_attrs={
        "geoprocessing": [
            "_assert_is_valid_pixel_size",
            "align_and_resize_raster_stack",
            "align_bbox",
            "array_equals_nodata",
            "build_overviews",
            "calculate_disjoint_polygon_set",
            "choose_dtype",
            "choose_nodata",
            "convolve_2d",
            "create_raster_from_bounding_box",
            "create_raster_from_vector_extents",
            "distance_transform_edt",
            "get_gis_type",
            "get_raster_info",
            "get_vector_info",
            "interpolate_points",
            "iterblocks",
            "mask_raster",
            "merge_bounding_box_list",
            "new_raster_from_base",
            "numpy_array_to_raster",
            "raster_calculator",
            "raster_map",
            "raster_reduce",
            "raster_to_numpy_array",
            "rasterize",
            "ReclassificationMissingValuesError",
            "reclassify_raster",
            "reproject_vector",
            "shapely_geometry_to_vector",
            "stitch_rasters",
            "transform_bounding_box",
            "warp_raster",
            "zonal_statistics",
        ],
        "slurm_utils": [
            "log_warning_if_gdal_will_exhaust_slurm_memory",
        ],
        "utils": [
            "GDALUseExceptions",
            "gdal_use_exceptions",
        ],
        "geoprocessing_core": [
            "calculate_slope",
            "raster_band_percentile",
        ],
    }
)

def get_include():
    """Return the directory that includes the pygeoprocessing *.h header files.

    If any environment variables are included in the package location path
    (which sometimes happens in conda-forge), then these are expanded.

    Returns:
        The string path to the header files location."""
    package_location = os.path.dirname(os.path.expandvars(__file__))
    ext_location = os.path.join(package_location, 'extensions')
    LOGGER.info(f"Pygeoprocessing extensions location: {ext_location}")
    return ext_location
