"""pygeoprocessing: geoprocessing routines for GIS.

__init__ module imports all the geoprocessing functions into this namespace.
"""
import logging
import types

import pkg_resources
import pkg_resources.extern.packaging.version

from . import geoprocessing
from .geoprocessing import _assert_is_valid_pixel_size
from .geoprocessing import align_and_resize_raster_stack
from .geoprocessing import calculate_disjoint_polygon_set
from .geoprocessing import convolve_2d
from .geoprocessing import create_raster_from_vector_extents
from .geoprocessing import distance_transform_edt
from .geoprocessing import get_gis_type
from .geoprocessing import get_raster_info
from .geoprocessing import get_vector_info
from .geoprocessing import interpolate_points
from .geoprocessing import iterblocks
from .geoprocessing import mask_raster
from .geoprocessing import merge_bounding_box_list
from .geoprocessing import new_raster_from_base
from .geoprocessing import numpy_array_to_raster
from .geoprocessing import raster_calculator
from .geoprocessing import raster_to_numpy_array
from .geoprocessing import rasterize
from .geoprocessing import ReclassificationMissingValuesError
from .geoprocessing import reclassify_raster
from .geoprocessing import reproject_vector
from .geoprocessing import shapely_geometry_to_vector
from .geoprocessing import stitch_rasters
from .geoprocessing import transform_bounding_box
from .geoprocessing import warp_raster
from .geoprocessing import zonal_statistics
from .geoprocessing_core import calculate_slope
from .geoprocessing_core import raster_band_percentile

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except (pkg_resources.extern.packaging.version.InvalidVersion,
        DeprecationWarning):
    # InvalidVersion is raised when our version string isn't PEP440.
    # If there was an error, just skip the version string.
    # Not having a version string shouldn't change how pygeoprocessing behaves
    # or whether it imports.
    # DeprecationWarning sometimes happens when we have an invalid version
    # string as well.
    __version__ = 'unknown'

# Programmatically defining __all__ based on what's been imported.
# Thus, the imports are the source of truth for __all__.
__all__ = ('calculate_slope', 'raster_band_percentile',
           'ReclassificationMissingValuesError')
for attrname in [k for k in locals().keys()]:
    try:
        if isinstance(getattr(geoprocessing, attrname), types.FunctionType):
            __all__ += (attrname,)
    except AttributeError:
        pass

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())  # silence logging by default

# these are bit masks for the known PyGeoprocessing types
UNKNOWN_TYPE = 0
RASTER_TYPE = 1
VECTOR_TYPE = 2
