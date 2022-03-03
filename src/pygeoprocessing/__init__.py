"""pygeoprocessing: geoprocessing routines for GIS.

__init__ module imports all the geoprocessing functions into this namespace.
"""
import logging
import sys
import types

import pkg_resources
import pkg_resources.extern.packaging.version

from . import geoprocessing
from .geoprocessing import ReclassificationMissingValuesError
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

__all__ = ('calculate_slope', 'raster_band_percentile',
           'ReclassificationMissingValuesError')
for attrname in dir(geoprocessing):
    attribute = getattr(geoprocessing, attrname)
    if isinstance(attribute, types.FunctionType):
        __all__ += (attrname,)
        setattr(sys.modules['pygeoprocessing'], attrname, attribute)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())  # silence logging by default

# these are bit masks for the known PyGeoprocessing types
UNKNOWN_TYPE = 0
RASTER_TYPE = 1
VECTOR_TYPE = 2
