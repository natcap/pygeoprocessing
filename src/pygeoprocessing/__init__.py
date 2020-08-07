"""pygeoprocessing: geoprocessing routines for GIS.

__init__ module imports all the geoprocessing functions into this namespace.
"""
import logging
import sys
import types

from . import geoprocessing
from .geoprocessing import ReclassificationMissingValuesError
from .geoprocessing_core import calculate_slope
from .geoprocessing_core import raster_band_percentile
import pkg_resources
__version__ = pkg_resources.get_distribution(__name__).version

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
