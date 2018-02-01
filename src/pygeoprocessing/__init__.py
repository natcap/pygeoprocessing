"""__init__ module for pygeprocessing, imports all the geoprocessing functions
    into the pygeoprocessing namespace"""
from __future__ import absolute_import

import logging
import types
import sys

import pkg_resources

from . import geoprocessing
from .geoprocessing_core import calculate_slope


try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed.
    pass

LOGGER = logging.getLogger('pygeoprocessing')
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.NullHandler())

__all__ = ['calculate_slope']
for attrname in dir(geoprocessing):
    attribute = getattr(geoprocessing, attrname)
    if isinstance(attribute, types.FunctionType):
        __all__.append(attrname)
        setattr(sys.modules['pygeoprocessing'], attrname, attribute)
