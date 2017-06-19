"""__init__ module for pygeprocessing, imports all the geoprocessing functions
    into the pygeoprocessing namespace"""
from __future__ import absolute_import

import natcap.versioner
__version__ = natcap.versioner.get_version('pygeoprocessing')

import logging
import types
import sys

from . import geoprocessing
from .geoprocessing_core import calculate_slope

LOGGER = logging.getLogger('pygeoprocessing')
LOGGER.setLevel(logging.DEBUG)

__all__ = ['calculate_slope']
for attrname in dir(geoprocessing):
    attribute = getattr(geoprocessing, attrname)
    if isinstance(attribute, types.FunctionType):
        __all__.append(attrname)
        setattr(sys.modules['pygeoprocessing'], attrname, attribute)
