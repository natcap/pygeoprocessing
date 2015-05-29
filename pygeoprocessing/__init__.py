"""__init__ module for pygeprocessing, imports all the geoprocessing functions
    into the pygeoprocessing namespace"""

import os

with open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
            '__version__')) as versionfile:
    __version__ = versionfile.read().rstrip()

import unittest
import logging
import types

import pygeoprocessing.geoprocessing as geoprocessing
from geoprocessing import *

__all__ = []
for attrname in dir(geoprocessing):
    if type(getattr(geoprocessing, attrname)) is types.FunctionType:
        __all__.append(attrname)

LOGGER = logging.getLogger('pygeoprocessing')
LOGGER.setLevel(logging.DEBUG)

def test():
    """run modulewide tests"""
    LOGGER.info('running tests on %s', os.path.dirname(__file__))
    suite = unittest.TestLoader().discover(os.path.dirname(__file__))
    unittest.TextTestRunner(verbosity=2).run(suite)
