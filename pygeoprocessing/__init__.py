"""__init__ module for pygeprocessing, imports all the geoprocessing functions
    into the pygeoprocessing namespace"""

import unittest
import logging
import os

from pygeoprocessing.geoprocessing import *

LOGGER = logging.getLogger('pygeoprocessing')
LOGGER.setLevel(logging.DEBUG)

def test():
    """run modulewide tests"""
    LOGGER.info('running tests on %s', os.path.dirname(__file__))
    suite = unittest.TestLoader().discover(os.path.dirname(__file__))
    unittest.TextTestRunner(verbosity=2).run(suite)
