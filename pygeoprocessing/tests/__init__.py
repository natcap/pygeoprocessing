
"""
Top-level package for containing unit tests for pygeoprocessing.
"""

import os
import unittest
import logging

import nose.tools

LOGGER = logging.getLogger('pygeoprocessing.tests')


@nose.tools.nottest
def test():
    """run modulewide tests"""

    LOGGER.info('running tests on %s', os.path.dirname(__file__))
    suite = unittest.TestLoader().discover(os.path.dirname(__file__))
    unittest.TextTestRunner(verbosity=2).run(suite)
