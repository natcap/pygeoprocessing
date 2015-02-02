import unittest
import natcap.geoprocessing
import os

class TestImportPackage(unittest.TestCase):
    def test_import_package(self):
        print
        print os.getcwd()
        if 'site-packages' in natcap.geoprocessing.__file__:
            # assert that the version was set correctly on distribution.
            self.assertNotEqual(natcap.geoprocessing.__version__, 'dev')
        else:
            get_version = lambda: natcap.geoprocessing.__version__
            self.assertRaises(AttributeError, get_version)
