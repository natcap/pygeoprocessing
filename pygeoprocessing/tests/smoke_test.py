import unittest
import pygeoprocessing
import os


class TestImportPackage(unittest.TestCase):
    def test_import_package(self):
        print
        print os.getcwd()
        if 'site-packages' in pygeoprocessing.__file__:
            # assert that the version was set correctly on distribution.
            self.assertNotEqual(pygeoprocessing.__version__, 'dev')
        else:
            get_version = lambda: pygeoprocessing.__version__
            self.assertRaises(AttributeError, get_version)
