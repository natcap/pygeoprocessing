import unittest
import natcap.geoprocessing

class TestImportPackage(unittest.TestCase):
    def test_import_package(self):
        # assert that the version was set correctly on distribution.
        self.assertNotEqual(natcap.geoprocessing.__version__, 'dev')
