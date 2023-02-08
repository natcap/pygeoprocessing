import os
import shutil
import tempfile
import unittest

import numpy
import numpy.testing
from osgeo import gdal


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()
        self.filepath = os.path.join(self.workspace, 'kernel.tif')

    def tearDown(self):
        shutil.rmtree(self.workspace)

    def test_kernel_from_array(self):
        """Kernels: test kernel creation from a numpy array."""
        import pygeoprocessing
        import pygeoprocessing.kernels

        # let's use a sharpening mask, just for fun.
        # https://en.wikipedia.org/wiki/Kernel_(image_processing)
        kernel = numpy.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]], dtype=numpy.int32)

        pygeoprocessing.kernels.kernel_from_numpy_array(
            kernel, self.filepath)

        # This both verifies that we can read the kernel with GDAL and that
        # it's the datatype we expect.
        raster_info = pygeoprocessing.get_raster_info(self.filepath)
        self.assertEqual(raster_info['datatype'], gdal.GDT_Int32)

        numpy.testing.assert_equal(
            pygeoprocessing.raster_to_numpy_array(self.filepath),
            kernel)

    def test_kernel_invalid_dimensions(self):
        """Kernels: test error with invalid dimensions."""
        import pygeoprocessing.kernels

        kernel = numpy.array([1, 2, 3])
        with self.assertRaises(ValueError) as cm:
            pygeoprocessing.kernels.kernel_from_numpy_array(
                kernel, self.filepath)

        self.assertIn("array must have exactly 2 dimensions, not 1",
                      str(cm.exception))

    def test_dichotomy(self):
        from pygeoprocessing import kernels

        kernels.dichotomous_kernel(
            self.filepath, max_distance=30, normalize=True)
