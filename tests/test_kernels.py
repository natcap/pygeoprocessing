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
        """Kernels: test dichotomous kernel."""
        import pygeoprocessing.kernels

        max_dist = 30
        n_nonzero_pixels = 2821
        for normalize, expected_sum in [(True, 1), (False, n_nonzero_pixels)]:
            pygeoprocessing.kernels.dichotomous_kernel(
                self.filepath, max_distance=max_dist, normalize=normalize)

            kernel_array = pygeoprocessing.raster_to_numpy_array(
                self.filepath)

            self.assertEqual(kernel_array.shape, (max_dist*2+1, max_dist*2+1))
            numpy.testing.assert_allclose(kernel_array.sum(), expected_sum)
            self.assertEqual(numpy.count_nonzero(kernel_array),
                             n_nonzero_pixels)

    def test_exponential_decay(self):
        """Kernels: test exponential decay."""
        import pygeoprocessing.kernels

        max_dist = 30
        expected_distance = 15
        n_nonzero_pixels = 2821
        for normalize, expected_sum in [(True, 1), (False, 838.87195)]:
            pygeoprocessing.kernels.exponential_decay_kernel(
                self.filepath, max_distance=max_dist,
                expected_distance=expected_distance, normalize=normalize)

            kernel_array = pygeoprocessing.raster_to_numpy_array(
                self.filepath)

            self.assertEqual(kernel_array.shape, (max_dist*2+1, max_dist*2+1))
            numpy.testing.assert_allclose(kernel_array.sum(), expected_sum)
            self.assertEqual(numpy.count_nonzero(kernel_array),
                             n_nonzero_pixels)

    def test_linear_decay(self):
        """Kernels: test linear decay."""
        import pygeoprocessing.kernels

        max_dist = 30
        n_nonzero_pixels = 2809
        for normalize, expected_sum in [(True, 1), (False, 942.44055)]:
            pygeoprocessing.kernels.linear_decay_kernel(
                self.filepath, max_distance=max_dist,
                normalize=normalize)

            kernel_array = pygeoprocessing.raster_to_numpy_array(
                self.filepath)

            self.assertEqual(kernel_array.shape, (max_dist*2+1, max_dist*2+1))
            numpy.testing.assert_allclose(kernel_array.sum(), expected_sum)
            self.assertEqual(numpy.count_nonzero(kernel_array),
                             n_nonzero_pixels)

    def test_normal_distribution(self):
        """Kernels: test normal distribution."""
        import pygeoprocessing.kernels

        sigma = 3
        n_std_dev = 10
        n_nonzero_pixels = 2821
        for normalize, expected_sum in [(True, 1), (False, 0.98877025)]:
            pygeoprocessing.kernels.normal_distribution_kernel(
                self.filepath, sigma=10, n_std_dev=3,
                normalize=normalize)

            kernel_array = pygeoprocessing.raster_to_numpy_array(
                self.filepath)

            max_dist = sigma * n_std_dev
            self.assertEqual(kernel_array.shape, (max_dist*2+1, max_dist*2+1))
            numpy.testing.assert_allclose(kernel_array.sum(), expected_sum)
            self.assertEqual(numpy.count_nonzero(kernel_array),
                             n_nonzero_pixels)

    def test_create_distance_decay_kernel_callable(self):
        """Kernels: test kernel creation from function."""
        import pygeoprocessing.kernels

        # should create a sort of bullseye effect.
        # pixel outside max_distance are 0.
        # The kernel's rings alternate between -1 and 1.
        def _my_weird_kernel(dist):
            return ((numpy.floor(dist) % 2) * 2) - 1

        # do the same thing, but as a string
        my_string_kernel = "((floor(dist) % 2) * 2) - 1"

        for function in (_my_weird_kernel, my_string_kernel):
            pygeoprocessing.kernels.create_distance_decay_kernel(
                self.filepath, _my_weird_kernel, max_distance=30,
                normalize=False)

            array = pygeoprocessing.raster_to_numpy_array(self.filepath)
            self.assertEqual(array.shape, (61, 61))
            self.assertEqual(set(numpy.unique(array)), {-1, 0, 1})
