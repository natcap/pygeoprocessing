import dataclasses
import functools
import logging
import math
from typing import Callable
from typing import Union

import numpy
import pygeoprocessing
from osgeo import gdal
from osgeo import osr

FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
LOGGER = logging.getLogger(__name__)

# note that convolve_2d requires that all pixels are valid
#    pixels may not be nodata.
# TODO: are kernels required to be square?
#     From what I can tell, no, they just need to have same num. dimensions.
# TODO: what happens if kernels are not centered on the target pixel?
# TODO: use type hints for these modules and note it in the changelog.

# ok, so the questions needed when we have a kernel is:
#   * how big should the kernel be (pixel radius)
#   * what function should be used to determine pixel values (function of
#     distance from centerpoint)
#   * should the kernel be normalized.


def kernel_from_numpy_array(numpy_array, target_kernel_path):
    """Create a convolution kernel from a numpy array.
    """
    pygeoprocessing.numpy_array_to_raster(
        numpy_array, target_nodata=None, pixel_size=None, origin=None,
        projection_wkt=None, target_path=target_kernel_path)


def dichotomous_kernel(target_kernel_path, max_distance, normalize):
    """Create a binary kernel indicating presence/absence within a distance.

    Given a centerpoint pixel C and an arbitrary pixel P in the target kernel,
    if the distance between C and P exceeds ``max_distance``, the value of P
    will be 0.  The value of P will be 1 otherwise.

    Args:
        target_kernel_path (string): Where the target kernel file should be
            stored.
        max_distance (float): The maximum distance within which pixels should
            indicate presence (1) in the output kernel.  Pixels that are more
            than this distance (in units of pixels) from the center pixel will
            indicate absence (0) in the output kernel.

    Returns:
        ``None``
    """
    create_kernel(
        target_kernel_path, "dist <= max_dist", max_distance,
        normalize=normalize)


# UNA calls this a density kernel
# really, this is quite specific to UNA
def parabolic_decay_kernel(target_kernel_path, max_distance, normalize=True):
    """Create an inverted parabola that reaches 0 at ``max_distance``
    """
    create_kernel(
        target_kernel_path, "0.75 * (1-(dist / max_dist) ** 2)",
        max_distance, normalize=normalize)


def numexpr_kernel(target_kernel_path, max_distance, expression, extras=None):
    import numexpr
    if not extras:
        extras = {}

    def _numexpr_expression(distance_from_center):
        # evaluate a numexpr expression provided by the user
        return numexpr.evaluate(expression)


def exponential_decay_kernel(target_kernel_path, max_distance,
                             normalize=True):
    create_kernel(
        target_kernel_path, "exp(-dist / max_dist)",
        max_distance, normalize=normalize)


def linear_decay_kernel(target_kernel_path, max_distance, normalize=True):
    create_kernel(
        target_kernel_path, "(max_dist - dist) / max_dist",
        max_distance, normalize=normalize)


def normal_distribution_kernel(target_kernel_path, sigma, n_std_dev=3,
                               normalize=True):
    create_kernel(
        target_kernel_path,
        expression=f"(1/(2*pi*{sigma}**2))*exp((-dist**2)/(2*{sigma}**2))",
        max_distance=(sigma * n_std_dev),
        normalize=normalize)


def create_kernel(
        target_kernel_path: str,
        function: Union[str, Callable],
        max_distance: Union[int, float],
        normalize=True):
    """
    Create a kernel raster based on pixel distance from the centerpoint.

    Args:
        target_kernel_path (string): The path to where the target kernel should
            be written on disk.  If this file does not have the suffix
            ``.tif``, it will be added to the filepath.
        function (callable): A python callable that takes as input a
            2D numpy array and returns a 2D numpy array.  The input array will
            contain float32 distances to the centerpoint pixel of the kernel.
        max_distance (float): The maximum distance of kernel values from
            the center point.  Values outside of this distance will be set to
            ``0.0``.
        normalize=False (bool): Whether to normalize the resulting kernel.

    Returns:
        ``None``
    """
    pixel_radius = math.ceil(max_distance)
    kernel_size = pixel_radius * 2 + 1  # allow for a center pixel
    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        target_kernel_path.encode('utf-8'), kernel_size, kernel_size, 1,
        gdal.GDT_Float32, options=[
            'BIGTIFF=IF_SAFER', 'TILED=YES', 'BLOCKXSIZE=256',
            'BLOCKYSIZE=256'])

    # NOTE: We are deliberately NOT setting a coordinate system because it
    # isn't needed.  By omitting this, we're telling GDAL to just create a
    # TIFF.

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_nodata = FLOAT32_NODATA
    kernel_band.SetNoDataValue(kernel_nodata)

    kernel_band = None
    kernel_dataset = None

    kernel_raster = gdal.OpenEx(target_kernel_path, gdal.GA_Update)
    kernel_band = kernel_raster.GetRasterBand(1)
    band_x_size = kernel_band.XSize
    band_y_size = kernel_band.YSize
    running_sum = 0

    # If the user provided a string rather than a callable, assume it's a
    # python expression appropriate for evaling.
    if isinstance(function, str):
        # Avoid recompiling on each iteration.
        code = compile(function, '<string>', 'eval')
        numpy_namespace = {name: getattr(numpy, name) for name in dir(numpy)}

        def function(d):
            result = eval(
                code,
                numpy_namespace,  # globals
                {'dist': d, 'max_dist': max_distance})  # locals
            return result

    for block_data in pygeoprocessing.iterblocks(
            (target_kernel_path, 1), offset_only=True):
        array_xmin = block_data['xoff'] - pixel_radius
        array_xmax = min(
            array_xmin + block_data['win_xsize'],
            band_x_size - pixel_radius)
        array_ymin = block_data['yoff'] - pixel_radius
        array_ymax = min(
            array_ymin + block_data['win_ysize'],
            band_y_size - pixel_radius)

        pixel_dist_from_center = numpy.hypot(
            *numpy.mgrid[
                array_ymin:array_ymax,
                array_xmin:array_xmax])

        valid_pixels = (pixel_dist_from_center <= max_distance)

        kernel = numpy.zeros(pixel_dist_from_center.shape,
                             dtype=numpy.float32)
        kernel[valid_pixels] = function(pixel_dist_from_center[valid_pixels])

        if normalize:
            running_sum += kernel.sum()

        kernel_band.WriteArray(
            kernel,
            yoff=block_data['yoff'],
            xoff=block_data['xoff'])

    kernel_raster.FlushCache()
    kernel_band = None
    kernel_raster = None

    if normalize:
        kernel_raster = gdal.OpenEx(target_kernel_path, gdal.GA_Update)
        kernel_band = kernel_raster.GetRasterBand(1)
        for block_data, kernel_block in pygeoprocessing.iterblocks(
                (target_kernel_path, 1)):
            # divide by sum to normalize
            kernel_block /= running_sum
            kernel_band.WriteArray(
                kernel_block, xoff=block_data['xoff'], yoff=block_data['yoff'])

        kernel_raster.FlushCache()
        kernel_band = None
        kernel_raster = None
