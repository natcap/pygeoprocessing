"""A library of kernels for use in ``pygeoprocessing.convolve_2d``.

``pygeoprocessing.convolve_2d`` has some stringent requirements for its kernels
that require a thorough understanding of GDAL and GeoTiffs to be able to use,
including:

    * Kernels must be GDAL-readable rasters, preferably GeoTiffs
    * Kernels must not be striped, and, for more efficient disk I/O, should be
      tiled.
    * The pixel values of a kernel may not represent nodata (GDAL's internal
      way of indicating invalid pixel data)
    * Pixel values must be finite, real numbers.  That is, they may not
      represent positive or negative infinity or NaN.

The functions in this module represent kernels that we have found to be
commonly used, including:

    * Dichotomous kernel: :meth:`dichotomous_kernel`
    * Gaussian (normal) decay kernel: :meth:`normal_distribution_kernel`
    * Linear decay kernel: :meth:`linear_decay_kernel`
    * Exponential decay kernel: :meth:`exponential_decay_kernel`

Additionally, the user may define their own kernel using the helper functions
included here:

    * :meth:`kernel_from_numpy_array`, for kernels where the kernel array is
      already available as a ``numpy`` array, such as those used in common
      image processing operations like sharpening and edge detection.
      https://en.wikipedia.org/wiki/Kernel_(image_processing)
    * :meth:`create_distance_decay_kernel`, for kernels that are a function of
      distance from the center pixel
"""
import logging
import math
from typing import Callable
from typing import Text
from typing import Union

import numpy
import pygeoprocessing
from numpy.typing import ArrayLike
from osgeo import gdal

from .geoprocessing_core import gdal_use_exceptions

FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
LOGGER = logging.getLogger(__name__)


def kernel_from_numpy_array(
        numpy_array: ArrayLike, target_kernel_path: Text) -> None:
    """Create a convolution kernel from a numpy array.

    Args:
        numpy_array: A 2-dimensional numpy array to convert into a raster.
        target_kernel_path: The path to where the kernel should be written.

    Returns:
        ``None``
    """
    n_dimensions = numpy.array(numpy_array).ndim
    if n_dimensions != 2:
        raise ValueError(
            "The kernel array must have exactly 2 dimensions, not "
            f"{n_dimensions}")

    # The kernel is technically a TIFF, not a GeoTiff.
    pygeoprocessing.numpy_array_to_raster(
        numpy_array, target_nodata=None, pixel_size=None, origin=None,
        projection_wkt=None, target_path=target_kernel_path)


def dichotomous_kernel(
        target_kernel_path: Text,
        max_distance: Union[int, float],
        normalize: bool = True) -> None:
    """Create a binary kernel indicating presence/absence within a distance.

    This is equivalent to ``int(dist <= max_distance)`` for each pixel in the
    kernel, where ``dist`` is the euclidean distance of the pixel's centerpoint
    to the centerpoint of the center pixel.

    Args:
        target_kernel_path: The path to where the kernel will be written.
            Must have a file extension of ``.tif``.
        max_distance: The distance threshold, in pixels. Kernel
            pixels that are greater than ``max_distance`` from the centerpoint
            of the kernel will have values of ``0.0``.
        normalize: Whether to normalize the kernel.

    Returns:
        ``None``
    """
    def _dichotomous(dist):
        return dist <= max_distance

    create_distance_decay_kernel(
        target_kernel_path=target_kernel_path,
        distance_decay_function=_dichotomous,
        max_distance=max_distance,
        normalize=normalize
    )


def exponential_decay_kernel(
        target_kernel_path: Text,
        max_distance: Union[int, float],
        expected_distance: Union[int, float],
        normalize: bool = True) -> None:
    """Create an exponential decay kernel.

    This is equivalent to ``e**(-dist / expected_distance)`` for each pixel in
    the kernel, where ``dist`` is the euclidean distance of the pixel's
    centerpoint to the centerpoint of the center pixel and
    ``expected_distance`` represents the distance at which the kernel's values
    reach ``1/e``.  The kernel will continue to have nonzero values out to
    ``max_distance``.

    Args:
        target_kernel_path: The path to where the kernel will be written.
            Must have a file extension of ``.tif``.
        max_distance: The maximum distance of the kernel, in pixels. Kernel
            pixels that are greater than ``max_distance`` from the centerpoint
            of the kernel will have values of ``0.0``.
        expected_distance: The distance, in pixels, from the centerpoint at
            which decayed values will equal ``1/e``.
        normalize: Whether to normalize the kernel.

    Returns:
        ``None``
    """
    def _exponential_decay(dist):
        return numpy.exp(-dist / expected_distance)

    create_distance_decay_kernel(
        target_kernel_path=target_kernel_path,
        distance_decay_function=_exponential_decay,
        max_distance=max_distance,
        normalize=normalize
    )


def linear_decay_kernel(
        target_kernel_path: Text,
        max_distance: Union[int, float],
        normalize: bool = True) -> None:
    """Create a linear decay kernel.

    This is equivalent to ``(max_distance - dist) / max_distance`` for each
    pixel in the kernel, where ``dist`` is the euclidean distance between the
    centerpoint of the current pixel and the centerpoint of the center pixel in
    the kernel.

    Args:
        target_kernel_path: The path to where the kernel will be written.
            Must have a file extension of ``.tif``.
        max_distance: The maximum distance of the kernel, in pixels. Kernel
            pixels that are greater than ``max_distance`` from the centerpoint
            of the kernel will have values of ``0.0``.
        normalize: Whether to normalize the kernel.

    Returns:
        ``None``
    """
    def _linear_decay(dist):
        return (max_distance - dist) / max_distance

    create_distance_decay_kernel(
        target_kernel_path=target_kernel_path,
        distance_decay_function=_linear_decay,
        max_distance=max_distance,
        normalize=normalize
    )


def normal_distribution_kernel(
        target_kernel_path: Text,
        sigma: Union[int, float],
        n_std_dev: Union[int, float] = 3,
        normalize: bool = True):
    """Create an decay kernel following a normal distribution.

    This is equivalent to
    ``(1/(2*pi*sigma**2))*(e**((-dist**2)/(2*sigma**2)))`` for each pixel,
    where ``dist`` is the euclidean distance between the current pixel and the
    centerpoint of the center pixel.

    Args:
        target_kernel_path: The path to where the kernel will be written.
            Must have a file extension of ``.tif``.
        sigma: The width (in pixels) of a standard deviation.
        n_std_dev: The number of standard deviations to include in the kernel.
            The kernel will have values of 0 when at a distance of
            ``(sigma * n_std_dev)`` away from the centerpoint.
        normalize: Whether to normalize the kernel.

    Returns:
        ``None``
    """
    def _normal_decay(dist):
        return (1 / (2 * numpy.pi * sigma ** 2)) * numpy.exp(
            (-dist ** 2) / (2 * sigma ** 2))

    create_distance_decay_kernel(
        target_kernel_path=target_kernel_path,
        distance_decay_function=_normal_decay,
        max_distance=(sigma * n_std_dev),
        normalize=normalize
    )


@gdal_use_exceptions
def create_distance_decay_kernel(
        target_kernel_path: str,
        distance_decay_function: Union[str, Callable],
        max_distance: Union[int, float],
        normalize=True):
    """
    Create a kernel raster based on pixel distance from the centerpoint.

    Args:
        target_kernel_path (string): The path to where the target kernel should
            be written on disk.  If this file does not have the suffix
            ``.tif``, it will be added to the filepath.
        distance_decay_function (callable or str): A python callable that takes as
            input a single 1D numpy array and returns a 1D numpy array.  The
            input array will contain float32 distances to the centerpoint pixel
            of the kernel, in units of pixels. If a ``str``, then it must be a
            python expression using the local variables:

                * ``dist`` - a 1D numpy array of distances from the
                  centerpoint.
                * ``max_dist`` - a float indicating the max distance.

        max_distance (float): The maximum distance of kernel values from
            the center point.  Values outside of this distance will be set to
            ``0.0``.
        normalize=False (bool): Whether to normalize the resulting kernel.

    Returns:
        ``None``
    """
    apothem = math.floor(max_distance)
    kernel_size = apothem * 2 + 1  # allow for a center pixel
    assert kernel_size % 2 == 1
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
    if isinstance(distance_decay_function, str):
        # Avoid recompiling on each iteration.
        code = compile(distance_decay_function, '<string>', 'eval')
        numpy_namespace = {name: getattr(numpy, name) for name in dir(numpy)}

        def distance_decay_function(d):
            result = eval(
                code,
                numpy_namespace,  # globals
                {'dist': d, 'max_dist': max_distance})  # locals
            return result

    for block_data in pygeoprocessing.iterblocks(
            (target_kernel_path, 1), offset_only=True):
        array_xmin = block_data['xoff'] - apothem
        array_xmax = min(
            array_xmin + block_data['win_xsize'],
            band_x_size - apothem)
        array_ymin = block_data['yoff'] - apothem
        array_ymax = min(
            array_ymin + block_data['win_ysize'],
            band_y_size - apothem)

        pixel_dist_from_center = numpy.hypot(
            *numpy.mgrid[
                array_ymin:array_ymax,
                array_xmin:array_xmax])

        valid_pixels = (pixel_dist_from_center <= max_distance)

        kernel = numpy.zeros(pixel_dist_from_center.shape,
                             dtype=numpy.float32)
        kernel[valid_pixels] = distance_decay_function(
            pixel_dist_from_center[valid_pixels])

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
