import functools

from osgeo import gdal

class GDALUseExceptions:
    """Context manager that enables GDAL exceptions and restores state after."""

    def __init__(self):
        pass

    def __enter__(self):
        self.currentUseExceptions = gdal.GetUseExceptions()
        gdal.UseExceptions()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.currentUseExceptions == 0:
            gdal.DontUseExceptions()


def gdal_use_exceptions(func):
    """Decorator that enables GDAL exceptions and restores state after.

    Args:
        func (callable): function to call with GDAL exceptions enabled

    Returns:
        Wrapper function that calls ``func`` with GDAL exceptions enabled
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with GDALUseExceptions():
            return func(*args, **kwargs)
    return wrapper
