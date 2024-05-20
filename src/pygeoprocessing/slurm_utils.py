import logging
import os
import warnings

from osgeo import gdal

from .geoprocessing_core import gdal_use_exceptions

LOGGER = logging.getLogger(__name__)


@gdal_use_exceptions
def log_warning_if_gdal_will_exhaust_slurm_memory():
    """Warn if GDAL's cache max size exceeds SLURM's allocated memory.

    This function checks GDAL's max cache (set by the ``GDAL_CACHEMAX``
    environment variable or ``gdal.SetCacheMax()`` function) against the amount
    of memory available to the current SLURM node, identified by the
    ``SLURM_MEM_PER_NODE`` environment variable.

    This function uses a primitive check of environment variables to verify
    whether this function is operating on a SLURM node.  If any environment
    variables have the prefix ``SLURM``, we assume we are running within a
    SLURM environment.

    If the GDAL cache size may exceed the SLURM available memory, then a
    warning is issued.  If ``logging.captureWarnings(True)`` is in effect, a
    warning is logged with the logging system.  Otherwise, the warnings system
    is used directly.
    """
    if {'SLURM_MEM_PER_NODE'}.issubset(set(os.environ.keys())):
        gdal_cache_size = gdal.GetCacheMax()
        if gdal_cache_size < 100000:
            # If the cache size is 100,000 or greater, it's assumed to be in
            # bytes.  Otherwise, units are interpreted as megabytes.
            # See gcore/gdalrasterblock.cpp for reference.
            gdal_cache_size_mb = gdal_cache_size
        else:
            # Convert from bytes to megabytes
            gdal_cache_size_mb = gdal_cache_size / 1024 / 1024

        slurm_mem_per_node = os.environ['SLURM_MEM_PER_NODE']
        if gdal_cache_size_mb > int(slurm_mem_per_node):
            message = (
                "GDAL's cache max exceeds the memory SLURM has "
                "allocated for this node. The process will probably be "
                "killed by the kernel's oom-killer. "
                f"GDAL_CACHEMAX={gdal_cache_size} (interpreted as "
                f"{gdal_cache_size_mb} MB), "
                f"SLURM_MEM_PER_NODE={slurm_mem_per_node}")

            # If logging is not configured to capture warnings, send the output
            # to the usual warnings stream.  If logging is configured to
            # capture warnings, log the warning as normal.
            # This appears to be the easiest way to identify whether we're in a
            # logging.captureWarnings(True) block.
            if logging._warnings_showwarning is None:
                warnings.warn(message)
            else:
                LOGGER.warning(message)
