import logging
import os
import warnings

from osgeo import gdal

LOGGER = logging.getLogger(__name__)


def log_warning_if_gdal_will_exhaust_slurm_memory():
    slurm_env_vars = set(k for k in os.environ.keys() if k.startswith('SLURM'))
    if slurm_env_vars:
        gdal_cache_size = gdal.GetCacheMax()
        if gdal_cache_size < 100000:
            # If the cache size is 100,000 or greater, it's assumed to be in
            # bytes.  Otherwise, units are interpreted as megabytes.
            # See gcore/gdalrasterblock.cpp for reference.
            gdal_cache_size_mb = gdal_cache_size
        else:
            gdal_cache_size_mb = gdal_cache_size * 1024 * 1024

        slurm_mem_per_node = os.environ['SLURM_MEM_PER_NODE']
        if gdal_cache_size_mb > int(os.environ['SLURM_MEM_PER_NODE']):
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
