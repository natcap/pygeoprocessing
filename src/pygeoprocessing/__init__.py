"""pygeoprocessing: geoprocessing routines for GIS.

__init__ module imports all the geoprocessing functions into this namespace.
"""
from __future__ import absolute_import

import types
import sys

import pkg_resources

from . import geoprocessing

from .geoprocessing_core import calculate_slope

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # Package is not installed, so the package metadata is not available.
    # This should only happen if a package is importable but the package
    # metadata is not, as might happen if someone copied files into their
    # system site-packages or they're importing this package from the CWD.
    raise RuntimeError(
        "Could not load version from installed metadata.\n\n"
        "This is often because the package was not installed properly. "
        "Ensure that the package is installed in a way that the metadata is "
        "maintained.  Calls to ``pip`` and this package's ``setup.py`` "
        "maintain metadata.  Examples include:\n"
        "  * python setup.py install\n"
        "  * python setup.py develop\n"
        "  * pip install <distribution>")

__all__ = ('calculate_slope',)
for attrname in dir(geoprocessing):
    attribute = getattr(geoprocessing, attrname)
    if isinstance(attribute, types.FunctionType):
        __all__ += (attrname,)
        setattr(sys.modules['pygeoprocessing'], attrname, attribute)

# these are bit masks for the known PyGeoprocessing types
UNKNOWN_TYPE = 0
RASTER_TYPE = 1
VECTOR_TYPE = 2
