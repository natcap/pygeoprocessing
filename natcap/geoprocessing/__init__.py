import subprocess
import logging
import platform
import sys
import tempfile
import os
import atexit

if platform.system() != 'Windows':
    import shutil
    from shutil import WindowsError

LOGGER = logging.getLogger('natcap.geoprocessing')
LOGGER.setLevel(logging.ERROR)

__version__ = 'dev'
build_data = None

if __version__ == 'dev' and build_data == None:
    import imp
    versioning = imp.load_source('natcap.geoprocessing.versioning',
        os.path.join(os.path.dirname(__file__), 'versioning.py'))
    __version__ = versioning.REPO.version
    build_data = versioning._build_data()
    for key, value in sorted(build_data.iteritems()):
        setattr(sys.modules[__name__], key, value)

    del sys.modules[__name__].key
    del sys.modules[__name__].value
