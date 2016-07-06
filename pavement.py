"""PyGeoprocessing pavement script."""
import subprocess
import logging
import glob

import paver.easy
import paver.path

logging.basicConfig(
    format='%(asctime)s %(name)-18s %(levelname)-8s %(message)s',
    level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')
LOGGER = logging.getLogger('pavement')
logging.getLogger('pip').setLevel(logging.ERROR)

_VIRTUAL_ENV_DIR = 'dev_env'


@paver.easy.task
def dev():
    """Build development environment."""
    subprocess.call('virtualenv --system-site-packages %s' % _VIRTUAL_ENV_DIR)
    subprocess.call(r'dev_env\Scripts\python setup.py install')
    subprocess.call(
        'dev_env\\Scripts\\python -c "import pygeoprocessing; '
        'print \'***\\npygeoprocessing version: \' + '
        'pygeoprocessing.__version__ + \'\\n***\'"')
    print (
        "Installed virtualenv launch with:\n.\\%s\\Scripts\\activate" %
        _VIRTUAL_ENV_DIR)

@paver.easy.task
def clean():
    """Clean build environment."""
    folders_to_rm = ['build', 'dist', 'tmp', 'bin', _VIRTUAL_ENV_DIR]

    for folder in folders_to_rm:
        for globbed_dir in glob.glob(folder):
            paver.path.path(globbed_dir).rmtree()
