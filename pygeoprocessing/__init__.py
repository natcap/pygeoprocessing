"""__init__ module for pygeprocessing, imports all the geoprocessing functions
	into the pygeoprocessing namespace"""

import logging
LOGGER = logging.getLogger('pygeoprocessing')
LOGGER.setLevel(logging.ERROR)

from pygeoprocessing.geoprocessing import *
