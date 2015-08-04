"""
Fetch the subversion data repository based on the configuration
stored in svn_config.json.

To execute:
    $ python get_data.py
"""


import os
import json

from pygeoprocessing.testing import scm

_FILE = os.path.abspath(os.path.dirname(__file__))
SVN_CONFIG = os.path.join(_FILE, 'svn_config.json')

if __name__ == '__main__':
    config = json.load(open(SVN_CONFIG))
    data_dir = os.path.join(_FILE, config['local'])
    scm.checkout_svn(data_dir, config['remote'], config['rev'])

