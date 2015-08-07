"""
Fetch the subversion data repository based on the configuration
stored in svn_config.json.

To execute:
    $ python get_data.py
"""


import os
import json
import imp


# Import pygeoprocessing.testing.scm from source instead of via pygeoprocessing
# Handy, since I might not want to have to install pygeoprocessing to clone
# svn data.  Also, scm doesn't import pygeoprocessing, so that's cool.
scm = imp.load_source('scm', os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '..', 'pygeoprocessing', 'testing', 'scm.py')))

_FILE = os.path.abspath(os.path.dirname(__file__))
SVN_CONFIG = os.path.join(_FILE, 'svn_config.json')

if __name__ == '__main__':
    config = json.load(open(SVN_CONFIG))
    data_dir = os.path.join(_FILE, config['local'])
    scm.checkout_svn(data_dir, config['remote'], config['rev'])

