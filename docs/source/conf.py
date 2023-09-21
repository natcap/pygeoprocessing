# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import datetime
import os
import sphinx.ext.apidoc
from pkg_resources import get_distribution

# -- Project information -----------------------------------------------------

project = 'PyGeoprocessing'
copyright = f'{datetime.datetime.today().year}, The Natural Capital Project'
author = 'The Natural Capital Project'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # support google style docstrings
    'sphinx.ext.autosummary',
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = os.path.join(os.path.dirname(__file__), '_static',
                         'pygeoprocessing_logo.jpg')
html_css_files = ['custom.css']


# -- Extension configuration -------------------------------------------------

# Nitpicky=True will make sphinx complain about the types matching python types
# _exactly_.  So "string" will be wrong, but "str" right.  I don't think we
# need to be so picky.
nitpicky = False

DOCS_SOURCE_DIR = os.path.dirname(__file__)
sphinx.ext.apidoc.main([
    '--force',
    '-d', '1',  # max depth for TOC
    '--separate',  # Put docs for each module on their own pages
    '-o', os.path.join(DOCS_SOURCE_DIR, 'api'),
    os.path.join(DOCS_SOURCE_DIR, '..', '..', 'src', 'pygeoprocessing'),
])

release = get_distribution('pygeoprocessing').version
version = '.'.join(release.split('.')[:2])
