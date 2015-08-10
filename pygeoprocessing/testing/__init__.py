"""The testing subpackage provides reasonable testing functionality for
building programmatic tests with and of geospatial data.  It provides functions
to generate input data of both raster and vector formats, and offers assertions
to verify the correctness of geospatial outputs.

Most useful features of this package have been exposed at the
pygeoprocessing.testing level.

Select locations have been chosen for their spatial references.  These
references are available in the `sampledata` module and al have the prefix
`SRS_`."""

from assertions import assert_equal, assert_almost_equal, \
    assert_rasters_equal, assert_vectors_equal, assert_csv_equal, \
    assert_md5, assert_matrixes, assert_archives, assert_workspace, \
    assert_json, assert_text_equal, assert_files, assert_snapshot
from utils import get_hash, save_workspace, regression, \
    build_regression_archives, snapshot_folder
from sampledata import raster, vector

__all__ = [
    'raster',
    'vector',
    'get_hash',
    'save_workspace',
    'regression',
    'build_regression_archives',
    'snapshot_folder',
    'assert_equal',
    'assert_almost_equal',
    'assert_rasters_equal',
    'assert_vectors_equal',
    'assert_csv_equal',
    'assert_md5',
    'assert_matrixes',
    'assert_archives',
    'assert_workspace',
    'assert_json',
    'assert_text_equal',
    'assert_files',
    'assert_snapshot',
]
