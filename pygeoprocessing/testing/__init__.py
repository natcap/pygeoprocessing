"""The testing subpackage provides reasonable testing functionality for
building programmatic tests with and of geospatial data.  It provides functions
to generate input data of both raster and vector formats, and offers assertions
to verify the correctness of geospatial outputs.

Most useful features of this package have been exposed at the
pygeoprocessing.testing level.

Select locations have been chosen for their spatial references.  These
references are available in the ``sampledata`` module and al have the prefix
``SRS_``::

    from pygeoprocessing.testing import sampledata

    sampledata.SRS_WILLAMETTE
    sampledata.SRS_COLOMBIA


For writing tests, write them as you normally do!  Assertions are generally
file-based and raise ``AssertionError`` when a failure is encountered.

This example is relatively simplistic, since there will often be many more
assertions you may need to make to be able to test your model
effectively::

    import unittest
    import pygeoprocessing.testing
    import natcap.invest.example_model

    class ExampleTest(unittest.TestCase):
        def test_some_model(self):
            example_args = {
                'workspace_dir': './workspace',
                'arg_1': 'foo',
                'arg_2': 'bar',
            }
            natcap.invest.example_model.execute(example_args)

            # example assertion
            pygeoprocessing.testing.assert_rasters_equal('workspace/raster_1.tif',
                'regression_data/raster_1.tif')

"""


from assertions import assert_almost_equal, \
    assert_rasters_equal, assert_vectors_equal, assert_csv_equal, \
    assert_md5, assert_matrixes, assert_archives, assert_workspace, \
    assert_json, assert_text_equal, assert_files, assert_snapshot
from utils import get_hash, save_workspace, regression, \
    build_regression_archives, snapshot_folder, iterblocks
from sampledata import raster, vector

__all__ = [
    'raster',
    'vector',
    'get_hash',
    'save_workspace',
    'regression',
    'build_regression_archives',
    'snapshot_folder',
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
    'iterblocks',
]
