"""
Assertions for geospatial testing.
"""

import os
import csv
import logging
import json
import glob

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import pygeoprocessing

from . import utils
from . import data_storage

LOGGER = logging.getLogger('natcap.testing.assertions')
TOLERANCE = 7


def assert_almost_equal(value_a, value_b, places=TOLERANCE, msg=None):
    """
    Assert that values a and b are equal out to `places` places.
    If msg is not provided, a standard one will be used.

    Parameters:
        value_a (int or float): The first value to test.
        value_b (int or float): The second value to test.
        places=TOLERANCE (int): The number of places to which the values
            should be tested.
        msg=None (string or None): The assertion message to use if value_a
            and value_b are not found to be equal to `places` places.

    Returns:
        None.

    Raises:
        AssertionError: Raised when the values are not equal out to the
        desired precision.
    """
    if round(abs(value_b - value_a), places) == 0:
        return

    if msg is None:
        msg = "{a} != {b} within {pl} places".format(
            a=value_a, b=value_b, pl=places)
    raise AssertionError(msg)


def assert_rasters_equal(a_uri, b_uri):
    """Tests if datasets a and b are 'almost equal' to each other on a per
    pixel basis

    This assertion method asserts the equality of these raster
    characteristics:
        + Raster height and width

        + The number of layers in the raster

        + Each pixel value, out to a precision of TOLERANCE decimal places if
            the pixel value is a float.

        + Projection

    Args:
        a_uri (string): a URI to a GDAL dataset
        b_uri (string): a URI to a GDAL dataset

    Returns:
        None

    Raises:
        IOError: Raised when one of the input files is not found on disk.

        AssertionError: Raised when the two rasters are found to be not\
        equal to each other.

    """

    LOGGER.debug('Asserting datasets A: %s, B: %s', a_uri, b_uri)

    for uri in [a_uri, b_uri]:
        if not os.path.exists(uri):
            raise IOError('File "%s" not found on disk' % uri)

    a_dataset = gdal.Open(a_uri)
    b_dataset = gdal.Open(b_uri)

    assert a_dataset.RasterXSize == b_dataset.RasterXSize, (
        "x dimensions are different a=%s, second=%s" %
        (a_dataset.RasterXSize, b_dataset.RasterXSize))
    assert a_dataset.RasterYSize == b_dataset.RasterYSize, (
        "y dimensions are different a=%s, second=%s" %
        (a_dataset.RasterYSize, b_dataset.RasterYSize))
    assert a_dataset.RasterCount == b_dataset.RasterCount, (
        "different number of rasters a=%s, b=%s" %
        ((a_dataset.RasterCount, b_dataset.RasterCount)))

    a_sr = osr.SpatialReference()
    a_sr.ImportFromWkt(a_dataset.GetProjection())

    b_sr = osr.SpatialReference()
    b_sr.ImportFromWkt(b_dataset.GetProjection())

    assert bool(a_sr.IsSame(b_sr)) is True, 'Projections differ'

    for band_number in range(1, a_dataset.RasterCount + 1):
        for (a_data, a_block), (b_data, b_block) in zip(
                pygeoprocessing.testing.iterblocks(a_uri, band_number),
                pygeoprocessing.testing.iterblocks(b_uri, band_number)):
            try:
                numpy.testing.assert_array_almost_equal(a_block, b_block,
                                                        verbose=True)
            except AssertionError:
                iterator = numpy.nditer([a_block, b_block],
                                        flags=['multi_index'],
                                        op_flags=['readonly'])
                while not iterator.finished:
                    col = a_data['xoff'] + iterator.multi_index[0]
                    row = a_data['yoff'] + iterator.multi_index[1]
                    pixel_a = a_block[iterator.multi_index]
                    pixel_b = b_block[iterator.multi_index]
                    assert pixel_a == pixel_b, (
                        '{a_val} != {b_val} at col {col}, row {row}').format(
                            a_val=pixel_a,
                            b_val=pixel_b,
                            col=col,
                            row=row
                        )
                    iterator.iternext()


def assert_vectors_equal(a_uri, b_uri):
    """
    Tests if vector datasources are equal to each other.

    This assertion method asserts the equality of these vector
    characteristics:
        + Number of layers in the vector

        + Number of features in each layer

        + Geometry type of the layer

        + Feature geometry type

        + Number of fields in each feature

        + Name of each field

        + Field values for each feature

        + Projection

    Args:
        a_uri (string): a URI to an OGR vector
        b_uri (string): a URI to an OGR vector

    Raises:
        IOError: Raised if one of the input files is not found on disk.
        AssertionError: Raised if the vectors are not found to be equal to\
        one another.

    Returns
        None
    """

    for uri in [a_uri, b_uri]:
        if not os.path.exists(uri):
            raise IOError('File "%s" not found on disk' % uri)

    shape = ogr.Open(a_uri)
    shape_regression = ogr.Open(b_uri)

    # Check that the shapefiles have the same number of layers
    layer_count = shape.GetLayerCount()
    layer_count_regression = shape_regression.GetLayerCount()
    assert layer_count == layer_count_regression, (
        'The shapes DO NOT have the same number of layers')

    for layer_num in range(layer_count):
        # Get the current layer
        layer = shape.GetLayer(layer_num)
        layer_regression = shape_regression.GetLayer(layer_num)
        # Check that each layer has the same number of features
        feat_count = layer.GetFeatureCount()
        feat_count_regression = layer_regression.GetFeatureCount()
        assert feat_count == feat_count_regression, (
            'The layers DO NOT have the same number of features')

        assert layer.GetGeomType() == layer_regression.GetGeomType(), (
            'The layers do not have the same geometry type')

        a_sr = layer.GetSpatialRef()
        b_sr = layer_regression.GetSpatialRef()
        assert bool(a_sr.IsSame(b_sr)) is True, 'Projections differ'

        # Get the first features of the layers and loop through all the
        # features
        feat = layer.GetNextFeature()
        feat_regression = layer_regression.GetNextFeature()
        while feat is not None:
            # Check that the field counts for the features are the same
            layer_def = layer.GetLayerDefn()
            layer_def_regression = layer_regression.GetLayerDefn()
            field_count = layer_def.GetFieldCount()
            field_count_regression = layer_def_regression.GetFieldCount()
            assert field_count == field_count_regression, (
                'The shapes DO NOT have the same number of fields')

            for fld_index in range(field_count):
                # Check that the features have the same field values
                field = feat.GetField(fld_index)
                field_regression = feat_regression.GetField(fld_index)
                assert field == field_regression, ('The field values DO NOT '
                                                   'match')
                # Check that the features have the same field name
                field_ref = feat.GetFieldDefnRef(fld_index)
                field_ref_regression = \
                    feat_regression.GetFieldDefnRef(fld_index)
                field_name = field_ref.GetNameRef()
                field_name_regression = field_ref_regression.GetNameRef()
                assert field_name == field_name_regression, (
                    'The fields DO NOT have the same name')
            # Check that the features have the same geometry
            geom = feat.GetGeometryRef()
            geom_regression = feat_regression.GetGeometryRef()

            feature_fid = feat.GetFID()
            reg_feature_fid = feat_regression.GetFID()
            assert bool(geom.Equals(geom_regression)) is True, (
                'Geometries are not equal in feature %s, '
                'regression feature %s') % (feature_id, reg_feature_id)

            feat = None
            feat_regression = None
            feat = layer.GetNextFeature()
            feat_regression = layer_regression.GetNextFeature()

    shape = None
    shape_regression = None


def assert_csv_equal(a_uri, b_uri, tolerance=TOLERANCE):
    """Tests if csv files a and b are 'almost equal' to each other on a per
    cell basis.  Numeric cells are asserted to be equal out to TOLERANCE decimal
    places.  Other cell types are asserted to be equal.

    Args:
        a_uri (string): a URI to a csv file
        b_uri (string): a URI to a csv file
        tolerance=TOLERANCE (int): The number of places out to which to test
            floating-point cell values.

    Raises:
        AssertionError: Raised when the two CSV files are found to be
        different.

    Returns:
        None
    """

    a = open(a_uri, 'rb')
    b = open(b_uri, 'rb')

    reader_a = csv.reader(a)
    reader_b = csv.reader(b)

    for index, (a_row, b_row) in enumerate(zip(reader_a, reader_b)):
        try:
            assert a_row == b_row, ('Rows differ at row'
                                    '%s: a=%s b=%s' % (index, a_row, b_row))
        except AssertionError:
            for col_index, (a_element, b_element) in enumerate(zip(a_row,
                                                                   b_row)):
                try:
                    a_element = float(a_element)
                    b_element = float(b_element)
                    assert_almost_equal(a_element, b_element, places=tolerance,
                        msg=('Values are significantly different at row %s'
                             'col %s: a=%s b=%s' % (index, col_index,
                                                    a_element,
                                                    b_element)))
                except ValueError:
                    # we know for sure they arenot floats, so compare as
                    # non-floats.
                    assert a_element == b_element, (
                        'Elements differ at row %s col%s: a=%s '
                        'b=%s' % (index, col_index, a_element, b_element))


def assert_md5_equal(uri, regression_hash):
    """Assert the MD5sum of a file against a regression MD5sum.

    This method is a convenience method that uses
    ``natcap.invest.testing.get_hash()`` to determine the MD5sum of the
    file located at `uri`.  It is functionally equivalent to calling::

        assert get_hash(uri) == '<some md5sum>'

    Regression MD5sums can be calculated for you by using
    ``natcap.invest.testing.get_hash()`` or a system-level md5sum program.

    Args:
        uri (string): a string URI to the file to be tested.
        regression_hash (string): a string md5sum to be tested against.

    Raises:
        AssertionError: Raised when the MD5sum of  the file at `uri`
        differs from the provided regression md5sum hash.

    Returns:
        None
    """

    assert utils.get_hash(uri) == regression_hash, "MD5 Hashes differ."


def assert_matrixes(matrix_a, matrix_b, decimal=TOLERANCE):
    """Tests if the input numpy matrices are equal up to `decimal` places.

    This is a convenience function that wraps up required functionality in
    ``numpy.testing``.

    Args:
        matrix_a (numpy.ndarray): a numpy matrix
        matrix_b (numpy.ndarray): a numpy matrix
        decimal (int): an integer of the desired precision.

    Raises:
        AssertionError: Raised when the two matrices are determined to be\
        different.

    Returns:
        None
    """

    numpy.testing.assert_array_almost_equal(matrix_a, matrix_b, decimal)


def assert_archives(archive_1_uri, archive_2_uri):
    """
    Compare the contents of two archived workspaces against each other.

    Takes two archived workspaces, each generated from
    ``build_regression_archives()``, unzips them and
    compares the resulting workspaces against each other.

    Args:
        archive_1_uri (string): a URI to a .tar.gz workspace archive
        archive_2_uri (string): a URI to a .tar.gz workspace archive

    Raises:
        AssertionError: Raised when the two workspaces are found to be\
        different.

    Returns:
        None
    """

    archive_1_folder = pygeoprocessing.geoprocessing.temporary_folder()
    data_storage.extract_archive(archive_1_folder, archive_1_uri)

    archive_2_folder = pygeoprocessing.geoprocessing.temporary_folder()
    data_storage.extract_archive(archive_2_folder, archive_2_uri)

    assert_workspace(archive_1_folder, archive_2_folder)


def assert_workspace(archive_1_folder, archive_2_folder,
                     glob_exclude=''):
    """
    Check the contents of two folders against each other.

    This method iterates through the contents of each workspace folder and
    verifies that all files exist in both folders.  If this passes, then
    each file is compared against each other using
    ``GISTest.assertFiles()``.

    If one of these workspaces includes files that are known to be
    different between model runs (such as logs, or other files that include
    timestamps), you may wish to specify a glob pattern matching those
    filenames and passing it to `glob_exclude`.

    Args:
        archive_1_folder (string): a uri to a folder on disk
        archive_2_folder (string): a uri to a folder on disk
        glob_exclude (string): a string in glob format representing files to
            ignore

    Raises:
        AssertionError: Raised when the two folders are found to have\
        different contents.

    Returns:
        None
    """

    # uncompress the two archives

    archive_1_files = []
    archive_2_files = []
    for files_list, workspace in [
            (archive_1_files, archive_1_folder),
            (archive_2_files, archive_2_folder)]:
        for root, dirs, files in os.walk(workspace):
            root = root.replace(workspace + os.sep, '')
            ignored_files = glob.glob(glob_exclude)
            for filename in files:
                if filename not in ignored_files:
                    full_path = os.path.join(root, filename)
                    files_list.append(full_path)

    archive_1_files = sorted(archive_1_files)
    archive_2_files = sorted(archive_2_files)

    archive_1_size = len(archive_1_files)
    archive_2_size = len(archive_2_files)
    if archive_1_size != archive_2_size:
        # find out which archive had more files.
        archive_1_files = [x.replace(archive_1_folder, '')
                           for x in archive_1_files]
        archive_2_files = [x.replace(archive_2_folder, '')
                           for x in archive_2_files]
        missing_from_archive_1 = list(set(archive_2_files) -
                                      set(archive_1_files))
        missing_from_archive_2 = list(set(archive_1_files) -
                                      set(archive_2_files))
        raise AssertionError('Elements missing from A:%s, from B:%s' %
                             (missing_from_archive_1, missing_from_archive_2))
    else:
        # archives have the same number of files that we care about
        for file_1, file_2 in zip(archive_1_files, archive_2_files):
            file_1_uri = os.path.join(archive_1_folder, file_1)
            file_2_uri = os.path.join(archive_2_folder, file_2)
            LOGGER.debug('Checking %s, %s', file_1, file_2)
            assert_file_contents_equal(file_1_uri, file_2_uri)


def assert_json_equal(json_1_uri, json_2_uri):
    """Assert two JSON files against each other.

    The two JSON files provided will be opened, read, and their
    contents will be asserted to be equal.  If the two are found to be
    different, the diff of the two files will be printed.

    Args:
        json_1_uri (string): a uri to a JSON file.
        json_2_uri (string): a uri to a JSON file.

    Raises:
        AssertionError: Raised when the two JSON objects differ.

    Returns:
        None
    """

    dict_1 = json.loads(open(json_1_uri).read())
    dict_2 = json.loads(open(json_2_uri).read())

    assert dict_1 == dict_2, 'JSON objects differ: %s\n%s' % (dict_1, dict_2)


def assert_text_equal(text_1_uri, text_2_uri):
    """Assert that two text files are equal

    This comparison is done line-by-line.

    Args:
        text_1_uri (string): a python string uri to a text file. \
            Considered the file to be tested.
        text_2_uri (string): a python string uri to a text file. \
            Considered the regression file.

    Raises:
        AssertionError: Raised when a line differs in the two files.

    Returns:
        None
    """

    def lines(f):
        """Return a list of lines in the opened file."""
        return [line for line in open(f)]
    for index, (a_line, b_line) in enumerate(zip(lines(text_1_uri),
                                                 lines(text_2_uri))):
        assert a_line == b_line, ('Line %s in %s does not match regression '
                                  'file. Output  "%s" Regression "%s"') % (
                                      index, text_1_uri, a_line, b_line)


def assert_file_contents_equal(file_1_uri, file_2_uri):
    """Assert two files are equal.

    If the extension of the provided file is recognized, the relevant
    filetype-specific function is called and a more detailed check of the
    file can be done.  If the extension is not recognized, the MD5sums of
    the two files are compared instead.

    Known extensions: ``json``, ``tif``, ``shp``, ``csv``, ``txt``,
    ``html``

    Args:
        file_1_uri (string): a string URI to a file on disk.
        file_2_uri (string): a string URI to a file on disk.

    Raises:
        AssertionError: Raised when one of the input files does not exist,\
        when the extensions of the input files differ, or if the two files\
        are found to differ.

    Returns:
        None
    """

    for uri in [file_1_uri, file_2_uri]:
        assert os.path.exists(uri) is True, 'File not found: %s' % uri

    # assert the extensions are the same
    file_1_ext = os.path.splitext(file_1_uri)[1]
    file_2_ext = os.path.splitext(file_2_uri)[1]
    assert file_1_ext == file_2_ext, 'Extensions differ: %s, %s' % (file_1_ext,
                                                                    file_2_ext)

    assert_funcs = {
        '.json': assert_json,
        '.tif': assert_rasters_equal,
        '.shp': assert_vectors_equal,
        '.csv': assert_csv_equal,
        '.txt': assert_text_equal,
        '.html': assert_text_equal,
    }

    try:
        assert_funcs[file_1_ext](file_1_uri, file_2_uri)
    except KeyError:
        # When we're given an extension we don't have a function for, assert
        # the MD5s.
        file_1_md5 = utils.get_hash(file_1_uri)
        file_2_md5 = utils.get_hash(file_2_uri)
        assert file_1_md5 == file_2_md5, 'Files %s and %s differ (MD5sum)' % (
            file_1_uri, file_2_uri)


def assert_snapshot(folder, snapshot_file):
    """
    Assert all files in a directory according to the snapshot file recorded
    by `natcap.testing.utils.snapshot_folder()`.  Any files not in the
    snapshot file are ignored.

    Parameters:
        folder (string): the path to the folder to recurse through and check
            md5sums for.
        snapshot_file (string): the path to the snapshot file to use.

    Raises:
        AssertionError: when a nonmatching md5sum is found.
    """

    snapshot = open(snapshot_file)
    env_params = {}
    for line in snapshot:
        # a blank line signals the end of the env section
        if line.strip() == '':
            break
        name, value = line.split('=')
        env_params[name.strip()] = value.strip()

    files = {}
    for line in snapshot:
        filename, md5sum = line.split('::')
        files[filename.strip()] = md5sum.strip()

    missing_files = []
    nonmatching_files = []
    for filepath, expected_md5sum in files.iteritems():
        full_filepath = os.path.join(folder, filepath)
        try:
            current_md5sum = utils.get_hash(full_filepath)
        except IOError:
            # When the file we're looking for doesn't exist
            missing_files.append(full_filepath)
            continue

        if current_md5sum != expected_md5sum:
            nonmatching_files.append(filepath)

    if len(missing_files) != 0:
        if len(missing_files) == len(files):
            raise AssertionError((
                'All files recorded in the snapshot are missing.  Are you '
                'testing against the right folder?  Testing {test_dir}. '
                'Snapshot taken from {snap_dir}.').format(
                    test_dir=folder, snap_dir=env_params['orig_workspace']))
        raise AssertionError(
            ('{num_missing} files out of {num_files} are '
             'missing.').format(num_missing=len(missing_files),
                                num_files=len(files)))

    if len(nonmatching_files) != 0:
        raise AssertionError((
            '{num_err} files out of {num_files} have differing '
            'md5sums: {files}').format(num_err=len(nonmatching_files),
                                       num_files=len(files),
                                       files=nonmatching_files))
