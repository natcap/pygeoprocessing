import unittest
import tempfile
import os
import shutil
import glob
import hashlib

import numpy
from osgeo import gdal
from osgeo import ogr
from shapely.geometry import Point, LineString

from pygeoprocessing.testing import scm
import pygeoprocessing.testing as testing
from pygeoprocessing.testing import utils
from pygeoprocessing.testing import sampledata
import pygeoprocessing as raster_utils
import pygeoprocessing

SVN_LOCAL_DIR = scm.load_config(os.path.join(os.path.dirname(__file__),
                                             'svn_config.json'))['local']
POLLINATION_DATA = os.path.join(SVN_LOCAL_DIR, 'pollination', 'samp_input')
SAMPLE_RASTERS = os.path.join(SVN_LOCAL_DIR, 'sample_rasters')
SAMPLE_VECTORS = os.path.join(SVN_LOCAL_DIR, 'sample_vectors')
TESTING_REGRESSION = os.path.join(SVN_LOCAL_DIR, 'testing_regression')
CARBON_DATA = os.path.join(SVN_LOCAL_DIR, 'carbon', 'input')
REGRESSION_ARCHIVES = os.path.join(SVN_LOCAL_DIR, 'data_storage', 'regression')
WRITING_ARCHIVES = os.path.join(SVN_LOCAL_DIR, 'test_writing')
TEST_INPUT = os.path.join(SVN_LOCAL_DIR, 'data_storage', 'test_input')
TEST_OUT = os.path.join(SVN_LOCAL_DIR, 'test_out')
BASE_DATA = os.path.join(SVN_LOCAL_DIR, 'base_data')
REGRESSION_INPUT = os.path.join(SVN_LOCAL_DIR, 'testing_regression')


class DataStorageUtilsTest(unittest.TestCase):
    def test_digest_file(self):
        """Test for digesting and comparing files and folders"""
        # make a file in tempdir
        file_handle, new_file = tempfile.mkstemp()
        os.close(file_handle)
        open_file = open(new_file, 'w')
        open_file.write('foobarbaz')
        open_file.close()

        first_md5sum = utils.digest_file_list([new_file])

        # move it to a different filename
        moved_file = new_file + '.txt'
        shutil.move(new_file, moved_file)
        second_md5sum = utils.digest_file_list([moved_file])

        # verify md5sum stays the same across all cases.
        self.assertEqual(first_md5sum, second_md5sum)

        # move it to a new folder, verify md5sum
        dirname = tempfile.mkdtemp()
        shutil.move(moved_file, os.path.join(dirname,
                                             os.path.basename(moved_file)))
        dir_md5sum = utils.digest_folder(dirname)
        self.assertEqual(first_md5sum, dir_md5sum)

        file_handle, new_file_2 = tempfile.mkstemp()
        os.close(file_handle)
        with open(new_file_2, 'w') as new_file:
            new_file.write('hello world!')

        new_file_md5sum = utils.digest_file_list([new_file_2])
        self.assertNotEqual(new_file_md5sum, first_md5sum)

        os.remove(new_file_2)
        shutil.rmtree(dirname)


class GISTestTester(unittest.TestCase):
    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_raster_assertion_fileio(self):
        """Verify correct behavior for assertRastersEqual"""

        # check that IOError is raised if a file is not found.
        raster_on_disk = os.path.join(TEST_INPUT, 'landuse_cur_200m.tif')
        self.assertRaises(IOError, testing.assert_rasters_equal,
                          'file_not_on_disk', 'other_file_not_on_disk',
                          tolerance=1e-9)
        self.assertRaises(IOError, testing.assert_rasters_equal,
                          'file_not_on_disk', raster_on_disk, tolerance=1e-9)
        self.assertRaises(IOError, testing.assert_rasters_equal,
                          raster_on_disk, 'file_not_on_disk',
                          tolerance=1e-9)
        testing.assert_rasters_equal(raster_on_disk, raster_on_disk,
                                     tolerance=1e-9)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_raster_assertion_failed_cleanup(self):
        """If raster assertion fails, we should still be able to remove it."""
        # create a raster on disk that we intend to remove.
        tempdir = tempfile.mkdtemp()
        lulc_current_base = os.path.join(TEST_INPUT, 'landuse_cur_200m.tif')
        lulc_copied = os.path.join(tempdir, 'landuse_copied.tif')
        lulc_future = os.path.join(TEST_INPUT, 'landuse_fut_200m.tif')

        pygeoprocessing.geoprocessing.tile_dataset_uri(lulc_current_base,
                                                       lulc_copied, 256)
        try:
            self.assertRaises(
                AssertionError, pygeoprocessing.testing.assert_rasters_equal,
                lulc_copied, lulc_future, tolerance=1e-9)
            shutil.rmtree(tempdir)
        except OSError as file_not_removed_error:
            # This should technically be a WindowsError, which is a subclass of
            # OSError.  If we can't remove the copied raster because the test
            # has it open, then this test fails (and the raster will sit there
            # since we won't be able to remove it until after python quits).
            raise AssertionError(('Raster objects not cleaned up properly: '
                                  '%s') % file_not_removed_error)
        except AssertionError as test_failure_error:
            # If assertRaises raises an assertionError, then something really
            # did go wrong!  Clean up the workspace and raise the exception.
            shutil.rmtree(tempdir)
            raise test_failure_error

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_vector_assertion_failed_cleanup(self):
        """If vector assertion fails, we should still be able to remove it."""
        # create a vector on disk that we intend to remove.
        tempdir = tempfile.mkdtemp()
        sample_vector_base = os.path.join(SAMPLE_VECTORS, 'harv_samp_cur.shp')
        sample_vector_copy = os.path.join(tempdir, 'sample_vector.shp')
        comparison_vector = os.path.join(SAMPLE_VECTORS, 'harv_samp_fut.shp')

        pygeoprocessing.copy_datasource_uri(sample_vector_base,
                                            sample_vector_copy)
        try:
            self.assertRaises(
                AssertionError, pygeoprocessing.testing.assert_vectors_equal,
                sample_vector_copy, comparison_vector, field_tolerance=1e-9)
            shutil.rmtree(tempdir)
        except OSError as file_not_removed_error:
            # This should technically be a WindowsError, which is a subclass of
            # OSError.  If we can't remove the copied vector because the test
            # has it open, then this test fails (and the vector will sit there
            # since we won't be able to remove it until after python quits).
            raise AssertionError(('Vector objects not cleaned up properly: '
                                  '%s') % file_not_removed_error)
        except AssertionError as test_failure_error:
            # If assertRaises raises an assertionError, then something really
            # did go wrong!  Clean up the workspace and raise the exception.
            shutil.rmtree(tempdir)
            raise test_failure_error

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_raster_assertion_files_equal(self):
        """Verify when rasters are, in fact, equal."""
        temp_folder = raster_utils.temporary_folder()
        new_raster = os.path.join(temp_folder, 'new_file.tif')

        source_file = os.path.join(TEST_INPUT, 'landuse_cur_200m.tif')
        shutil.copyfile(source_file, new_raster)
        testing.assert_rasters_equal(source_file, new_raster, tolerance=1e-9)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_raster_assertion_different_dims(self):
        """Verify when rasters are different"""
        source_raster = os.path.join(TEST_INPUT, 'landuse_cur_200m.tif')
        different_raster = os.path.join(SAMPLE_RASTERS, 'lulc_samp_cur')
        self.assertRaises(AssertionError, testing.assert_rasters_equal,
                          source_raster, different_raster, tolerance=1e-9)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_raster_assertion_different_values(self):
        """Verify when rasters have different values"""
        lulc_cur_raster = os.path.join(TEST_INPUT, 'landuse_cur_200m.tif')
        lulc_fut_raster = os.path.join(TEST_INPUT, 'landuse_fut_200m.tif')
        self.assertRaises(AssertionError, testing.assert_rasters_equal,
                          lulc_cur_raster, lulc_fut_raster, tolerance=1e-9)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_vector_assertion_fileio(self):
        """Verify correct behavior for assertVectorsEqual"""
        vector_on_disk = os.path.join(TEST_INPUT, 'farms.dbf')
        self.assertRaises(IOError, testing.assert_rasters_equal,
                          'file_not_on_disk', 'other_file_not_on_disk',
                          tolerance=1e-9)
        self.assertRaises(IOError, testing.assert_rasters_equal,
                          'file_not_on_disk', vector_on_disk, tolerance=1e-9)
        self.assertRaises(IOError, testing.assert_rasters_equal,
                          vector_on_disk, 'file_not_on_disk', tolerance=1e-9)
        testing.assert_vectors_equal(vector_on_disk, vector_on_disk,
                                     field_tolerance=1e-9)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_vector_assertion_files_equal(self):
        """Verify when vectors are equal."""
        temp_folder = raster_utils.temporary_folder()
        for vector_file in glob.glob(os.path.join(TEST_INPUT, 'farms.*')):
            base_name = os.path.basename(vector_file)
            new_file = os.path.join(temp_folder, base_name)
            shutil.copyfile(vector_file, new_file)

        sample_shape = os.path.join(TEST_INPUT, 'farms.shp')
        copied_shape = os.path.join(temp_folder, 'farms.shp')
        testing.assert_vectors_equal(sample_shape, copied_shape,
                                     field_tolerance=1e-9)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_vectors_different_attributes(self):
        """Verify when two vectors have different attributes"""
        base_file = os.path.join(TEST_INPUT, 'farms.shp')
        different_file = os.path.join(REGRESSION_ARCHIVES, 'farms.shp')

        self.assertRaises(AssertionError, testing.assert_vectors_equal,
                          base_file, different_file, field_tolerance=1e-9)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_vectors_very_different(self):
        """Verify when two vectors are very, very different."""
        base_file = os.path.join(TEST_INPUT, 'farms.shp')
        different_file = os.path.join(SAMPLE_VECTORS, 'harv_samp_cur.shp')
        self.assertRaises(AssertionError, testing.assert_vectors_equal,
                          base_file, different_file, field_tolerance=1e-9)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_csv_assertion_fileio(self):
        bad_file_1 = 'aaa'
        bad_file_2 = 'bbbbb'
        good_file = os.path.join(TEST_INPUT, 'Guild.csv')

        self.assertRaises(IOError, testing.assert_csv_equal, bad_file_1,
                          bad_file_2, tolerance=1e-9)
        self.assertRaises(IOError, testing.assert_csv_equal, bad_file_1,
                          good_file, tolerance=1e-9)
        self.assertRaises(IOError, testing.assert_csv_equal, good_file,
                          bad_file_2, tolerance=1e-9)
        testing.assert_csv_equal(good_file, good_file, tolerance=1e-9)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_csv_assertion_fails(self):
        sample_file = os.path.join(TEST_INPUT, 'Guild.csv')
        different_file = os.path.join(TEST_INPUT, 'LU.csv')

        self.assertRaises(AssertionError, testing.assert_csv_equal,
                          sample_file, different_file, tolerance=1e-9)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_md5_same(self):
        """Check that the MD5 is equal."""
        test_file = os.path.join(SAMPLE_VECTORS, 'harv_samp_cur.shp')
        md5_sum = testing.digest_file(test_file)
        testing.assert_md5_equal(test_file, md5_sum)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_md5_different(self):
        """Check that the MD5 is equal."""
        test_file = os.path.join(SAMPLE_VECTORS, 'harv_samp_cur.shp')
        md5_sum = 'bad md5sum!'

        self.assertRaises(AssertionError, testing.assert_md5_equal, test_file,
                          md5_sum)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_workspaces_ignore(self):
        """Check that ignoring certain files works as expected."""
        new_folder = os.path.join(raster_utils.temporary_folder(), 'foo')
        shutil.copytree(TEST_INPUT, new_folder)

        # make a file in TEST_INPUT by opening a writeable file there.
        copied_filepath = os.path.join(TEST_INPUT, 'test_file.txt')
        fp = open(copied_filepath, 'w')
        fp.close()

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_json_same(self):
        """Check that asserting equal json objects passes."""
        json_path = os.path.join(TESTING_REGRESSION, 'sample_json.json')
        testing.assert_json_equal(json_path, json_path)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_json_different(self):
        """Check that asserting different json objects fails"""
        json_path = os.path.join(TESTING_REGRESSION, 'sample_json.json')
        json_path_new = os.path.join(TESTING_REGRESSION, 'sample_json_2.json')
        self.assertRaises(AssertionError, testing.assert_json_equal, json_path,
                          json_path_new)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_assert_text_same(self):
        """Check that asserting two identical text files passes"""
        sample_file = os.path.join(REGRESSION_INPUT, 'sample_text_file.txt')
        testing.assert_text_equal(sample_file, sample_file)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_assert_text_different(self):
        """Check that asserting two different text files fails."""
        sample_file = os.path.join(REGRESSION_INPUT, 'sample_text_file.txt')
        regression_file = os.path.join(REGRESSION_INPUT, 'sample_json.json')
        self.assertRaises(AssertionError, testing.assert_text_equal,
                          sample_file, regression_file)

    def test_raster_equality_to_tolerance(self):
        """Verify assert_rasters_equal asserts out to the given tolerance."""
        reference = sampledata.SRS_COLOMBIA
        filename_a = pygeoprocessing.temporary_filename()
        sampledata.create_raster_on_disk(
            [numpy.array([[0.1234567]])], reference.origin,
            reference.projection, -1, reference.pixel_size(30),
            datatype=gdal.GDT_Float32, format='GTiff', filename=filename_a)

        filename_b = pygeoprocessing.temporary_filename()
        sampledata.create_raster_on_disk(
            [numpy.array([[0.123]])], reference.origin,
            reference.projection, -1, reference.pixel_size(30),
            datatype=gdal.GDT_Float32, format='GTiff', filename=filename_b)

        # 0.005 is greater than the difference between the pixel values in
        # these two matrices.  We're only testing that we can use a
        # user-defined tolerance here.
        pygeoprocessing.testing.assert_rasters_equal(filename_a, filename_b,
                                                     tolerance=0.005)

    def test_raster_inequality_to_tolerance(self):
        """Verify assert_rasters_equal fails if inequal past a tolerance."""
        reference = sampledata.SRS_COLOMBIA
        filename_a = pygeoprocessing.temporary_filename()
        sampledata.create_raster_on_disk(
            [numpy.array([[0.1234567]])], reference.origin,
            reference.projection, -1, reference.pixel_size(30),
            datatype=gdal.GDT_Float32, format='GTiff', filename=filename_a)

        filename_b = pygeoprocessing.temporary_filename()
        sampledata.create_raster_on_disk(
            [numpy.array([[0.123]])], reference.origin,
            reference.projection, -1, reference.pixel_size(30),
            datatype=gdal.GDT_Float32, format='GTiff', filename=filename_b)

        # 0.005 is smaller than the difference between the pixel values in
        # these two matrices, so the relative tolerance check should fail.
        self.assertRaises(
            AssertionError, pygeoprocessing.testing.assert_rasters_equal,
            filename_a, filename_b, tolerance=0.00005)

    def test_assert_close_no_message(self):
        """Verify assert_close provides a default message."""
        from pygeoprocessing.testing import assert_close

        with self.assertRaises(AssertionError):
            assert_close(1, 2, 0.0001)

    def test_raster_different_y_dimension(self):
        """Verify assert_rasters_equal fails when y dimensions differ"""
        reference = sampledata.SRS_COLOMBIA
        filename_a = pygeoprocessing.temporary_filename()
        sampledata.create_raster_on_disk(
            [numpy.array([[0.1], [0.1]])], reference.origin,
            reference.projection, -1, reference.pixel_size(30),
            datatype=gdal.GDT_Float32, format='GTiff', filename=filename_a)

        filename_b = pygeoprocessing.temporary_filename()
        sampledata.create_raster_on_disk(
            [numpy.array([[0.1]])], reference.origin,
            reference.projection, -1, reference.pixel_size(30),
            datatype=gdal.GDT_Float32, format='GTiff', filename=filename_b)

        with self.assertRaises(AssertionError):
            pygeoprocessing.testing.assert_rasters_equal(
                filename_a, filename_b, tolerance=0.00005)

    def test_raster_different_count(self):
        """Verify assert_rasters_equal catches different layer counts."""
        reference = sampledata.SRS_COLOMBIA
        filename_a = pygeoprocessing.temporary_filename()
        sampledata.create_raster_on_disk(
            [numpy.array([[0.1]])], reference.origin,
            reference.projection, -1, reference.pixel_size(30),
            datatype=gdal.GDT_Float32, format='GTiff', filename=filename_a)

        filename_b = pygeoprocessing.temporary_filename()
        sampledata.create_raster_on_disk(
            [numpy.array([[0.1]]), numpy.array([[0.1]])], reference.origin,
            reference.projection, -1, reference.pixel_size(30),
            datatype=gdal.GDT_Float32, format='GTiff', filename=filename_b)

        with self.assertRaises(AssertionError):
            pygeoprocessing.testing.assert_rasters_equal(
                filename_a, filename_b, tolerance=0.00005)

    def test_raster_different_projections(self):
        """Verify assert_rasters_equal catches differing projections."""
        reference = sampledata.SRS_COLOMBIA
        filename_a = pygeoprocessing.temporary_filename()
        sampledata.create_raster_on_disk(
            [numpy.array([[0.1]])], reference.origin,
            reference.projection, -1, reference.pixel_size(30),
            datatype=gdal.GDT_Float32, format='GTiff', filename=filename_a)

        reference = sampledata.SRS_WILLAMETTE
        filename_b = pygeoprocessing.temporary_filename()
        sampledata.create_raster_on_disk(
            [numpy.array([[0.1]])], reference.origin,
            reference.projection, -1, reference.pixel_size(30),
            datatype=gdal.GDT_Float32, format='GTiff', filename=filename_b)

        with self.assertRaises(AssertionError):
            pygeoprocessing.testing.assert_rasters_equal(
                filename_a, filename_b, tolerance=0.00005)


class VectorEquality(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace)

    def test_file_not_found(self):
        """IOError should be raised when a vector does not exist."""
        from pygeoprocessing.testing import assert_vectors_equal
        nonexistent_file_a = os.path.join(self.workspace, 'foo')
        nonexistent_file_b = os.path.join(self.workspace, 'bar')

        with self.assertRaises(IOError):
            assert_vectors_equal(nonexistent_file_a, nonexistent_file_b, 0.1)

    def test_mismatched_layer_counts(self):
        """Mismatched layer counts should raise assertionerror."""
        from pygeoprocessing.testing import assert_vectors_equal

        # creating manually, since create_vector_on_disk can't create vectors
        # with multiple layers at the moment.
        # Using KML for readability and multi-layer support.
        driver = ogr.GetDriverByName('KML')

        filepath_a = os.path.join(self.workspace, 'foo.kml')
        out_vector_a = driver.CreateDataSource(filepath_a)
        out_vector_a.CreateLayer('a')
        out_vector_a.CreateLayer('b')
        out_vector_a = None

        filepath_b = os.path.join(self.workspace, 'bar.kml')
        out_vector_b = driver.CreateDataSource(filepath_b)
        out_vector_b.CreateLayer('a')
        out_vector_b = None

        with self.assertRaises(AssertionError):
            assert_vectors_equal(filepath_a, filepath_b, 0.1)

    def test_mismatched_geometry_type(self):
        """Assert mismatched geometry types."""
        from pygeoprocessing.testing import assert_vectors_equal
        # creating manually, since create_vector_on_disk can't create vectors
        # with multiple layers at the moment.
        reference = sampledata.SRS_WILLAMETTE
        filename_a = os.path.join(self.workspace, 'foo')
        sampledata.create_vector_on_disk(
            [Point(0, 0)], reference.projection, filename=filename_a)

        filename_b = os.path.join(self.workspace, 'bar')
        sampledata.create_vector_on_disk(
            [LineString([(0, 0), (1, 1)])], reference.projection,
            filename=filename_b)

        with self.assertRaises(AssertionError):
            assert_vectors_equal(filename_a, filename_b, 0.1)

    def test_mismatched_projections(self):
        """Raise assertionerror when projections differ."""
        from pygeoprocessing.testing import assert_vectors_equal
        reference = sampledata.SRS_WILLAMETTE
        filename_a = os.path.join(self.workspace, 'foo')
        sampledata.create_vector_on_disk(
            [Point(0, 0)], reference.projection, filename=filename_a)

        reference = sampledata.SRS_COLOMBIA
        filename_b = os.path.join(self.workspace, 'bar')
        sampledata.create_vector_on_disk(
            [Point(0, 0)], reference.projection, filename=filename_b)

        with self.assertRaises(AssertionError):
            assert_vectors_equal(filename_a, filename_b, 0.1)

    def test_different_nonnumeric_field_values(self):
        """Assert that nonnumeric field values can be checked correctly."""
        from pygeoprocessing.testing import assert_vectors_equal
        reference = sampledata.SRS_WILLAMETTE
        filename_a = os.path.join(self.workspace, 'foo')
        sampledata.create_vector_on_disk(
            [Point(0, 0)], reference.projection, fields={'a': 'string'},
            attributes=[{'a': 'aaa'}], filename=filename_a)

        filename_b = os.path.join(self.workspace, 'bar')
        sampledata.create_vector_on_disk(
            [Point(0, 0)], reference.projection, fields={'a': 'string'},
            attributes=[{'a': 'bbb'}], filename=filename_b)

        with self.assertRaises(AssertionError):
            assert_vectors_equal(filename_a, filename_b, 0.1)

    def test_different_geometries(self):
        """Assert we can test geometric values of the same type."""
        from pygeoprocessing.testing import assert_vectors_equal
        reference = sampledata.SRS_WILLAMETTE
        filename_a = os.path.join(self.workspace, 'foo')
        sampledata.create_vector_on_disk(
            [Point(0, 1)], reference.projection, filename=filename_a)

        filename_b = os.path.join(self.workspace, 'bar')
        sampledata.create_vector_on_disk(
            [Point(0, 0)], reference.projection, filename=filename_b)

        with self.assertRaises(AssertionError):
            assert_vectors_equal(filename_a, filename_b, 0.1)


class CSVEquality(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace)

    def test_numeric_value_equality(self):
        """Test that numeric equality testing fails when expected."""
        from pygeoprocessing.testing import assert_csv_equal

        filename_a = os.path.join(self.workspace, 'foo.csv')
        filename_b = os.path.join(self.workspace, 'bar.csv')

        for filename, value in [(filename_a, 0.1),
                                (filename_b, 0.2)]:
            with open(filename, 'w') as csv_file:
                csv_file.write('a\n%s\n' % value)

        with self.assertRaises(AssertionError):
            assert_csv_equal(filename_a, filename_b, 0.001)


class DigestEquality(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace)

    @staticmethod
    def create_sample_folder(dirname):
        """Create a sample file/directory structure at `dirname`.

        Parameters:
            dirname (string): the path to a directory where simple
                files/folders should be created.

        Returns:
            None"""
        filename_a = os.path.join(dirname, '_a')
        filename_b = os.path.join(dirname, '_b')
        filename_c = os.path.join(dirname, '_c', '_d')

        for filename in [filename_a, filename_b, filename_c]:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            # Just write the name of the file to the file.
            with open(filename, 'w') as open_file:
                open_file.write(os.path.basename(filename))

    def test_checksum_assertion_in_cwd(self):
        """Verify testing from CWD if none is provided."""
        from pygeoprocessing.testing import assert_checksums_equal

        sample_folder = tempfile.mkdtemp(dir=self.workspace)
        DigestEquality.create_sample_folder(sample_folder)

        checksum_file = os.path.join(self.workspace, 'checksum.md5')
        utils.checksum_folder(sample_folder, checksum_file)

        try:
            cwd = os.getcwd()
            os.chdir(sample_folder)
            assert_checksums_equal(checksum_file)
        finally:
            os.chdir(cwd)

    def test_bsd_checksum_file(self):
        """Verify a BSD-style checksum file."""
        from pygeoprocessing.testing import assert_checksums_equal

        sample_folder = tempfile.mkdtemp(dir=self.workspace)
        DigestEquality.create_sample_folder(sample_folder)

        checksum_file = os.path.join(self.workspace, 'checksum.md5')
        utils.checksum_folder(sample_folder, checksum_file, style='BSD')

        assert_checksums_equal(checksum_file, base_folder=sample_folder)

    def test_gnu_checksum_file(self):
        """Verify a GNU-style checksum file."""
        from pygeoprocessing.testing import assert_checksums_equal

        sample_folder = tempfile.mkdtemp(dir=self.workspace)
        DigestEquality.create_sample_folder(sample_folder)

        checksum_file = os.path.join(self.workspace, 'checksum.md5')
        utils.checksum_folder(sample_folder, checksum_file, style='GNU')

        assert_checksums_equal(checksum_file, base_folder=sample_folder)

    def test_digest_creation_bad_type(self):
        """Verify an exception is raised when a bad digest type is given."""
        from pygeoprocessing.testing import checksum_folder

        with self.assertRaises(IOError):
            checksum_folder("doesn't matter", "doesn't matter", 'bad type!')

    def test_digest_creation_gnu(self):
        """Verify the correct creation of a GNU-style file."""
        from pygeoprocessing.testing import checksum_folder

        sample_folder = tempfile.mkdtemp(dir=self.workspace)
        DigestEquality.create_sample_folder(sample_folder)

        checksum_file = os.path.join(self.workspace, 'checksum.md5')
        checksum_folder(sample_folder, checksum_file, style='GNU')

        expected_checksum_lines = [
            '# orig_workspace = /tmp/tmpcQNvKj/tmpW_AShy',
            '# OS = Linux',
            '# plat_string = Linux-3.2.0-4-amd64-x86_64-with-debian-jessie-sid',
            '# GDAL = 1.10.1',
            '# numpy = 1.10.4',
            '# pygeoprocessing = 0.3.0a15.post24+n314e47b1d4e9',
            '# checksum_style = GNU',
            '5c855e094bdf284e55e9d16627ddd64b  _a',
            'c716fb29298ad96a3b31757ec9755763  _b',
            '6bc947566bb3f50d712efb0de07bfb19  _c/_d']

        checksum_file_lines = open(checksum_file).read().split('\n')
        for expected_line, found_line in zip(expected_checksum_lines,
                                             checksum_file_lines):
            if expected_line.startswith('# ') and found_line.startswith('# '):
                continue
            self.assertEqual(expected_line, found_line)

    def test_digest_creation_bsd(self):
        """Verify the correct creation of a BSD-style file."""
        from pygeoprocessing.testing import checksum_folder

        sample_folder = tempfile.mkdtemp(dir=self.workspace)
        DigestEquality.create_sample_folder(sample_folder)

        checksum_file = os.path.join(self.workspace, 'checksum.md5')
        checksum_folder(sample_folder, checksum_file, style='BSD')

        expected_checksum_lines = [
            '# orig_workspace = /tmp/tmpcQNvKj/tmpW_AShy',
            '# OS = Linux',
            '# plat_string = Linux-3.2.0-4-amd64-x86_64-with-debian-jessie-sid',
            '# GDAL = 1.10.1',
            '# numpy = 1.10.4',
            '# pygeoprocessing = 0.3.0a15.post24+n314e47b1d4e9',
            '# checksum_style = BSD',
            'MD5 (_a) = 5c855e094bdf284e55e9d16627ddd64b',
            'MD5 (_b) = c716fb29298ad96a3b31757ec9755763',
            'MD5 (_c/_d) = 6bc947566bb3f50d712efb0de07bfb19']

        checksum_file_lines = open(checksum_file).read().split('\n')

        for expected_line, found_line in zip(expected_checksum_lines,
                                             checksum_file_lines):
            if expected_line.startswith('# ') and found_line.startswith('# '):
                continue
            self.assertEqual(expected_line, found_line)



