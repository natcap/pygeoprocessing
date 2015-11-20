import unittest
import tempfile
import os
import shutil
import glob

import numpy
from osgeo import gdal

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
        self.assertRaises(IOError, testing.assert_rasters_equal, 'file_not_on_disk',
            'other_file_not_on_disk', tolerance=1e-9)
        self.assertRaises(IOError, testing.assert_rasters_equal, 'file_not_on_disk',
            raster_on_disk, tolerance=1e-9)
        self.assertRaises(IOError, testing.assert_rasters_equal,
            raster_on_disk, 'file_not_on_disk', tolerance=1e-9)
        testing.assert_rasters_equal(raster_on_disk, raster_on_disk, tolerance=1e-9)

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
        self.assertRaises(IOError, testing.assert_rasters_equal, 'file_not_on_disk',
            'other_file_not_on_disk', tolerance=1e-9)
        self.assertRaises(IOError, testing.assert_rasters_equal, 'file_not_on_disk',
            vector_on_disk, tolerance=1e-9)
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

        self.assertRaises(AssertionError, testing.assert_vectors_equal, base_file,
            different_file, field_tolerance=1e-9)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_vectors_very_different(self):
        """Verify when two vectors are very, very different."""
        base_file = os.path.join(TEST_INPUT, 'farms.shp')
        different_file = os.path.join(SAMPLE_VECTORS, 'harv_samp_cur.shp')
        self.assertRaises(AssertionError, testing.assert_vectors_equal, base_file,
            different_file, field_tolerance=1e-9)

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

        self.assertRaises(AssertionError, testing.assert_csv_equal, sample_file,
            different_file, tolerance=1e-9)

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

        self.assertRaises(AssertionError, testing.assert_md5_equal, test_file, md5_sum)

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
        self.assertRaises(AssertionError, testing.assert_text_equal, sample_file,
            regression_file)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_snapshot(self):
        """Check that a new snapshot of a folder asserts properly."""
        snapshot_file = os.path.join(TEST_OUT, 'snapshot.snap')
        utils.checksum_folder(REGRESSION_INPUT, snapshot_file)

        testing.assert_checksums_equal(snapshot_file, REGRESSION_INPUT)
        self.assertRaises(AssertionError, testing.assert_checksums_equal,
                          snapshot_file, POLLINATION_DATA)

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
        pygeoprocessing.testing.assert_rasters_equal(filename_a, filename_b, tolerance=0.005)

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
