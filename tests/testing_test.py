import unittest
import tempfile
import os
import shutil
import glob

import numpy
from osgeo import gdal

from pygeoprocessing.testing import scm
import pygeoprocessing.testing as testing
from pygeoprocessing.testing import data_storage
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


class DataStorageTest(unittest.TestCase):
    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_collect_parameters_simple(self):
        params = {
            'a': 1,
            'b': 2,
            'c': os.path.join(TEST_INPUT, 'LU.csv'),
        }

        archive_uri = os.path.join(TEST_OUT, 'archive')

        data_storage.collect_parameters(params, archive_uri)
        archive_uri = archive_uri + '.tar.gz'

        regression_archive_uri = os.path.join(REGRESSION_ARCHIVES,
            'collect_parameters_simple.tar.gz')

        testing.assert_archives_equal(archive_uri, regression_archive_uri)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_collect_parameters_nested_dict(self):
        params = {
            'a': 1,
            'b': 2,
            'd': {
                'one': 1,
                'two': 2,
                'three': os.path.join(TEST_INPUT, 'Guild.csv')
            },
            'c': os.path.join(TEST_INPUT, 'LU.csv'),
        }

        archive_uri = os.path.join(TEST_OUT, 'archive_nested_dict')

        data_storage.collect_parameters(params, archive_uri)
        archive_uri = archive_uri + '.tar.gz'

        regression_archive_uri = os.path.join(REGRESSION_ARCHIVES,
            'simple_nested_dict.tar.gz')

        testing.assert_archives_equal(archive_uri, regression_archive_uri)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_archive_geotiff(self):
        """Archive a simple geotiff."""
        params = {
            'raster': os.path.join(TEST_INPUT, 'landuse_cur_200m.tif')
        }
        archive_uri = os.path.join(TEST_OUT, 'raster_geotiff')
        data_storage.collect_parameters(params, archive_uri)
        archive_uri += '.tar.gz'

        regression_archive_uri = os.path.join(REGRESSION_ARCHIVES,
            'raster_geotiff.tar.gz')
        testing.assert_archives_equal(archive_uri, regression_archive_uri)


    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_archive_arc_raster_nice(self):
        """
        Test that messy arc raster organization collection will work.

        Arc/Info Binary Grid (AIG) rasters are usually stored in nice, neat
        directories, where all of the files in that directory belong to that
        raster.  Verify that when a raster is organized nicely like this,
        archiving works as expected.
        """
        params = {
            'raster': os.path.join(SAMPLE_RASTERS, 'lulc_samp_cur')
        }

        archive_uri = os.path.join(TEST_OUT, 'raster_nice')
        data_storage.collect_parameters(params, archive_uri)

        archive_uri += '.tar.gz'
        regression_archive_uri = os.path.join(REGRESSION_ARCHIVES,
            'arc_raster_nice.tar.gz')
        testing.assert_archives_equal(archive_uri, regression_archive_uri)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_archive_arc_raster_messy(self):
        """
        Test that messy arc raster organization collection will work.

        Arc/Info Binary Grid (AIG) rasters are usually stored in nice, neat
        directories, where all of the files in that directory belong to that
        raster.  This test verifies that when there is an extra file in that
        directory, the correct files are collected.

        Unfortunately, GDAL collects all the files in that folder, so anything
        in that folder is considered part of an AIG raster.
        """
        params = {
            'raster': os.path.join(TEST_INPUT, 'messy_raster_organization',
                'hdr.adf')
        }

        archive_uri = os.path.join(TEST_OUT, 'raster_messy')
        data_storage.collect_parameters(params, archive_uri)

        archive_uri += '.tar.gz'
        regression_archive_uri = os.path.join(REGRESSION_ARCHIVES,
            'arc_raster_messy.tar.gz')
        testing.assert_archives_equal(archive_uri, regression_archive_uri)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_archive_esri_shapefile(self):
        params = {
            'vector': os.path.join(SAMPLE_VECTORS, 'harv_samp_cur.shp')
        }

        archive_uri = os.path.join(TEST_OUT, 'vector_collected')
        data_storage.collect_parameters(params, archive_uri)

        archive_uri += '.tar.gz'
        regression_archive_uri = os.path.join(REGRESSION_ARCHIVES,
            'vector_collected.tar.gz')
        testing.assert_archives_equal(archive_uri, regression_archive_uri)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_archive_pollination_input(self):
        """Archive all the inputs for an example InVEST model (pollination)"""
        params = {
            u'ag_classes': '67 68 71 72 73 74 75 76 78 79 80 81 82 83 84 85 88 90 91 92',
            u'do_valuation': True,
            u'farms_shapefile': os.path.join(TEST_INPUT, 'farms.shp'),
            u'guilds_uri': os.path.join(TEST_INPUT, 'Guild.csv'),
            u'half_saturation': 0.125,
            u'landuse_attributes_uri': os.path.join(TEST_INPUT, 'LU.csv'),
            u'landuse_cur_uri': os.path.join(SAMPLE_RASTERS, 'lulc_samp_cur', 'hdr.adf'),
            u'landuse_fut_uri': os.path.join(SAMPLE_RASTERS, 'lulc_samp_fut', 'hdr.adf'),
            u'results_suffix': 'suff',
            u'wild_pollination_proportion': 1.0,
            u'workspace_dir': os.path.join(TEST_OUT, 'pollination_workspace'),
        }

        archive_uri = os.path.join(TEST_OUT, 'pollination_input')
        data_storage.collect_parameters(params, archive_uri)

        archive_uri += '.tar.gz'
        regression_archive_uri = os.path.join(REGRESSION_ARCHIVES,
            'pollination_input.tar.gz')
        testing.assert_archives_equal(archive_uri, regression_archive_uri)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_extract_archive(self):
        """Verify an archive extracts as expected."""
        workspace = raster_utils.temporary_folder()
        archive_uri = os.path.join(REGRESSION_ARCHIVES,
            'pollination_input.tar.gz')
        input_folder = raster_utils.temporary_folder()
        parameters = data_storage.extract_parameters_archive(workspace,
            archive_uri, input_folder)

        self.maxDiff = None
        regression_params = {
            u'ag_classes': u'67 68 71 72 73 74 75 76 78 79 80 81 82 83 84 85 88 90 91 92',
            u'do_valuation': True,
            u'farms_shapefile': os.path.join(input_folder, u'vector_NY3T16'),
            u'guilds_uri': os.path.join(input_folder, u'Guild.csv'),
            u'half_saturation': 0.125,
            u'landuse_attributes_uri': os.path.join(input_folder, u'LU.csv'),
            u'landuse_cur_uri': os.path.join(input_folder, u'raster_SST0AO'),
            u'landuse_fut_uri': os.path.join(input_folder, u'raster_NFQKOB'),
            u'results_suffix': u'suff',
            u'wild_pollination_proportion': 1.0,
            u'workspace_dir': workspace,
        }

        self.assertEqual(parameters, regression_params)

        for key in ['farms_shapefile', 'guilds_uri', 'landuse_attributes_uri',
            'landuse_cur_uri', 'landuse_fut_uri']:
            self.assertEqual(True, os.path.exists(parameters[key]))

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_extract_archive_nested_args(self):
        """Test that a complicated args dictionary archives properly"""
        input_parameters = {
            'a': 1,
            'b': 2,
            'd': {
                'one': 1,
                'two': 2,
                'three': os.path.join(TEST_INPUT, 'Guild.csv')
            },
            'c': os.path.join(SAMPLE_VECTORS, 'harv_samp_cur.shp'),
            'raster_list': [
                os.path.join(SAMPLE_RASTERS, 'lulc_samp_fut'),
                {
                    'lulc_samp_cur': os.path.join(SAMPLE_RASTERS, 'lulc_samp_cur'),
                    'do_biophysical': True,
                }
            ],
            'c_again': os.path.join(SAMPLE_VECTORS, 'harv_samp_cur.shp'),
        }
        archive_uri = os.path.join(TEST_OUT, 'nested_args')
        data_storage.collect_parameters(input_parameters, archive_uri)
        archive_uri += '.tar.gz'
        self.maxDiff=None

        workspace = raster_utils.temporary_folder()
        input_folder = raster_utils.temporary_folder()
        regression_parameters = {
            u'a': 1,
            u'b': 2,
            u'd': {
                u'one': 1,
                u'two': 2,
                u'three': os.path.join(input_folder, u'Guild.csv')
            },
            u'c': os.path.join(input_folder, u'vector_W4DTZB'),
            u'raster_list': [
                os.path.join(input_folder, u'raster_NFQKOB'),
                {
                    u'lulc_samp_cur': os.path.join(input_folder, u'raster_SST0AO'),
                    u'do_biophysical': True,
                }
            ],
            u'c_again': os.path.join(input_folder, u'vector_W4DTZB'),
            u'workspace_dir': workspace,
        }
        parameters = data_storage.extract_parameters_archive(workspace,
            archive_uri, input_folder)
        self.assertEqual(parameters, regression_parameters)

        files_to_check = [
            regression_parameters['d']['three'],
            regression_parameters['c'],
            regression_parameters['raster_list'][0],
            regression_parameters['raster_list'][1]['lulc_samp_cur'],
            regression_parameters['c_again'],
        ]

        for file_uri in files_to_check:
            self.assertEqual(True, os.path.exists(file_uri))

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_archive_dbf(self):
        """
        Test that a DBF archives correctly

        This requires its own test because OGR happily will open a DBF file.
        We do not, however, want to treat it like a vector, we want to treat
        it like a regular file.
        """
        input_parameters = {
            'dbf_file': os.path.join(TEST_INPUT, 'carbon_pools_samp.dbf'),
        }
        archive_uri = os.path.join(TEST_OUT, 'dbf_archive')
        data_storage.collect_parameters(input_parameters, archive_uri)

        archive_uri += '.tar.gz'
        reg_archive_uri = os.path.join(REGRESSION_ARCHIVES,
            'dbf_archive.tar.gz')
        testing.assert_archives_equal(archive_uri, reg_archive_uri)


class GISTestTester(unittest.TestCase):
    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_raster_assertion_fileio(self):
        """Verify correct behavior for assertRastersEqual"""

        # check that IOError is raised if a file is not found.
        raster_on_disk = os.path.join(TEST_INPUT, 'landuse_cur_200m.tif')
        self.assertRaises(IOError, testing.assert_rasters_equal, 'file_not_on_disk',
            'other_file_not_on_disk')
        self.assertRaises(IOError, testing.assert_rasters_equal, 'file_not_on_disk',
            raster_on_disk)
        self.assertRaises(IOError, testing.assert_rasters_equal,
            raster_on_disk, 'file_not_on_disk')
        testing.assert_rasters_equal(raster_on_disk, raster_on_disk)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_raster_assertion_files_equal(self):
        """Verify when rasters are, in fact, equal."""
        temp_folder = raster_utils.temporary_folder()
        new_raster = os.path.join(temp_folder, 'new_file.tif')

        source_file = os.path.join(TEST_INPUT, 'landuse_cur_200m.tif')
        shutil.copyfile(source_file, new_raster)
        testing.assert_rasters_equal(source_file, new_raster)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_raster_assertion_different_dims(self):
        """Verify when rasters are different"""
        source_raster = os.path.join(TEST_INPUT, 'landuse_cur_200m.tif')
        different_raster = os.path.join(SAMPLE_RASTERS, 'lulc_samp_cur')
        self.assertRaises(AssertionError, testing.assert_rasters_equal,
            source_raster, different_raster)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_raster_assertion_different_values(self):
        """Verify when rasters have different values"""
        lulc_cur_raster = os.path.join(TEST_INPUT, 'landuse_cur_200m.tif')
        lulc_fut_raster = os.path.join(TEST_INPUT, 'landuse_fut_200m.tif')
        self.assertRaises(AssertionError, testing.assert_rasters_equal,
            lulc_cur_raster, lulc_fut_raster)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_vector_assertion_fileio(self):
        """Verify correct behavior for assertVectorsEqual"""
        vector_on_disk = os.path.join(TEST_INPUT, 'farms.dbf')
        self.assertRaises(IOError, testing.assert_rasters_equal, 'file_not_on_disk',
            'other_file_not_on_disk')
        self.assertRaises(IOError, testing.assert_rasters_equal, 'file_not_on_disk',
            vector_on_disk)
        self.assertRaises(IOError, testing.assert_rasters_equal,
            vector_on_disk, 'file_not_on_disk')
        testing.assert_vectors_equal(vector_on_disk, vector_on_disk)

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
        testing.assert_vectors_equal(sample_shape, copied_shape)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_vectors_different_attributes(self):
        """Verify when two vectors have different attributes"""
        base_file = os.path.join(TEST_INPUT, 'farms.shp')
        different_file = os.path.join(REGRESSION_ARCHIVES, 'farms.shp')

        self.assertRaises(AssertionError, testing.assert_vectors_equal, base_file,
            different_file)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_vectors_very_different(self):
        """Verify when two vectors are very, very different."""
        base_file = os.path.join(TEST_INPUT, 'farms.shp')
        different_file = os.path.join(SAMPLE_VECTORS, 'harv_samp_cur.shp')
        self.assertRaises(AssertionError, testing.assert_vectors_equal, base_file,
            different_file)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_csv_assertion_fileio(self):
        bad_file_1 = 'aaa'
        bad_file_2 = 'bbbbb'
        good_file = os.path.join(TEST_INPUT, 'Guild.csv')

        self.assertRaises(IOError, testing.assert_csv_equal, bad_file_1, bad_file_2)
        self.assertRaises(IOError, testing.assert_csv_equal, bad_file_1, good_file)
        self.assertRaises(IOError, testing.assert_csv_equal, good_file, bad_file_2)
        testing.assert_csv_equal(good_file, good_file)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_csv_assertion_fails(self):
        sample_file = os.path.join(TEST_INPUT, 'Guild.csv')
        different_file = os.path.join(TEST_INPUT, 'LU.csv')

        self.assertRaises(AssertionError, testing.assert_csv_equal, sample_file,
            different_file)

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
    def test_archive_assertion(self):
        """Check that two archives are equal"""
        archive_file = os.path.join(REGRESSION_ARCHIVES,
            'arc_raster_nice.tar.gz')
        testing.assert_archives_equal(archive_file, archive_file)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_archive_assertion_fails(self):
        """Check that assertion fails when two archives are different"""
        archive_file = os.path.join(REGRESSION_ARCHIVES,
            'arc_raster_nice.tar.gz')
        different_archive = os.path.join(REGRESSION_ARCHIVES,
            'arc_raster_messy.tar.gz')
        self.assertRaises(AssertionError, testing.assert_archives_equal, archive_file,
            different_archive)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_workspaces_passes(self):
        """Check that asserting equal workspaces passes"""
        workspace_uri = os.path.join(REGRESSION_ARCHIVES, '..')
        testing.assert_workspace(workspace_uri, workspace_uri)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_workspaces_differ(self):
        """Check that asserting equal workspaces fails."""
        self.assertRaises(AssertionError, testing.assert_workspace,
            POLLINATION_DATA, REGRESSION_ARCHIVES)

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
    def test_assert_file_contents_equal_ext_diff(self):
        uri_1 = os.path.join(TESTING_REGRESSION, 'sample_json.json')
        uri_2 = os.path.join(REGRESSION_ARCHIVES, 'arc_raster_nice.tar.gz')
        self.assertRaises(AssertionError, testing.assert_file_contents_equal, uri_1, uri_2)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_assert_file_contents_equal_dne(self):
        uri_1 = os.path.join('invest-data/test/data', 'file_not_exists.txt')
        uri_2 = os.path.join(REGRESSION_ARCHIVES, 'arc_raster_nice.tar.gz')
        self.assertRaises(IOError, testing.assert_file_contents_equal, uri_1, uri_2)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_assert_file_contents_equal_json_same(self):
        json_path = os.path.join(TESTING_REGRESSION, 'sample_json.json')
        testing.assert_file_contents_equal(json_path, json_path)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_assert_file_contents_equal_json_different(self):
        """Check that asserting different json objects fails"""
        json_path = os.path.join(TESTING_REGRESSION, 'sample_json.json')
        json_path_new = os.path.join(TESTING_REGRESSION, 'sample_json_2.json')
        self.assertRaises(AssertionError, testing.assert_file_contents_equal, json_path,
            json_path_new)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_assert_file_contents_equal_gdal_same(self):
        source_file = os.path.join(TEST_INPUT, 'landuse_cur_200m.tif')
        testing.assert_file_contents_equal(source_file, source_file)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_assert_file_contents_equal_gdal_different(self):
        source_raster = os.path.join(TEST_INPUT, 'landuse_cur_200m.tif')
        different_raster = os.path.join(SAMPLE_RASTERS, 'lulc_samp_cur')
        self.assertRaises(AssertionError, testing.assert_file_contents_equal,
            source_raster, different_raster)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_assert_file_contents_equal_ogr_same(self):
        sample_shape = os.path.join(TEST_INPUT, 'farms.shp')
        testing.assert_file_contents_equal(sample_shape, sample_shape)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_assert_file_contents_equal_ogr_different(self):
        base_file = os.path.join(TEST_INPUT, 'farms.shp')
        different_file = os.path.join(REGRESSION_ARCHIVES, 'farms.shp')
        self.assertRaises(AssertionError, testing.assert_file_contents_equal, base_file,
            different_file)

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
    def test_assert_file_contents_equal_text_same(self):
        """Check that asserting two identical text files passes"""
        sample_file = os.path.join(REGRESSION_INPUT, 'sample_text_file.txt')
        testing.assert_file_contents_equal(sample_file, sample_file)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_assert_file_contents_equal_text_different(self):
        """Check that asserting two different text files fails."""
        sample_file = os.path.join(REGRESSION_INPUT, 'sample_text_file.txt')
        regression_file = os.path.join(REGRESSION_INPUT, 'sample_json.json')
        self.assertRaises(AssertionError, testing.assert_file_contents_equal, sample_file,
            regression_file)

    @scm.skip_if_data_missing(SVN_LOCAL_DIR)
    def test_snapshot(self):
        """Check that a new snapshot of a folder asserts properly."""
        snapshot_file = os.path.join(TEST_OUT, 'snapshot.snap')
        utils.checksum_folder(REGRESSION_INPUT, snapshot_file)

        testing.assert_checksums_equal(snapshot_file, REGRESSION_INPUT)
        self.assertRaises(AssertionError, testing.assert_checksums_equal,
                          snapshot_file, POLLINATION_DATA)

    def test_raster_equality_to_places(self):
        """Verify assert_rasters_equal can assert out to n places."""
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

        pygeoprocessing.testing.assert_rasters_equal(filename_a, filename_b, places=3)

    def test_raster_inequality_to_places(self):
        """Verify assert_rasters_equal fails if inequal past n places."""
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

        self.assertRaises(
            AssertionError, pygeoprocessing.testing.assert_rasters_equal,
            filename_a, filename_b, places=5)
