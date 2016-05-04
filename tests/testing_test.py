import unittest
import tempfile
import os
import shutil
import glob
import hashlib
import platform
import subprocess

import numpy
from osgeo import gdal
from osgeo import ogr
from shapely.geometry import Point, LineString
import mock

from pygeoprocessing.testing import scm
import pygeoprocessing.testing as testing
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

    """Test fixture for asserting CSV files."""

    def setUp(self):
        """Pre-test setUp function.

        Overridden from unittest.TestCase.setUp()
        """
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        """Post-test teardown function.

        Overridden from unittest.TestCase.tearDown()
        """
        shutil.rmtree(self.workspace)

    @staticmethod
    def make_guild_csv(workspace):
        """Make a guilds CSV at workspace/guild.csv.

        Parameters:
            workspace (string): The absolute path to the directory where the
                new file should be stored.
        Returns:
            The absolute path to the new file.
        """
        filename = os.path.join(workspace, 'guild.csv')
        open(filename, 'w').write(
            '"SPECIES","NS_cavity","NS_ground","FS_spring","FS_summer",'
            '"ALPHA","SPECIES_WEIGHT\n"'
            '"Apis",1.0,1.0,0.5,0.5,500.0,1.0\n'
            '"Bombus",1.0,0.0,0.4,0.6,1500.0,1.0')
        return filename

    @staticmethod
    def make_landuse_csv(workspace):
        """Make a landuse CSV at workspace/LU.csv.

        Parameters:
            workspace (string): The absolute path to the directory where the
                new file should be stored.
        Returns:
            The absolute path to the new file.
        """
        filename = os.path.join(workspace, 'LU.csv')
        open(filename, 'w').write(
            '"LULC","DESCRIPTIO","LULC_GROUP","N_cavity"\n'
            '1,"01_Residential 0-4 DU/ac","Built",0.3\n'
            '2,"02_Residential 4-9 DU/ac","Built",0.3')
        return filename

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

    def test_csv_not_found_bad_param_1(self):
        """Test that IOError is raised when param 1 is not found."""
        from pygeoprocessing.testing import assert_csv_equal
        nonexistent_file_a = os.path.join(self.workspace, 'foo')
        guilds = CSVEquality.make_guild_csv(self.workspace)
        with self.assertRaises(IOError):
            assert_csv_equal(nonexistent_file_a, guilds, 1)

    def test_csv_not_found_bad_param_2(self):
        """Test that IOError is raised when param 2 is not found."""
        from pygeoprocessing.testing import assert_csv_equal
        nonexistent_file_a = os.path.join(self.workspace, 'foo')
        guilds = CSVEquality.make_guild_csv(self.workspace)
        with self.assertRaises(IOError):
            assert_csv_equal(guilds, nonexistent_file_a, 1)

    def test_csv_unequal_when_comparing_different_types(self):
        """Assert a failure when comparing differing non-float values."""
        from pygeoprocessing.testing import assert_csv_equal
        guilds = CSVEquality.make_guild_csv(self.workspace)
        landuse = CSVEquality.make_landuse_csv(self.workspace)

        with self.assertRaises(AssertionError):
            assert_csv_equal(guilds, landuse, 1e-9)


class DigestEquality(unittest.TestCase):

    """Test fixture for testing MD5 digest functionality."""

    def setUp(self):
        """Pre-test setUp function.

        Overridden from unittest.TestCase.setUp()
        """
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        """Post-test teardown function.

        Overridden from unittest.TestCase.tearDown()
        """
        shutil.rmtree(self.workspace)

    @staticmethod
    def create_sample_folder(dirname):
        """Create a sample file/directory structure at `dirname`.

        Parameters:
            dirname (string): the path to a directory where simple
                files/folders should be created.

        Returns:
            None
        """
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
        return [filename_a, filename_b, filename_c]

    def test_checksum_assertion_in_cwd(self):
        """Verify testing from CWD if none is provided."""
        from pygeoprocessing.testing import assert_checksums_equal
        from pygeoprocessing.testing import checksum_folder

        sample_folder = tempfile.mkdtemp(dir=self.workspace)
        DigestEquality.create_sample_folder(sample_folder)

        checksum_file = os.path.join(self.workspace, 'checksum.md5')
        checksum_folder(sample_folder, checksum_file)

        try:
            cwd = os.getcwd()
            os.chdir(sample_folder)
            assert_checksums_equal(checksum_file)
        finally:
            os.chdir(cwd)

    def test_bsd_checksum_file(self):
        """Verify a BSD-style checksum file."""
        from pygeoprocessing.testing import assert_checksums_equal
        from pygeoprocessing.testing import checksum_folder

        sample_folder = tempfile.mkdtemp(dir=self.workspace)
        DigestEquality.create_sample_folder(sample_folder)

        checksum_file = os.path.join(self.workspace, 'checksum.md5')
        checksum_folder(sample_folder, checksum_file, style='BSD')

        assert_checksums_equal(checksum_file, base_folder=sample_folder)

    def test_gnu_checksum_file(self):
        """Verify a GNU-style checksum file."""
        from pygeoprocessing.testing import assert_checksums_equal
        from pygeoprocessing.testing import checksum_folder

        sample_folder = tempfile.mkdtemp(dir=self.workspace)
        DigestEquality.create_sample_folder(sample_folder)

        checksum_file = os.path.join(self.workspace, 'checksum.md5')
        checksum_folder(sample_folder, checksum_file, style='GNU')

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
            '# plat_string = Linux-3.2.0-4-amd64-x86_64-with-debian-jessie',
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
            '# plat_string = Linux-3.2.0-4-amd64-x86_64-with-debian-jessie',
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

    def test_digest_creation_skip_extensions(self):
        """Test the ignore_exts parameter for checksum file creation."""
        from pygeoprocessing.testing import checksum_folder

        sample_folder = tempfile.mkdtemp(dir=self.workspace)
        DigestEquality.create_sample_folder(sample_folder)

        # create a new file with the .ignore extension within the sample
        # folder.  This file should not appear in the checksum file at all.
        open(os.path.join(sample_folder, 'foo.ignore'), 'w').write('foo')

        checksum_file = os.path.join(self.workspace, 'checksum.md5')
        checksum_folder(sample_folder, checksum_file, style='GNU',
                        ignore_exts=['.ignore'])

        self.assertTrue('.ignore' not in open(checksum_file).read())

    def test_windows_pathsep_replacement(self):
        """Verify that windows-style path separation works as expected."""
        from pygeoprocessing.testing import checksum_folder

        sample_folder = tempfile.mkdtemp(dir=self.workspace)
        DigestEquality.create_sample_folder(sample_folder)
        checksum_file = os.path.join(self.workspace, 'checksum.md5')

        # Simulate being on Windows.
        # On *NIX systems, this shouldn't affect the output files at all, since
        # we're replacing os.sep ('/' on *NIX) with '/'.
        with mock.patch('platform.system', lambda: 'Windows'):
            # Just to verify that sys.platform() is currently set to Windows.
            self.assertEqual(platform.system(), 'Windows')
            checksum_folder(sample_folder, checksum_file, style='GNU')
            last_line = open(checksum_file).read().split('\n')[-2]
            self.assertEqual(last_line,
                             '6bc947566bb3f50d712efb0de07bfb19  _c/_d')

    def test_digest_file_list_skip_if_dir(self):
        """Verify that file digesting skips directories as expected."""
        from pygeoprocessing.testing import digest_file_list

        # Get the list of files that are being digested and add a folder to it
        filepaths = DigestEquality.create_sample_folder(self.workspace)
        filepaths.append(self.workspace)

        file_hexdigest = digest_file_list(filepaths, ifdir='skip')
        self.assertEqual('7b13e71a17fee2a14e179726281e85cc', file_hexdigest)

    def test_digest_file_list_raise_if_dir(self):
        """Verify that file list digesting raises IOError when dir found."""
        from pygeoprocessing.testing import digest_file_list

        # Get the list of files that are being digested and add a folder to it
        filepaths = DigestEquality.create_sample_folder(self.workspace)
        filepaths.append(self.workspace)

        with self.assertRaises(IOError):
            digest_file_list(filepaths, ifdir='raise')

    def test_digest_folder(self):
        """Verify directory digesting works as expected."""
        from pygeoprocessing.testing import digest_folder

        # Get the list of files that are being digested and add a folder to it
        DigestEquality.create_sample_folder(self.workspace)

        dir_hexdigest = digest_folder(self.workspace)
        self.assertEqual('7b13e71a17fee2a14e179726281e85cc', dir_hexdigest)

    def test_digest_file(self):
        """Verify that we can digest a single file."""
        from pygeoprocessing.testing import digest_file

        # Get the list of files that are being digested and add a folder to it
        files = DigestEquality.create_sample_folder(self.workspace)

        file_digest = digest_file(files[0])
        self.assertEqual('5c855e094bdf284e55e9d16627ddd64b', file_digest)

    def test_checksum_a_missing_file(self):
        """Test for when a checksummed file is missing."""
        from pygeoprocessing.testing import assert_checksums_equal
        from pygeoprocessing.testing import checksum_folder

        # Create the sample files, checksum the folder, then remove one of the
        # files.
        sample_folder = tempfile.mkdtemp(dir=self.workspace)
        files = DigestEquality.create_sample_folder(sample_folder)
        checksum_file = os.path.join(self.workspace, 'checksum.md5')
        checksum_folder(sample_folder, checksum_file, style='GNU')
        os.remove(files[0])

        with self.assertRaises(AssertionError):
            assert_checksums_equal(checksum_file, sample_folder)

    def test_checksum_a_modified_file(self):
        """Test for when a checksummed file has been modified."""
        from pygeoprocessing.testing import assert_checksums_equal
        from pygeoprocessing.testing import checksum_folder

        # Create the sample files, checksum the folder, then remove one of the
        # files.
        sample_folder = tempfile.mkdtemp(dir=self.workspace)
        files = DigestEquality.create_sample_folder(sample_folder)
        checksum_file = os.path.join(self.workspace, 'checksum.md5')
        checksum_folder(sample_folder, checksum_file, style='GNU')
        open(files[0], 'a').write('foo!')

        with self.assertRaises(AssertionError):
            assert_checksums_equal(checksum_file, sample_folder)


class SCMTest(unittest.TestCase):

    """Test fixture for testing SCM-related functionality."""

    def setUp(self):
        """Pre-test setUp function.

        Overridden from unittest.TestCase.setUp()
        """
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        """Post-test teardown function.

        Overridden from unittest.TestCase.tearDown()
        """
        shutil.rmtree(self.workspace)

    def test_skip_decorator_missing_dir(self):
        """Test that SkipTest is raised when a data dir is missing."""
        from pygeoprocessing.testing.scm import skip_if_data_missing
        nonexistent_dir = os.path.join(self.workspace, 'foo')

        with self.assertRaises(unittest.SkipTest):
            # sample_function needs a parameter to simulate the passing of self
            # within the context of a unittest.TestCase class.
            @skip_if_data_missing(nonexistent_dir)
            def sample_function(a):
                pass

            # True doesn't mean anything here, it's just to simulate the
            # passing of self to object methods, as discussed above.
            sample_function(True)

    def test_skip_decorator_noskip(self):
        """Verify decorated 'test' function passes when data dir exists."""
        from pygeoprocessing.testing.scm import skip_if_data_missing

        # sample_function needs a parameter to simulate the passing of self
        # within the context of a unittest.TestCase class.
        @skip_if_data_missing(self.workspace)
        def sample_function(a):
            pass

        # True doesn't mean anything here, it's just to simulate the
        # passing of self to object methods, as discussed above.
        sample_function(True)

    def test_checkout_svn(self):
        """Verify that SVN checkout is called with the correct parameters."""
        from pygeoprocessing.testing.scm import checkout_svn
        nonexistent_folder = os.path.join(self.workspace, 'dir_not_found')
        remote_path = 'svn://foo'

        with mock.patch('subprocess.call'):
            checkout_svn(nonexistent_folder, remote_path)
            self.assertTrue(subprocess.call.called)
            self.assertEqual(subprocess.call.call_args[0][0],
                             ['svn', 'checkout', remote_path,
                              nonexistent_folder, '-r', 'HEAD'])

    def test_update_svn(self):
        """Verify that SVN update is called with the correct parameters."""
        from pygeoprocessing.testing.scm import checkout_svn

        with mock.patch('subprocess.call'):
            checkout_svn(self.workspace, 'svn://foo')
            self.assertTrue(subprocess.call.called)
            self.assertEqual(subprocess.call.call_args[0][0],
                             ['svn', 'update', '-r', 'HEAD'])

    def test_update_svn_to_rev(self):
        """Verify SVN update -r <rev> is called with the correct params."""
        from pygeoprocessing.testing.scm import checkout_svn

        with mock.patch('subprocess.call'):
            checkout_svn(self.workspace, 'svn://foo', rev='25')
            self.assertTrue(subprocess.call.called)
            self.assertEqual(subprocess.call.call_args[0][0],
                             ['svn', 'update', '-r', '25'])
