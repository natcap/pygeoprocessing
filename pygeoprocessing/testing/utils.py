
import hashlib
import functools
import shutil
import logging
import os
import platform

import numpy
from osgeo import gdal
import pygeoprocessing

from . import data_storage

LOGGER = logging.getLogger('natcap.testing.utils')

def get_hash(filepath):
    """Get the MD5 hash for a single file or folder.  If a folder, all files
        in the folder will be combined into a single md5sum.  Files are read in
        a memory-efficient fashion.

        Args:
            filepath (string): a string path to the file or folder to be tested
                or a list of files to be analyzed.

        Returns:
            An md5sum of the input file"""

    if isinstance(filepath, list):
        # User provided list of files, ensure they are files and not folders.
        file_list = []
        for filepath in filepath:
            if not os.path.isdir(filepath):
                file_list.append(filepath)
    elif os.path.isdir(filepath):
        # User provided a folder, so recurse through all the files.
        file_list = []
        for path, subdirs, files in os.walk(filepath):
            for name in files:
                file_list.append(os.path.join(path, name))
    else:
        # User only provided a single file.
        file_list = [filepath]

    block_size = 2**20
    all_md5 = hashlib.md5()
    for filename in sorted(file_list):
        file_handler = open(filename, 'rb')
        file_md5 = hashlib.md5()
        for chunk in iter(lambda: file_handler.read(block_size), ''):
            file_md5.update(chunk)
        all_md5.update(file_md5.hexdigest())
        file_handler.close()

    return all_md5.hexdigest()


def save_workspace(new_workspace):
    """Decorator to save a workspace to a new location.

        If `new_workspace` already exists on disk, it will be recursively
        removed.

        Example usage with a test case::

            import pygeoprocessing.testing

            @pygeoprocessing.testing.save_workspace('/path/to/workspace')
            def test_workspaces(self):
                model.execute(self.args)

        Note:
            + Target workspace folder must be saved to ``self.workspace_dir``
                This decorator is only designed to work with test functions
                from subclasses of ``unittest.TestCase`` such as
                ``pygeoprocessing.testing.GISTest``.

            + If ``new_workspace`` exists, it will be removed.
                So be careful where you save things.

        Args:
            new_workspace (string): a URI to the where the workspace should be
                copied.

        Returns:
            A composed test case function which will execute and then save your
            workspace to the specified location."""

    # item is the function being decorated
    def test_inner_func(item):

        # this decorator indicates that this innermost function is wrapping up
        # the function passed in as item.
        @functools.wraps(item)
        def test_and_remove_workspace(self, *args, **kwargs):
            # This inner function actually executes the test function and then
            # moves the workspace to the folder passed in by the user.
            item(self)

            # remove the contents of the old folder
            try:
                shutil.rmtree(new_workspace)
            except OSError:
                pass

            # copy the workspace to the target folder
            old_workspace = self.workspace_dir
            shutil.copytree(old_workspace, new_workspace)
        return test_and_remove_workspace
    return test_inner_func


def regression(input_archive, workspace_archive):
    """Decorator to unzip input data, run the regression test and compare the
        outputs against the outputs on file.

        Example usage with a test case::

            import pygeoprocessing.testing

            @pygeoprocessing.testing.regression('/data/input.tar.gz', /data/output.tar.gz')
            def test_workspaces(self):
                model.execute(self.args)

        Args:
            input_archive (string): The path to a .tar.gz archive with the input data.
            workspace_archive (string): The path to a .tar.gz archive with the workspace to
                assert.

        Returns:
            Composed function with regression testing.
         """

    # item is the function being decorated
    def test_inner_function(item):

        @functools.wraps(item)
        def test_and_assert_workspace(self, *args, **kwargs):
            workspace = pygeoprocessing.geoprocessing.temporary_folder()
            self.args = data_storage.extract_parameters_archive(workspace, input_archive)

            # Actually run the test.  Assumes that self.args is used as the
            # input arguments.
            item(self)

            # Extract the archived workspace to a new temporary folder and
            # compare the two workspaces.
            archived_workspace = pygeoprocessing.geoprocessing.temporary_folder()
            data_storage.extract_archive(archived_workspace, workspace_archive)
            self.assertWorkspace(workspace, archived_workspace)
        return test_and_assert_workspace
    return test_inner_function


def build_regression_archives(file_uri, input_archive_uri, output_archive_uri):
    """Build regression archives for a target model run.

        With a properly formatted JSON configuration file at `file_uri`, all
        input files and parameters are collected and compressed into a single
        gzip.  Then, the target model is executed and the output workspace is
        zipped up into another gzip.  These could then be used for regression
        testing, such as with the ``pygeoprocessing.testing.regression`` decorator.

        Example configuration file contents (serialized to JSON)::

            {
                    "model": "pygeoprocessing.pollination.pollination",
                    "arguments": {
                        # the full set of model arguments here
                    }
            }

        Example function usage::

            import pygeoprocessing.testing

            file_uri = "/path/to/config.json"
            input_archive_uri = "/path/to/archived_inputs.tar.gz"
            output_archive_uri = "/path/to/archived_outputs.tar.gz"
            pygeoprocessing.testing.build_regression_archives(file_uri,
                input_archive_uri, output_archive_uri)

        Args:
            file_uri (string): a URI to a json file on disk containing the
            above configuration options.

            input_archive_uri (string): the URI to where the gzip archive
            of inputs should be saved once it is created.

            output_archive_uri (string): the URI to where the gzip output
            archive of output should be saved once it is created.

        Returns:
            Nothing.
        """
    file_handler = fileio.JSONHandler(file_uri)

    saved_data = file_handler.get_attributes()

    arguments = saved_data['arguments']
    model_id = saved_data['model']

    model_list = model_id.split('.')
    model = executor.locate_module(model_list)

    # guarantee that we're running this in a new workspace
    arguments['workspace_dir'] = pygeoprocessing.geoprocessing.temporary_folder()
    workspace = arguments['workspace_dir']

    # collect the parameters into a single folder
    input_archive = input_archive_uri
    if input_archive[-7:] == '.tar.gz':
        input_archive = input_archive[:-7]
    data_storage.collect_parameters(arguments, input_archive)
    input_archive += '.tar.gz'

    model_args = data_storage.extract_parameters_archive(workspace, input_archive)

    model.execute(model_args)

    archive_uri = output_archive_uri
    if archive_uri[-7:] == '.tar.gz':
        archive_uri = archive_uri[:-7]
    LOGGER.debug('Archiving the output workspace')
    shutil.make_archive(archive_uri, 'gztar', root_dir=workspace, logger=LOGGER)


def snapshot_folder(workspace_uri, logfile_uri):
    """Recurse through the workspace_uri and for every file in the workspace,
    record the filepath and md5sum to the logfile.  Additional environment
    metadata will also be recorded to help debug down the road.

    This output logfile will have two sections separated by a blank line.
    The first section will have relevant system information, with keys and values
    separated by '=' and some whitespace.

    This second section will identify the files we're snapshotting and the
    md5sums of these files, separated by '::' and some whitspace on each line.
    MD5sums are determined by calling `natcap.testing.utils.get_hash()`.

    Args:
        workspace_uri (string): A URI to the workspace to analyze

        logfile_uri (string): A URI to the logfile to which md5sums and paths
        will be recorded.

        workspace_varname (string): The string variable name that should be
        used when setting the ws_uri variable initially.

    Returns:
        Nothing.
    """

    logfile = open(logfile_uri, 'w')
    def _write(line):
        """
        Write a line to the logfile with a trailing newline character.
        """
        logfile.write(line + '\n')

    _write('orig_workspace = %s' % os.path.abspath(workspace_uri))
    _write('OS = %s' % platform.system())
    _write('plat_string = %s' % platform.platform())
    _write('GDAL = %s' % gdal.__version__)
    _write('numpy = %s' % numpy.__version__)
    _write('')  # blank line to signal end of section

    ignore_exts = ['.shx']
    for dirpath, _, filenames in os.walk(workspace_uri):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)

            # if the extension is in our set of extensions to ignore, skip it.
            if os.path.splitext(filepath)[-1] in ignore_exts:
                continue

            md5sum = get_hash(filepath)
            relative_filepath = filepath.replace(workspace_uri + os.sep, '')

            # Convert to unix path separators for all cases.
            if platform.system() == 'Windows':
                relative_filepath = relative_filepath.replace(os.sep, '/')

            _write('{file} :: {md5}'.format(file=relative_filepath, md5=md5sum))
