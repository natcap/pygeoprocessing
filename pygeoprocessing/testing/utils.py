
import hashlib
import functools
import shutil
import logging
import os
import platform
import importlib
import json
import math

import numpy
from osgeo import gdal
import pygeoprocessing

from . import data_storage

LOGGER = logging.getLogger('natcap.testing.utils')

def digest_file_list(filepath_list, ifdir='skip'):
    """
    Create a single MD5sum from all the files in `filepath_list`.

    This is done by creating an MD5sum from the string MD5sums calculated
    from each individual file in the list. The filepath_list will be sorted
    before generating an MD5sum. If a given file is in this list multiple
    times, it will be double-counted.

    Note:
        When passed a list with a single file in it, this function will produce
        a different MD5sum than if you were to simply take the md5sum of that
        single file.  This is because this function produces an MD5sum of
        MD5sums.

    Parameters:
        filepath_list (list of strings): A list of files to analyze.
        ifdir (string): Either 'skip' or 'raise'.  Indicates what to do
            if a directory is encountered in this list.  If 'skip', the
            directory skipped will be logged.  If 'raise', IOError will
            be raised with the directory name.

    Returns:
        A string MD5sum generated for all of the files in the list.

    Raises:
        IOError: When a file in `filepath_list` is a directory and
            `ifdir == skip` or a file could not be found.
    """
    summary_md5 = hashlib.md5()
    for filepath in sorted(filepath_list):
        if os.path.isdir(filepath):
            # We only want to pass files down to the digest_file function
            message = 'Skipping md5sum for directory %s' % filepath
            if ifdir == 'skip':
                LOGGER.warn(message)
                continue
            else:  # ifdir == 'raise'
                raise IOError(message)
        summary_md5.update(digest_file(filepath))

    return summary_md5.hexdigest()


def digest_folder(folder):
    """
    Create a single MD5sum from all of the files in a folder.  This
    recurses through `folder` and will take the MD5sum of all files found
    within.

    Parameters:
        folder (string): A string path to a folder on disk.

    Note:
        When there is a single file within this folder, the return value
        of this function will be different than if you were to take the MD5sum
        of just that one file.  This is because we are taking an MD5sum of MD5sums.

    Returns:
        A string MD5sum generated from all of the files contained in
            this folder.
    """
    file_list = []
    for path, subpath, files in os.walk(folder):
        for name in files:
            file_list.append(os.path.join(path, name))

    return digest_file_list(file_list)


def digest_file(filepath):
    """
    Get the MD5sum for a single file on disk.  Files are read in
    a memory-efficient fashion.

    Args:
        filepath (string): a string path to the file or folder to be tested
            or a list of files to be analyzed.

    Returns:
        An md5sum of the input file
    """

    block_size = 2**20
    file_handler = open(filepath, 'rb')
    file_md5 = hashlib.md5()
    for chunk in iter(lambda: file_handler.read(block_size), ''):
        file_md5.update(chunk)
    file_handler.close()

    return file_md5.hexdigest()


def build_regression_archives(file_uri, input_archive_uri, output_archive_uri):
    """
    Build regression archives for a target model run.

    With a properly formatted JSON configuration file at `file_uri`, all
    input files and parameters are collected and compressed into a single
    gzip.  Then, the target model is executed and the output workspace is
    zipped up into another gzip.  These could then be used for regression
    testing, such as with the ``pygeoprocessing.testing.regression``
    decorator.

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
    saved_data = json.loads(open(file_uri).read())

    arguments = saved_data['arguments']
    model_id = saved_data['model']

    model = importlib.import_module(model_id)

    # guarantee that we're running this in a new workspace
    arguments['workspace_dir'] = pygeoprocessing.temporary_folder()
    workspace = arguments['workspace_dir']

    # collect the parameters into a single folder
    input_archive = input_archive_uri
    if input_archive[-7:] == '.tar.gz':
        input_archive = input_archive[:-7]
    data_storage.collect_parameters(arguments, input_archive)
    input_archive += '.tar.gz'

    model_args = data_storage.extract_parameters_archive(workspace,
                                                         input_archive)

    model.execute(model_args)

    archive_uri = output_archive_uri
    if archive_uri[-7:] == '.tar.gz':
        archive_uri = archive_uri[:-7]
    LOGGER.debug('Archiving the output workspace')
    shutil.make_archive(archive_uri, 'gztar', root_dir=workspace,
                        logger=LOGGER)


def checksum_folder(workspace_uri, logfile_uri, style='GNU'):
    """Recurse through the workspace_uri and for every file in the workspace,
    record the filepath and md5sum to the logfile.  Additional environment
    metadata will also be recorded to help debug down the road.

    This output logfile will have two sections separated by a blank line.
    The first section will have relevant system information, with keys and
    values separated by '=' and some whitespace.

    This second section will identify the files we're snapshotting and the
    md5sums of these files, separated by '::' and some whitspace on each line.
    MD5sums are determined by calling `natcap.testing.utils.digest_file()`.

    Args:
        workspace_uri (string): A URI to the workspace to analyze
        logfile_uri (string): A URI to the logfile to which md5sums and paths
            will be recorded.
        stle='GNU' (string): Either 'GNU' or 'BSD'.  Corresponds to the style
            of the output file.

    Returns:
        Nothing.
    """

    format_styles = {
        'GNU': "{md5}  {filepath}",
        'BSD': "MD5 ({filepath}) = {md5}",
    }
    try:
        md5sum_string = format_styles[style]
    except KeyError:
        raise IOError('Invalid style: %s.  Valid styles: %s' % (
            style, format_styles.keys()))


    logfile = open(logfile_uri, 'w')
    def _write(line):
        """
        Write a line to the logfile with a trailing newline character.
        """
        logfile.write(line + '\n')

    _write('# orig_workspace = %s' % os.path.abspath(workspace_uri))
    _write('# OS = %s' % platform.system())
    _write('# plat_string = %s' % platform.platform())
    _write('# GDAL = %s' % gdal.__version__)
    _write('# numpy = %s' % numpy.__version__)
    _write('# pygeoprocessing = %s' % pygeoprocessing.__version__)
    _write('# checksum_style = %s' % style)

    ignore_exts = ['.shx']
    for dirpath, _, filenames in os.walk(workspace_uri):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)

            # if the extension is in our set of extensions to ignore, skip it.
            if os.path.splitext(filepath)[-1] in ignore_exts:
                continue

            md5sum = digest_file(filepath)
            relative_filepath = filepath.replace(workspace_uri + os.sep, '')

            # Convert to unix path separators for all cases.
            if platform.system() == 'Windows':
                relative_filepath = relative_filepath.replace(os.sep, '/')

            _write(md5sum_string.format(md5=md5sum,
                                        filepath=relative_filepath))


def iterblocks(raster_uri, band=1):
    """
    Return a generator to interate across all the memory blocks in the input
    raster.  Generated values are numpy arrays, read block by block.

    This is especially useful when a single value needs to be derived from the
    pixel values in a raster, such as the sum total of all pixel values, or
    a sequence of unique raster values.  In such cases, `vectorize_datasets`
    is overkill, since it writes out a raster.

    As a generator, this can be combined multiple times with zip() to iterate
    'simultaneously' over multiple rasters, though the user should be careful
    to do so only with prealigned rasters.

    Parameters:
        raster_uri (string): The string filepath to the raster to iterate over.
        band=1 (int): The band number to operate on.  Defaults to 0 if not
            provided.

    Returns:
        On each iteration, a tuple containing a dict of block data and a numpy
        array is returned.  The dict of block data has these attributes:

            data['xoff'] - The X offset of the upper-left-hand corner of the
                block.
            data['yoff'] - The Y offset of the upper-left-hand corner of the
                block.
            data['win_xsize'] - The width of the block.
            data['win_ysize'] - The height of the block.
    """
    dataset = gdal.Open(raster_uri)
    band = dataset.GetRasterBand(band)

    block = band.GetBlockSize()
    cols_per_block = block[0]
    rows_per_block = block[1]

    n_cols = dataset.RasterXSize
    n_rows = dataset.RasterYSize

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    for row_block_index in xrange(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block

        for col_block_index in xrange(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            offset_dict = {
                'xoff': col_offset,
                'yoff': row_offset,
                'win_xsize': col_block_width,
                'win_ysize': row_block_width,
            }
            yield (offset_dict, band.ReadAsArray(**offset_dict))

