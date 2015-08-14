"""
A module for InVEST test-related data storage.
"""

import os
import json
import tarfile
import shutil
import inspect
import logging
import string
import random
import glob

from osgeo import gdal
from osgeo import ogr

import pygeoprocessing.geoprocessing
import pygeoprocessing.testing


DATA_ARCHIVES = os.path.join('data', 'regression_archives')
INPUT_ARCHIVES = os.path.join(DATA_ARCHIVES, 'input')
OUTPUT_ARCHIVES = os.path.join(DATA_ARCHIVES, 'output')

# ArcInfo Binary Grid files documented here:
# http://support.esri.com/en/knowledgebase/techarticles/detail/30616
# ESRI Shapefile documented here:
# https://www.esri.com/library/whitepapers/pdfs/shapefile.pdf
COMPLEX_FILES = {
    'ArcInfo Binary Grid': ['dblbnd.adf', 'hdr.adf', 'log', 'metadata.xml',
                            'prj.adf', 'sta.adf', 'vat.adf', 'w001001.adf',
                            'w001001x.adf'],
    'ESRI Shapefile': ['.dbf', '.shp', '.prj', '.shx'],
}

LOGGER = logging.getLogger('natcap.invest.testing.data_storage')


def archive_uri(name=None):
    if name is None:
        calling_function = inspect.stack()[1]
        name = calling_function.__name__

    return(os.path.join(INPUT_ARCHIVES, name))


def make_random_dir(workspace, seed_string, prefix, make_dir=True):
    """
    Make a directory in `workspace` where the name is a combination of
    `prefix` and a 6-character random string based off of `seed_string`.

    Parameters:
        workspace (string): The string workspace in which to create the new
            folder.
        seed_string (string): A string to use as a seed for the random folder
            name.
        prefix (string): The prefix of the new folder name.  The first part
            of the new folder name.
        make_dir=True (boolean): Whether to make the new folder we are naming.

    Returns:
        The path to the new folder.  The name of the folder will be of the
        pattern: <prefix><random chars>.  Example: raster_1G3GES
    """
    LOGGER.debug('Random directory seed: %s', seed_string)
    random.seed(seed_string)
    new_dirname = ''.join(random.choice(string.ascii_uppercase + string.digits)
                          for x in range(6))
    new_dirname = prefix + new_dirname
    LOGGER.debug('New directory name: %s', new_dirname)
    raster_dir = os.path.join(workspace, new_dirname)

    if make_dir:
        os.mkdir(raster_dir)

    return raster_dir


def _get_multi_part_gdal(filepath, workspace):
    """Collect all GDAL files into a new folder inside of the temp_workspace
    (a closure from the collect_parameters function).

    This function uses gdal's internal knowledge of the files it contains to
    determine which files are to be included.

        filepath - a URI to a file that is in a GDAL raster.
        workspace - the path to the workspace in which this gdal file should
            be saved.

    Returns the name of the new folder within the temp_workspace that
    contains all the files in this raster."""

    # Get the file list from GDAL
    # NOTE: with ESRI Rasters, other files are sometimes included in this
    # list that are not actually part of the raster.  This is an acceptable
    # cost of this function, since we are now able to handle ALL raster
    # types supported by GDAL.
    dataset = gdal.Open(filepath)
    file_list = dataset.GetFileList()
    LOGGER.debug('Files in raster: %s', file_list)
    dataset = None

    if len(file_list) == 1:
        # If there is only one file in the raster, just return the file name
        raster_file = file_list[0]
        new_file_location = os.path.join(workspace,
                                         os.path.basename(raster_file))
        shutil.copyfile(raster_file, new_file_location)
        return os.path.basename(file_list[0])
    else:
        # Regardless of whether the raster is passed in as a folder or a
        # single file, use its md5sum as a seed to the new raster's folder
        # name.
        seed = pygeoprocessing.testing.get_hash(file_list)
        # Casting to an int affords better compatibility between *nix and
        # Windows.
        seed = int(seed, 16)

        new_raster_dir = make_random_dir(workspace, seed, 'raster_',
                                         True)
        for raster_file in file_list:
            # raster_file may be a folder ... we can't copy a folder with
            # copyfile.
            if os.path.isfile(raster_file):
                file_basename = os.path.basename(raster_file)
                new_raster_uri = os.path.join(new_raster_dir, file_basename)
                shutil.copyfile(raster_file, new_raster_uri)

        return os.path.basename(new_raster_dir)


class UnsupportedFormat(Exception):
    pass


class NotAVector(Exception):
    pass


def _get_multi_part_ogr(filepath, workspace):
    """
    Collect multi-part vectors into a single folder within the workspace.

    This function currently only supports ESRI Shapefiles.  Any other formats
    will cause NotAVector to be raised.

    Parameters:
        filepath (string): The path to one of the member files of the
            multi-part file.
        workspace (string): The path to the output folder wherein a new folder
            will be created that contains the files in the multi-part vector.

    Returns:
        A string filepath to the new folder created within the `workspace`.
        This folder will have the name 'vector_[0-9]{6}'.
    """
    shapefile = ogr.Open(filepath)
    driver = shapefile.GetDriver()

    seed = pygeoprocessing.testing.get_hash(filepath)
    # Casting the md5sum seed to an int affords better
    # cross-platform.compatibility between *nix and Windows.
    seed = int(seed, 16)
    LOGGER.debug('Temp folder seed: %s', seed)
    new_vector_dir = make_random_dir(workspace, seed, 'vector_', True)

    if driver.name == 'ESRI Shapefile':
        LOGGER.debug('%s is an ESRI Shapefile', filepath)
        # get the layer name
        layer = shapefile.GetLayer()

        # get the layer files
        parent_folder_path = os.path.dirname(filepath)
        glob_pattern = os.path.join(parent_folder_path, '%s.*' %
                                    layer.GetName())
        layer_files = sorted(glob.glob(glob_pattern))
        LOGGER.debug('Layer files: %s', layer_files)

        layer_extensions = map(lambda x: os.path.splitext(x)[1],
                               layer_files)
        LOGGER.debug('Layer extensions: %s', layer_extensions)

        # It's not a shapefile if there's no file with a .shp extension.
        if '.shp' not in layer_extensions:
            shutil.rmtree(new_vector_dir)
            raise NotAVector()

        # copy the layer files to the new folder.
        for file_name in layer_files:
            file_basename = os.path.basename(file_name)
            new_filename = os.path.join(new_vector_dir, file_basename)
            shutil.copyfile(file_name, new_filename)
    else:
        raise UnsupportedFormat('%s is not a supported OGR Format',
                                driver.name)

    return os.path.basename(new_vector_dir)


def collect_parameters(parameters, archive_uri):
    """Collect an InVEST model's arguments into a dictionary and archive all
        the input data.

        parameters - a dictionary of arguments
        archive_uri - a URI to the target archive.

        Returns nothing."""

    parameters = parameters.copy()
    temp_workspace = pygeoprocessing.geoprocessing.temporary_folder()

    # For tracking existing files so we don't copy things twice
    files_found = {}

    def _get_multi_part(filepath):
        """
        Attempt to open a file at `filepath`, first as a gdal dataset, then
        as an OGR vector.  If the file can be opened by either library, bundle
        up the multipart file and save its contents to a new folder.  The path
        to this new folder is returned.

        Parameters:
            filepath (string): The path to some part of the multipart file on
            disk.

        Returns:
            Either a string or None.

            If a string is returned, it's the path to a newly created folder
            within `temp_workspace` (from the closure) that contains a copy of
            all files in the multi-part dataset.

            If None is returned, the filepath was not a multi-part file.
        """
        # If the user provides a multi-part file, wrap it into a folder and
        # grab that instead of the individual file.

        raster_obj = gdal.Open(filepath)
        if raster_obj is not None:
            # file is a raster
            raster_obj = None
            LOGGER.debug('%s is a raster', filepath)
            return _get_multi_part_gdal(filepath, temp_workspace)

        vector_obj = ogr.Open(filepath)
        if vector_obj is not None:
            # Need to check the driver name to be sure that this isn't a CSV.
            driver = vector_obj.GetDriver()
            if driver.name != 'CSV':
                # file is a vector
                vector_obj = None
                try:
                    return _get_multi_part_ogr(filepath, temp_workspace)
                except NotAVector:
                    # For some reason, the file actually turned out to not be a
                    # vector, so we just want to return from this function.
                    LOGGER.debug(('Thought %s was a vector, but I was '
                                  'wrong.'), filepath)

        # If the file is neither a raster nor a vector, return None.
        return None

    def _get_if_file(parameter):
        """
        If the input parameter exists on disk as a file or folder, collect
        the appropriate contents and return the path to the new folder.
        """
        try:
            # files_found is a dictionary from the outer scope
            uri = files_found[os.path.abspath(parameter)]
            LOGGER.debug('Found %s from a previous parameter', uri)
            return uri
        except KeyError:
            # we haven't found this file before, so we still need to process
            # it.
            pass

        # initialize the return_path
        return_path = None
        try:
            multi_part_folder = _get_multi_part(parameter)
            if multi_part_folder is not None:
                LOGGER.debug('%s is a multi-part file', parameter)
                return_path = multi_part_folder

            elif os.path.isfile(parameter):
                LOGGER.debug('%s is a single file', parameter)
                new_filename = os.path.basename(parameter)
                shutil.copyfile(parameter, os.path.join(temp_workspace,
                                new_filename))
                return_path = new_filename

            elif os.path.isdir(parameter):
                LOGGER.debug('%s is a directory', parameter)
                # parameter is a folder, so we want to copy the folder and all
                # its contents to temp_workspace.
                folder_name = os.path.basename(parameter)
                new_foldername = make_random_dir(temp_workspace, folder_name,
                                                 'data_', False)
                shutil.copytree(parameter, new_foldername)
                return_path = new_foldername

            else:
                # Parameter does not exist on disk.  Print an error to the
                # logger and move on.
                LOGGER.error('File %s does not exist on disk.  Skipping.',
                             parameter)
        except TypeError as e:
            # When the value is not a string.
            LOGGER.warn('%s', e)

        LOGGER.debug('Return path: %s', return_path)
        if return_path is not None:
            files_found[os.path.abspath(parameter)] = return_path
            return return_path

        LOGGER.debug('Returning original parameter %s', parameter)
        return parameter

    # Recurse through the parameters to locate any URIs
    #   If a URI is found, copy that file to a new location in the temp
    #   workspace and update the URI reference.
    #   Duplicate URIs should also have the same replacement URI.
    #
    # If a workspace or suffix is provided, ignore that key.
    LOGGER.debug('Keys: %s', parameters.keys())
    ignored_keys = []
    for key, restore_key in [
            ('workspace_dir', False),
            ('suffix', True),
            ('results_suffix', True)]:
        try:
            if restore_key:
                ignored_keys.append((key, parameters[key]))
                LOGGER.debug('tracking key %s', key)
            del parameters[key]
        except KeyError:
            LOGGER.warn(('Parameters missing the workspace key \'%s\'.'
                         ' Be sure to check your archived data'), key)

    types = {
        str: _get_if_file,
        unicode: _get_if_file,
    }
    new_args = format_dictionary(parameters, types)

    for (key, value) in ignored_keys:
        LOGGER.debug('Restoring %s: %s', key, value)
        new_args[key] = value

    LOGGER.debug('new arguments: %s', new_args)
    # write parameters to a new json file in the temp workspace
    param_file_uri = os.path.join(temp_workspace, 'parameters.json')
    parameter_file = open(param_file_uri, mode='w+')
    parameter_file.writelines(json.dumps(new_args))
    parameter_file.close()

    # archive the workspace.
    if archive_uri[-7:] == '.tar.gz':
        archive_uri = archive_uri[:-7]
    shutil.make_archive(archive_uri, 'gztar', root_dir=temp_workspace,
                        logger=LOGGER)


def extract_archive(workspace_dir, archive_uri):
    """Extract a .tar.gzipped file to the given workspace.

        workspace_dir - the folder to which the archive should be extracted
        archive_uri - the uri to the target archive

        Returns nothing."""

    archive = tarfile.open(archive_uri)
    archive.extractall(workspace_dir)
    archive.close()


def format_dictionary(input_dict, types_lookup={}):
    """Recurse through the input dictionary and return a formatted dictionary.

        As each element is encountered, the correct function to use is looked
        up in the types_lookup input.  If a type is not found, we assume that
        the element should be returned verbatim.

        input_dict - a dictionary to process
        types_lookup - a dictionary mapping types to functions.  These
            functions must take a single parameter of the type that is the
            key.  These functions must return a formatted version of the input
            parameter.

        Returns a formatted dictionary."""

    def format_dict(parameter):
        new_dict = {}
        for key, value in parameter.iteritems():
            try:
                new_dict[key] = types[value.__class__](value)
            except KeyError:
                new_dict[key] = value
        return new_dict

    def format_list(parameter):
        new_list = []
        for item in parameter:
            try:
                new_list.append(types[item.__class__](item))
            except KeyError:
                new_list.append(item)
        return new_list

    types = {
        dict: format_dict,
        list: format_list,
    }

    types.update(types_lookup)

    return format_dict(input_dict)


def extract_parameters_archive(workspace_dir, archive_uri, input_folder=None):
    """Extract the target archive to the target workspace folder.

        workspace_dir - a uri to a folder on disk.  Must be an empty folder.
        archive_uri - a uri to an archive to be unzipped on disk.  Archive must
            be in .tar.gz format.
        input_folder=None - either a URI to a folder on disk or None.  If None,
            temporary folder will be created and then erased using the atexit
            register.

        Returns a dictionary of the model's parameters for this run."""

    # create a new temporary folder just for the input parameters, if the user
    # has not provided one already.
    if input_folder is None:
        input_folder = pygeoprocessing.geoprocessing.temporary_folder()

    # extract the archive to the workspace
    extract_archive(input_folder, archive_uri)

    # get the arguments dictionary
    arguments_dict = json.load(open(os.path.join(input_folder,
                                                 'parameters.json')))

    def _get_if_uri(parameter):
        """If the parameter is a file, returns the filepath relative to the
        extracted workspace.  If the parameter is not a file, returns the
        original parameter."""
        try:
            temp_file_path = os.path.join(input_folder, parameter)
            if os.path.exists(temp_file_path) and not len(parameter) == 0:
                return temp_file_path
        except TypeError:
            # When the parameter is not a string
            pass
        except AttributeError:
            # when the parameter is not a string
            pass

        return parameter

    types = {
        str: _get_if_uri,
        unicode: _get_if_uri,
    }
    formatted_args = format_dictionary(arguments_dict, types)
    formatted_args[u'workspace_dir'] = workspace_dir

    return formatted_args
