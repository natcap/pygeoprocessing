"""Determine the version of the package based on source control."""
import subprocess
import logging
import platform
import sys
import tempfile
import os
import atexit

if platform.system() != 'Windows':
    import shutil
    from shutil import WindowsError

try:
    from setuptools.command.sdist import sdist as _sdist
    from setuptools.command.build_py import build_py as _build_py
except ImportError:
    from distutils.command.sdist import sdist as _sdist
    from distutils.command.build_py import build_py as _build_py


LOGGER = logging.getLogger('versioning')
LOGGER.setLevel(logging.ERROR)

class VCSQuerier(object):
    def _run_command(self, cmd):
        """Run a subprocess.Popen command.  This function is intended for internal
        use only and ensures a certain degree of uniformity across the various
        subprocess calls made in this module.

        cmd - a python string to be executed in the shell.

        Returns a python bytestring of the output of the input command."""
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.stdout.read().replace('\n', '')

    @property
    def tag_distance(self):
        pass

    @property
    def build_id(self):
        pass

    @property
    def latest_tag(self):
        pass

    @property
    def branch(self):
        pass

    @property
    def py_arch(self):
        """This function gets the python architecture string.  Returns a string."""
        return platform.architecture()[0]

    @property
    def release_version(self):
        """This function gets the release version.  Returns either the latest tag
        (if we're on a release tag) or None, if we're on a dev changeset."""
        if self.tag_distance == 0:
            return self.latest_tag
        return None

    @property
    def version(self):
        """This function gets the module's version string.  This will be either the
        dev build ID (if we're on a dev build) or the current tag if we're on a
        known tag.  Either way, the return type is a string."""
        release_version = self.release_version
        if release_version == None:
            return self.build_dev_id(self.build_id)
        return release_version

    def build_dev_id(self, build_id=None):
        """This function builds the dev version string.  Returns a string."""
        if build_id == None:
            build_id = self.build_id
        return '%s' % (build_id)

    def get_architecture_string(self):
        """Return a string representing the operating system and the python
        architecture on which this python installation is operating (which may be
        different than the native processor architecture.."""
        return '%s%s' % (platform.system().lower(),
            platform.architecture()[0][0:2])

    @property
    def pep440(self):
        """Return a string representing the PEP440 version string for the
        current version of the package."""
        latest_tag = self.latest_tag
        if latest_tag == 'null':
            latest_tag = '0.0'
        return "%s.dev%s" % (latest_tag, self.tag_distance)


class HgRepo(VCSQuerier):
    HG_CALL = 'hg log -r . --config ui.report_untrusted=False'

    @property
    def build_id(self):
        """Call mercurial with a template argument to get the build ID.  Returns a
        python bytestring."""
        cmd = self.HG_CALL + ' --template "{branch}-{node|short}"'
        return self._run_command(cmd)

    @property
    def tag_distance(self):
        """Call mercurial with a template argument to get the distance to the latest
        tag.  Returns an int."""
        cmd = self.HG_CALL + ' --template "{latesttagdistance}"'
        return int(self._run_command(cmd))

    @property
    def latest_tag(self):
        """Call mercurial with a template argument to get the latest tag.  Returns a
        python bytestring."""
        cmd = self.HG_CALL + ' --template "{latesttag}"'
        return self._run_command(cmd)

    @property
    def branch(self):
        """Call mercurial to figure out which branch we're on'  Resturns a python
        bytestring."""
        return self._run_command('hg branch')

    @property
    def version(self):
        """Return the version string, where the format is dependent on the
        branch and tag."""
        if self.branch == 'master':
            if self.tag_distance == 0:
                return self.latest_tag
            return self.latest_tag + '+%s' % self.tag_distance
        else:
            cmd = self.HG_CALL + ' --template "{branch}-{node|short}"'
            return self._run_command(cmd)

REPO = HgRepo()

def _build_data():
    """Returns a dictionary of relevant build data."""
    data = {
        'release': REPO.latest_tag,
        'build_id': REPO.build_id,
        'py_arch': REPO.py_arch,
        'version_str': REPO.version,
        'branch': REPO.branch,
    }
    return data

def write_build_info(source_file_uri, ver_type='dvcs'):
    """Write the build information to the file specified as `source_file_uri`.
        source_file_uri (string): The file to write version info to.
        ver_type='dvcs' (string): One of 'dvcs', or 'pep440'.
    """
    temp_file_uri = _temporary_filename()
    temp_file = open(temp_file_uri, 'w+')

    # write whatever's in the file first, then write the build info.
    source_file = open(os.path.abspath(source_file_uri))
    for line in source_file:
        temp_file.write(line)

    if ver_type == 'dvcs':
        version = REPO.version
    elif ver_type == 'pep440':
        version = REPO.pep440
    else:
        raise RuntimeError('Version type %s not known' % ver_type)
    temp_file.write('__version__ = "%s"\n' % version)
    build_information = _build_data()
    temp_file.write("build_data = %s\n" % str(build_information.keys()))
    for key, value in sorted(build_information.iteritems()):
        temp_file.write("%s = '%s'\n" % (key, value))
    source_file.close()

    temp_file.flush()
    temp_file.close()

    source_file_removed = False
    for index in range(10):
        try:
            os.remove(source_file_uri)
            source_file_removed = True
        except WindowsError:
            time.sleep(0.25)
        except OSError:
            print 'Not removing existing file, file not found.'
            source_file_removed = True
            break
    if not source_file_removed:
        raise IOError('Could not remove %s' % source_file_uri)

    # This whole block of try/except logic is an attempt to mitigate a problem
    # we've experienced on Windows, where a file had not quite been deleted
    # before we tried to copy the new file over the old one.
    file_copied = False
    for index in range(10):
        try:
            shutil.copyfile(temp_file_uri, source_file_uri)
            file_copied = True
            break  # if we successfully copy, end the loop.
        except WindowsError:
            time.sleep(0.25)

    if not file_copied:
        raise IOError('Could not copy %s to %s', temp_file_uri,
            source_file_uri)

def _temporary_filename():
    """Returns a temporary filename using mkstemp. The file is deleted
        on exit using the atexit register.  This function was migrated from
        the invest-3 raster_utils file, rev 11354:1029bd49a77a.

        returns a unique temporary filename"""

    file_handle, path = tempfile.mkstemp()
    os.close(file_handle)

    def remove_file(path):
        """Function to remove a file and handle exceptions to register
            in atexit"""
        try:
            os.remove(path)
        except OSError as exception:
            #This happens if the file didn't exist, which is okay because maybe
            #we deleted it in a method
            pass

    atexit.register(remove_file, path)
    return path

geoprocessing_init = lambda base_dir: os.path.join(base_dir,
    'pygeoprocessing', '__init__.py')

class CustomSdist(_sdist):
    """Custom source distribution builder.  Builds a source distribution via the
    distutils sdist command, but then writes the adept version information to
    the temp source tree before everything is archived for distribution."""
    def make_release_tree(self, base_dir, files):
        _sdist.make_release_tree(self, base_dir, files)

        # Write version information (which is derived from the adept mercurial
        # source tree) to the build folder's copy of adept.__init__.
        filename = geoprocessing_init(base_dir)
        print 'Writing version data to %s' % filename
        write_build_info(filename, 'pep440')

class CustomPythonBuilder(_build_py):
    """Custom python build step for distutils.  Builds a python distribution in
    the specified folder ('build' by default) and writes the adept version
    information to the temporary source tree therein."""
    def run(self):
        _build_py.run(self)

        # Write version information (which is derived from the mercurial
        # source tree) to the build folder's copy of adept.__init__.
        filename = geoprocessing_init(self.build_lib)
        print 'Writing version data to %s' % filename
        write_build_info(filename, 'pep440')

if __name__ == '__main__':
    print REPO.pep440

