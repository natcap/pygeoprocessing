"""PyGeoprocessing pavement script."""
import sys
import textwrap
import warnings
import json
import os
import subprocess
import logging
import glob

import paver.easy
import paver.path
import paver.svn
import paver.virtual

logging.basicConfig(
    format='%(asctime)s %(name)-18s %(levelname)-8s %(message)s',
    level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')
LOGGER = logging.getLogger('pavement')
logging.getLogger('pip').setLevel(logging.ERROR)

_VIRTUAL_ENV_DIR = 'dev_env'

paver.easy.options(
    bdist_windows=paver.easy.Bunch(
        bootstrap_file='bootstrap.py',
        upload=False,
        envname='release_env'
    )
)


class Repository(object):
    """Abstract class representing a version-controlled repository."""

    tip = ''  # The string representing the latest revision
    statedir = ''  # Where the SCM stores its data, relative to repo root
    cmd = ''  # The command-line exe to call.

    def __init__(self, local_path, remote_url):
        """Initialize the Repository instance.

        Parameters:
            local_path (string): The filepath on disk to the repo
                (relative to pavement.py)
            remote_url (string): The string URL to use to identify the repo.
                Used for clones, updates.

        Returns:
            An instance of Repository.
        """
        self.local_path = local_path
        self.remote_url = remote_url

    def get(self, rev):
        """Update to the target revision.

        Parameters:
            rev (string): The revision to update the repo to.

        Returns:
            None
        """
        if not self.ischeckedout():
            self.clone()
        else:
            print 'Repository %s exists' % self.local_path

        # If we're already updated to the correct rev, return.
        if self.at_expected_rev():
            print 'Repo %s is in an expected state' % self.local_path
            return

        # Try to update to the correct revision.  If we can't pull, then
        # update.
        try:
            self.update(rev)
        except paver.easy.BuildFailure:
            print (
                'Repo %s not found, falling back to fresh clone and update'
                % self.local_path)
            # When the current repo can't be updated because it doesn't know
            # the change we want to update to
            self.clone(rev)

    def ischeckedout(self):
        """Identify whether the repository is checked out on disk."""
        return os.path.exists(os.path.join(self.local_path, self.statedir))

    def clone(self, rev=None):
        """Clone the repository from the remote URL.

        This method is to be overridden in a subclass with the SCM-specific
        commands.

        Parameters:
            rev=None (string or None): The revision to update to.  If None,
                the revision will be fetched from versions.json.

        Returns:
            None

        Raises:
            NotImplementedError: When the method is not overridden in a
                subclass
            BuildFailure: When an error is encountered in the clone
                command.
        """
        raise NotImplementedError

    def update(self, rev=None):
        """Update the repository to a revision.

        This method is to be overridden in a subclass with the SCM-specific
        commands.

        Parameters:
            rev=None (string or None): The revision to update to.  If None,
                the revision will be fetched from versions.json.

        Returns:
            None

        Raises:
            NotImplementedError: When the method is not overridden in a
                subclass
            BuildFailure: When an error is encountered in the clone command.
        """
        raise NotImplementedError

    def expected_version(self):
        """Get this repository's expected version from versions.json.

        Returns:
            A string representation of the tracked version.
        """
        tracked_rev = json.load(open('versions.json'))[self.local_path]
        return tracked_rev

    def at_expected_rev(self):
        """Identify whether the Repository is at the expected revision.

        Returns:
            A boolean.
        """
        if not self.ischeckedout():
            return False

        expected_version = self.format_rev(self.expected_version())
        return self.current_rev() == expected_version

    def format_rev(self, rev):
        """Get the uniquely-identifiable commit ID of `rev`.

        This is particularly useful for SCMs that have multiple ways of
        identifying commits.

        Parameters:
            rev (string): The revision to identify.

        Returns:
            The string id of the commit.

        Raises:
            NotImplementedError: When the method is not overridden in a
                subclass
            BuildFailure: When an error is encountered in the clone command.
        """
        raise NotImplementedError

    def current_rev(self):
        """Fetch the current revision of the repository on disk.

        This method should be overridden in a subclass with the SCM-specific
        command(s).

        Raises:
            NotImplementedError: When the method is not overridden in a
                subclass
            BuildFailure: When an error is encountered in the clone command.
        """
        raise NotImplementedError


class SVNRepository(Repository):
    """Subversion repository handler."""

    tip = 'HEAD'
    statedir = '.svn'
    cmd = 'svn'

    def at_expected_rev(self):
        """Determine repo status from `svn status`.

        Overridden from Repository.at_expected_rev(...).  SVN info does not
        correctly report the status of the repository in the version number,
        so we must parse the output of `svn status` to see if a checkout or
        update was interrupted.

        Returns True if the repository is up-to-date.  False if not.
        """
        # Parse the output of SVN status.
        repo_status = paver.easy.sh(
            'svn status', cwd=self.local_path, capture=True)
        for line in repo_status:
            # If the line is empty, skip it.
            if line.strip() == '':
                continue

            if line.split()[0] in ['!', 'L']:
                print 'Checkout or update incomplete!  Repo NOT at known rev.'
                return False

        print 'Status ok.'
        return Repository.at_expected_rev(self)

    def _cleanup_and_retry(self, cmd, *args, **kwargs):
        """Run SVN cleanup."""
        for retry in [True, False]:
            try:
                cmd(*args, **kwargs)
            except paver.easy.BuildFailure as failure:
                if retry and self.ischeckedout():
                    # We should only retry if the repo is checked out.
                    print 'Cleaning up SVN repository %s' % self.local_path
                    paver.easy.sh('svn cleanup', cwd=self.local_path)
                    # Now we'll try the original command again!
                else:
                    # If there was a failure before the repo is checked out,
                    # then the issue is probably identified in stderr.
                    raise failure

    def clone(self, rev=None):
        """Clone SVN repo."""
        if rev is None:
            rev = self.expected_version()
        self._cleanup_and_retry(paver.svn.checkout, self.remote_url,
                                self.local_path, revision=rev)

    def update(self, rev=None):
        """Update SVN to revision."""
        # check that the repository URL hasn't changed.  If it has, update to
        # the new URL
        local_copy_info = paver.svn.info(self.local_path)
        if local_copy_info.repository_root != self.remote_url:
            paver.easy.sh('svn switch --relocate {orig_url} {new_url}'.format(
                orig_url=local_copy_info.repository_root,
                new_url=self.remote_url), cwd=self.local_path)

        self._cleanup_and_retry(paver.svn.update, self.local_path, rev)

    def current_rev(self):
        """Return current revision."""
        try:
            return paver.svn.info(self.local_path).revision
        except AttributeError:
            # happens when we're in a dry run
            # In this case, paver.svn.info() returns an empty Bunch object.
            # Returning 'Unknown' for now until we implement something more
            # stable.
            warnings.warn('SVN version info does not work when in a dry run')
            return 'Unknown'

    def format_rev(self, rev):
        """Match abstract implementation that trivially returns rev."""
        return rev

REPOS_DICT = {
    'test-data': SVNRepository(
        'data/pygeoprocessing-test-data',
        'svn://scm.naturalcapitalproject.org/svn/pygeoprocessing-test-data'),
}


@paver.easy.task
def env():
    """Build development environment."""
    subprocess.call('virtualenv --system-site-packages %s' % _VIRTUAL_ENV_DIR)
    subprocess.call(r'dev_env\Scripts\python setup.py install')
    subprocess.call(r'dev_env\Scripts\pip install nose -I')
    subprocess.call(
        'dev_env\\Scripts\\python -c "import pygeoprocessing; '
        'print \'***\\npygeoprocessing version: \' + '
        'pygeoprocessing.__version__ + \'\\n***\'"')
    print (
        "Installed virtualenv launch with:\n.\\%s\\Scripts\\activate" %
        _VIRTUAL_ENV_DIR)


@paver.easy.task
def clean():
    """Clean build environment."""
    folders_to_rm = ['build', 'dist', 'tmp', 'bin', 'data', _VIRTUAL_ENV_DIR]

    for folder in folders_to_rm:
        for globbed_dir in glob.glob(folder):
            paver.path.path(globbed_dir).rmtree()


@paver.easy.task
def fetch():
    """Fetch data repositories."""
    for known_repo_path, repo_obj in REPOS_DICT.iteritems():
        print 'Fetching {path}'.format(path=known_repo_path)
        repo_obj.get(repo_obj.expected_version())


@paver.easy.task
@paver.easy.cmdopts([
    ('upload', '', ('Upload the binaries to PyPI.  Requires a configured '
                    '.pypirc')),
    ('envname=', '', ('The environment to use for building the binaries')),
])
def bdist_windows(options):
    """Build a Windows wheel of pygeoprocessing against an early numpy ABI."""
    import virtualenv
    # paver provides paver.virtual.bootstrap(), but this does not afford the
    # degree of control that we want and need with installing needed packages.
    # We therefore make our own bootstrapping function calls here.
    install_string = """
import os, subprocess, platform
def after_install(options, home_dir):
    if platform.system() == 'Windows':
        bindir = 'Scripts'
    else:
        bindir = 'bin'
    subprocess.check_output([os.path.join(home_dir, bindir, 'easy_install'),
                             'numpy==1.6.1'])
"""

    output = virtualenv.create_bootstrap_script(
        textwrap.dedent(install_string))
    open(options.bdist_windows.bootstrap_file, 'w').write(output)

    paver.easy.sh((
        '{python} {bootstrap} {envname} '
        '--system-site-packages --clear').format(
            python=sys.executable,
            envname=options.bdist_windows.envname,
            bootstrap=options.bdist_windows.bootstrap_file))

    @paver.virtual.virtualenv(options.bdist_windows.envname)
    def _build_files():
        upload_string = ''
        if options.bdist_windows.upload:
            upload_string = 'upload'

        paver.easy.sh('python setup.py sdist bdist_wheel {upload}'.format(
            upload=upload_string))

    _build_files()
