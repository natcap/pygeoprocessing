import os
import functools
from unittest import SkipTest

import svn
import svn.local
import svn.remote

def skipIfDataMissing():
    """
    Decorator for unittest.TestCase test functions.  Raises SkipTest if the
    pygeoprocessing data repository has not been cloned.
    """
    message = 'Data repo is not cloned'

    def data_repo_aware_skipper(item):

        @functools.wraps(item)
        def skip_if_data_not_cloned(self, *args, **kwargs):
            if _SVN_REPO.missing():
                raise SkipTest(message)
            item(self, *args, **kwargs)
        return skip_if_data_not_cloned
    return data_repo_aware_skipper

class SVNRepo:
    def __init__(self, local_path, remote_path, rev):
        self.local_path = local_path
        self.remote_path = remote_path
        self.rev = rev

    def missing(self):
        """
        Is the repository present?  Returns a boolean.
        """
        if os.path.exists(self.local_path):
            return False
        return True

    def get(self):
        """
        If the repository does not exist, clone it.
        """
        if not os.path.exists(self.local_path):
            self.checkout()
        else:
            self.update()

    def checkout(self):
        """
        Check out the repository to the target rev (self.rev).
        """
        repo = svn.remote.RemoteClient(self.remote_path)
        repo.checkout(self.local_path, self.rev)

    def update(self):
        """
        Update an existing repository to the target rev (self.rev).
        """
        repo = svn.local.LocalClient(self.local_path)
        repo.run_command(['update', '-r', str(self.rev)])

_NATCAP_SVN = 'svn://naturalcapitalproject.org/svn/'
_SVN_REPO = SVNRepo(
    local_path=os.path.join(os.path.dirname(__file__), 'data'),
    remote_path=_NATCAP_SVN + 'pygeoprocessing-test-data',
    rev=0
)

