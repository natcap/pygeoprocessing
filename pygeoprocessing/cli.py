
import argparse
import sys
import os

from pygeoprocessing.testing import utils

def snapshot_cli(args=None):
    """CLI interface to snapshot a directory via
    pygeoprocessing.testing.utils.snapshot_folder()

    Accessed as an entry point via setuptools or else as a function that
    is called with a list of string arguments.

    Parameters:
        args (None or list of strings): If None, `sys.argv[1:]` will be used.
            Otherwise, a list of strings is expected, where the strings
            are each parameter (optional and position) that would be passed
            to the program normally through the command line.

    Returns:
        Nothing.
    """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog='pygeo-snapshot',
        description=(
            'Command-line access to common testing functionality.')
        )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        dest='overwrite',
        help=('Overwrite an existing snapshot file, if one exists')
    )
    parser.add_argument(
        'directory',
        help='The folder to snapshot'
    )
    parser.add_argument(
        'snapshotfile',
        help='Where to save the output file'
    )
    user_args = parser.parse_args(args)
    if os.path.exists(user_args.snapshotfile) and user_args.overwrite is False:
        parser.exit(('E: Snapshot file already exists.  Use --overwrite to '
                'ignore'))
    utils.snapshot_folder(user_args.directory, user_args.snapshotfile)


