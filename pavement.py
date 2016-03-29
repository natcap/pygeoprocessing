"""Automated build and development processes for PyGeoProcessing.

Execute ``paver --help`` for more information about available tasks.
"""
import textwrap
import sys

import paver.easy

paver.easy.options(
    bdist_windows=paver.easy.Bunch(
        bootstrap_file='bootstrap.py',
        upload=False,
        envname='release_env'
    )
)


@paver.easy.task
@paver.easy.cmdopts([
    ('upload', '', ('Upload the binaries to PyPI.  Requires a configured '
                    '.pypirc')),
    ('envname', '', ('The environment to use for building the binaries')),
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
    subprocess.check_output([home_dir, bindir, 'easy_install', 'numpy==1.6.1'])
"""

    output = virtualenv.create_bootstrap_script(
        textwrap.dedent(install_string))
    open(options.bdist_windows.bootstrap_file, 'w').write(output)

    paver.easy.sh(('{python} {bootstrap} {envname} --system-site-packages '
                   '--clear').format(
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
