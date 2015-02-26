import os
import sys
try:
    from setuptools import setup

    # Monkeypatch os.link to prevent hard lnks from being formed.  Useful when
    # running tests across filesystems, like in our test docker containers.
    # Only an issue pre python 2.7.9.
    # See http://bugs.python.org/issue8876
    py_version = sys.version_info[0:3]
    if py_version[0] == 2 and py_version[1] <= 7 and py_version[2] < 9:
        del os.link
except ImportError:
    from distutils.core import setup

try:
    import versioning
    version = versioning.REPO.version
    _sdist = versioning.CustomSdist
    _build_py = versioning.CustomPythonBuilder
except ImportError:
    exec(open('pygeoprocessing/__init__.py', 'r').read())
    version = __version__
    try:
        from setuptools.command.sdist import sdist as _sdist
        from setuptools.command.build_py import build_py as _build_py
    except ImportError:
        from distutils.command.sdist import sdist as _sdist
        from distutils.command.build_py import build_py as _build_py


readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
license = open('LICENSE.TXT').read()

#requirements = ['cython>=0.17.1', 'numpy', 'scipy', 'nose>=1.0']
requirements = []

setup(
    name='pygeoprocessing',
    version=version,
    description="Geoprocessing routines for GIS",
    long_description=readme + '\n\n' + history,
    maintainer='Rich Sharp',
    maintainer_email='richsharp@stanford.edu',
    url='http://bitbucket.org/richpsharp/pygeoprocessing',
    packages=[
        'pygeoprocessing',
        'pygeoprocessing.routing',
        'pygeoprocessing.tests',
    ],
    package_dir={'pygeoprocessing': 'pygeoprocessing'},
    include_package_data=True,
    install_requires=requirements,
    setup_requires=['nose>=1.0'],
    cmdclass={
        'sdist': _sdist,
        'build_py': _build_py,
    },
    license=license,
    zip_safe=False,
    keywords='pygeoprocessing',
    classifiers=[
        'Intended Audience :: Developers',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 2 :: Only',
        'Topic :: Scientific/Engineering :: GIS'
    ]
)
