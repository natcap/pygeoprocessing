import os
import imp
#import natcap.geoprocessing
try:
    from setuptools import setup
    from setuptools.command.sdist import sdist as _sdist
    from setuptools.command.build_py import build_py as _build_py
except ImportError:
    from distutils.core import setup
    from distutils.command.sdist import sdist as _sdist
    from distutils.command.build_py import build_py as _build_py

geoprocessing = imp.load_source('natcap.geoprocessing',
    'natcap/geoprocessing/__init__.py')

class CustomSdist(_sdist):
    """Custom source distribution builder.  Builds a source distribution via the
    distutils sdist command, but then writes the adept version information to
    the temp source tree before everything is archived for distribution."""
    def make_release_tree(self, base_dir, files):
        _sdist.make_release_tree(self, base_dir, files)

        # Write version information (which is derived from the adept mercurial
        # source tree) to the build folder's copy of adept.__init__.
        filename = os.path.join(base_dir, 'natcap', 'geoprocessing', '__init__.py')
        print 'Writing version data to %s' % filename
        geoprocessing.write_build_info(filename)

class CustomPythonBuilder(_build_py):
    """Custom python build step for distutils.  Builds a python distribution in
    the specified folder ('build' by default) and writes the adept version
    information to the temporary source tree therein."""
    def run(self):
        _build_py.run(self)

        # Write version information (which is derived from the mercurial
        # source tree) to the build folder's copy of adept.__init__.
        filename = os.path.join(self.build_lib, 'natcap', 'geoprocessing', '__init__.py')
        print 'Writing version data to %s' % filename
        geoprocessing.write_build_info(filename)

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
license = open('LICENSE.txt').read()

#requirements = ['cython>=0.17.1', 'numpy', 'scipy', 'nose>=1.0']
requirements = []

setup(
    name='natcap.geoprocessing',
    version=geoprocessing.__version__,
    description="Geoprocessing routines for GIS",
    long_description=readme + '\n\n' + history,
    maintainer='Rich Sharp',
    maintainer_email='richsharp@stanford.edu',
    url='http://bitbucket.org/richpsharp/pygeoprocessing',
    namespace_packages=['natcap'],
    packages=[
        'natcap.geoprocessing',
        'natcap.geoprocessing.routing',
        'natcap.geoprocessing.tests',
    ],
    package_dir={'natcap/geoprocessing': 'natcap.geoprocessing'},
    include_package_data=True,
    install_requires=requirements,
    setup_requires=['nose>=1.0'],
    cmdclass={
        'sdist': CustomSdist,
        'build_py': CustomPythonBuilder,
    },
    license=license,
    zip_safe=False,
    keywords='natcap.geoprocessing',
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
