"""setup.py module for PyGeoprocessing."""
import os
import platform
import subprocess

import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

# Read in requirements.txt and populate the python readme with the non-comment
# contents.
_REQUIREMENTS = [
    x for x in open('requirements.txt').read().split('\n')
    if not x.startswith('#') and len(x) > 0]
LONG_DESCRIPTION = open('README.rst').read().format(
    requirements='\n'.join(['    ' + r for r in _REQUIREMENTS]))
LONG_DESCRIPTION += '\n' + open('HISTORY.rst').read() + '\n'

include_dirs = [
    numpy.get_include(),
    'src/pygeoprocessing/routing',
    'src/pygeoprocessing/extensions']
if platform.system() == 'Windows':
    compiler_args = ['/std:c++20']
    compiler_and_linker_args = []
    if 'PYGEOPROCESSING_GDAL_LIB_PATH' not in os.environ:
        raise RuntimeError(
            'env variable PYGEOPROCESSING_GDAL_LIB_PATH is not defined. '
            'This env variable is required when building on Windows. If '
            'using conda to manage your gdal installation, you may set '
            'PYGEOPROCESSING_GDAL_LIB_PATH=%CONDA_PREFIX%/Library".')
    library_dirs = [os.path.join(
        os.environ["PYGEOPROCESSING_GDAL_LIB_PATH"].rstrip(), "lib")]
    include_dirs.append(os.path.join(
        os.environ["PYGEOPROCESSING_GDAL_LIB_PATH"].rstrip(), "include"))
else:
    compiler_args = [subprocess.run(
        ['gdal-config', '--cflags'], capture_output=True, text=True
    ).stdout.strip()]
    compiler_and_linker_args = ['-std=c++20']
    library_dirs = [subprocess.run(
        ['gdal-config', '--libs'], capture_output=True, text=True
    ).stdout.split()[0][2:]] # get the first argument which is the library path

setup(
    name='pygeoprocessing',
    description="PyGeoprocessing: Geoprocessing routines for GIS",
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/x-rst',
    url='https://github.com/natcap/pygeoprocessing',
    packages=[
        'pygeoprocessing',
        'pygeoprocessing.routing',
        'pygeoprocessing.multiprocessing',
        'pygeoprocessing.extensions'
    ],
    package_dir={
        'pygeoprocessing': 'src/pygeoprocessing'
    },
    setup_requires=['cython', 'numpy'],
    include_package_data=True,
    install_requires=_REQUIREMENTS,
    license='BSD',
    zip_safe=False,
    ext_modules=cythonize([
        Extension(
            name="pygeoprocessing.routing.routing",
            sources=["src/pygeoprocessing/routing/routing.pyx"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=['gdal'],
            extra_compile_args=compiler_args + compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            language="c++",
        ),
        Extension(
            "pygeoprocessing.routing.watershed",
            sources=["src/pygeoprocessing/routing/watershed.pyx"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=['gdal'],
            extra_compile_args=compiler_args + compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            language="c++",
        ),
        Extension(
            "pygeoprocessing.geoprocessing_core",
            sources=[
                'src/pygeoprocessing/geoprocessing_core.pyx'],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=['gdal'],
            extra_compile_args=compiler_args + compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            language="c++"
        ),
    ])
)
