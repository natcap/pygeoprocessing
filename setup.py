"""setup.py module for PyGeoprocessing."""
import platform

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

# Since OSX Mavericks, the stdlib has been renamed.  So if we're on OSX, we
# need to be sure to define which standard c++ library to use.  I don't have
# access to a pre-Mavericks mac, so hopefully this won't break on someone's
# older system.  Tested and it works on Mac OSX Catalina.
compiler_and_linker_args = []
if platform.system() == 'Darwin':
    compiler_and_linker_args = ['-stdlib=libc++']

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
            include_dirs=[
                numpy.get_include(),
                'src/pygeoprocessing/routing'],
            extra_compile_args=compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            language="c++",
        ),
        Extension(
            "pygeoprocessing.routing.watershed",
            sources=["src/pygeoprocessing/routing/watershed.pyx"],
            include_dirs=[
                numpy.get_include(),
                'src/pygeoprocessing/routing'],
            extra_compile_args=compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            language="c++",
        ),
        Extension(
            "pygeoprocessing.geoprocessing_core",
            sources=[
                'src/pygeoprocessing/geoprocessing_core.pyx'],
            include_dirs=[numpy.get_include()],
            extra_compile_args=compiler_and_linker_args,
            extra_link_args=compiler_and_linker_args,
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            language="c++"
        ),
    ])
)
