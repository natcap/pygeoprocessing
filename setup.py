try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
license = open('LICENSE.txt').read()

requirements = ['cython>=0.17.1', 'numpy', 'scipy']

setup(
    name='natcap.geoprocessing',
    version=__version__,
    description="Geoprocessing routines for GIS",
    long_description=readme + '\n\n' + history,
    author='Rich Sharp',
    author_email='richsharp@stanford.edu',
    url='http://bitbucket.org/richpsharp/pygeoprocessing',
    packages=[
        'natcap.geoprocessing',
        'natcap.geoprocessing.routing',
    ],
    package_dir={'natcap.geoprocessing': 'natcap.geoprocessing'},
    include_package_data=True,
    install_requires=requirements,
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
