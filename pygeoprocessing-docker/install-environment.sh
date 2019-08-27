#!/bin/bash

# 2) note the pip install command does not build an index to mimize docker
#    image size
# 3) pre-build a development build of pygeoprocessing in anticipation of a
#    faster compile step
# 4) clean the conda cache to minimize docker image size

# 1) build an environment for both python 3.6 and 3.7, install as much as
#    possible from pip to minimize the expensive conda dep resolution
#    algorithm
conda create -y --name $1$2 -c conda-forge \
    gdal=2.4.2 \
    python=$1.$2
conda run -v -n $1$2 pip install --no-cache-dir \
    cython \
    pytest \
    pytest-cov \
    mock \
    numpy \
    rtree \
    scipy \
    setuptools-scm \
    shapely \
    sympy
pushd pygeoprocessing
conda run -v -n \$1 python setup.py install
popd
conda clean -a -y
