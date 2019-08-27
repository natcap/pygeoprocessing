#!/bin/bash

# install pygeoprocessing pip dependencies and avoid building a cache
conda run -v -n py$1$2 pip install --no-cache-dir \
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
conda clean -a -y
