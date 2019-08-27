#!/bin/bash

# build an environment for given python version $1.$2 with gdal
conda create -y --name $1$2 -c conda-forge \
    gdal=2.4.2 \
    python=$1.$2
conda clean -a -y
