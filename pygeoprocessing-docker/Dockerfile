FROM conda/miniconda3

# install build essentials & mercurial then delete the cache to minimize
# the docker size
RUN apt-get update \
&& apt-get install -y \
    build-essential \
    mercurial \
&& rm -rf /var/lib/apt/lists/*

# make a new user called pygeoprocessing to avoid running everything as root
ENV SETUSER pygeoprocessing
RUN useradd -m $SETUSER
USER $SETUSER
WORKDIR /home/$SETUSER

# pre-fetch pygeoprocessing in anticipation of a faster build step
RUN hg clone https://bitbucket.org/natcap/pygeoprocessing -r develop

# 1) build an environment for both python 3.6 and 3.7, install as much as
#    possible from pip to minimize the expensive conda dep resolution
#    algorithm
# 2) note the pip install command does not build an index to mimize docker
#    image size
# 3) pre-build a development build of pygeoprocessing in anticipation of a
#    faster compile step
# 4) clean the conda cache to minimize docker image size
RUN /bin/bash -c "for i in py36,3.6 py37,3.7; do IFS=\",\"; set -- \$i; \
    conda create -y --name \$1 -c conda-forge \
        gdal=2.4.2 \
        python=\$2 \
        rtree \
        shapely; \
    conda run -v -n \$1 pip install --no-cache-dir \
        cython \
        pytest \
        pytest-cov \
        mock \
        numpy \
        scipy \
        setuptools-scm \
        sympy; \
    pushd pygeoprocessing; \
    conda run -v -n \$1 python setup.py install; \
    popd; \
    done; \
    conda clean -a -y;"
