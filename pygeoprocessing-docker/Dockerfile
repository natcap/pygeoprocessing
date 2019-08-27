FROM conda/miniconda3

# install build essentials & mercurial then delete the cache to minimize
# the docker size:
#   build-essential to build cython extensions
#   libspatialindex-c4v5 for the rtree python extension
#   mercurial for source control repo of PyGeoprocessing
RUN apt-get update \
&& apt-get install -y \
    build-essential \
    libspatialindex-c4v5 \
    mercurial \
&& rm -rf /var/lib/apt/lists/*

# update conda to fix conda error reported that is not an error
RUN conda update -n base -c defaults conda && conda clean -a -y

# make a new user called pygeoprocessing to avoid running everything as root
ENV SETUSER pygeoprocessing
RUN useradd -m $SETUSER
USER $SETUSER
WORKDIR /home/$SETUSER
COPY install-conda-environment.sh .
COPY install-pip-packages.sh .

# pre-fetch pygeoprocessing in anticipation of a faster build step
RUN hg clone https://bitbucket.org/natcap/pygeoprocessing -r develop

# build the python environments for 3.6 and 3.7 using conda, pip, and
# python setup.py install
RUN /bin/bash ./install-conda-environment.sh 3 7
RUN /bin/bash ./install-pip-packages.sh 3 7
RUN /bin/bash -c "\
    pushd pygeoprocessing; \
    conda run -v -n 37 python setup.py install; \
    popd"

RUN /bin/bash ./install-conda-environment.sh 3 6
RUN /bin/bash ./install-pip-packages.sh 3 6
RUN /bin/bash -c "\
    pushd pygeoprocessing; \
    conda run -v -n 36 python setup.py install; \
    popd"
