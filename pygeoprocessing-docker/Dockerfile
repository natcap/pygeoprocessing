FROM atlassian/default-image:2

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && /bin/bash ./Miniconda3-latest-Linux-x86_64.sh -b -p/usr/local/miniconda
ENV PATH="/usr/local/miniconda/bin":$PATH

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

# build the python environments for 3.6 and 3.7 using conda, pip, and
# python setup.py install
COPY requirements.txt .
RUN /bin/bash -c "\
    conda create -y --name py37 python=3.7 \
    && conda clean -a -y"
RUN /bin/bash -c "conda run -v -n py37 conda install -c conda-forge gdal=2.4.1"
RUN /bin/bash -c "conda run -v -n py37 pip install -r requirements.txt"
RUN /bin/bash -c "conda run -v -n py37 pip install pytest mock"


RUN /bin/bash -c \
    "conda create -y --name py36 python=3.6 \
    && conda clean -a -y"
RUN /bin/bash -c "conda run -v -n py36 conda install -c conda-forge gdal=2.4.1"
RUN /bin/bash -c "conda run -v -n py36 pip install -r requirements.txt"
RUN /bin/bash -c "conda run -v -n py36 pip install pytest mock"
