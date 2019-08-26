FROM debian:buster

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        gfortran \
        git \
        libc6-dev \
        libc6 \
        libblas3 \
        liblapack3 \
        mercurial \
        subversion \
        tox \
        unzip \
        wget

ENV SETUSER myuser
ENV PATH /home/$SETUSER/anaconda3/bin:$PATH
ENV CONDA_ENV_NAME py36
RUN useradd -m $SETUSER
USER $SETUSER
WORKDIR /home/$SETUSER

RUN wget -O Anaconda3-2019.07-Linux-x86_64.sh https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh \
    && /bin/bash ./Anaconda3-2019.07-Linux-x86_64.sh -b \
    && rm ./Anaconda3-2019.07-Linux-x86_64.sh

RUN /home/$SETUSER/anaconda3/condabin/conda create -q --name py36 python=3.6
RUN conda run -n py36 pip install sympy pytest pytest-cov setuptools-scm cython numpy scipy mock
RUN conda run -n py36 conda install -c conda-forge shapely rtree gdal=2.4.2
RUN /home/$SETUSER/anaconda3/condabin/conda create -q --name py37 python=3.7
RUN conda run -n py37 pip install sympy pytest pytest-cov setuptools-scm cython numpy scipy mock
RUN conda run -n py37 conda install -c conda-forge shapely rtree gdal=2.4.2

# preclone and precompile pygeoprocessing so the build goes a little faster
RUN /bin/bash -c "hg clone https://bitbucket.org/natcap/pygeoprocessing -r develop; cd pygeoprocessing; conda run -n py36 python setup.py install"
RUN /bin/bash -c "hg clone https://bitbucket.org/natcap/pygeoprocessing -r develop; cd pygeoprocessing; conda run -n py37 python setup.py install"
CMD bash
