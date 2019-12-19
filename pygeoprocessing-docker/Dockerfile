FROM conda/miniconda3

# this builds natcap/pygeoprocessing-test, should be run in the same directory
# as the pygeoprocessing repository root like so:
# docker run -it --rm -v `cwd`:/usr/local/pygeoprocessing natcap/pygeoprocessing-test:[TAG]
# the [TAG] is defined in the .travis.yml file

USER root
RUN apt-get update \
&& apt-get install -y \
    build-essential \
    git \
    libspatialindex-c4v5 \
    mercurial \
&& rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
RUN conda create -y --name py37 python=3.7 && conda clean -a -y
RUN conda run -v -n py37 conda install -c conda-forge gdal=2.4.1
RUN conda run -v -n py37 pip install --no-cache-dir \
    cython \
    ecoshard==0.3.1 \
    pytest \
    pytest-cov \
    mock \
    numpy \
    retrying \
    rtree \
    scipy \
    setuptools-scm \
    shapely \
    sympy \
    taskgraph && conda clean -a -y

RUN conda init bash && echo "source activate py37" > ~/.bashrc
WORKDIR /usr/local/pygeoprocessing
RUN echo "python setup.py install && pytest -xv tests" > /usr/local/run_pgp_tests.sh && chmod a+x /usr/local/run_pgp_tests.sh
ENTRYPOINT /bin/bash -xci /usr/local/run_pgp_tests.sh
