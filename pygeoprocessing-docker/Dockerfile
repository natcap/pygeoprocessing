FROM conda/miniconda3

RUN apt-get update \
&& apt-get install -y \
    build-essential \
    mercurial \
&& rm -rf /var/lib/apt/lists/*

ENV SETUSER pygeoprocessing
RUN useradd -m $SETUSER
USER $SETUSER
WORKDIR /home/$SETUSER

RUN hg clone https://bitbucket.org/natcap/pygeoprocessing -r develop
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
