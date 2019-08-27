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

RUN conda init bash
RUN /bin/bash -c "for i in py36,3.6 py37,3.7; do IFS=\",\"; set -- \$i; \
    conda create -q --name \$1 python=\$2;\
    conda run -n \$1 pip install \
        cython \
        pytest \
        pytest-cov \
        mock \
        numpy \
        setuptools-scm \
        scipy \
        sympy; \
    conda run -n \$1 conda install -c conda-forge \
        gdal=2.4.2 \
        rtree \
        shapely; \
    hg clone https://bitbucket.org/natcap/pygeoprocessing -r develop; \
    pushd pygeoprocessing; \
    conda run -n \$1 python setup.py install; \
    popd; \
    done"
