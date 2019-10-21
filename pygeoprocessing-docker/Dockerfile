FROM conda/miniconda3

# install build essentials & mercurial then delete the cache to minimize
# the docker size:
#   build-essential to build cython extensions
#   libspatialindex-c4v5 for the rtree python extension
#   mercurial for source control repo of PyGeoprocessing
USER root
RUN apt-get update \
&& apt-get install -y \
    build-essential \
    libspatialindex-c4v5 \
    mercurial \
&& rm -rf /var/lib/apt/lists/*


# build the python environments for 3.6 and 3.7 using conda, pip, and
# python setup.py install
RUN /bin/bash -c "\
    conda create -y --name py37 python=3.7 \
    && conda clean -a -y"
RUN /bin/bash -c "conda run -v -n py37 conda install -c conda-forge gdal=2.4.1 --target=/src"
RUN /bin/bash -c "conda run -v -n py37 pip install -r requirements.txt --target=/src"

RUN /bin/bash -c \
    "conda create -y --name py36 python=3.6 \
    && conda clean -a -y"
RUN /bin/bash -c "conda run -v -n py36 conda install -c conda-forge gdal=2.4.1 --target=/src"
RUN /bin/bash -c "conda run -v -n py36 pip install -r requirements.txt --target=/src"

RUN conda init bash && echo "source activate py37" > ~/.bashrc

WORKDIR /usr/local/pgp_workdir
ENTRYPOINT ["/bin/bash", "-l", "-c"]
