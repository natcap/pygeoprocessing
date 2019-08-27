FROM conda/miniconda3

# install build essentials & mercurial then delete the cache to minimize
# the docker size
RUN apt-get update \
&& apt-get install -y \
    build-essential \
    libspatialindex-c4v5 \
    mercurial \
&& rm -rf /var/lib/apt/lists/*

# make a new user called pygeoprocessing to avoid running everything as root
ENV SETUSER pygeoprocessing
RUN useradd -m $SETUSER
USER $SETUSER
WORKDIR /home/$SETUSER
COPY install-environment.sh .

# pre-fetch pygeoprocessing in anticipation of a faster build step
RUN hg clone https://bitbucket.org/natcap/pygeoprocessing -r develop

# build the python environments for 3.6 and 3.7 using conda, pip, and
# python setup.py install
RUN /bin/bash ./install-environment.sh 3 6
RUN /bin/bash ./install-environment.sh 3 7
