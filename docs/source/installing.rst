Installing PyGeoprocessing
==========================

Basic Installation
******************

.. _InstallWithPip:

Installing via pip
------------------

PyGeoprocessing can be installed via pip::

    pip install pygeoprocessing

Note that `GDAL <https://gdal.org/>`_ is a required dependency of
``pyeoprocessing`` and prebuilt binaries are not available on PyPI.
See `Dependencies`_ section for alternatives.

.. _InstallWithConda:

Installing via conda
--------------------

PyGeoprocessing can also be installed from ``conda-forge``::

    conda install -c conda-forge pygeoprocessing

Unlike the ``pip`` approach, this will install the complete dependency tree.


Installing from source
----------------------

If you have a ``git`` installation and the `appropriate compiler for your system
<https://wiki.python.org/moin/WindowsCompilers>`_
available, PyGeoprocessing can also be installed from its source tree.  This is
particularly useful for installing the very latest development build of
PyGeoprocessing::

    pip install git+https://github.com/natcap/pygeoprocessing.git@main


Numpy dtype error
+++++++++++++++++

If you ``import pygeoprocessing`` and see a ``ValueError: numpy.dtype has the
wrong size, try recompiling``, do this::

    pip uninstall -y pygeoprocessing
    pip install pygeoprocessing --no-deps --no-binary :all:

This error is the result of a version compatibility issue with the numpy API in
the precompiled pygeoprocessing wheel. The solution is to recompile
pygeoprocessing on your computer using the above steps.


.. _Dependencies:

Dependencies
************

PyGeoprocessing's dependencies are listed in ``requirements.txt``, reproduced here:

.. include:: ../../requirements.txt
    :literal:

All of these dependencies will be installed automatically to your conda
environment if you're using conda (see `Installing via conda`_).  Below are a few
alternate ways to install the required packages.

Ubuntu & Debian: apt
--------------------
::
    sudo apt install python3-dev python3-gdal cython3 python3-shapely python3-rtree

Fedora: yum
-----------
::
    sudo yum install python3-devel python3-gdal python3-rtree python3-shapely

Mac: brew
---------
::
    brew install gdal spatialindex

Windows: pip
------------

For installing on Windows, take a look at Christoph Gohlke's page of unofficial builds:
http://www.lfd.uci.edu/~gohlke/pythonlibs/

