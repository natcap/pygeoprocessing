.. default-role:: code

About PyGeoprocessing
=====================

PyGeoprocessing is a Python/Cython based library that provides a set of
commonly used raster, vector, and hydrological operations for GIS processing.
Similar functionality can be found in ArcGIS/QGIS raster algebra, ArcGIS zonal
statistics, and ArcGIS/GRASS/TauDEM hydrological routing routines.

PyGeoprocessing is developed at the Natural Capital Project to create a
programmable, open source, and free Python based GIS processing library to
support the InVEST toolset.  PyGeoprocessing's design prioritizes
computation and memory efficient runtimes, easy installation and cross
compatibility with other open source and proprietary software licenses, and a
simplified set of orthogonal GIS processing routines that interact with GIS
data via filename. Specifically the functionally provided by PyGeoprocessing
includes:

* a suite of raster manipulation functions (warp, align, raster calculator,
  reclassification, distance transform, convolution, and fast iteration)
* a suite of vector based manipulation function (zonal statistics,
  rasterization, interpolate points, reprojection, and disjoint polygon sets)
* a simplified hydrological routing library (D8inf/MFD flow direction,
  plateau drainage, weighted and unweighted flow accumulation, and weighted
  and unweighted flow distance)

Installing PyGeoprocessing
==========================

.. code-block:: console

    $ pip install pygeoprocessing

If you `import pygeoprocessing` and see a `ValueError: numpy.dtype has the
wrong size, try recompiling`, this is the result of a version compatibility
issue with the numpy API in the precompiled pygeoprocessing binaries.
The solution is to recompile pygeoprocessing on your computer:

.. code-block:: console

    $ pip uninstall -y pygeoprocessing
    $ pip install pygeoprocessing --no-deps --no-binary :all:

Requirements
------------

Note the pip-installable requirements in `requirements.txt` are for best
results, but older package versions may also work. If necessary,
PyGeoprocessing can be installed without dependencies with `pip install
--no-deps`.

API Documentation
=================

API documentation is available at https://pygeoprocessing.readthedocs.io/en/latest/

Test Matrix Policy
==================

Python
------
We test against all currently supported (security & bugfix) 
Python versions.
https://devguide.python.org/versions/  

GDAL
----
We test against the latest GDAL release of each minor version starting 
with the version included in Debian Stable.
https://packages.debian.org/stable/science/gdal-bin
