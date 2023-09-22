PyGeoprocessing
===============

Release: |release|

PyGeoprocessing is a Python/Cython based library that provides a set of
commonly used raster, vector, and hydrological operations for GIS processing.
Similar functionality can be found in ArcGIS/QGIS raster algebra, ArcGIS zonal
statistics, and ArcGIS/GRASS/TauDEM hydrological routing routines.

PyGeoprocessing is developed at the Natural Capital Project to create a
programmable, open source, and free Python based GIS processing library to
support the InVEST toolset. PyGeoprocessing's design prioritizes computation
and memory efficient runtimes, easy installation and cross compatibility with
other open source and proprietary software licenses, and a simplified set of
orthogonal GIS processing routines that interact with GIS data via filename.
Specifically the functionally provided by PyGeoprocessing includes:

    * a suite of raster manipulation functions (warp, align, raster calculator,
      reclassification, distance transform, convolution, and fast iteration)
    * a suite of vector based manipulation function (zonal statistics,
      rasterization, interpolate points, reprojection, and disjoint polygon
      sets)
    * a simplified hydrological routing library (D8inf/MFD flow direction,
      plateau drainage, weighted and unweighted flow accumulation, and weighted
      and unweighted flow distance)

PyGeoprocessing is developed by the `Natural Capital Project <https://naturalcapitalproject.stanford.edu>`_.


Getting Started
---------------

.. toctree::
    :maxdepth: 1

    installing
    basic_usage


.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/pygeoprocessing.rst


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

