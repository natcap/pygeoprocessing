PyGeoprocessing is a Python/Cython based library that provides a set of commonly used raster, vector, and hydrological operations for GIS processing.  Similar functionality can be found in ArcGIS/QGIS raster algebra, ArcGIS zonal statistics, and ArcGIS/GRASS/TauDEM hydrological routing routines.

PyGeoprocessing was developed at the Natural Capital Project to create a programmable, open source, and free GIS processing library to support the ecosystem service software InVEST.  PyGeoprocessing's design prioritizes computation and memory efficient runtimes, easy installation and cross compatibility with other open source and proprietary software licenses, and a simplified set of orthogonal GIS processing routines that interact with GIS data via filename. Specifically the functionally provided by PyGeoprocessing includes

* programmable raster algebra routine (vectorize_datasets)
* routines for simplified raster creation and statistics
* integration with vector based geometry in many of the routines
* a simplified hydrological routing library including,
   * d-infinity flow direction
   * support for plateau drainage
   * weighted and unweighted flow accumulation
   * and weighted and unweighted flow distance

Dependencies include
 * cython>=0.20.2
 * numpy>=1.8.2
 * scipy>=0.13.3
 * shapely>=1.3.3
 * gdal>=1.10.1
