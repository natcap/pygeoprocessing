#######
History
#######

0.2.3dev (XXX)
--------------

* Fixing a memory leak in block caches that held on to dataset, band, and block references even after the object was destroyed.

0.2.2 (2015-05-07)
------------------

* Adding MinGW-specific compiler flags for statically linking pygeoprocessing binaries against libstdc++ and libgcc.  Fixes an issue on many user's computers when installing from a wheel on the Python Package Index without having two needed DLLs on the PATH, resuling in an ImportError on pygeoprocessing.geoprocessing_core.pyd.
* Fixing an issue with versioning where 'dev' was displayed instead of the version recorded in pygeoprocessing/__init__.py.
* Adding all pygeoprocessing.geoprocessing functions to pygeoprocessing.__all__, which allows those functions to appear when calling help(pygeoprocessing).
* Adding routing_core.pxd to the manifest.  This fixes an issue where some users were unable to compiler pygeoprocessing from source.

0.2.1 (2015-04-23)
------------------

* Fixed a bug on the test that determines if a raster should be memory blocked.  Rasters were not getting square blocked if the memory block was row aligned.  Now creates 256x256 blocks on rasters larger than 256x256.
* Updates to reclassify_dataset_uri to use numpy.digitize rather than Python loops across the number of keys.
* More informative error messages raised on incorrect bounding box mode.
* Updated docstring on get_lookup_from_table to indicate the headers are case insensitive.
* Added updates to align dataset list that report which dataset is being aligned.  This is helpful for logging feedback when many datasets are passed in that don't take long enough to get a report from the underlying reproject dataset function.
* pygeoprocessing.routing.routing_core includes pxd to be `cimport`able from a Cython module.

0.2.0 (2015-04-14)
------------------

* Fixed a library wide issue relating to the underlying numpy types of GDT_Byte Datasets.  Now correctly identify the signed and unsigned versions and removed all instances where code used to mod byte data to unsigned data and correctly creates signed/unsigned byte datasets during resampling.
* Removed extract_band_and_nodata function since it exposes the underlying GDAL types.
* Removed reclassify_by_dictionary since reclassify_dataset_uri provided almost the same functionality and was widely used.
* Removed the class OrderedDict that was not used.
* Removed the function calculate_value_not_in_dataset since it loaded the entire dataset into memory and was not useful.

0.1.8 (2015-04-13)
------------------

* Fixed an issue on reclassifying signed byte rasters that had negative nodata values but the internal type stored for vectorize datasets was unsigned.

0.1.7 (2015-04-02)
------------------

* Package logger objects are now identified by python heirarchical package paths (e.g. pygeoprocessing.routing)
* Fixed an issue where rasters that had undefined nodata values caused striping in the reclassify_dataset_uri function.

0.1.6 (2015-03-24)
---------------------

* Fixing LICENSE.TXT to .txt issue that keeps reoccuring.

0.1.5 (2015-03-16)
---------------------

* Fixed an issue where int32 dems with INT_MIN as the nodata value were being treated as real DEM values because of an internal cast to a float for the nodata type, but a cast to double for the DEM values.
* Fixed an issue where flat regions, such as reservoirs, that could only drain off the edge of the DEM now correctly drain as opposed to having undefined flow directions.

0.1.4 (2015-03-13)
---------------------

* Fixed a memory issue for DEMs on the order of 25k X 25k, still may have issues with larger DEMs.

0.1.3 (2015-03-08)
---------------------

* Fixed an issue so tox correctly executes on the repository.
* Created a history file to document current and previous releases.
* Created an informative README.rst.

0.1.2 (2015-03-04)
---------------------

* Fixing issue that caused "LICENSE.TXT not found" during pip install.

0.1.1 (2015-03-04)
---------------------

* Fixing issue with automatic versioning scheme.

0.1.0 (2015-02-26)
---------------------

* First release on PyPI.
