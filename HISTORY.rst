.. :changelog:

#######
History
#######

0.1.8 (2015-04-13)
------------------

* Fixed an issue on reclassifying signed byte rasters that had negative nodata values but the internal type stored for vectorize datasets was unsigned.

0.1.7
-----

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
