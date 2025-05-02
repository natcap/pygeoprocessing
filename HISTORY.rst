Release History
===============

2.4.8 (2025-05-02)
------------------
* ``zonal_statistics`` raises an error if the vector is not a Polygon
  or MultiPolygon geometry.
  https://github.com/natcap/pygeoprocessing/issues/322
* Handling a case in ``raster_map`` where an exception would be raised when a
  float32 array was passed along with a float64 nodata value.
  https://github.com/natcap/pygeoprocessing/issues/358
* ``reproject_vector`` will skip copying field values from the base layer
  to the target if doing so would raise a RuntimeError,
  such as when a string value cannot be represented by UTF-8.
  https://github.com/natcap/pygeoprocessing/issues/418
* ``raster_band_percentile`` can now optionally log a warning if the raster
  has a geographic CRS.
  https://github.com/natcap/pygeoprocessing/issues/299

2.4.7 (2025-01-23)
------------------
* Dropped support for Python 3.8. Added support for Python 3.13 and GDAL 3.9.
  https://github.com/natcap/pygeoprocessing/issues/415
* Added validation to ``reclassify_raster`` to raise a ``TypeError`` with a
  descriptive message if ``value_map`` contains non-numeric keys.
* In ``warp_raster``, if either ``base_raster_path`` or ``target_raster_path``
  are not strings, a ``ValueError`` is now raised with a more helpful error
  message.  https://github.com/natcap/pygeoprocessing/issues/421

2.4.6 (2024-10-15)
------------------
* Removing the ``numpy<2`` constraint for requirements.txt that should have
  been included in the 2.4.5 release. https://github.com/natcap/pygeoprocessing/issues/396
* Fixed an issue in ``convolve_2d`` where a long-running convolution would
  raise a cryptic exception involving ``queue.Empty``.  This will instead now
  raise ``RuntimeError`` with a more helpful exception message.  We also fixed
  an issue where the ``max_timeout`` parameter of ``convolve_2d`` was unused,
  so it is now used correctly. https://github.com/natcap/pygeoprocessing/issues/360
* Handling GDAL-based ``RuntimeError`` raised during ``pygeoprocessing.reproject_vector``.
  https://github.com/natcap/pygeoprocessing/issues/409

2.4.5 (2024-10-08)
------------------
* Updating for numpy 2.0 API changes.  Pygeoprocessing is now compatible with
  numpy 2.0 and later.  https://github.com/natcap/pygeoprocessing/issues/396

2.4.4 (2024-05-21)
------------------
* Our github actions for building python distributions now use
  `actions/setup-python@v5`, which uses node 20.
  https://github.com/natcap/pygeoprocessing/issues/384
* ``warp_raster`` and ``build_overviews`` no longer raise a ``ValueError`` if
  called with an invalid resampling algorithm. We now fall back to the
  underlying GDAL functions' error messages.
  https://github.com/natcap/pygeoprocessing/issues/387
* Implementing decaying flow accumulation for D8 routing.
  https://github.com/natcap/pygeoprocessing/issues/386
* Updated to Cython 3.
* Dropped support for Python 3.7.
* Enabled GDAL exceptions (``gdal.UseExceptions``) throughout pygeoprocessing.
  This is done with a context manager so should not affect other code or
  libraries. https://github.com/natcap/pygeoprocessing/issues/391

2.4.3 (2024-03-06)
------------------
* Wheels for python 3.12 are now built during our github actions runs.
  https://github.com/natcap/pygeoprocessing/issues/381
* ``get_gis_type`` can accept a path to a remote file, allowing the GDAL driver
  to open it if the driver supports the protocol.
  https://github.com/natcap/pygeoprocessing/issues/375
* If running on a SLURM system (identified by the presence of ``SLURM*``
  environment variables), the GDAL cache max is checked against the amount of
  memory available on the compute node.  If GDAL may exceed the available slurm
  memory, a warning is issued or logged.
  https://github.com/natcap/pygeoprocessing/issues/361
* Fixed an issue in ``extract_strahler_streams_d8`` where a nodata pixel
  could be mistakenly treated as a stream seed point, ultimately creating
  a stream feature with no geometry.
  https://github.com/natcap/pygeoprocessing/issues/361
* Improved ``align_and_resize_raster_stack`` so that rasterization of a vector
  mask only happens once, regardless of the number of rasters in the stack.
  In addition, the created mask raster's path may be defined by the caller so
  that it persists across calls to ``align_and_resize_raster_stack``.
  https://github.com/natcap/pygeoprocessing/issues/366
* Improved ``warp_raster`` to allow for a pre-defined mask raster to be
  provided instead of a vector.  If both are provided, the mask raster alone is
  used.  The new mask raster must have the same dimensions and geotransform as
  the output warped raster. https://github.com/natcap/pygeoprocessing/issues/366
* Fixed a bug in ``zonal_statistics`` where the wrong number of disjoint
  polygon sets were being reported in the logs.
  https://github.com/natcap/pygeoprocessing/issues/368
* Pygeoprocessing is now tested against python 3.12.
  https://github.com/natcap/pygeoprocessing/issues/355

2.4.2 (2023-10-24)
------------------
* Fixed an issue where MFD flow direction was producing many nodata holes given
  a large-enough DEM.  These nodata holes would then propagate to flow
  accumulation and stream extraction, producing very disjointed stream
  networks. https://github.com/natcap/pygeoprocessing/issues/350
* Improved progress logging in MFD flow direction, MFD flow accumulation, MFD
  stream extraction to prevent integer overflows in percentages and improve
  the readability of log messages. https://github.com/natcap/pygeoprocessing/issues/246
* Fixed an issue introduced in 2.4.1 where ``zonal_statistics`` results could
  vary depending on the bounding boxes of the disjoint polygon sets calculated
  from the input vector.
* Added a new function, ``pygeoprocessing.align_bbox``, which pads a bounding
  box to align with the grid of a given geotransform.
* Added support for int64 and uint64 (GDAL 3.5+)
  https://github.com/natcap/pygeoprocessing/issues/352
* Added support for signed bytes (GDAL 3.7+)
  https://github.com/natcap/pygeoprocessing/issues/329
* Fixed a bug where ``calculate_slope`` did not work as expected when the DEM
  raster's nodata value is NaN (https://github.com/natcap/pygeoprocessing/issues/352)

2.4.1 (2023-09-05)
------------------
* The ``pygeoprocessing`` package metadata has been updated to use
  ``importlib.metadata`` (python >= 3.8) or ``importlib_metadata``
  (python < 3.8) for retrieving the package version, in keeping with
  recommendations from ``setuptools_scm``.  The dependency
  ``importlib_metadata`` is now required for installs on python < 3.8.
* Fixed another memory leak in ``pygeoprocessing.raster_calculator``, where
  shared memory objects under certain circumstances were not being unlinked at
  the end of the function, resulting in excess memory usage and a warning
  during the shutdown of the python process.
  https://github.com/natcap/pygeoprocessing/issues/247
* Added a new function, ``pygeoprocessing.array_equals_nodata``, which returns
  a boolean array indicating which elements have nodata. It handles integer,
  float, and ``nan`` comparison, and the case where the nodata value is `None`.
* Standardized the approach used in ``warp_raster`` and
  ``create_raster_from_bounding_box`` for determining the dimensions of the
  target raster given a target bounding box and pixel sizes.
  https://github.com/natcap/pygeoprocessing/issues/321
* ``pygeoprocessing.routing.delineate_watersheds_d8`` now handles the case
  where the input flow direction raster does not have a defined spatial
  reference. https://github.com/natcap/pygeoprocessing/issues/254
* Updating internal documentation describing TauDEM flow directions, and adding
  for how to convert from a flow direction raster from what TauDEM expects to
  what pygeoprocessing expects.
  https://github.com/natcap/pygeoprocessing/issues/255
* Users may now specify the overview level to use when calling ``warp_raster``.
  By default, ``pygeoprocessing`` will use the base layer.
  https://github.com/natcap/pygeoprocessing/issues/326
* Fixed a bug across ``pygeoprocessing`` where some valid resampling methods
  would throw an exception because they were not recognized.  This was only
  happening when ``pygeoprocessing`` was installed alongside GDAL < 3.4.
* Fixing an issue with ``pygeoprocessing.multiprocessing.raster_calculator``
  where the function would raise an Exception when the target raster path was
  provided as a filename only, not within a directory, even though the parent
  directory could be inferred. https://github.com/natcap/pygeoprocessing/issues/313
* Fixing a bug where the statistics worker in
  ``pygeoprocessing.raster_calculator`` may return a complex value. This is
  only an issue when pygeoprocessing is compiled against Cython 3.0.0 and
  later. https://github.com/natcap/pygeoprocessing/issues/342
* Added the new function ``pygeoprocessing.raster_map``, a higher-level version
  of ``raster_calculator``. https://github.com/natcap/pygeoprocessing/issues/235

2.4.0 (2023-03-03)
------------------
* A new submodule, ``pygeoprocessing.kernels`` has been added to facilitate the
  creation of kernel rasters needed for calls to
  ``pygeoprocessing.convolve_2d``. Functions for creating common decay kernels
  have been added, along with functions to facilitate the creation of
  distance-based kernels using a user-defined function
  (``pygeoprocessing.create_kernel``) and to facilitate the creation of kernels
  using custom 2D numpy arrays, such as those commonly used in image processing
  (``pygeoprocessing.kernel_from_numpy_array``).
  https://github.com/natcap/pygeoprocessing/issues/268
* Logging across functions in ``pygeoprocessing.geoprocessing`` now correctly
  reports the function that it's being called from rather than ``<lambda>``.
  https://github.com/natcap/pygeoprocessing/issues/300
* The function ``pygeoprocessing.reproject_vector`` now accepts an optional
  parameter ``layer_name`` to allow the target vector layer name to be defined
  by the user.  If the user does not provide a ``layer_name``, the layer name
  will be copied from the source vector.
  https://github.com/natcap/pygeoprocessing/issues/301
* Implement the proposed new function ``pygeoprocessing.raster_reduce``, a
  wrapper around ``pygeoprocessing.iterblocks``
  (https://github.com/natcap/pygeoprocessing/issues/285)
* Nodata value checking in ``pygeoprocessing.routing`` now correctly handles
  comparison of ``nan`` values.  This is explicitly tested in
  ``pygeoprocessing.routing.fill_pits``, but should also improve the
  experience of other routing functions as well.
  https://github.com/natcap/pygeoprocessing/issues/248
* Added a function to build overviews for a raster,
  ``pygeoprocessing.build_overviews``. Related to this,
  ``pygeoprocessing.get_raster_info()`` now includes an ``'overviews'`` key
  listing the pixel dimensions of each overview layer in a raster.
  https://github.com/natcap/pygeoprocessing/issues/280
* Added a D8 stream extraction function at
  ``pygeoprocessing.routing.extract_streams_d8`` which takes a D8 flow
  accumulation raster and a flow accumulation threshold, setting all pixels
  with accumulation above that threshold to 1 and all other valid pixels to 0.
  https://github.com/natcap/pygeoprocessing/issues/272
* Adding a new function, ``pygeoprocessing.create_raster_from_bounding_box``,
  that enables the creation of a new raster from a bounding box.
  https://github.com/natcap/pygeoprocessing/issues/276
* Win32 wheels of PyGeoprocessing are no longer created through our GitHub
  Actions workflows and will no longer be produced or distributed as part of
  our release checklist.  For details (and metrics!) see:
  https://github.com/natcap/pygeoprocessing/issues/232

2.3.5 (2022-12-13)
------------------
* ``pygeoprocessing.calculate_disjoint_polygon_set`` now offers an optional
  parameter, ``geometries_may_touch`` for cases where geometries are known to
  have intersecting boundaries but nonintersecting interiors.
  https://github.com/natcap/pygeoprocessing/issues/269
* Pygeoprocessing is now tested against Python 3.11.
* Adding the target filename to progress logging in
  ``pygeoprocessing.raster_calculator``.
* ``pygeoprocessing.zonal_statistics`` will now optionally include a count of
  the number of pixels per value encountered under each polygon. A warning
  will be logged when invoked on floating-point rasters, as using this on
  continuous rasters can result in excessive memory consumption. To use this
  feature, set ``include_value_counts=True`` when calling ``zonal_statistics``.
* ``pygeoprocessing.get_gis_type`` will now raise a ``ValueError`` if the file
  cannot be opened as ``gdal.OF_RASTER`` or ``gdal.OF_VECTOR``.
  https://github.com/natcap/pygeoprocessing/issues/244
* Fixing an error message in ``convolve_2d`` when the signal or kernel is a
  row based blocksize. https://github.com/natcap/pygeoprocessing/issues/228.

2.3.4 (2022-08-22)
------------------
* Fixing an issue with imports at the ``pygeoprocessing`` module level that was
  causing linters like PyLint and IDE command-completion programs like JEDI-vim
  to not be able to identify the attributes of the ``pygeoprocessing`` module
  namespace.
* Reducing the amount of memory used by
  ``pygeoprocessing.new_raster_from_base`` when filling an array with values to
  only as much memory as is needed for the datatype.
* Fixing a memory leak in ``pygeoprocessing.raster_calculator`` where
  shared memory objects were being inadvertently created when they should not
  have been and then they were not subsequently destroyed.
* ``calculate_disjoint_polygon_set`` will now skip over empty geometries.
  Previously, the presence of empty geometries would cause an error to be
  raised.
* Fixed a ``DeprecationWarning`` in ``calculate_disjoint_polygon_set`` caused
  by the use of a deprecated logging ``warn`` method.

2.3.3.post0 (2022-01-28)
------------------------
* Post-release due to corrupted sdist released on Github and PyPI. The sdist
  for 2.3.3 failed to install. The wheels were unaffected. No code changes.

2.3.3 (2022-01-26)
------------------
* Fixing a bug in ``shapely_geometry_to_vector`` where a feature name mismatch
  between the ``fields`` and ``attribute_list`` inputs would silently pass
  under most circumstances.  Now an informative ``ValueError`` is raised.
* Testing against Python 3.10.
* Pinned ``numpy`` versions in ``pyproject.toml`` to the lowest compatible
  version for each supported python version. This prevents issues when
  ``pygeoprocessing`` is used in an environment with a lower numpy version
  than it was built with (https://github.com/cython/cython/issues/4452).

2.3.2 (2021-09-08)
------------------
* Restore functionality in ``reclassify_raster`` that allows for nodata
  values to be reclassified. This was accidentally removed in 2.3.1.

2.3.1 (2021-08-24)
------------------
* Slightly change the error message displayed for a
  ``ReclassificationMissingValuesError``.
* Handle float nodata values correctly in ``reclassify_raster``.
* Fix a bug in ``distance_to_channel_mfd`` where pixels that only partially
  drained to a stream were given incorrect distances.
* Fixed an issue with ``raster_band_percentile`` that would raise an
  exception if an input raster had an undefined nodata value and/or
  would cause an invalid result if the raster contained non-finite
  values.
* After many years with the Natural Capital Project, Rich Sharp has stepped
  down from the Project and as the maintainer of ``pygeoprocessing``.  James
  Douglass is taking his place, and this change is now reflected in
  ``setup.py``.

2.3.0 (2021-06-21)
------------------
* Added a ``single_outlet_tuple`` parameter to ``routing.fill_pits`` that
  forces a DEM to only have one outlet at any point on the raster. The
  fill effect is that all pixels will drain to the raster coordinate at
  ``single_outlet_tuple``.
* Added a ``detect_lowest_sink_and_drain`` function that finds the lowest
  DEM pixel that drains to nodata/edge and the lowest DEM pixel that could
  be a sink. The values that result from this call can be used to condition
  a DEM that is known to have a single drain using the
  ``single_outlet_tuple`` parameter in ``routing.fill_pits``.
* Fixed a bug in ``routing.fill_pits`` that could cause the nodata region of
  a DEM to be incorrectly filled with non-nodata values.
* Fixed an issue where ``create_raster_from_vector_extent`` would use an
  approximation of the layer extent to infer the input vector extent. Now
  uses the individual geometry to infer vector extents.
* Fixed a bug in ``routing.distance_to_channel_mfd`` where pixels that drained
  off the edge of the raster had negative values rather than nodata values.
* Expanded the error message raised by ``transform_bounding_box`` when the
  bounding box cannot be transformed to provide more helpful details.
* Add support and testing for GDAL 3.3.0.
* Fixed exception occuring in ``stitch_rasters`` when base rasters were
  larger than the target raster on the ``y`` axis.

2.2.0 (2021-05-14)
------------------
* Adding explicit support for Python 3.9 and testing on Python 3.9.
* Fixed an issue in ``create_raster_from_vector_extents`` that would cause a
  confusing exception to be raised if there was no geometry in the vector.
  Now raises a ``ValueError`` with a helpful error message.
* Changed parameters in ``convolve_2d`` to allow API to set
  ``ignore_nodata_and_edges=True`` while ``mask_nodata=False`` and updated
  docstring to indicate this is useful in cases such as filling nodata holes
  in missing datasets. Additionally added a logger ``debug`` message to note
  this "unusual" setting of these parameters in case of accidental usage
  which could be noted during development.
* Fixed issue in zonal statistics that could cause the aggregate vector to be
  locked due to a dereferencing race condition. This was present in some
  cases with a flaky unit test but could have been seen in practice if the
  vector was deleted immediately after the call to ``zonal_statistics``.
* Fixed issue in ``routing.fill_pits`` that used a numerical "is close" test
  when determining if two pixels were at equal height. In the case of
  hydrological pitfilling, pixels must be exactly equal height to be
  considered a plateau otherwise a drain into or out of the pixel is
  resolved. Testing is close allowed a hydrological pit to remain since it
  was "close" to the same height as a draining pixel.
* Added a D8 Strahler order stream extraction function at
  ``pygeoprocessing.extract_strahler_streams_d8``. Creates segmented
  geometric lines which correspond to streams on the landscape.
* Added a D8 subwatershed calculation function at
  ``pygeoprocessing.calculate_subwatershed_boundary``. Creates subwatersheds
  that are segmented at the junctions of the streams created by
  ``pygeoprocessing.extract_strahler_streams_d8``.
* Removing all instances of ``__swig_destroy__`` to prevent multiprocessing
  memory corruption.
* Exposing a ``use_shared_memory`` flag on ``raster_calculator`` to allow
  a user to use shared memory objects when calculating statistics. This
  feature is only available for Python >= 3.8. If available, this
  feature creates a significant runtime improvement but can be unstable
  in multiprocessing configurations. For this reason it is set to
  ``False`` as the default value.
* Added a ``max_timeout`` parameter to ``convolve_2d`` and
  ``raster_calculator`` to allow the user to specify the maximum amount of
  time to wait for worker threads to terminate. In normal operation these
  threads should terminate in a short amount of time but are generously
  timed with the ``pygeoprocessing._MAX_TIMEOUT`` parameter. This parameter
  allows a user to tune in cases that may involve significant latency such
  as in a heavy multiprocess environment.
* Added a ``pygeoprocessing.routing.detect_outlets`` function that creates a
  vector with points placed on the center of the pixels in a D8 or MFD flow
  direction raster that are outlet pixels. The vector has additional metadata
  about the (i, j) locations of the pixel in raster coordinate space.
* Replacing ``merge_rasters`` with ``stitch_rasters`` that can take an
  arbitrary list of raster path bands and stitch them into an existing raster
  of same or different projection. Additional functionality to preserve
  area variant values when stitching from a linear meters projection to
  WGS84. Useful when the units of the base raster are in "units per pixel"
  rather than "units per area".
* Added a ``max_pixel_fill_count`` parameter to ``routing.fill_pits`` to
  guard against pitfilling large natural depression. Defaults to 500.
* Fixed an issue in ``align_and_resize_raster_stack`` that would ignore
  the bounds of a feature in a mask vector if the
  ``"mask_vector_where_filter"`` clause was invoked and instead only
  considered the entire bounds of the vector.
* Fixed an issue with ``convolve_2d`` that allowed output rasters to be
  created without a defined nodata value.
* Fixed a LOGGER message bug that occurred in ``zonal_statistics``.
* Fixed an issue in ``distance_to_channel_mfd`` that would generate a raster
  with distances to the edge of the raster even if there was no channel. Now
  generates nodata so it is consistent with ``distance_to_channel_d8``.
* The ``convolve_2d`` function now raises a ValueError if either the signal
  or kernel raster has a row based blocksize since this could result in
  very long runtimes due to gdal cache thrashing.
* Fixed an issue that would cause a ``convolve_2d`` worker to crash if the
  integrating kernel was not a float and also ``normalize_kernel=True``
  thus causing the entire function call to deadlock.
* Added a check on ``transform_bounding_box`` to ensure the target bounding
  box's coordinates were finite. This guards against cases where a transform
  into another coordinate system creates a degenerate bounding box.
  Previously the function would silently return non-finite coordinates.
* Fixing issue when calculating histogram for floating point rasters the
  logging progress percent would be incorrectly calculated.

2.1.2 (2020-12-03)
------------------
* ``pygeoprocessing.warp_raster`` now raises a ``ValueError`` when an invalid
  resampling method is provided.
* Fixed bug in ``pygeoprocessing.warp_raster`` that would not properly handle
  GDAL Byte type signing when masking warped raster with a vector.
* Fixed issue in ``convolve_2d`` that would cause excessive memory use
  leading to out of memory errors.
* Fixed issue in ``convolve_2d`` that could lead to a file removal race
  condition and raise a ``FileNotFoundException`` when ignoring nodata
  regions.

2.1.1 (2020-09-16)
------------------
* Fixed a critical bug introduced in 2.1.0 that generated invalid results in
  `convolve_2d` for any raster larger than 256x256.
* Added an optional parameter, ``target_layer_name`` to
  ``pygeoprocessing.routing.delineate_watersheds_d8`` for cases where a
  caller would like to define the output layer name.

2.1.0 (2020-08-25)
------------------
* Modified ``pygeoprocessing.numpy_array_to_raster`` to take boolean arrays.
* Modified ``pygeoprocessing.convolve_2d`` to guard against nonsensical queries
  to both ``ignore_nodata_and_edges=True`` but also ``mask_nodata=False``.
  A query of this combination now raises a ``ValueError`` to guard against
  programmer error.
* Added a custom exception class ``ReclassificationMissingValuesError`` to
  ``pygeoprocessing``. ``pygeoprocessing.reclassify_raster`` raises this
  exception instead of ``ValueError`` when a raster pixel value is not
  represented in ``value_map``. This custom exception provides a list of
  missing raster pixel values in a ``missing_values`` attribute that allows
  the caller access to the pixel values that are missing through a Python type
  rather than indirectly through an error message.
* Correcting the docstring for ``pygeoprocessing.numpy_array_to_raster`` to
  specify that the ``pixel_size`` parameter must be a tuple or list, not an
  int.
* ``pygeoprocessing.routing.delineate_watersheds_d8`` now has an optional
  parameter ``write_diagnostic_vector``.  When ``True``, this parameter will
  cause a new vector per outflow feature to be created in the ``working_dir``.
  This parameter defaults to ``False``.  This is a change from prior behavior,
  when the diagnostic vectors were always created, which could occupy a lot of
  computational time under large outflow geometries.
* Added a ``pygeoprocessing.multiprocessing.raster_calculator`` function which
  matches the API and results of ``pygeoprocessing.raster_calculator`` but uses
  multiple processing cores to compute raster calculation blocks.
  when the diagnostic vectors were always created, which could occupy
  significant computational time under large outflow geometries.
* Minor performance improvement to ``pygeoprocessing.convolve_2d`` by
  preventing a pre-processing step that initialized temporary rasters with zero
  values as well as added asynchronous work distribution for kernel/signal
  block processing.
* Modified logging message for ``pygeoprocessing.new_raster_from_base`` when
  filling a raster such that an informative error message is printed with
  context as to the function, file, status, and value being filled.

2.0.0 (05-19-2020)
------------------
* Adding Python 3.8 support and dropping Python 3.6 support.
* Adding GDAL 3 support and dropping GDAL 2 support. The only non-backwards
  compatible issue in GDAL 2 to GDAL 3 is the need to handle Axis Ordering with
  osr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) because of
  https://trac.osgeo.org/gdal/wiki/rfc73_proj6_wkt2_srsbarn#Axisorderissues?.
  Since Axis ordering now matters for Geographic CRS the expected order is
  Lat,Lon but we use osr.OAMS_TRADITIONAL_GIS_ORDER to swap to Lon,Lat.
* Using osr.CreateCoordinateTransformation() instead of
  osr.CoordinateTransformation() as the GDAL 3 call.
* Fixed a bug in convolve_2d that would not ``ignore_nodata`` if the signal
  raster's nodata value was undefined. Changed the name of this flag to
  ``ignore_nodata_and_edges`` to reflect its expected functionality.
* Warped signed byte rasters are now also signed byte rasters.
* Adding a GitHub Actions-based build job for building wheels and a source
  distribution for a given commit of pygeoprocessing.
* Updated ``setup.py`` to point the URL project link to the project's new
  home on GitHub.
* Updated ``MANIFEST.in`` to only include files that should be there in the
  pygeoprocessing source distribution.  This fixes an issue where files
  matching a variety of extensions anywhere in the pygeoprocessing directory
  might be included with the source distribution.
* Added ``set_tol_to_zero`` to ``convolve_2d`` to allow for in-function masking
  of near-zero results to be set to 0.0.
* Fixed malformed logging outputs which could be seen during long running
  ``rasterize`` calls.
* Renamed all parameters involving Spatial Projections to the form
  ``[var_id]_projection_wkt``, this involves optional arguments in
  ``reproject_vector``, ``warp_raster``, ``transform_bounding_box``,
  and ``align_and_resize_raster_stack`` as well as the return value from
  ``get_raster_info`` and ``get_vector_info``.
* Fixed an issue in ``zonal_statistics`` that would crash if an aggregate
  vector had a feature with no geometry defined. Now the function ignores
  such features and prints a warning to the log.
* Fixed a malformed ``ValueError`` message when a corrupt raster was
  encountered in ``raster_calculator``.
* Fixes an unnecessary calculation that pre-fills slope raster GeoTIFFs with
  nodata values.
* Added a check to ``convolve_2d`` to verify that raster path/band tuples were
  passed where expected and raise a useful Exception  if not.
* Removed most of the `pygeoprocessing.testing` module and added the following
  convenience functions to ``pygeoprocessing``, which should not be used
  for production code but are useful for testing and scripting:

  * ``raster_to_numpy_array`` - read a single band of a raster into a ``numpy``
    array, runs the risk of memory error if the raster is too large.
  * ``numpy_array_to_raster`` - writes a ``numpy`` array to a raster on disk.
  * ``shapely_geometry_to_vector`` - creates a vector from a list of
    ``Shapely`` geometry.

* Fixed an issue in ``flow_dir_mfd`` that would cause invalid flow directions
  on DEMs that had very small numerical delta heights.
* Fixes an issue in ``convolve_2d`` that would occasionally cause undefined
  numerical noise in regions where the signal was nodata but ``mask_nodata``
  was set to ``False``.

1.9.2 (2020-02-06)
------------------
* Removed the ``multiprocessing`` dependency to avoid an occasional deadlock
  that occurred on Mac OS X during ``align_and_resize_raster_stack``.
  That function now operates serially, but multithreading can be used by
  passing ``gdal_warp_options``.

1.9.1 (2019-12-19)
------------------
* Fixed a compilation issue on Mac OS X Catalina related to the compilation
  of a template in the file iteration component of the out-of-core percentile
  function.
* Resolved a compilation issue on Mac OS X (Mavericks and later) where
  pygeoprocessing would not compile unless some additional compiler and linker
  flags were provided.  These are now accounted for in the package's compilation
  steps in ``setup.py``.
* ``pygeoprocessing.symbolic.evaluate_raster_calculator_expression``
  no longer depends on ``sympy`` for its expression evaluation.

1.9.0 (2019-10-22)
------------------
* Fixed a memory error issue that could occur on multiple flow direction flow
  accumulation calculations.
* Added PEP518-compatible build dependencies to ``pyproject.toml``, which has
  been added to source distributions of pygeoprocessing.
* Added an out-of-core high performance raster percentile function at
  pygeoprocessing.raster_band_percentile.
* Added a ``pygeoprocessing.symbolic.evaluate_raster_calculator_expression``
  function that can parse a string expression and a map of symbols to
  calculate a raster calculation operation.
* Modified ``pygeoprocessing.routing.fill_pits`` to create a single band
  raster of the input DEM raster/path band rather than a copy of the input
  DEM raster irrespective of the band number.
* Fixed an issue that would cause the bounding box generated by
  ```transform_bounding_box`` to flip the min and max coordinates in some
  spatial contexts.
* Defaulting compression algorithm back to LZW as ZSTD is not a widely
  implemented compression standard for geotiffs.

1.8.0 (2019-08-12)
------------------
* Added a ``'numpy_type'`` field to the result of ``get_raster_info`` that
  contains the equivalent numpy datatype of the GDAL type in the raster. This
  includes functionality differentate between the unsigned and signed
  ``gdal.GDT_Byte`` vs. ``numpy.int8`` and ``numpy.uint8``.
* Changed default compression routine for GeoTIFFs to ZSTD (thanks Facebook
  https://facebook.github.io/zstd/).
* Added a **non-backwards compatible change** by replacing the
  ``gtiff_creation_options`` string to a driver/option string named
  ``raster_driver_creation_tuple``. This allows the caller to create any type
  of ``GDAL`` writable driver along with the option list associated with that
  driver.
* Added a ``'numpy_type'`` field to the result of ``get_raster_info`` that
  contains the equivalent numpy datatype of the GDAL type in the raster. This
  includes functionality differentate between the unsigned and signed
  ``gdal.GDT_Byte`` vs. ``numpy.int8`` and ``numpy.uint8``.
* Changed default compression routine for GeoTIFFs to ZSTD (thanks Facebook
  https://facebook.github.io/zstd/).
* Added a ``'file_list'`` key to the dictionary returned by
  ``get_raster_info`` and ``get_vector_info`` that contains a list of all the
  files associated with that GIS object. The first parameter of these lists
  can be passed to ``gdal.OpenEx`` to open the object directly.
* Added a ``get_gis_type`` function to ``pygeoprocessing`` that takes a
  filepath and returns a bitmask of ``pygeoprocessing.RASTER_TYPE`` and/or
  ``pygeoprocessing.VECTOR_TYPE``.
* Modified ``iterblocks`` to raise a helpful ValueError instead of a general
  NoneTypeError if a raster does not open.

1.7.0 (2019-06-27)
------------------
* Removing support for Python 2.7.
* Adding D8 watershed delineation as
  ``pygeoprocessing.routing.delineate_watersheds_d8``.
* Corrected an issue with ``pygeoprocessing.create_raster_from_vector_extents``
  where a vector with no width or no height (a vector with a single point, for
  example) would result in invalid raster dimensions being passed to GDAL.
  These edge cases are now guarded against.
* ``pygeoprocessing.calculate_disjoint_polygon_set`` will now raise
  ``RuntimeError`` if it is passed a vector with no features in it.
* ``pygeoprocessing.rasterize`` will now raise ``RuntimeError`` if the
  underlying call to ``gdal.RasterizeLayer`` encounters an error.
* Correcting an issue with the docstring in
  ``pygeoprocessing.reclassify_raster`` to reflect the current parameters.
* Changed ``zonal_statistics`` to always return a ``dict`` instead of
  sometimes a ``defaultdict``. This allows pickling of the result, if desired.
* Adding automated testing via bitbucket pipelines.
* Correcting an issue with ``pygeoprocessing.zonal_statistics`` that was
  causing test failures on Python 3.6.
* Pygeoprocessing is now tested against Python 3.7.
* Fixed an issue in distance transform where a vertical striping artifact
  would occur in the masked region of some large rasters when distances should
  be 0.
* Fixed an issue in all functionality that used a cutline polygon with
  invalid geometry which would cause a crash. This was caused by `gdal.Warp`
  when using the cutline functionality. Instead this functionality was
  replaced with manual rasterization. In turn this introduces two optional
  parameters:

    * ``rasterize`` and ``mask_raster`` have a ``where_clause`` parameter
      which takes a string argument in SQL WHERE syntax to filter
      rasterization based on attribute values.
    * ``warp_raster`` takes a ``working_dir`` parameter to manage local
      temporary mask rasters.

* Removing a temporary working directory that is created when executing
  pygeoprocessing.convolve_2d.
* Changed optional parameters involving layer indexes to be either indexes
  or string ids. In all cases changing ``layer_index`` to ``layer_id`` in
  the functions: ``get_vector_info``, ``reproject_vector``, ``warp_raster``,
  ``rasterize``, ``calculate_disjoint_polygon_set``, and ``mask_raster``.

1.6.1 (2019-02-13)
------------------
* Added error checking in ``raster_calculator`` to help ensure that the
  ``target_datatype`` value is a valid GDAL type.
* Fixed an issue in ``distance_transform_edt`` that would occasionally
  cause incorrect distance calculations when the x sampling distance was > 1.

1.6.0 (2019-01-23)
------------------
* Changed ``iterblocks`` API to take a raster/path band as an input rather
  than a path and a list of bands. Also removed the ``astype_list`` due to
  its lack of orthogonality.
* Fixed bugs in ``convolve_2d`` involving inputs with nodata masking.
* Changing default raster creation compression algorithm from LZW to DEFLATE,
  this is to address issues where we were seeing recreatable, but
  unexplainable LZWDecode errors in large raster data.
* Fixed an issue that could cause the distance transform to be incorrect when
  the sampling distance was in the noninclusive range of (0.0, 1.0).

1.5.0 (2018-12-12)
------------------
* Specific type checking for ``astype_list`` in ``iterblocks`` to avoid
  confusing exceptions.
* Renamed test suite to be consistent with the pattern
  ``test_[component].tif``.
* Added a function ``pygeoprocessing.routing.extract_streams_mfd`` that
  creates a contiguous stream layer raster to accounts for the divergent flow
  that can occur with multiple flow direction. If the flow direction raster is
  otherwise directly thresholded, small disjoint streams can appear where
  the downstream flow drops below the threshold level.
* Fixed an issue that could cause some custom arguments to geotiff creation
  options to be ignored.
* Added a ``mask_raster`` function that can be used to mask out pixels in
  an existing raster that don't overlap with a given vector.
* Fixed a bug in the ``distance_transform_edt`` function that would cause
  incorrect distances to be calculated in the case of nodata pixels in the
  region raster. The algorithm has been modified to treat nodata as though
  pixel values were 0 (non-region) and the distance transform will be defined
  for the entire raster.
* Added a ``sampling_distance`` parameter to ``distance_transform_edt`` that
  linearly scales the distance transform by this value.
* Fixed an issue in ``calculate_slope`` that would raise an exception if the
  input dem did not have a nodata value defined.
* Changed the behavior of ``zonal_statistics`` for polygons that that do not
  intersect any pixels. These FIDs are now also included in the result from
  ``zonal_statistics`` where previously they were absent. This is to remain
  consistent with how other GIS libraries calculate zonal stats.

1.4.1 (2018-11-12)
------------------
* Hotfix that fixes an issue that would cause ``zonal_statistics`` to crash if
  a polygon were outside of the raster's bounding box.

1.4.0 (2018-11-12)
------------------
* Adding error checking to ensure that ``target_pixel_size`` passed to
  ``warp_raster`` and ``align_and_resize_raster_stack`` are validated to ensure
  they are in the correct format. This solves an issue where an incorrect
  value, such as a single numerical value, resolve into readable exception
  messages.
* Added a ``gdal_warp_options`` parameter to ``align_and_resize_raster_stack``
  and ``warp_raster`` whose contents get passed to gdal.Warp's ``warpOptions``
  parameter. This was implemented to expose the CUTLINE_TOUCH_ALL
  functionality but could be used for any gdal functionality.
* Modified ``rasterize`` API call to make ``burn_values`` and ``option_list``
  both optional parameters, along with error checking to ensure a bad input's
  behavior is understood.
* Exposing GeoTIFF creation options for all the ``pygeoprocessing.routing``
  functions which create rasters. This is consistent with the creation
  options exposed in the main ``pygeoprocessing`` API.
* Removing ``'mean_pixel_size'`` as a return value from ``get_raster_info``,
  this is because this parameter is easily misused and easily calculated if
  needed. This is a "What good programmers need, not what bad programmers
  want." feature.

1.3.1 (2018-10-25)
------------------
* Hotfix to patch an infinite loop when aggregating upstream or downstream
  with custom rasters.

1.3.0 (2018-10-25)
------------------
* Fixed a handful of docstring errors.
* Improved runtime of ``zonal_statistics`` by a couple of orders of magnitude
  for large vectors by using spatial indexes when calculating disjoint polygon
  overlap sets, using database transactions, and memory buffers.
* Improved runtime performance of ``reproject_vector`` by using database
  transactions.
* Improved logging for long runtimes in ``zonal_statistics``.
* Changed ``zonal_statistics`` API and functionality to aggregate across the
  FIDs of the aggregate vector. This is to be consistent with QGIS and other
  zonal statistics functionality. Additionally, fixed a bug where very small
  polygons might not get aggregated if they lie in the same pixel as another
  polygon that does not intersect it. The algorithm now runs in two passes:

    * aggregate pixels whose centers intersect the aggregate polygons
    * any polygons that were not aggregated are geometrically intersected
      with pixels to determine coverage.

* Removed the ``calculate_raster_stats`` function since it duplicates GDAL
  functionality, but with a slower runtime, and now functions in
  ``pygeoprocessing`` that create rasters also calculate stats on the fly if
  desired.
* Fixes an issue in ``get_raster_info`` and ``get_vector_info`` where the path
  to the raster/vector includes non-standard OS pathing (such as a NETCDF),
  info will still calculate info.
* Added functionality to ``align_raster_stack`` and ``warp_raster`` to define
  a base spatial reference system for rasters if not is not defined or one
  wishes to override the existing one. This functionality is useful when
  reprojecting a rasters that does not have a spatial reference defined in the
  dataset but is otherwise known.
* Added a ``weight_raster_path_band`` parameter to both
  ``flow_accumulation_d8`` and ``flow_accumulation_mfd`` that allows the
  caller to use per-pixel weights from a parallel raster as opposed to
  assuming a weight of 1 per pixel.
* Added a ``weight_raster_path_band`` parameter to both
  ``distance_to_channel_mfd`` and ``distance_to_channel_d8`` that allows the
  caller to use per-pixel weights from a parallel raster as opposed to
  assuming a distance of 1 between neighboring pixels or sqrt(2) between
  diagonal ones.
* Added an option to ``reproject_vector`` that allows a caller to specify
  which fields, if any, to copy to the target vector after reprojection.
* Adding a check in ``align_and_resize_raster_stack`` for duplicate target
  output paths to avoid problems where multiple rasters are being warped to
  the same path.
* Created a public ``merge_bounding_box_list`` function that's useful for
  union or intersection of bounding boxes consistent with the format in
  PyGeoprocessing.
* Added functionality in ``align_and_resize_raster_stack`` and ``warp_raster``
  to use a vector to mask out pixel values that lie outside of the polygon
  coverage area. This parameter is called ``vector_mask_options`` and is
  fully documented in both functions. It is similar to the cutline
  functionality provided in ``gdal.Warp``.
* Fixed an issue in the ``flow_accumulation_*`` functions where a weight
  raster whose values were equal to the nodata value of the flow accumulation
  raster OR simply nodata would cause infinite loops.

1.2.3 (2018-07-25)
------------------
* Exposing a parameter and setting reasonable defaults for the number of
  processes to allocate to ``convolve_2d`` and ``warp_raster``. Fixes an issue
  where the number of processes could exponentiate if many processes were
  calling these functions.
* Fixing an issue on ``zonal_statistics`` and ``convolve_2d`` that would
  attempt to both read and write to the target raster with two different GDAL
  objects. This caused an issue on Linux where the read file was not caught up
  with the written one. Refactored to use only one handle.
* Fixing a rare race condition where an exception could occur in
  ``raster_calculator`` that would be obscured by an access to an object that
  had not yet been assigned.
* ``align_and_resize_raster_stack`` now terminates its process pool.
* Increased the timeout in joining ``raster_calculator``'s stats worker.
  On a slow system 5 seconds was not quite enough time.

1.2.2 (2018-07-25)
------------------
* Hotfixed a bug that would cause numpy arrays to be treated as broadcastable
  even if they were passed in "raw".

1.2.1 (2018-07-22)
------------------
* Fixing an issue with ``warp_raster`` that would round off bounding boxes
  for rasters that did not fit perfectly into the target raster's provided
  pixel size.
* Cautiously ``join``\ing all process pools to avoid a potential bug where a
  deamonized subprocess in a process pool may still have access to a raster
  but another process may require write access to it.

1.2.0 (2018-07-19)
------------------
* Several PyGeoprocessing functions now take advantage of multiple CPU cores:

  * ``raster_calculator`` uses a separate thread to calculate raster
    statistics in a ``nogil`` section of Cython code. In timing with a big
    rasters we saw performance improvements of about 35%.
  * ``align_and_resize_raster_stack`` uses as many CPU cores, up to the number
    of CPUs reported by multiprocessing.cpu_count (but no less than 1), to
    process each raster warp while also accounting for the fact that
    ``gdal.Warp`` uses 2 cores on its own.
  * ``warp_raster`` now directly uses ``gdal.Warp``'s multithreading directly.
    In practice it seems to utilize two cores.
  * ``convolve_2d`` attempts to use ``multiprocessing.cpu_count`` cpus to
    calculate separable convolutions per block while using the main thread to
    aggregate  and write the result to the target raster. In practice we saw
    this improve runtimes by about 50% for large rasters.
* Fixed a bug that caused some nodata values to not be treated as nodata
  if there was a numerical roundoff.
* A recent GDAL upgrade (might have been 2.0?) changed the reference to
  nearest neighbor interpolation from 'nearest' to 'near'. This PR changes
  PyGeoprocessing to be consistent with that change.
* ``raster_calculator`` can now also take "raw" arguments in the form of a
  (value, "raw") tuple. The parameter ``value`` will be passed directly to
  ``local_op``. Scalars are no longer a special case and need to be passed as
  "raw" parameters.
* Raising ``ValueError`` in ``get_raster_info`` and ``get_vector_info`` in
  cases where non-filepath non-GIS values are passed as parameters. Previously
  such an error would result in an unhelpful error in the GDAL library.

1.1.0 (2018-07-06)
------------------
* PyGeoprocessing now supports Python 2 and 3, and is tested on python 2.7
  and 3.6  Testing across multiple versions is configured to be run via
  ``tox``.
* After testing (tox configuration included under ``tox-libcompat.ini``),
  numpy requirement has been dropped to ``numpy>=1.10.0`` and scipy has been
  modified to be ``scipy>=0.14.1,!=0.19.1``.
* A dependency on ``future`` has been added for compatibility between python
  versions.
* Fixed a crash in ``pygeoprocessing.routing.flow_dir_mfd`` and
  ``flow_dir_d8`` if a base raster was passed in that did not have a power of
  two blocksize.
* ``raster_calculator`` can now take numpy arrays and scalar values along with
  raster path band tuples. Arrays and scalars are broadcast to the raster size
  according to numpy array broadcasting rules.
* ``align_and_resize_raster_stack`` can now take a desired target projection
  which causes all input rasters to be warped to that projection on output.

1.0.1 (2018-05-16)
------------------
* Hotfix patch to remove upper bound on required numpy version. This was
  causing a conflict with InVEST's looser requirement. Requirement is now
  set to >=1.13.0.

1.0.0 (2018-04-29)
------------------
* This release marks a feature-complete version of PyGeoprocessing with a
  full suite of routing and geoprocessing capabilities.
* ``pygeoprocessing.routing`` module has a ``flow_dir_mfd`` function that
  calculates a 32 bit multiple flow direction raster.
* ``pygeoprocessing.routing`` module has a ``flow_accumulation_mfd`` function
  that uses the flow direction raster from
  ``pygeoprocessing.routing.flow_dir_mfd`` to calculate a per-pixel continuous
  flow accumulation raster.
* ``pygeoprocessing.routing`` module has a ``distance_to_channel_mfd``
  function that calculates distance to a channel raster given a
  pygeoprocessing MFD raster.
* ``pygeoprocessing.routing`` module has a ``distance_to_channel_d8`` function
  that calculates distance to a channel raster given a pygeoprocessing D8
  raster.

0.7.0 (2018-04-18)
------------------
* Versioning is now handled by ``setuptools_scm`` rather than
  ``natcap.versioner``.  ``pygeoprocessing.__version__`` is now fetched from
  the package metadata.
* Raster creation defaults now set "COMPRESS=LZW" for all rasters created in
  PyGeoprocessing, including internal temporary rasters. This option was
  chosen after profiling large raster creation runs on platter hard drives.
  In many cases processing time was dominated by several orders of magnitude
  as a write-to-disk. When compression is turned on overall runtime of very
  large rasters is significantly reduced. Note this otherwise increases the
  runtime small raster creation and processing by a small amount.
* ``pygeoprocessing.routing`` module now has a ``fill_pits``, function which
   fills hydrological pits with a focus on runtime efficiency, memory space
   efficiency, and cache locality.
* ``pygeoprocessing.routing`` module has a ``flow_dir_d8`` that uses largest
  slope to determine the downhill flow direction.
* ``pygeoprocessing.routing`` module has a ``flow_accumulation_d8`` that uses
  a pygeoprocessing D8 flow direction raster to calculate per-pixel flow
  accumulation.
* Added a ``merge_rasters`` function to ``pygeoprocessing`` that will mosaic a
  set of rasters in the same projection, pixel size, and band count.

0.6.0 (2017-01-10)
------------------
* Added an optional parameter to ``iterblocks`` to allow the ``largest_block``
  to be set something other than the PyGeoprocessing default. This in turn
  allows the ``largest_block`` parameter in ``raster_calculator`` to be passed
  through to ``iterblocks``.
* Upgraded PyGeoprocessing GDAL dependency to >=2.0.
* Added a ``working_dir`` optional parameter to ``zonal_statistics``,
  ``distance_transform_edt``, and ``convolve_2d`` which specifies a directory
  in which temporary files will be created during execution of the function.
  If set to ``None`` files are created in the default system temporary
  directory.

0.5.0 (2017-09-14)
------------------
* Fixed an issue where NETCDF files incorrectly raised Exceptions in
  ``raster_calculator``  and ``rasterize`` because they aren't filepaths.
* Added a NullHandler so that users wouldn't get an error that a logger
  handler was undefined.
* Added ``ignore_nodata``, ``mask_nodata``, and ``normalize_kernel`` options
  to ``convolve_2d`` which make this function capable of adapting the nodata
  overlap with the kernel rather than zero out the result, as well as on
  the fly normalization of the kernel for weighted averaging purposes. This
  is in part to make this functionality more consistent with ArcGIS's
  spatial filters.

0.4.4 (2017-08-18)
------------------
* When testing for raster alignment ``raster_calculator`` no longer checks the
  string equality for projections or geotransforms.  Instead it only checks
  raster size equality.  This fixes issues where users rasters DO align, but
  have a slightly different text format of the WKT of projection.  It also
  abstracts the problem of georeferencing away from raster_calculator that is
  only a grid based operation.

0.4.3 (2017-08-16)
------------------
* Changed the error message in ``reclassify_raster`` so it's more informative
  about how many values are missing and the values in the input lookup table.
* Added an optional parameter ``target_nodata`` to ``convolve_2d`` to set the
  desired target nodata value.

0.4.2 (2017-06-20)
------------------
* Hotfix to fix an issue with ``iterblocks`` that would return signed values
  on unsigned raster types.
* Hotfix to correctly cite Natural Capital Project partners in license and
  update the copyright year.
* Hotfix to patch an issue that gave incorrect results in many PyGeoprocessing
  functions when a raster was passed with an NoData value.  In these cases the
  internal raster block masks would blindly pass through on the first row
  since a test for ``numpy.ndarray == None`` is ``False`` and later
  ``x[False]`` is the equivalent of indexing the first row of the array.

0.4.1 (2017-06-19)
------------------
* Non-backwards compatible refactor of core PyGeoprocessing geoprocessing
  pipeline. This is to in part expose only orthogonal functionality, address
  runtime complexity issues, and follow more conventional GIS naming
  conventions. Changes include:

    * Full test coverage for ``pygeoprocessing.geoprocessing`` module
    * Dropping "uri" moniker in lieu of "path".
    * If a raster path is specified and operation requires a single band,
      argument is passed as a "(path, band)" tuple where the band index starts
      at 1 as convention for raster bands.
    * Shapefile paths are assumed to operate on the first layer.  It is so
      rare for a shapefile to have more than one layer, functions that would
      be confused by multiple layers have a layer_index that defaults to 0
      that can be overridden in the call.
    * Be careful, many of the parameter orders have been changed and renamed.
      Generally inputs come first, outputs last.  Input parameters are
      often prefixed with "base\_" while output parameters are prefixed with
      "target\_".
    * Functions that take rasters as inputs must have their rasters aligned
      before the call to that function.  The function
      ``align_and_resize_raster_stack`` can handle this.
    * ``vectorize_datasets`` refactored to ``raster_calculator`` since that
      name is often used as a convention when referring to raster
      calculations.
    * ``vectorize_points`` refactored to meaningful ``interpolate_points``.
    * ``aggregate_by_shapefile`` refactored to ``zonal_statistics`` and now
      returns a dictionary rather than a named tuple.
    * All functions that create rasters expose the underlying GeoTIFF options
      through a default parameter ``gtiff_creation_options`` which default to
      "('TILED=YES', 'BIGTIFF=IF_SAFER')".
    * Individual functions for raster and vector properties have been
      aggregated into ``get_raster_info`` and ``get_vector_info``
      respectively.
    * Introducing ``warp_raster`` to wrap GDAL's ``ReprojectImage``
      functionality that also works on bounding box clips.
    * Removed the ``temporary_filename()`` paradigm.  Users should manage
      temporary filenames directly.
    * Numerous API changes from the 0.3.x version of PyGeoprocessing.
* Fixing an issue with aggregate_raster_values that caused a crash if feature
  IDs were not in increasing order starting with 0.
* Removed "create_rat/create_rat_uri" and migrated it to
  natcap.invest.wind_energy; the only InVEST model that uses that function.
* Fixing an issue with aggregate_raster_values that caused a crash if feature
  IDs were not in increasing order starting with 0.
* Removed "create_rat/create_rat_uri" and migrated it to
  natcap.invest.wind_energy; the only InVEST model that uses that function.

0.3.3 (2017-02-09)
------------------
* Fixing a memory leak with large polygons when calculating disjoint set.

0.3.2 (2017-01-24)
------------------
* Hotfix to patch an issue with watershed delineation packing that causes some
  field values to lose precision due to default field widths being set.

0.3.1 (2017-01-18)
------------------
* Hotfix patch to address an issue in watershed delineation that doesn't pack
  the target watershed output file.  Half the shapefile consists of features
  polygonalized around nodata values that are flagged for deletion, but not
  removed from the file.  This patch packs those features and returns a clean
  watershed.

0.3.0 (2016-10-21)
------------------
* Added ``rel_tol`` and ``abs_tol`` parameters to ``testing.assertions`` to be
  consistent with PEP485 and deal with real world testing situations that
  required an absolute tolerance.
* Removed calls to ``logging.basicConfig`` throughout pygeoprocessing.  Client
  applications may need to adjust their logging if pygeoprocessing's log
  messages are desired.
* Added a flag  to ``aggregate_raster_values_uri`` that can be used to
  indicate incoming polygons do not overlap, or the user does not care about
  overlap. This can be used in cases where there is a computational or memory
  bottleneck in calculating the polygon disjoint sets that would ultimately be
  unnecessary if it is known a priori that such a check is unnecessary.
* Fixed an issue where in some cases different nodata values for 'signal' and
  'kernel' would cause incorrect convolution results in ``convolve_2d_uri``.
* Added functionality to ``pygeoprocessing.iterblocks`` to iterate over
  largest memory aligned block that fits into the number of elements provided
  by the parameter.  With default parameters, this uses a ceiling around 16MB
  of memory per band.
* Added functionality to ``pygeoprocessing.iterblocks`` to return only the
  offset dictionary.  This functionality would be used in cases where memory
  aligned writes are desired without first reading arrays from the band.
* Refactored ``pygeoprocessing.convolve_2d_uri`` to use ``iterblocks`` to take
  advantage of large block sizes for FFT summing window method.
* Refactoring source side to migrate source files from [REPO]/pygeoprocessing
  to [REPO]/src/pygeoprocessing.
* Adding a pavement script with routines to fetch SVN test data, build a
  virtual environment, and clean the environment in a Windows based operating
  system.
* Adding ``transform_bounding_box`` to calculate the largest projected
  bounding box given the four corners on a local coordinate system.
* Removing GDAL, Shapely from the hard requirements in setup.py.  This will
  allow pygeoprocessing to be built by package managers like pip without these
  two packages being installed.  GDAL and Shapely will still need to be
  installed for pygeoprocessing to run as expected.
* Fixed a defect in ``pygeoprocessing.testing.assert_checksums_equal``
  preventing BSD-style checksum files from being analyzed correctly.
* Fixed an issue in reclassify_dataset_uri that would cause an exception if
  the incoming raster didn't have a nodata value defined.
* Fixed a defect in ``pygeoprocessing.geoprocessing.get_lookup_from_csv``
  where the dialect was unable to be detected when analyzing a CSV that was
  larger than 1K in size.  This fix enables the correct detection of comma or
  semicolon delimited CSV files, so long as the header row by itself is not
  larger than 1K.
* Intra-package imports are now relative.  Addresses an import issue for users
  with multiple copies of pygeoprocessing installed across multiple Python
  installations.
* Exposed cython routing functions so they may be imported from C modules.
* ``get_lookup_from_csv`` attempts to determine the dialect of the CSV instead
  of assuming comma delimited.
* Added relative numerical tolerance parameters to the PyGeoprocessing raster
  and csv tests with in the same API style as ``numpy.testing.allclose``.
* Fixed an incomparability with GDAL 1.11.3 bindings that expects a boolean
  type in ``band.ComputeStatistics``.  Before this fix PyGeoprocessing would
  crash with a TypeError on many operations.
* Fixed a defect in pygeoprocessing.routing.calculate_transport where the
  nodata types were cast as int even though the base type of the routing
  rasters were floats.  In extreme cases this could cause a crash on a type
  that could not be converted to an int, like an ``inf``, and in subtle cases
  this would result in nodata values in the raster being ignored during
  routing.
* Added functions to construct raster and vectors on disk from reasonable
  datatypes (numpy matrices for rasters, lists of Shapely geometries for
  vectors).
* Fixed an issue where reproject_datasource_uri would add geometry that
  couldn't be projected directly into the output datasource.  Function now
  only adds geometries that transformed without error and reports if any
  features failed to transform.
* Added file flushing and dataset swig deletion in reproject_datasource_uri to
  handle a race condition that might have been occurring.
* Fixed an issue when "None" was passed in on new raster creation that would
  attempt to directly set that value as the nodata value in the raster.
* Added basic filetype-specific assertions for many geospatial filetypes, and
  tests for these assertions.  These assertions are exposed in
  ``pygeoprocessing.testing``.
* Pygeoprocessing package tests can be run by invoking
  ``python setup.py nosetests``.  A subset of tests may also be run from an
  installed pygeoprocessing distribution by calling
  ``pygeoprocessing.test()``.
* Fixed an issue with reclassify dataset that would occur when small rasters
  whose first memory block would extend beyond the size of the raster thus
  passing in "0" values in the out of bounds area. Reclassify dataset
  identified these as valid pixels, even though vectorize_datsets would mask
  them out later.  Now vectorize_datasets only passes memory blocks that
  contain valid pixel data to its kernel op.
* Added support for very small AOIs that result in rasters less than a pixel
  wide.  Additionally an ``all_touched`` flag was added to allow the
  ALL_TOUCHED=TRUE option to be passed to RasterizeLayer in the AOI mask
  calculation.
* Added watershed delineation routine to
  pygeoprocessing.routing.delineate_watershed.  Operates on a DEM and point
  shapefile, optionally snaps outlet points to nearest stream as defined by a
  thresholded flow accumulation raster and copies the outlet point fields into
  the constructed watershed shapefile.
* Fixing a memory leak in block caches that held on to dataset, band, and
  block references even after the object was destroyed.
* Add an option to route_flux that lets the current pixel's source be included
  in the flux, or not.  Previous version would include on the source no matter
  what.
* Now using natcap.versioner for versioning instead of local versioning logic.

0.2.2 (2015-05-07)
------------------
* Adding MinGW-specific compiler flags for statically linking pygeoprocessing
  binaries against libstdc++ and libgcc.  Fixes an issue on many user's
  computers when installing from a wheel on the Python Package Index without
  having two needed DLLs on the PATH, resulting in an ImportError on pygeoprocessing.geoprocessing_core.pyd.
* Fixing an issue with versioning where 'dev' was displayed instead of the
  version recorded in pygeoprocessing/__init__.py.
* Adding all pygeoprocessing.geoprocessing functions to
  pygeoprocessing.__all__, which allows those functions to appear when
  calling help(pygeoprocessing).
* Adding routing_core.pxd to the manifest.  This fixes an issue where some
  users were unable to compiler pygeoprocessing from source.

0.2.1 (2015-04-23)
------------------
* Fixed a bug on the test that determines if a raster should be memory
  blocked.  Rasters were not getting square blocked if the memory block was
  row aligned.  Now creates 256x256 blocks on rasters larger than 256x256.
* Updates to reclassify_dataset_uri to use numpy.digitize rather than Python
  loops across the number of keys.
* More informative error messages raised on incorrect bounding box mode.
* Updated docstring on get_lookup_from_table to indicate the headers are case
  insensitive.
* Added updates to align dataset list that report which dataset is being
  aligned.  This is helpful for logging feedback when many datasets are passed
  in that don't take long enough to get a report from the underlying reproject
  dataset function.
* pygeoprocessing.routing.routing_core includes pxd to be ``cimport``\able
  from a Cython module.

0.2.0 (2015-04-14)
------------------
* Fixed a library wide issue relating to the underlying numpy types of
  GDT_Byte Datasets.  Now correctly identify the signed and unsigned versions
  and removed all instances where code used to mod byte data to unsigned data
  and correctly creates signed/unsigned byte datasets during resampling.
* Removed extract_band_and_nodata function since it exposes the underlying
  GDAL types.
* Removed reclassify_by_dictionary since reclassify_dataset_uri provided
  almost the same functionality and was widely used.
* Removed the class OrderedDict that was not used.
* Removed the function calculate_value_not_in_dataset since it loaded the
  entire dataset into memory and was not useful.

0.1.8 (2015-04-13)
------------------
* Fixed an issue on reclassifying signed byte rasters that had negative nodata
  values but the internal type stored for vectorize datasets was unsigned.

0.1.7 (2015-04-02)
------------------
* Package logger objects are now identified by python hierarchical package
  paths (e.g. pygeoprocessing.routing)
* Fixed an issue where rasters that had undefined nodata values caused
  striping in the reclassify_dataset_uri function.

0.1.6 (2015-03-24)
------------------
* Fixing LICENSE.TXT to .txt issue that keeps reoccurring.

0.1.5 (2015-03-16)
------------------
* Fixed an issue where int32 dems with INT_MIN as the nodata value were being
  treated as real DEM values because of an internal cast to a float for the
  nodata type, but a cast to double for the DEM values.
* Fixed an issue where flat regions, such as reservoirs, that could only drain
  off the edge of the DEM now correctly drain as opposed to having undefined
  flow directions.

0.1.4 (2015-03-13)
------------------
* Fixed a memory issue for DEMs on the order of 25k X 25k, still may have
  issues with larger DEMs.

0.1.3 (2015-03-08)
------------------
* Fixed an issue so tox correctly executes on the repository.
* Created a history file to document current and previous releases.
* Created an informative README.rst.

0.1.2 (2015-03-04)
------------------
* Fixing issue that caused "LICENSE.TXT not found" during pip install.

0.1.1 (2015-03-04)
------------------
* Fixing issue with automatic versioning scheme.

0.1.0 (2015-02-26)
------------------
* First release on PyPI.
