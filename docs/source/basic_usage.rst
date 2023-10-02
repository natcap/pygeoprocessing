Basic Usage
===========

``pygeoprocessing`` uses GDAL to read and write all GDAL-supported
`raster <https://gdal.org/user/raster_data_model.html>`_ and
`vector <https://gdal.org/user/vector_data_model.html>`_ file formats.
The utility functions :func:`get_raster_info <pygeoprocessing.get_raster_info>`
and :func:`get_vector_info <pygeoprocessing.get_vector_info>` read a file’s
metadata into a dictionary. These values can be used as parameters for a
variety of ``pygeoprocessing`` functions where properties like pixel sizes,
bounding boxes, and nodata values need to be defined.

.. code::

    >>> import pygeoprocessing

.. code::

    >>> raster_a_path = 'dem_utm.tif'
    >>> raster_b_path = 'mswep.tif'
    >>> raster_info = pygeoprocessing.get_raster_info(raster_a_path)
    >>> raster_info
    {'block_size': [256, 256],
     'bounding_box': [364636.41313796316,
                      5229554.621510817,
                      530566.4131379632,
                      5362724.621510817],
     'datatype': 3,
     'file_list': ['dem_utm.tif'],
     'geotransform': (364636.41313796316, 30.0, 0.0, 5362724.621510817, 0.0, -30.0),
     'n_bands': 1,
     'nodata': [-32768.0],
     'numpy_type': <class 'numpy.int16'>,
     'overviews': [],
     'pixel_size': (30.0, -30.0),
     'projection_wkt': 'PROJCS["WGS 84 / UTM zone 10N",GEOGCS["WGS '
                       '84",DATUM["WGS_1984",SPHEROID["WGS '
                       '84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-123],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32610"]]',
     'raster_size': (5531, 4439)}


Aligning rasters
****************

| A basic raster-processing workflow often begins by aligning a stack of
  rasters.
| The :func:`align_and_resize_raster_stack <pygeoprocessing.align_and_resize_raster_stack>`
  function takes a list of overlapping rasters that may all have different
  extents, pixel sizes, and projections. The resulting stack of rasters will
  all be aligned and ready for use in pixel-stack operations such as
  :func:`raster_map <pygeoprocessing.raster_map>` or
  :func:`raster_calculator <pygeoprocessing.raster_calculator>`.

.. code::

    >>> base_raster_list = [raster_a_path, raster_b_path]
    >>> target_raster_list = [x.replace('.tif', '_aligned.tif') for x in base_raster_list]

.. code::

    >>> pygeoprocessing.align_and_resize_raster_stack(
    ...    base_raster_path_list=base_raster_list,
    ...    target_raster_path_list=target_raster_list,
    ...    resample_method_list=['bilinear', 'bilinear'],
    ...    target_pixel_size=raster_info['pixel_size'],
    ...    bounding_box_mode='intersection',
    ...    target_projection_wkt=raster_info['projection_wkt'])

Interfacing with files
**********************

Pygeoprocessing functions typically interact with GIS datasets via their
filename. Some functions, such as
:func:`align_and_resize_raster_stack <pygeoprocessing.align_and_resize_raster_stack>`,
operate on all bands of a raster. Other times it is necessary to specify
which band of a raster, or which layer of a vector should be used.

For example, the :func:`zonal_statistics <pygeoprocessing.zonal_statistics>`
function requires the user to specify which band of the raster from which to
calculate statistics. This is done using a ``tuple``, or ``list``, where the
first element is the filepath, and the second is the band index:

.. code::

    >>> path_band_tuple = (raster_b_path, 1)  # band indices start at 1 (not 0), by GDAL convention

.. code::

    >>> stats_dict = pygeoprocessing.zonal_statistics(
    ...    base_raster_path_band=path_band_tuple,
    ...    aggregate_vector_path='watersheds.gpkg',
    ...    aggregate_layer_name='watersheds')  # if the vector only contains 1 layer, this can be `None`, or ommitted

| An example of the path-band object when a function operates on multiple
  rasters, such as for
  :func:`raster_calculator <pygeoprocessing.raster_calculator>`:

.. code::

    >>> raster_path_band_list = [(raster_a_path, 1), (raster_b_path, 1)]

