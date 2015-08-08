import os
import shutil
import collections
import tempfile
import subprocess
import logging
import inspect
import warnings

import numpy

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

LOGGER = logging.getLogger('pygeoprocessing.testing.sampledata')

ReferenceData = collections.namedtuple('ReferenceData',
                                       'projection origin pixel_size')
gdal.AllRegister()
GDAL_DRIVERS = sorted([gdal.GetDriver(i).GetDescription()
                       for i in range(1, gdal.GetDriverCount())])
OGR_DRIVERS = sorted([ogr.GetDriver(i).GetName()
                      for i in range(ogr.GetDriverCount())])

# Higher index in list represents higher precision
DTYPES = [
    (numpy.byte, gdal.GDT_Byte),
    (numpy.int16, gdal.GDT_Int16),
    (numpy.int32, gdal.GDT_Int32),
    (numpy.uint16, gdal.GDT_UInt16),
    (numpy.uint32, gdal.GDT_UInt32),
    (numpy.float32, gdal.GDT_Float32),
    (numpy.float64, gdal.GDT_Float64),
]

# Mappings of numpy -> GDAL types and GDAL -> numpy types.
NUMPY_GDAL_DTYPES = dict(DTYPES)
GDAL_NUMPY_TYPES = dict((g, n) for (n, g) in NUMPY_GDAL_DTYPES.iteritems())

# Build up an index mapping GDAL datatype index to the string label of the
# datatype.  Helpful for Debug messages.
GDAL_DTYPE_LABELS = {}
for _attrname in dir(gdal):
    if _attrname.startswith('GDT_'):
        _dtype_value = getattr(gdal, _attrname)
        GDAL_DTYPE_LABELS[_dtype_value] = _attrname

SRS_COLOMBIA = ReferenceData(
    projection="""PROJCS["MAGNA-SIRGAS / Colombia Bogota zone",
    GEOGCS["MAGNA-SIRGAS",
        DATUM["Marco_Geocentrico_Nacional_de_Referencia",
        SPHEROID["GRS 1980",6378137,298.2572221010002,
            AUTHORITY["EPSG","7019"]],
        TOWGS84[0,0,0,0,0,0,0],
        AUTHORITY["EPSG","6686"]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433],
        AUTHORITY["EPSG","4686"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",4.596200416666666],
    PARAMETER["central_meridian",-74.07750791666666],
    PARAMETER["scale_factor",1],
    PARAMETER["false_easting",1000000],
    PARAMETER["false_northing",1000000],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AUTHORITY["EPSG","3116"]]""",
    origin=(444720, 3751320),
    pixel_size=lambda x: (x, -1. * x)
)
SRS_WILLAMETTE = ReferenceData(
    projection="""PROJCS["UTM Zone 10, Northern Hemisphere",
        GEOGCS["NAD83",
            DATUM["North_American_Datum_1983",
                SPHEROID["GRS 1980",6378137,298.257222101,
                    AUTHORITY["EPSG","7019"]],
                TOWGS84[0,0,0,0,0,0,0],
                AUTHORITY["EPSG","6269"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.0174532925199433,
                AUTHORITY["EPSG","9108"]],
            AUTHORITY["EPSG","4269"]],
        PROJECTION["Transverse_Mercator"],
        PARAMETER["latitude_of_origin",0],
        PARAMETER["central_meridian",-123],
        PARAMETER["scale_factor",0.9996],
        PARAMETER["false_easting",500000],
        PARAMETER["false_northing",0],
        UNIT["METERS",1]]""",
    origin=(443723.127327877911739,4956546.905980412848294),
    pixel_size=lambda x: (x, -1. * x)
)

VECTOR_FIELD_TYPES = {
    'int': ogr.OFTInteger,
    'intlist': ogr.OFTIntegerList,
    'real': ogr.OFTReal,
    'reallist': ogr.OFTRealList,
    'string': ogr.OFTString,
    'stringlist': ogr.OFTStringList,
    'widestring': ogr.OFTWideString,
    'widestringlist': ogr.OFTWideStringList,
    'binary': ogr.OFTBinary,
    'date': ogr.OFTDate,
    'time': ogr.OFTTime,
    'datetime': ogr.OFTDateTime,
}

# Later versions of OGR include 64-bit integer/integerlist types.
# Add them to the available types if they are available.
for keyname, typename in [('int64', 'OFTInteger64'),
                          ('int64list', 'OFTInteger64List')]:
    try:
        VECTOR_FIELD_TYPES[keyname] = getattr(ogr, typename)
    except AttributeError:
        pass


def dtype_precision(dtype):
    """
    Return the precision index of the datatype provided.

    Parameters:
        dtype (numpy.dtype or int GDAL datatype)

    Returns:
        The precision index relative to the other numpy/gdal type pairs.
    """

    if isinstance(dtype, type):
        # It's a numpy type.
        dtype_tuple_index = 0
    elif isinstance(dtype, int):
        # It's a GDAL type.
        dtype_tuple_index = 1
    else:
        raise RuntimeError(('Datatype %s not recognized.  Must be a numpy or '
                            'gdal datatype') % dtype)
    return map(lambda x: x[dtype_tuple_index], DTYPES).index(dtype)


def make_geotransform(x_len, y_len, origin):
    """
    Build an array of affine transformation coefficients.


    Parameters:
        x_len (int or float): The length of a pixel along the x axis.
        y_len (int of float): The length of a pixel along the y axis.
        origin (tuple of ints or floats): The origin of the raster, a
            2-element tuple.

    Returns:
        A 6-element list with this structure:
        [
            Origin along x-axis,
            Length of a pixel along the x axis,
            0.0,  (this is the standard in north-up projections)
            Origin along y-axis,
            Length of a pixel along the y axis,
            0.0   (this is the standard in north-up projections)
        ]
    """
    return [origin[0], x_len, 0, origin[1], 0, y_len]


def cleanup(uri):
    """Remove the uri.  If it's a folder, recursively remove its contents."""
    if os.path.isdir(uri):
        shutil.rmtree(uri)
    else:
        os.remove(uri)


def raster(band_matrix, origin, projection_wkt, nodata, pixel_size,
           datatype='auto', format='GTiff', dataset_opts=None, filename=None):
    """
    Create a temp GDAL raster on disk.

    Parameters:
        band_matrix (numpy.ndarray) - a numpy matrix representing pixel
            values.
        origin (tuple of numbers) - A 2-element tuple representing the origin
            of the pixel values in the raster.  This must be a tuple of numbers.
        projection_wkt (string) - A string WKT represntation of the projection
            to use in the output raster.
        nodata (int or float) - The nodata value for the raster.
        pixel_size (tuple of numbers) - A 2-element tuple representing the
            size of all pixels in the raster arranged in the order (x, y).
            Either or both of these values could be negative.
        datatype (int or 'auto') - A GDAL datatype. If 'auto', a reasonable
            datatype will be chosen based on the datatype of the numpy matrix.
        format='GTiff' (string) - The string driver name to use.  Determines
            the output format of the raster.  Defaults to GeoTiff.  See
            http://www.gdal.org/formats_list.html for a list of available
            formats.
        dataset_opts=None (list of strings) - A list of strings to pass to
            the underlying GDAL driver for creating this raster.  Possible
            options are usually format dependent.  If None, no options will
            be passed to the driver.
        filename=None (string) - If provided, the new raster should be created
            at this filepath.  If None, a new temporary file will be created
            within your tempfile directory (within `tempfile.gettempdir()`)
            and this path will be returned from the function.

    Outputs:
        Writes a raster created with the given options.

    Returns:
        The string path to the new raster created on disk.
    """
    if filename is None:
        _, out_uri = tempfile.mkstemp()
        out_uri += '.tif'
    else:
        out_uri = filename

    # Derive reasonable gdal dtype from numpy matrix dtype if needed
    numpy_dtype = band_matrix.dtype.type
    if datatype == 'auto':
        datatype = NUMPY_GDAL_DTYPES[numpy_dtype]

    # Warn the user if loss of precision is likely when converting from numpy
    # datatype to a gdal datatype.
    if dtype_precision(numpy_dtype) > dtype_precision(datatype):
        gdal_dtype_label = GDAL_DTYPE_LABELS[datatype]
        numpy_dtype_label = numpy_dtype.__name__
        message = ('Pixels have a datatype of %s, which is greater than the '
                   'allowed precision of the GDAL datatype %s.  Loss of '
                   'precision is likely when saving to the new raster.')
        message %= (numpy_dtype_label, gdal_dtype_label)
        warnings.warn(message)

    # Create a raster given the shape of the pixels given the input driver
    n_rows, n_cols = band_matrix.shape
    driver = gdal.GetDriverByName(format)
    if driver is None:
        raise RuntimeError(
            ('GDAL driver "%s" not found.  '
             'Available drivers: %s') % (format, ', '.join(GDAL_DRIVERS)))

    if dataset_opts is None:
        dataset_opts = []

    new_raster = driver.Create(out_uri, n_cols, n_rows, 1, datatype,
                               options=dataset_opts)

    # create some projection information based on the GDAL tutorial at
    # http://www.gdal.org/gdal_tutorial.html
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection_wkt)
    new_raster.SetProjection(srs.ExportToWkt())
    geotransform = make_geotransform(pixel_size[0], pixel_size[1], origin)
    new_raster.SetGeoTransform(geotransform)

    band = new_raster.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(band_matrix, 0, 0)

    band = None
    new_raster = None
    return out_uri


def vector(
        geometries, projection, fields=None, attributes=None,
        vector_format='GeoJSON', filename=None):
    """Create a temp OGR vector on disk.

        geometries (list) - a list of Shapely objects.
        projection (string)- a WKT representation of the vector's projection.
        fields (dict or None)- a python dictionary mapping string fieldname
            to a string datatype representation of the target ogr fieldtype.
            Example: {'ws_id': 'int'}.  See `VECTOR_FIELD_TYPES.keys()`
            for the complete list of all allowed types.  If None, the datatype
            will be determined automatically based on the types of the
            attribute values.
        attributes (list of dicts) - a list of python dictionary mapping
            fieldname to field value.  The field value's type must match the
            type defined in the fields input.  It is an error if it doesn't.
        vector_format (string)- a python string indicating the OGR format to
            write. GeoJSON is pretty good for most things, but doesn't handle
            multipolygons very well. 'ESRI Shapefile' is usually a good bet.
        filename=None (None or string)- None or a python string where the file
            should be saved. If None, the vector will be saved to a temporary
            folder.

    Returns a string URI to the location of the vector on disk."""

    if fields is None:
        fields = {}

    if attributes is None:
        attributes = [{} for _ in range(len(geometries))]

    num_geoms = len(geometries)
    num_attrs = len(attributes)
    assert num_geoms == num_attrs, ("Geometry count (%s) and attribute count "
                                    "(%s) do not match.") % (num_geoms, num_attrs)

    for field_name, field_type in fields.iteritems():
        assert field_type in VECTOR_FIELD_TYPES, \
            ("Vector field type for field %s not "
             "reconized: %s") % (field_name, field_type)

    if filename is None:
        if vector_format == 'GeoJSON':
            ext = 'geojson'
        else:
            # assume ESRI Shapefile
            ext = 'shp'

        temp_dir = tempfile.mkdtemp()
        vector_uri = os.path.join(temp_dir, 'vector.%s' % ext)
    else:
        vector_uri = filename

    out_driver = ogr.GetDriverByName(vector_format)
    assert out_driver is not None, ('Vector format "%s" not recognized. '
                                    'Valid formats: %s' ) % (vector_format,
                                                             OGR_DRIVERS)
    out_vector = out_driver.CreateDataSource(vector_uri)

    layer_name = str(os.path.basename(os.path.splitext(vector_uri)[0]))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    out_layer = out_vector.CreateLayer(layer_name, srs=srs)

    for field_name, field_type in fields.iteritems():
        field_defn = ogr.FieldDefn(field_name, VECTOR_FIELD_TYPES[field_type])
        out_layer.CreateField(field_defn)
    layer_defn = out_layer.GetLayerDefn()

    for shapely_feature, fields in zip(geometries, attributes):
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)

        for field_name, field_value in fields.iteritems():
            new_feature.SetField(field_name, field_value)
        out_layer.CreateFeature(new_feature)

    out_layer = None
    out_vector = None

    return vector_uri


class Factory(object):
    """
    The Factory object allows for multiple geospatial files of a consistent
    type (e.g. rasters, vectors) to be created from common parameters.

    For example, a RasterFactory could be instantiated with default pixel
    values and SRS, and then the user would only be required to provide the
    nodata value when calling Factory.new(), as the factory instance would use
    the default values provided on creation.

    Calls to factory.new() allow the user to override any defaults provided by
    the Factory.  User options provided at this stage will always take
    precendent over any settings provided at instantiation.

    Creating a Factory object does not require that the user provide any
    arguments, but if you're doing this, then you might as well just call
    raster() or vector() directly!

    Also, Factory objects will not accept parameters not allowed by their
    creator functions.
    """
    allowed_params = set([])

    def __init__(self, **kwargs):
        self._check_allowed_params(kwargs)

        for user_param, user_value in kwargs.iteritems():
            setattr(self, user_param, user_value)

    def creation_func(self):
        """
        Define the identity function for now.  This method should be
        overridden in the appropriate subclass.
        """
        def foo(x): x
        return foo

    def _check_allowed_params(self, kwargs):
        """
        Check the input dictionary to see if there are any provided
        parameters that are not allowed by the creation function.

        Raise a RuntimeError if so.
        """
        for user_param, user_value in kwargs.iteritems():
            if user_param not in self.allowed_params:
                raise RuntimeError(
                    'Provided parameters "%s" not allowed' % user_param)

    def __getattr__(self, key):
        """
        Get any attribute that has been set in this object.
        """
        return getattr(self, key)

    def __setattr__(self, key, value):
        """
        Overridden attribute setting object method, for allowing raster
        attributes to be set.  The only attributes allowed are those allowed
        in the creation function.

        If an attribute is set that is not allowed in the creation function,
        AttributeError is raised.
        """
        if key not in self.allowed_params:
            error_str = (
                'The attribute %s is not allowed.  Allowed '
                'attrs: %s') % (key, self.allowed_params.items())
            raise AttributeError(error_str)
        setattr(self, key, value)

    def _get_attrs(self):
        """
        Return a dictionary of the parameters that the user provided when
        creating this Factory instance, or that have been set manually
        sometime thereafter.  Attributes are only returned if they are in
        `self.allowed_params`.

        Returns:
            A dictionary mapping {'argname': arg_value}, where 'argname'
            is the parameter name for the target creation function.
        """
        factory_func_attrs = {}
        for attribute in dir(self):
            if attribute in self.allowed_params:
                factory_func_attrs[attribute] = getattr(self, attribute)
        return factory_func_attrs

    def new(self, *args, **kwargs):
        """
        Create a new geospatial object, where the object is defined by the user
        as the return value of the function self.creation_func(), and the
        function takes some (or no) input parameters.

        This object method allows the user to override any defaults that the
        Factory instance was created with (if any overrides are provided),
        and any parameters not overridden will fall back to what the user
        provided as a default parameter on creation of the Factory.

        Any parameter omissions will raise a `TypeError`, as the
        underlying creation function will not fail unless appropriate
        parameters are provided.
        """

        # Inspect the factory's creation function to get the names of the
        # function arguments.  Useful to converting args to kwargs.
        argnames, _, _, _ = inspect.getargspec(self.creation_func())

        out_kwargs = {}
        exclusions = set(['self'])

        # check whether the user added positional or keyword arguments.
        # If so, add them accordingly to the out_kwargs dictionary.
        # Tuple of (args_value to check, iterator yielding (key, value))
        user_inputs = [
            (args, zip(argnames, args)),
            (kwargs, kwargs.iteritems())
        ]

        for user_input, iterator in user_inputs:
            if len(user_input) > 0:
                for argname, argvalue in iterator:
                    if argname not in exclusions:
                        out_kwargs[argname] = argvalue

        # Check the user-defined kwargs to be sure an invalid parameter has not
        # been provided.
        self._check_allowed_params(out_kwargs)

        # Fill in any defaults from the factory that the user didn't provide.
        for base_attr_name, base_attr_value in self._get_attrs().iteritems():
            out_kwargs[base_attr_name] = base_attr_value

        # Run the factory's creation function.
        return self.creation_func()(**out_kwargs)


class RasterFactory(Factory):
    """
    Factory subclass for creating rasters.
    """
    allowed_params = set(['band_matrix', 'origin', 'projection_wkt', 'nodata',
                          'pixel_size', 'datatype', 'format', 'dataset_opts',
                          'filename'])

    def creation_func(self):
        return raster


class VectorFactory(Factory):
    """
    Factory subclass for creating vectors.
    """
    allowed_params = set(['geometries', 'projection', 'fields', 'features',
                          'vector_format', 'filename'])

    def creation_func(self):
        return vector


def visualize(file_list):
    """
    Open the specified geospatial files with the provided application or
    visualization method (qgis is the only supported method at the moment).

    NOTE: This functionality does not appear to work on Darwin systems
    with QGIS installed via homebrew.  In this case, QGIS will open, but
    will not show the files requested.

    Parameters:
        file_list (list): a list of string filepaths to geospatial files.

    Returns:
        Nothing.
    """

    application_call = ['qgis'] + file_list
    LOGGER.debug('Executing %s', application_call)

    subprocess.call(application_call, shell=True)
