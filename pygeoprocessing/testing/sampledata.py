import os
import shutil
import collections
import tempfile
import subprocess
import logging
import inspect

import numpy

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

LOGGER = logging.getLogger('natcap.testing.sampledata')

CoordSystem = collections.namedtuple('CoordSystem', 'wkt origin geotransform')

GDAL_TYPES = {
    numpy.byte: gdal.GDT_Byte,
    numpy.int16: gdal.GDT_Int16,
    numpy.int32: gdal.GDT_Int32,
    numpy.uint16: gdal.GDT_UInt16,
    numpy.uint32: gdal.GDT_UInt32,
    numpy.float32: gdal.GDT_Float32,
    numpy.float64: gdal.GDT_Float64,
}
NUMPY_TYPES = dict((g, n) for (n, g) in GDAL_TYPES.iteritems())

_COLOMBIA_SRS = """PROJCS["MAGNA-SIRGAS / Colombia Bogota zone",
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
    AUTHORITY["EPSG","3116"]]"""

_COLOMBIA_ORIGIN = (444720, 3751320)

def make_geotransform(x_len, y_len, origin):
    return [origin[0], x_len, 0, origin[1], 0, y_len]

def build_srs(srs_wkt, x_len, y_len, origin):
    geotransform = make_geotransform(x_len, y_len, origin)
    return CoordSystem(
        wkt=srs_wkt,
        origin=origin,
        geotransform=geotransform
    )

# Basic Colombia reference 30m coordinate system
SRS_COLOMBIA_30M = build_srs(_COLOMBIA_SRS, 30, -30, _COLOMBIA_ORIGIN)

VECTOR_FIELD_TYPES = {
    str: ogr.OFTString,
    float: ogr.OFTReal,
    int: ogr.OFTInteger,
}

def cleanup(uri):
    """Remove the uri.  If it's a folder, recursively remove its contents."""
    if os.path.isdir(uri):
        shutil.rmtree(uri)
    else:
        os.remove(uri)

def raster(pixels, nodata, reference, filename=None):
    """
    Create a temp GDAL raster on disk.

    Parameters:
        pixels (numpy.ndarray)- a numpy matrix representing pixel values.  The datatype
            of the resulting raster will be determined by the datatype of this
            matrix.
        nodata - the nodata value for this raster.
        reference - a natcap.testing.sampledata.CoordSystem object representing
            the coordinate system.
        filename=None - the URI to which this raster should be written. If
            none, a random one will be generated.

    Returns a URI to the resulting GDAL raster (a GeoTiff) on disk."""
    if filename is None:
        _, out_uri = tempfile.mkstemp()
        out_uri += '.tif'
    else:
        out_uri = filename

    datatype = GDAL_TYPES[pixels.dtype.type]
    n_rows, n_cols = pixels.shape
    driver = gdal.GetDriverByName('GTiff')
    new_raster = driver.Create(out_uri, n_cols, n_rows, 1, datatype)

    # create some projection information based on the GDAL tutorial at
    # http://www.gdal.org/gdal_tutorial.html
    srs = osr.SpatialReference()
    srs.ImportFromWkt(reference.wkt)
    new_raster.SetProjection(srs.ExportToWkt())
    new_raster.SetGeoTransform(reference.geotransform)

    band = new_raster.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(pixels, 0, 0)

    band = None
    new_raster = None
    return out_uri

def vector(
        geometries, reference, fields=None, features=None,
        vector_format='GeoJSON', filename=None):
    """Create a temp OGR vector on disk.

        geometries (list) - a list of Shapely objects.
        reference (CoordSystem)- a CoordSystem reference object
        fields (dict)- a python dictionary mapping string fieldname to a python
            primitive type.  Example: {'ws_id': int}
        features (dict) - a python dictionary mapping fieldname to field value.  The
            field value's type must match the type defined in the fields input.
            It is an error if it doesn't.
        vector_format (string)- a python string indicating the OGR format to write.
            GeoJSON is pretty good for most things, but doesn't handle
            multipolygons very well. 'ESRI Shapefile' is usually a good bet.
        filename=None (None or string)- None or a python string where the file should be saved.
            If None, the vector will be saved to a temporary folder.

    Returns a string URI to the location of the vector on disk."""

    if fields is None:
        fields = {}

    if features is None:
        features = [{} for _ in range(len(geometries))]

    assert len(geometries) == len(features)

    if vector_format == 'GeoJSON':
        ext = 'geojson'
    else:
        # assume ESRI Shapefile
        ext = 'shp'

    if filename is None:
        temp_dir = tempfile.mkdtemp()
        vector_uri = os.path.join(temp_dir, 'shapefile.%s' % ext)
    else:
        vector_uri = filename

    out_driver = ogr.GetDriverByName(vector_format)
    out_vector = out_driver.CreateDataSource(vector_uri)

    layer_name = str(os.path.basename(os.path.splitext(vector_uri)[0]))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(reference.wkt)
    out_layer = out_vector.CreateLayer(layer_name, srs=srs)

    for field_name, field_type in fields.iteritems():
        field_defn = ogr.FieldDefn(field_name, VECTOR_FIELD_TYPES[field_type])
        out_layer.CreateField(field_defn)
    layer_defn = out_layer.GetLayerDefn()

    for shapely_feature, fields in zip(geometries, features):
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)

        for field_name, field_value in fields.iteritems():
            #assert type(field_value) == fields[field_name]
            new_feature.SetField(field_name, field_value)
        out_layer.CreateFeature(new_feature)

    out_layer = None
    out_vector = None

    return vector_uri

class Factory(object):
    """
    The Factory object allows for multiple geospatial files of a consistent
    type (e.g. rasters, vectors) to be created from common parameters.

    For example, a RasterFactory could be instantiated with default pixel values
    and SRS, and then the user would only be required to provide the nodata value
    when calling Factory.new(), as the factory instance would use the default
    values provided on creation.

    Calls to factory.new() allow the user to override any defaults provided by
    the Factory.  User options provided at this stage will always take precendent
    over any settings provided at instantiation.

    Creating a Factory object does not require that the user provide any arguments,
    but if you're doing this, then you might as well just call raster() or vector()
    directly!

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
        return lambda x: x

    def _check_allowed_params(self, kwargs):
        """
        Check the input dictionary to see if there are any provided
        parameters that are not allowed by the creation function.

        Raise a RuntimeError if so.
        """
        for user_param, user_value in kwargs.iteritems():
            if user_param not in self.allowed_params:
                raise RuntimeError('Provided parameters "%s" not allowed' % user_param)

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
            error_str = ('The attribute %s is not allowed.  Allowed '
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
    allowed_params = set(['pixels', 'nodata', 'reference', 'filename'])

    def creation_func(self):
        return raster

class VectorFactory(Factory):
    """
    Factory subclass for creating vectors.
    """
    allowed_params = set(['geometries', 'reference', 'fields', 'features',
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




