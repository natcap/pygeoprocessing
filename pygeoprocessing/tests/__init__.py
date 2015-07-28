"""__init__ module for PyGeoprocessing testing.  Provides some helper
    functions to create Shapefiles and Rasters based on constant input."""

import os
import shutil
import unittest
import tempfile
import hashlib
import logging

import nose.tools
from osgeo import ogr
from osgeo import osr
from osgeo import gdal
import numpy

from pygeoprocessing.testing import scm
from pygeoprocessing.testing.scm import skipIfDataMissing, _SVN_REPO

LOGGER = logging.getLogger('pygeoprocessing.tests')

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

COLOMBIA_SRS = """PROJCS["MAGNA-SIRGAS / Colombia Bogota zone",
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
def COLOMBIA_GEOTRANSFORM(x_len, y_len, origin=_COLOMBIA_ORIGIN):
    # in the case of Colombia, expect that x_len is positive, y_len is
    # negative.
    #assert x_len > 0
    #assert y_len < 0
    return [origin[0], x_len, 0, origin[1], 0, y_len]

@nose.tools.nottest
def test_suite():
    return unittest.TestLoader().discover(os.path.dirname(__file__))

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

def get_md5sum(uri):
    """Get the MD5 hash for a single file.  The file is read in a
        memory-efficient fashion.

        uri - a string uri to the file to be tested.

        Returns a string hash for the file."""

    block_size = 2**20
    file_handler = open(uri, 'rb')
    md5 = hashlib.md5()
    while True:
        data = file_handler.read(block_size)
        if not data:
            break
        md5.update(data)
    return md5.hexdigest()

def record_workspace_md5sums(workspace_uri, logfile_uri,
        workspace_varname='self.workspace_uri'):
    """Recurse through the workspace_uri and for every file in the workspace,
    record the filepath and md5sum to the logfile.

    Args:
        workspace_uri (string): A URI to the workspace to analyze

        logfile_uri (string): A URI to the logfile to which md5sums and paths
        will be recorded.

        workspace_varname (string): The string variable name that should be
        used when setting the ws_uri variable initially.

    Returns:
        Nothing.
    """

    logfile = open(logfile_uri, 'w')
    _write = lambda x: logfile.write(x + '\n')
    _write(
        'ws_uri = lambda x: os.path.join(%s, x)' % workspace_varname)

    ignore_exts = ['.shx']
    for dirpath, _, filenames in os.walk(workspace_uri):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.splitext(filepath)[-1] in ignore_exts:
                continue

            md5sum = get_md5sum(filepath)
            relative_filepath = filepath.replace(workspace_uri + os.sep, '')
            _write(
                "self.assertMD5(ws_uri('%s'), '%s')" %
                (relative_filepath, md5sum))

def vector(
        geometries, srs_wkt, fields=None, features=None,
        vector_format='GeoJSON', filename=None):
    """Create a temp OGR vector on disk.

        geometries - a list of Shapely objects.
        srs_wkt - the WKT spatial reference string.
        fields - a python dictionary mapping string fieldname to a python
            primitive type.  Example: {'ws_id': int}
        features - a python dictionary mapping fieldname to field value.  The
            field value's type must match the type defined in the fields input.
            It is an error if it doesn't.
        vector_format - a python string indicating the OGR format to write.
            GeoJSON is pretty good for most things, but doesn't handle
            multipolygons very well.
        filename=None - None or a python string where the file should be saved.
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
    srs.ImportFromWkt(srs_wkt)
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

def create_raster_from_array(
        pixel_matrix, srs_wkt, geotransform, nodata, filename=None):
    """Create a temp GDAL raster on disk.

        pixel_matrix - a numpy matrix representing pixel values.  The datatype
            of the resulting raster will be determined by the datatype of this
            matrix.
        srs_wkt - the WKT spatial reference string.
        geotransform - a 6-element python list of the geotransform for this
            raster.
        nodata - the nodat value for this raster.
        filename=None - the URI to which this raster should be written. If
            none, a random one will be generated.

    Returns a URI to the resulting GDAL raster on disk."""

    if filename is None:
        _, out_uri = tempfile.mkstemp()
        out_uri += '.tif'
    else:
        out_uri = filename

    datatype = GDAL_TYPES[pixel_matrix.dtype.type]
    n_rows, n_cols = pixel_matrix.shape
    driver = gdal.GetDriverByName('GTiff')
    new_raster = driver.Create(out_uri, n_cols, n_rows, 1, datatype)

    # create some projection information based on the GDAL tutorial at
    # http://www.gdal.org/gdal_tutorial.html
    srs = osr.SpatialReference()
    srs.ImportFromWkt(srs_wkt)
    new_raster.SetProjection(srs.ExportToWkt())
    new_raster.SetGeoTransform(geotransform)

    band = new_raster.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(pixel_matrix, 0, 0)

    band = None
    new_raster = None
    return out_uri


def test(with_data=False):
    """run modulewide tests"""

    if with_data is True:
        _SVN_REPO.get()

    LOGGER.info('running tests on %s', os.path.dirname(__file__))
    suite = unittest.TestLoader().discover(os.path.dirname(__file__))
    unittest.TextTestRunner(verbosity=2).run(suite)
