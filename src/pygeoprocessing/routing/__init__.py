from pygeoprocessing.routing.routing import *
import time

from pygeoprocessing.routing.watershed import *

import logging
from osgeo import gdal
from osgeo import ogr
import shapely.wkb
import shapely.geometry

LOGGER = logging.getLogger(__name__)


def join_watershed_fragments(watershed_fragments_vector,
                             target_watersheds_vector):
    """Join watershed fragments by their IDs.

    This function takes a watershed fragments vector and creates a new vector
    where all geometries represent the full watershed represented by any nested
    watershed fragments contained within the watershed fragments vector.

    Parameters:
        watershed_fragments_vector (string): A path to a vector on disk.  This
            vector must have at least two fields: 'ws_id' (int) and
            'upstream_fragments' (string).  The 'upstream_fragments' field's
            values must be formatted as a list of comma-separated integers
            (example: ``1,2,3,4``), where each integer matches the ``ws_id``
            field of a polygon in this vector.  This field should only contain
            the ``ws_id``s of watershed fragments that directly touch the
            current fragment, as this function will recurse through the
            fragments to determine the correct nested geometry.  If a fragment
            has no nested watersheds, this field's value will have an empty
            string.  The ``ws_id`` field represents a unique integer ID for the
            watershed fragment.
        target_watersheds_vector (string): A path to a vector on disk.  This
            vector will be written as a GeoPackage.

    Returns:
        ``None``

    """
    fragments_vector = ogr.Open(watershed_fragments_vector)
    fragments_layer = fragments_vector.GetLayer()
    fragments_layer_name = fragments_layer.GetName()
    fragments_srs = fragments_layer.GetSpatialRef()

    driver = gdal.GetDriverByName('GPKG')
    watersheds_vector = driver.Create(target_watersheds_vector, 0, 0, 0,
                                      gdal.GDT_Unknown)
    watersheds_layer = watersheds_vector.CreateLayer(
        'watersheds', fragments_srs, ogr.wkbPolygon)
    watersheds_layer_defn = watersheds_layer.GetLayerDefn()

    for field_defn in fragments_layer.schema:
        field_type = field_defn.GetType()
        if field_type in (ogr.OFTInteger, ogr.OFTReal):
            field_defn.SetWidth(24)
        watersheds_layer.CreateField(field_defn)

    upstream_fragments = {}
    fragment_geometries = {}
    fragment_field_values = {}
    LOGGER.info('Loading fragment geometries.')
    for feature in fragments_layer:
        ws_id = feature.GetField('ws_id')
        fragment_field_values[ws_id] = feature.items()
        upstream_fragments_string = feature.GetField('upstream_fragments')
        if upstream_fragments_string:
            upstream_fragments[ws_id] = [int(f) for f in
                                         upstream_fragments_string.split(',')]
        else:
            # If no upstream fragments, the string will be '', and
            # ''.split(',') turns into [''], which crashes when you cast it to
            # an int.
            upstream_fragments[ws_id] = []
        shapely_polygon = shapely.wkb.loads(
            feature.GetGeometryRef().ExportToWkb())
        try:
            fragment_geometries[ws_id].append(shapely_polygon)
        except KeyError:
            fragment_geometries[ws_id] = [shapely_polygon]

    # Create multipolygons from each of the lists of fragment geometries.
    fragment_multipolygons = {}
    for ws_id, geometry_list in fragment_geometries.items():
        fragment_multipolygons[ws_id] = shapely.geometry.MultiPolygon(
            geometry_list)
    #del fragment_geometries  # don't need this dict any longer.

    # Populate the watershed geometries dict with fragments that are as
    # upstream as you can go.
    watershed_geometries = dict(
        (ws_id, fragment_multipolygons[ws_id]) for (ws_id, upstream_fragments)
        in upstream_fragments.items() if not upstream_fragments)

    encountered_ws_ids = set([])
    def _recurse_watersheds(ws_id):
        """Find or build geometries for the given ``ws_id``.

        This is a dynamic programming, recursive approach to determining
        watershed geometries.  If a geometry already exists, we use that.
        If it doesn't, we create the appropriate geometries as needed by
        taking the union of that fragment's upstream fragments with the
        geometry of the fragment itself.

        This function has the side effect of modifying the
        ``watershed_geometries`` dict.

        Parameters:
            ws_id (int): The integer ws_id that identifies the watershed
                geometry to build.

        Returns:
            shapely.geometry.Polygon object of the union of the watershed
                fragment identified by ws_id as well as all watersheds upstream
                of this fragment.

        """

        try:
            geometries = [fragment_multipolygons[ws_id]]
            encountered_ws_ids.add(ws_id)
        except KeyError:
            LOGGER.warn('Upstream watershed fragment %s not found. '
                        'Do you have overlapping geometries?', ws_id)
            return [shapely.geometry.Polygon([])]

        for upstream_fragment_id in upstream_fragments[ws_id]:
            # If we've already encountered this upstream fragment on this
            # run through the recursion, skip it.
            if upstream_fragment_id in encountered_ws_ids:
                continue

            encountered_ws_ids.add(upstream_fragment_id)
            if upstream_fragment_id not in watershed_geometries:
                watershed_geometries[upstream_fragment_id] = (
                    shapely.ops.cascaded_union(
                        _recurse_watersheds(upstream_fragment_id)))
            geometries.append(watershed_geometries[upstream_fragment_id])
        return geometries

    # Iterate over the ws_ids that have upstream geometries and create the
    # watershed geometries.
    # This iteration must happen in sorted order because real-world watersheds
    # sometimes have thousands of nested watersheds, which will exhaust
    # python's max recursion depth here if we're not careful.
    # TODO: avoid max recursion depth here.
    last_log_time = time.time()
    n_watersheds_processed = len(watershed_geometries)
    n_features = len(upstream_fragments)
    LOGGER.info('Joining watershed geometries')
    for ws_id in sorted(
            set(upstream_fragments.keys()).difference(
                set(watershed_geometries.keys())),
            key=lambda ws_id_key: len(upstream_fragments[ws_id_key])):
        current_time = time.time()
        if current_time - last_log_time >= 5.0:
            LOGGER.info("%s fragments of %s processed", n_watersheds_processed, n_features)
            last_log_time = current_time

        # The presence of a ws_id key in watershed_geometries could be
        # altered during a call to _recurse_watersheds.  This condition
        # ensures that we don't call _recurse_watersheds more than we need to.
        encountered_ws_ids.clear()
        watershed_geometries[ws_id] = shapely.ops.cascaded_union(
            _recurse_watersheds(ws_id))
        n_watersheds_processed +=1 

    # Copy fields from the fragments vector and set the geometries to the
    # newly-created, unioned geometries.
    LOGGER.info('Copying field values to the target vector')
    watersheds_layer.StartTransaction()
    for ws_id, watershed_geometry in sorted(watershed_geometries.items(),
                                            key=lambda x: x[0]):
        # Creating the feature here works properly, unlike
        # fragments_feature.Clone(), which raised SQL errors.
        watershed_feature = ogr.Feature(watersheds_layer_defn)
        try:
            for field_name, field_value in fragment_field_values[ws_id].items():
                watershed_feature.SetField(field_name, field_value)
        except KeyError:
            LOGGER.info('Skipping ws_id %s', ws_id)

        watershed_feature.SetGeometry(ogr.CreateGeometryFromWkb(
            shapely.wkb.dumps(watershed_geometry)))
        watersheds_layer.CreateFeature(watershed_feature)
    watersheds_layer.CommitTransaction()

    fragments_srs = None
    fragments_feature = None
    fragments_layer = None
    fragments_vector = None

    watersheds_layer = None
    watersheds_vector = None
