import time
import collections

from pygeoprocessing.routing.routing import *
from pygeoprocessing.routing.watershed import *

import logging
from osgeo import gdal
from osgeo import ogr
import shapely.wkb
import shapely.geometry

LOGGER = logging.getLogger(__name__)


def join_watershed_fragments_stack(watershed_fragments_vector,
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
    # Components:
    #  * Create the target watersheds vector with layers for:
    #       * Flow-contiguous fragments
    #       * The output watersheds
    #  * Open the watershed fragments vector and get the fragments layer.
    #  * Iterate through the fragments
    #       * Determine which features belong to which watersheds
    #       * Determine the upstream fragments and be able to identify geometries correctly
    #       * Notes:
    #           * Fragments are all assumed to be valid polygons or multipolygons
    #           * It's an error for any geometry to be invalid.
    #           * It's an error for any two fragments to have the same fragment ID
    #  * Any fragments with no upstream geometries are considered 'compiled', meaning
    #    that they do not need to have any further geometric operations performed on
    #    them (aside from unioning by watershed).
    #  * Use stack-recursion to go through the fragments to find all of the fragments
    #    that nest and buffer(0) them together, saving the results to an intermediate
    #    data structure.
    #  * Loop through all watersheds to buffer(0) their component fragments and write
    #    out the watershed geometries to the final output layer along with all of the
    #    field values.
    #  * Delete the flow-contuguous fragment layer from the target watersheds vector.
    fragments_vector = ogr.Open(watershed_fragments_vector)
    fragments_layer = fragments_vector.GetLayer('watershed_fragments')
    watershed_attributes_layer = fragments_vector.GetLayer('watershed_attributes')
    fragments_srs = fragments_layer.GetSpatialRef()

    driver = gdal.GetDriverByName('GPKG')
    watersheds_vector = driver.Create(target_watersheds_vector, 0, 0, 0,
                                      gdal.GDT_Unknown)
    # ogr.wkbUnknown allows both polygons and multipolygons
    watersheds_layer = watersheds_vector.CreateLayer(
        'watersheds', fragments_srs, ogr.wkbUnknown)
    for field_defn in watershed_attributes_layer.schema:
        # ``outflow_feature_id`` not needed in the final watersheds.
        if field_defn.GetName() == 'outflow_feature_id':
            continue

        field_type = field_defn.GetType()
        if field_type in (ogr.OFTInteger, ogr.OFTReal):
            field_defn.SetWidth(24)
        watersheds_layer.CreateField(field_defn)

    upstream_fragments = {}
    fragments_in_watershed = collections.defaultdict(set)
    fragment_geometries = {}
    LOGGER.info('Loading fragments')
    for feature in fragments_layer:
        fragment_id = int(feature.GetField('fragment_id'))
        upstream_fragments_string = feature.GetField('upstream_fragments')
        if upstream_fragments_string:
            feature_upstream_fragments = set([
                int(f) for f in upstream_fragments_string.split(',')
                if int(f) != fragment_id])
        else:
            feature_upstream_fragments = set([])

        # Every fragment will have member watershed IDs.
        for member_ws_id in feature.GetField('member_ws_ids').split(','):
            fragments_in_watershed[int(member_ws_id)].add(fragment_id)

        shapely_geometry = shapely.wkb.loads(
            feature.GetGeometryRef().ExportToWkb()).buffer(0)

        # If the two geometries (fragment_geometries[fragment_id] and shapely_geometry)
        # do not intersect, ``union`` will return a MultiPolygon.  If they do intersect,
        # ``union`` will return a Polygon.
        if fragment_id in fragment_geometries:
            shapely_geometry = shapely_geometry.union(fragment_geometries[fragment_id])
            feature_upstream_fragments = feature_upstream_fragments.union(
                upstream_fragments[fragment_id])

        fragment_geometries[fragment_id] = shapely_geometry
        upstream_fragments[fragment_id] = feature_upstream_fragments

    # Write compiled fragments to a layer for compiled fragments
    # ogr.wkbUnknown supports both polygons and multipolygons.
    compiled_fragments_layer = watersheds_vector.CreateLayer(
        'compiled_fragments', fragments_srs, ogr.wkbUnknown)
    compiled_fragments_layer.CreateField(
        ogr.FieldDefn('fragment_id', ogr.OFTInteger))
    compiled_fragments_layer.StartTransaction()
    compiled_fragment_ids = set([])  # TODO: just use compiled_fragment_fids keys
    compiled_fragment_fids = {}  # {fragment_id: FID}
    for fragment_id in fragment_geometries:
        if len(upstream_fragments[fragment_id]) > 0:
            continue

        new_feature = ogr.Feature(compiled_fragments_layer.GetLayerDefn())
        new_feature.SetField('fragment_id', fragment_id)
        new_feature.SetGeometry(
            ogr.CreateGeometryFromWkb(fragment_geometries[fragment_id].wkb))
        compiled_fragments_layer.CreateFeature(new_feature)
        compiled_fragment_fids[fragment_id] = new_feature.GetFID()
        compiled_fragment_ids.add(fragment_id)
    compiled_fragments_layer.CommitTransaction()

    # Double-check that the committed FIDs match what we expect them to be.
    for feature in compiled_fragments_layer:
        fragment_id = feature.GetField('fragment_id')
        fragment_fid = feature.GetFID()
        assert compiled_fragment_fids[fragment_id] == fragment_fid

    def _load_fragment(fragment_id):
        compiled_fid = compiled_fragment_fids[fragment_id]
        compiled_feature = compiled_fragments_layer.GetFeature(compiled_fid)
        compiled_geom = compiled_feature.GetGeometryRef()
        shapely_geometry = shapely.wkb.loads(
            compiled_geom.ExportToWkb()).buffer(0)
        return shapely_geometry

    def _write_fragment(fragment_id, geometry_list):
        unioned_geometry = shapely.ops.cascaded_union(geometry_list).buffer(0)

        compiled_feature = ogr.Feature(compiled_fragments_layer.GetLayerDefn())
        compiled_feature.SetField('fragment_id', fragment_id)
        compiled_feature.SetGeometry(
            ogr.CreateGeometryFromWkb(unioned_geometry.wkb))
        compiled_fragments_layer.CreateFeature(compiled_feature)
        compiled_fragment_fids[fragment_id] = compiled_feature.GetFID()
        compiled_fragment_ids.add(fragment_id)

    # Now, iterate over all of the fragments and compile them into
    # flow-contiguous fragments by recursing upstream.
    LOGGER.info('Compiling flow-contiguous fragments')
    last_time = time.time()
    for fragment_id in fragment_geometries:
        # If the fragment has already been compiled, no need to recompile.
        if fragment_id in compiled_fragment_ids:
            continue

        fragment_ids_encountered = set([fragment_id])
        fragment_geometry_list = [fragment_geometries[fragment_id]]

        stack = [fragment_id]
        stack_set = set([fragment_id])
        compiled_fragments_layer.StartTransaction()
        while len(stack) > 0:
            if time.time() - last_time > 5.0:
                LOGGER.info('%s complete so far',
                            compiled_fragments_layer.GetFeatureCount())
                last_time = time.time()

            stack_fragment_id = stack.pop()
            stack_set.remove(stack_fragment_id)
            fragment_ids_encountered.add(stack_fragment_id)

            if stack_fragment_id in compiled_fragment_fids:
                fragment_geometry_list.append(_load_fragment(stack_fragment_id))
            else:
                fragment_geometry_list.append(
                    fragment_geometries[stack_fragment_id])

                for upstream_fragment_id in upstream_fragments[stack_fragment_id]:
                    if upstream_fragment_id in fragment_ids_encountered:
                        continue

                    if upstream_fragment_id in stack_set:
                        continue

                    fragment_geometry_list.append(
                        fragment_geometries[upstream_fragment_id])
                    stack.append(upstream_fragment_id)
                    stack_set.add(upstream_fragment_id)

        _write_fragment(fragment_id, fragment_geometry_list)
        compiled_fragments_layer.CommitTransaction()

    # Having compiled individual fragments, we can now join together fragments
    # by their watersheds.
    # Before we do this, we need to load up the attributes so we can index them by
    # their watershed ID (represented by the ``outflow_feature_id`` column)
    LOGGER.info('Joining fragments into watersheds')
    attributes_by_ws_id = {}
    for feature in watershed_attributes_layer:
        ws_id = feature.GetField('outflow_feature_id')
        attributes_by_ws_id[ws_id] = feature.items()

    watersheds_layer.StartTransaction()
    for ws_id, member_fragments in fragments_in_watershed.items():
        member_geometries = []
        for fragment_id in member_fragments:
            member_geometries.append(_load_fragment(fragment_id))

        # Cascaded_union does not always return valid geometries.
        unioned_geometry = shapely.ops.cascaded_union(member_geometries).buffer(0)

        watershed_feature = ogr.Feature(watersheds_layer.GetLayerDefn())
        for field_name, field_value in attributes_by_ws_id[ws_id].items():
            if field_name == 'outflow_feature_id':
                continue

            watershed_feature.SetField(field_name, field_value)

        watershed_feature.SetGeometry(ogr.CreateGeometryFromWkb(unioned_geometry.wkb))
        watersheds_layer.CreateFeature(watershed_feature)
    watersheds_layer.CommitTransaction()

    # OK to remove the compiled layer.
    compiled_fragments_layer = None
    watersheds_vector.DeleteLayer('compiled_fragments')
