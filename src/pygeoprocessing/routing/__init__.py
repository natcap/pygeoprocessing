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
    watersheds_layer = watersheds_vector.CreateLayer(
        'watersheds', fragments_srs, ogr.wkbPolygon)
    for field_defn in watershed_attributes_layer.schema:
        field_type = field_defn.GetType()
        if field_type in (ogr.OFTInteger, ogr.OFTReal):
            field_defn.SetWidth(24)
        watersheds_layer.CreateField(field_defn)

    upstream_fragments = {}
    fragments_in_watershed = collections.defaultdict(set)
    fragment_geometries = {}
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
            feature.GetGeometryRef().ExportToWkb())

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
    compiled_fragments_layer = watersheds_vector.CreateLayer(
        'compiled_fragments', fragments_srs, ogr.wkbMultiPolygon)
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

    # Now, recurse over the remaining geometries and compile the results.
    stack = []
    stack_set = set([])
    for fragment_id in (set(fragment_geometries) - compiled_fragment_ids):
        if fragment_id in compiled_fragment_ids:
            continue

        stack.append(fragment_id)
        stack_set.add(fragment_id)

        for upstream_fragment_id in upstream_fragments[fragment_id]:
            if upstream_fragment_id in compiled_fragment_ids:
                continue

            if upstream_fragment_id == fragment_id:
                continue

            stack.append(upstream_fragment_id)
            stack_set.add(upstream_fragment_id)

        compiled_fragments_layer.StartTransaction()
        while len(stack) > 0:
            stack_fragment_id = stack.pop()
            stack_set.remove(stack_fragment_id)

            # base case: this fragment has already been compiled and has geometry
            # that can be retrieved from ``compiled_fragments_layer``.
            if stack_fragment_id in compiled_fragment_ids:
                continue

            # If the geometry has not been compiled, see if it can be compiled
            # right now.  The geometry can be compiled immediately if all of
            # the upstream fragments have geometries in the compiled layer.
            if upstream_fragments[stack_fragment_id] <= compiled_fragment_ids:
                upstream_geometries = []
                for upstream_fragment_id in upstream_fragments[stack_fragment_id]:
                    compiled_fid = compiled_fragment_fids[upstream_fragment_id]
                    compiled_feature = compiled_fragments_layer.GetFeature(compiled_fid)
                    compiled_geom = compiled_feature.GetGeometryRef()
                    shapely_geometry = shapely.wkb.loads(compiled_geom.ExportToWkb())
                    upstream_geometries.append(shapely_geometry)

                # Buffer by 0 is needed because cascaded_union does not always
                # return valid geometries.
                unioned_geometry = shapely.ops.cascaded_union(upstream_geometries).buffer(0)

                compiled_feature = ogr.Feature(compiled_fragments_layer.GetLayerDefn())
                compiled_feature.SetField('fragment_id', stack_fragment_id)
                compiled_feature.SetGeometry(
                    ogr.CreateGeometryFromWkb(unioned_geometry.wkb))
                compiled_fragments_layer.CreateFeature(compiled_feature)

                compiled_fragment_ids.add(stack_fragment_id)
                compiled_fragment_fids[stack_fragment_id] = compiled_feature.GetFID()
            else:
                # If the geometry cannot be compiled immediately, see which of
                # its upstream fragments still need compilation and push onto
                # the stack as needed.
                if stack_fragment_id not in stack_set:
                    stack.append(stack_fragment_id)
                    stack_set.add(stack_fragment_id)

                for upstream_fragment_id in upstream_fragments[stack_fragment_id]:
                    if upstream_fragment_id in compiled_fragment_ids:
                        continue

                    if upstream_fragment_id not in stack_set:
                        stack.append(upstream_fragment_id)
                        stack_set.add(upstream_fragment_id)

        compiled_fragments_layer.CommitTransaction()

    # Having compiled individual fragments, we can now join together fragments
    # by their watersheds.
    watersheds_layer.StartTransaction()
    for ws_id, member_fragments in fragments_in_watershed.items():
        member_geometries = []
        for fragment_id in member_fragments:
            compiled_fid = compiled_fragment_fids[fragment_id]
            compiled_feature = compiled_fragments_layer.GetFeature(compiled_fid)
            compiled_geom = compiled_feature.GetGeometryRef()
            shapely_geometry = shapely.wkb.loads(compiled_geom.ExportToWkb())
            member_geometries.append(shapely_geometry)

        # Cascaded_union does not always return valid geometries.
        unioned_geometry = shapely.ops.cascaded_union(member_geometries).buffer(0)

        watershed_feature = ogr.Feature(watersheds_layer.GetLayerDefn())
        watershed_feature.SetGeometry(ogr.CreateGeometryFromWkb(unioned_geometry.wkb))
        watersheds_layer.CreateFeature(watershed_feature)
    watersheds_layer.CommitTransaction()































def join_watershed_fragments_new(watershed_fragments_vector,
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
    fragments_layer = fragments_vector.GetLayer('watershed_fragments')
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
    fragment_geometries = collections.defaultdict(list)  # {fragment_id: [geometries in fragment]}
    fragments_in_watershed = collections.defaultdict(list)  # {ws_id: [fragment_ids]}
    fragment_field_values = {}
    LOGGER.info('Loading fragment geometries.')
    for feature in fragments_layer:
        fragment_id = int(feature.GetField('fragment_id'))
        fragment_field_values[fragment_id] = feature.items()
        upstream_fragments_string = feature.GetField('upstream_fragments')
        if upstream_fragments_string:
            upstream_fragments[fragment_id] = [int(f) for f in
                                         upstream_fragments_string.split(',')]
        else:
            # If no upstream fragments, the string will be '', and
            # ''.split(',') turns into [''], which crashes when you cast it to
            # an int.
            upstream_fragments[fragment_id] = []

        # Every fragment will have member watershed IDs, so no need to guard
        # like we do for upstream fragments.
        member_ws_id_string = feature.GetField('member_ws_ids')
        for member_ws_id in member_ws_id_string.split(','):
            fragments_in_watershed[int(member_ws_id)].append(fragment_id)

        shapely_polygon = shapely.wkb.loads(
            feature.GetGeometryRef().ExportToWkb())

        # Buffering by 0 often fixes validity issues and while it isn't
        # perfect, it's been enough for what I've needed it for.
        fragment_geometries[fragment_id].append(shapely_polygon.buffer(0))

    # Create multipolygons from each of the lists of fragment geometries.
    fragment_multipolygons = {}
    for fragment_id, geometry_list in fragment_geometries.items():
        sub_geometry_list = []
        try:
            for geometry in geometry_list:
                sub_geometry_list += [g for g in geometry.geoms]
        except AttributeError:
            # When geometry is a Polygon
            sub_geometry_list = geometry_list

        fragment_multipolygons[fragment_id] = shapely.geometry.MultiPolygon(
            sub_geometry_list)

    # Populate the watershed geometries dict with fragments that are as
    # upstream as you can go.
    watershed_geometries = dict(
        (fragment_id, fragment_multipolygons[fragment_id]) for (fragment_id, upstream_fragments)
        in upstream_fragments.items() if not upstream_fragments)

    encountered_ws_ids = set([])
    def _recurse_watersheds(fragment_id):
        """Find or build geometries for the given ``fragment_id``.

        This is a dynamic programming, recursive approach to determining
        watershed geometries.  If a geometry already exists, we use that.
        If it doesn't, we create the appropriate geometries as needed by
        taking the union of that fragment's upstream fragments with the
        geometry of the fragment itself.

        This function has the side effect of modifying the
        ``watershed_geometries`` dict.

        Parameters:
            fragment_id (int): The integer fragment_id that identifies the watershed
                geometry to build.

        Returns:
            shapely.geometry.Polygon object of the union of the watershed
                fragment identified by fragment_id as well as all watersheds upstream
                of this fragment.

        """

        try:
            geometries = [fragment_multipolygons[fragment_id]]
            encountered_ws_ids.add(fragment_id)
        except KeyError:
            LOGGER.warn('Upstream watershed fragment %s not found. '
                        'Do you have overlapping geometries?', fragment_id)
            return [shapely.geometry.Polygon([])]

        for upstream_fragment_id in upstream_fragments[fragment_id]:
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
    for fragment_id in sorted(
            set(upstream_fragments.keys()).difference(
                set(watershed_geometries.keys())),
            key=lambda ws_id_key: len(upstream_fragments[ws_id_key])):
        current_time = time.time()
        if current_time - last_log_time >= 5.0:
            LOGGER.info("%s fragments of %s processed", n_watersheds_processed, n_features)
            last_log_time = current_time

        # The presence of a fragment_id key in watershed_geometries could be
        # altered during a call to _recurse_watersheds.  This condition
        # ensures that we don't call _recurse_watersheds more than we need to.
        encountered_ws_ids.clear()
        watershed_geometries[fragment_id] = shapely.ops.cascaded_union(
            _recurse_watersheds(fragment_id))
        n_watersheds_processed +=1

    # join fragments by member watersheds.
    fragment_attributes = fragments_vector.GetLayer('watershed_attributes')
    watershed_field_values = {}
    for attr_feature in fragment_attributes:
        outflow_id = attr_feature.GetField('outflow_feature_id')
        watershed_field_values[outflow_id] = attr_feature.items()

    watersheds_layer.CreateFields(fragment_attributes.schema)

    final_watersheds = {}
    watersheds_layer.StartTransaction()
    last_log_time = time.time()
    n_watersheds_processed = 0
    n_watersheds = len(fragments_in_watershed)
    for ws_id, member_fragment_ids in fragments_in_watershed.items():
        current_time = time.time()
        if current_time - last_log_time >= 5.0:
            LOGGER.info("%s fragments of %s processed", n_watersheds_processed,
                        n_watersheds)
            last_log_time = current_time

        member_fragment_polygons = []
        for member_fragment_id in member_fragment_ids:
            member_fragment_polygons.append(watershed_geometries[member_fragment_id])

        watershed_feature = ogr.Feature(watersheds_layer.GetLayerDefn())
        watershed_geom = ogr.CreateGeometryFromWkb(shapely.ops.cascaded_union(member_fragment_polygons).wkb)
        for field_name, field_value in watershed_field_values[ws_id].items():
            watershed_feature.SetField(field_name, field_value)

        watershed_feature.SetGeometry(watershed_geom)
        watersheds_layer.CreateFeature(watershed_feature)
        n_watersheds_processed += 1
    watersheds_layer.CommitTransaction()

    fragments_srs = None
    fragments_feature = None
    fragments_layer = None
    fragments_vector = None

    watersheds_layer = None
    watersheds_vector = None


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
    fragments_layer = fragments_vector.GetLayer('watershed_fragments')
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
        ws_id = feature.GetField('fragment_id')
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
        if shapely_polygon.is_empty:
            print 'WSID', ws_id, 'is empty'
            continue

        try:
            fragment_geometries[ws_id].append(shapely_polygon)
        except KeyError:
            fragment_geometries[ws_id] = [shapely_polygon]

    # Create multipolygons from each of the lists of fragment geometries.
    fragment_multipolygons = {}
    for ws_id, geometry_list in fragment_geometries.items():
        sub_geometry_list = []
        try:
            for geometry in geometry_list:
                sub_geometry_list += [g for g in geometry.geoms]
        except AttributeError:
            # When geometry is a Polygon
            sub_geometry_list = geometry_list

        try:
            fragment_multipolygons[ws_id] = shapely.geometry.MultiPolygon(
                sub_geometry_list)
        except:
            import pprint
            pprint.pprint(sub_geometry_list)
            import pdb; pdb.set_trace()
            raise
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
