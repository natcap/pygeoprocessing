import collections
import logging
import os
import contextlib
import time
import cProfile
import StringIO
import pstats
import glob

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

from osgeo import gdal

import pygeoprocessing
import pygeoprocessing.testing
import pygeoprocessing.routing


@contextlib.contextmanager
def time_it(message):
    LOGGER.info(message)
    start_time = time.time()
    yield
    LOGGER.info('Finished %s in %ss', message.lower(), time.time() - start_time)


@contextlib.contextmanager
def profile(filename=None):
    profiler = cProfile.Profile()
    profiler.enable()
    yield
    profiler.disable()

    if not filename:
        stream = StringIO.StringIO()
    else:
        stream = open(filename, 'w')

    ps = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    ps.print_stats()

    if not filename:
        print stream.getvalue()
    else:
        stream.close()

WILLAMETTE_ROADS = (
    'workspace-willamette-roads',
    r'C:\Users\jdouglass\workspace\natcap\invest\data\invest-data\Base_Data\Terrestrial\roads.shp',
    #r'roadsegment.shp',
    #r'roadsegments.shp',
    r'C:\Users\jdouglass\workspace\natcap\invest\data\invest-data\Base_Data\Freshwater\dem\hdr.adf')


STACIES_SUBSET = (
    'workspace-stacie-subset',
    r'C:\Users\jdouglass\Downloads\waterbodies_DEM_Montana_sample\waterbodies_DEM_Montana_sample\waterbodies_NHD_3fc.shp',
    r'C:\Users\jdouglass\Downloads\waterbodies_DEM_Montana_sample\waterbodies_DEM_Montana_sample\dem_reprojected.tif')

ALL_OF_MONTANA = (
    'workspace-montana',
    r'C:\Users\jdouglass\workspace\jdouglass\snippets\in-development-watershed-delination-sample\NDD_Montana_waterbodies1.gpkg',
    r'C:\Users\jdouglass\workspace\jdouglass\snippets\in-development-watershed-delination-sample\merged_montana.tif')


def compare_scratch_to_fragments(vector, rasterized_path, raster2):
    pygeoprocessing.new_raster_from_base(
        raster2, rasterized_path, gdal.GDT_UInt32, [2**32-1], fill_value_list=[0])

    pygeoprocessing.rasterize(
        vector_path=vector,
        target_raster_path=rasterized_path,
        option_list=['ATTRIBUTE=fragment_id'])

    pygeoprocessing.testing.assert_rasters_equal(rasterized_path, raster2)


def find_fragments_split_between_features(vector_path):
    vector = gdal.OpenEx(vector_path)
    layer = vector.GetLayer()
    features = collections.defaultdict(set)
    for feature in layer:
        features[feature.GetField('fragment_id')].add(feature.GetFID())

    return [fragment_id for (fragment_id, fids) in features.items() if len(fids)]


def doit():
    handler = logging.FileHandler('latest-logfile.txt', 'w', encoding='UTF-8')
    formatter = logging.Formatter(
            '%(asctime)s %(name)-18s %(levelname)-8s %(message)s',
            '%m/%d/%Y %H:%M:%S ')
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.NOTSET)
    handler.setFormatter(formatter)
    handler.setLevel(logging.NOTSET)

    # Willamette roads, from the sample data
    workspace, lakes, dem = WILLAMETTE_ROADS

    # Stacie's subset of lakes from NHD.
    #workspace, lakes, dem = STACIES_SUBSET

    # Biggest Montana dataset I have at the moment.
    #workspace, lakes, dem = ALL_OF_MONTANA

    #workspace = 'D:\\new-watershed-delineation'
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    filled_dem = os.path.join(workspace, 'filled_dem.tif')
    flow_dir = os.path.join(workspace, 'flow_dir_d8.tif')
    fragments_path = os.path.join(workspace, 'fragments.gpkg')
    joined_fragments = os.path.join(workspace, 'joined_fragments.gpkg')
    trivial_watersheds = os.path.join(workspace, 'watersheds_trivial.gpkg')

    #with time_it('Filling pits'):
    #    pygeoprocessing.routing.fill_pits((dem, 1), filled_dem)

    #with time_it('D8 flow direction'):
    #    pygeoprocessing.routing.flow_dir_d8((filled_dem, 1), flow_dir, workspace)

    #with time_it('delineating watersheds'):
    #    pygeoprocessing.routing.delineate_watersheds_d8(
    #            (flow_dir, 1), lakes, fragments_path, workspace)

    #with time_it('delineating watersheds trivial'):
    #    pygeoprocessing.routing.delineate_watersheds_trivial_d8(
    #            (flow_dir, 1), lakes, trivial_watersheds)

    #with time_it('delineating watersheds'):
    #    pygeoprocessing.routing.delineate_watersheds_d8(
    #            (flow_dir, 1), lakes, fragments_path, working_dir=workspace)
    #compare_scratch_to_fragments(
    #    fragments_path,
    #    os.path.join(workspace, 'rasterized_fragments.tif'),
    #    os.path.join(sorted(glob.glob(os.path.join(workspace, 'watershed_delineation*')))[-1], 'scratch_raster.tif'))

    with time_it('joining watershed fragments'):
        pygeoprocessing.routing.join_watershed_fragments(
                fragments_path, joined_fragments)

    handler.close()
    root_logger.removeHandler(handler)

if __name__ == '__main__':
    doit()
