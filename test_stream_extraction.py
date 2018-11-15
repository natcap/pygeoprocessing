"""Script to tracer together a stream extraction algorithm."""
import logging

import pygeoprocessing.routing
import taskgraph

logging.basicConfig(level=logging.DEBUG)


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph('stream_testing_dir', -1)
    dem_path = r"D:\Dropbox\big_dems_for_testing_routing\DEM.tif"
    filled_dem = 'filled_dem.tif'
    fill_pits_task = task_graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=((dem_path, 1), filled_dem),
        target_path_list=[filled_dem],
        task_name='fill pits')
    flow_dir_mfd_path = 'flow_dir.tif'
    flow_dir_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_dir_mfd,
        args=((filled_dem, 1), flow_dir_mfd_path),
        dependent_task_list=[fill_pits_task],
        target_path_list=[flow_dir_mfd_path],
        task_name='flow_dir_mfd')
    flow_accum_path = 'flow_accum.tif'
    _ = task_graph.add_task(
        func=pygeoprocessing.routing.flow_accumulation_mfd,
        args=((flow_dir_mfd_path, 1), flow_accum_path),
        dependent_task_list=[flow_dir_task],
        target_path_list=[flow_accum_path],
        task_name='flow_accumulation_mfd')
    task_graph.close()
    task_graph.join()

    flow_threshold = 1000
    target_stream_raster_path = 'streams_non.tif'
    pygeoprocessing.routing.extract_streams_mfd(
        (flow_accum_path, 1), flow_threshold,
        target_stream_raster_path,
        flow_dir_mfd_path_band=(flow_dir_mfd_path, 1),
        remove_stream_fragments=False)
    return
    target_stream_connect_raster_path = 'streams_connect.tif'
    pygeoprocessing.routing.extract_streams_mfd(
        (flow_accum_path, 1), flow_threshold,
        target_stream_connect_raster_path,
        flow_dir_mfd_path_band=(flow_dir_mfd_path, 1),
        remove_stream_fragments=True)


if __name__ == '__main__':
    main()
