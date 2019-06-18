"""Process a raster calculator plain text expression."""
import sys
import os
import logging

import pygeoprocessing
import taskgraph

WORKSPACE_DIR = 'raster_expression_workspace'
try:
    os.makedirs(WORKSPACE_DIR)
except OSError:
    pass

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def main():
    """Write your expression here."""
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1, 5.0)
    expression_list = [
        {
            'expression': '(load-export)/load',
            'symbol_to_path_band_map': {
                'load': 'https://storage.googleapis.com/ipbes-natcap-ecoshard-data-for-publication/water_2015_n_load_degree_md5_c9cd446d3263fc1f8125a9cc2c388bec.tif',
                'export': 'https://storage.googleapis.com/ipbes-natcap-ecoshard-data-for-publication/water_2015_n_export_degree_md5_74d9ed09f379ea7370c5db88c4826d44.tif',
            },
            'target_nodata': -1,
            'target_raster_path': "NC_nutrient_ssp5.tif",
        }
    ]

    for expression in expression_list:
        # this sets a common target sr, pixel size, and resample method .
        expression.update({
            'churn_dir': WORKSPACE_DIR,
            'target_sr_wkt': None,
            'target_pixel_size': None,
            'resample_method': 'near'
            })
        task_graph.add_task(
            func=pygeoprocessing.evaluate_raster_calculator_expression,
            kwargs=expression,
            task_name='%s = %s' % (
                expression['expression'],
                os.path.basename(expression['target_raster_path'])))


if __name__ == '__main__':
    main()
