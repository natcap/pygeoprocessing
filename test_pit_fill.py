"""Script to help with fill pits tuning and testing."""
import time
import logging

import pygeoprocessing.routing
import pygeoprocessing.testing

logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

def main():
    raster_path = r"C:\Users\Rich\Dropbox\big_dems_for_testing_routing\dem_with_pits.tif"
    #raster_path = r"C:\Users\Rich\Dropbox\big_dems_for_testing_routing\DEM_SRTM_90m_my_fill_v2.tif"
    target_filled_dem_raster_path = 'dem_with_pits_filled.tif'
    target_flow_direction_raster_path = 'flow_direction.tif'
    base_test_dem_path = 'for_testing_dem_with_pits_filled.tif'

    start_time = time.time()
    pygeoprocessing.routing.fill_pits(
        (raster_path, 1), target_filled_dem_raster_path,
        target_flow_direction_raster_path)
    print 'total time: %f' % (time.time() - start_time)

    pygeoprocessing.testing.assert_rasters_equal(
        target_filled_dem_raster_path, base_test_dem_path)

if __name__ == '__main__':
    main()
