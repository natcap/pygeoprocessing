"""Script to help with fill pits tuning and testing."""
import time
import logging

import pygeoprocessing.routing

logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

def main():
    raster_path = r"C:\Users\Rich\Dropbox\big_dems_for_testing_routing\DEM_30m_from_perrine_bad_routing.tif"
    target_filled_dem_raster_path = 'filled_dem2.tif'

    start_time = time.time()
    pygeoprocessing.routing.fill_pits(
        (raster_path, 1), target_filled_dem_raster_path)
    print 'total time: %f' % (time.time() - start_time)


if __name__ == '__main__':
    main()
