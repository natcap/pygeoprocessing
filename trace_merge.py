"""Tracer code for merge rasters."""
import os

import pygeoprocessing


def main():
    """Entry point."""
    raster_path_list = [
        os.path.join(r"D:\dataplatformdata\dem_globe_ASTER_1arcsecond", x)
        for x in [
            'ASTGTM2_N40W111_dem.tif', 'ASTGTM2_N40W113_dem.tif',
            'ASTGTM2_N41W112_dem.tif']]
    target_path = 'merged.tif'
    pygeoprocessing.merge_rasters(raster_path_list, target_path)

if __name__ == '__main__':
    main()
