========
Examples
========

Vectorize Datasets
------------------

The `vectorize_datasets` function is a programmable raster algebra routine.

Demo::

	import numpy
	 
	import gdal
	from pygeoprocessing import geoprocessing
	 
	 
	def main():
	    """main entry point"""
	 
	    dataset_uri_list = [
	        r"C:\path\to\landuse_90",
	        r"C:\path\to\precip",
	        r"C:\path\to\erodibility",
	        ]
	 
	    pixel_size_out = geoprocessing.get_cell_size_from_uri(
	    	dataset_uri_list[0])
	 
	    dataset_out_uri = r"C:\path\to\masked.tif"
	    datatype_out = gdal.GDT_Float32
	    nodata_out = 99
	    bounding_box_mode = 'intersection'
	    vectorize_op = False
	    aoi_uri = r"C:\path\to\subwatersheds.shp"
	 
	 
	    def mask_wet_regions(precip):
	        return precip > 2000
	 
	    wet_regions_uri = r"C:\path\to\wet_regions.tif"
	 
	    geoprocessing.vectorize_datasets(
	        [dataset_uri_list[1]], mask_wet_regions, wet_regions_uri, datatype_out,
	        nodata_out, pixel_size_out, bounding_box_mode,
	        vectorize_op=vectorize_op)
	 
	 
	    valid_landcovers = numpy.array([58, 59, 60, 61])
	    precip_threshold = 2000
	    erodibility_nodata = geoprocessing.get_nodata_from_uri(
	    	dataset_uri_list[2])
	    def dataset_pixel_op(landuse, precip, erodibility):
	        """mask forest pixels that have lots of precip"""
	        precip_mask = precip > precip_threshold
	        landuse_mask = numpy.in1d(landuse, valid_landcovers).reshape(
	            landuse.shape)
	        erodibility_mask = erodibility != erodibility_nodata
	        return numpy.where(
	            precip_mask & landuse_mask & erodibility_mask,
	            precip * erodibility,
	            numpy.where(erodibility_mask, erodibility, nodata_out))
	 
	    geoprocessing.vectorize_datasets(
	        dataset_uri_list, dataset_pixel_op, dataset_out_uri, datatype_out,
	        nodata_out, pixel_size_out, bounding_box_mode,
	        vectorize_op=vectorize_op, aoi_uri=aoi_uri)
	 
	 
	if __name__ == '__main__':
	    main()


Routing
-------

Examples of functions in the hydrological routing library

Demo::

	from pygeoprocessing import routing
	 
	def main():
	    """main entry point"""
	 
	    #dem_uri = r"C:\path\to\dem"
	    dem_uri = r"C:\path\to\srtm_1sec_uga.tif"
	    flow_direction_uri = (
	        r"C:\path\to\flow_dir.tif")
	    flow_accumulation_uri = (
	        r"C:\path\to\flow_accumulation.tif")
	    flow_threshold = 1000
	    stream_uri = r"C:\path\to\stream.tif"
	    distance_uri = r"C:\path\to\distance.tif"
	 
	    routing.flow_direction_d_inf(dem_uri, flow_direction_uri)
	    routing.flow_accumulation(
	        flow_direction_uri, dem_uri, flow_accumulation_uri)
	    routing.stream_threshold(
	        flow_accumulation_uri, flow_threshold, stream_uri)
	    routing.distance_to_stream(
	        flow_direction_uri, stream_uri, distance_uri)
	 
	if __name__ == '__main__':
	    main()

