from libcpp.deque cimport deque

cdef find_outlets(dem_uri, flow_direction_uri, deque[int] &outlet_deque)
cdef calculate_transport(
    outflow_direction_uri, outflow_weights_uri, deque[int] &sink_cell_deque,
    source_uri, absorption_rate_uri, loss_uri, flux_uri, absorption_mode,
    stream_uri=?, include_source=?)