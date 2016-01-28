from libc.math cimport sqrt as sqrt_c

from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cpdef get_barycentric_coords(DTYPE_FLOAT_t x, DTYPE_FLOAT_t y,
        DTYPE_FLOAT_t[:] x_tri, DTYPE_FLOAT_t[:] y_tri, DTYPE_FLOAT_t[:] phi)

cpdef DTYPE_FLOAT_t shephard_interpolation(DTYPE_FLOAT_t x,
        DTYPE_FLOAT_t y, DTYPE_INT_t npts, DTYPE_FLOAT_t[:] xpts, 
        DTYPE_FLOAT_t[:] ypts, DTYPE_FLOAT_t[:] vals)

cpdef inline DTYPE_FLOAT_t get_euclidian_distance(DTYPE_FLOAT_t x1,
        DTYPE_FLOAT_t y1, DTYPE_FLOAT_t x2, DTYPE_FLOAT_t y2):
     return sqrt_c((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

cpdef inline DTYPE_FLOAT_t get_time_fraction(DTYPE_FLOAT_t t, 
        DTYPE_FLOAT_t t1, DTYPE_FLOAT_t t2):
    return (t - t1) / (t2 - t1)

cpdef inline DTYPE_FLOAT_t interpolate_in_time(DTYPE_FLOAT_t time_fraction,
        DTYPE_FLOAT_t val_last, DTYPE_FLOAT_t val_next):
    return (1.0 - time_fraction) * val_last + time_fraction * val_next

cpdef inline DTYPE_FLOAT_t interpolate_within_element(DTYPE_FLOAT_t[:] var, 
        DTYPE_FLOAT_t[:] phi):
    return var[0] + phi[0] * (var[1] - var[0]) + phi[1] * (var[2] - var[0])

cdef inline DTYPE_FLOAT_t interpolate_sigma_within_element(DTYPE_FLOAT_t var[3], 
        DTYPE_FLOAT_t[:] phi):
    return var[0] + phi[0] * (var[1] - var[0]) + phi[1] * (var[2] - var[0])

