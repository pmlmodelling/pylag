from libc.math cimport sqrt as sqrt_c

from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef get_barycentric_coords(DTYPE_FLOAT_t x, DTYPE_FLOAT_t y,
        DTYPE_FLOAT_t x_tri[3], DTYPE_FLOAT_t y_tri[3], DTYPE_FLOAT_t phi[3])

cpdef DTYPE_FLOAT_t shephard_interpolation(DTYPE_FLOAT_t x,
        DTYPE_FLOAT_t y, DTYPE_INT_t npts, DTYPE_FLOAT_t[:] xpts, 
        DTYPE_FLOAT_t[:] ypts, DTYPE_FLOAT_t[:] vals)

cpdef inline DTYPE_FLOAT_t get_euclidian_distance(DTYPE_FLOAT_t x1,
        DTYPE_FLOAT_t y1, DTYPE_FLOAT_t x2, DTYPE_FLOAT_t y2):
     return sqrt_c((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

cpdef inline DTYPE_FLOAT_t get_linear_fraction(DTYPE_FLOAT_t var, 
        DTYPE_FLOAT_t var1, DTYPE_FLOAT_t var2):
    return (var - var1) / (var2 - var1)

cpdef inline DTYPE_FLOAT_t interpolate_in_time(DTYPE_FLOAT_t time_fraction,
        DTYPE_FLOAT_t val_last, DTYPE_FLOAT_t val_next):
    return (1.0 - time_fraction) * val_last + time_fraction * val_next

cpdef inline DTYPE_FLOAT_t interpolate_in_sigma(DTYPE_FLOAT_t sigma_fraction,
        DTYPE_FLOAT_t val_last, DTYPE_FLOAT_t val_next):
    return (1.0 - sigma_fraction) * val_last + sigma_fraction * val_next

cdef inline DTYPE_FLOAT_t interpolate_within_element(DTYPE_FLOAT_t var[3], 
        DTYPE_FLOAT_t phi[3]):
    return var[0] + phi[0] * (var[1] - var[0]) + phi[1] * (var[2] - var[0])
