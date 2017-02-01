from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef inline DTYPE_FLOAT_t float_min(DTYPE_FLOAT_t a, DTYPE_FLOAT_t b): return a if a <= b else b
cdef inline DTYPE_FLOAT_t float_max(DTYPE_FLOAT_t a, DTYPE_FLOAT_t b): return a if a >= b else b

cdef inline DTYPE_INT_t int_min(DTYPE_INT_t a, DTYPE_INT_t b): return a if a <= b else b
cdef inline DTYPE_INT_t int_max(DTYPE_INT_t a, DTYPE_INT_t b): return a if a >= b else b

cdef inline det(DTYPE_FLOAT_t a[2], DTYPE_FLOAT_t b[2]): return a[0]*b[1] - a[1]*b[0]

cdef inline inner_product(DTYPE_FLOAT_t a[2], DTYPE_FLOAT_t b[2]): return a[0]*b[0] + a[1]*b[1]

cdef get_intersection_point(DTYPE_FLOAT_t x1[2], DTYPE_FLOAT_t x2[2],
        DTYPE_FLOAT_t x3[2], DTYPE_FLOAT_t x4[2], DTYPE_FLOAT_t xi[2])

cdef inline DTYPE_FLOAT_t sigma_to_cartesian_coords(DTYPE_FLOAT_t sigma, DTYPE_FLOAT_t h,
        DTYPE_FLOAT_t zeta):
    return zeta + sigma * (h + zeta)

cdef inline DTYPE_FLOAT_t cartesian_to_sigma_coords(DTYPE_FLOAT_t z, DTYPE_FLOAT_t h,
        DTYPE_FLOAT_t zeta):
    return (z - zeta) / (h + zeta)