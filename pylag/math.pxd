from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef inline DTYPE_FLOAT_t float_min(DTYPE_FLOAT_t a, DTYPE_FLOAT_t b): return a if a <= b else b
cdef inline DTYPE_FLOAT_t float_max(DTYPE_FLOAT_t a, DTYPE_FLOAT_t b): return a if a >= b else b

cdef inline DTYPE_INT_t int_min(DTYPE_INT_t a, DTYPE_INT_t b): return a if a <= b else b
cdef inline DTYPE_INT_t int_max(DTYPE_INT_t a, DTYPE_INT_t b): return a if a >= b else b

cdef inline det(DTYPE_FLOAT_t a[2], DTYPE_FLOAT_t b[2]): return a[0]*b[1] - a[1]*b[0]

cdef inline inner_product(DTYPE_FLOAT_t a[2], DTYPE_FLOAT_t b[2]): return a[0]*b[0] + a[1]*b[1]
