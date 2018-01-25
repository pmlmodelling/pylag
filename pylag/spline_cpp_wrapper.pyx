include "constants.pxi"

cdef class SplineWrapper:
    def __cinit__(self):
        self.c_spline = spline()

    cdef set_points(self, vector[DTYPE_FLOAT_t] x, vector[DTYPE_FLOAT_t] y):
        self.c_spline.set_points(x,y)

    cdef DTYPE_FLOAT_t call(self, DTYPE_FLOAT_t x) except FLOAT_ERR:
        return self.c_spline(x)
    
    cdef DTYPE_FLOAT_t deriv(self, DTYPE_INT_t order, DTYPE_FLOAT_t x) except FLOAT_ERR:
        return self.c_spline.deriv(order, x)
