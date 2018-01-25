include "constants.pxi"

cdef class SplineWrapper:
    """ Cython wrapper for the class spline, which is implemented in spline.cpp
    
    The method for wrapping spline follows the method laid out in Cython's
    documentation (see https://cython.readthedocs.io/en/latest/).
    """

    def __cinit__(self):
        self.c_spline = spline()

    cdef set_points(self, vector[DTYPE_FLOAT_t] x, vector[DTYPE_FLOAT_t] y):
        self.c_spline.set_points(x,y)

    cdef DTYPE_FLOAT_t call(self, DTYPE_FLOAT_t x) except FLOAT_ERR:
        return self.c_spline(x)
    
    cdef DTYPE_FLOAT_t deriv(self, DTYPE_INT_t order, DTYPE_FLOAT_t x) except FLOAT_ERR:
        return self.c_spline.deriv(order, x)
