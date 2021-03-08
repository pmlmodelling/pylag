include "constants.pxi"

from libc.math cimport sqrt as sqrt_c

from libcpp.vector cimport vector

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from pylag.particle cimport Particle
from pylag.spline_cpp_wrapper cimport SplineWrapper

cdef class Interpolator:
    cdef set_points(self, DTYPE_FLOAT_t[:] xp, DTYPE_FLOAT_t[:] fp)

    cdef DTYPE_FLOAT_t get_value(self, Particle* particle) except FLOAT_ERR

    cdef DTYPE_FLOAT_t get_first_derivative(self, Particle* particle) except FLOAT_ERR

cdef class Linear1DInterpolator(Interpolator):
    cdef DTYPE_INT_t _n_elems
    cdef DTYPE_FLOAT_t[:] _xp
    cdef DTYPE_FLOAT_t[:] _fp
    cdef DTYPE_FLOAT_t[:] _fp_prime

    cdef set_points(self, DTYPE_FLOAT_t[:] xp, DTYPE_FLOAT_t[:] fp)

    cdef DTYPE_FLOAT_t get_value(self, Particle* particle) except FLOAT_ERR

    cdef DTYPE_FLOAT_t get_first_derivative(self, Particle* particle) except FLOAT_ERR

cdef class CubicSpline1DInterpolator(Interpolator):
    cdef DTYPE_INT_t _n_elems
    cdef DTYPE_INT_t _first_order
    cdef DTYPE_INT_t _second_order

    cdef SplineWrapper _spline

    cdef set_points(self, DTYPE_FLOAT_t[:] xp, DTYPE_FLOAT_t[:] fp)

    cdef DTYPE_FLOAT_t get_value(self, Particle* particle) except FLOAT_ERR

    cdef DTYPE_FLOAT_t get_first_derivative(self, Particle* particle) except FLOAT_ERR

cdef DTYPE_FLOAT_t get_linear_fraction_safe(const DTYPE_FLOAT_t &var,
        const DTYPE_FLOAT_t &var1, const DTYPE_FLOAT_t &var2) except FLOAT_ERR

cpdef inline DTYPE_FLOAT_t get_linear_fraction(const DTYPE_FLOAT_t &var,
        const DTYPE_FLOAT_t &var1, const DTYPE_FLOAT_t &var2) except FLOAT_ERR:
    return (var - var1) / (var2 - var1)

cpdef inline DTYPE_FLOAT_t linear_interp(const DTYPE_FLOAT_t &fraction,
        const DTYPE_FLOAT_t &val_last, const DTYPE_FLOAT_t &val_next):
    return (1.0 - fraction) * val_last + fraction * val_next

cdef inline DTYPE_FLOAT_t interpolate_within_element(const DTYPE_FLOAT_t var[3],
        const DTYPE_FLOAT_t phi[3]):
    return var[0] * phi[0] +  var[1] * phi[1] + var[2] * phi[2]

cpdef inline DTYPE_FLOAT_t get_euclidian_distance(const DTYPE_FLOAT_t &x1,
        const DTYPE_FLOAT_t &y1, const DTYPE_FLOAT_t &x2, const DTYPE_FLOAT_t &y2):
    return sqrt_c((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

