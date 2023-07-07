"""
Interpolators and interpolation helpers

Note
----
interpolation is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

include "constants.pxi"

import numpy as np

# Cython imports
cimport numpy as np
np.import_array()

from libcpp.vector cimport vector

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t


cdef class Interpolator:
    """ An abstract base class for interpolation schemes
    
    The following method(s) should be implemented in the derived class:
    
    * :meth: `set_points`
    * :meth: `get_value`
    * :meth: `get_first_derivative`
    """

    def __cinit__(self, n_elems):
        pass
    
    cdef set_points(self, DTYPE_FLOAT_t[:] xp, DTYPE_FLOAT_t[:] fp):
        raise NotImplementedError

    cdef DTYPE_FLOAT_t get_value(self, Particle* particle) except FLOAT_ERR:
        raise NotImplementedError

    cdef DTYPE_FLOAT_t get_first_derivative(self, Particle* particle) except FLOAT_ERR:
        raise NotImplementedError

cdef class Linear1DInterpolator:
    """ Linear 1D Interpolator
    
    This class contains methods that perform linear 1D interpolation.
    
    Parameters
    ----------
    n_elems : int
        The number of points at which data are defined. This is used to
        create data arrays of the correct size, which are later used for
        the interpolation.
    """

    def __cinit__(self, n_elems):
        self._n_elems = n_elems
        self._xp = np.empty((self._n_elems), dtype=DTYPE_FLOAT)
        self._fp = np.empty((self._n_elems), dtype=DTYPE_FLOAT)
        self._fp_prime = np.empty((self._n_elems), dtype=DTYPE_FLOAT)
    
    cdef set_points(self, DTYPE_FLOAT_t[:] xp, DTYPE_FLOAT_t[:] fp):
        """ Set coordinate and function value data

        The two memory view objects should have the same length, equal to 
        `n_elems'.

        Parameters
        ----------
        xp : 1D MemoryView
            A MemoryView giving the location coordinates of points at which
            the interpolating function is defined.
        
        fp : 1D MemoryView
            A MemoryView giving the function value at each of the location
            coordinates.
        """
        cdef DTYPE_INT_t i
        
        # Read in data
        for i in xrange(self._n_elems):
            self._xp[i] = xp[i]
            self._fp[i] = fp[i]

        # Compute the first derivative through central differencing
        for i in xrange(1, self._n_elems - 1):
            self._fp_prime[i] = (self._fp[i+1] - self._fp[i-1]) / (self._xp[i+1] - self._xp[i-1])
        self._fp_prime[0] = (self._fp[1] - self._fp[0]) / (self._xp[1] - self._xp[0])
        self._fp_prime[self._n_elems-1] = (self._fp[self._n_elems-1] - self._fp[self._n_elems-2]) / (self._xp[self._n_elems-1] - self._xp[self._n_elems-2])

    cdef DTYPE_FLOAT_t get_value(self, Particle* particle) except FLOAT_ERR:
        """ Evaluate the interpolating function at the particle's location

        Parameters
        ----------
        *particle: C pointer
            C Pointer to a Particle struct
        """
        return linear_interp(particle.get_omega_interfaces(),
                self._fp[particle.get_k_layer()],
                self._fp[particle.get_k_layer()+1])

    cdef DTYPE_FLOAT_t get_first_derivative(self, Particle* particle) except FLOAT_ERR:
        """ Evaluate the derivative of the interpolating function

        Parameters
        ----------
        *particle: C pointer
            C Pointer to a Particle struct
        """
        return linear_interp(particle.get_omega_interfaces(),
                self._fp_prime[particle.get_k_layer()],
                self._fp_prime[particle.get_k_layer()+1])

cdef class CubicSpline1DInterpolator:
    """ Cubic spline 1D Interpolator
    
    This class contains methods that perform cubic spline 1D interpolation.
    
    Parameters
    ----------
    n_elems : int
        The number of points at which data are defined. This is used to
        create data arrays of the correct size, which are later used for
        the interpolation.
    """

    def __cinit__(self, n_elems):
        self._n_elems = n_elems
        self._first_order = 1
        self._second_order = 2
        self._spline = SplineWrapper()

    cdef set_points(self, DTYPE_FLOAT_t[:] xp, DTYPE_FLOAT_t[:] fp):
        """ Set points for the cubic spline interpolator
        
        This function is basically a wrapper for the same function that is
        implemented in C++. The supplied data arrays are first converted into
        C++ vectors before being passed in. This is the data type the C++
        spline interpolator expects.
        
        TODO
        ----
        1) Check whether it is necessary to create a C++ vector before passing
        the data on. Posts on stack overflow suggest numpy arrays can be
        passed to C++ functions that expect C++ vectors, but I am not sure how
        Cython handles this, or whether it would work here.
        2) Exception handling.
        """
        cdef vector[DTYPE_FLOAT_t] xp_tmp
        cdef vector[DTYPE_FLOAT_t] fp_tmp
        
        # Generate C++ vectors from the supplied data arrays
        for i in xrange(self._n_elems):
            xp_tmp.push_back(xp[i])
            fp_tmp.push_back(fp[i])

        self._spline.set_points(xp_tmp, fp_tmp)

    cdef DTYPE_FLOAT_t get_value(self, Particle* particle) except FLOAT_ERR:
        """ Evaluate the interpolating function at the particle's location

        This function is basically a wrapper - all the hard work is done in C++.

        Parameters
        ----------
        *particle: C pointer
            C Pointer to a Particle struct
        """
        return self._spline.call(particle.get_x3())

    cdef DTYPE_FLOAT_t get_first_derivative(self, Particle* particle) except FLOAT_ERR:
        """ Evaluate the derivative of the interpolating function

        The derivative is computed at the particle's location. As before, this
        function is basically a wrapper - all the hard work is done in C++.

        Parameters
        ----------
        *particle: C pointer
            C Pointer to a Particle struct
        """
        return self._spline.deriv(self._first_order, particle.get_x3())


def interpolate_within_element_wrapper(const vector[DTYPE_FLOAT_t] &val, const vector[DTYPE_FLOAT_t] &phi):
    cdef DTYPE_FLOAT_t _val[3]
    cdef DTYPE_FLOAT_t _phi[3]
    cdef DTYPE_INT_t i

    if val.size() != 3 or phi.size() !=3:
        raise ValueError('Input arrays should have size three')

    for i in range(3):
        _val[i] = val[i]
        _phi[i] = phi[i]

    return interpolate_within_element(_val, _phi)

def get_interpolator(config, n_elems):
    """ Interpolator factory method

        Parameters
        ----------
        config : ConfigParser
            Configuration object.
        
        n_elems : int
            The number of points at which data is defined. Used when allocating
            memory to the interpolating object.
    """
    if config.get("OCEAN_DATA", "vertical_interpolation_scheme") == "linear":
        return Linear1DInterpolator(n_elems)
    elif config.get("OCEAN_DATA", "vertical_interpolation_scheme") == "cubic_spline":
        return CubicSpline1DInterpolator(n_elems)
    else:
        raise ValueError('Unsupported vertical interpolation scheme.')


cdef DTYPE_FLOAT_t get_linear_fraction_safe(const DTYPE_FLOAT_t &var, const DTYPE_FLOAT_t &var1,
        const DTYPE_FLOAT_t &var2) except FLOAT_ERR:
    """ Compute the fractional linear distance of a point between two numbers
    
    The function is deemed safe as it raises an exception if `var' does not lie
    between `var1' and`var2'. Clients should call `get_linear_fraction' if this
    behaviour is not desired.
    
    """
    cdef DTYPE_FLOAT_t frac
    
    frac = get_linear_fraction(var, var1, var2)
    
    if frac >= 0.0 and frac <= 1.0:
        return frac
    else:
        raise ValueError('{} does not lie between {} and {}.'.format(var, var1, var2))

