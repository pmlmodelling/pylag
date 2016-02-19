# cython: profile=True
# cython: linetrace=True

import numpy as np

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from data_reader cimport DataReader

cdef class AnalyticDataReader(DataReader):
    """
    
    The primary purpose of the analytic data reader if to test pylag
    numerical integration schemes.
    
    Author: James Clark (PML)
    """
    cdef DTYPE_FLOAT_t _zmin
    cdef DTYPE_FLOAT_t _zmax
    
    def __init__(self, config):
        self._zmin = config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmin')
        self._zmax = config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmax')

    cpdef find_host(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_INT_t guess):
        return 0    
    
    cdef get_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_FLOAT_t vel[3]):
        """
        Object passes back u/v/w velocity components for the system of ODEs:
            dx/dt = x           (1)
            dy/dt = 1.5y        (2)
            dz/dt = 0.0         (3)
        """         
        vel[0] = self._get_u_component(xpos)
        vel[1] = self._get_v_component(ypos)
        vel[2] = 0.0

    def get_velocity_analytic(self, xpos, ypos, zpos=0.0):
        """
        Python friendly version of get_velocity(...).
        """  
        u = self._get_u_component(xpos)
        v = self._get_v_component(ypos)
        w = 0.0
        return u,v,w

    def get_position_analytic(self, x0, y0, t):
        """
        The velocity eqns are uncoupled and can be solved analytically, giving:
            x = x_0 * exp(t)    (4)
            y = y_0 * exp(1.5t) (5)
            z = 0.0             (6)
        """
        x = self._get_x(x0, t)
        y = self._get_y(y0, t)
        return x,y

    cdef get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host):
        """
        Object returns the vertical eddy diffusivity based on the profile:
            k = 0.001 + 0.0136245*zpos - 0.00263245*zpos**2 + 2.11875e-4 * zpos**3 - \
                8.65898e-6 * zpos**4 + 1.7623e-7 * zpos**5 - 1.40918e-9 * zpos**6
        """
        return self._get_diffusivity(zpos)

    cdef get_vertical_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t time, 
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos,
            DTYPE_INT_t host):
        """
        Returns the derivative of the vertical eddy diffusivity, approximated
        numerically.
        """
        cdef DTYPE_FLOAT_t zmax = 40.0 # TODO this should be a config parameter
        cdef DTYPE_FLOAT_t zpos_increment, zpos_incremented
        cdef k1, k2

        zpos_increment = (self._zmax - self._zmin) / 1000.0
        
        # Use the negative of zpos_increment at the top of the water column
        if ((zpos + zpos_increment) > self._zmax):
            z_increment = -zpos_increment
        
        zpos_incremented = zpos + zpos_increment

        k1 = self._get_diffusivity(zpos)
        k2 = self._get_diffusivity(zpos_incremented)
        
        return (k2 - k1) / zpos_increment

    cpdef get_vertical_eddy_diffusivity_analytic(self, zpos):
        """
        Python friendly version of get_vertical_eddy_diffusivity.
        """
        return self._get_diffusivity(zpos)

    def _get_x(self, x0, t):
        return x0 * np.exp(t)
    
    def _get_y(self, y0, t):
        return y0 * np.exp(1.5*t)

    def _get_u_component(self, DTYPE_FLOAT_t xpos):
        return xpos

    def _get_v_component(self, DTYPE_FLOAT_t ypos):
        return 1.5 * ypos

    def _get_diffusivity(self, DTYPE_FLOAT_t zpos):
        cdef DTYPE_FLOAT_t k
        k = 0.001 + 0.0136245*zpos - 0.00263245*zpos**2 + 2.11875e-4 * zpos**3 - \
                8.65898e-6 * zpos**4 + 1.7623e-7 * zpos**5 - 1.40918e-9 * zpos**6
        return k
