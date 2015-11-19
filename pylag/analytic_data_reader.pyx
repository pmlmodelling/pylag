import numpy as np

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

class AnalyticDataReader(object):
    """
    Object passes back u/v/w velocity components for the system of ODEs:

    dx/dt = x           (1)
    dy/dt = 1.5y        (2)
    dz/dt = 0.0         (3)
    
    These eqns are uncoupled and can be solved analytically, giving:
    
    x = x_0 * exp(t)    (4)
    y = y_0 * exp(1.5t) (5)
    z = 0.0             (6)
    
    The primary purpose of the analytic data reader if to test pylag
    numerical integration schemes.
    
    Author: James Clark (PML)
    """
    
    def get_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_FLOAT_t[:] vel):
                
        vel[0] = self._get_u_component(xpos)
        vel[1] = self._get_v_component(ypos)
        vel[2] = 0.0

    def find_host(self, xpos, ypos, host=0):
        return 0
    
    def get_velocity_analytic(self, xpos, ypos, zpos=0.0):
        u = self._get_u_component(xpos)
        v = self._get_v_component(ypos)
        w = 0.0
        return u,v,w

    def get_position_analytic(self, x0, y0, t):
        x = self._get_x(x0, t)
        y = self._get_y(y0, t)
        return x,y

    def _get_x(self, x0, t):
        return x0 * np.exp(t)
    
    def _get_y(self, y0, t):
        return y0 * np.exp(1.5*t)

    def _get_u_component(self, DTYPE_FLOAT_t xpos):
        return xpos

    def _get_v_component(self, DTYPE_FLOAT_t ypos):
        return 1.5 * ypos
