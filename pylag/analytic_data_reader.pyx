import numpy as np

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from data_reader cimport DataReader

cdef class TestVelocityDataReader(DataReader):
    """
    Test data reader for different numerical integration schemes. Object passes
    back u/v/w velocity components for the system of ODEs:
            dx/dt = x           (1)
            dy/dt = 1.5y        (2)
            dz/dt = 0.0         (3)
    
    The velocity eqns are uncoupled and can be solved analytically, giving:
            x = x_0 * exp(t)    (4)
            y = y_0 * exp(1.5t) (5)
            z = 0.0             (6)
    
    which provides a useful test of different integration schemes.
    
    Author: James Clark (PML)
    """
    cpdef find_host(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_INT_t guess):
        return 0    
    
    cdef get_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_FLOAT_t vel[3]):
        """
        Return velocity field array for the given space/time coordinates.
        """  
        vel[0] = self._get_u_component(xpos)
        vel[1] = self._get_v_component(ypos)
        vel[2] = self._get_w_component(zpos)
        
    cdef get_horizontal_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_FLOAT_t vel[2]):
        """
        Return horizontal velocity field array for the given space/time 
        coordinates.
        """  
        vel[0] = self._get_u_component(xpos)
        vel[1] = self._get_v_component(ypos)

    cdef get_vertical_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host):
        """
        Return vertical velocity field for the given space/time coordinates.
        """  
        return self._get_w_component(zpos)

    def get_velocity_analytic(self, xpos, ypos, zpos=0.0):
        """
        Python friendly version of get_velocity(...).
        """  
        u = self._get_u_component(xpos)
        v = self._get_v_component(ypos)
        w = self._get_w_component(zpos)
        return u,v,w

    def get_position_analytic(self, x0, y0, t):
        """
        Return particle positions according to the analytic soln.
        """  
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

    def _get_w_component(self, DTYPE_FLOAT_t zpos):
        return 0.0

cdef class TestDiffusivityDataReader(DataReader):
    """
    Test data reader for random displacement models. Typically these use the 
    vertical or horizontal eddy diffusivity to compute particle displacements.
    This data reader returns vertical diffusivities drawn from the analytic
    profile:

    k = 0.001 + 0.0136245*zpos - 0.00263245*zpos**2 + 2.11875e-4 * zpos**3 - \
        8.65898e-6 * zpos**4 + 1.7623e-7 * zpos**5 - 1.40918e-9 * zpos**6    
    
    where k (m^2/s) is the vertical eddy diffusivity and zpos (m) is the height
    above the sea bed (positivite up). See Visser (1997) and Ross and 
    Sharples (2004).
    
    References:
    Visser, A. Using random walk models to simulate the vertical distribution of
    particles in a turbulent water column Marine Ecology Progress Series, 1997,
    158, 275-281
    
    Ross, O. & Sharples, J. Recipe for 1-D Lagrangian particle tracking models 
    in space-varying diffusivity Limnology and Oceanography Methods, 2004, 2, 
    289-302
    
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
        Returns a zeroed velocity vector. The advective velocity is used by
        random displacement models adapted to work in non-homogenous diffusivity
        fields.
        """         
        vel[0] = 0.0
        vel[1] = 0.0
        vel[2] = 0.0

    cpdef get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host):
        """
        Returns the vertical eddy diffusivity at zpos.
        """  
        return self._get_vertical_eddy_diffusivity(zpos)

    def _get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t zpos):
        cdef DTYPE_FLOAT_t k
        k = 0.001 + 0.0136245*zpos - 0.00263245*zpos**2 + 2.11875e-4 * zpos**3 - \
                8.65898e-6 * zpos**4 + 1.7623e-7 * zpos**5 - 1.40918e-9 * zpos**6
        return k

    cpdef get_vertical_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t time, 
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos,
            DTYPE_INT_t host):
        """
        Returns the derivative of the vertical eddy diffusivity. This is
        approximated numerically, as in PyLag, as opposed to being computed
        directly using the derivative of k.
        """
        return self._get_vertical_eddy_diffusivity_derivative(zpos)
    
    def _get_vertical_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t zpos):
        cdef DTYPE_FLOAT_t zpos_increment, zpos_incremented
        cdef k1, k2

        zpos_increment = (self._zmax - self._zmin) / 1000.0
        
        # Use the negative of zpos_increment at the top of the water column
        if ((zpos + zpos_increment) > self._zmax):
            z_increment = -zpos_increment
        
        zpos_incremented = zpos + zpos_increment

        k1 = self._get_vertical_eddy_diffusivity(zpos)
        k2 = self._get_vertical_eddy_diffusivity(zpos_incremented)
        
        return (k2 - k1) / zpos_increment
