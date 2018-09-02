include "constants.pxi"

import numpy as np

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag python imports
from pylag.numerics import get_num_method, get_adv_iterative_method, get_diff_iterative_method
from pylag.boundary_conditions import get_vert_boundary_condition_calculator

from particle cimport Particle, ParticleSmartPtr
from data_reader cimport DataReader
from pylag.delta cimport Delta, reset
from pylag.numerics cimport NumMethod, ItMethod
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator

cdef class MockVelocityDataReader(DataReader):
    """ Test data reader for numerical integration schemes
    
    Object passes back u/v/w velocity components for the system of ODEs:
            dx/dt = x           (1)
            dy/dt = 1.5y        (2)
            dz/dt = 0.0         (3)
    
    The velocity eqns are uncoupled and can be solved analytically, giving:
            x = x_0 * exp(t)    (4)
            y = y_0 * exp(1.5t) (5)
            z = 0.0             (6)
    
    """
    cdef DTYPE_INT_t find_host(self, Particle *particle_old,
                               Particle *particle_new) except INT_ERR:
        return IN_DOMAIN

    cdef set_local_coordinates(self, Particle *particle):
        raise NotImplementedError

    cdef DTYPE_INT_t set_vertical_grid_vars(self, DTYPE_FLOAT_t time, Particle *particle) except INT_ERR:
        return IN_DOMAIN

    cdef get_velocity(self, DTYPE_FLOAT_t time, Particle* particle, 
            DTYPE_FLOAT_t vel[3]):
        """ Return velocity field array for the given space/time coordinates.
        
        """  
        vel[0] = self._get_u_component(particle.xpos)
        vel[1] = self._get_v_component(particle.ypos)
        vel[2] = self._get_w_component(particle.zpos)

    cdef get_horizontal_velocity(self, DTYPE_FLOAT_t time, Particle* particle,
            DTYPE_FLOAT_t vel[2]):
        """ Return horizontal velocity for the given space/time coordinates.
        
        """  
        vel[0] = self._get_u_component(particle.xpos)
        vel[1] = self._get_v_component(particle.ypos)

    cdef get_vertical_velocity(self, DTYPE_FLOAT_t time, Particle* particle):
        """ Return vertical velocity field for the given space/time coordinates.
        
        """ 
        return self._get_w_component(particle.zpos)

    def get_velocity_analytic(self, xpos, ypos, zpos=0.0):
        """ Python friendly version of get_velocity(...).
        
        """  
        u = self._get_u_component(xpos)
        v = self._get_v_component(ypos)
        w = self._get_w_component(zpos)
        
        return u,v,w

    def get_position_analytic(self, x0, y0, t):
        """ Return particle positions according to the analytic soln.
        
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

cdef class MockVerticalDiffusivityDataReader(DataReader):
    """Test data reader for vertical random displacement models.
    
    The data reader returns vertical eddy diffusivities drawn from the analytic
    profile:

    k = 0.001 + 0.0136245*zpos - 0.00263245*zpos**2 + 2.11875e-4 * zpos**3 - \
        8.65898e-6 * zpos**4 + 1.7623e-7 * zpos**5 - 1.40918e-9 * zpos**6    
    
    where k (m^2/s) is the vertical eddy diffusivity and zpos (m) is the height
    above the sea bed (positivite up). See Visser (1997) and Ross and 
    Sharples (2004).
    
    Attributes:
    -----------
    _zmin : float
        The minimum depth in m.
    
    _zmax : float
        The maximum depth in m.
    
    References:
    -----------
    Visser, A. Using random walk models to simulate the vertical distribution of
    particles in a turbulent water column Marine Ecology Progress Series, 1997,
    158, 275-281
    
    Ross, O. & Sharples, J. Recipe for 1-D Lagrangian particle tracking models 
    in space-varying diffusivity Limnology and Oceanography Methods, 2004, 2, 
    289-302
    
    """
    cdef DTYPE_FLOAT_t _zmin
    cdef DTYPE_FLOAT_t _zmax
    
    def __init__(self):
        self._zmin = 0.0
        self._zmax = 40.0

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle):
        return self._zmin

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle):
        return self._zmax

    cdef DTYPE_INT_t find_host(self, Particle *particle_old,
                               Particle *particle_new) except INT_ERR:
        return IN_DOMAIN

    cdef set_local_coordinates(self, Particle *particle):
        pass

    cdef DTYPE_INT_t set_vertical_grid_vars(self, DTYPE_FLOAT_t time, Particle *particle) except INT_ERR:
        return IN_DOMAIN

    cdef get_velocity(self, DTYPE_FLOAT_t time, Particle* particle,
            DTYPE_FLOAT_t vel[3]):
        """ Returns a zeroed velocity vector.
        
        The advective velocity is used by random displacement models adapted to
        work in non-homogeneous diffusivity fields, thus the need to implement
        this method here.
        """         
        vel[0] = 0.0
        vel[1] = 0.0
        vel[2] = 0.0

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time, 
            Particle* particle) except FLOAT_ERR:
        """ Returns the vertical eddy diffusivity at zpos.
        
        """  
        return self._get_vertical_eddy_diffusivity(particle.zpos)

    def _get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t zpos):
        cdef DTYPE_FLOAT_t k
        k = 0.001 + 0.0136245*zpos - 0.00263245*zpos**2 + 2.11875e-4 * zpos**3 - \
                8.65898e-6 * zpos**4 + 1.7623e-7 * zpos**5 - 1.40918e-9 * zpos**6
        return k

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self, 
            DTYPE_FLOAT_t time, Particle* particle) except FLOAT_ERR:
        """ Returns the derivative of the vertical eddy diffusivity.

        This is approximated numerically, as in PyLag, as opposed to being
        computed directly using the derivative of k.
        """
        return self._get_vertical_eddy_diffusivity_derivative(particle.zpos)
    
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

cdef class MockHorizontalEddyViscosityDataReader(DataReader):
    """Test data reader for horizontal random displacement models.
    
    The data reader returns horizontal eddy viscosities using the analytic
    formula:
    
    Ah = x^2 + y^2 + C (1)
    
    where `x' is the x cartesian coordinate, `y' is the y cartesian coordinate
    and C is some constant, set equal to 1.0 m2/s. Ah has units
    m^2/s; the time unit is implicit within equation (1). This class is designed
    to help test 2D random displacement models; for example, using the well
    mixed condition.
    
    Attributes:
    -----------
    _C : float
        Constant used in equation (1)

    _xmin, _xmax : float
        Min and max x values between which Ah is defined.

    _ymin, _ymax : float
        Min and max y values between which Ah is defined.
        
    TODO:
    -----
    Will not work as is! It needs find_host() and get_boundary_intersection()
    implementing properly in order to allow horizontal boundary conditions to
    be applied.
    """
    cdef DTYPE_FLOAT_t _C
    cdef DTYPE_FLOAT_t _xmin, _xmax, _ymin, _ymax, _zmin, _zmax
    
    def __init__(self):
        """ Initialise class data members
        """
        self._C = 1.0
        self._xmin = -10.0
        self._xmax = 10.0
        self._ymin = -10.0
        self._ymax = 10.0
        self._zmin = 0.0
        self._zmax = 0.0

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle):
        return self._zmin

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle):
        return self._zmax

    cdef DTYPE_INT_t find_host(self, Particle *particle_old,
                               Particle *particle_new) except INT_ERR:
        return IN_DOMAIN

    cdef set_local_coordinates(self, Particle *particle):
        pass

    cdef DTYPE_INT_t set_vertical_grid_vars(self, DTYPE_FLOAT_t time, Particle *particle) except INT_ERR:
        return IN_DOMAIN

    cdef get_horizontal_eddy_viscosity(self, DTYPE_FLOAT_t time,
            Particle* particle):
        """ Returns the horizontal eddy viscosity
        
        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object. 
        
        Returns:
        --------
        Ah : float
            The horizontal eddy viscosity. 
        """
        return particle.xpos**2 + particle.ypos**2 + self._C

    cdef get_horizontal_eddy_viscosity_derivative(self, DTYPE_FLOAT_t time,
            Particle* particle, DTYPE_FLOAT_t Ah_prime[2]):
        """ Returns the gradient in the horizontal eddy viscosity

        This is computed by taking the derivate of eqn (1) wrt x and y.

        Parameters:
        -----------
        time : float
            Time at which to interpolate.
        
        particle: *Particle
            Pointer to a Particle object.

        Ah_prime : C array, float
            dAh_dx and dH_dy components stored in a C array of length two.
        """
        Ah_prime[0] = 2.0 * particle.xpos
        Ah_prime[1] = 2.0 * particle.ypos

        return

cdef class MockVelocityEddyViscosityDataReader(DataReader):
    """ Test data reader for advection-diffsion numerical integrations schemes
    
    The data reader represents a very simple 2D advection-diffusion test case
    based upon the point release of a tracer into an environment with a
    time independent, spatially homogeneous horizontal velocity field
    described by the variables u and v; and a time independent, spatially homogeneous,
    isotropic horizontal eddy viscosity field described by the variable Ah.
    
    Under these conditions, the evolution of the tracer C (units kg m-2) is 
    described by the equation:
    
    C(t, x, y) = M/(4 * Pi * Ah * t) * exp (-((x - u * t)**2 + (y - v * t)**2)/(4 * Ah * t))
    
    where M (units kg) is the amount of tracer released at time t = 0 (s) and
    position (x,y) = (0,0) (m).
    
    For testing purposes, u = 1.0 m s-1, v = 1.0 m s-1, Ah = 10 m2 s-1 and
    M = 1 kg. Furthermore, w = 0.0 m s-1 and Kh = 0.0 m2 s-1.
    
    Attributes:
    -----------
    _u : float
        x velocity component in Cartesian space.
    _v : float
        y velocity component in Cartesian space.
    _Ah : float
        Horizontal eddy viscosity.
    M : float
        Mass of tracer released at t = t0
    """
    cdef DTYPE_FLOAT_t _u, _v, _w
    cdef DTYPE_FLOAT_t _Ah
    cdef DTYPE_FLOAT_t _M

    cdef DTYPE_FLOAT_t _zmin, _zmax

    def __cinit__(self):
        self._u = 1.0
        self._v = 1.0
        self._w = 0.0
        self._Ah = 10.0
        self._M = 10000.0

        self._zmin = 0.0
        self._zmax = 0.0

    @property
    def M(self):
        return self._M

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle):
        return self._zmin

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle):
        return self._zmax
    
    cdef DTYPE_INT_t find_host(self, Particle *particle_old,
                               Particle *particle_new) except INT_ERR:
        return IN_DOMAIN

    cdef set_local_coordinates(self, Particle *particle):
        pass

    cdef DTYPE_INT_t set_vertical_grid_vars(self, DTYPE_FLOAT_t time, Particle *particle) except INT_ERR:
        return IN_DOMAIN

    cdef get_velocity(self, DTYPE_FLOAT_t time, Particle* particle, 
            DTYPE_FLOAT_t vel[3]):
        """ Return velocity field array for the given space/time coordinates.
        
        """  
        vel[0] = self._u
        vel[1] = self._v
        vel[2] = self._w

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t time, 
            Particle* particle) except FLOAT_ERR:
        return 0.0

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self, 
            DTYPE_FLOAT_t time, Particle* particle) except FLOAT_ERR:
        return 0.0

    cdef get_horizontal_eddy_viscosity(self, DTYPE_FLOAT_t time,
            Particle* particle):
        """ Returns the horizontal eddy viscosity
        """
        return self._Ah

    cdef get_horizontal_eddy_viscosity_derivative(self, DTYPE_FLOAT_t time,
            Particle* particle, DTYPE_FLOAT_t Ah_prime[2]):
        """ Returns the gradient in the horizontal eddy viscosity
        """
        Ah_prime[0] = 0.0
        Ah_prime[1] = 0.0
        return

    def get_concentration_analytic(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos):
        """ Return the mass concentration C(t, x, y) using the analytic formula
        """
        P = self._M / (4.*np.pi*self._Ah*time)
        Q = np.exp(-((xpos - self._u*time)**2.0 + (ypos - self._v*time)**2.0)/(4.*self._Ah*time))
        return P*Q

cdef class MockOneDNumMethod:
    """ Test class for 1D numerical methods
    
    Parameters:
    -----------
    config : ConfigParser
        Configuration object.
    """
    cdef NumMethod _num_method
    
    def __init__(self, config):

        self._num_method = get_num_method(config)
    
    def step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, zpos_arr, DTYPE_INT_t host):
        cdef ParticleSmartPtr particle
        cdef DTYPE_FLOAT_t zpos_new, zmin, zmax
        cdef DTYPE_INT_t i, n_zpos

        # Create particle
        particle = ParticleSmartPtr(xpos=xpos, ypos=ypos, host=host,
                group_id=0, in_domain=True)

        # Number of z positions
        n_zpos = len(zpos_arr)
        
        # Array in which to store updated z positions
        zpos_new_arr = np.empty(n_zpos, dtype=DTYPE_FLOAT)

        for i in xrange(n_zpos):
            # Set zpos, local coordinates and variables that define the location
            # of the particle within the vertical grid
            particle.get_ptr().zpos = zpos_arr[i]
            data_reader.set_local_coordinates(particle.get_ptr())
            if data_reader.set_vertical_grid_vars(time, particle.get_ptr()) != IN_DOMAIN:
                raise RuntimeError('Test particle is not in the domain.')

            if self._num_method.step(data_reader, time, particle.get_ptr()) == IN_DOMAIN:
                zpos_new_arr[i] = particle.get_ptr().zpos
            else:
                raise RuntimeError('Test particle left the domain.')

        # Return the updated position
        return zpos_new_arr
    
cdef class MockTwoDNumMethod:
    """ Test class for 2D numerical methods
    
    Parameters:
    -----------
    config : ConfigParser
        Configuration object.
    """
    cdef NumMethod _num_method
    
    def __init__(self, config):

        self._num_method = get_num_method(config)
    
    def step(self, DataReader data_reader, time, xpos_arr, ypos_arr):
        cdef ParticleSmartPtr particle
        cdef DTYPE_FLOAT_t xpos_new, ypos_new

        if len(xpos_arr) != len(ypos_arr):
            raise ValueError('xpos and ypos array lengths do not match')
        n_particles = len(xpos_arr)

        particle = ParticleSmartPtr(zpos=0.0, group_id=0, host=0, in_domain=True)

        xpos_new_arr = np.empty(n_particles, dtype=DTYPE_FLOAT)
        ypos_new_arr = np.empty(n_particles, dtype=DTYPE_FLOAT)
        
        for i in xrange(n_particles):
            particle.get_ptr().xpos = xpos_arr[i]
            particle.get_ptr().ypos = ypos_arr[i]

            data_reader.set_local_coordinates(particle.get_ptr())
            if data_reader.set_vertical_grid_vars(time, particle.get_ptr()) != IN_DOMAIN:
                raise RuntimeError('Test particle is not in the domain.')

            if self._num_method.step(data_reader, time, particle.get_ptr()) == IN_DOMAIN:
                xpos_new_arr[i] = particle.get_ptr().xpos
                ypos_new_arr[i] = particle.get_ptr().ypos
            else:
                raise RuntimeError('Test particle left the domain.')

        return xpos_new_arr, ypos_new_arr

cdef class MockThreeDNumMethod:
    """ Test class for 3D numerical methods
    
    Parameters:
    -----------
    config : ConfigParser
        Configuration object.
    """
    cdef NumMethod _num_method
    
    def __init__(self, config):

        self._num_method = get_num_method(config)
    
    def step(self, DataReader data_reader, time, xpos_arr, ypos_arr, zpos_arr):
        cdef ParticleSmartPtr particle
        cdef DTYPE_FLOAT_t xpos_new, ypos_new, zpos_new

        if len(xpos_arr) != len(ypos_arr) != len(zpos_arr):
            raise ValueError('xpos, ypos and zpos array lengths do not match')
        n_particles = len(xpos_arr)

        particle = ParticleSmartPtr(group_id=0, host=0, in_domain=True)

        xpos_new_arr = np.empty(n_particles, dtype=DTYPE_FLOAT)
        ypos_new_arr = np.empty(n_particles, dtype=DTYPE_FLOAT)
        zpos_new_arr = np.empty(n_particles, dtype=DTYPE_FLOAT)
        
        for i in xrange(n_particles):
            particle.get_ptr().xpos = xpos_arr[i]
            particle.get_ptr().ypos = ypos_arr[i]
            particle.get_ptr().zpos = zpos_arr[i]

            data_reader.set_local_coordinates(particle.get_ptr())
            if data_reader.set_vertical_grid_vars(time, particle.get_ptr()) != IN_DOMAIN:
                raise RuntimeError('Test particle is not in the domain.')

            if self._num_method.step(data_reader, time, particle.get_ptr()) == IN_DOMAIN:
                # Save new position
                xpos_new_arr[i] = particle.get_ptr().xpos
                ypos_new_arr[i] = particle.get_ptr().ypos
                zpos_new_arr[i] = particle.get_ptr().zpos
            else:
                raise RuntimeError('Test particle left the domain.')

        return xpos_new_arr, ypos_new_arr, zpos_new_arr
