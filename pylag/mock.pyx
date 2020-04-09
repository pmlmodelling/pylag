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

from particle cimport Particle
from particle_cpp_wrapper cimport ParticleSmartPtr
from data_reader cimport DataReader
from pylag.delta cimport Delta, reset
from pylag.numerics cimport NumMethod, ItMethod
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator

cdef class MockVelocityDataReader(DataReader):
    """ Test data reader for numerical integration schemes
    
    The example is taken from Kreyszig, E. (2006) Advanced Engineering
    Mathematics, Ch. 18. P762. In the example, the complex potential
    with F(z) = z^2 = x^2 - y^2 + 2ixy models a flow with:

    Equipotential lines Phi = x^2 - y^2 = const,

    Streamlines Psi = 2xy = const,

    The velocity vector is then:

    V = 2(x - iy) (1),
    
    with components:

    V1 = u =  2x (2),
    V2 = v = -2y (3),

    The speed is:

    |V| = sqrt(x^2 + y^2).

    An analytical expression for the postion vector (X, Y) can be found by
    integrating (2) and (3):

    X(t, x_0) = x_0 * exp^(2t)
    Y(t, y_0) = y_0 * exp^(-2t)
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
        vel[0] = self._get_u_component(particle.get_x1())
        vel[1] = self._get_v_component(particle.get_x2())
        vel[2] = self._get_w_component(particle.get_x3())

    cdef get_horizontal_velocity(self, DTYPE_FLOAT_t time, Particle* particle,
            DTYPE_FLOAT_t vel[2]):
        """ Return horizontal velocity for the given space/time coordinates.
        
        """  
        vel[0] = self._get_u_component(particle.get_x1())
        vel[1] = self._get_v_component(particle.get_x2())

    cdef get_vertical_velocity(self, DTYPE_FLOAT_t time, Particle* particle):
        """ Return vertical velocity field for the given space/time coordinates.
        
        """ 
        return self._get_w_component(particle.get_x3())

    cdef DTYPE_INT_t is_wet(self, DTYPE_FLOAT_t time, Particle *particle) except INT_ERR:
        """ Return is_wet status

        """
        return 1

    def get_velocity_analytic(self, x1, x2, x3=0.0):
        """ Python friendly version of get_velocity(...).
        
        """  
        u = self._get_u_component(x1)
        v = self._get_v_component(x2)
        w = self._get_w_component(x3)
        
        return u,v,w

    def get_position_analytic(self, x0, y0, t):
        """ Return particle positions according to the analytic soln.
        
        """  
        x = self._get_x(x0, t)
        y = self._get_y(y0, t)
        
        return x,y

    def _get_x(self, x0, t):
        return x0 * np.exp(2*t)
    
    def _get_y(self, y0, t):
        return y0 * np.exp(-2*t)

    def _get_u_component(self, DTYPE_FLOAT_t x1):
        return 2.0 * x1

    def _get_v_component(self, DTYPE_FLOAT_t x2):
        return -2.0 * x2

    def _get_w_component(self, DTYPE_FLOAT_t x3):
        return 0.0

cdef class MockVerticalDiffusivityDataReader(DataReader):
    """Test data reader for vertical random displacement models.
    
    The data reader returns vertical eddy diffusivities drawn from the analytic
    profile:

    k = 0.001 + 0.0136245*x3 - 0.00263245*x3**2 + 2.11875e-4 * x3**3 - \
        8.65898e-6 * x3**4 + 1.7623e-7 * x3**5 - 1.40918e-9 * x3**6    
    
    where k (m^2/s) is the vertical eddy diffusivity and x3 (m) is the height
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

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        return self._zmin

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
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
        """ Returns the vertical eddy diffusivity at x3.
        
        """  
        return self._get_vertical_eddy_diffusivity(particle.get_x3())

    def _get_vertical_eddy_diffusivity(self, DTYPE_FLOAT_t x3):
        cdef DTYPE_FLOAT_t k
        k = 0.001 + 0.0136245*x3 - 0.00263245*x3**2 + 2.11875e-4 * x3**3 - \
                8.65898e-6 * x3**4 + 1.7623e-7 * x3**5 - 1.40918e-9 * x3**6
        return k

    cdef DTYPE_FLOAT_t get_vertical_eddy_diffusivity_derivative(self, 
            DTYPE_FLOAT_t time, Particle* particle) except FLOAT_ERR:
        """ Returns the derivative of the vertical eddy diffusivity.

        This is approximated numerically, as in PyLag, as opposed to being
        computed directly using the derivative of k.
        """
        return self._get_vertical_eddy_diffusivity_derivative(particle.get_x3())
    
    def _get_vertical_eddy_diffusivity_derivative(self, DTYPE_FLOAT_t x3):
        cdef DTYPE_FLOAT_t x3_increment, x3_incremented
        cdef k1, k2

        x3_increment = (self._zmax - self._zmin) / 1000.0
        
        # Use the negative of x3_increment at the top of the water column
        if ((x3 + x3_increment) > self._zmax):
            z_increment = -x3_increment
        
        x3_incremented = x3 + x3_increment

        k1 = self._get_vertical_eddy_diffusivity(x3)
        k2 = self._get_vertical_eddy_diffusivity(x3_incremented)
        
        return (k2 - k1) / x3_increment

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

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        return self._zmin

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
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
        return particle.get_x1()**2 + particle.get_x2()**2 + self._C

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
        Ah_prime[0] = 2.0 * particle.get_x1()
        Ah_prime[1] = 2.0 * particle.get_x2()

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

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        return self._zmin

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
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

    def get_concentration_analytic(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t x1,
            DTYPE_FLOAT_t x2):
        """ Return the mass concentration C(t, x, y) using the analytic formula
        """
        P = self._M / (4.*np.pi*self._Ah*time)
        Q = np.exp(-((x1 - self._u*time)**2.0 + (x2 - self._v*time)**2.0)/(4.*self._Ah*time))
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
            DTYPE_FLOAT_t x1, DTYPE_FLOAT_t x2, x3_arr, DTYPE_INT_t host):
        cdef ParticleSmartPtr particle
        cdef DTYPE_FLOAT_t x3_new, zmin, zmax
        cdef DTYPE_INT_t i, n_x3

        # Create particle
        particle = ParticleSmartPtr(x1=x1, x2=x2, host=host,
                group_id=0, in_domain=True)

        # Number of z positions
        n_x3 = len(x3_arr)
        
        # Array in which to store updated z positions
        x3_new_arr = np.empty(n_x3, dtype=DTYPE_FLOAT)

        for i in xrange(n_x3):
            # Set x3, local coordinates and variables that define the location
            # of the particle within the vertical grid
            particle.get_ptr().set_x3(x3_arr[i])
            data_reader.set_local_coordinates(particle.get_ptr())
            if data_reader.set_vertical_grid_vars(time, particle.get_ptr()) != IN_DOMAIN:
                raise RuntimeError('Test particle is not in the domain.')

            if self._num_method.step(data_reader, time, particle.get_ptr()) == IN_DOMAIN:
                x3_new_arr[i] = particle.get_ptr().get_x3()
            else:
                raise RuntimeError('Test particle left the domain.')

        # Return the updated position
        return x3_new_arr
    
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
    
    def step(self, DataReader data_reader, time, x1_arr, x2_arr):
        cdef ParticleSmartPtr particle
        cdef DTYPE_FLOAT_t x1_new, x2_new

        if len(x1_arr) != len(x2_arr):
            raise ValueError('x1 and x2 array lengths do not match')
        n_particles = len(x1_arr)

        particle = ParticleSmartPtr(x3=0.0, group_id=0, host=0, in_domain=True)

        x1_new_arr = np.empty(n_particles, dtype=DTYPE_FLOAT)
        x2_new_arr = np.empty(n_particles, dtype=DTYPE_FLOAT)
        
        for i in xrange(n_particles):
            particle.get_ptr().set_x1(x1_arr[i])
            particle.get_ptr().set_x2(x2_arr[i])

            data_reader.set_local_coordinates(particle.get_ptr())
            if data_reader.set_vertical_grid_vars(time, particle.get_ptr()) != IN_DOMAIN:
                raise RuntimeError('Test particle is not in the domain.')

            if self._num_method.step(data_reader, time, particle.get_ptr()) == IN_DOMAIN:
                x1_new_arr[i] = particle.get_ptr().get_x1()
                x2_new_arr[i] = particle.get_ptr().get_x2()
            else:
                raise RuntimeError('Test particle left the domain.')

        return x1_new_arr, x2_new_arr

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
    
    def step(self, DataReader data_reader, time, x1_arr, x2_arr, x3_arr):
        cdef ParticleSmartPtr particle
        cdef DTYPE_FLOAT_t x1_new, x2_new, x3_new

        if len(x1_arr) != len(x2_arr) != len(x3_arr):
            raise ValueError('x1, x2 and x3 array lengths do not match')
        n_particles = len(x1_arr)

        particle = ParticleSmartPtr(group_id=0, host=0, in_domain=True)

        x1_new_arr = np.empty(n_particles, dtype=DTYPE_FLOAT)
        x2_new_arr = np.empty(n_particles, dtype=DTYPE_FLOAT)
        x3_new_arr = np.empty(n_particles, dtype=DTYPE_FLOAT)
        
        for i in xrange(n_particles):
            particle.get_ptr().set_x1(x1_arr[i])
            particle.get_ptr().set_x2(x2_arr[i])
            particle.get_ptr().set_x3(x3_arr[i])

            data_reader.set_local_coordinates(particle.get_ptr())
            if data_reader.set_vertical_grid_vars(time, particle.get_ptr()) != IN_DOMAIN:
                raise RuntimeError('Test particle is not in the domain.')

            if self._num_method.step(data_reader, time, particle.get_ptr()) == IN_DOMAIN:
                # Save new position
                x1_new_arr[i] = particle.get_ptr().get_x1()
                x2_new_arr[i] = particle.get_ptr().get_x2()
                x3_new_arr[i] = particle.get_ptr().get_x3()
            else:
                raise RuntimeError('Test particle left the domain.')

        return x1_new_arr, x2_new_arr, x3_new_arr
