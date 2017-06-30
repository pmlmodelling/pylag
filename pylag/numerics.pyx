include "constants.pxi"

from libc.math cimport sqrt

import logging

from pylag.boundary_conditions import get_horiz_boundary_condition_calculator
from pylag.boundary_conditions import get_vert_boundary_condition_calculator

# PyLag cimports
from pylag.boundary_conditions cimport HorizBoundaryConditionCalculator
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator
from delta cimport reset
cimport pylag.random as random

# Objects of type NumMethod
# -------------------------
#
# Different types of NumMethod object encode different approaches to combining
# the effects of advection and diffusion.

# Base class for NumMethod objects
cdef class NumMethod:

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle) except INT_ERR:
        pass

# Standard numerical method without operator splitting
cdef class StdNumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    cdef ItMethod _iterative_method
    cdef HorizBoundaryConditionCalculator _horiz_bc_calculator
    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

        self._iterative_method = get_iterative_method(config)
        
        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)
        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle) except INT_ERR:
        """ Compute changes in a particle's position due to advection

         Parameters:
         -----------
         data_reader : DataReader
             DataReader object used for calculating point velocities.

         time : float
             The current time.

         particle : *Particle
             Pointer to a particle object.

         Returns:
         --------
         flag : int
             Flag identifying if a boundary crossing has occurred.
         """
        cdef DTYPE_FLOAT_t xpos, ypos, zpos
        cdef DTYPE_FLOAT_t zmin, zmax
        cdef Delta delta_X
        cdef DTYPE_INT_t flag, host

        reset(&delta_X)

        # Compute Delta
        flag = self._iterative_method.step(time, particle, data_reader, &delta_X)

        # Return if the particle crossed an open boundary
        if flag == -2: return flag
                
        # Compute new position
        xpos = particle.xpos + delta_X.x
        ypos = particle.ypos + delta_X.y
        zpos = particle.zpos + delta_X.z
        flag, host = data_reader.find_host(particle.xpos, particle.ypos, xpos,
                ypos, particle.host_horizontal_elem)
              
        # First check for a land boundary crossing
        while flag == -1:
            xpos, ypos = self.horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, xpos, ypos, host)
            flag, host = data_reader.find_host(particle.xpos,
                particle.ypos, xpos, ypos, particle.host_horizontal_elem)

        # Second check for an open boundary crossing
        if flag == -2: return flag

        # If the particle still resides in the domain update its position.
        if flag == 0:
            # Update the particle's position
            particle.xpos = xpos
            particle.ypos = ypos
            particle.zpos = zpos
            particle.host_horizontal_elem = host

            # Update particle local coordinates
            data_reader.set_local_coordinates(particle)

            # Apply surface/bottom boundary conditions and set zpos
            # NB zmin and zmax evaluated at t+dt
            zmin = data_reader.get_zmin(time+self._time_step, particle)
            zmax = data_reader.get_zmax(time+self._time_step, particle)
            if particle.zpos < zmin or particle.zpos > zmax:
                particle.zpos = self.vert_bc_calculator.apply(particle.zpos, zmin, zmax)

            # Determine the new host zlayer
            data_reader.set_vertical_grid_vars(time+self._time_step, particle)
        else:
            raise ValueError('Unrecognised host element flag {}.'.format(flag))

        return flag

# Advection-diffusion numerical method that uses a form of operator splitting - 
# first the advection step is computed, then the diffusion step.
cdef class OS0NumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    cdef ItMethod _adv_iterative_method
    cdef ItMethod _diff_iterative_method
    cdef HorizBoundaryConditionCalculator _horiz_bc_calculator
    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

        self._adv_iterative_method = get_adv_iterative_method(config)
        self._diff_iterative_method = get_diff_iterative_method(config)
        
        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)
        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle) except INT_ERR:
        pass


# Advection-diffusion numerical method that uses a form or operator splitting 
# known using Strang Splitting
cdef class OS1NumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    cdef ItMethod _adv_iterative_method
    cdef ItMethod _diff_iterative_method
    cdef HorizBoundaryConditionCalculator _horiz_bc_calculator
    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

        self._adv_iterative_method = get_adv_iterative_method(config)
        self._diff_iterative_method = get_diff_iterative_method(config)

        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)
        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle) except INT_ERR:
        pass

def get_num_method(config):
    if not config.has_option("NUMERICS", "num_method"):
        raise ValueError("Failed to find the option `num_method' in the "\
                "supplied configuration file.")
    
    # Return the specified numerical integrator.
    if config.get("NUMERICS", "num_method") == "standard":
        return StdNumMethod(config)
    elif config.get("NUMERICS", "num_method") == "operator_split_0":
        return OS0NumMethod(config)
    elif config.get("NUMERICS", "num_method") == "operator_split_1":
        return OS1NumMethod(config)
    else:
        raise ValueError('Unsupported numerical method specified.')

# Base class for ItMethod objects
# -------------------------------

cdef class ItMethod:
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

# Iterative methods for pure advection
# ------------------------------------

cdef class AdvRK42DItMethod(ItMethod):
    """ 2D deterministic Fourth Order Runga Kutta numerical integration scheme.
    
    Parameters:
    -----------
    config : SafeConfigParser
        Configuration object.
    """
    cdef DTYPE_FLOAT_t _time_step
    cdef HorizBoundaryConditionCalculator _horiz_bc_calculator

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')
        
        # Create horizontal boundary conditions calculator
        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)

    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute changes in a particle's position due to advection
        
        Use a basic fourth order Runga Kutta scheme to compute changes in a
        particle's position in two dimensions (e_i,e_j). These are saved in an 
        object of type Delta. If the particle moves outside of the model domain
        delta_X is left unchanged and the flag identifying that a boundary
        crossing has occurred is returned. This function returns a value
        of 0 if a boundary crossing has not occured.
        
        Parameters:
        -----------
        time : float
            The current time.
        
        particle : Particle
            Particle object that stores the current particle's position.
        
        data_reader : DataReader
            DataReader object used for calculating point velocities.
        
        delta_X : Delta
            Delta object in which the change in the particle's position is
            stored.
        
        Returns:
        --------
        host : int
            Flag identifying if a boundary crossing has occurred. A return value
            of 0 means a boundary crossing did not occur.
        """
        # Arrays for RK4 stages
        cdef DTYPE_FLOAT_t k1[2]
        cdef DTYPE_FLOAT_t k2[2]
        cdef DTYPE_FLOAT_t k3[2]
        cdef DTYPE_FLOAT_t k4[2]

        # Calculated vel
        cdef DTYPE_FLOAT_t vel[2]
        
        # Temporary particle object
        cdef Particle _particle

        # Array indices/loop counters
        cdef DTYPE_INT_t ndim = 2
        cdef DTYPE_INT_t i
        
        # Stage 1
        t = time
        _particle = particle[0]
        data_reader.get_horizontal_velocity(t, &_particle, vel) 
        for i in xrange(ndim):
            k1[i] = self._time_step * vel[i]
        
        # Stage 2
        t = time + 0.5 * self._time_step
        _particle.xpos = particle.xpos + 0.5 * k1[0]
        _particle.ypos = particle.ypos + 0.5 * k1[1]
        
        flag, _particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos, _particle.xpos,
                _particle.ypos, particle.host_horizontal_elem)

        # Check for land boundary crossing
        while flag == -1:
            _particle.xpos, _particle.ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, _particle.xpos, _particle.ypos,
                    _particle.host_horizontal_elem)
            flag, _particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos,
                    _particle.xpos, _particle.ypos, particle.host_horizontal_elem)

        # Check for open boundary crossing
        if flag == -2: return flag

        # Update particle local coordinates
        data_reader.set_local_coordinates(&_particle)

        data_reader.set_vertical_grid_vars(t, &_particle)
        data_reader.get_horizontal_velocity(t, &_particle, vel) 
        for i in xrange(ndim):
            k2[i] = self._time_step * vel[i]

        # Stage 3
        t = time + 0.5 * self._time_step
        _particle.xpos = particle.xpos + 0.5 * k2[0]
        _particle.ypos = particle.ypos + 0.5 * k2[1]
        
        flag, _particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos, _particle.xpos,
                _particle.ypos, particle.host_horizontal_elem)

        # Check for land boundary crossing
        while flag == -1:
            _particle.xpos, _particle.ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, _particle.xpos, _particle.ypos,
                    _particle.host_horizontal_elem)
            flag, _particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos,
                    _particle.xpos, _particle.ypos, particle.host_horizontal_elem)

        # Check for open boundary crossing
        if flag == -2: return flag

        # Update particle local coordinates
        data_reader.set_local_coordinates(&_particle)

        data_reader.set_vertical_grid_vars(t, &_particle)
        data_reader.get_horizontal_velocity(t, &_particle, vel)
        for i in xrange(ndim):
            k3[i] = self._time_step * vel[i]

        # Stage 4
        t = time + self._time_step
        _particle.xpos = particle.xpos + k3[0]
        _particle.ypos = particle.ypos + k3[1]

        flag, _particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos, _particle.xpos,
                _particle.ypos, particle.host_horizontal_elem)

        # Check for land boundary crossing
        while flag == -1:
            _particle.xpos, _particle.ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, _particle.xpos, _particle.ypos,
                    _particle.host_horizontal_elem)
            flag, _particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos,
                    _particle.xpos, _particle.ypos, particle.host_horizontal_elem)

        # Check for open boundary crossing
        if flag == -2: return flag

        # Update particle local coordinates
        data_reader.set_local_coordinates(&_particle)

        data_reader.set_vertical_grid_vars(t, &_particle)
        data_reader.get_horizontal_velocity(t, &_particle, vel)
        for i in xrange(ndim):
            k4[i] = self._time_step * vel[i]

        # Sum changes and save
        delta_X.x += (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])/6.0
        delta_X.y += (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])/6.0
    
        return flag

cdef class AdvRK43DItMethod(ItMethod):
    """ 3D deterministic Fourth Order Runga Kutta numerical integration scheme.
    
    Parameters:
    -----------
    config : SafeConfigParser
        Configuration object.
    """

    cdef DTYPE_FLOAT_t _time_step
    cdef HorizBoundaryConditionCalculator _horiz_bc_calculator
    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

        # Create horizontal boundary conditions calculator
        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)

        # Create vertical boundary conditions calculator
        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)    
    
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute changes in a particle's position due to advection
        
        Use a basic fourth order Runga Kutta scheme to compute changes in a
        particle's position in three dimensions (e_i, e_j, e_k). These are saved
        in an object of type Delta. If the particle moves outside of the model
        domain delta_X is left unchanged and the flag identifying that a boundary
        crossing has occurred is returned. This function returns a value
        of 0 if a boundary crossing has not occured.
        
        Parameters:
        -----------
        time : float
            The current time.
        
        particle : Particle
            Particle object that stores the current particle's position.
        
        data_reader : DataReader
            DataReader object used for calculating point velocities.
        
        delta_X : Delta
            Delta object in which the change in the particle's position is
            stored.
        
        Returns:
        --------
        host : int
            Flag identifying if a boundary crossing has occurred. A return value
            of 0 means a boundary crossing did not occur.
        """
        # Arrays for RK4 stages
        cdef DTYPE_FLOAT_t k1[3]
        cdef DTYPE_FLOAT_t k2[3]
        cdef DTYPE_FLOAT_t k3[3]
        cdef DTYPE_FLOAT_t k4[3]

        # Calculated vel
        cdef DTYPE_FLOAT_t vel[3]
        
        # Temporary particle object
        cdef Particle _particle
        
        # For applying vertical boundary conditions
        cdef DTYPE_FLOAT_t zmin, zmax

        # Array indices/loop counters
        cdef DTYPE_INT_t ndim = 3
        cdef DTYPE_INT_t i
        
        # Stage 1
        t = time
        _particle = particle[0]
        data_reader.get_velocity(t, &_particle, vel) 
        for i in xrange(ndim):
            k1[i] = self._time_step * vel[i]
        
        # Stage 2
        t = time + 0.5 * self._time_step
        _particle.xpos = particle.xpos + 0.5 * k1[0]
        _particle.ypos = particle.ypos + 0.5 * k1[1]
        _particle.zpos = particle.zpos + 0.5 * k1[2]
        
        flag, _particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos, _particle.xpos,
                _particle.ypos, particle.host_horizontal_elem)

        # Check for land boundary crossing
        while flag == -1:
            _particle.xpos, _particle.ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, _particle.xpos, _particle.ypos,
                    _particle.host_horizontal_elem)
            flag, _particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos,
                    _particle.xpos, _particle.ypos, particle.host_horizontal_elem)

        # Check for open boundary crossing
        if flag == -2: return flag

        # Update particle local coordinates
        data_reader.set_local_coordinates(&_particle)

        # Impose boundary condition in z
        zmin = data_reader.get_zmin(t, &_particle)
        zmax = data_reader.get_zmax(t, &_particle)
        if _particle.zpos < zmin or _particle.zpos > zmax:
            _particle.zpos = self._vert_bc_calculator.apply(_particle.zpos, zmin, zmax)

        data_reader.set_vertical_grid_vars(t, &_particle)

        data_reader.get_velocity(t, &_particle, vel)
        for i in xrange(ndim):
            k2[i] = self._time_step * vel[i]

        # Stage 3
        t = time + 0.5 * self._time_step
        _particle.xpos = particle.xpos + 0.5 * k2[0]
        _particle.ypos = particle.ypos + 0.5 * k2[1]
        _particle.zpos = particle.zpos + 0.5 * k2[2]
        
        flag, _particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos, _particle.xpos,
                _particle.ypos, particle.host_horizontal_elem)

        # Check for land boundary crossing
        while flag == -1:
            _particle.xpos, _particle.ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, _particle.xpos, _particle.ypos,
                    _particle.host_horizontal_elem)
            flag, _particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos,
                    _particle.xpos, _particle.ypos, particle.host_horizontal_elem)

        # Check for open boundary crossing
        if flag == -2: return flag

        # Update particle local coordinates
        data_reader.set_local_coordinates(&_particle)

        # Impose boundary condition in z
        zmin = data_reader.get_zmin(t, &_particle)
        zmax = data_reader.get_zmax(t, &_particle)
        if _particle.zpos < zmin or _particle.zpos > zmax:
            _particle.zpos = self._vert_bc_calculator.apply(_particle.zpos, zmin, zmax)

        data_reader.set_vertical_grid_vars(t, &_particle)

        data_reader.get_velocity(t, &_particle, vel)
        for i in xrange(ndim):
            k3[i] = self._time_step * vel[i]

        # Stage 4
        t = time + self._time_step
        _particle.xpos = particle.xpos + k3[0]
        _particle.ypos = particle.ypos + k3[1]
        _particle.zpos = particle.zpos + k3[2]

        flag, _particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos, _particle.xpos,
                _particle.ypos, particle.host_horizontal_elem)

        # Check for land boundary crossing
        while flag == -1:
            _particle.xpos, _particle.ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, _particle.xpos, _particle.ypos,
                    _particle.host_horizontal_elem)
            flag, _particle.host_horizontal_elem = data_reader.find_host(particle.xpos, particle.ypos,
                    _particle.xpos, _particle.ypos, particle.host_horizontal_elem)

        # Check for open boundary crossing
        if flag == -2: return flag

        # Update particle local coordinates
        data_reader.set_local_coordinates(&_particle)

        # Impose boundary condition in z
        zmin = data_reader.get_zmin(t, &_particle)
        zmax = data_reader.get_zmax(t, &_particle)
        if _particle.zpos < zmin or _particle.zpos > zmax:
            _particle.zpos = self._vert_bc_calculator.apply(_particle.zpos, zmin, zmax)

        data_reader.set_vertical_grid_vars(t, &_particle)

        data_reader.get_velocity(t, &_particle, vel)
        for i in xrange(ndim):
            k4[i] = self._time_step * vel[i]

        # Sum changes and save
        delta_X.x += (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])/6.0
        delta_X.y += (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])/6.0
        delta_X.z += (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])/6.0

        return flag

# Iterative methods for pure diffusion
# ------------------------------------

cdef class DiffNaive1DItMethod(ItMethod):
    """ Stochastic Naive Euler 1D iterative method
    """
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        self._time_step = config.getfloat('NUMERICS', 'time_step')
        
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 1D using Naive Euler iterative method
        
        This method should only be used when the vertical eddy diffusivity field
        is homogeneous. When it is not, particles will accumulate in regions of
        low diffusivity.
        
        Parameters:
        -----------
        time: float
            The current time.
        particle: object of type Particle
            A Particle object. The object's z position will be updated.
        data_reader: object of type DataReader
            A DataReader object. Used for reading the vertical eddy diffusivity.
        delta_X: object of type Delta
            A Delta object. Used for storing position deltas.
            
        Returns:
        --------
        flat : int
        """
        # The vertical eddy diffusiviy
        cdef DTYPE_FLOAT_t D

        D = data_reader.get_vertical_eddy_diffusivity(time, particle)
        
        # Change in position
        delta_X.z += sqrt(2.0*D*self._time_step) * random.gauss(0.0, 1.0)
        
        return 0

cdef class DiffEuler1DItMethod(ItMethod):
    """ Stochastic Euler 1D iterative method
    """
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        self._time_step = config.getfloat('NUMERICS', 'time_step')

    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 1D using Euler iterative method
        
        The scheme includes a deterministic advective term that counteracts the
        tendency for particles to accumulate in regions of low diffusivity
        (c.f. the NaiveEuler scheme). See Grawe (2012) for more details.

        Parameters:
        -----------
        time: float
            The current time.
        particle: object of type Particle
            A Particle object. The object's z position will be updated.
        data_reader: object of type DataReader
            A DataReader object. Used for reading the vertical eddy diffusivity.
        delta_X: object of type Delta
            A Delta object. Used for storing position deltas.
            
        Returns:
        --------
        flat : int
        """
        # Temporary particle object
        cdef Particle _particle
        
        # The vertical eddy diffusiviy
        #     k - at the advected location
        #     dk_dz - gradient in k
        cdef DTYPE_FLOAT_t k, dk_dz

        # Change in position (units: m)
        cdef DTYPE_FLOAT_t dz_advection, dz_random

        # Create a copy of particle
        _particle = particle[0]

        # Compute the vertical eddy diffusivity at the particle's current location.
        k = data_reader.get_vertical_eddy_diffusivity(time, &_particle)

        # Compute an approximate value for the gradient in the vertical eddy
        # diffusivity at the particle's current location.
        dk_dz = data_reader.get_vertical_eddy_diffusivity_derivative(time, &_particle)

        # Compute the random displacement
        dz_random = sqrt(2.0*k*self._time_step) * random.gauss(0.0, 1.0)

        # Compute the advective displacement for inhomogeneous turbluence. This
        # assumes advection due to the resolved flow is computed elsewhere or
        # is zero.
        dz_advection = dk_dz * self._time_step

        # Change in position
        delta_X.z = dz_advection + dz_random

        return 0

cdef class DiffVisser1DItMethod(ItMethod):
    """ Stochastic Visser 1D iterative method
    """
    cdef DTYPE_FLOAT_t _time_step

    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        self._time_step = config.getfloat('NUMERICS', 'time_step')
        
        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)

    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 1D using Visser iterative method
        
        The scheme includes a deterministic advective term that counteracts the
        tendency for particles to accumulate in regions of low diffusivity
        (c.f. the NaiveEuler scheme). See Visser (1997) and Ross and Sharples
        (2004) for a more detailed discussion.

        Parameters:
        -----------
        time: float
            The current time.

        particle: object of type Particle
            A Particle object. The object's z position will be updated.

        data_reader: object of type DataReader
            A DataReader object. Used for reading the vertical eddy diffusivity.

        delta_X: object of type Delta
            A Delta object. Used for storing position deltas.
            
        Returns:
        --------
        flat : int
        """
        # Temporary particle object
        cdef Particle _particle
        
        # Temporary containers for the particle's location
        cdef DTYPE_FLOAT_t zmin, zmax
        
        # Temporary containers for z position offset from the current position -
        # used in dk_dz calculations
        cdef DTYPE_FLOAT_t zpos_offset
        
        # The vertical eddy diffusiviy
        #     k - at the advected location
        #     dk_dz - gradient in k
        cdef DTYPE_FLOAT_t k, dk_dz

        # The velocity at the particle's current location
        cdef DTYPE_FLOAT_t vel[3]
        vel[:] = [0.0, 0.0, 0.0]

        # Change in position (units: m)
        cdef DTYPE_FLOAT_t dz_advection, dz_random

        # Create a copy of particle
        _particle = particle[0]

        # Compute an approximate value for the gradient in the vertical eddy
        # diffusivity at the particle's current location.
        dk_dz = data_reader.get_vertical_eddy_diffusivity_derivative(time, &_particle)
        
        # Compute the velocity at the particle's current location
        data_reader.get_velocity(time, &_particle, vel)
        
        # Compute the vertical eddy diffusivity at a position that lies roughly
        # between the current particle's position and the position it would be
        # advected too. Apply reflecting boundary condition if the computed
        # offset falls outside of the model domain
        zpos_offset = _particle.zpos + 0.5 * (vel[2] + dk_dz) * self._time_step
        zmin = data_reader.get_zmin(time, &_particle)
        zmax = data_reader.get_zmax(time, &_particle)
        if zpos_offset < zmin or zpos_offset > zmax:
            zpos_offset = self._vert_bc_calculator.apply(zpos_offset, zmin, zmax)

        _particle.zpos = zpos_offset
        data_reader.set_vertical_grid_vars(time, &_particle)
        k = data_reader.get_vertical_eddy_diffusivity(time, &_particle)

        # Compute the random displacement
        dz_random = sqrt(2.0*k*self._time_step) * random.gauss(0.0, 1.0)

        # Compute the advective displacement for inhomogenous turbluence. This
        # assumes advection due to the resolved flow is computed elsewhere or
        # is zero.
        dz_advection = dk_dz * self._time_step

        # Change in position
        delta_X.z = dz_advection + dz_random

        return 0

cdef class DiffMilstein1DItMethod(ItMethod):
    """ Stochastic Milstein 1D iterative method
    """
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        self._time_step = config.getfloat('NUMERICS', 'time_step')

    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 1D using Milstein iterative method
        
        This scheme was highlighted by Grawe (2012) as being more
        accurate that the Euler or Visser schemes, but still computationally
        efficient.
        
        Parameters:
        -----------
        time: float
            The current time.

        particle: object of type Particle
            A Particle object. The object's z position will be updated.

        data_reader: object of type DataReader
            A DataReader object. Used for reading the vertical eddy diffusivity.

        delta_X: object of type Delta
            A Delta object. Used for storing position deltas.
            
        Returns:
        --------
        flat : int
        """
        # Temporary particle object
        cdef Particle _particle
        
        # Random deviate
        cdef DTYPE_FLOAT_t deviate
        
        # The vertical eddy diffusiviy
        #     k - at the advected location
        #     dk_dz - gradient in k
        cdef DTYPE_FLOAT_t k, dk_dz

        # Create a copy of particle
        _particle = particle[0]

        # Compute the random deviate for the update. It is Gaussian, with zero
        # mean and standard deviation equal to 1.0. It is transformed into a
        # Wiener increment later. Not doing this hear minimises the number of
        # square root operations we need to perform.
        deviate = random.gauss(0.0, 1.0)

        # Compute the vertical eddy diffusivity at the particle's current location.
        k = data_reader.get_vertical_eddy_diffusivity(time, &_particle)

        # Compute an approximate value for the gradient in the vertical eddy
        # diffusivity at the particle's current location.
        dk_dz = data_reader.get_vertical_eddy_diffusivity_derivative(time, &_particle)

        # Compute the random displacement
        delta_X.z  = 0.5 * dk_dz * self._time_step * (deviate*deviate + 1.0) + sqrt(2.0 * k * self._time_step) * deviate

        return 0

cdef class DiffMilstein2DItMethod(ItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class DiffMilstein3DItMethod(ItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

# Iterative methods for advection and diffusion
# -----------------------------------------
cdef class AdvDiffMilstein3DItMethod(ItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

def get_iterative_method(config):
    """ Factory method for iterative methods
    
    The type of iterative method to be constructed is read from an object of
    type ConfigParser which is passed in as a function argument. All types of
    ItMethod object are supported.
    
    Parameters:
    -----------
    config : ConfigParser
        Object of type ConfigParser.
    
    """
    
    if not config.has_option("NUMERICS", "iterative_method"):
        raise ValueError("Failed to find the option `iterative_method' in the "\
                "supplied configuration file.")
    
    # Return the specified iterative method
    if config.get("NUMERICS", "iterative_method") == "Adv_RK4_2D":
        return AdvRK42DItMethod(config)
    elif config.get("NUMERICS", "iterative_method") == "Adv_RK4_3D":
        return AdvRK43DItMethod(config)
    elif config.get("NUMERICS", "iterative_method") == "Diff_Naive_1D":
        return DiffNaive1DItMethod(config)
    elif config.get("NUMERICS", "iterative_method") == "Diff_Euler_1D":
        return DiffEuler1DItMethod(config)
    elif config.get("NUMERICS", "iterative_method") == "Diff_Visser_1D":
        return DiffVisser1DItMethod(config)
    elif config.get("NUMERICS", "iterative_method") == "Diff_Milstein_1D":
        return DiffMilstein1DItMethod(config)
    elif config.get("NUMERICS", "iterative_method") == "Diff_Milstein_2D":
        return DiffMilstein2DItMethod(config)
    elif config.get("NUMERICS", "iterative_method") == "Diff_Milstein_3D":
        return DiffMilstein3DItMethod(config)
    elif config.get("NUMERICS", "iterative_method") == "AdvDiff_Milstein_3D":
        return AdvDiffMilstein3DItMethod(config)
    else:
        raise ValueError('Unsupported deterministic-stochastic iterative method.')

def get_adv_iterative_method(config):
    """ Factory method for iterative methods that handle advection only
    
    The type of iterative method to be constructed is read from an object of
    type ConfigParser which is passed in as a function argument. The method will
    only create types that handle pure advection.
    
    Parameters:
    -----------
    config : ConfigParser
        Object of type ConfigParser.
    
    """

    if not config.has_option("NUMERICS", "adv_iterative_method"):
        raise ValueError("Failed to find the option `det_iterative_method' in "\
                "the supplied configuration file.")
    
    # Return the specified numerical integrator.
    if config.get("NUMERICS", "adv_iterative_method") == "Adv_RK4_2D":
        return AdvRK42DItMethod(config)
    elif config.get("NUMERICS", "adv_iterative_method") == "Adv_RK4_3D":
        return AdvRK43DItMethod(config)
    else:
        raise ValueError('Unsupported deterministic iterative method.')

def get_diff_iterative_method(config):
    """ Factory method for iterative methods that handle diffusion only
    
    The type of iterative method to be constructed is read from an object of
    type ConfigParser which is passed in as a function argument. The method will
    only create types that handle pure diffusion.
    
    Parameters:
    -----------
    config : ConfigParser
        Object of type ConfigParser.
    
    """
    if not config.has_option("NUMERICS", "diff_iterative_method"):
        raise ValueError("Failed to find the option `diff_iterative_method' in"\
                " the supplied configuration file.")
    
    # Return the specified numerical integrator.
    if config.get("NUMERICS", "diff_iterative_method") == "Diff_Naive_1D":
        return DiffNaive1DItMethod(config)
    elif config.get("NUMERICS", "diff_iterative_method") == "Diff_Euler_1D":
        return DiffEuler1DItMethod(config)
    elif config.get("NUMERICS", "diff_iterative_method") == "Diff_Visser_1D":
        return DiffVisser1DItMethod(config)
    elif config.get("NUMERICS", "diff_iterative_method") == "Diff_Milstein_1D":
        return DiffMilstein1DItMethod(config)
    elif config.get("NUMERICS", "diff_iterative_method") == "Diff_Milstein_2D":
        return DiffMilstein2DItMethod(config)
    elif config.get("NUMERICS", "diff_iterative_method") == "Diff_Milstein_3D":
        return DiffMilstein3DItMethod(config)
    else:
        raise ValueError('Unsupported iterative method specified.')
