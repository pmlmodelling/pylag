include "constants.pxi"

import logging

from pylag.boundary_conditions import get_horiz_boundary_condition_calculator
from pylag.boundary_conditions import get_vert_boundary_condition_calculator

# PyLag cimports
from pylag.boundary_conditions cimport HorizBoundaryConditionCalculator
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator
from delta cimport reset

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
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class DiffEuler1DItMethod(ItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class DiffVisser1DItMethod(ItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class DiffMilstein1DItMethod(ItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

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
