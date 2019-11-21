include "constants.pxi"

from libc.math cimport sqrt

import logging

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.boundary_conditions import get_horiz_boundary_condition_calculator
from pylag.boundary_conditions import get_vert_boundary_condition_calculator

# PyLag cimports
from pylag.particle cimport ParticleSmartPtr
from pylag.boundary_conditions cimport HorizBoundaryConditionCalculator
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator
from delta cimport reset
cimport pylag.random as random

cdef class NumMethod:
    """ An abstract base class for numerical integration schemes
    
    The following method(s) should be implemented in the derived class:
    
    * :meth: `step`
    """

    def step_wrapper(self, DataReader data_reader, DTYPE_FLOAT_t time,
                     ParticleSmartPtr particle):
        """ Python friendly wrapper for step()
        
        """
        return self.step(data_reader, time, particle.get_ptr())

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
                          Particle *particle) except INT_ERR:
        """ Perform one iteration of the numerical method
        
        Following the update, the particle's new position is saved.
        """
        raise NotImplementedError


cdef class TestNumMethod(NumMethod):
    """ Test iterative method

    Class to assist in testing other parts of the code that may
    require an object of type NumMethod to exist, but which does
    nothing.
    """
    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
                          Particle *particle) except INT_ERR:
        return IN_DOMAIN


cdef class StdNumMethod(NumMethod):
    """ Standard numerical method
    
    The method can be used for cases in which pure advection, pure diffusion
    or advection and diffusion are modelled. In the case of the latter,
    the deterministic and stochastic components of particle movement share the
    same time step. If you would prefer to use some form of operator splitting
    (e.g. to reduce simulation times) use the methods `OS1NumMethod' or 
    `OS2NumMethod' instead.
    
    Attributes:
    -----------
    _time_step : float
        Time step to be used by the iterative method
    
    _iterative_method : _ItMethod
        The iterative method used (e.g. Euler etc)
    
    _horiz_bc_calculator : HorizBoundaryConditionCalculator
        The method used for computing horizontal boundary conditions.

    _vert_bc_calculator : VertBoundaryConditionCalculator
        The method used for computing vertical boundary conditions.
    """
    cdef DTYPE_FLOAT_t _time_step

    cdef bint _depth_restoring
    cdef DTYPE_FLOAT_t _fixed_depth_below_surface

    cdef ItMethod _iterative_method
    cdef HorizBoundaryConditionCalculator _horiz_bc_calculator
    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
            Configuration object
        """
        self._iterative_method = get_iterative_method(config)
        
        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)
        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)

        self._time_step = self._iterative_method.get_time_step()

        try:
            self._depth_restoring = config.getboolean("SIMULATION", "depth_restoring")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self._depth_restoring = False

        try:
            self._fixed_depth_below_surface = config.getfloat("SIMULATION", "fixed_depth")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self._fixed_depth_below_surface = FLOAT_ERR

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle) except INT_ERR:
        """ Perform one iteration of the numerical method
        
        If the particle's new position lies outside of the model domain, the
        specified boundary conditions are applied. The `flag' variable is used
        to tell the caller whether the particle's position was successfully
        updated.

        Parameters:
        -----------
        data_reader : DataReader
            DataReader object used for calculating point velocities
            and/or diffusivities.

        time : float
            The current time.

        particle : C pointer
            C pointer to a particle struct.

        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred.
        """
        cdef Particle _particle_copy
        cdef DTYPE_FLOAT_t zmin, zmax
        cdef Delta _delta_X
        cdef DTYPE_INT_t flag
        cdef DTYPE_INT_t counter

        # Create a clone of the current particle to work on
        _particle_copy = particle[0]

        reset(&_delta_X)

        # Compute Delta
        flag = self._iterative_method.step(data_reader, time, &_particle_copy, &_delta_X)

        # Return if the particle crossed an open boundary
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag
                
        # Compute new position
        _particle_copy.xpos += _delta_X.x1
        _particle_copy.ypos += _delta_X.x2
        _particle_copy.zpos += _delta_X.x3
        flag = data_reader.find_host(particle, &_particle_copy)
        
        if flag == LAND_BDY_CROSSED:
            flag = self._horiz_bc_calculator.apply(data_reader, particle, &_particle_copy)

        # Second check for an open boundary crossing
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag
        
        # Restore to a fixed depth?
        if self._depth_restoring is True:
            zmax = data_reader.get_zmax(time+self._time_step, &_particle_copy)
            _particle_copy.zpos = self._fixed_depth_below_surface + zmax
            
            # Determine the new host zlayer
            flag = data_reader.set_vertical_grid_vars(time+self._time_step, &_particle_copy)

            # Return if failure recorded
            if flag != IN_DOMAIN:
                return flag

            # Copy back particle properties
            particle[0] = _particle_copy

            return flag

        # Set vertical grid vars. NB use t + dt!
        flag = data_reader.set_vertical_grid_vars(time+self._time_step, &_particle_copy)

        # Apply surface/bottom boundary conditions if required
        if flag != IN_DOMAIN:
            flag = self._vert_bc_calculator.apply(data_reader, time+self._time_step, &_particle_copy)

            # Return if failure recorded
            if flag == BDY_ERROR:
                return flag

        # Copy back particle properties
        particle[0] = _particle_copy

        return flag

cdef class OS0NumMethod(NumMethod):
    """ Numerical method that employs operator splitting
    
    The numerical method should be used when the effects of advection and
    diffusion are combined using a form of operator splitting in which
    first the advection step is computed, then `n' diffusion steps. The two
    processes can use different time steps - typically, the time step used
    for diffusion will be smaller than that used for advection - which has
    the potential to significantly reduce run times. Note the advection time 
    step, which must be set in the supplied config, should be an exact multiple
    of the diffusion time step; if it isn't, an exception will be raised.

    Attributes:
    -----------
    _adv_time_step : float
        Time step used for advection

    _diff_time_step : float
        Time step used for diffusion

    _n_sub_time_steps : int
        The number of diffusion time steps for each advection step

    _adv_iterative_method : _ItMethod
        The iterative method used for advection (e.g. Euler etc)

    _diff_iterative_method : _ItMethod
        The iterative method used for diffusion (e.g. Euler etc)

    _horiz_bc_calculator : HorizBoundaryConditionCalculator
        The method used for computing horizontal boundary conditions.

    _vert_bc_calculator : VertBoundaryConditionCalculator
        The method used for computing vertical boundary conditions.
    """
    cdef DTYPE_FLOAT_t _adv_time_step
    cdef DTYPE_FLOAT_t _diff_time_step
    cdef DTYPE_INT_t _n_sub_time_steps

    cdef bint _depth_restoring
    cdef DTYPE_FLOAT_t _fixed_depth_below_surface

    cdef ItMethod _adv_iterative_method
    cdef ItMethod _diff_iterative_method
    cdef HorizBoundaryConditionCalculator _horiz_bc_calculator
    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    def __init__(self, config):
        """ Initialise class data members

        Parameters:
        -----------
        config : ConfigParser
            Configuration object
        """
        self._adv_iterative_method = get_adv_iterative_method(config)
        self._diff_iterative_method = get_diff_iterative_method(config)
        
        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)
        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)

        self._adv_time_step = self._adv_iterative_method.get_time_step()
        self._diff_time_step = self._diff_iterative_method.get_time_step()
        
        if self._diff_time_step > self._adv_time_step:
            raise ValueError("The time step for advection "
                    "(time_step_adv, {} s) must be greater than or equal to "
                    "the time step for diffusion "
                    "(time_step_diff, {} s)".format(self._adv_time_step,
                    self._diff_time_step))

        if self._adv_time_step % self._diff_time_step != 0.0:
            raise ValueError("The time step for advection "
                    "(time_step_adv, {} s) must be an exact multiple of the "
                    "time step for diffusion (time_step_diff, {} s)"
                    "".format(self._adv_time_step, self._diff_time_step))
        
        self._n_sub_time_steps = int(self._adv_time_step / self._diff_time_step)

        self._depth_restoring = config.getboolean("SIMULATION", "depth_restoring")
        self._fixed_depth_below_surface = config.getfloat("SIMULATION", "fixed_depth")

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle) except INT_ERR:
        """ Perform one iteration of the numerical integration
        
        If the particle's new position lies outside of the model domain, the
        specified boundary conditions are applied. The `flag' variable is used
        to tell the caller whether the particle's position was successfully
        updated.

        Parameters:
        -----------
        data_reader: object of type DataReader
            A DataReader object used for reading velocities and eddy
            diffusivities/viscosities.

        time: float
            The current time.

        *particle: C pointer
            C Pointer to a Particle struct

        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred.
        """
        cdef DTYPE_FLOAT_t zmin, zmax
        cdef DTYPE_INT_t flag
        cdef Particle _particle_copy_a
        cdef Particle _particle_copy_b
        cdef Delta _delta_X
        cdef DTYPE_FLOAT_t t
        cdef DTYPE_INT_t i
        cdef DTYPE_INT_t counter

        # Advection
        # ---------

        # Create a clone of the current particle to work on
        _particle_copy_a = particle[0]

        # Set delta to zero
        reset(&_delta_X)

        # Compute Delta
        flag = self._adv_iterative_method.step(data_reader, time, &_particle_copy_a, &_delta_X)

        # Return if the particle crossed an open boundary
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # Compute new position
        _particle_copy_a.xpos += _delta_X.x1
        _particle_copy_a.ypos += _delta_X.x2
        _particle_copy_a.zpos += _delta_X.x3
        flag = data_reader.find_host(particle, &_particle_copy_a)

        if flag == LAND_BDY_CROSSED:
                flag = self._horiz_bc_calculator.apply(data_reader, particle, &_particle_copy_a)

        # Second check for an open boundary crossing
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # Set vertical grid vars NB these are evaluated at time `time',
        # since this is when the diffusion loop starts
        flag = data_reader.set_vertical_grid_vars(time, &_particle_copy_a)

        # Apply surface/bottom boundary conditions if required
        if flag != IN_DOMAIN:
            flag = self._vert_bc_calculator.apply(data_reader, time, &_particle_copy_a)

            # Return if failure recorded
            if flag == BDY_ERROR:
                return flag

        # Diffusion
        # ---------

        # Create a copy of the current particle
        _particle_copy_b = _particle_copy_a

        # Diffusion inner loop
        for i in xrange(self._n_sub_time_steps):
            t = time + i * self._diff_time_step

            # Perform the diffusion step
            reset(&_delta_X)
            flag = self._diff_iterative_method.step(data_reader, t, &_particle_copy_b, &_delta_X)

            # Return if the particle crossed an open boundary
            if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
                return flag

            # Compute new position
            _particle_copy_b.xpos += _delta_X.x1
            _particle_copy_b.ypos += _delta_X.x2
            _particle_copy_b.zpos += _delta_X.x3
            flag = data_reader.find_host(&_particle_copy_a, &_particle_copy_b)

            if flag == LAND_BDY_CROSSED:
                flag = self._horiz_bc_calculator.apply(data_reader, &_particle_copy_a, &_particle_copy_b)

            if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
                return flag

            flag = data_reader.set_vertical_grid_vars(t+self._diff_time_step, &_particle_copy_b)

            # Apply surface/bottom boundary conditions if required
            if flag != IN_DOMAIN:
                flag = self._vert_bc_calculator.apply(data_reader, t+self._diff_time_step, &_particle_copy_b)

                # Return if failure recorded
                if flag == BDY_ERROR:
                    return flag

            # Save the particle's last position to help with host element searching
            _particle_copy_a = _particle_copy_b

        # Restore to a fixed depth?
        if self._depth_restoring is True:
            zmax = data_reader.get_zmax(time+self._adv_time_step, &_particle_copy_b)
            _particle_copy_b.zpos = self._fixed_depth_below_surface + zmax
            
            # Determine the new host zlayer
            flag = data_reader.set_vertical_grid_vars(time+self._adv_time_step, &_particle_copy_b)

            # Return if failure recorded
            if flag != IN_DOMAIN:
                return flag

        particle[0] = _particle_copy_b

        return flag

cdef class OS1NumMethod(NumMethod):
    """ Numerical method that employs strang splitting
    
    The numerical method should be used when the effects of advection and
    diffusion are combined using a form of operator splitting in which
    first a half diffusion step is computed, then a full advection step, then
    a half diffusion step.

    Attributes:
    -----------
    _adv_time_step : float
        Time step used for advection

    _diff_time_step : float
        Time step used for diffusion
    
    _adv_iterative_method : _ItMethod
        The iterative method used for advection (e.g. Euler etc)

    _diff_iterative_method : _ItMethod
        The iterative method used for diffusion (e.g. Euler etc)

    _horiz_bc_calculator : HorizBoundaryConditionCalculator
        The method used for computing horizontal boundary conditions.

    _vert_bc_calculator : VertBoundaryConditionCalculator
        The method used for computing vertical boundary conditions.
    """
    cdef DTYPE_FLOAT_t _adv_time_step
    cdef DTYPE_FLOAT_t _diff_time_step

    cdef bint _depth_restoring
    cdef DTYPE_FLOAT_t _fixed_depth_below_surface

    cdef ItMethod _adv_iterative_method
    cdef ItMethod _diff_iterative_method
    cdef HorizBoundaryConditionCalculator _horiz_bc_calculator
    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
            Configuration object
        """
        self._adv_iterative_method = get_adv_iterative_method(config)
        self._diff_iterative_method = get_diff_iterative_method(config)

        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)
        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)

        self._adv_time_step = self._adv_iterative_method.get_time_step()
        self._diff_time_step = self._diff_iterative_method.get_time_step()

        if self._adv_time_step !=  2.0 * self._diff_time_step:
            raise ValueError("The time step for advection ("\
                    "time_step_adv, {} s) must be exactly twice that for "\
                    "diffusion (time_step_diff, {} s) when using "\
                    "the numerical integration scheme OS1NumMethod."\
                    "".format(self._adv_time_step, self._diff_time_step))

        self._depth_restoring = config.getboolean("SIMULATION", "depth_restoring")
        self._fixed_depth_below_surface = config.getfloat("SIMULATION", "fixed_depth")

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle) except INT_ERR:
        """ Perform one iteration of the numerical integration
        
        If the particle's new position lies outside of the model domain, the
        specified boundary conditions are applied. The `flag' variable is used
        to tell the caller whether the particle's position was successfully
        updated.

        Parameters:
        -----------
        data_reader: object of type DataReader
            A DataReader object used for reading velocities and eddy
            diffusivities/viscosities.

        time: float
            The current time.

        *particle: C pointer
            C Pointer to a Particle struct

        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred.
        """
        cdef DTYPE_FLOAT_t zmin, zmax
        cdef DTYPE_INT_t flag
        cdef Particle _particle_copy_a
        cdef Particle _particle_copy_b
        cdef Delta _delta_X
        cdef DTYPE_FLOAT_t t
        cdef DTYPE_INT_t counter

        # 1st Diffusion step
        # ------------------

        # Create a clone of the current particle to work on
        _particle_copy_a = particle[0]

        # Compute Delta
        reset(&_delta_X)
        flag = self._diff_iterative_method.step(data_reader, time, &_particle_copy_a, &_delta_X)

        # Return if the particle crossed an open boundary
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # Compute new position
        _particle_copy_a.xpos += _delta_X.x1
        _particle_copy_a.ypos += _delta_X.x2
        _particle_copy_a.zpos += _delta_X.x3
        flag = data_reader.find_host(particle, &_particle_copy_a)

        if flag == LAND_BDY_CROSSED:
                flag = self._horiz_bc_calculator.apply(data_reader, particle, &_particle_copy_a)

        # Second check for an open boundary crossing
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # NB these are evaluated at time `time', since this is when the
        # advection update starts
        flag = data_reader.set_vertical_grid_vars(time, &_particle_copy_a)

        # Apply surface/bottom boundary conditions if required
        if flag != IN_DOMAIN:
            flag = self._vert_bc_calculator.apply(data_reader, time, &_particle_copy_a)

            # Return if failure recorded
            if flag == BDY_ERROR:
                return flag

        # Advection step
        # --------------

        # Time at which to start the advection step
        t = time

        # Create a copy of the current particle
        _particle_copy_b = _particle_copy_a

        # Compute Delta
        reset(&_delta_X)
        flag = self._adv_iterative_method.step(data_reader, t, &_particle_copy_b, &_delta_X)

        # Return if the particle crossed an open boundary
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # Compute new position
        _particle_copy_b.xpos += _delta_X.x1
        _particle_copy_b.ypos += _delta_X.x2
        _particle_copy_b.zpos += _delta_X.x3
        flag = data_reader.find_host(&_particle_copy_a, &_particle_copy_b)

        if flag == LAND_BDY_CROSSED:
                flag = self._horiz_bc_calculator.apply(data_reader, &_particle_copy_a, &_particle_copy_b)

        # Second check for an open boundary crossing
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # 2nd Diffusion step
        # ------------------

        # Time at which to start the second diffusion step
        t = time + self._diff_time_step

        flag = data_reader.set_vertical_grid_vars(t, &_particle_copy_b)

        # Apply surface/bottom boundary conditions if required
        if flag != IN_DOMAIN:
            flag = self._vert_bc_calculator.apply(data_reader, t, &_particle_copy_b)

            # Return if failure recorded
            if flag == BDY_ERROR:
                return flag

        # Save the particle's last position to help with host element searching
        _particle_copy_a = _particle_copy_b

        # Compute Delta
        reset(&_delta_X)
        flag = self._diff_iterative_method.step(data_reader, t, &_particle_copy_b, &_delta_X)

        # Return if the particle crossed an open boundary
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # Compute new position
        _particle_copy_b.xpos += _delta_X.x1
        _particle_copy_b.ypos += _delta_X.x2
        _particle_copy_b.zpos += _delta_X.x3
        flag = data_reader.find_host(&_particle_copy_a, &_particle_copy_b)

        if flag == LAND_BDY_CROSSED:
                flag = self._horiz_bc_calculator.apply(data_reader, &_particle_copy_a, &_particle_copy_b)

        # Second check for an open boundary crossing
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # All steps complete - update the original particle's position
        # ------------------------------------------------------------

        t = time + self._adv_time_step

        # Restore to a fixed depth?
        if self._depth_restoring is True:
            zmax = data_reader.get_zmax(t, &_particle_copy_b)
            _particle_copy_b.zpos = self._fixed_depth_below_surface + zmax
            
            # Determine the new host zlayer
            flag = data_reader.set_vertical_grid_vars(t, &_particle_copy_b)

            # Return if failure recorded
            if flag != IN_DOMAIN:
                return flag

            # Copy back particle properties
            particle[0] = _particle_copy_b

            return flag

        flag = data_reader.set_vertical_grid_vars(t, &_particle_copy_b)

        # Apply surface/bottom boundary conditions if required
        if flag != IN_DOMAIN:
            flag = self._vert_bc_calculator.apply(data_reader, t, &_particle_copy_b)

            # Return if failure recorded
            if flag == BDY_ERROR:
                return flag
        
        # Copy back particle properties
        particle[0] = _particle_copy_b

        return flag


def get_num_method(config):
    """ Factory method for constructing NumMethod objects
    
    Parameters:
    -----------
    config : ConfigParser
        Object of type ConfigParser.
    """
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
    elif config.get("NUMERICS", "num_method") == "test":
        return TestNumMethod(config)
    else:
        raise ValueError('Unsupported numerical method specified.')


cdef class ItMethod:
    """ An abstract base class for iterative methods
    
    The following method(s) should be implemented in the derived class:
    
    * :meth: `step`

    Attributes:
    -----------
    _time_step : float
        Time step to be used by the iterative method

    _time_direction : float
        Multiplier indicating the integration direction (forward or reverse)
    """
    cdef DTYPE_FLOAT_t get_time_step(self):
        return self._time_step
    
    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        raise NotImplementedError


cdef class AdvRK42DItMethod(ItMethod):
    """ 2D deterministic Fourth Order Runge-Kutta iterative method

    Attributes:
    -----------
    _horiz_bc_calculator : HorizBoundaryConditionCalculator
        The method used for computing horizontal boundary conditions.
    """
    cdef HorizBoundaryConditionCalculator _horiz_bc_calculator

    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
            Configuration object
        """
        # Time direction (forward or reverse)
        self._time_direction = get_time_direction(config)

        # Set time step (-ve if reverse tracking)
        time_step = config.getfloat('NUMERICS', 'time_step_adv')
        self._time_step = time_step * self._time_direction
        
        # Create horizontal boundary conditions calculator
        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        """ Compute changes in a particle's position due to lateral advection
        
        Use a basic fourth order Runge-Kutta scheme to compute changes in a
        particle's position in two dimensions (e_i,e_j). These are saved in an
        object of type Delta. If the particle moves outside of the model domain
        delta_X is left unchanged and the flag identifying that a boundary
        crossing has occurred is returned.
        
        Parameters:
        -----------
        data_reader : DataReader
            DataReader object used for calculating point velocities.

        time : float
            The current time.
        
        particle : C pointer
            C pointer to a Particle struct
        
        delta_X : C pointer
            C pointer to a Delta struct
        
        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred.
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
        cdef DTYPE_INT_t counter

        # Host search flag
        cdef DTYPE_INT_t flag
        
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
        
        flag = data_reader.find_host(particle, &_particle)

        if flag == LAND_BDY_CROSSED:
            flag = self._horiz_bc_calculator.apply(data_reader, particle, &_particle)

        # Check for open boundary crossing
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # Update particle local coordinates
        flag = data_reader.set_vertical_grid_vars(t, &_particle)
        if flag != IN_DOMAIN:
            return flag

        data_reader.get_horizontal_velocity(t, &_particle, vel) 
        for i in xrange(ndim):
            k2[i] = self._time_step * vel[i]

        # Stage 3
        t = time + 0.5 * self._time_step
        _particle.xpos = particle.xpos + 0.5 * k2[0]
        _particle.ypos = particle.ypos + 0.5 * k2[1]
        
        flag = data_reader.find_host(particle, &_particle)

        if flag == LAND_BDY_CROSSED:
            flag = self._horiz_bc_calculator.apply(data_reader, particle, &_particle)

        # Check for open boundary crossing
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR: return flag

        # Update particle local coordinates
        flag = data_reader.set_vertical_grid_vars(t, &_particle)
        if flag != IN_DOMAIN:
            return flag

        data_reader.get_horizontal_velocity(t, &_particle, vel)
        for i in xrange(ndim):
            k3[i] = self._time_step * vel[i]

        # Stage 4
        t = time + self._time_step
        _particle.xpos = particle.xpos + k3[0]
        _particle.ypos = particle.ypos + k3[1]

        flag = data_reader.find_host(particle, &_particle)

        if flag == LAND_BDY_CROSSED:
            flag = self._horiz_bc_calculator.apply(data_reader, particle, &_particle)

        # Check for open boundary crossing
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # Update particle local coordinates
        flag = data_reader.set_vertical_grid_vars(t, &_particle)
        if flag != IN_DOMAIN:
            return flag

        data_reader.get_horizontal_velocity(t, &_particle, vel)
        for i in xrange(ndim):
            k4[i] = self._time_step * vel[i]

        # Sum changes and save
        delta_X.x1 += (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])/6.0
        delta_X.x2 += (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])/6.0
        return flag

cdef class AdvRK43DItMethod(ItMethod):
    """ 3D deterministic Fourth Order Runge-Kutta iterative method
    
    Attributes:
    -----------
    _horiz_bc_calculator : HorizBoundaryConditionCalculator
        The method used for computing horizontal boundary conditions.

    _vert_bc_calculator : VertBoundaryConditionCalculator
        The method used for computing vertical boundary conditions.
    """
    cdef HorizBoundaryConditionCalculator _horiz_bc_calculator
    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
            Configuration object
        """
        # Time direction (forward or reverse)
        self._time_direction = get_time_direction(config)
       
        # Set time step (-ve if reverse tracking)
        time_step = config.getfloat('NUMERICS', 'time_step_adv')
        self._time_step = time_step * self._time_direction

        # Create horizontal boundary conditions calculator
        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)

        # Create vertical boundary conditions calculator
        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)    
    
    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        """ Compute changes in a particle's position due to advection
        
        Use a basic fourth order Runge-Kutta scheme to compute changes in a
        particle's position in three dimensions (e_i, e_j, e_k). These are saved
        in an object of type Delta. If the particle moves outside of the model
        domain delta_X is left unchanged and the flag identifying that a boundary
        crossing has occurred is returned.
        
        Parameters:
        -----------
        data_reader : DataReader
            DataReader object used for calculating point velocities.

        time : float
            The current time.
        
        particle : C pointer
            C pointer to a Particle struct
        
        delta_X : C pointer
            C pointer to a Delta struct
        
        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred.
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
        cdef DTYPE_INT_t counter

        # Host search flag
        cdef DTYPE_INT_t flag

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
        
        flag = data_reader.find_host(particle, &_particle)

        if flag == LAND_BDY_CROSSED:
            flag = self._horiz_bc_calculator.apply(data_reader, particle, &_particle)

        # Check for open boundary crossing
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # Set vertical grid vars
        flag = data_reader.set_vertical_grid_vars(t, &_particle)

        # Apply surface/bottom boundary conditions if required
        if flag != IN_DOMAIN:
            flag = self._vert_bc_calculator.apply(data_reader, t, &_particle)

            # Return if failure recorded
            if flag == BDY_ERROR:
                return flag

        data_reader.get_velocity(t, &_particle, vel)
        for i in xrange(ndim):
            k2[i] = self._time_step * vel[i]

        # Stage 3
        t = time + 0.5 * self._time_step
        _particle.xpos = particle.xpos + 0.5 * k2[0]
        _particle.ypos = particle.ypos + 0.5 * k2[1]
        _particle.zpos = particle.zpos + 0.5 * k2[2]
        
        flag = data_reader.find_host(particle, &_particle)

        if flag == LAND_BDY_CROSSED:
            flag = self._horiz_bc_calculator.apply(data_reader, particle, &_particle)

        # Check for open boundary crossing
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # Set vertical grid vars.
        flag = data_reader.set_vertical_grid_vars(t, &_particle)

        # Apply surface/bottom boundary conditions if required
        if flag != IN_DOMAIN:
            flag = self._vert_bc_calculator.apply(data_reader, t, &_particle)

            # Return if failure recorded
            if flag == BDY_ERROR:
                return flag

        data_reader.get_velocity(t, &_particle, vel)
        for i in xrange(ndim):
            k3[i] = self._time_step * vel[i]

        # Stage 4
        t = time + self._time_step
        _particle.xpos = particle.xpos + k3[0]
        _particle.ypos = particle.ypos + k3[1]
        _particle.zpos = particle.zpos + k3[2]

        flag = data_reader.find_host(particle, &_particle)

        if flag == LAND_BDY_CROSSED:
            flag = self._horiz_bc_calculator.apply(data_reader, particle, &_particle)

        # Check for open boundary crossing
        if flag == OPEN_BDY_CROSSED or flag == BDY_ERROR:
            return flag

        # Set vertical grid vars.
        flag = data_reader.set_vertical_grid_vars(t, &_particle)

        # Apply surface/bottom boundary conditions if required
        if flag != IN_DOMAIN:
            flag = self._vert_bc_calculator.apply(data_reader, t, &_particle)

            # Return if failure recorded
            if flag == BDY_ERROR:
                return flag

        data_reader.get_velocity(t, &_particle, vel)
        for i in xrange(ndim):
            k4[i] = self._time_step * vel[i]

        # Sum changes and save
        delta_X.x1 += (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])/6.0
        delta_X.x2 += (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])/6.0
        delta_X.x3 += (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])/6.0

        return flag

cdef class DiffNaive1DItMethod(ItMethod):
    """ Stochastic Naive Euler 1D iterative method
    """
    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        # Set time step
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 1D using Naive Euler iterative method
        
        This method should only be used when the vertical eddy diffusivity field
        is homogeneous. When it is not, particles will accumulate in regions of
        low diffusivity.
        
        Parameters:
        -----------
        data_reader : DataReader
            DataReader object.

        time : float
            The current time.
        
        particle : C pointer
            C pointer to a Particle struct
        
        delta_X : C pointer
            C pointer to a Delta struct
            
        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        """
        cdef DTYPE_FLOAT_t Kh

        Kh = data_reader.get_vertical_eddy_diffusivity(time, particle)
        
        delta_X.x3 += sqrt(2.0*Kh*self._time_step) * random.gauss(0.0, 1.0)
        
        return 0

cdef class DiffEuler1DItMethod(ItMethod):
    """ Stochastic Euler 1D iterative method
    """
    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        # Set time step
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 1D using Euler iterative method
        
        The scheme includes a deterministic advective term that counteracts the
        tendency for particles to accumulate in regions of low diffusivity
        (c.f. the NaiveEuler scheme). See Grawe (2012) for more details.

        Parameters:
        -----------
        data_reader : DataReader
            DataReader object.

        time : float
            The current time.
        
        particle : C pointer
            C pointer to a Particle struct
        
        delta_X : C pointer
            C pointer to a Delta struct
            
        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        """
        # The vertical eddy diffusiviy and its derivative wrt z
        cdef DTYPE_FLOAT_t Kh, Kh_prime

        Kh = data_reader.get_vertical_eddy_diffusivity(time, particle)
        Kh_prime = data_reader.get_vertical_eddy_diffusivity_derivative(time, particle)

        delta_X.x3 = Kh_prime * self._time_step + sqrt(2.0*Kh*self._time_step) * random.gauss(0.0, 1.0)

        return 0

cdef class DiffVisser1DItMethod(ItMethod):
    """ Stochastic Visser 1D iterative method

    The scheme includes a deterministic advective term that counteracts the
    tendency for particles to accumulate in regions of low diffusivity
    (c.f. the NaiveEuler scheme). In this scheme, the vertical eddy
    diffusivity is computed at a position that lies roughly half way between
    the particle's current position and the position it would be advected
    too. Vertical boundary conditions are invoked if the computed offset is 
    outside of the model grid. See Visser (1997).

    Attributes:
    -----------
    _vert_bc_calculator : VertBoundaryConditionCalculator
        The method used for computing vertical boundary conditions.
    """
    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        # Set time step
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 1D using Visser iterative method

        Parameters:
        -----------
        data_reader : DataReader
            DataReader object.

        time : float
            The current time.
        
        particle : C pointer
            C pointer to a Particle struct
        
        delta_X : C pointer
            C pointer to a Delta struct
            
        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        
        References:
        -----------
        Visser, A. (1997) Using random walk models to simulate the vertical 
        distribution of particles in a turbulent water column.
        Marine Ecology Progress Series, 158, 275-281
        """
        cdef Particle _particle
        cdef DTYPE_FLOAT_t zmin, zmax, zpos_offset
        cdef DTYPE_FLOAT_t Kh, Kh_prime
        cdef DTYPE_FLOAT_t vel[3]

        vel[:] = [0.0, 0.0, 0.0]

        Kh_prime = data_reader.get_vertical_eddy_diffusivity_derivative(time, particle)

        data_reader.get_velocity(time, particle, vel)
        
        zpos_offset = particle.zpos + 0.5 * (vel[2] + Kh_prime) * self._time_step

        # Create a copy of the particle and move it to the offset position
        _particle = particle[0]
        _particle.zpos = zpos_offset

        # Set vertical grid vars.
        flag = data_reader.set_vertical_grid_vars(time, &_particle)

        # Apply surface/bottom boundary conditions if required
        if flag != IN_DOMAIN:
            flag = self._vert_bc_calculator.apply(data_reader, time, &_particle)

            # Return if failure recorded
            if flag == BDY_ERROR:
                return flag

        # Compute Kh at the offset position
        Kh = data_reader.get_vertical_eddy_diffusivity(time, &_particle)

        delta_X.x3 = Kh_prime * self._time_step + sqrt(2.0*Kh*self._time_step) * random.gauss(0.0, 1.0)

        return 0

cdef class DiffMilstein1DItMethod(ItMethod):
    """ Stochastic Milstein 1D iterative method

    This scheme was highlighted by Grawe (2012) as being more
    accurate than the Euler or Visser schemes, but still computationally
    efficient.
    """
    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        # Set time step
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 1D using Milstein iterative method
        
        Parameters:
        -----------
        data_reader : DataReader
            DataReader object.

        time : float
            The current time.
        
        particle : C pointer
            C pointer to a Particle struct
        
        delta_X : C pointer
            C pointer to a Delta struct
        
        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        
        References:
        -----------
        Gräwe, U. (2011) Implementation of high-order particle-tracking schemes
        in a water column model Ocean Modelling, 36, 80 - 89
        """
        cdef DTYPE_FLOAT_t deviate
        cdef DTYPE_FLOAT_t Kh, Kh_prime

        deviate = random.gauss(0.0, 1.0)

        Kh = data_reader.get_vertical_eddy_diffusivity(time, particle)
        Kh_prime = data_reader.get_vertical_eddy_diffusivity_derivative(time, particle)

        delta_X.x3  = 0.5 * Kh_prime * self._time_step * (deviate*deviate + 1.0) + sqrt(2.0 * Kh * self._time_step) * deviate

        return 0

cdef class DiffConst2DItMethod(ItMethod):
    """ Stochastic Constant 2D iterative method

    Attributes:
    -----------
    _Ah : float
        Horizontal eddy viscosity constant
    """
    cdef DTYPE_FLOAT_t _Ah

    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        # Set time step
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

        self._Ah = config.getfloat("OCEAN_CIRCULATION_MODEL", "horizontal_eddy_viscosity_constant")
        
    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 2D using a constant eddy viscosity
        
        This method uses a constant value for the horizontal eddy viscosity that
        is set in the run config.
        
        Parameters:
        -----------
        data_reader : DataReader
            DataReader object.

        time : float
            The current time.
        
        particle : C pointer
            C pointer to a Particle struct
        
        delta_X : C pointer
            C pointer to a Delta struct
            
        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        """
        delta_X.x1 += sqrt(2.0*self._Ah*self._time_step) * random.gauss(0.0, 1.0)
        delta_X.x2 += sqrt(2.0*self._Ah*self._time_step) * random.gauss(0.0, 1.0)
        
        return 0

cdef class DiffNaive2DItMethod(ItMethod):
    """ Stochastic Naive Euler 2D iterative method
    
    This method is very similar to that implemented in DiffConst2DItMethod
    with the difference being the eddy viscosity is provided by DataReader.
    As in the 1D case, this method should not be used when the eddy 
    viscosity field is inhomogeneous.
    """
    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        # Set time step
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 2D using Naive Euler iterative method
        
        Parameters:
        -----------
        data_reader : DataReader
            DataReader object.

        time : float
            The current time.
        
        particle : C pointer
            C pointer to a Particle struct
        
        delta_X : C pointer
            C pointer to a Delta struct
            
        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        """
        # The horizontal eddy viscosity
        cdef DTYPE_FLOAT_t Ah
        
        # The horizontal eddy viscosity at the particle's current location
        Ah = data_reader.get_horizontal_eddy_viscosity(time, particle)
        
        # Change in position
        delta_X.x1 += sqrt(2.0*Ah*self._time_step) * random.gauss(0.0, 1.0)
        delta_X.x2 += sqrt(2.0*Ah*self._time_step) * random.gauss(0.0, 1.0)
        
        return 0

cdef class DiffMilstein2DItMethod(ItMethod):
    """ Stochastic Milstein 2D iterative method

    This method is a 2D implementation of the Milstein scheme.
    """
    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        # Set time step
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 2D using Milstein iterative method
        
        Parameters:
        -----------
        data_reader : DataReader
            DataReader object.

        time : float
            The current time.
        
        particle : C pointer
            C pointer to a Particle struct
        
        delta_X : C pointer
            C pointer to a Delta struct
            
        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        """
        cdef DTYPE_FLOAT_t deviate_x, deviate_y
        cdef DTYPE_FLOAT_t Ah
        cdef DTYPE_FLOAT_t Ah_prime[2]

        Ah = data_reader.get_horizontal_eddy_viscosity(time, particle)
        data_reader.get_horizontal_eddy_viscosity_derivative(time, particle, Ah_prime)

        deviate_x = random.gauss(0.0, 1.0)
        deviate_y = random.gauss(0.0, 1.0)

        delta_X.x1  = 0.5 * Ah_prime[0] * self._time_step * (deviate_x*deviate_x + 1.0) \
                + sqrt(2.0 * Ah * self._time_step) * deviate_x
        delta_X.x2  = 0.5 * Ah_prime[1] * self._time_step * (deviate_y*deviate_y + 1.0) \
                + sqrt(2.0 * Ah * self._time_step) * deviate_y

        return 0

cdef class DiffMilstein3DItMethod(ItMethod):
    """ Stochastic Milstein 3D iterative method

    This method is a 3D implementation of the Milstein scheme.
    """
    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        # Set time step
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 3D using Milstein iterative method
        
        Parameters:
        -----------
        data_reader : DataReader
            DataReader object.

        time : float
            The current time.
        
        particle : C pointer
            C pointer to a Particle struct
        
        delta_X : C pointer
            C pointer to a Delta struct
            
        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        """
        cdef DTYPE_FLOAT_t deviate_x, deviate_y, deviate_z
        cdef DTYPE_FLOAT_t Ah
        cdef DTYPE_FLOAT_t Kh
        cdef DTYPE_FLOAT_t Ah_prime[2]
        cdef DTYPE_FLOAT_t Kh_prime

        Ah = data_reader.get_horizontal_eddy_viscosity(time, particle)
        data_reader.get_horizontal_eddy_viscosity_derivative(time, particle, Ah_prime)

        Kh = data_reader.get_vertical_eddy_diffusivity(time, particle)
        Kh_prime = data_reader.get_vertical_eddy_diffusivity_derivative(time, particle)

        deviate_x = random.gauss(0.0, 1.0)
        deviate_y = random.gauss(0.0, 1.0)
        deviate_z = random.gauss(0.0, 1.0)

        delta_X.x1  = 0.5 * Ah_prime[0] * self._time_step * (deviate_x*deviate_x + 1.0) \
                + sqrt(2.0 * Ah * self._time_step) * deviate_x
        delta_X.x2  = 0.5 * Ah_prime[1] * self._time_step * (deviate_y*deviate_y + 1.0) \
                + sqrt(2.0 * Ah * self._time_step) * deviate_y
        delta_X.x3  = 0.5 * Kh_prime * self._time_step * (deviate_z*deviate_z + 1.0) \
                + sqrt(2.0 * Kh * self._time_step) * deviate_z

        return 0

cdef class AdvDiffMilstein3DItMethod(ItMethod):
    """ Milstein 3D iterative method

    In this class the contributions of both advection and diffusion are
    accounted for.
    """
    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        # Set time step
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 3D using Milstein iterative method
        
        This method is a 3D implementation of the Milstein scheme that accounts
        for the contributions of both advection and diffusion.
        
        Parameters:
        -----------
        data_reader : DataReader
            DataReader object.

        time : float
            The current time.
        
        particle : C pointer
            C pointer to a Particle struct
        
        delta_X : C pointer
            C pointer to a Delta struct
            
        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        """
        cdef DTYPE_FLOAT_t vel[3]
        cdef DTYPE_FLOAT_t deviate_x, deviate_y, deviate_z
        cdef DTYPE_FLOAT_t Ah
        cdef DTYPE_FLOAT_t Kh
        cdef DTYPE_FLOAT_t Ah_prime[2]
        cdef DTYPE_FLOAT_t Kh_prime

        data_reader.get_velocity(time, particle, vel) 

        Ah = data_reader.get_horizontal_eddy_viscosity(time, particle)
        data_reader.get_horizontal_eddy_viscosity_derivative(time, particle, Ah_prime)

        Kh = data_reader.get_vertical_eddy_diffusivity(time, particle)
        Kh_prime = data_reader.get_vertical_eddy_diffusivity_derivative(time, particle)

        deviate_x = random.gauss(0.0, 1.0)
        deviate_y = random.gauss(0.0, 1.0)
        deviate_z = random.gauss(0.0, 1.0)

        delta_X.x1  = vel[0] * self._time_step \
                + 0.5 * Ah_prime[0] * self._time_step * (deviate_x*deviate_x + 1.0) \
                + sqrt(2.0 * Ah * self._time_step) * deviate_x

        delta_X.x2  = vel[1] * self._time_step \
                + 0.5 * Ah_prime[1] * self._time_step * (deviate_y*deviate_y + 1.0) \
                + sqrt(2.0 * Ah * self._time_step) * deviate_y

        delta_X.x3  = vel[2] * self._time_step \
                + 0.5 * Kh_prime * self._time_step * (deviate_z*deviate_z + 1.0) \
                + sqrt(2.0 * Kh * self._time_step) * deviate_z

        return 0

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
    
    # Prevent backtracking when using RDMs
    if "Diff" in config.get("NUMERICS", "iterative_method") and config.get("SIMULATION", "time_direction") == "reverse":
        raise ValueError("The use of RDMs when reverse tracking is prohibited")

    # Return the specified iterative method
    if config.get("NUMERICS", "iterative_method") == "Adv_RK4_2D":
        return AdvRK42DItMethod(config)
    elif config.get("NUMERICS", "iterative_method") == "Adv_RK4_3D":
        return AdvRK43DItMethod(config)
    elif config.get("NUMERICS", "iterative_method") == "Diff_Const_2D":
        return DiffConst2DItMethod(config)
    elif config.get("NUMERICS", "iterative_method") == "Diff_Naive_1D":
        return DiffNaive1DItMethod(config)
    elif config.get("NUMERICS", "iterative_method") == "Diff_Naive_2D":
        return DiffNaive2DItMethod(config)
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
        raise ValueError("Failed to find the option `adv_iterative_method' in "\
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

    # Prevent backtracking when using a RDM
    if config.get("SIMULATION", "time_direction") == "reverse":
        raise ValueError("The use of RDMs when reverse tracking is prohibited")
    
    # Return the specified numerical integrator.
    if config.get("NUMERICS", "diff_iterative_method") == "Diff_Const_2D":
        return DiffConst2DItMethod(config)
    elif config.get("NUMERICS", "diff_iterative_method") == "Diff_Naive_1D":
        return DiffNaive1DItMethod(config)
    elif config.get("NUMERICS", "diff_iterative_method") == "Diff_Naive_2D":
        return DiffNaive2DItMethod(config)
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

def get_global_time_step(config):
    """ Return the global time step
    
    This is a utility function that returns the global time step adopted within
    the model. It is included to ensure clients have a robust method of
    calculating the global time step that takes into account the fact that
    different NumMethod and ItMethod objects can use different time steps (or
    combinations of time steps if separate iterative methods are used for 
    advection and diffusion).
    
    The rules are:
    1) Return the advection time step if operator splitting is being used
    2) Return the advection time step if operator splitting isn't being used
    and the iterative method is for advection only.
    3) Return the diffusion tim step if operator splitting isn't being used,
    and the iterative method is for diffusion only or advection+diffusion.

    Parameters:
    -----------
    config : ConfigParser
        Object of type ConfigParser.
    
    """
    num_method = config.get("NUMERICS", "num_method")
    if num_method == "operator_split_0" or num_method == "operator_split_1":
        return config.getfloat("NUMERICS", "time_step_adv")
    elif num_method == "standard":
        iterative_method = config.get("NUMERICS", "iterative_method")
        if iterative_method.find('Diff') == -1 and iterative_method.find('Adv') == -1:
            raise ValueError("Expected the config option `iterative_method' "
                "to contain one or both of the substrings `Diff' and `Adv'")

        if iterative_method.find('Diff') != -1:
            return config.getfloat("NUMERICS", "time_step_diff")
        else:
            return config.getfloat("NUMERICS", "time_step_adv")
    else:
        raise ValueError("Unrecognised config option num_method={}".format(num_method))    

def get_time_direction(config):
    # Time direction (forward or reverse tracking)
    try:
        time_direction = config.get("SIMULATION", "time_direction")
    except configparser.NoOptionError:
        time_direction = "forward"

    if time_direction == "forward":
        return 1.
    elif time_direction == "reverse":
        return -1.
    else:
        raise ValueError("Invalid time direction option `{}'".format(time_direction))

