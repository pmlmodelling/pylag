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

cdef class NumMethod:
    """ An abstract base class for numerical integration schemes
    
    The following method(s) should be implemented in the derived class:
    
    * :meth: `step`
    """

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle) except INT_ERR:
        """ Perform one iteration of the numerical method
        
        Following the update, the particle's new position is saved.
        """
        raise NotImplementedError

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
            xpos, ypos = self._horiz_bc_calculator.apply(data_reader,
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
                particle.zpos = self._vert_bc_calculator.apply(particle.zpos, zmin, zmax)

            # Determine the new host zlayer
            data_reader.set_vertical_grid_vars(time+self._time_step, particle)
        else:
            raise ValueError('Unrecognised host element flag {}.'.format(flag))

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

        *delta_X: C pointer
            C Pointer to a Delta struct

        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred.
        """
        cdef DTYPE_FLOAT_t xpos, ypos, zpos
        cdef DTYPE_FLOAT_t zmin, zmax
        cdef DTYPE_INT_t flag, host
        cdef Particle _particle
        cdef Delta _delta_X
        cdef DTYPE_FLOAT_t t
        cdef DTYPE_INT_t i

        # Advection
        # ---------

        # Compute Delta
        reset(&_delta_X)
        flag = self._adv_iterative_method.step(time, particle, data_reader, &_delta_X)

        # Return if the particle crossed an open boundary
        if flag == -2: return flag

        # Compute new position
        xpos = particle.xpos + _delta_X.x
        ypos = particle.ypos + _delta_X.y
        zpos = particle.zpos + _delta_X.z
        flag, host = data_reader.find_host(particle.xpos, particle.ypos, xpos,
                ypos, particle.host_horizontal_elem)
              
        # First check for a land boundary crossing
        while flag == -1:
            xpos, ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, xpos, ypos, host)
            flag, host = data_reader.find_host(particle.xpos,
                particle.ypos, xpos, ypos, particle.host_horizontal_elem)

        # Second check for an open boundary crossing
        if flag == -2: return flag

        # Diffusion
        # ---------

        # First clone the original particle
        _particle = particle[0]

        _particle.xpos = xpos
        _particle.ypos = ypos
        _particle.zpos = zpos
        _particle.host_horizontal_elem = host

        data_reader.set_local_coordinates(&_particle)

        # NB these are evaluated at time `time', since this is when the
        # diffusion loop starts
        zmin = data_reader.get_zmin(time, &_particle)
        zmax = data_reader.get_zmax(time, &_particle)
        if _particle.zpos < zmin or _particle.zpos > zmax:
            _particle.zpos = self._vert_bc_calculator.apply(_particle.zpos, zmin, zmax)

        data_reader.set_vertical_grid_vars(time, &_particle)

        # Diffusion inner loop
        for i in xrange(self._n_sub_time_steps):
            t = time + i * self._diff_time_step

            # Perform the diffusion step
            reset(&_delta_X)
            flag = self._diff_iterative_method.step(t, &_particle, data_reader, &_delta_X)

            # Return if the particle crossed an open boundary
            if flag == -2: return flag

            # Compute new position
            xpos = _particle.xpos + _delta_X.x
            ypos = _particle.ypos + _delta_X.y
            zpos = _particle.zpos + _delta_X.z
            flag, host = data_reader.find_host(_particle.xpos, _particle.ypos, xpos,
                    ypos, _particle.host_horizontal_elem)

            # Check for a land boundary crossing
            while flag == -1:
                xpos, ypos = self._horiz_bc_calculator.apply(data_reader,
                        _particle.xpos, _particle.ypos, xpos, ypos, host)
                flag, host = data_reader.find_host(_particle.xpos,
                    _particle.ypos, xpos, ypos, _particle.host_horizontal_elem)

            if flag == -2: return flag

            # The particle still resides in the domain - update its position
            _particle.xpos = xpos
            _particle.ypos = ypos
            _particle.zpos = zpos
            _particle.host_horizontal_elem = host

            data_reader.set_local_coordinates(&_particle)

            zmin = data_reader.get_zmin(t+self._diff_time_step, &_particle)
            zmax = data_reader.get_zmax(t+self._diff_time_step, &_particle)
            if _particle.zpos < zmin or _particle.zpos > zmax:
                _particle.zpos = self._vert_bc_calculator.apply(_particle.zpos, zmin, zmax)

            data_reader.set_vertical_grid_vars(t+self._diff_time_step, &_particle)

        # Diffusion loop complete - update the original particle's position
        particle[0] = _particle

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

        *delta_X: C pointer
            C Pointer to a Delta struct

        Returns:
        --------
        flag : int
            Flag identifying if a boundary crossing has occurred.
        """
        cdef DTYPE_FLOAT_t xpos, ypos, zpos
        cdef DTYPE_FLOAT_t zmin, zmax
        cdef DTYPE_INT_t flag, host
        cdef Particle _particle
        cdef Delta _delta_X
        cdef DTYPE_FLOAT_t t

        # 1st Diffusion step
        # ------------------

        # Compute Delta
        reset(&_delta_X)
        flag = self._diff_iterative_method.step(time, particle, data_reader, &_delta_X)

        # Return if the particle crossed an open boundary
        if flag == -2: return flag

        # Compute new position
        xpos = particle.xpos + _delta_X.x
        ypos = particle.ypos + _delta_X.y
        zpos = particle.zpos + _delta_X.z
        flag, host = data_reader.find_host(particle.xpos, particle.ypos, xpos,
                ypos, particle.host_horizontal_elem)
              
        # First check for a land boundary crossing
        while flag == -1:
            xpos, ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, xpos, ypos, host)
            flag, host = data_reader.find_host(particle.xpos,
                particle.ypos, xpos, ypos, particle.host_horizontal_elem)

        # Second check for an open boundary crossing
        if flag == -2: return flag

        # Advection step
        # --------------

        # Time at which to start the advection step
        t = time + self._diff_time_step

        # First clone the original particle then update its position
        _particle = particle[0]

        _particle.xpos = xpos
        _particle.ypos = ypos
        _particle.zpos = zpos
        _particle.host_horizontal_elem = host

        data_reader.set_local_coordinates(&_particle)

        zmin = data_reader.get_zmin(t, &_particle)
        zmax = data_reader.get_zmax(t, &_particle)
        if _particle.zpos < zmin or _particle.zpos > zmax:
            _particle.zpos = self._vert_bc_calculator.apply(_particle.zpos, zmin, zmax)

        data_reader.set_vertical_grid_vars(t, &_particle)

        # Compute Delta
        reset(&_delta_X)
        flag = self._adv_iterative_method.step(t, &_particle, data_reader, &_delta_X)

        # Return if the particle crossed an open boundary
        if flag == -2: return flag

        # Compute new position
        xpos = _particle.xpos + _delta_X.x
        ypos = _particle.ypos + _delta_X.y
        zpos = _particle.zpos + _delta_X.z
        flag, host = data_reader.find_host(_particle.xpos, _particle.ypos, xpos,
                ypos, _particle.host_horizontal_elem)
              
        # First check for a land boundary crossing
        while flag == -1:
            xpos, ypos = self._horiz_bc_calculator.apply(data_reader,
                    _particle.xpos, _particle.ypos, xpos, ypos, host)
            flag, host = data_reader.find_host(_particle.xpos,
                _particle.ypos, xpos, ypos, _particle.host_horizontal_elem)

        # Second check for an open boundary crossing
        if flag == -2: return flag

        # 2nd Diffusion step
        # ------------------

        # Time at which to start the second diffusion step
        t = time + self._diff_time_step + self._adv_time_step

        _particle.xpos = xpos
        _particle.ypos = ypos
        _particle.zpos = zpos
        _particle.host_horizontal_elem = host

        data_reader.set_local_coordinates(&_particle)

        zmin = data_reader.get_zmin(t, &_particle)
        zmax = data_reader.get_zmax(t, &_particle)
        if _particle.zpos < zmin or _particle.zpos > zmax:
            _particle.zpos = self._vert_bc_calculator.apply(_particle.zpos, zmin, zmax)

        data_reader.set_vertical_grid_vars(t, &_particle)

        # Compute Delta
        reset(&_delta_X)
        flag = self._diff_iterative_method.step(t, &_particle, data_reader, &_delta_X)

        # Return if the particle crossed an open boundary
        if flag == -2: return flag

        # Compute new position
        xpos = _particle.xpos + _delta_X.x
        ypos = _particle.ypos + _delta_X.y
        zpos = _particle.zpos + _delta_X.z
        flag, host = data_reader.find_host(_particle.xpos, _particle.ypos, xpos,
                ypos, _particle.host_horizontal_elem)
              
        # First check for a land boundary crossing
        while flag == -1:
            xpos, ypos = self._horiz_bc_calculator.apply(data_reader,
                    _particle.xpos, _particle.ypos, xpos, ypos, host)
            flag, host = data_reader.find_host(_particle.xpos,
                _particle.ypos, xpos, ypos, _particle.host_horizontal_elem)

        # Second check for an open boundary crossing
        if flag == -2: return flag

        # All steps complete - update the original particle's position
        # ------------------------------------------------------------

        particle.xpos = xpos
        particle.ypos = ypos
        particle.zpos = zpos
        particle.host_horizontal_elem = host

        data_reader.set_local_coordinates(particle)

        t = time + 2.0*self._diff_time_step + self._adv_time_step 
        zmin = data_reader.get_zmin(t, particle)
        zmax = data_reader.get_zmax(t, particle)
        if particle.zpos < zmin or particle.zpos > zmax:
            particle.zpos = self._vert_bc_calculator.apply(particle.zpos, zmin, zmax)

        data_reader.set_vertical_grid_vars(t, particle)
        
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
    """
    cdef DTYPE_FLOAT_t get_time_step(self):
        return self._time_step
    
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        raise NotImplementedError

cdef class AdvRK42DItMethod(ItMethod):
    """ 2D deterministic Fourth Order Runga Kutta iterative method

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
        self._time_step = config.getfloat('NUMERICS', 'time_step_adv')
        
        # Create horizontal boundary conditions calculator
        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)

    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute changes in a particle's position due to lateral advection
        
        Use a basic fourth order Runga Kutta scheme to compute changes in a
        particle's position in two dimensions (e_i,e_j). These are saved in an 
        object of type Delta. If the particle moves outside of the model domain
        delta_X is left unchanged and the flag identifying that a boundary
        crossing has occurred is returned.
        
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
    """ 3D deterministic Fourth Order Runga Kutta iterative method
    
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
        self._time_step = config.getfloat('NUMERICS', 'time_step_adv')

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
        crossing has occurred is returned.
        
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

cdef class DiffNaive1DItMethod(ItMethod):
    """ Stochastic Naive Euler 1D iterative method
    """

    def __init__(self, config):
        """ Initialise class data members
        
        Parameters:
        -----------
        config : ConfigParser
        """
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')
        
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
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        """
        cdef DTYPE_FLOAT_t Kh

        Kh = data_reader.get_vertical_eddy_diffusivity(time, particle)
        
        delta_X.z += sqrt(2.0*Kh*self._time_step) * random.gauss(0.0, 1.0)
        
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
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

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
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        """
        # The vertical eddy diffusiviy and its derivative wrt z
        cdef DTYPE_FLOAT_t Kh, Kh_prime

        Kh = data_reader.get_vertical_eddy_diffusivity(time, particle)
        Kh_prime = data_reader.get_vertical_eddy_diffusivity_derivative(time, particle)

        delta_X.z = Kh_prime * self._time_step + sqrt(2.0*Kh*self._time_step) * random.gauss(0.0, 1.0)

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
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')
        
        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)

    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 1D using Visser iterative method

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
        
        zpos_offset = _particle.zpos + 0.5 * (vel[2] + Kh_prime) * self._time_step
        zmin = data_reader.get_zmin(time, particle)
        zmax = data_reader.get_zmax(time, particle)
        if zpos_offset < zmin or zpos_offset > zmax:
            zpos_offset = self._vert_bc_calculator.apply(zpos_offset, zmin, zmax)

        # Create a copy of the particle and move it to the offset position
        _particle = particle[0]
        _particle.zpos = zpos_offset
        data_reader.set_vertical_grid_vars(time, &_particle)

        # Compute Kh at the offset position
        Kh = data_reader.get_vertical_eddy_diffusivity(time, &_particle)

        delta_X.z = Kh_prime * self._time_step + sqrt(2.0*Kh*self._time_step) * random.gauss(0.0, 1.0)

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
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 1D using Milstein iterative method
        
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

        delta_X.z  = 0.5 * Kh_prime * self._time_step * (deviate*deviate + 1.0) + sqrt(2.0 * Kh * self._time_step) * deviate

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
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

        self._Ah = config.getfloat("OCEAN_CIRCULATION_MODEL", "horizontal_eddy_viscosity_constant")
        
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 2D using a constant eddy viscosity
        
        This method uses a constant value for the horizontal eddy viscosity that
        is set in the run config.
        
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
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        """
        delta_X.x += sqrt(2.0*self._Ah*self._time_step) * random.gauss(0.0, 1.0)
        delta_X.y += sqrt(2.0*self._Ah*self._time_step) * random.gauss(0.0, 1.0)
        
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
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')
        
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 2D using Naive Euler iterative method
        
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
        flag : int
            Flag identifying if a boundary crossing has occurred. This should
            always be zero since the method does not check for boundary
            crossings.
        """
        # The horizontal eddy viscosity
        cdef DTYPE_FLOAT_t Ah
        
        # The horizontal eddy viscosity at the particle's current location
        Ah = data_reader.get_horizontal_eddy_diffusivity(time, particle)
        
        # Change in position
        delta_X.x += sqrt(2.0*Ah*self._time_step) * random.gauss(0.0, 1.0)
        delta_X.y += sqrt(2.0*Ah*self._time_step) * random.gauss(0.0, 1.0)
        
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
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 2D using Milstein iterative method
        
        Parameters:
        -----------
        time: float
            The current time.

        particle: object of type Particle
            A Particle object. The object's z position will be updated.

        data_reader: object of type DataReader
            A DataReader object. Used for reading the horizontal eddy viscosity.

        delta_X: object of type Delta
            A Delta object. Used for storing position deltas.
            
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

        Ah = data_reader.get_horizontal_eddy_diffusivity(time, particle)
        data_reader.get_horizontal_eddy_diffusivity_derivative(time, particle, Ah_prime)

        deviate_x = random.gauss(0.0, 1.0)
        deviate_y = random.gauss(0.0, 1.0)

        delta_X.x  = 0.5 * Ah_prime[0] * self._time_step * (deviate_x*deviate_x + 1.0) \
                + sqrt(2.0 * Ah * self._time_step) * deviate_x
        delta_X.y  = 0.5 * Ah_prime[1] * self._time_step * (deviate_y*deviate_y + 1.0) \
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
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 3D using Milstein iterative method
        
        Parameters:
        -----------
        time: float
            The current time.

        *particle: C pointer
            C Pointer to a Particle struct

        data_reader: object of type DataReader
            A DataReader object used for reading eddy diffusivities/viscosities.

        *delta_X: C pointer
            C Pointer to a Delta struct
            
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

        Ah = data_reader.get_horizontal_eddy_diffusivity(time, particle)
        data_reader.get_horizontal_eddy_diffusivity_derivative(time, particle, Ah_prime)

        Kh = data_reader.get_vertical_eddy_diffusivity(time, particle)
        Kh_prime = data_reader.get_vertical_eddy_diffusivity_derivative(time, particle)

        deviate_x = random.gauss(0.0, 1.0)
        deviate_y = random.gauss(0.0, 1.0)
        deviate_z = random.gauss(0.0, 1.0)

        delta_X.x  = 0.5 * Ah_prime[0] * self._time_step * (deviate_x*deviate_x + 1.0) \
                + sqrt(2.0 * Ah * self._time_step) * deviate_x
        delta_X.y  = 0.5 * Ah_prime[1] * self._time_step * (deviate_y*deviate_y + 1.0) \
                + sqrt(2.0 * Ah * self._time_step) * deviate_y
        delta_X.z  = 0.5 * Kh_prime * self._time_step * (deviate_z*deviate_z + 1.0) \
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
        self._time_step = config.getfloat('NUMERICS', 'time_step_diff')

    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Compute position delta in 3D using Milstein iterative method
        
        This method is a 3D implementation of the Milstein scheme that accounts
        for the contributions of both advection and diffusion.
        
        Parameters:
        -----------
        time: float
            The current time.

        *particle: C pointer
            C Pointer to a Particle struct

        data_reader: object of type DataReader
            A DataReader object used for reading velocities and eddy 
            diffusivities/viscosities.

        *delta_X: C pointer
            C Pointer to a Delta struct
            
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

        Ah = data_reader.get_horizontal_eddy_diffusivity(time, particle)
        data_reader.get_horizontal_eddy_diffusivity_derivative(time, particle, Ah_prime)

        Kh = data_reader.get_vertical_eddy_diffusivity(time, particle)
        Kh_prime = data_reader.get_vertical_eddy_diffusivity_derivative(time, particle)

        deviate_x = random.gauss(0.0, 1.0)
        deviate_y = random.gauss(0.0, 1.0)
        deviate_z = random.gauss(0.0, 1.0)

        delta_X.x  = vel[0] * self._time_step \
                + 0.5 * Ah_prime[0] * self._time_step * (deviate_x*deviate_x + 1.0) \
                + sqrt(2.0 * Ah * self._time_step) * deviate_x

        delta_X.y  = vel[1] * self._time_step \
                + 0.5 * Ah_prime[1] * self._time_step * (deviate_y*deviate_y + 1.0) \
                + sqrt(2.0 * Ah * self._time_step) * deviate_y

        delta_X.z  = vel[2] * self._time_step \
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
