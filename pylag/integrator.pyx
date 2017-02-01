include "constants.pxi"

import logging

from pylag.boundary_conditions import get_horiz_boundary_condition_calculator
from pylag.boundary_conditions import get_vert_boundary_condition_calculator

# PyLag cimports
from pylag.boundary_conditions cimport HorizBoundaryConditionCalculator
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator

cdef class NumIntegrator:
    cdef DTYPE_INT_t advect(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class RK4Integrator2D(NumIntegrator):
    """ 2D Fourth Order Runga Kutta numerical integration scheme.
    
    Parameters:
    -----------
    config : SafeConfigParser
        Configuration object.
    """
    cdef DTYPE_FLOAT_t _time_step
    cdef HorizBoundaryConditionCalculator _horiz_bc_calculator

    def __init__(self, config):
        self._time_step = config.getfloat('SIMULATION', 'time_step')
        
        # Create horizontal boundary conditions calculator
        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)

    cdef DTYPE_INT_t advect(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Advect particles forward in time.
        
        Use a basic fourth order Runga Kutta scheme to compute changes in a
        particle's position in two dimensions. These are saved in an object
        of type Delta. If the particle moves outside of the model domain
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
        
        # Temporary containers
        cdef DTYPE_FLOAT_t t, xpos, ypos, zpos
        cdef DTYPE_INT_t flag, host, zlayer

        # Array indices/loop counters
        cdef DTYPE_INT_t ndim = 2
        cdef DTYPE_INT_t i
        
        # Stage 1
        t = time
        xpos = particle.xpos
        ypos = particle.ypos
        zpos = particle.zpos
        host = particle.host_horizontal_elem
        zlayer = particle.host_z_layer
        data_reader.get_horizontal_velocity(t, xpos, ypos, zpos, host, zlayer, vel) 
        for i in xrange(ndim):
            k1[i] = self._time_step * vel[i]
        
        # Stage 2
        t = time + 0.5 * self._time_step
        xpos = particle.xpos + 0.5 * k1[0]
        ypos = particle.ypos + 0.5 * k1[1]
        
        flag, host = data_reader.find_host(particle.xpos, particle.ypos, xpos,
                ypos, particle.host_horizontal_elem)

        # Check for land boundary crossing
        while flag == -1:
            xpos, ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, xpos, ypos, host)
            flag, host = data_reader.find_host(particle.xpos, particle.ypos,
                    xpos, ypos, particle.host_horizontal_elem)

        # Check for open boundary crossing
        if flag == -2: return flag

        zlayer = data_reader.find_zlayer(t, xpos, ypos, zpos, host, zlayer)
        data_reader.get_horizontal_velocity(t, xpos, ypos, zpos, host, zlayer, vel) 
        for i in xrange(ndim):
            k2[i] = self._time_step * vel[i]

        # Stage 3
        t = time + 0.5 * self._time_step
        xpos = particle.xpos + 0.5 * k2[0]
        ypos = particle.ypos + 0.5 * k2[1]
        
        flag, host = data_reader.find_host(particle.xpos, particle.ypos, xpos,
                ypos, particle.host_horizontal_elem)

        # Check for land boundary crossing
        while flag == -1:
            xpos, ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, xpos, ypos, host)
            flag, host = data_reader.find_host(particle.xpos, particle.ypos,
                    xpos, ypos, particle.host_horizontal_elem)

        # Check for open boundary crossing
        if flag == -2: return flag

        zlayer = data_reader.find_zlayer(t, xpos, ypos, zpos, host, zlayer)
        data_reader.get_horizontal_velocity(t, xpos, ypos, zpos, host, zlayer, vel) 
        for i in xrange(ndim):
            k3[i] = self._time_step * vel[i]

        # Stage 4
        t = time + self._time_step
        xpos = particle.xpos + k3[0]
        ypos = particle.ypos + k3[1]

        flag, host = data_reader.find_host(particle.xpos, particle.ypos, xpos,
                ypos, particle.host_horizontal_elem)

        # Check for land boundary crossing
        while flag == -1:
            xpos, ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, xpos, ypos, host)
            flag, host = data_reader.find_host(particle.xpos, particle.ypos,
                    xpos, ypos, particle.host_horizontal_elem)

        # Check for open boundary crossing
        if flag == -2: return flag

        zlayer = data_reader.find_zlayer(t, xpos, ypos, zpos, host, zlayer)
        data_reader.get_horizontal_velocity(t, xpos, ypos, zpos, host, zlayer, vel)
        for i in xrange(ndim):
            k4[i] = self._time_step * vel[i]

        # Sum changes and save
        delta_X.x += (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])/6.0
        delta_X.y += (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])/6.0
    
        return flag

cdef class RK4Integrator3D(NumIntegrator):
    """ 3D Fourth Order Runga Kutta numerical integration scheme.
    
    Parameters:
    -----------
    config : SafeConfigParser
        Configuration object.
    """

    cdef DTYPE_FLOAT_t _time_step
    cdef HorizBoundaryConditionCalculator _horiz_bc_calculator
    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    def __init__(self, config):
        self._time_step = config.getfloat('SIMULATION', 'time_step')

        # Create horizontal boundary conditions calculator
        self._horiz_bc_calculator = get_horiz_boundary_condition_calculator(config)

        # Create vertical boundary conditions calculator
        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)    
    
    cdef DTYPE_INT_t advect(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        """ Advect particles forward in time.
        
        Use a basic fourth order Runga Kutta scheme to compute changes in a
        particle's position in three dimensions. These are saved in an object
        of type Delta. If the particle moves outside of the model domain
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
        cdef DTYPE_FLOAT_t k1[3]
        cdef DTYPE_FLOAT_t k2[3]
        cdef DTYPE_FLOAT_t k3[3]
        cdef DTYPE_FLOAT_t k4[3]

        # Calculated vel
        cdef DTYPE_FLOAT_t vel[3]
        
        # Temporary containers
        cdef DTYPE_FLOAT_t t, xpos, ypos, zpos
        cdef DTYPE_INT_t flag, host, zlayer
        cdef DTYPE_FLOAT_t zmin, zmax

        # Array indices/loop counters
        cdef DTYPE_INT_t ndim = 3
        cdef DTYPE_INT_t i
        
        # Stage 1
        t = time
        xpos = particle.xpos
        ypos = particle.ypos
        zpos = particle.zpos
        host = particle.host_horizontal_elem
        zlayer = particle.host_z_layer
        data_reader.get_velocity(t, xpos, ypos, zpos, host, zlayer, vel) 
        for i in xrange(ndim):
            k1[i] = self._time_step * vel[i]
        
        # Stage 2
        t = time + 0.5 * self._time_step
        xpos = particle.xpos + 0.5 * k1[0]
        ypos = particle.ypos + 0.5 * k1[1]
        zpos = particle.zpos + 0.5 * k1[2]
        
        flag, host = data_reader.find_host(particle.xpos, particle.ypos, xpos,
                ypos, particle.host_horizontal_elem)

        # Check for land boundary crossing
        while flag == -1:
            xpos, ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, xpos, ypos, host)
            flag, host = data_reader.find_host(particle.xpos, particle.ypos,
                    xpos, ypos, particle.host_horizontal_elem)

        # Check for open boundary crossing
        if flag == -2: return flag

        # Impose reflecting boundary condition in z
        zmin = data_reader.get_zmin(t, xpos, ypos)
        zmax = data_reader.get_zmax(t, xpos, ypos)
        if zpos < zmin or zpos > zmax:
            zpos = self._vert_bc_calculator.apply(zpos, zmin, zmax)

        zlayer = data_reader.find_zlayer(t, xpos, ypos, zpos, host, zlayer)

        data_reader.get_velocity(t, xpos, ypos, zpos, host, zlayer, vel) 
        for i in xrange(ndim):
            k2[i] = self._time_step * vel[i]

        # Stage 3
        t = time + 0.5 * self._time_step
        xpos = particle.xpos + 0.5 * k2[0]
        ypos = particle.ypos + 0.5 * k2[1]
        zpos = particle.zpos + 0.5 * k2[2]
        
        flag, host = data_reader.find_host(particle.xpos, particle.ypos, xpos,
                ypos, particle.host_horizontal_elem)

        # Check for land boundary crossing
        while flag == -1:
            xpos, ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, xpos, ypos, host)
            flag, host = data_reader.find_host(particle.xpos, particle.ypos,
                    xpos, ypos, particle.host_horizontal_elem)

        # Check for open boundary crossing
        if flag == -2: return flag

        # Impose reflecting boundary condition in z
        zmin = data_reader.get_zmin(t, xpos, ypos)
        zmax = data_reader.get_zmax(t, xpos, ypos)
        if zpos < zmin or zpos > zmax:
            zpos = self._vert_bc_calculator.apply(zpos, zmin, zmax)

        zlayer = data_reader.find_zlayer(t, xpos, ypos, zpos, host, zlayer)

        data_reader.get_velocity(t, xpos, ypos, zpos, host, zlayer, vel) 
        for i in xrange(ndim):
            k3[i] = self._time_step * vel[i]

        # Stage 4
        t = time + self._time_step
        xpos = particle.xpos + k3[0]
        ypos = particle.ypos + k3[1]
        zpos = particle.zpos + k3[2]

        flag, host = data_reader.find_host(particle.xpos, particle.ypos, xpos,
                ypos, particle.host_horizontal_elem)

        # Check for land boundary crossing
        while flag == -1:
            xpos, ypos = self._horiz_bc_calculator.apply(data_reader,
                    particle.xpos, particle.ypos, xpos, ypos, host)
            flag, host = data_reader.find_host(particle.xpos, particle.ypos,
                    xpos, ypos, particle.host_horizontal_elem)

        # Check for open boundary crossing
        if flag == -2: return flag

        # Impose reflecting boundary condition in z
        zmin = data_reader.get_zmin(t, xpos, ypos)
        zmax = data_reader.get_zmax(t, xpos, ypos)
        if zpos < zmin or zpos > zmax:
            zpos = self._vert_bc_calculator.apply(zpos, zmin, zmax)

        zlayer = data_reader.find_zlayer(t, xpos, ypos, zpos, host, zlayer)

        data_reader.get_velocity(t, xpos, ypos, zpos, host, zlayer, vel) 
        for i in xrange(ndim):
            k4[i] = self._time_step * vel[i]

        # Sum changes and save
        delta_X.x += (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])/6.0
        delta_X.y += (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])/6.0
        delta_X.z += (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])/6.0

        return flag

def get_num_integrator(config):
    if not config.has_option("SIMULATION", "num_integrator"):
        if config.get('GENERAL', 'log_level') == 'DEBUG':
            logger = logging.getLogger(__name__)
            logger.info('Configuation option num_integrator not found. The model '\
                'will run without advection.')
        return None
    
    # Return the specified numerical integrator.
    if config.get("SIMULATION", "num_integrator") == "RK4_2D":
        return RK4Integrator2D(config)
    elif config.get("SIMULATION", "num_integrator") == "RK4_3D":
        return RK4Integrator3D(config)
    elif config.get("SIMULATION", "num_integrator") == "None":
        return None
    else:
        raise ValueError('Unsupported numerical integration scheme.')

