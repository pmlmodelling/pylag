import logging

from libc.math cimport sqrt

cimport pylag.random as random

from pylag.boundary_conditions import get_vert_boundary_condition_calculator

cdef class LSM:
    cdef apply(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X):
        pass

 # Vertical Lagrangian Stochastic Models
 # -------------------------------------
 
cdef class OneDLSM(LSM):
    cdef apply(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X):
        pass

cdef class NaiveOneDLSM(OneDLSM):
    def __init__(self, config):
        self._time_step = config.getfloat('SIMULATION', 'time_step')

    cdef apply(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X):
        """
        Apply the naive LSM in 1D, which is taken to be the vertical dimension.
        This method should only be used when the vertical eddy diffusivity field
        is homogenous. When it is not, particles will accumulate in regions of
        low diffusivity. In this situation a more sophisticated scheme should be
        used (e.g. VisserOneDLSM).
        
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
        N/A
        """
        # The vertical eddy diffusiviy
        cdef DTYPE_FLOAT_t D

        D = data_reader.get_vertical_eddy_diffusivity(time, particle)
        
        # Change in position
        delta_X.z += sqrt(2.0*D*self._time_step) * random.gauss(0.0, 1.0)

cdef class VisserOneDLSM(OneDLSM):
    def __init__(self, config):
        self._time_step = config.getfloat('SIMULATION', 'time_step')
        
        self._vert_bc_calculator = get_vert_boundary_condition_calculator(config)

    cdef apply(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X):
        """
        Apply the Visser LSM in 1D, which is assumed to be the vertical dimension.
        The model includes a deterministic advective terms that counteracts the
        tendency for particles to accumulate in regions of low diffusivity
        (c.f. the Naive LSM). See Visser (1997) and Ross and Sharples (2004) 
        for a more detailed discussion.
        
        Parameters:
        -----------
        time: float
            The current time.

        particle: *Particle
            Pointer to a Particle object.

        data_reader: object of type DataReader
            A DataReader object. Used for reading the vertical eddy diffusivity.
            
        Returns:
        --------
        N/A
        """
        # Temporary particle object
        cdef Particle _particle
        
        # Temporary containers for the particle's location
        cdef DTYPE_FLOAT_t zmin, zmax
        
        # Temporary containers for z position offset from the current position -
        # used in dk_dz calculations
        cdef DTYPE_FLOAT_t zpos_offset
        
        # The vertical eddy diffusiviy
        #     k - at the location the particle is advected to
        #     dk_dz - gradient in k
        cdef DTYPE_FLOAT_t k, dk_dz
        
        # Change in position (units can be m, or sigma)
        cdef DTYPE_FLOAT_t dz_advection, dz_random, dz

        # Create a copy of particle
        _particle = particle[0]

        # Compute an approximate value for the gradient in the vertical eddy
        # diffusivity at the particle's current location.
        dk_dz = data_reader.get_vertical_eddy_diffusivity_derivative(time, &_particle)

        # Compute the advective component of the lagrangian stochastic model
        dz_advection = dk_dz * self._time_step
        
        # Compute the vertical eddy diffusivity at a position offset by a distance
        # dz_advection/2. Apply reflecting boundary condition if the computed
        # offset falls outside of the model domain
        zpos_offset = _particle.zpos + 0.5 * dz_advection
        zmin = data_reader.get_zmin(time, &_particle)
        zmax = data_reader.get_zmax(time, &_particle)
        if zpos_offset < zmin or zpos_offset > zmax:
            zpos_offset = self._vert_bc_calculator.apply(zpos_offset, zmin, zmax)

        _particle.zpos = zpos_offset
        data_reader.set_vertical_grid_vars(time, &_particle)
        k = data_reader.get_vertical_eddy_diffusivity(time, &_particle)

        # Compute the random component of the particle's motion
        dz_random = sqrt(2.0*k*self._time_step) * random.gauss(0.0, 1.0)

        # Change in position
        delta_X.z = dz_advection + dz_random

cdef class VisserSplineOneDLSM(OneDLSM):
    def __init__(self, config):
        pass

    cdef apply(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X):
        pass


# Horizontal Lagrangian Stochastic Models
# ---------------------------------------
 
cdef class HorizontalLSM(LSM):
    cdef apply(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X):
        pass  

cdef class ConstantHorizontalLSM(HorizontalLSM):
    def __init__(self, config):
        self._time_step = config.getfloat('SIMULATION', 'time_step')
        self._kh = config.getfloat("SIMULATION", "horizontal_eddy_diffusivity_constant")

    cdef apply(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X):
        """
        Horizontal lagrangian stochastic model using a constant value for the
        horizontal eddy diffusivity that is provided as a parameter.
        
        Parameters:
        -----------
        time: float
            The current time.

        particle: object of type Particle
            A Particle object. The object's z position will be updated.

        data_reader: object of type DataReader
            A DataReader object. Used for reading the vertical eddy diffusivity.
            
        Returns:
        --------
        N/A
        """
        # Change in position
        delta_X.x += sqrt(2.0*self._kh*self._time_step) * random.gauss(0.0, 1.0)
        delta_X.y += sqrt(2.0*self._kh*self._time_step) * random.gauss(0.0, 1.0)

cdef class NaiveHorizontalLSM(HorizontalLSM):
    def __init__(self, config):
        self._time_step = config.getfloat('SIMULATION', 'time_step')

    cdef apply(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X):
        """
        Naive horizontal lagrangian stochastic model. This method should only be
        used when the horizontal eddy diffusivity is both homogenous and
        isotropic in x and y.
        
        Parameters:
        -----------
        time: float
            The current time.

        particle: object of type Particle
            A Particle object. The object's z position will be updated.

        data_reader: object of type DataReader
            A DataReader object. Used for reading the vertical eddy diffusivity.
            
        Returns:
        --------
        N/A
        """
        # The horizontal eddy diffusiviy
        cdef DTYPE_FLOAT_t kh

        # The vertical eddy diffusivity at the particle's current location
        kh = data_reader.get_horizontal_eddy_diffusivity(time, particle)
        
        # Change in position
        delta_X.x += sqrt(2.0*kh*self._time_step) * random.gauss(0.0, 1.0)
        delta_X.y += sqrt(2.0*kh*self._time_step) * random.gauss(0.0, 1.0)

cdef class AR0HorizontalLSM(HorizontalLSM):
    def __init__(self, config):
        pass

    cdef apply(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X):
        pass

def get_vertical_lsm(config):
    if not config.has_option("SIMULATION", "vertical_lsm"):
        if config.get('GENERAL', 'log_level') == 'DEBUG':
            logger = logging.getLogger(__name__)
            logger.info('Configuation option vertical_lsm not found. '\
            'The model will run without vertical lagrangian stochastic model.')
        return None

    # Return the specified vertical lagrangian stochastic model.
    if config.get("SIMULATION", "vertical_lsm") == "naive":
        return NaiveOneDLSM(config)
    elif config.get("SIMULATION", "vertical_lsm") == "visser":
        return VisserOneDLSM(config)
    elif config.get("SIMULATION", "vertical_lsm") == "none":
        return None
    else:
        raise ValueError('Unrecognised vertical lagrangian stochastic model.')
    
def get_horizontal_lsm(config):
    if not config.has_option("SIMULATION", "horizontal_lsm"):
        if config.get('GENERAL', 'log_level') == 'DEBUG':
            logger = logging.getLogger(__name__)
            logger.info('Configuation option horizontal_lsm not found. '\
                'The model will run without horizontal lagrangian stochastic model.')
        return None

    # Return the specified horizontal lagrangian stochastic model.
    if config.get("SIMULATION", "horizontal_lsm") == "constant":
        return ConstantHorizontalLSM(config)
    elif config.get("SIMULATION", "horizontal_lsm") == "naive":
        return NaiveHorizontalLSM(config)
    elif config.get("SIMULATION", "horizontal_lsm") == "none":
        return None
    else:
        raise ValueError('Unrecognised horizontal lagrangian stochastic model.')
