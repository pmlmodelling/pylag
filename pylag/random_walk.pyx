import logging

from libc.math cimport sqrt

cimport pylag.random as random

cdef class RandomWalk:
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X):
        pass

 # Vertical Random Walks
 # ---------------------
 
cdef class VerticalRandomWalk(RandomWalk):
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X):
        pass    

cdef class NaiveVerticalRandomWalk(VerticalRandomWalk):
    def __init__(self, config):
        self._time_step = config.getfloat('SIMULATION', 'time_step')
        self._zmin = config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmin')
        self._zmax = config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmax')

    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X):
        """
        Naive vertical random walk. This method should only be used when
        the vertical eddy diffusivity is homogenous. When it is not, particles
        will accumulate in regions of low diffusivity. In this situation a more
        sophisticated scheme should be used (e.g. AR0VerticalRandomWalk).
        
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
        # Temporary containers
        cdef DTYPE_FLOAT_t t, xpos, ypos, zpos
        cdef DTYPE_INT_t host
        
        # The vertical eddy diffusiviy
        cdef DTYPE_FLOAT_t D        
        
        # Change in position (m)
        cdef DTYPE_FLOAT_t dz
        
        # Water depth and sea surface elevation (only used for sigma coordinates)
        cdef DTYPE_FLOAT_t h, zeta

        # The vertical eddy diffusivity at the particle's current location
        t = time
        xpos = particle.xpos
        ypos = particle.ypos
        zpos = particle.zpos
        host = particle.host_horizontal_elem        
        D = data_reader.get_vertical_eddy_diffusivity(t, xpos, ypos, zpos, host)
        
        # Change in position
        delta_X.z += sqrt(2.0*D*self._time_step) * random.gauss(1.0)

cdef class AR0VerticalRandomWalk(VerticalRandomWalk):
    def __init__(self, config):
        self._time_step = config.getfloat('SIMULATION', 'time_step')
        self._zmin = config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmin')
        self._zmax = config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmax')

    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X):
        """
        AR0 vertical random walk. This method is an extension of the
        NaiveVerticalRandomWalk model to situations in which the eddy diffusivity
        is not necessarily homogenous. The extension prevents the articifical
        accumulation of particles in regions of low diffusivity. See Visser (1997)
        and Ross and Sharples (2004) for a more detailed discussion.
        
        The method is independent of the vertical coordiante system - DataReader
        objects are expected to provide diffusivities that are consistent in
        terms of their units.
        
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
        # Temporary containers for the particle's location
        cdef DTYPE_FLOAT_t t, xpos, ypos, zpos
        cdef DTYPE_INT_t host
        
        # Temporary containers for z position offset from the current position -
        # used in dk_dz calculations
        cdef DTYPE_FLOAT_t zpos_offset
        
        # The vertical eddy diffusiviy
        #     k - at the location the particle is advected too
        #     dk_dz - gradient in k
        cdef DTYPE_FLOAT_t k, dk_dz
        
        # Change in position (units can be m, or sigma)
        cdef DTYPE_FLOAT_t dz_advection, dz_random, dz

        # Compute the vertical eddy diffusivity at the particle's current location
        t = time
        xpos = particle.xpos
        ypos = particle.ypos
        zpos = particle.zpos
        host = particle.host_horizontal_elem        

        # Compute an approximate value for the gradient in the vertical eddy
        # diffusivity at the particle's current location.
        dk_dz = data_reader.get_vertical_eddy_diffusivity_derivative(t, xpos, ypos, zpos, host)

        # Compute the advective component of the random walk
        dz_advection = dk_dz * self._time_step
        
        # Compute the vertical eddy diffusivity at a position offset by a distance
        # dz_advection/2. Apply reflecting boundary condition if the computed
        # offset falls outside of the model domain
        zpos_offset = zpos + 0.5 * dz_advection
        if zpos_offset < self._zmin:
            zpos_offset = self._zmin + self._zmin - zpos_offset
        elif zpos_offset > self._zmax:
            zpos_offset = self._zmax + self._zmax - zpos_offset
        k = data_reader.get_vertical_eddy_diffusivity(t, xpos, ypos, zpos_offset, host)

        # Compute the random component of the particle's motion
        dz_random = sqrt(2.0*k*self._time_step) * random.gauss(1.0)

        # Change in position
        delta_X.z = dz_advection + dz_random

cdef class AR0VerticalRandomWalkWithSpline(VerticalRandomWalk):
    def __init__(self, config):
        pass

    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X):
        pass


# Horizontal Random Walks
# -----------------------
 
cdef class HorizontalRandomWalk(RandomWalk):
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X):
        pass  

cdef class NaiveVHorizontalRandomWalk(HorizontalRandomWalk):
    def __init__(self, config):
        pass

    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X):
        """
        Naive horizontal random walk. This method should only be used when
        the horizontal eddy diffusivity is both homogenous and isotropic in x
        and y.
        
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
        pass

cdef class AR0HorizontalRandomWalk(HorizontalRandomWalk):
    def __init__(self, config):
        pass

    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader, Delta delta_X):
        pass

def get_vertical_random_walk_model(config):
    if not config.has_option("SIMULATION", "vertical_random_walk_model"):
        logger = logging.getLogger(__name__)
        logger.info('Configuation option vertical_random_walk_model not found. '\
        'The model will run without vertical random walk.')
        return None

    # Return the specified vertical random walk model.
    if config.get("SIMULATION", "vertical_random_walk_model") == "naive":
        return NaiveVerticalRandomWalk(config)
    elif config.get("SIMULATION", "vertical_random_walk_model") == "AR0":
        return AR0VerticalRandomWalk(config)
    elif config.get("SIMULATION", "vertical_random_walk_model") == "None":
        return None
    else:
        raise ValueError('Unrecognised vertical random walk model.')
