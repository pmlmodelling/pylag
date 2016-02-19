from libc.math cimport sqrt

cimport pylag.random as random

cdef class RandomWalk:
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader):
        pass

 # Vertical Random Walks
 # ---------------------
 
cdef class VerticalRandomWalk(RandomWalk):
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader):
        pass    

cdef class NaiveVerticalRandomWalk(VerticalRandomWalk):
    def __init__(self, config):
        self._time_step = config.getfloat('SIMULATION', 'time_step')
        self._zmin = config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmin')
        self._zmax = config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmax')
        self._vertical_coordinate_system = config.get('OCEAN_CIRCULATION_MODEL', 'vertical_coordinate_system')

    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader):
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
        
        # Change in position (in meters)
        dz = sqrt(2.0*D*self._time_step) * random.gauss(1.0)
        
        # Update zpos. This step is dependent on the vertical coordinate system
        # used. dz has units of length. If zpos is in nomralised sigma 
        # coordiantes one must first divide by the water column depth
        # before updating zpos.
        if self._vertical_coordinate_system == "cartesian":
            zpos = zpos + dz
        elif self._vertical_coordinate_system == "sigma":
            h = data_reader.get_bathymetry(xpos, ypos, host)
            zeta = data_reader.get_zeta(t, zpos, ypos, host)
            zpos = zpos + dz/(h+zeta)
        else:
            raise ValueError('Vertical coordinate system not recognised.')
        
        # Apply reflecting boundary conditions
        if zpos < self._zmin:
            zpos = self._zmin + self._zmin - zpos
        elif zpos > self._zmax:
            zpos = self._zmax + self._zmax - zpos
        
        # Update the particle's position
        particle.zpos = zpos

cdef class AR0VerticalRandomWalk(VerticalRandomWalk):
    def __init__(self, config):
        pass

    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader):
        pass

cdef class AR0VerticalRandomWalkWithSpline(VerticalRandomWalk):
    def __init__(self, config):
        pass

    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader):
        pass


# Horizontal Random Walks
# -----------------------
 
cdef class HorizontalRandomWalk(RandomWalk):
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader):
        pass  

cdef class NaiveVHorizontalRandomWalk(HorizontalRandomWalk):
    def __init__(self, config):
        pass

    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader):
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

    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader):
        pass
