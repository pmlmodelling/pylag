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
        self._time_step = config.getfloat('SIMULATION', 'time_step')
        self._zmin = config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmin')
        self._zmax = config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmax')
        self._vertical_coordinate_system = config.get('OCEAN_CIRCULATION_MODEL', 'vertical_coordinate_system')

    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader):
        """
        AR0 vertical random walk. This method is an extension of the
        NaiveVerticalRandomWalk model to situations in which the eddy diffusivity
        is not necessarily homogenous. The extension prevenets the articifical
        accumulation of particles in regions of low diffusivity. See Visser (1997)
        and Ross and Sharples (2004) for a more detailed discussion.
        
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
        
        # Temporary containers for z positions slightly offset from the current
        # position - used in k and dk_dz calculations
        cdef DTYPE_FLOAT_t zpos_incremented, zpos_offset
        
        # The vertical eddy diffusiviy
        #     k1 - at tha particle's location
        #     k2 - at a small distance away from the particle's location
        #     k3 - at the location the particle is advected too
        #     dk_dz - gradient in k computed from D1 and D2
        cdef DTYPE_FLOAT_t k1, k2, k3, dk_dz
        
        # Change in position (m)
        cdef DTYPE_FLOAT_t dz_advection, dz_random, dz
        
        # Change in position (sigma)
        cdef DTYPE_FLOAT_t ds_advection, ds_random, ds
        
        # Depth increment used in the computation of dk/dz
        cdef DTYPE_FLOAT_t z_increment = (self._zmax - self._zmin)/1000.0
        
        # Use the negative of depth_increment at the top of the water column
        if ((particle.zpos + z_increment) > self._zmax):
            z_increment = -z_increment

        # Compute the vertical eddy diffusivity at the particle's current location
        t = time
        xpos = particle.xpos
        ypos = particle.ypos
        zpos = particle.zpos
        host = particle.host_horizontal_elem        
        k1 = data_reader.get_vertical_eddy_diffusivity(t, xpos, ypos, zpos, host)
        
        # Compute the vertical eddy diffusiviy a small distance away from the particle's
        # current location
        zpos_incremented = zpos + z_increment
        k2 = data_reader.get_vertical_eddy_diffusivity(t, xpos, ypos, zpos_incremented, host)

        # The following calculations depend on the nature of the grid.
        # TODO leave this as if logic, subclass, or look into using data_reader
        # for coordinate transformations?
        if self._vertical_coordinate_system == "cartesian":
            # Compute an approximate value for the gradient in the vertical eddy
            # diffusivity at the particles current location.
            dk_dz = (k2 - k1) / z_increment

            # Compute the advective component of the random walk
            dz_advection = dk_dz * self._time_step

            # Compute the vertical eddy diffusivity at a position offset by a distance
            # dz_advection/2. TODO Although the diffusivity will generally be
            # lower at the boundaries, and this terms acts in the direction of 
            # increasing diffusivity, is there a chance this could walk us outside
            # of the domain?
            zpos_offset = zpos + 0.5 * dz_advection
            k3 = data_reader.get_vertical_eddy_diffusivity(t, xpos, ypos, zpos_offset, host)

            # Compute the random component of the particle's motion
            dz_random = sqrt(2.0*k3*self._time_step) * random.gauss(1.0)

            # Change in position (in meters)
            dz = dz_advection + dz_random
            
            # New z (in m)
            zpos = zpos + dz

        elif self._vertical_coordinate_system == "sigma":
            # Compute the water depth
            h = data_reader.get_bathymetry(xpos, ypos, host)
            zeta = data_reader.get_sea_sur_elev(t, xpos, ypos, host)
            depth = h + zeta

            # Compute an approximate value for the gradient in the vertical eddy
            # diffusivity at the particle's current location.
            dk_ds = (k2 - k1) / z_increment

            # Compute the advective component of the random walk (carteisan coords)
            dz_advection = dk_ds * self._time_step / depth

            # Compute the advective component of the random walk (sigma coords)
            ds_advection = dz_advection / depth

            # Compute the vertical eddy diffusivity at a position offset by a distance
            # ds_advection/2.
            zpos_offset = zpos + 0.5 * ds_advection
            k3 = data_reader.get_vertical_eddy_diffusivity(t, xpos, ypos, zpos_offset, host)

            # Compute the random component of the particle's motion
            ds_random = sqrt(2.0*k3*self._time_step) * random.gauss(1.0) / depth

            # Change in position (in sigma)
            ds = ds_advection + ds_random
            
            # New z (in sigma)
            zpos = zpos + ds
        
        # Apply reflecting boundary conditions
        if zpos < self._zmin:
            zpos = self._zmin + self._zmin - zpos
        elif zpos > self._zmax:
            zpos = self._zmax + self._zmax - zpos
        
        # Update the particle's position
        particle.zpos = zpos

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
