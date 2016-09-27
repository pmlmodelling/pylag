import sys
import logging
import copy

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from pylag.fvcom_data_reader import FVCOMDataReader
from pylag.integrator import get_num_integrator
from pylag.random_walk import get_vertical_random_walk_model, get_horizontal_random_walk_model
from pylag.particle_positions_reader import read_particle_initial_positions
from pylag.particle import Particle
from pylag.delta import Delta

from pylag.data_reader cimport DataReader
from pylag.integrator cimport NumIntegrator
from pylag.random_walk cimport VerticalRandomWalk, HorizontalRandomWalk

cdef class OPTModel:
    def set_particle_data(self, group_ids, x_positions, y_positions, z_positions):
        pass
    
    def setup_input_data_access(self, start_datetime, end_datetime):
        pass

    def read_input_data(self, time):
        pass

    def seed(self, time):
        pass
    
    def update(self, time):
        pass
    
    def get_diagnostics(self, time):
        pass
    
cdef class FVCOMOPTModel(OPTModel):
    cdef object config
    cdef DataReader data_reader
    cdef NumIntegrator num_integrator
    cdef VerticalRandomWalk vert_rand_walk_model
    cdef HorizontalRandomWalk horiz_rand_walk_model
    cdef object particle_seed
    cdef object particle_set

    # Grid boundary limits
    cdef DTYPE_FLOAT_t _zmin
    cdef DTYPE_FLOAT_t _zmax
    
    # Seed particle data (as read from file)
    cdef DTYPE_INT_t[:] _group_ids
    cdef DTYPE_FLOAT_t[:] _x_positions
    cdef DTYPE_FLOAT_t[:] _y_positions
    cdef DTYPE_FLOAT_t[:] _z_positions

    def __init__(self, config, data_reader, *args, **kwargs):
        super(FVCOMOPTModel, self).__init__(*args, **kwargs)

        # Initialise config
        self.config = config

        # Initialise model data reader
        self.data_reader = data_reader
        
        # Create numerical integrator
        self.num_integrator = get_num_integrator(self.config)

        # Create vertical random walk model
        self.vert_rand_walk_model = get_vertical_random_walk_model(self.config)

        # Create horizontal random walk model
        self.horiz_rand_walk_model = get_horizontal_random_walk_model(self.config)
        
        # Vertical min and max values - used to check for boundary crossings
        self._zmin = self.config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmin')
        self._zmax = self.config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmax')

    def set_particle_data(self, group_ids, x_positions, y_positions, z_positions):
        """Initialise memory views for data describing the particle seed.
        
        """
        self._group_ids = group_ids
        self._x_positions = x_positions
        self._y_positions = y_positions
        self._z_positions = z_positions

    def setup_input_data_access(self, start_datetime, end_datetime):
        """Setup access to FVCOM time dependent variables.

        """
        self.data_reader.setup_data_access(start_datetime, end_datetime)

    def read_input_data(self, time):
        """Update reading frames for FVCOM data fields.
        
        """
        self.data_reader.read_data(time)

    def seed(self, time=None):
        """Set particle positions equal to those of the particle seed.
        
        Create the particle seed if it has not been created already. Make
        an `active' copy of the particle seed.
        """
        if self.particle_seed is None:
            self._create_seed(time)

        self.particle_set = copy.deepcopy(self.particle_seed)

    def _create_seed(self, time):
        """Create the particle seed.
        
        Create the particle seed using the supplied arguments. Initialise
        `particle_set' using `particle_seed'. A separate copy of the particle
        seed is stored so that the model can be reseeded at a later time, as
        required.
        """
        # Create particle seed - particles stored in a list object
        self.particle_seed = []

        guess = None
        particles_in_domain = 0
        for group, x, y, z_temp in zip(self._group_ids, self._x_positions, self._y_positions, self._z_positions):
            # Find particle host element
            if guess is not None:
                # Try a local search first
                host_horizontal_elem = self.data_reader.find_host_using_local_search(x, y, guess)
                if host_horizontal_elem < 0:
                    # Local search failed - try a global search
                    host_horizontal_elem = self.data_reader.find_host_using_global_search(x, y)
            else:
                # Global search ...
                host_horizontal_elem = self.data_reader.find_host_using_global_search(x, y)

            if host_horizontal_elem >= 0:
                in_domain = True

                # Set z depending on the specified coordinate system
                if self.config.get("SIMULATION", "depth_coordinates") == "cartesian":
                    # z is given as the distance below the free surface. We use
                    # this and zeta to determine the distance below the mean
                    # free surface, which is then used with h to calculate sigma
                    h = self.data_reader.get_bathymetry(x, y, host_horizontal_elem)
                    zeta = self.data_reader.get_sea_sur_elev(time, x, y, host_horizontal_elem)

                    z = z_temp + zeta
                    sigma = self._cartesian_to_sigma_coords(z, h, zeta)
                    
                elif self.config.get("SIMULATION", "depth_coordinates") == "sigma":
                    # sigma ranges from 0.0 at the sea surface to -1.0 at the 
                    # sea floor.
                    sigma = z_temp
                
                # Check that the given depth is valid
                if sigma < (self._zmin - sys.float_info.epsilon):
                    raise ValueError("Supplied depth z (= {}) lies below the sea floor (h = {}).".format(z,h))
                elif sigma > (self._zmax + sys.float_info.epsilon):
                    raise ValueError("Supplied depth z (= {}) lies above the free surface (zeta = {}).".format(z,zeta))

                # Create particle
                particle = Particle(group, x, y, sigma, host_horizontal_elem, in_domain)
                self.particle_seed.append(particle)

                particles_in_domain += 1

                # Use the location of the last particle to guide the search for the
                # next. This should be fast if particle initial positions are colocated.
                guess = host_horizontal_elem
            else:
                in_domain = False
                particle = Particle(group_id=group, in_domain=in_domain)
                self.particle_seed.append(particle)

        if self.config.getboolean('GENERAL', 'full_logging'):
            logger = logging.getLogger(__name__)
            logger.info('{} of {} particles are located in the model domain.'.format(particles_in_domain, len(self.particle_seed)))

    def update(self, DTYPE_FLOAT_t time):
        """
        Compute the net effect of resolved and unresolved processes on particle
        motion in the interval t -> t + dt. Resolved velocities are used to
        advect particles. A random displacement model is used to model the
        effect of unresolved (subgrid scale) processes. Particle displacements
        are first stored and accumulated in an object of type Delta before
        then being used to update a given particle's position. For now, if a
        particle crosses a lateral boundary its motion is temporarily arrested.
        Reflecting boundary conditions are applied at the bottom and surface
        boundaries.
        """
        cdef DTYPE_FLOAT_t xpos, ypos, zpos
        cdef DTYPE_INT_t host
        cdef DTYPE_INT_t i, n_particles

        # Object for storing position deltas resulting from advection and random
        # displacement in the interval t -> t + dt
        delta_X = Delta()
        
        # Cycle over the particle set, updating the position of only those
        # particles that remain in the model domain
        n_particles = len(self.particle_set)
        for i in xrange(n_particles):
            if self.particle_set[i].in_domain:
                delta_X.reset()
                
                # Advection
                if self.num_integrator is not None:
                    self.num_integrator.advect(time, self.particle_set[i], 
                            self.data_reader, delta_X)
                
                # Vertical random walk
                if self.vert_rand_walk_model is not None:
                    self.vert_rand_walk_model.random_walk(time, self.particle_set[i], 
                            self.data_reader, delta_X)

                # Horizontal random walk
                if self.horiz_rand_walk_model is not None:
                    self.horiz_rand_walk_model.random_walk(time, self.particle_set[i], 
                            self.data_reader, delta_X)  
                
                # Check for boundary crossings. TODO For now, arrest particle 
                # motion.
                xpos = self.particle_set[i].xpos + delta_X.x
                ypos = self.particle_set[i].ypos + delta_X.y
                zpos = self.particle_set[i].zpos + delta_X.z
                host = self.data_reader.find_host(xpos, ypos, self.particle_set[i].host_horizontal_elem)
                if host == -1: continue

                # Apply reflecting surface/bottom boundary conditions
                if zpos < self._zmin:
                    zpos = self._zmin + self._zmin - zpos
                elif zpos > self._zmax:
                    zpos = self._zmax + self._zmax - zpos

                # Check for valid zpos
                if zpos < (self._zmin - sys.float_info.epsilon):
                    raise ValueError("New zpos (= {}) lies below the sea floor.".format(zpos))
                elif zpos > (self._zmax + sys.float_info.epsilon):
                    raise ValueError("New zpos (= {}) lies above the free surface.".format(zpos))                
                
                # Update the particle's position
                self.particle_set[i].xpos = xpos
                self.particle_set[i].ypos = ypos
                self.particle_set[i].zpos = zpos
                self.particle_set[i].host_horizontal_elem = host

    def get_diagnostics(self, time):
        diags = {'xpos': [], 'ypos': [], 'zpos': [], 'host_horizontal_elem': [], 'h': [], 'zeta': []}
        for particle in self.particle_set:
            diags['xpos'].append(particle.xpos)
            diags['ypos'].append(particle.ypos)
            diags['host_horizontal_elem'].append(particle.host_horizontal_elem)            
            
            # Derived vars including depth, which is first converted to cartesian coords
            h = self.data_reader.get_bathymetry(particle.xpos, particle.ypos, particle.host_horizontal_elem)
            zeta = self.data_reader.get_sea_sur_elev(time, particle.xpos, particle.ypos, particle.host_horizontal_elem)
            z = self._sigma_to_cartesian_coords(particle.zpos, h, zeta)
            diags['h'].append(h)
            diags['zeta'].append(zeta)
            diags['zpos'].append(z)
        return diags
        
    def _sigma_to_cartesian_coords(self, DTYPE_FLOAT_t sigma, DTYPE_FLOAT_t h,
            DTYPE_FLOAT_t zeta):
        return zeta + sigma * (h + zeta)
    
    def _cartesian_to_sigma_coords(self, DTYPE_FLOAT_t z, DTYPE_FLOAT_t h,
            DTYPE_FLOAT_t zeta):
        return (z - zeta) / (h + zeta)
