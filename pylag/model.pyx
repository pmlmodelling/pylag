import sys
import logging

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from pylag.fvcom_data_reader import FVCOMDataReader
from pylag.integrator import get_num_integrator
from pylag.random_walk import get_vertical_random_walk_model
from pylag.particle_positions_reader import read_particle_initial_positions
from pylag.particle import Particle
from pylag.delta import Delta
from pylag.netcdf_logger import NetCDFLogger

from pylag.data_reader cimport DataReader
from pylag.integrator cimport NumIntegrator
from pylag.random_walk cimport VerticalRandomWalk

cdef class OPTModel:
    cdef object config
    def __init__(self, config):
        self.config = config

    def initialise(self, time):
        pass
    
    def update_reading_frame(self, time):
        pass
    
    def update(self, time):
        pass
    
    def record(self, time):
        pass
    
    def shutdown(self):
        pass
    
cdef class FVCOMOPTModel(OPTModel):
    cdef DataReader data_reader
    cdef NumIntegrator num_integrator
    cdef VerticalRandomWalk vert_rand_walk_model 
    cdef object particle_set
    cdef object data_logger

    # Grid boundary limits
    cdef DTYPE_FLOAT_t _zmin
    cdef DTYPE_FLOAT_t _zmax    

    def __init__(self, *args, **kwargs):
        super(FVCOMOPTModel, self).__init__(*args, **kwargs)
        self._zmin = self.config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmin')
        self._zmax = self.config.getfloat('OCEAN_CIRCULATION_MODEL', 'zmax')

    def initialise(self, time):     
        # Create FVCOM data reader
        self.data_reader = FVCOMDataReader(self.config)
        
        # Create numerical integrator
        self.num_integrator = get_num_integrator(self.config)

        # Create vertical random walk model
        self.vert_rand_walk_model = get_vertical_random_walk_model(self.config)

        # Create particle seed - particles stored in a list object
        self.particle_set = []

        # Read in particle initial positions from file - these will be used to
        # create the initial particle set.
        group_id, xpos, ypos, zpos_temp = read_particle_initial_positions(self.config.get('SIMULATION', 'initial_positions_file'))

        guess = None
        particles_in_domain = 0
        for group, x, y, z_temp in zip(group_id, xpos, ypos, zpos_temp):
            # Find particle host element
            if guess is not None:
                # Try local search first, then global search if this fails
                host_horizontal_elem = self.data_reader.find_host_using_local_search(x, y, guess)
                if host_horizontal_elem == -1:
                    host_horizontal_elem = self.data_reader.find_host_using_global_search(x, y)
            else:
                host_horizontal_elem = self.data_reader.find_host_using_global_search(x, y)

            if host_horizontal_elem != -1:
                in_domain = 1

                h = self.data_reader.get_bathymetry(x, y, host_horizontal_elem)

                zeta = self.data_reader.get_sea_sur_elev(time, x, y, host_horizontal_elem)

                # Set z depending on the specified coordinate system
                if self.config.get("SIMULATION", "depth_coordinates") == "cartesian":
                    # z is given as the distance below the free surface. We use this,
                    # and zeta to determine the distance below the mean free
                    # surface, which is then used to calculate sigma
                    z = z_temp + zeta
                    sigma = self._cartesian_to_sigma_coords(z, h, zeta)
                    
                elif self.config.get("SIMULATION", "depth_coordinates") == "sigma":
                    # sigma ranges from 0.0 at the sea surface to -1.0 at the 
                    # sea floor.
                    sigma = z_temp
                
                # Check that the given depth is valid
                if sigma < (-1.0 - sys.float_info.epsilon):
                    raise ValueError("Supplied depth z (= {}) lies below the sea floor (h = {}).".format(z,h))
                elif sigma > (0.0 + sys.float_info.epsilon):
                    raise ValueError("Supplied depth z (= {}) lies above the free surface (zeta = {}).".format(z,zeta))

                # Create particle
                particle = Particle(group, x, y, sigma, host_horizontal_elem, in_domain)
                self.particle_set.append(particle)

                particles_in_domain += 1

                # Use the location of the last particle to guide the search for the
                # next. This should be fast if particle initial positions are colocated.
                guess = host_horizontal_elem
            else:
                in_domain = 0
                particle = Particle(group_id=group, in_domain=in_domain)
                self.particle_set.append(particle)

        logger = logging.getLogger(__name__)
        logger.info('{} of {} particles are located in the model domain.'.format(particles_in_domain, len(self.particle_set)))

        # Data logger
        self.data_logger = NetCDFLogger(self.config, len(self.particle_set))
        
        # Write particle group ids to file
        self.data_logger.write_group_ids(group_id)

    def update_reading_frame(self, time):
        self.data_reader.update_time_dependent_vars(time)

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
            if self.particle_set[i].in_domain != -1:
                delta_X.reset()
                
                # Advection
                self.num_integrator.advect(time, self.particle_set[i], 
                        self.data_reader, delta_X)
                
                # Vertical random walk
                self.vert_rand_walk_model.random_walk(time, self.particle_set[i], 
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
                
                # Update the particle's position
                self.particle_set[i].xpos = xpos
                self.particle_set[i].ypos = ypos
                self.particle_set[i].zpos = zpos
                self.particle_set[i].host_horizontal_elem = host

    def record(self, time):
        # Write particle data to file
        data = {'xpos': [], 'ypos': [], 'zpos': [], 'h': [], 'zeta': []}
        for particle in self.particle_set:
            data['xpos'].append(particle.xpos)
            data['ypos'].append(particle.ypos)
            
            # Derived vars including depth, which is first converted to cartesian coords
            h = self.data_reader.get_bathymetry(particle.xpos, particle.ypos, particle.host_horizontal_elem)
            zeta = self.data_reader.get_sea_sur_elev(time, particle.xpos, particle.ypos, particle.host_horizontal_elem)
            z = self._sigma_to_cartesian_coords(particle.zpos, h, zeta)
            data['h'].append(h)
            data['zeta'].append(zeta)
            data['zpos'].append(z)
        self.data_logger.write(time, data)
        
    def shutdown(self):
        self.data_logger.close()
        
    def _sigma_to_cartesian_coords(self, DTYPE_FLOAT_t sigma, DTYPE_FLOAT_t h,
            DTYPE_FLOAT_t zeta):
        return zeta + sigma * (h + zeta)
    
    def _cartesian_to_sigma_coords(self, DTYPE_FLOAT_t z, DTYPE_FLOAT_t h,
            DTYPE_FLOAT_t zeta):
        return (z - zeta) / (h + zeta)

def get_model(config):
    if config.get("OCEAN_CIRCULATION_MODEL", "name") == "FVCOM":
        return FVCOMOPTModel(config)
    else:
        raise ValueError('Unsupported ocean circulation model.')
