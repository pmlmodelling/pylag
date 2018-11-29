include "constants.pxi"

import logging

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from pylag.numerics import get_num_method
from pylag.particle_positions_reader import read_particle_initial_positions
from pylag.particle import ParticleSmartPtr

from libcpp.vector cimport vector

from pylag.data_reader cimport DataReader
from pylag.math cimport sigma_to_cartesian_coords, cartesian_to_sigma_coords
from pylag.numerics cimport NumMethod
from pylag.particle cimport Particle, ParticleSmartPtr, copy, to_string

cdef class OPTModel:
    """ Offline Particle Tracking Model
    
    This class provides a generic interface for setting up and running a
    particle tracking simulation. This includes:
    
    1) Creating a particle seed
    2) Reading input data
    3) Updating particle positions
    4) Returning diagnostic data
    """
    cdef object config
    cdef DataReader data_reader
    cdef NumMethod num_method
    cdef object particle_seed_smart_ptrs
    cdef object particle_smart_ptrs
    cdef vector[Particle*] particle_ptrs
    
    # Seed particle data (as read from file)
    cdef DTYPE_INT_t[:] _group_ids
    cdef DTYPE_FLOAT_t[:] _x_positions
    cdef DTYPE_FLOAT_t[:] _y_positions
    cdef DTYPE_FLOAT_t[:] _z_positions

    def __init__(self, config, data_reader):
        """ Initialise class data members

        Parameters:
        -----------
        config : SafeConfigParser
            Configuration obect.

        data_reader : DataReader
            Data reader object.
        """
        # Initialise config
        self.config = config

        # Initialise model data reader
        self.data_reader = data_reader

        # Create num method object
        self.num_method = get_num_method(self.config)

    def set_particle_data(self, group_ids, x_positions, y_positions, z_positions):
        """Initialise memory views for data describing the particle seed.

        Parameters:
        -----------
        group_ids : ndarray, int
            Particle groups IDs.

        x_positions : ndarray, float
            Particle x-positions.

        y_positions : ndarray, float
            Particle y-positions.

        z_positions : ndarray, float
            Particle z-positions.
        """
        self._group_ids = group_ids
        self._x_positions = x_positions
        self._y_positions = y_positions
        self._z_positions = z_positions

    def setup_input_data_access(self, start_datetime, end_datetime):
        """Setup access to FVCOM time dependent variables.

        Parameters:
        -----------
        start_datime : Datetime
            The simulation start date and time.

        end_datime : Datetime
            The simulation end date and time.
        """
        self.data_reader.setup_data_access(start_datetime, end_datetime)

    def read_input_data(self, time):
        """Update reading frames for FVCOM data fields.

        Parameters:
        -----------
        time : float
            The current time.
        """
        self.data_reader.read_data(time)

    def seed(self, time):
        """Set particle positions equal to those of the particle seed.
        
        Create the particle seed if it has not been created already. Make
        an `active' copy of the particle seed.

        Parameters:
        -----------
        time : float
            The current time.
        """
        # Grid boundary limits
        cdef DTYPE_FLOAT_t zmin
        cdef DTYPE_FLOAT_t zmax
        
        # Particle raw pointer
        cdef Particle* particle_ptr

        if self.particle_seed_smart_ptrs is None:
            self._create_seed()

        # Destroy the current active particle set and all pointers to it
        self.particle_smart_ptrs = []
        self.particle_ptrs.clear()

        # Set all time dependent quantities while adding seed particles to the
        # active particle set
        for particle_seed_smart_ptr in self.particle_seed_smart_ptrs:
            particle_smart_ptr = copy(particle_seed_smart_ptr)
            particle_ptr = particle_smart_ptr.get_ptr()
            
            # Set z depending on the specified coordinate system
            zmin = self.data_reader.get_zmin(time, particle_ptr)
            zmax = self.data_reader.get_zmax(time, particle_ptr)
            if self.config.get("SIMULATION", "depth_coordinates") == "depth_below_surface":
                # z_temp is given as the depth below the moving free surface
                # Use this and zeta (zmax) to compute z
                particle_ptr.zpos = particle_ptr.zpos + zmax

            elif self.config.get("SIMULATION", "depth_coordinates") == "height_above_bottom":
                # z_temp is given as the height above the sea floor. Use this
                # and h (zmin) to compute z
                particle_ptr.zpos = particle_ptr.zpos + zmin

            # Check that the given depth is valid
            if particle_ptr.zpos < zmin:
                raise ValueError("Supplied depth z (= {}) lies below the sea floor (h = {}).".format(particle_ptr.zpos, zmin))
            elif particle_ptr.zpos > zmax:
                raise ValueError("Supplied depth z (= {}) lies above the free surface (zeta = {}).".format(particle_ptr.zpos, zmax))

            # Find the host z layer
            flag = self.data_reader.set_vertical_grid_vars(time, particle_ptr)

            if flag != IN_DOMAIN:
                raise ValueError("Supplied depth z (= {}) is not within the grid (h = {}, zeta={}).".format(particle_ptr.zpos, zmin, zmax))

            # Determine if the host element is presently dry
            if self.data_reader.is_wet(time, particle_ptr.host_horizontal_elem) == 1:
                particle_ptr.is_beached = 0
            else:
                particle_ptr.is_beached = 1
            
            self.particle_smart_ptrs.append(particle_smart_ptr)
            self.particle_ptrs.push_back(particle_ptr)

    def _create_seed(self):
        """Create the particle seed.
        
        Create the particle seed. The particle seed is distinct from the active
        particle set, whose positions are updated throughout the model
        simulation. The separation allows the model to reseeded at later times;
        for example, during an ensemble simulation.
        
        Note that while zpos is set here, this value is ultimately overwritten
        in seed(), which accounts for the current position of the moving free
        surface and the specified depth coordinates. Other time dependent
        quantities (e.g. is_beached) are also set in seed().
        """
        # Particle raw pointer
        cdef ParticleSmartPtr particle_smart_ptr
        
        # Create particle seed - particles stored in a list object
        self.particle_seed_smart_ptrs = []

        # Grid offsets
        xmin = self.data_reader.get_xmin()
        ymin = self.data_reader.get_ymin()

        guess = None
        particles_in_domain = 0
        id = 0
        for group, x, y, z in zip(self._group_ids, self._x_positions, self._y_positions, self._z_positions):
            # Unique particle ID.
            id += 1

            # Create particle
            particle_smart_ptr = ParticleSmartPtr(group_id=group, xpos=x-xmin, ypos=y-ymin, zpos=z, id=id)

            # Find particle host element
            if guess is not None:
                # Try a local search first
                flag = self.data_reader.find_host_using_local_search(particle_smart_ptr.get_ptr(), guess)
                if flag != IN_DOMAIN:
                    # Local search failed - try a global search
                    flag = self.data_reader.find_host_using_global_search(particle_smart_ptr.get_ptr())
            else:
                # Global search ...
                flag = self.data_reader.find_host_using_global_search(particle_smart_ptr.get_ptr())

            if flag == IN_DOMAIN:
                particle_smart_ptr.get_ptr().in_domain = True

                # Add particle to the particle set
                self.particle_seed_smart_ptrs.append(particle_smart_ptr)

                particles_in_domain += 1

                # Use the location of the last particle to guide the search for the
                # next. This should be fast if particle initial positions are colocated.
                guess = particle_smart_ptr.host_horizontal_elem
            else:
                particle_smart_ptr.get_ptr().in_domain = False
                self.particle_seed_smart_ptrs.append(particle_smart_ptr)

        if particles_in_domain == 0:
            raise RuntimeError('All seed particles lie outside of the model domain!')

        if self.config.get('GENERAL', 'log_level') == 'DEBUG':
            logger = logging.getLogger(__name__)
            logger.info('{} of {} particles are located in the model domain.'.format(particles_in_domain, len(self.particle_seed_smart_ptrs)))
    
    cpdef update(self, DTYPE_FLOAT_t time):
        """ Compute and update each particle's position.

        Cycle over the particle set, updating the position of only those
        particles that remain in the model domain. If a particle has beached
        update its status.
        
        Parameters:
        -----------
        time : float
            The current time.
        """
        cdef Particle* particle_ptr
        cdef DTYPE_INT_t flag

        for particle_ptr in self.particle_ptrs:
            if particle_ptr.in_domain:
                if particle_ptr.is_beached == 0:
                    flag = self.num_method.step(self.data_reader, time, particle_ptr)

                    if flag == OPEN_BDY_CROSSED: 
                        particle_ptr.in_domain = False
                        continue
                    elif flag == BDY_ERROR:
                        s = to_string(particle_ptr)
                        msg = "WARNING BDY_ERROR encountered at time {} \n\n"\
                              "PyLag failed to successfully update the position of a particle \n"\
                              "resulting in a BDY_ERROR flag being returned. This can occur \n"\
                              "when a particle has crossed a land boundary and the model \n"\
                              "fails to apply the specified land boundary condition. The \n"\
                              "particle will be flagged as having entered an error state \n"\
                              "and its position will no longer be updated. The following \n"\
                              "information may be used to study the failure in more detail. \n\n"\
                              "{}".format(time, s)
                        print msg
                    
                        particle_ptr.in_domain = False
                        particle_ptr.status = 1
                        continue
                
                if self.data_reader.is_wet(time, particle_ptr.host_horizontal_elem) == 1:
                    particle_ptr.is_beached = 0
                else:
                    particle_ptr.is_beached = 1
                
    def get_diagnostics(self, time):
        """ Get particle diagnostics
        
        Parameters:
        -----------
        time : float
            The current time.
        
        Returns:
        --------
        diags : dict
            Dictionary holding particle diagnostic data.
        """
        cdef Particle* particle_ptr

        # Grid offsets
        xmin = self.data_reader.get_xmin()
        ymin = self.data_reader.get_ymin()

        diags = {'xpos': [], 'ypos': [], 'zpos': [], 'host_horizontal_elem': [],
                 'h': [], 'zeta': [], 'is_beached': [], 'in_domain': [],
                 'status': []}
        for particle_ptr in self.particle_ptrs:
            diags['xpos'].append(particle_ptr.xpos + xmin)
            diags['ypos'].append(particle_ptr.ypos + ymin)
            diags['zpos'].append(particle_ptr.zpos)
            diags['host_horizontal_elem'].append(particle_ptr.host_horizontal_elem)
            diags['is_beached'].append(particle_ptr.is_beached)
            diags['in_domain'].append(particle_ptr.in_domain)
            diags['status'].append(particle_ptr.status)
            
            # Environmental environmental variables
            h = self.data_reader.get_zmin(time, particle_ptr)
            zeta = self.data_reader.get_zmax(time, particle_ptr)
            diags['h'].append(h)
            diags['zeta'].append(zeta)
        return diags
