include "constants.pxi"

import logging

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from pylag.numerics import get_num_method
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
    cdef object environmental_variables
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

        # Save a list of environmental variables to be returned as diagnostics
        try:
            var_names = self.config.get("OUTPUT", "environmental_variables").strip().split(',')
            self.environmental_variables = [var_name.strip() for var_name in var_names]
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self.environmental_variables = []

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
            
            # Set vertical grid vars for particles that lie inside the domain
            if particle_ptr.in_domain == True:

                # Grid limits for error checking
                zmin = self.data_reader.get_zmin(time, particle_ptr)
                zmax = self.data_reader.get_zmax(time, particle_ptr)

                # If not starting from a restart, set z depending on the specified coordinate system
                if self.config.get("SIMULATION", "initialisation_method") != "restart_file":

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

        # Initialise lists
        diags = {'x1': [], 'x2': [], 'x3': [], 'host_horizontal_elem': [],
                 'h': [], 'zeta': [], 'is_beached': [], 'in_domain': [],
                 'status': []}

        # Initialise env var lists
        for var_name in self.environmental_variables:
            diags[var_name] = []

        for particle_ptr in self.particle_ptrs:
            # Particle location data
            diags['x1'].append(particle_ptr.xpos + xmin)
            diags['x2'].append(particle_ptr.ypos + ymin)
            diags['x3'].append(particle_ptr.zpos)
            diags['host_horizontal_elem'].append(particle_ptr.host_horizontal_elem)

            # Particle state data
            diags['is_beached'].append(particle_ptr.is_beached)
            diags['in_domain'].append(particle_ptr.in_domain)
            diags['status'].append(particle_ptr.status)
            
            # Grid variables
            if particle_ptr.in_domain:
                h = self.data_reader.get_zmin(time, particle_ptr)
                zeta = self.data_reader.get_zmax(time, particle_ptr)
            else:
                h = FLOAT_ERR
                zeta = FLOAT_ERR
            diags['h'].append(h)
            diags['zeta'].append(zeta)

            # Environmental variables
            for var_name in self.environmental_variables:
                if particle_ptr.in_domain:
                    var = self.data_reader.get_environmental_variable(var_name, time, particle_ptr)
                    diags[var_name].append(var)
                else:
                    diags[var_name].append(FLOAT_ERR)

        return diags

    def get_particle_data(self):
        """ Get particle data

        Pool, sort and return data describing the basic state of each particle. The
        main purpose of this function is to assist with the creation of restart files.
        
        Returns:
        --------
        all_particle_data : dict
            Dictionary holding particle data.
        """
        # Initialise a dictionary of particle data with empty lists
        all_particle_data = {}
        for key in self.particle_smart_ptrs[0].get_particle_data().keys():
            all_particle_data[key] = []

        # Add data for the remaining particles
        for particle_smart_ptr in self.particle_smart_ptrs:
            particle_data = particle_smart_ptr.get_particle_data()
            for key, value in list(particle_data.items()):
                all_particle_data[key].append(value)

        # Add grid offsets
        if 'x1' in all_particle_data.keys():
            x1_min = self.data_reader.get_xmin()
            x1_positions = all_particle_data['x1']
            all_particle_data['x1'] = [x1 + x1_min for x1 in x1_positions]

        if 'x2' in all_particle_data.keys():
            x2_min = self.data_reader.get_xmin()
            x2_positions = all_particle_data['x2']
            all_particle_data['x2'] = [x2 + x2_min for x2 in x2_positions]

        return all_particle_data

