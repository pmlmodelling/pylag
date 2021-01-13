"""
Model objects are used to calculate changes in particle states during
the course of a typical simulation. They hold and manage all particle data.

Note
----
model is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

include "constants.pxi"

import logging

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# Error flagging
from pylag.data_types_python import INT_INVALID, FLOAT_INVALID
from pylag.variable_library import get_invalid_value

from pylag.numerics import get_num_method, get_global_time_step

from libcpp.vector cimport vector

from pylag.data_reader cimport DataReader
from pylag.math cimport sigma_to_cartesian_coords, cartesian_to_sigma_coords
from pylag.numerics cimport NumMethod, ParticleStateNumMethod
from pylag.particle cimport Particle
from pylag.particle_cpp_wrapper cimport ParticleSmartPtr, copy, to_string
from pylag.bio_model cimport BioModel


cdef class OPTModel:
    """ Offline Particle Tracking Model
    
    This class provides a generic interface for setting up and running a
    particle tracking simulation. This includes:
    
    1) Creating a particle seed
    2) Reading input data
    3) Updating particle positions
    4) Returning diagnostic data

    Parameters
    ----------
    config : configparser.ConfigParser
        PyLag configuration object

    data_reader : pylag.data_reader.DataReader
        PyLag data reader
    """
    cdef object config
    cdef object coordinate_system
    cdef DataReader data_reader
    cdef NumMethod num_method
    cdef ParticleStateNumMethod particle_state_num_method
    cdef object environmental_variables
    cdef object particle_seed_smart_ptrs
    cdef object particle_smart_ptrs
    cdef vector[Particle*] particle_ptrs
    
    # Seed particle data (as read from file)
    cdef DTYPE_INT_t[:] _group_ids
    cdef DTYPE_FLOAT_t[:] _x1_positions
    cdef DTYPE_FLOAT_t[:] _x2_positions
    cdef DTYPE_FLOAT_t[:] _x3_positions

    # Copy of the global time step
    cdef DTYPE_FLOAT_t _global_time_step

    # Include a biological model?
    cdef bint use_bio_model
    cdef BioModel bio_model

    def __init__(self, config, data_reader):
        # Initialise config
        self.config = config

        # Initialise model data reader
        self.data_reader = data_reader

        # Create num method object
        self.num_method = get_num_method(self.config)

        # Read in the coordinate system
        coordinate_system = self.config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system").strip().lower()
        if coordinate_system in ["cartesian", "geographic"]:
            self.coordinate_system = coordinate_system
        else:
            raise ValueError("Unsupported model coordinate system `{}'".format(coordinate_system))

        # Save a list of environmental variables to be returned as diagnostics
        try:
            var_names = self.config.get("OUTPUT", "environmental_variables").strip().split(',')
            self.environmental_variables = [var_name.strip() for var_name in var_names]
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self.environmental_variables = []

        self._global_time_step = get_global_time_step(self.config)

        # Are we running with biology?
        try:
            self.use_bio_model = self.config.getboolean("BIO_MODEL", "use_bio_model")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self.use_bio_model = False

        if self.use_bio_model:
            self.bio_model = BioModel()

    def set_particle_data(self, group_ids, x1_positions, x2_positions, x3_positions):
        """Initialise memory views for data describing the particle seed.

        Parameters
        ----------
        group_ids : ndarray, int
            Particle groups IDs.

        x1_positions : ndarray, float
            Particle x-positions.

        x2_positions : ndarray, float
            Particle y-positions.

        x3_positions : ndarray, float
            Particle z-positions.
        """
        self._group_ids = group_ids
        self._x1_positions = x1_positions
        self._x2_positions = x2_positions
        self._x3_positions = x3_positions

    def setup_input_data_access(self, start_datetime, end_datetime):
        """Setup access to FVCOM time dependent variables.

        Parameters
        ----------
        start_datime : Datetime
            The simulation start date and time.

        end_datime : Datetime
            The simulation end date and time.
        """
        self.data_reader.setup_data_access(start_datetime, end_datetime)

    def read_input_data(self, time):
        """Update reading frames for FVCOM data fields.

        Parameters
        ----------
        time : float
            The current time.
        """
        self.data_reader.read_data(time)

    def get_grid_names(self):
        """ Return a list of grid names
        """
        return self.data_reader.get_grid_names()

    def seed(self, time):
        """Set particle positions equal to those of the particle seed.
        
        Create the particle seed if it has not been created already. Make
        an `active` copy of the particle seed.

        Parameters
        ----------
        time : float
            The current time.
        """
        # Grid boundary limits
        cdef DTYPE_FLOAT_t zmin
        cdef DTYPE_FLOAT_t zmax

        # Depth used in intermediate calculations
        cdef DTYPE_FLOAT_t z_test

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
            if particle_ptr.get_in_domain() == True:

                # Grid limits for error checking
                zmin = self.data_reader.get_zmin(time, particle_ptr)
                zmax = self.data_reader.get_zmax(time, particle_ptr)

                # If not starting from a restart, set z depending on the specified coordinate system
                if self.config.get("SIMULATION", "initialisation_method") != "restart_file":

                    if self.config.get("SIMULATION", "depth_coordinates") == "depth_below_surface":
                        z_test = particle_ptr.get_x3() + zmax

                        # Block the user from trying to start particles off above the free surface
                        if z_test > zmax:
                            raise ValueError("Supplied depth z (= {}) lies above the free surface (zeta = {}).".format(particle_ptr.get_x3(), zmax))

                        # If z_test lies below the sea floor, move it up so that the particle starts on the sea floor
                        if z_test < zmin:
                            z_test = zmin

                        particle_ptr.set_x3(z_test)

                    elif self.config.get("SIMULATION", "depth_coordinates") == "height_above_bottom":
                        z_test = particle_ptr.get_x3() + zmin

                        # Block the user from trying to start off particles below the sea floor
                        if z_test < zmin:
                            raise ValueError("Supplied depth z (= {}) lies below the sea floor (h = {}).".format(particle_ptr.get_x3(), zmin))

                        # Shift down to the sea floor
                        if zmin > zmax and z_test > zmin:
                            z_test = zmin

                        # Shift down to the free surface
                        if zmax > zmin and z_test > zmax:
                            z_test = zmax

                        # x3 is given as the height above the sea floor. Use this and h (zmin) to compute z
                        particle_ptr.set_x3(z_test)

                # Determine if the host element is presently dry
                if self.data_reader.is_wet(time, particle_ptr) == 1:
                    particle_ptr.set_is_beached(0)

                    # Confirm the given depth is valid for wet cells
                    if particle_ptr.get_x3() < zmin:
                        raise ValueError("Supplied depth z (= {}) lies below the sea floor (h = {}).".format(particle_ptr.get_x3(), zmin))
                    elif particle_ptr.get_x3() > zmax:
                        raise ValueError("Supplied depth z (= {}) lies above the free surface (zeta = {}).".format(particle_ptr.get_x3(), zmax))

                    # Find the host z layer
                    flag = self.data_reader.set_vertical_grid_vars(time, particle_ptr)

                    if flag != IN_DOMAIN:
                        raise ValueError("Supplied depth z (= {}) is not within the grid (h = {}, zeta={}).".format(particle_ptr.get_x3(), zmin, zmax))

                else:
                    # Don't set vertical grid vars as this will fail if zeta < h. They will be set later.
                    particle_ptr.set_is_beached(1)

                # Initialise the age of the particle to 0 s.
                particle_ptr.set_age(0.0)

                # Initialise bio particle properties
                if self.use_bio_model:
                    if self.config.get("SIMULATION", "initialisation_method") != "restart_file":
                        self.bio_model.set_initial_particle_properties(particle_ptr)
                    else:
                        raise NotImplementedError('It is not yet possible to run bio models with restarts')

            self.particle_smart_ptrs.append(particle_smart_ptr)
            self.particle_ptrs.push_back(particle_ptr)

    def _create_seed(self):
        """Create the particle seed.
        
        Create the particle seed. The particle seed is distinct from the active
        particle set, whose positions are updated throughout the model
        simulation. The separation allows the model to reseeded at later times;
        for example, during an ensemble simulation.
        
        Note that while x3 is set here, this value is ultimately overwritten
        in seed(), which accounts for the current position of the moving free
        surface and the specified depth coordinates. Other time dependent
        quantities (e.g. is_beached) are also set in seed().
        """
        # Particle raw pointer
        cdef ParticleSmartPtr particle_smart_ptr
        
        # Create particle seed - particles stored in a list object
        self.particle_seed_smart_ptrs = []

        # Grid offsets
        if self.coordinate_system == "cartesian":
            xmin = self.data_reader.get_xmin()
            ymin = self.data_reader.get_ymin()
        elif self.coordinate_system == "geographic":
            xmin = 0.0
            ymin = 0.0

        host_elements = None
        particles_in_domain = 0
        id = 0
        for group, x1, x2, x3 in zip(self._group_ids, self._x1_positions, self._x2_positions, self._x3_positions):
            # Unique particle ID.
            id += 1

            # Create particle
            particle_smart_ptr = ParticleSmartPtr(group_id=group, x1=x1-xmin, x2=x2-ymin, x3=x3, id=id)

            # Find particle host element
            if host_elements is not None:
                # Try a local search first using guess as a starting point
                particle_smart_ptr.set_all_host_horizontal_elems(host_elements)
                flag = self.data_reader.find_host_using_local_search(particle_smart_ptr.get_ptr())
                if flag != IN_DOMAIN:
                    # Local search failed - try a global search
                    flag = self.data_reader.find_host_using_global_search(particle_smart_ptr.get_ptr())
            else:
                # Global search ...
                flag = self.data_reader.find_host_using_global_search(particle_smart_ptr.get_ptr())

            if flag == IN_DOMAIN:
                particle_smart_ptr.get_ptr().set_in_domain(True)

                # Add particle to the particle set
                self.particle_seed_smart_ptrs.append(particle_smart_ptr)

                particles_in_domain += 1

                # Use the location of the last particle to guide the search for the
                # next. This should be fast if particle initial positions are collocated.
                host_elements = particle_smart_ptr.get_all_host_horizontal_elems()
            else:
                # Flag host elements as being invalid
                for grid_name in self.get_grid_names():
                    particle_smart_ptr.set_host_horizontal_elem(grid_name, INT_INVALID)
                particle_smart_ptr.get_ptr().set_in_domain(False)
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
        
        Parameters
        ----------
        time : float
            The current time.
        """
        cdef Particle* particle_ptr
        cdef DTYPE_INT_t flag
        cdef DTYPE_FLOAT_t new_time

        # Time following the update. Used to set the particle's age.
        new_time = time + self._global_time_step

        # Update particle biological properties
        if self.use_bio_model:
            for particle_ptr in self.particle_ptrs:
                if particle_ptr.get_in_domain() and particle_ptr.get_is_alive():
                    self.bio_model.update(self.data_reader, time, particle_ptr)

        for particle_ptr in self.particle_ptrs:
            if particle_ptr.get_in_domain():
                flag = self.num_method.step(self.data_reader, time, particle_ptr)

                if flag == OPEN_BDY_CROSSED:
                    particle_ptr.set_in_domain(False)
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
                    print(msg)

                    particle_ptr.set_in_domain(False)
                    particle_ptr.set_status(1)
                    continue

            # Update the particle's age
            particle_ptr.set_age(new_time)

    def get_diagnostics(self, time):
        """ Get particle diagnostics
        
        Parameters
        ----------
        time : float
            The current time.
        
        Returns
        -------
        diags : dict
            Dictionary holding particle diagnostic data.
        """
        cdef ParticleSmartPtr particle_smart_ptr

        # Grid offsets
        if self.coordinate_system == "cartesian":
            xmin = self.data_reader.get_xmin()
            ymin = self.data_reader.get_ymin()
        elif self.coordinate_system == "geographic":
            xmin = 0.0
            ymin = 0.0

        # Initialise lists
        diags = {'x1': [], 'x2': [], 'x3': [], 'h': [], 'zeta': [], 'is_beached': [],
                 'in_domain': [], 'status': []}

        # Initialise host data
        for grid_name in self.get_grid_names():
            diags['host_{}'.format(grid_name)] = []

        # Initialise env var lists
        for var_name in self.environmental_variables:
            diags[var_name] = []

        for particle_smart_ptr in self.particle_smart_ptrs:

            # Particle location data
            diags['x1'].append(particle_smart_ptr.x1 + xmin)
            diags['x2'].append(particle_smart_ptr.x2 + ymin)
            diags['x3'].append(particle_smart_ptr.x3)

            # Grid specific host element data
            host_elements = particle_smart_ptr.get_all_host_horizontal_elems()
            for grid_name, host in host_elements.items():
                diags['host_{}'.format(grid_name)].append(host)

            # Particle state data
            diags['is_beached'].append(particle_smart_ptr.is_beached)
            diags['in_domain'].append(particle_smart_ptr.in_domain)
            diags['status'].append(particle_smart_ptr.status)
            
            # Grid variables
            if particle_smart_ptr.in_domain:
                h = self.data_reader.get_zmin(time, particle_smart_ptr.get_ptr())
                zeta = self.data_reader.get_zmax(time, particle_smart_ptr.get_ptr())
            else:
                h = get_invalid_value('h')
                zeta = get_invalid_value('zeta')
            diags['h'].append(h)
            diags['zeta'].append(zeta)

            # Environmental variables
            for var_name in self.environmental_variables:
                if particle_smart_ptr.in_domain:
                    var = self.data_reader.get_environmental_variable(var_name, time, particle_smart_ptr.get_ptr())
                    diags[var_name].append(var)
                else:
                    diags[var_name].append(get_invalid_value(var_name))

        return diags

    def get_particle_data(self):
        """ Get particle data

        Pool, sort and return data describing the basic state of each particle. The
        main purpose of this function is to assist with the creation of restart files.
        
        Returns
        -------
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

        # Grid offsets
        if self.coordinate_system == "cartesian":
            xmin = self.data_reader.get_xmin()
            ymin = self.data_reader.get_ymin()
        elif self.coordinate_system == "geographic":
            xmin = 0.0
            ymin = 0.0

        # Add grid offsets
        all_particle_data['x1'] = [x1 + xmin for x1 in all_particle_data['x1']]
        all_particle_data['x2'] = [x2 + ymin for x2 in all_particle_data['x2']]

        return all_particle_data

