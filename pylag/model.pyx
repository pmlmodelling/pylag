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
import numpy as np

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# Error flagging
from pylag.data_types_python import INT_INVALID
from pylag import variable_library

from pylag.numerics import get_num_method, get_global_time_step
from pylag.settling import get_settling_velocity_calculator
from pylag.exceptions import PyLagValueError, PyLagRuntimeError

from libcpp.vector cimport vector

from pylag.parameters cimport seconds_per_day, radians_to_deg, deg_to_radians
from pylag.data_reader cimport DataReader
from pylag.numerics cimport NumMethod, ParticleStateNumMethod
from pylag.settling cimport SettlingVelocityCalculator
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
    cdef SettlingVelocityCalculator settling_velocity_calculator
    cdef object extra_grid_variables
    cdef object environmental_variables
    cdef object particle_seed_smart_ptrs
    cdef object particle_smart_ptrs
    cdef vector[Particle*] particle_ptrs
    
    # Seed particle data (as read from file)
    cdef DTYPE_INT_t _n_particles
    cdef DTYPE_INT_t[:] _group_ids
    cdef DTYPE_FLOAT_t[:] _x1_positions
    cdef DTYPE_FLOAT_t[:] _x2_positions
    cdef DTYPE_FLOAT_t[:] _x3_positions

    # Copy of the global time step
    cdef DTYPE_FLOAT_t _global_time_step

    # Include a biological model?
    cdef bint use_bio_model
    cdef BioModel bio_model

    # Output flags
    cdef bint save_ocean_current_vars
    cdef bint save_surface_wind_vars
    cdef bint save_stokes_drift_vars

    # Data precision, used when constructing diagnostic variable arrays
    cdef object precision

    def __init__(self, config, data_reader):
        # Initialise config
        self.config = config

        # Initialise model data reader
        self.data_reader = data_reader

        # Create num method object
        self.num_method = get_num_method(self.config)

        # Create settling velocity calculator object
        self.settling_velocity_calculator = get_settling_velocity_calculator(self.config)

        # If depth restoring is also being used, raise a warning
        if self.settling_velocity_calculator is not None:
            try:
                depth_restoring = self.config.getboolean("SIMULATION",
                                                         "depth_restoring")
            except (configparser.NoSectionError, configparser.NoOptionError) as e:
                depth_restoring = False

            if depth_restoring == True:
                logger = logging.getLogger(__name__)
                logger.warning(f'Depth restoring (`depth_restoring = True`) is '
                               f'being used with settling. The impact '
                               f'of settling may be ignored or conflict with '
                               f'this setting. To run with settling and '
                               f'no depth restoring, set `depth_restoring = '
                               f'False`.')

            try:
                height_restoring = self.config.getboolean("SIMULATION",
                                                          "height_restoring")
            except (configparser.NoSectionError, configparser.NoOptionError) as e:
                height_restoring = False
            
            if height_restoring == True:
                logger = logging.getLogger(__name__)
                logger.warning(f'Height restoring (`height_restoring = True`) is '
                               f'being used with settling. The impact '
                               f'of settling may be ignored or conflict with '
                               f'this setting. To run with settling and '
                               f'no height restoring, set `height_restoring = '
                               f'False`.')

        # Read in the coordinate system
        coordinate_system = self.config.get("SIMULATION",
                                            "coordinate_system").strip().lower()
        if coordinate_system in ["cartesian", "geographic"]:
            self.coordinate_system = coordinate_system
        else:
            raise PyLagValueError(f"Unsupported model coordinate "
                                  f"system `{coordinate_system}`")

        # Save extra grid variables; to be returned as diagnostics
        try:
            var_names = self.config.get("OUTPUT",
                    "extra_grid_variables").strip().split(',')
            self.extra_grid_variables = \
                    [var_name.strip() for var_name in var_names]
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self.extra_grid_variables = []

        # Save ocean current variables; to be returned as diagnostics
        try:
            save_ocean_current_vars = self.config.getboolean(
                    "OUTPUT", "save_ocean_current_variables")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            save_ocean_current_vars = False
        self.save_ocean_current_vars = save_ocean_current_vars

        # Save surface wind variables; to be returned as diagnostics
        try:
            save_surface_wind_vars = self.config.getboolean(
                    "OUTPUT", "save_surface_wind_variables")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            save_surface_wind_vars = False
        self.save_surface_wind_vars = save_surface_wind_vars

        # Save Stokes drift variables; to be returned as diagnostics
        try:
            save_stokes_drift_vars = self.config.getboolean(
                    "OUTPUT", "save_stokes_drift_variables")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            save_stokes_drift_vars = False
        self.save_stokes_drift_vars = save_stokes_drift_vars

        # Save environmental variables; to be returned as diagnostics
        try:
            var_names = self.config.get("OUTPUT",
                    "environmental_variables").strip().split(',')
            self.environmental_variables = [var_name.strip() for var_name in var_names]
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self.environmental_variables = []

        self._global_time_step = get_global_time_step(self.config)

        # Are we running with biology?
        try:
            self.use_bio_model = self.config.getboolean("BIO_MODEL",
                                                        "use_bio_model")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self.use_bio_model = False

        if self.use_bio_model:
            self.bio_model = BioModel(config)

        # Initialise n_particles to zero as the seed has not been created yet
        self._n_particles = 0

        # Set precision for diagnostic variables
        try:
            precision = self.config.get("OUTPUT", "precision")
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            precision = "single"

        if precision == "single":
            self.precision = 's'
        elif precision == "double":
            self.precision = 'd'
        else:
            raise PyLagValueError(f'Unknown precision flag `{precision}')

    def set_particle_data(self, group_ids, x1_positions, x2_positions,
                          x3_positions):
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

        if self.coordinate_system == "cartesian":
            self._x1_positions = x1_positions
            self._x2_positions = x2_positions
        elif self.coordinate_system == "geographic":
            self._x1_positions = x1_positions * deg_to_radians
            self._x2_positions = x2_positions * deg_to_radians

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

                # If not starting from a restart, set z depending on the
                # specified coordinate system
                if self.config.get("SIMULATION",
                                   "initialisation_method") != "restart_file":

                    if self.config.get("SIMULATION",
                            "depth_coordinates") == "depth_below_surface":
                        z_test = particle_ptr.get_x3() + zmax

                        # Block the user from trying to start particles off
                        # above the free surface
                        if z_test > zmax:
                            raise PyLagValueError(f"Supplied depth z  "
                                                  f"(= {particle_ptr.get_x3()})"
                                                  f" lies above the free "
                                                  f"surface (zeta = {zmax}).")

                        # If z_test lies below the sea floor, move it up so
                        # that the particle starts on the sea floor
                        if z_test < zmin:
                            z_test = zmin

                        particle_ptr.set_x3(z_test)

                    elif self.config.get("SIMULATION",
                            "depth_coordinates") == "height_above_bottom":
                        z_test = particle_ptr.get_x3() + zmin

                        # Block the user from trying to start off particles
                        # below the sea floor
                        if z_test < zmin:
                            raise PyLagValueError(f"Supplied depth z "
                                    f"(= {particle_ptr.get_x3()}) lies below "
                                    f"the sea floor (h = {zmin}).")

                        # Shift down to the sea floor
                        if zmin > zmax and z_test > zmin:
                            z_test = zmin

                        # Shift down to the free surface
                        if zmax > zmin and z_test > zmax:
                            z_test = zmax

                        # x3 is given as the height above the sea floor.
                        # Use this and h (zmin) to compute z
                        particle_ptr.set_x3(z_test)

                # Determine if the host element is presently dry
                if self.data_reader.is_wet(time, particle_ptr) == 1:
                    particle_ptr.set_is_beached(0)

                    # Confirm the given depth is valid for wet cells
                    if particle_ptr.get_x3() < zmin:
                        raise PyLagValueError(f"Supplied depth z "
                                f"(= {particle_ptr.get_x3()}) lies below the "
                                f"sea floor (h = {zmin}).")
                    elif particle_ptr.get_x3() > zmax:
                        raise ValueError(f"Supplied depth z "
                                f"(= {particle_ptr.get_x3()}) lies above the "
                                f"free surface (zeta = {zmax}).")

                    # Find the host z layer
                    flag = self.data_reader.set_vertical_grid_vars(time,
                                                                   particle_ptr)

                    if flag != IN_DOMAIN:
                        raise ValueError(f"Supplied depth z "
                                f"(= {particle_ptr.get_x3()}) is not within "
                                f"the grid (h = {zmin}, zeta={zmax}).")

                else:
                    # Don't set vertical grid vars as this will fail if
                    # zeta < h. They will be set later.
                    particle_ptr.set_is_beached(1)
                
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
        for group, x1, x2, x3 in zip(self._group_ids, self._x1_positions,
                                     self._x2_positions, self._x3_positions):
            # Unique particle ID.
            id += 1

            # Create particle
            particle_smart_ptr = ParticleSmartPtr(group_id=group, x1=x1-xmin,
                                                  x2=x2-ymin, x3=x3, id=id)

            # Find particle host element
            if host_elements is not None:
                # Try a local search first using guess as a starting point
                particle_smart_ptr.set_all_host_horizontal_elems(host_elements)
                flag = self.data_reader.find_host_using_local_search(
                        particle_smart_ptr.get_ptr())
                if flag != IN_DOMAIN:
                    # Local search failed. Check to see if the particle is in a
                    # masked element. If not, do a global search.
                    if flag != IN_MASKED_ELEM:
                        flag = self.data_reader.find_host_using_global_search(
                                particle_smart_ptr.get_ptr())
            else:
                # Global search ...
                flag = self.data_reader.find_host_using_global_search(
                        particle_smart_ptr.get_ptr())

            # Initialise particle settling velocity parameters
            if self.settling_velocity_calculator is not None:
                self.settling_velocity_calculator.init_particle_settling_velocity(
                        particle_smart_ptr.get_ptr())

            # Initialise the age of the particle to 0 s.
            particle_smart_ptr.set_age(0.0)

            # Initialise bio particle properties
            if self.use_bio_model:
                if self.config.get("SIMULATION",
                        "initialisation_method") != "restart_file":
                    self.bio_model.set_initial_particle_properties(
                        particle_smart_ptr.get_ptr())
                else:
                    # TODO: Add support for bio model restarts
                    #
                    # NB This would require one to read in bio model data
                    # from the restart file and set the particle properties
                    # accordingly. This is not yet possible - only particle
                    # position data is read from restart files. So, for now,
                    # raise an error.
                    raise NotImplementedError('It is not yet possible to '
                                              'run bio models with '
                                              'restarts')

            if flag == IN_DOMAIN:
                particle_smart_ptr.get_ptr().set_in_domain(True)

                # Will the depth of the particle be restored to a fixed depth?
                try:
                    depth_restoring = self.config.getboolean("SIMULATION",
                                                             "depth_restoring")
                except (configparser.NoSectionError, configparser.NoOptionError) as e:
                    depth_restoring = False

                particle_smart_ptr.get_ptr().set_restore_to_fixed_depth(
                        depth_restoring)

                try:
                    fixed_depth_below_surface = self.config.getfloat(
                            "SIMULATION", "fixed_depth")
                except (configparser.NoSectionError, configparser.NoOptionError) as e:
                    if depth_restoring == True:
                        raise PyLagRuntimeError(f'Depth restoring is being used '
                                f'but a restoring depth has not been given. '
                                f'You can choose a restoring depth with the '
                                f'configuration option `fixed_depth`.')
                    fixed_depth_below_surface = FLOAT_ERR

                particle_smart_ptr.get_ptr().set_fixed_depth(
                        fixed_depth_below_surface)

                # Will the height of the particle be restored to a fixed height?
                try:
                    height_restoring = self.config.getboolean("SIMULATION",
                                                              "height_restoring")
                except (configparser.NoSectionError, configparser.NoOptionError) as e:
                    height_restoring = False
                
                # Block attempts to both restore to a fixed depth and height
                if height_restoring == True and depth_restoring == True:
                    raise PyLagValueError(f'Both depth and height restoring '
                            f'are being used. Only one restoring method can be '
                            f'used at a time. To run with height restoring, '
                            f'set `height_restoring = True` and '
                            f'`depth_restoring = False`, and vice versa. If you '
                            f'want to run with neither, set both to `False`.')
                
                particle_smart_ptr.get_ptr().set_restore_to_fixed_height(
                        height_restoring)

                try:
                    fixed_height_above_bottom = self.config.getfloat(
                            "SIMULATION", "fixed_height")
                except (configparser.NoSectionError, configparser.NoOptionError) as e:
                    if height_restoring == True:
                        raise PyLagRuntimeError(f'Height restoring is being used '
                                f'but a restoring height has not been given. '
                                f'You can choose a restoring height with the '
                                f'configuration option `fixed_height`.')
                    fixed_height_above_bottom = FLOAT_ERR

                particle_smart_ptr.get_ptr().set_fixed_height(
                        fixed_height_above_bottom)

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
            raise PyLagRuntimeError(f'All seed particles lie outside of '
                                    f'the model domain!')

        # Save the total number of particles
        self._n_particles = len(self.particle_seed_smart_ptrs)

        if self.config.get('GENERAL', 'log_level') == 'DEBUG':
            logger = logging.getLogger(__name__)
            logger.info(f'{particles_in_domain} of '
                        f'{len(self.particle_seed_smart_ptrs)} particles are '
                        f'located in the model domain.')
    
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
                # Update the settling velocity
                if self.settling_velocity_calculator is not None:
                    self.settling_velocity_calculator.set_particle_settling_velocity(
                            self.data_reader, time, particle_ptr)

                # Step the model forward in time
                flag = self.num_method.step(self.data_reader, time, particle_ptr)

                if flag == IN_DOMAIN:
                    pass
                elif (flag == OPEN_BDY_CROSSED or
                      flag == BOTTOM_BDY_CROSSED or
                      flag == IS_PERMANENTLY_BEACHED):
                    particle_ptr.set_in_domain(False)
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

        # Velocity components
        cdef DTYPE_FLOAT_t ocean_velocity[3]
        cdef DTYPE_FLOAT_t wind_velocity[2]
        cdef DTYPE_FLOAT_t stokes_drift_velocity[2]

        # Grid offsets
        if self.coordinate_system == "cartesian":
            xmin = self.data_reader.get_xmin()
            ymin = self.data_reader.get_ymin()
        elif self.coordinate_system == "geographic":
            xmin = 0.0
            ymin = 0.0

        # Initialise diagnostic variable arrays
        # -------------------------------------
        diags = {}

        # x1, x2, x3
        for var_name in ['x1', 'x2', 'x3']:
            # Get the variable name which is specific to the coordinate system
            _var_name = variable_library.get_coordinate_variable_name(
                self.coordinate_system, var_name)

            # Add to the diagnostics dictionary
            diags[var_name] = np.empty(self._n_particles,
                dtype=variable_library.get_data_type(_var_name, self.precision))

        # Host elements
        for grid_name in self.get_grid_names():
            diags[f'host_{grid_name}'] = np.empty(self._n_particles,
                dtype=variable_library.get_integer_type(self.precision))

        # Status variables
        for var_name in ['error_status', 'in_domain', 'is_beached']:
            dtype = variable_library.get_data_type(var_name, self.precision)
            diags[var_name] = np.empty(self._n_particles, dtype)

        # Number of boundary encounters
        dtype = variable_library.get_data_type('land_boundary_encounters',
                                               self.precision)
        diags['land_boundary_encounters'] = np.empty(self._n_particles, dtype)

        # Extra grid variables
        for var_name in self.extra_grid_variables:
            dtype = variable_library.get_data_type(var_name, self.precision)
            diags[var_name] = np.empty(self._n_particles, dtype)

        # Ocean current variables
        if self.save_ocean_current_vars:
            for var_name in ['uo', 'vo', 'wo']:
                dtype = variable_library.get_data_type(var_name, self.precision)
                diags[var_name] = np.empty(self._n_particles, dtype)

        # Surface wind variables
        if self.save_surface_wind_vars:
            for var_name in ['u10', 'v10']:
                dtype = variable_library.get_data_type(var_name, self.precision)
                diags[var_name] = np.empty(self._n_particles, dtype)

        # Stokes drift variables
        if self.save_stokes_drift_vars:
            for var_name in ['usd', 'vsd']:
                dtype = variable_library.get_data_type(var_name, self.precision)
                diags[var_name] = np.empty(self._n_particles, dtype)

        # Environmental variables
        for var_name in self.environmental_variables:
            dtype = variable_library.get_data_type(var_name, self.precision)
            diags[var_name] = np.empty(self._n_particles, dtype)

        # Bio model variables
        if self.use_bio_model:
            dtype = variable_library.get_data_type('age', self.precision)
            diags['age'] = np.empty(self._n_particles, dtype)

            dtype = variable_library.get_data_type('is_alive', self.precision)
            diags['is_alive'] = np.empty(self._n_particles, dtype)

        # Fill the diagnostic arrays with data
        # ------------------------------------
        for i, particle_smart_ptr in enumerate(self.particle_smart_ptrs):

            # Particle location data
            if self.coordinate_system == "cartesian":
                diags['x1'][i] = particle_smart_ptr.x1 + xmin
                diags['x2'][i] = particle_smart_ptr.x2 + ymin
            elif self.coordinate_system == "geographic":
                diags['x1'][i] = particle_smart_ptr.x1 * radians_to_deg
                diags['x2'][i] = particle_smart_ptr.x2 * radians_to_deg

            diags['x3'][i] = particle_smart_ptr.x3

            # Grid specific host element data
            host_elements = particle_smart_ptr.get_all_host_horizontal_elems()
            for grid_name, host in host_elements.items():
                diags[f'host_{grid_name}'][i] = host

            # Status variables
            diags['is_beached'][i] = particle_smart_ptr.is_beached
            diags['in_domain'][i] = particle_smart_ptr.in_domain
            diags['error_status'][i] = particle_smart_ptr.status

            # Extra grid variables
            if 'h' in self.extra_grid_variables:
                if particle_smart_ptr.in_domain:
                    h = self.data_reader.get_zmin(time,
                            particle_smart_ptr.get_ptr())
                else:
                    h = variable_library.get_invalid_value(diags['h'].dtype)

                diags['h'][i] = h

            if 'zeta' in self.extra_grid_variables:
                if particle_smart_ptr.in_domain:
                    zeta = self.data_reader.get_zmax(time,
                            particle_smart_ptr.get_ptr())
                else:
                    zeta = variable_library.get_invalid_value(
                            diags['zeta'].dtype)

                diags['zeta'][i] = zeta

            # Ocean current variables
            if self.save_ocean_current_vars:
                if particle_smart_ptr.in_domain:
                    self.data_reader.get_velocity(time, particle_smart_ptr.get_ptr(), ocean_velocity)
                    diags['uo'][i] = ocean_velocity[0]
                    diags['vo'][i] = ocean_velocity[1]
                    diags['wo'][i] = ocean_velocity[2]
                else:
                    diags['uo'][i] = variable_library.get_invalid_value(
                        diags['uo'].dtype)
                    diags['vo'][i] = variable_library.get_invalid_value(
                        diags['vo'].dtype)
                    diags['wo'][i] = variable_library.get_invalid_value(
                        diags['wo'].dtype)
            
            # Surface wind variables
            if self.save_surface_wind_vars:
                if particle_smart_ptr.in_domain:
                    self.data_reader.get_ten_meter_wind_velocity(time,
                            particle_smart_ptr.get_ptr(), wind_velocity)
                    diags['u10'][i] = wind_velocity[0]
                    diags['v10'][i] = wind_velocity[1]
                else:
                    diags['u10'][i] = variable_library.get_invalid_value(
                        diags['u10'].dtype)
                    diags['v10'][i] = variable_library.get_invalid_value(
                        diags['v10'].dtype)

            # Stokes drift variables
            if self.save_stokes_drift_vars:
                if particle_smart_ptr.in_domain:
                    self.data_reader.get_surface_stokes_drift_velocity(time, 
                            particle_smart_ptr.get_ptr(), stokes_drift_velocity)
                    diags['usd'][i] = stokes_drift_velocity[0]
                    diags['vsd'][i] = stokes_drift_velocity[1]
                else:
                    diags['usd'][i] = variable_library.get_invalid_value(
                        diags['usd'].dtype)
                    diags['vsd'][i] = variable_library.get_invalid_value(
                        diags['vsd'].dtype)

            # Number of boundary encounters
            diags['land_boundary_encounters'][i] = \
                particle_smart_ptr.land_boundary_encounters

            # Environmental variables
            for var_name in self.environmental_variables:
                if particle_smart_ptr.in_domain:
                    var = self.data_reader.get_environmental_variable(var_name,
                            time, particle_smart_ptr.get_ptr())
                else:
                    var = variable_library.get_invalid_value(
                        diags[var_name].dtype)

                diags[var_name][i] = var

            # Bio model variables
            if self.use_bio_model:
                diags['age'][i] = particle_smart_ptr.age / seconds_per_day
                diags['is_alive'][i] = particle_smart_ptr.is_alive

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

            # Add grid offsets
            all_particle_data['x1'] = [x1 + xmin for x1 in all_particle_data['x1']]
            all_particle_data['x2'] = [x2 + ymin for x2 in all_particle_data['x2']]
        elif self.coordinate_system == "geographic":
            all_particle_data['x1'] = [x1 * radians_to_deg for x1 in all_particle_data['x1']]
            all_particle_data['x2'] = [x2 * radians_to_deg for x2 in all_particle_data['x2']]

        return all_particle_data

