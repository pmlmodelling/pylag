"""
Regridder objects can be used to regrid input data.

Note
----
regridder is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

include "constants.pxi"

import numpy as np

from libcpp.vector cimport vector

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# Error flagging
from pylag.data_types_python import INT_INVALID, FLOAT_INVALID
from pylag import variable_library

# PyLag python imports
from pylag.arakawa_a_data_reader import ArakawaADataReader
from pylag.fvcom_data_reader import FVCOMDataReader
from pylag.roms_data_reader import ROMSDataReader
from pylag.mediator import SerialMediator

# PyLag cython imports
from pylag.parameters cimport seconds_per_day, radians_to_deg, deg_to_radians
from pylag.particle cimport Particle
from pylag.data_reader cimport DataReader
from pylag.particle_cpp_wrapper cimport ParticleSmartPtr

cdef class Regridder:
    """ A Regridder object

    An object of type Regridder can be used to interpolate data onto
    a new given grid. Regridder's leverage PyLag's interpolation
    routines to facilitate the task.

    Parameters
    ----------
    config : configparser.ConfigParser
        PyLag configuration object

    lons : 1D ND Array
        1D array of longitudes to interpolate data to in deg. E.

    lats : 1D ND Array
        1D array of latitudes to interpolate data to in deg. N.

    depths : 1D NDArray
        1D array of depths to interpolate data to.

    datetime_start : datetime.datetime
        The earliest date and time at which regridded data is desired.
        Requested here as PyLag will first check that the given date and
        time is covered by the input data.

    datetime_end : datetime.datetime
        The latest date and time at which regridded data is desired.
        Requested here as PyLag will first check that the given date and
        time is covered by the input data.

    Attributes
    ----------
    datetime_start : datetime.datetime
        The earliest date and time at which regridded data is desired.
        Requested here as PyLag will first check that the given date and
        time is covered by the input data.

    datetime_end : datetime.datetime
        The latest date and time at which regridded data is desired.
        Requested here as PyLag will first check that the given date and
        time is covered by the input data.

    data_reader : pylag.data_reader.DataReader
        A PyLag DataReader object.
    """
    cdef object config
    cdef object coordinate_system
    cdef object datetime_start
    cdef object datetime_end
    cdef DTYPE_FLOAT_t[:] lons
    cdef DTYPE_FLOAT_t[:] lats
    cdef DTYPE_FLOAT_t[:] depths
    cdef DTYPE_INT_t n_points
    cdef object particle_smart_ptrs
    cdef vector[Particle*] particle_ptrs
    cdef DataReader data_reader

    def __init__(self, config, lons, lats, depths, datetime_start, datetime_end):
        # Save config
        self.config = config

        # Save reference times
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end

        # Intialise the data reader
        if config.get("OCEAN_CIRCULATION_MODEL", "name") == "ArakawaA":
            mediator = SerialMediator(config, datetime_start, datetime_end)
            self.data_reader = ArakawaADataReader(config, mediator)
        elif config.get("OCEAN_CIRCULATION_MODEL", "name") == "FVCOM":
            mediator = SerialMediator(config, datetime_start, datetime_end)
            self.data_reader = FVCOMDataReader(config, mediator)
        elif config.get("OCEAN_CIRCULATION_MODEL", "name") == "ROMS":
            mediator = SerialMediator(config, datetime_start, datetime_end)
            self.data_reader = ROMSDataReader(config, mediator)
        else:
            raise ValueError('Unsupported ocean circulation model.')

        self.coordinate_system = config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system")

        # Generate particle set, including interpolation coefficients
        print('Computing weights ', end='... ')
        self.set_spatial_coordinates(lons, lats, depths)
        print('done!')

    def get_grid_names(self):
        """ Return a list of grid names
        """
        return self.data_reader.get_grid_names()

    def set_spatial_coordinates(self, lons, lats, depths):
        """ Set coordinates to interpolate to

        Internally, a PyLag particle set is created with one particle located
        at each of the specified coordinates. Interpolation coefficients are
        set at the same time. Some of these may be time dependent
        (e.g. vertical coordinates, which may vary with changes in
        free surface elevation. Thus, when interpolating data at different
        time points, these should be reset as required.

        Parameters
        ----------
        lons : 1D NumPy array
            1D array of longitudes to interpolate data to.

        lats : 1D NumPy array
            1D array of latitudes to interpolate data to.

        depths : 1D NumPy array
            1D array of depths to interpolate data to. Depths should be
            negative down relative to the sea surface.
        """
         # Particle raw pointer
        cdef ParticleSmartPtr particle_smart_ptr

        cdef DTYPE_FLOAT_t x1, x2, x3
        cdef DTYPE_INT_t n_points
        cdef DTYPE_INT_t i

        # Create particle seed - particles stored in a list object
        self.particle_smart_ptrs = []

        # Confirm we have been provided 1D arrays
        assert lons.ndim == 1, 'Longitude array is not 1D'
        assert lats.ndim == 1, 'Latitude array is not 1D'
        assert depths.ndim == 1, 'Depth array is not 1D'

        # Confirm arrays have the same shape
        assert lons.shape[0] == lats.shape[0], 'Array lengths do not match (n_lon={}, n_lat={})'.format(lons.shape[0], lats.shape[0])
        assert lons.shape[0] == depths.shape[0], 'Array lengths do not match (n_lon={}, n_depth={})'.format(lons.shape[0], depths.shape[0])

        # Save spatial coordinates
        self.lons = (lons * deg_to_radians).astype(DTYPE_FLOAT)
        self.lats = (lats * deg_to_radians).astype(DTYPE_FLOAT)
        self.depths = depths.astype(DTYPE_FLOAT)

        # The number of points at which data will  be interpolated to
        self.n_points = self.lons.shape[0]

        # Check that the coordinate system is supported
        # TODO if support for Cartesian input grids is added, will need to apply xmin and xmax offsets
        self.coordinate_system = self.config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system").strip().lower()
        if not self.coordinate_system == "geographic":
            raise ValueError('Input oordinate sytem {} is not supported in regridding tasks.'.format(self.coordinate_system))

        # Initialise variables involved in creating the particle set
        host_elements = None
        particles_in_domain = 0
        group = 0
        id = 0

        # Create the particle set
        for i in range(self.n_points):
            # Unique particle ID.
            id += 1

            # lon, lat and depth coordinates
            x1 = self.lons[i]
            x2 = self.lats[i]
            x3 = self.depths[i]

            # Create particle
            particle_smart_ptr = ParticleSmartPtr(group_id=group, x1=x1, x2=x2, x3=x3, id=id)

            # Find particle host element
            if host_elements is not None:
                # Try a local search first using guess as a starting point
                particle_smart_ptr.set_all_host_horizontal_elems(host_elements)
                flag = self.data_reader.find_host_using_local_search(particle_smart_ptr.get_ptr())
                if flag != IN_DOMAIN:
                    # Local search failed. Check to see if the particle is in a masked element. If not, do a global search.
                    if flag != IN_MASKED_ELEM:
                        flag = self.data_reader.find_host_using_global_search(particle_smart_ptr.get_ptr())
            else:
                # Global search ...
                flag = self.data_reader.find_host_using_global_search(particle_smart_ptr.get_ptr())

            if flag == IN_DOMAIN:
                particle_smart_ptr.get_ptr().set_in_domain(True)

                particles_in_domain += 1

                # Use the location of the last particle to guide the search for the
                # next. This should be fast if particle initial positions are collocated.
                host_elements = particle_smart_ptr.get_all_host_horizontal_elems()
            else:
                # Flag host elements as being invalid
                for grid_name in self.get_grid_names():
                    particle_smart_ptr.set_host_horizontal_elem(grid_name, INT_INVALID)
                particle_smart_ptr.get_ptr().set_in_domain(False)

            # Add particles to the particle set
            self.particle_smart_ptrs.append(particle_smart_ptr)
            self.particle_ptrs.push_back(particle_smart_ptr.get_ptr())

        if particles_in_domain == 0:
            raise RuntimeError('All points lie outside of the model domain!')

        #if self.config.get('GENERAL', 'log_level') == 'DEBUG':
        #  logger = logging.getLogger(__name__)
        #    logger.info('{} of {} particles are located in the model domain.'.format(particles_in_domain, len(self.particle_seed_smart_ptrs)))

    def interpolate(self, datetime_now, var_names):
        """ Return values for the given variables at the specified time point(s)

        Parameters
        ----------
        time : datetime.datetime
            The date/time at which the interpolation should be performed.

        var_names : list[str]
            List of variable names that for which interpolated data is required.
            Names should correspond to PyLag standard names (see
            `pylag.variable_library.standard_variable_names` for the full list.

        Returns
        -------
        interpolated_vars : dict(str : 1D NumPy array)
            Dictionary giving the interpolated data.
        """
        cdef DTYPE_FLOAT_t[:] var_c
        cdef DTYPE_FLOAT_t vel[3]
        cdef Particle* particle_ptr
        cdef DTYPE_FLOAT_t fill_value
        cdef bint uo_requested, vo_requested
        cdef DTYPE_FLOAT_t time_in_seconds
        cdef DTYPE_INT_t i

        uo_requested = False
        if 'uo' in var_names:
            uo_requested = True

        vo_requested = False
        if 'vo' in var_names:
            vo_requested = True

        env_var_names = []
        for var_name in var_names:
            if var_name not in ['uo', 'vo']:
                if var_name in variable_library.standard_variable_names.keys():
                    env_var_names.append(var_name)
                else:
                    raise ValueError('Unrecognised or unsupported variable {}'.format(var_name))

        # Establish the current time in seconds
        time_in_seconds = (datetime_now - self.datetime_start).total_seconds()

        # Read data for the current time
        self.data_reader.read_data(time_in_seconds)

        # Set vertical positions for the interpolation
        self._set_vertical_positions(time_in_seconds)

        # Dictionary in which to hold the interpolated data
        data = {}

        if uo_requested or vo_requested:
            # Create array in which to hold the interpolated uo and vo data
            uo, uo_fill_value = self._create_float_array()
            vo, vo_fill_value = self._create_float_array()
            uo_c = uo
            vo_c = vo

            # Interpolate
            for i in range(self.n_points):
                particle_ptr = self.particle_ptrs[i]
                if particle_ptr.get_in_domain() == True and particle_ptr.get_is_beached() == 0:
                    self.data_reader.get_velocity(time_in_seconds, particle_ptr, vel)
                    uo_c[i] = vel[0]
                    vo_c[i] = vel[1]
                else:
                    uo_c[i] = uo_fill_value
                    vo_c[i] = vo_fill_value

            # Add uo to dictionary
            if uo_requested:
                data['uo'] = np.ma.masked_values(uo, uo_fill_value)

            # Add vo to dictionary
            if vo_requested:
                data['vo'] = np.ma.masked_values(vo, vo_fill_value)

        # Loop over all environmental variables
        for var_name in env_var_names:
            var, var_fill_value = self._create_float_array()
            var_c = var

            # Interpolate
            for i in range(self.n_points):
                particle_ptr = self.particle_ptrs[i]
                if particle_ptr.get_in_domain() == True and particle_ptr.get_is_beached() == 0:
                    var_c[i] = self.data_reader.get_environmental_variable(var_name, time_in_seconds, particle_ptr)
                else:
                    var_c[i] = var_fill_value

            # Save data
            data[var_name] = np.ma.masked_values(var, var_fill_value)

        return data

    def _set_vertical_positions(self, DTYPE_FLOAT_t time):
        cdef Particle* particle_ptr
        cdef DTYPE_FLOAT_t zmin, zmax, z
        cdef DTYPE_INT_t i

        for i in range(self.n_points):
            particle_ptr = self.particle_ptrs[i]

            # Set vertical grid vars for particles that lie inside the domain
            if particle_ptr.get_in_domain() == True:
                # Grid limits for error checking
                zmin = self.data_reader.get_zmin(time, particle_ptr)
                zmax = self.data_reader.get_zmax(time, particle_ptr)

                # Compute depth relative to the free surface height at time t.
                # NB Depths are positive up from the sea surface.
                z = self.depths[i] + zmax
                particle_ptr.set_x3(z)

                # Determine if the host element is presently dry
                if self.data_reader.is_wet(time, particle_ptr) == 1:

                    # Confirm the given depth is valid for wet cells
                    if particle_ptr.get_x3() < zmin or particle_ptr.get_x3() > zmax:
                        particle_ptr.set_is_beached(1)
                    else:
                        particle_ptr.set_is_beached(0)

                        # Find the host z layer
                        flag = self.data_reader.set_vertical_grid_vars(time, particle_ptr)

                        if flag != IN_DOMAIN:
                            raise ValueError("Supplied depth z (= {}) is not within the grid (h = {}, zeta={}).".format(particle_ptr.get_x3(), zmin, zmax))
                else:
                    # Don't set vertical grid vars as this will fail if zeta < h. They will be set later.
                    particle_ptr.set_is_beached(1)

    def _create_float_array(self):
        var = np.zeros(self.n_points, dtype=DTYPE_FLOAT)
        fill_value = np.ma.default_fill_value(var)
        return var, fill_value

