import logging
from netCDF4 import Dataset
from cftime import num2pydate
from cftime import date2num

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT 

from pylag import variable_library

class NetCDFLogger(object):
    """ NetCDF data logger.

    Parameters:
    -----------
    file_name : str
        Name of the *.nc output file to be generated. If the `.nc' extension
        is not present it is automatically added.

    start_datetime : str
        String giving the simulation start date and time.
    
    n_particles : int
        The number of particles.
    """
    def __init__(self, config, file_name, start_datetime, n_particles):
        
        logger = logging.getLogger(__name__)

        self.config = config

        if file_name[-3:] == '.nc':
            self.file_name = file_name
        else:
            self.file_name = ''.join([file_name, '.nc'])

        try:
            logger.info('Creating output file: {}.'.format(self.file_name))
            self._ncfile = Dataset(self.file_name, mode='w', format='NETCDF4')
        except:
            raise RuntimeError('Failed to create output file {}.'.format(self.file_name))

        # Compression options for the netCDF variables.
        self._ncopts = {'zlib': True, 'complevel': 7}

        # Time units
        self._simulation_time_units = 'seconds since {}'.format(start_datetime)

        # Read in the coordinate system
        coordinate_system = self.config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system").strip().lower()
        if coordinate_system in ["cartesian", "spherical"]:
            self.coordinate_system = coordinate_system
        else:
            raise ValueError("Unsupported model coordinate system `{}'".format(coordinate_system))

        # Local environmental variables
        self._env_vars = {}

        # Add environmental variables, as requested
        try:
            var_names = self.config.get("OUTPUT", "environmental_variables").strip().split(',')
            self.environmental_variables = [var_name.strip() for var_name in var_names]
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self.environmental_variables = []
            pass

        # Create coordinate variables etc.
        self._create_file_structure(n_particles)
        
    def _create_file_structure(self, n_particles):

        self._ncfile.title = 'PyLag -- Plymouth Marine Laboratory'

        # Create coordinate dimensions
        self._ncfile.createDimension('particles', n_particles)
        self._ncfile.createDimension('time', None)
 
        # Add time variable
        self._time = self._ncfile.createVariable('time', DTYPE_INT,('time',))
        self._time.units = 'seconds since 1960-01-01 00:00:00'
        self._time.calendar = 'standard'
        self._time.long_name = 'Time'

        # Add particle group ids
        self._group_id = self._ncfile.createVariable('group_id', DTYPE_INT,('particles',))
        self._group_id.long_name = 'Particle group ID'
        
        # x1
        x1_var_name = variable_library.get_coordinate_variable_name(self.coordinate_system, 'x1')
        self._x1 = self._ncfile.createVariable(x1_var_name,
                                               variable_library.get_data_type(x1_var_name),
                                               ('time', 'particles',), **self._ncopts)
        self._x1.units = variable_library.get_units(x1_var_name)
        self._x1.long_name = variable_library.get_long_name(x1_var_name)

        # x2
        x2_var_name = variable_library.get_coordinate_variable_name(self.coordinate_system, 'x2')
        self._x2 = self._ncfile.createVariable(x2_var_name,
                                               variable_library.get_data_type(x2_var_name),
                                               ('time', 'particles',), **self._ncopts)
        self._x2.units = variable_library.get_units(x2_var_name)
        self._x2.long_name = variable_library.get_long_name(x2_var_name)

        # x3
        x3_var_name = variable_library.get_coordinate_variable_name(self.coordinate_system, 'x3')
        self._x3 = self._ncfile.createVariable(x3_var_name,
                                               variable_library.get_data_type(x3_var_name),
                                               ('time', 'particles',), **self._ncopts)
        self._x3.units = variable_library.get_units(x3_var_name)
        self._x3.long_name = variable_library.get_long_name(x3_var_name)

        # Add host
        self._host = self._ncfile.createVariable('host', DTYPE_INT, ('time', 'particles',), **self._ncopts)
        self._host.units = 'None'
        self._host.long_name = 'Host horizontal element'
        
        # Add status variables
        self._in_domain = self._ncfile.createVariable('in_domain', 'i4', ('time', 'particles',), **self._ncopts)
        self._in_domain.units = 'None'
        self._in_domain.long_name = 'In domain flag (1 - yes; 0 - no)'

        self._status = self._ncfile.createVariable('status', 'i4', ('time', 'particles',), **self._ncopts)
        self._status.units = 'None'
        self._status.long_name = 'Status flag (1 - error state; 0 - ok)'

        self._is_beached = self._ncfile.createVariable('is_beached', DTYPE_INT, ('time', 'particles',), **self._ncopts)
        self._is_beached.long_name = 'Is beached'
        
        # Add grid variables
        self._h = self._ncfile.createVariable('h', DTYPE_FLOAT, ('time', 'particles',), **self._ncopts)
        self._h.units = 'meters (m)'
        self._h.long_name = 'Water depth'
        
        self._zeta = self._ncfile.createVariable('zeta', DTYPE_FLOAT, ('time', 'particles',), **self._ncopts)
        self._zeta.units = 'meters (m)'
        self._zeta.long_name = 'Sea surface elevation'

        for var_name in self.environmental_variables:
            data_type = variable_library.get_data_type(var_name)
            units = variable_library.get_units(var_name)
            long_name = variable_library.get_long_name(var_name)

            self._env_vars[var_name] = self._ncfile.createVariable(var_name, data_type, ('time', 'particles',), **self._ncopts)
            self._env_vars[var_name].units = units
            self._env_vars[var_name].long_name = long_name

    def write_group_ids(self, group_ids):
        self._group_id[:] = group_ids

    def write(self, time, particle_data):
        # Next time index
        tidx = self._time.shape[0]
        
        # Rebase time units and save
        dt = num2pydate(time, units=self._simulation_time_units)
        self._time[tidx] = date2num(dt, units=self._time.units)
        
        self._x1[tidx, :] = particle_data['x1']
        self._x2[tidx, :] = particle_data['x2']
        self._x3[tidx, :] = particle_data['x3']
        self._host[tidx, :] = particle_data['host_horizontal_elem']
        self._h[tidx, :] = particle_data['h']
        self._zeta[tidx, :] = particle_data['zeta']
        self._is_beached[tidx, :] = particle_data['is_beached']
        self._in_domain[tidx, :] = particle_data['in_domain']
        self._status[tidx, :] = particle_data['status']

        # Add environmental variables
        for var_name in self.environmental_variables:
            self._env_vars[var_name][tidx, :] = particle_data[var_name]

    def sync(self):
        # Sync data to disk
        self._ncfile.sync()
        
    def close(self):
        logger = logging.getLogger(__name__)
        logger.info('Closing data logger.')
        self._ncfile.close()

