import logging
from netCDF4 import Dataset, date2num, num2date

from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT 

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
    def __init__(self, file_name, start_datetime, n_particles):
        
        logger = logging.getLogger(__name__)

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
        
        # Add position variables
        self._xpos = self._ncfile.createVariable('xpos', DTYPE_FLOAT, ('time', 'particles',), **self._ncopts)
        self._xpos.units = 'meters (m)'
        self._xpos.long_name = 'Particle x position'
        
        self._ypos = self._ncfile.createVariable('ypos', DTYPE_FLOAT, ('time', 'particles',), **self._ncopts)
        self._ypos.units = 'meters (m)'
        self._ypos.long_name = 'Particle y position'        
        
        self._zpos = self._ncfile.createVariable('zpos', DTYPE_FLOAT, ('time', 'particles',), **self._ncopts)
        self._zpos.units = 'meters (m)'
        self._zpos.long_name = 'Particle z position'

        self._host = self._ncfile.createVariable('host', DTYPE_INT, ('time', 'particles',), **self._ncopts)
        self._host.units = 'None'
        self._host.long_name = 'Host horizontal element'
        
        # Add local environmental variables
        self._h = self._ncfile.createVariable('h', DTYPE_FLOAT, ('time', 'particles',), **self._ncopts)
        self._h.units = 'meters (m)'
        self._h.long_name = 'Water depth'
        
        self._zeta = self._ncfile.createVariable('zeta', DTYPE_FLOAT, ('time', 'particles',), **self._ncopts)
        self._zeta.units = 'meters (m)'
        self._zeta.long_name = 'Sea surface elevation'

        self._is_beached = self._ncfile.createVariable('is_beached', DTYPE_INT, ('time', 'particles',), **self._ncopts)
        self._is_beached.long_name = 'Is beached'
        
        # Add status variables
        self._in_domain = self._ncfile.createVariable('in_domain', 'i4', ('time', 'particles',), **self._ncopts)
        self._in_domain.units = 'None'
        self._in_domain.long_name = 'In domain flag (1 - yes; 0 - no)'

        self._status = self._ncfile.createVariable('status', 'i4', ('time', 'particles',), **self._ncopts)
        self._status.units = 'None'
        self._status.long_name = 'Status flag (1 - error state; 0 - ok)'

    def write_group_ids(self, group_ids):
        self._group_id[:] = group_ids

    def write(self, time, particle_data):
        # Next time index
        tidx = self._time.shape[0]
        
        # Rebase time units and save
        dt = num2date(time, units=self._simulation_time_units)
        self._time[tidx] = date2num(dt, units=self._time.units)
        
        self._xpos[tidx, :] = particle_data['xpos']
        self._ypos[tidx, :] = particle_data['ypos']
        self._zpos[tidx, :] = particle_data['zpos']
        self._host[tidx, :] = particle_data['host_horizontal_elem']
        self._h[tidx, :] = particle_data['h']
        self._zeta[tidx, :] = particle_data['zeta']
        self._is_beached[tidx, :] = particle_data['is_beached']
        self._in_domain[tidx, :] = particle_data['in_domain']
        self._status[tidx, :] = particle_data['status']
    
    def sync(self):
        # Sync data to disk
        self._ncfile.sync()
        
    def close(self):
        logger = logging.getLogger(__name__)
        logger.info('Closing data logger.')
        self._ncfile.close()
