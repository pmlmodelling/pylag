import logging
from netCDF4 import Dataset, date2num, num2date

class NetCDFLogger(object):
    """
    NetCDFLogger

    Parameters:
    -----------
    file_name: str
        Name of the *nc output file to be generated.

    data_type: str
        Output variable type (e.g. 'f4' for 32-bit floating point)

    Original author:
    ----------------
    James Clark (PML)
    """
    def __init__(self, config, n_particles):
        
        self.config = config
        
        logger = logging.getLogger(__name__)

        file_name = self.config.get('GENERAL', 'output_file') + '.nc'
        try:
            logger.info('Creating output file: {}.'.format(file_name))
            self._ncfile = Dataset(file_name, mode='w', format='NETCDF4_CLASSIC')
        except:
            raise RuntimeError('Failed to create output file {}.'.format(file_name))

        # Variable data type
        self._data_type='f4'

        # Create coordinate variables etc.
        self._create_file_structure(n_particles)
        
    def _create_file_structure(self, n_particles):

        self._ncfile.title = 'PyLag -- Plymouth Marine Laboratory'

        # Create coordinate dimensions
        self._ncfile.createDimension('particles', n_particles)
        self._ncfile.createDimension('time', None)
 
        # Add time variable
        self._time = self._ncfile.createVariable('time','i4',('time',))
        self._time.units = 'seconds since 1960-01-01 00:00:00'
        self._time.calendar = 'standard'
        self._time.long_name = 'Time'

        # Add particle group ids
        self._group_id = self._ncfile.createVariable('group_id','i4',('particles',))
        self._group_id.long_name = 'Particle group ID'
        
        # Add position variables
        self._xpos = self._ncfile.createVariable('xpos', self._data_type, ('time', 'particles',))
        self._xpos.units = 'meters (m)'
        self._xpos.long_name = 'Particle x position'
        
        self._ypos = self._ncfile.createVariable('ypos', self._data_type, ('time', 'particles',))
        self._ypos.units = 'meters (m)'
        self._ypos.long_name = 'Particle y position'        
        
        self._zpos = self._ncfile.createVariable('zpos', self._data_type, ('time', 'particles',))
        self._zpos.units = 'meters (m)'
        self._zpos.long_name = 'Particle z position'
        
        # Add local environmental variables
        self._h = self._ncfile.createVariable('h', self._data_type, ('time', 'particles',))
        self._h.units = 'meters (m)'
        self._h.long_name = 'Water depth'
        
        self._zeta = self._ncfile.createVariable('zeta', self._data_type, ('time', 'particles',))
        self._zeta.units = 'meters (m)'
        self._zeta.long_name = 'Sea surface elevation'
        
        # Add extra grid variables
        #self._indomain = self._ncfile.createVariable('indomain', 'i4', ('Time', 'Particles',))
        #self._inwater = self._ncfile.createVariable('inwater', 'i4', ('Time', 'Particles',))

    def write_group_ids(self, group_ids):
        self._group_id[:] = group_ids

    def write(self, time, particle_data):
        # Next time index
        tidx = self._time.shape[0]
        
        # Rebase time units and save
        dt = num2date(time, units='seconds since {}'.format(self.config.get("SIMULATION", "start_datetime")))
        self._time[tidx] = date2num(dt, units=self._time.units)
        
        self._xpos[tidx, :] = particle_data['xpos']
        self._ypos[tidx, :] = particle_data['ypos']
        self._zpos[tidx, :] = particle_data['zpos']
        self._h[tidx, :] = particle_data['h']
        self._zeta[tidx, :] = particle_data['zeta']
        
    def close(self):
        logger = logging.getLogger(__name__)
        logger.info('Closing data logger.')
        self._ncfile.close()
