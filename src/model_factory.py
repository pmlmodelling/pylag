import numpy as np
from netCDF4 import Dataset
import logging

class ModelReader(object):
    def __init__(self, config):
        self.config = config
    
class FVCOMModelReader(ModelReader):
    def __init__(self, *args, **kwargs):
        super(FVCOMModelReader, self).__init__(*args, **kwargs)

        self._read_grid()

    def _read_grid(self):
        logger = logging.getLogger(__name__)
        logger.info('Reading FVCOM\'s grid')
        self._file_name = self.config.get('OCEAN_CIRCULATION_MODEL', 
                                          'data_file')
        
        self._ncfile = Dataset(self._file_name, 'r')
        
        # Number of nodes
        self._n_nodes = len(self._ncfile.dimensions['node'])
        
        # Number of elements
        self._n_elems = len(self._ncfile.dimensions['nele'])
        
        # Sigma lavels
        self._n_siglev = len(self._ncfile.dimensions['siglev'])
        
        # Number of sigma layers
        self._n_siglay = len(self._ncfile.dimensions['siglay'])
        
        # Grid connectivity
        self._nv = self._ncfile.variables['nv'][:] - 1 # TO CHECK
        
        # Grid adjacency
        self._nbe = self._ncfile.variables['nbe'][:] - 1 # TO CHECK
        
        # Nodal x coordinates
        self._x = self._ncfile.variables['x'][:]
        
        # Nodal y coordinates
        self._y = self._ncfile.variables['y'][:]

        # Element x coordinates (taken at face centre)
        self._xc = self._ncfile.variables['xc'][:]
        
        # Element y coordinates (taken at face centre)
        self._yc = self._ncfile.variables['yc'][:]

        # Sigma levels at nodal coordinates
        self._siglev = self._ncfile.variables['siglev'][:]
        
        # Sigma layers at nodal coordinates
        self._siglay = self._ncfile.variables['siglay'][:]
        
        # TODO Does it make sense to precompute the following (relatively
        # expensive on large grids) or to simply compute on the fly? From 
        # what I can tell so far these values are only needed during 
        # interpolation so maybe we just compute as we go?
        
        # Sigma level separation at nodal coordinates
        # TODO?
        
        # Sigma layer separation at nodal coordinates
        # TODO?
        
        # Sigma levels at element centre
        # TODO?
        
        # Sigma layers at alement centre
        # TODO?
        
        # Sigma level separation at element centre
        # TODO?
        
        # Sigma layer separation at element centre
        # TODO?
        
        # Interpolation parameters (a1u, a2u, aw0, awx, awy)
        # TODO?
        
        # Bathymetry
        self._h = self._ncfile.variables['h'][:]

        # Create upper right cartesian grid
        self._vxmin = np.min(self._x)
        self._x_upper_right_cart = self._x - self._vxmin
        self._vymin = np.min(self._y)
        self._y_upper_right_cart = self._y - self._vymin


class ModelFactory(object):
    def __init__(self, config):
        self.config = config
    
    def make_grid_reader(self):
        pass

class FVCOMModelFactory(ModelFactory):
    def __init__(self, *args, **kwargs):
        super(FVCOMModelFactory, self).__init__(*args, **kwargs)

    def make_grid_reader(self):
        return FVCOMModelReader(self.config)

def get_model_factory(config):
    if config.get("OCEAN_CIRCULATION_MODEL", "name") == "FVCOM":
        return FVCOMModelFactory(config)
    else:
        raise ValueError('Unsupported ocean circulation model.')    
