import sys
import numpy as np
from netCDF4 import Dataset
import logging

from unstruct_grid_tools import sort_adjacency_array

class ModelReader(object):
    def __init__(self, config):
        self.config = config
        
    def find_host_using_local_search(self, particle, guess=0):
        """
        Find host element for a single particle. Typically, if 
        """
        pass
    
    def find_host_using_global_search(self, particle):
        """
        Find host elements for a set of particles. Subclasses may use
        the location of one particle to speed up the locating of other particles
        within the domain -- assuming at least some particles are likely to 
        be co-located -- which should provide a speed up relative to algorithms
        that perform a global search for each and every particle in the set.
        """
        pass
    
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
        self._nv = self._ncfile.variables['nv'][:] - 1
        
        # Grid adjacency
        nbe = self._ncfile.variables['nbe'][:] - 1
        self._nbe = sort_adjacency_array(self._nv, nbe)
        
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
        
    def find_host_using_local_search(self, particle, guess=0):
        """
        Try to establish the host horizontal element for the particle.
        The algorithm adopted is as described in Shadden (2009), adapted for
        FVCOM's grid which is unstructured in the horizontal only.
        
        Parameters:
        -----------
        particle: Particle
        
        Returns:
        --------
        N/A
        
        Author(s):
        ----------------
        James Clark (PML) October 2015.
            Implemented algorithm based on Shadden (2009).
        
        References:
        -----------
        Shadden, S. 2009 TODO
        """

        while True:
            nodes = self._nv[:,guess].squeeze()
            
            # Transform to natural coordinates
            phi = get_natural_coords(particle.xpos, particle.ypos, self._x[nodes], self._y[nodes])

            # Check to see if the particle is in the current element
            if np.min(phi) >= 0.0:
                return guess

            # If not, use phi to select the next element to be searched
            if phi[0] < 0.0:
                guess = self._nbe[0,guess]
            elif phi[1] < 0.0:
                guess = self._nbe[1,guess]
            elif phi[2] < 0.0:
                guess = self._nbe[2,guess]
            else:
                raise RuntimeError('Host element search algorithm failed.')
            
            if guess == -1:
                # Local search failed
                raise ValueError('Particle not found using local search.')

    def find_host_using_global_search(self, particle):

        for i in range(self._n_elems):
            nodes = self._nv[:,i].squeeze()
            
            # Transform to natural coordinates
            phi = get_natural_coords(particle.xpos, particle.ypos, self._x[nodes], self._y[nodes])

            # Check to see if the particle is in the current element
            if np.min(phi) >= 0.0:
                return i
            
        # Particle is not in the domain
        raise ValueError('Particle is not in the domain')

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

def get_natural_coords(x, y, x_nodes, y_nodes):
    # Array entries
    a11 = y_nodes[2] - y_nodes[0]
    a12 = x_nodes[0] - x_nodes[2]
    a21 = y_nodes[0] - y_nodes[1]
    a22 = x_nodes[1] - x_nodes[0]

    # Determinant
    a = np.array([[a11,a12],[a21,a22]])
    det = np.linalg.det(a)

    # Transformation to natural coordinates
    phi = np.empty(3, dtype='float')
    phi[0] = (a11*(x - x_nodes[0]) + a12*(y - y_nodes[0]))/det
    phi[1] = (a21*(x - x_nodes[0]) + a22*(y - y_nodes[0]))/det
    phi[2] = 1.0 - phi[0] - phi[1]
    
    return phi
