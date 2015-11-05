import sys
import numpy as np
from netCDF4 import Dataset, num2date
import logging

from unstruct_grid_tools import sort_adjacency_array

class ModelReader(object):
    def __init__(self, config):
        self.config = config
        
    def find_host(self, xpos, ypos, guess=None):
        pass
    
class FVCOMModelReader(ModelReader):
    def __init__(self, *args, **kwargs):
        super(FVCOMModelReader, self).__init__(*args, **kwargs)

        self._read_grid()
        self._init_vars()
        
    def find_host(self, xpos, ypos, guess=None):
        if guess is not None:
            try:
                return self._find_host_using_local_search(xpos, ypos, guess)
            except ValueError:
                pass
            
        # Global search
        try:
            return self._find_host_using_global_search(xpos, ypos)
        except ValueError:
            return -1

    def _read_grid(self):
        logger = logging.getLogger(__name__)
        logger.info('Reading FVCOM\'s grid')
        
        # Try to read grid data from the grid metrics file, in which neighbour
        # element info (nbe) has been ordered to match node ordering in nv.
        grid_metrics_file = False
        if self.config.has_option('OCEAN_CIRCULATION_MODEL', 'grid_metrics_file'):
            grid_file = self.config.get('OCEAN_CIRCULATION_MODEL', 'grid_metrics_file')
            grid_metrics_file = True
        else:
            grid_file = self.config.get('OCEAN_CIRCULATION_MODEL', 'data_file')
        
        ncfile = Dataset(grid_file, 'r')
        
        # Number of nodes
        self._n_nodes = len(ncfile.dimensions['node'])
        
        # Number of elements
        self._n_elems = len(ncfile.dimensions['nele'])
        
        # Sigma lavels
        self._n_siglev = len(ncfile.dimensions['siglev'])
        
        # Number of sigma layers
        self._n_siglay = len(ncfile.dimensions['siglay'])
        
        # Grid connectivity/adjacency. If a separate grid metrics file has been
        # supplied "assume" that nv and nbe have been preprocessed
        if grid_metrics_file:
            self._nv = ncfile.variables['nv'][:]
            self._nbe = ncfile.variables['nbe'][:]
        else:
            self._nv = ncfile.variables['nv'][:] - 1
            
            logger.info('NO GRID METRICS FILE GIVEN. Grid adjacency will be ' \
            'computed in run. To save time, this should be precomputed using' \
            ' unstruct_grid_tools and saved in a separate grid metrics file.')
            nbe = ncfile.variables['nbe'][:] - 1
            self._nbe = sort_adjacency_array(self._nv, nbe)
        
        # Nodal x coordinates
        self._x = ncfile.variables['x'][:]
        
        # Nodal y coordinates
        self._y = ncfile.variables['y'][:]

        # Element x coordinates (taken at face centre)
        self._xc = ncfile.variables['xc'][:]
        
        # Element y coordinates (taken at face centre)
        self._yc = ncfile.variables['yc'][:]

        # Sigma levels at nodal coordinates
        self._siglev = ncfile.variables['siglev'][:]
        
        # Sigma layers at nodal coordinates
        self._siglay = ncfile.variables['siglay'][:]
        
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
        self._h = ncfile.variables['h'][:]

        # Create upper right cartesian grid
        self._vxmin = np.min(self._x)
        self._x_upper_right_cart = self._x - self._vxmin
        self._vymin = np.min(self._y)
        self._y_upper_right_cart = self._y - self._vymin
        
        ncfile.close()
        
    def _init_vars(self):
        """
        Create and initialise class data members pointing to time dependent
        variables. No data copying is performed at this time.
        """
        file_name = self.config.get('OCEAN_CIRCULATION_MODEL', 'data_file')
        self._data_file = Dataset(file_name, 'r')
        
        # Read in time and convert to basetime object
        time = self._data_file.variables['time']
        self._time = num2date(time[:], units=time.units)
        
        # Dictionary providing access to netcdf variables
        self._vars = {}
        
        # Initialise witout data copying
        var_names = ['u', 'v', 'zeta']
        for name in var_names:
            self._vars[name] = self._data_file.variables[name]
        
    def _find_host_using_local_search(self, xpos, ypos, guess=0):
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
            phi = get_natural_coords(xpos, ypos, self._x[nodes], self._y[nodes])

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

    def _find_host_using_global_search(self, xpos, ypos):

        for i in range(self._n_elems):
            nodes = self._nv[:,i].squeeze()
            
            # Transform to natural coordinates
            phi = get_natural_coords(xpos, ypos, self._x[nodes], self._y[nodes])

            # Check to see if the particle is in the current element
            if np.min(phi) >= 0.0:
                return i
            
        # Particle is not in the domain
        raise ValueError('Particle is not in the domain')

    def get_local_environment(self, time, x, y, z, host_elem):
        """
        Return local environmental conditions for the provided time, x, y, and 
        z coordinates. Conditions include:
        
        h: water depth (m)
        zeta: sea surface elevation (m)
        u: eastward velocity component (m/s)
        v: northward velocity component (m/s)
        
        TODO - Make host elem optional.
        """
        local_environment = {}
        
        nodes = self._nv[:,host_elem].squeeze()

        phi = get_natural_coords(x, y, self._x[nodes], self._y[nodes])

        t_fraction, tidx1, tidx2 = get_time_fraction(time, self._time)

        # Bathymetry - h
        local_environment['h'] = interpolate_within_element(self._h[nodes], phi)
        
        # Sea surface elevation - zeta
        zeta_nodes = (1.0 - t_fraction)*self._vars['zeta'][tidx1, nodes] + t_fraction * self._vars['zeta'][tidx2, nodes]
        local_environment['zeta'] = interpolate_within_element(zeta_nodes, phi)

        return local_environment
        
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

def interpolate_within_element(var, phi):
    return var[0] + phi[0] * (var[1] - var[0]) + phi[1] * (var[2] - var[0])

def get_time_fraction(time, time_array):
    # Find indices for times within time_array that bracket `time'
    tidx1 = None
    for idx, t_test in enumerate(time_array):
        if time >= t_test and idx < (len(time_array) - 1):
            tidx1 = idx
            break

    if tidx1 is None:
        logger = logging.getLogger(__name__)
        logger.info('The provided date {} lies outside of the range for which '\
        'there exists simulation output. Limits are: t_start = {}, t_end '\
        '= {}'.format(time, time_array[0], time_array[-1]))
        raise TypeError('Time out of range.')

    # Adjacent time index
    tidx2 = tidx1 + 1
    
    # Calculate time fraction according to the formula (t - t1)/(t2 - t1)
    tdelta_1 = float((time - time_array[tidx1]).total_seconds())
    tdelta_2 = float((time_array[tidx2] - time_array[tidx1]).total_seconds())
    return tdelta_1/tdelta_2, tidx1, tidx2
