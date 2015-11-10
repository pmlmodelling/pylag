import numpy as np
from netCDF4 import Dataset, num2date
import logging

# Cython imports
cimport numpy as np
np.import_array()

DTYPE_FLOAT = np.float32
DTYPE_INT = np.int32
ctypedef np.float32_t DTYPE_FLOAT_t
ctypedef np.int32_t DTYPE_INT_t

from unstruct_grid_tools import sort_adjacency_array

cdef class ModelReader(object):

    def find_host(self, xpos, ypos, guess=None):
        pass
    
cdef class FVCOMModelReader(ModelReader):
    # Name of file containing velocity field data
    cdef object data_file_name

    # Name of file containing grid data
    cdef object grid_file_name
    
    # NetCDF4 data file giving access to time dependent fields
    cdef object _data_file

    # Grid dimensions
    cdef DTYPE_INT_t _n_elems, _n_nodes, _n_siglay, _n_siglev
    
    # Element connectivity
    cdef DTYPE_INT_t[:,:] _nv
    
    # Element adjacency
    cdef DTYPE_INT_t[:,:] _nbe
    
    # Nodal coordinates
    cdef DTYPE_FLOAT_t[:] _x
    cdef DTYPE_FLOAT_t[:] _y

    # Element centre coordinates
    cdef DTYPE_FLOAT_t[:] _xc
    cdef DTYPE_FLOAT_t[:] _yc   
    
    # Bathymetry
    cdef DTYPE_FLOAT_t[:] _h
    
    # Sea surface elevation
    cdef DTYPE_FLOAT_t[:,:,:] _zeta
    
    # u/v velocity components
    cdef DTYPE_FLOAT_t[:,:,:] _u
    cdef DTYPE_FLOAT_t[:,:,:] _v
    
    # Time (datetime obj)
    cdef object _time
    
    def __init__(self, data_file_name, grid_file_name=None):
        self.data_file_name = data_file_name
        self.grid_file_name = grid_file_name

        self._read_grid()
        self._init_vars()     
        
    def find_host(self, xpos, ypos, guess=None):
        if guess is not None:
            try:
                return self._find_host_using_local_search(xpos, ypos, guess)
            except ValueError:
                pass

        return self._find_host_using_global_search(xpos, ypos)

    def _read_grid(self):
        logger = logging.getLogger(__name__)
        logger.info('Reading FVCOM\'s grid')
        
        # Try to read grid data from the grid metrics file, in which neighbour
        # element info (nbe) has been ordered to match node ordering in nv.
        if self.grid_file_name is not None:
            ncfile = Dataset(self.grid_file_name, 'r')
        else:
            ncfile = Dataset(self.data_file_name, 'r')
        
        self._n_nodes = len(ncfile.dimensions['node'])
        self._n_elems = len(ncfile.dimensions['nele'])
        self._n_siglev = len(ncfile.dimensions['siglev'])
        self._n_siglay = len(ncfile.dimensions['siglay'])
        
        # Grid connectivity/adjacency. If a separate grid metrics file has been
        # supplied "assume" that nv and nbe have been preprocessed
        if self.grid_file_name is not None:
            self._nv = ncfile.variables['nv'][:]
            self._nbe = ncfile.variables['nbe'][:]
        else:
            self._nv = ncfile.variables['nv'][:] - 1
            
            logger.info('NO GRID METRICS FILE GIVEN. Grid adjacency will be ' \
            'computed in run. To save time, this should be precomputed using' \
            ' unstruct_grid_tools and saved in a separate grid metrics file.')
            nbe = ncfile.variables['nbe'][:] - 1
            self._nbe = sort_adjacency_array(self._nv, nbe)

        self._x = ncfile.variables['x'][:]
        self._y = ncfile.variables['y'][:]
        self._xc = ncfile.variables['xc'][:]
        self._yc = ncfile.variables['yc'][:]

        # Sigma levels at nodal coordinates
        #self._siglev = ncfile.variables['siglev'][:]
        
        # Sigma layers at nodal coordinates
        #self._siglay = ncfile.variables['siglay'][:]
        
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
        #self._vxmin = np.min(self._x)
        #self._x_upper_right_cart = self._x - self._vxmin
        #self._vymin = np.min(self._y)
        #self._y_upper_right_cart = self._y - self._vymin
        
        ncfile.close()
        
    def _init_vars(self):
        """
        Set up access to the NetCDF data file and initialise time.
        """
        self._data_file = Dataset(self.data_file_name, 'r')
        
        # Read in time and convert to basetime object
        time = self._data_file.variables['time']
        self._time = num2date(time[:], units=time.units)
        
    cdef _find_host_using_local_search(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_INT_t guess):
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

        cdef int i # Loop counters
        cdef int vertex # Vertex identifier
        cdef int n_vertices = 3 # No. of vertices in a triangle

        # Intermediate arrays
        cdef DTYPE_FLOAT_t[:] x_tri = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] y_tri = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] phi = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t phi_test

        while True:
            for i in xrange(n_vertices):
                vertex = self._nv[i,guess]
                x_tri[i] = self._x[vertex]
                y_tri[i] = self._y[vertex]

            # Transform to natural coordinates
            self._get_barycentric_coords(xpos, ypos, x_tri, y_tri, phi)

            # Check to see if the particle is in the current element
            phi_test = phi[0]
            if phi[1] < phi_test: phi_test = phi[1]
            if phi[2] < phi_test: phi_test = phi[2]  
            if phi_test >= 0.0: return guess

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

    #@cython.boundscheck(False)
    cdef _find_host_using_global_search(self, DTYPE_FLOAT_t x, DTYPE_FLOAT_t y):

        cdef int i, j # Loop counters
        cdef int vertex # Vertex identifier
        cdef int n_vertices = 3 # No. of vertices in a triangle

        # Intermediate arrays
        cdef DTYPE_FLOAT_t[:] x_tri = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] y_tri = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] phi = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t phi_test
        
        for i in xrange(self._n_elems):
            for j in xrange(n_vertices):
                vertex = self._nv[j,i]
                x_tri[j] = self._x[vertex]
                y_tri[j] = self._y[vertex]

            # Transform to natural coordinates
            self._get_barycentric_coords(x, y, x_tri, y_tri, phi)

            # Check to see if the particle is in the current element
            phi_test = phi[0]
            if phi[1] < phi_test: phi_test = phi[1]
            if phi[2] < phi_test: phi_test = phi[2]  
            if phi_test >= 0.0: return i
        return -1
    
    #@cython.boundscheck(False)
    cdef _get_barycentric_coords(self, DTYPE_FLOAT_t x, DTYPE_FLOAT_t y,
            DTYPE_FLOAT_t[:] x_tri, DTYPE_FLOAT_t[:] y_tri, DTYPE_FLOAT_t[:] phi):

        cdef DTYPE_FLOAT_t a11, a12, a21, a22, det

        # Array elements
        a11 = y_tri[2] - y_tri[0]
        a12 = x_tri[0] - x_tri[2]
        a21 = y_tri[0] - y_tri[1]
        a22 = x_tri[1] - x_tri[0]

        # Determinant
        det = a11 * a22 - a12 * a21

        # Transformation to barycentric coordinates
        phi[0] = (a11*(x - x_tri[0]) + a12*(y - y_tri[0]))/det
        phi[1] = (a21*(x - x_tri[0]) + a22*(y - y_tri[0]))/det
        phi[2] = 1.0 - phi[0] - phi[1]

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
        pass
        #local_environment = {}
        
        #nodes = self._nv[:,host_elem].squeeze()

        #phi = get_natural_coords(x, y, self._x[nodes], self._y[nodes])

        #t_fraction, tidx1, tidx2 = get_time_fraction(time, self._time)

        # Bathymetry - h
        #local_environment['h'] = interpolate_within_element(self._h[nodes], phi)
        
        # Sea surface elevation - zeta
        #zeta_nodes = (1.0 - t_fraction)*self._vars['zeta'][tidx1, nodes] + t_fraction * self._vars['zeta'][tidx2, nodes]
        #local_environment['zeta'] = interpolate_within_element(zeta_nodes, phi)

        #return local_environment
    
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
