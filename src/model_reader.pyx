# # cython: profile=True
import numpy as np
from netCDF4 import Dataset, num2date
import datetime
import logging

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from unstruct_grid_tools import round_time, sort_adjacency_array

cdef class ModelReader(object):

    def find_host(self, xpos, ypos, guess=None):
        pass
    
    def update_time_vars(self, time_ref):
        pass

    def get_time_fraction(self, time_ref):
        pass    
    
    def get_bathymetry(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_INT_t host):
        pass
    
    def get_sea_sur_elev(self, DTYPE_FLOAT_t time_fraction, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t host):
        pass

    def get_velocity(self, DTYPE_INT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_FLOAT_t[:] vel):
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
    cdef DTYPE_FLOAT_t[:] _zeta_last
    cdef DTYPE_FLOAT_t[:] _zeta_next
    
    # u/v velocity components
    cdef DTYPE_FLOAT_t[:,:] _u_last
    cdef DTYPE_FLOAT_t[:,:] _u_next
    cdef DTYPE_FLOAT_t[:,:] _v_last
    cdef DTYPE_FLOAT_t[:,:] _v_next
    cdef DTYPE_FLOAT_t[:,:] _omega_last
    cdef DTYPE_FLOAT_t[:,:] _omega_next
    
    # Time array
    cdef DTYPE_INT_t[:] _time
    cdef DTYPE_INT_t _tidx_last
    cdef DTYPE_INT_t _tidx_next
    
    def __init__(self, config, datetime_start):
        self.data_file_name = config.get("OCEAN_CIRCULATION_MODEL", "data_file")
        self.grid_file_name = config.get("OCEAN_CIRCULATION_MODEL", "grid_metrics_file")

        self._read_grid()
        self._init_vars(datetime_start)     
        
    def find_host(self, xpos, ypos, guess=None):
        if guess is not None:
            try:
                return self._find_host_using_local_search(xpos, ypos, guess)
            except ValueError:
                pass

        return self._find_host_using_global_search(xpos, ypos)

    def update_time_vars(self, time_ref):
        # Find indices for times within time_array that bracket time_start
        tidx_last = None
        for idx, t_test in enumerate(self._time):
            if time_ref >= t_test and idx < (len(self._time) - 1):
                tidx_last = idx
                break

        if tidx_last is None:
            logger = logging.getLogger(__name__)
            logger.info('The provided time {} lies outside of the range for which '\
            'there exists input data.'.format(time_ref))
            raise TypeError('Time out of range.')

        # Adjacent time index
        tidx_next = tidx_last + 1
        
        # Save time indices
        self._tidx_last = tidx_last
        self._tidx_next = tidx_next
        
        # Initialise memory views for zeta
        self._zeta_last = self._data_file.variables['zeta'][self._tidx_last,:]
        self._zeta_next = self._data_file.variables['zeta'][self._tidx_next,:]
        
        # Initialise memory views for u, v and w
        self._u_last = self._data_file.variables['u'][self._tidx_last,:,:]
        self._u_next = self._data_file.variables['u'][self._tidx_next,:,:]
        self._v_last = self._data_file.variables['v'][self._tidx_last,:,:]
        self._v_next = self._data_file.variables['v'][self._tidx_next,:,:]
        self._omega_last = self._data_file.variables['omega'][self._tidx_last,:,:]
        self._omega_next = self._data_file.variables['omega'][self._tidx_next,:,:]

    def get_time_fraction(self, time_ref):
        # Calculate time fraction according to the formula (t - t1)/(t2 - t1)
        time_fraction = float(time_ref - self._time[self._tidx_last]) / float(self._time[self._tidx_next] - self._time[self._tidx_last])
        return time_fraction

    def get_bathymetry(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_INT_t host):
        """
        Return bathymetry at the supplied x/y coordinates.
        """
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier
        cdef int n_vertices = 3 # No. of vertices in a triangle
        cdef DTYPE_FLOAT_t h # Bathymetry at (xpos, ypos)

        # Intermediate arrays
        cdef DTYPE_FLOAT_t[:] x_tri = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] y_tri = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] h_tri = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] phi = np.empty(3, dtype=DTYPE_FLOAT)

        for i in xrange(n_vertices):
            vertex = self._nv[i,host]
            x_tri[i] = self._x[vertex]
            y_tri[i] = self._y[vertex]
            h_tri[i] = self._h[vertex]

        # Calculate barycentric coordinates
        self._get_barycentric_coords(xpos, ypos, x_tri, y_tri, phi)

        h = self._interpolate_within_element(h_tri, phi)

        return h
    
    def get_sea_sur_elev(self, DTYPE_FLOAT_t time_fraction, DTYPE_FLOAT_t xpos,
            DTYPE_FLOAT_t ypos, DTYPE_INT_t host):
        """
        Return sea surface elevation at the supplied x/y coordinates.
        """
        cdef int i # Loop counters
        cdef int vertex # Vertex identifier
        cdef int n_vertices = 3 # No. of vertices in a triangle
        cdef DTYPE_FLOAT_t zeta # Sea surface elevation at (t, xpos, ypos)

        # Intermediate arrays
        cdef DTYPE_FLOAT_t[:] x_tri = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] y_tri = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] zeta_tri_t_last = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] zeta_tri_t_next = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] zeta_tri = np.empty(3, dtype=DTYPE_FLOAT)
        cdef DTYPE_FLOAT_t[:] phi = np.empty(3, dtype=DTYPE_FLOAT)

        for i in xrange(n_vertices):
            vertex = self._nv[i,host]
            x_tri[i] = self._x[vertex]
            y_tri[i] = self._y[vertex]
            zeta_tri_t_last[i] = self._zeta_last[vertex]
            zeta_tri_t_next[i] = self._zeta_next[vertex]

        # Calculate barycentric coordinates
        self._get_barycentric_coords(xpos, ypos, x_tri, y_tri, phi)

        # Interpolate in time
        self._interpolate_in_time_at_tri_nodes(time_fraction, zeta_tri_t_last, zeta_tri_t_next, zeta_tri)

        # Interpolate in space
        zeta = self._interpolate_within_element(zeta_tri, phi)

        return zeta

    def get_velocity(self, DTYPE_INT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_FLOAT_t[:] vel):
        pass

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

        self._h = ncfile.variables['h'][:]

        # Create upper right cartesian grid
        #self._vxmin = np.min(self._x)
        #self._x_upper_right_cart = self._x - self._vxmin
        #self._vymin = np.min(self._y)
        #self._y_upper_right_cart = self._y - self._vymin
        
        ncfile.close()
        
    def _init_vars(self, datetime_start):
        """
        Set up access to the NetCDF data file and initialise time vars/counters.
        """
        self._data_file = Dataset(self.data_file_name, 'r')
        
        # Read in time and convert to a list of datetime objects, then round to 
        # the nearest hour (TODO pass in the rounding interval from the config file)
        time_raw = self._data_file.variables['time']
        datetime_raw = num2date(time_raw[:], units=time_raw.units)
        datetime_rounded = round_time(datetime_raw)
        
        # Convert to seconds using time_start as a reference point
        time_seconds = []
        for time in datetime_rounded:
            time_seconds.append((time - datetime_start).total_seconds())
        # TODO Not sure about type conversion here
        self._time = np.array(time_seconds, dtype=DTYPE_INT)

        # Set time indices for reading frames, and initialise time-dependent 
        # variable reading frames
        self.update_time_vars(0) # 0s for simulation start
        
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

    cdef _interpolate_in_time_at_tri_nodes(self, DTYPE_FLOAT_t time_fraction, 
            DTYPE_FLOAT_t[:] zeta_tri_t_last, DTYPE_FLOAT_t[:] zeta_tri_t_next, 
            DTYPE_FLOAT_t[:] zeta_tri):

        cdef DTYPE_INT_t i # Loop counter
        cdef DTYPE_INT_t n_vertices = 3 # No. of vertices in a triangle
        
        for i in xrange(n_vertices):
            zeta_tri[i] = (1.0 - time_fraction) * zeta_tri_t_last[i] + time_fraction * zeta_tri_t_next[i]

    cdef DTYPE_FLOAT_t _interpolate_within_element(self, DTYPE_FLOAT_t[:] var, DTYPE_FLOAT_t[:] phi):
        cdef DTYPE_FLOAT_t interpolated_var
        interpolated_var = var[0] + phi[0] * (var[1] - var[0]) + phi[1] * (var[2] - var[0])
        return interpolated_var
