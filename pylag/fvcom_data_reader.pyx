# cython: profile=True
# cython: linetrace=True

import numpy as np
from netCDF4 import Dataset, num2date
import datetime
import logging
import ConfigParser

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t
from cpython cimport bool

from data_reader cimport DataReader

cimport interpolation as interp

from math cimport int_min, float_min

from unstruct_grid_tools import round_time, sort_adjacency_array

cdef struct HostElementSearchArrays:
    # Temporary arrays used in host element searching
    DTYPE_FLOAT_t[:] x_tri
    DTYPE_FLOAT_t[:] y_tri
    DTYPE_FLOAT_t[:] phi

cdef class FVCOMDataReader(DataReader):
    # Configurtion object
    cdef object config

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
    
    # Interpolation coefficients
    cdef DTYPE_FLOAT_t[:,:] _a1u
    cdef DTYPE_FLOAT_t[:,:] _a2u
    
    # Sigma layers and levels
    cdef DTYPE_FLOAT_t[:,:] _siglev
    cdef DTYPE_FLOAT_t[:,:] _siglay
    
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
    cdef DTYPE_FLOAT_t[:] _time
    cdef DTYPE_INT_t _tidx_last
    cdef DTYPE_INT_t _tidx_next
    
    # Temporary arrays used in host element searching
    cdef HostElementSearchArrays host_elem_search_arrs

    def __init__(self, config):
        self.config = config

        self.data_file_name = config.get("OCEAN_CIRCULATION_MODEL", "data_file")
        try:
            self.grid_file_name = config.get("OCEAN_CIRCULATION_MODEL", "grid_metrics_file")
        except ConfigParser.NoOptionError:
            self.grid_file_name = None

        self._read_grid()
        self._init_time_dependent_vars()

    cpdef update_time_dependent_vars(self, DTYPE_FLOAT_t time):
        time_fraction = interp.get_time_fraction(time, self._time[self._tidx_last], self._time[self._tidx_next])
        if time_fraction < 0.0 or time_fraction >= 1.0:
            self._read_time_dependent_vars(time)

    cpdef get_bathymetry(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_INT_t host):
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
        interp.get_barycentric_coords(xpos, ypos, x_tri, y_tri, phi)

        h = interp.interpolate_within_element(h_tri, phi)

        return h
    
    cpdef get_sea_sur_elev(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos,
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
        interp.get_barycentric_coords(xpos, ypos, x_tri, y_tri, phi)

        # Interpolate in time
        time_fraction = interp.get_time_fraction(time, self._time[self._tidx_last], self._time[self._tidx_next])
        for i in xrange(n_vertices):
            zeta_tri[i] = interp.interpolate_in_time(time_fraction, zeta_tri_t_last[i], zeta_tri_t_next[i])

        # Interpolate in space
        zeta = interp.interpolate_within_element(zeta_tri, phi)

        return zeta

    cdef get_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_FLOAT_t[:] vel):
        """
        Returns the velocity field u(t,x,y,z) through linear interpolation for a 
        particle residing in the horizontal host element `host'. The actual 
        computation is split into two separate parts - one for computing u and 
        v, and one for computing omega. This reflects the fact that u and v are
        defined are element centres on sigma layers, while omega is defined at
        element nodes on sigma levels, which means the two must be handled
        separately.
        """
        self._get_uv_velocity(time, xpos, ypos, zpos, host, vel)
        vel[2] = 0 # TODO
        return

    cdef _get_uv_velocity(self, DTYPE_FLOAT_t time, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host,
            DTYPE_FLOAT_t[:] vel):
        """
        Steps:
        1) Determine coordinates of the host element's three neighbouring
        elements.
        2) Determine velocity at the coordinates of the host element and its
        three neighbouring elements.
        3) Interpolate in time in the overlying sigma layer
        4) Interpolate in time in the underlying sigma layer
        5) Interpolate in space in the over and underlying sigma layers
        6) Interpolate in the vertical between the two sigma layers to the depth
        of the particle.
        """
        # x/y coordinates of element centres
        cdef DTYPE_FLOAT_t xc[4]
        cdef DTYPE_FLOAT_t yc[4]

        # Temporary array for vel at element centres at last time point
        cdef DTYPE_FLOAT_t uc_last[4]
        cdef DTYPE_FLOAT_t vc_last[4]

        # Temporary array for vel at element centres at next time point
        cdef DTYPE_FLOAT_t uc_next[4]
        cdef DTYPE_FLOAT_t vc_next[4]

        # Vel at element centres in overlying sigma layer
        cdef DTYPE_FLOAT_t uc1[4]
        cdef DTYPE_FLOAT_t vc1[4]

        # Vel at element centres in underlying sigma layer
        cdef DTYPE_FLOAT_t uc2[4]
        cdef DTYPE_FLOAT_t vc2[4]     
        
        # Vel at the given location in the overlying sigma layer
        cdef DTYPE_FLOAT_t up1, vp1
        
        # Vel at the given location in the underlying sigma layer
        cdef DTYPE_FLOAT_t up2, vp2
        
        cdef DTYPE_FLOAT_t dudx, dudy, dvdx, dvdy
        
        cdef DTYPE_FLOAT_t rx, ry
        
        # No. of vertices and a temporary object used for determining host 
        # element barycentric coords
        cdef DTYPE_INT_t n_vertices
        cdef DTYPE_INT_t vertex
        
        # Variables used when determining indices for the sigma layers that
        # bound the particle's position
        cdef bool particle_at_surface_or_bottom_boundary
        cdef DTYPE_FLOAT_t sigma_test
        cdef DTYPE_FLOAT_t sigma_lower_layer, sigma_upper_layer
        cdef DTYPE_INT_t k_boundary, k_lower_layer, k_upper_layer
        
        # Time fraction for interpolation in time
        cdef DTYPE_FLOAT_t time_fraction

        # Array and loop indices
        cdef DTYPE_INT_t i, j, k, neighbour
        
        cdef DTYPE_INT_t nbe_min

        # Determine barycentric coordinates of the host element
        n_vertices = 3
        for i in xrange(n_vertices):
            vertex = self._nv[i,host]
            self.host_elem_search_arrs.x_tri[i] = self._x[vertex]
            self.host_elem_search_arrs.y_tri[i] = self._y[vertex]
        interp.get_barycentric_coords(xpos, ypos, self.host_elem_search_arrs.x_tri,
                self.host_elem_search_arrs.y_tri, self.host_elem_search_arrs.phi)
        
        # Find the sigma layers bounding the particle's position. First check
        # the upper and lower boundaries, then the centre of the water columnun.
        particle_at_surface_or_bottom_boundary = False
        
        # Try the top sigma layer
        k = 0
        sigma_test = self._interp_on_sigma_layer(self.host_elem_search_arrs.phi, host, k)
        if zpos >= sigma_test:
            particle_at_surface_or_bottom_boundary = True
            k_boundary = k
        else:
            # ... the bottom sigma layer
            k = self._n_siglay - 1
            sigma_test = self._interp_on_sigma_layer(self.host_elem_search_arrs.phi, host, k)
            if zpos <= sigma_test:
                particle_at_surface_or_bottom_boundary = True
                k_boundary = k
            else:
                # ... search the middle of the water column
                for k in xrange(1, self._n_siglay):
                    sigma_test = self._interp_on_sigma_layer(self.host_elem_search_arrs.phi, host, k)
                    if zpos >= sigma_test:
                        k_lower_layer = k
                        k_upper_layer = k - 1

                        sigma_lower_layer = self._interp_on_sigma_layer(self.host_elem_search_arrs.phi, host, k_lower_layer)
                        sigma_upper_layer = self._interp_on_sigma_layer(self.host_elem_search_arrs.phi, host, k_upper_layer)
                        break
                raise ValueError("Particle zpos (={} not found!".format(zpos))

        # Time fraction
        time_fraction = interp.get_time_fraction(time, self._time[self._tidx_last], self._time[self._tidx_next])
        if time_fraction < 0.0 or time_fraction > 1.0:
            logger = logging.getLogger(__name__)
            logger.info('Invalid time fraction computed at time {}s.'.format(time))
            raise ValueError('Time out of range.')

        nbe_min = int_min(int_min(self._nbe[0, host], self._nbe[1, host]), self._nbe[2, host])
        if nbe_min < 0:
            # Boundary element - no horizontal interpolation
            if particle_at_surface_or_bottom_boundary is True:
                vel[0] = interp.interpolate_in_time(time_fraction, self._u_last[k_boundary, host], self._u_next[k_boundary, host])
                vel[1] = interp.interpolate_in_time(time_fraction, self._v_last[k_boundary, host], self._v_next[k_boundary, host])
                return
            else:
                raise ValueError('TEMP ERROR: particle not at surface ...')

        else:
            # Non-boundary element - perform horizontal and temporal interpolation
            if particle_at_surface_or_bottom_boundary is True:
                xc[0] = self._xc[host]
                yc[0] = self._yc[host]
                uc1[0] = interp.interpolate_in_time(time_fraction, self._u_last[k_boundary, host], self._u_next[k_boundary, host])
                vc1[0] = interp.interpolate_in_time(time_fraction, self._v_last[k_boundary, host], self._v_next[k_boundary, host])
                for i in xrange(3):
                    neighbour = self._nbe[i, host]
                    j = i+1 # +1 as host is 0
                    xc[j] = self._xc[neighbour] 
                    yc[j] = self._yc[neighbour]
                    uc1[j] = interp.interpolate_in_time(time_fraction, self._u_last[0, neighbour], self._u_next[0, neighbour])
                    vc1[j] = interp.interpolate_in_time(time_fraction, self._v_last[0, neighbour], self._v_next[0, neighbour])
            else:
                raise ValueError('TEMP ERROR: particle not at surface ...')
        
            # Interpolate in space - overlying sigma layer
            #up1 = interp.shephard_interpolation(xpos, ypos, self._n_pts_vel_interp, xc, yc, uc1)
            #vp1 = interp.shephard_interpolation(xpos, ypos, self._n_pts_vel_interp, xc, yc, vc1)
            dudx = 0.0
            dudy = 0.0
            dvdx = 0.0
            dvdy = 0.0
            for i in xrange(4):
                dudx += uc1[i] * self._a1u[i, host]
                dudy += uc1[i] * self._a2u[i, host]
                dvdx += vc1[i] * self._a1u[i, host]
                dvdy += vc1[i] * self._a2u[i, host]
            
            rx = xpos - self._xc[host]
            ry = ypos - self._yc[host]
            up1 = uc1[0] + dudx*rx + dudy*ry
            vp1 = vc1[0] + dvdx*rx + dvdy*ry 
            
            # Interpolate in space - underlying sigma layer
            up2 = 0.0 # TODO
            vp2 = 0.0 # TODO
            
        # Vertical interpolation
        vel[0] = up1 # TODO
        vel[1] = vp1 # TODO

    cpdef find_host(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_INT_t guess):
        return self.find_host_using_local_search(xpos, ypos, guess)
    
    cpdef sigma_to_cartesian_coords(self, DTYPE_FLOAT_t sigma, DTYPE_FLOAT_t h,
            DTYPE_FLOAT_t zeta):
        return zeta + sigma * (h + zeta)
    
    cpdef cartesian_to_sigma_coords(self, DTYPE_FLOAT_t z, DTYPE_FLOAT_t h,
            DTYPE_FLOAT_t zeta):
        return (z - zeta) / (h + zeta)

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
        self._a1u = ncfile.variables['a1u'][:,:]
        self._a2u = ncfile.variables['a2u'][:,:]

        # Bathymetry
        self._h = ncfile.variables['h'][:]

        # Create upper right cartesian grid
        #self._vxmin = np.min(self._x)
        #self._x_upper_right_cart = self._x - self._vxmin
        #self._vymin = np.min(self._y)
        #self._y_upper_right_cart = self._y - self._vymin
        
        ncfile.close()
        
    def _init_time_dependent_vars(self):
        """
        Set up access to the NetCDF data file and initialise time vars/counters.
        """
        self._data_file = Dataset(self.data_file_name, 'r')
        
        # Read in time and convert to a list of datetime objects, then round to 
        # the nearest hour
        time_raw = self._data_file.variables['time']
        datetime_raw = num2date(time_raw[:], units=time_raw.units)
        rounding_interval = self.config.getint("OCEAN_CIRCULATION_MODEL", "rounding_interval")
        datetime_rounded = round_time(datetime_raw, rounding_interval)
        
        # Simulation start time
        datetime_start = datetime.datetime.strptime(self.config.get("SIMULATION", "start_datetime"), "%Y-%m-%d %H:%M:%S")
        
        # Convert to seconds using datetime_start as a reference point
        time_seconds = []
        for time in datetime_rounded:
            time_seconds.append((time - datetime_start).total_seconds())
        self._time = np.array(time_seconds, dtype=DTYPE_FLOAT)

        # Set time indices for reading frames, and initialise time-dependent 
        # variable reading frames
        self._read_time_dependent_vars(0.0) # 0s as simulation start
            
        # Temporary arays used in host element sarching
        self.host_elem_search_arrs.x_tri = np.empty(3, dtype=DTYPE_FLOAT)
        self.host_elem_search_arrs.y_tri = np.empty(3, dtype=DTYPE_FLOAT)
        self.host_elem_search_arrs.phi = np.empty(3, dtype=DTYPE_FLOAT)

    cdef _read_time_dependent_vars(self, time):
        # Find indices for times within time_array that bracket time_start
        cdef DTYPE_INT_t tidx_last, tidx_next
        cdef DTYPE_INT_t n_times
        cdef DTYPE_INT_t i
        
        n_times = len(self._time)
        
        tidx_last = -1
        tidx_next = -1
        for i in xrange(0, n_times-1):
            if time >= self._time[i] and time < self._time[i+1]:
                tidx_last = i
                tidx_next = tidx_last + 1
                break

        if tidx_last == -1:
            logger = logging.getLogger(__name__)
            logger.info('The provided time {}s lies outside of the range for which '\
            'there exists input data: {} to {}s'.format(time, self._time[0], self._time[-1]))
            raise TypeError('Time out of range.')
        
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
        
    cpdef find_host_using_local_search(self, DTYPE_FLOAT_t xpos, DTYPE_FLOAT_t ypos, DTYPE_INT_t guess):
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
        cdef DTYPE_FLOAT_t phi_test

        while True:
            for i in xrange(n_vertices):
                vertex = self._nv[i,guess]
                self.host_elem_search_arrs.x_tri[i] = self._x[vertex]
                self.host_elem_search_arrs.y_tri[i] = self._y[vertex]

            # Transform to natural coordinates
            interp.get_barycentric_coords(xpos, ypos, self.host_elem_search_arrs.x_tri, self.host_elem_search_arrs.y_tri, self.host_elem_search_arrs.phi)

            # Check to see if the particle is in the current element
            phi_test = float_min(float_min(self.host_elem_search_arrs.phi[0], self.host_elem_search_arrs.phi[1]), self.host_elem_search_arrs.phi[2])
            if phi_test >= 0.0: return guess

            # If not, use phi to select the next element to be searched
            # TODO epsilon for floating point comp
            if self.host_elem_search_arrs.phi[0] - 1.0e-10 <= phi_test and phi_test <= self.host_elem_search_arrs.phi[0] + 1.0e-10:
                guess = self._nbe[0,guess]
            elif self.host_elem_search_arrs.phi[1] - 1.0e-10 <= phi_test and phi_test <= self.host_elem_search_arrs.phi[1] + 1.0e-10:
                guess = self._nbe[1,guess]
            elif self.host_elem_search_arrs.phi[2] - 1.0e-10 <= phi_test and phi_test <= self.host_elem_search_arrs.phi[2] + 1.0e-10:
                guess = self._nbe[2,guess]
            else:
                raise RuntimeError('Host element search algorithm failed.')
            
            if guess == -1:
                # Local search failed
                return guess

    #@cython.boundscheck(False)
    cpdef find_host_using_global_search(self, DTYPE_FLOAT_t x, DTYPE_FLOAT_t y):

        cdef int i, j # Loop counters
        cdef int vertex # Vertex identifier
        cdef int n_vertices = 3 # No. of vertices in a triangle

        # Intermediate arrays
        cdef DTYPE_FLOAT_t phi_test
        
        for i in xrange(self._n_elems):
            for j in xrange(n_vertices):
                vertex = self._nv[j,i]
                self.host_elem_search_arrs.x_tri[j] = self._x[vertex]
                self.host_elem_search_arrs.y_tri[j] = self._y[vertex]

            # Transform to natural coordinates
            interp.get_barycentric_coords(x, y, self.host_elem_search_arrs.x_tri, self.host_elem_search_arrs.y_tri, self.host_elem_search_arrs.phi)

            # Check to see if the particle is in the current element
            phi_test = float_min(float_min(self.host_elem_search_arrs.phi[0], self.host_elem_search_arrs.phi[1]), self.host_elem_search_arrs.phi[2])
            if phi_test >= 0.0: return i
        return -1
    
    cdef _interp_on_sigma_layer(self, DTYPE_FLOAT_t[:] phi, DTYPE_INT_t host,
            DTYPE_INT_t kidx):
        """
        Return the linearly interpolated value of sigma on the specified sigma
        layer within the given host element.
        
        Parameters
        ----------
        phi: MemoryView, float
            Array of length three giving the barycentric coordinates at which 
            to interpolate
        host: int
            Host element index
        kidx: int
            Sigma layer on which to interpolate
        Returns
        -------
        sigma: float
            Interpolated value of sigma.
        """
        cdef int vertex # Vertex identifier
        cdef int n_vertices = 3 # No. of vertices in a triangle
        cdef DTYPE_FLOAT_t sigma_nodes[3]
        cdef DTYPE_FLOAT_t sigma # Sigma

        for i in xrange(n_vertices):
            vertex = self._nv[i,host]
            sigma_nodes[i] = self._siglay[kidx, vertex]                  

        sigma = interp.interpolate_sigma_within_element(sigma_nodes, phi)
        return sigma
