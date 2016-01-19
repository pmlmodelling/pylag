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

from data_reader cimport DataReader

cimport interpolation as interp

from math cimport int_min, float_min

from unstruct_grid_tools import round_time, sort_adjacency_array

cdef struct VelInterpArrays:
    # x/y coordinates of element centres
    DTYPE_FLOAT_t[:] xc
    DTYPE_FLOAT_t[:] yc

    # Temporary array for vel at element centres at last time point
    DTYPE_FLOAT_t[:] uc_last
    DTYPE_FLOAT_t[:] vc_last
    DTYPE_FLOAT_t[:] wc_last

    # Temporary array for vel at element centres at next time point
    DTYPE_FLOAT_t[:] uc_next
    DTYPE_FLOAT_t[:] vc_next
    DTYPE_FLOAT_t[:] wc_next

    # Vel at element centres in overlying sigma layer
    DTYPE_FLOAT_t[:] uc1
    DTYPE_FLOAT_t[:] vc1
    DTYPE_FLOAT_t[:] wc1

    # Vel at element centres in underlying sigma layer
    DTYPE_FLOAT_t[:] uc2
    DTYPE_FLOAT_t[:] vc2
    DTYPE_FLOAT_t[:] wc2

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
    
    # Struct of temporary arrays used in vel interpolation
    cdef DTYPE_INT_t _n_pts_vel_interp
    cdef VelInterpArrays vel_interp_arrs
    
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
        # Vel at the given location in the overlying sigma layer
        cdef DTYPE_FLOAT_t up1, vp1, wp1
        
        # Vel at the given location in the underlying sigma layer
        cdef DTYPE_FLOAT_t up2, vp2, wp2
        
        cdef DTYPE_FLOAT_t dudx, dudy, dvdx, dvdy
        
        cdef DTYPE_FLOAT_t rx, ry
        
        # Time fraction for interpolation in time
        cdef DTYPE_FLOAT_t time_fraction

        # Array and loop indices
        cdef DTYPE_INT_t i, j, neighbour
        
        cdef DTYPE_INT_t nbe_min
        
        # Time fraction
        time_fraction = interp.get_time_fraction(time, self._time[self._tidx_last], self._time[self._tidx_next])
        if time_fraction < 0.0 or time_fraction > 1.0:
            logger = logging.getLogger(__name__)
            logger.info('Invalid time fraction computed at time {}s.'.format(time))
            raise ValueError('Time out of range.')

        nbe_min = int_min(int_min(self._nbe[0, host], self._nbe[1, host]), self._nbe[2, host])
        if nbe_min < 0:
            # Boundary element - temporal interpolation only
            up1 = interp.interpolate_in_time(time_fraction, self._u_last[0, host], self._u_next[0, host])
            vp1 = interp.interpolate_in_time(time_fraction, self._v_last[0, host], self._v_next[0, host])
            wp1 = 0.0 # TODO
            up2 = 0.0 # TODO
            vp2 = 0.0 # TODO
            wp2 = 0.0 # TODO
        else:
            # Non-boundary element - perform horizontal and temporal interpolation
            self.vel_interp_arrs.xc[0] = self._xc[host]
            self.vel_interp_arrs.yc[0] = self._yc[host]
            self.vel_interp_arrs.uc1[0] = interp.interpolate_in_time(time_fraction, self._u_last[0, host], self._u_next[0, host])
            self.vel_interp_arrs.vc1[0] = interp.interpolate_in_time(time_fraction, self._v_last[0, host], self._v_next[0, host])
            self.vel_interp_arrs.wc1[0] = 0.0 # TODO
            for i in xrange(3):
                neighbour = self._nbe[i, host]
                j = i+1 # +1 as host is 0
                self.vel_interp_arrs.xc[j] = self._xc[neighbour] 
                self.vel_interp_arrs.yc[j] = self._yc[neighbour]
                self.vel_interp_arrs.uc1[j] = interp.interpolate_in_time(time_fraction, self._u_last[0, neighbour], self._u_next[0, neighbour])
                self.vel_interp_arrs.vc1[j] = interp.interpolate_in_time(time_fraction, self._v_last[0, neighbour], self._v_next[0, neighbour])
                self.vel_interp_arrs.wc1[j] = 0.0 # TODO
                self.vel_interp_arrs.uc2[j] = 0.0 # TODO
                self.vel_interp_arrs.vc2[j] = 0.0 # TODO
                self.vel_interp_arrs.wc2[j] = 0.0 # TODO
        
            # Interpolate in space - overlying sigma layer
            #up1 = interp.shephard_interpolation(xpos, ypos, self._n_pts_vel_interp, self.vel_interp_arrs.xc, self.vel_interp_arrs.yc, self.vel_interp_arrs.uc1)
            #vp1 = interp.shephard_interpolation(xpos, ypos, self._n_pts_vel_interp, self.vel_interp_arrs.xc, self.vel_interp_arrs.yc, self.vel_interp_arrs.vc1)
            dudx = 0.0
            dudy = 0.0
            dvdx = 0.0
            dvdy = 0.0
            for i in xrange(4):
                dudx += self.vel_interp_arrs.uc1[i] * self._a1u[i, host]
                dudy += self.vel_interp_arrs.uc1[i] * self._a2u[i, host]
                dvdx += self.vel_interp_arrs.vc1[i] * self._a1u[i, host]
                dvdy += self.vel_interp_arrs.vc1[i] * self._a2u[i, host]
            
            rx = xpos - self._xc[host]
            ry = ypos - self._yc[host]
            up1 = self.vel_interp_arrs.uc1[0] + dudx*rx + dudy*ry
            vp1 = self.vel_interp_arrs.vc1[0] + dvdx*rx + dvdy*ry 
            wp1 = 0.0 # TODO
            
            # Interpolate in space - underlying sigma layer
            up2 = 0.0 # TODO
            vp2 = 0.0 # TODO
            wp2 = 0.0 # TODO
            
        # Vertical interpolation
        vel[0] = up1 # TODO
        vel[1] = vp1 # TODO
        vel[2] = wp1 # TODO

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
        
        # Initialise temporary array objects used for vel interpolation

        # No. of points used for spatial interpolation
        self._n_pts_vel_interp = 4
    
        # Temporary array for x/y coordinates of element centres
        self.vel_interp_arrs.xc = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)
        self.vel_interp_arrs.yc = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)

        # Temporary array for vel at element centres at last time point
        self.vel_interp_arrs.uc_last = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)
        self.vel_interp_arrs.vc_last = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)
        self.vel_interp_arrs.wc_last = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)

        # Temporary array for vel at element centres at next time point
        self.vel_interp_arrs.uc_next = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)
        self.vel_interp_arrs.vc_next = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)
        self.vel_interp_arrs.wc_next = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)

        # Temporary array for vel at element centres in overlying sigma layer
        self.vel_interp_arrs.uc1 = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)
        self.vel_interp_arrs.vc1 = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)
        self.vel_interp_arrs.wc1 = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)

        # Temporary array for vel at element centres in underlying sigma layer
        self.vel_interp_arrs.uc2 = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)
        self.vel_interp_arrs.vc2 = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)
        self.vel_interp_arrs.wc2 = np.empty(self._n_pts_vel_interp, dtype=DTYPE_FLOAT)
        
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
