"""
The grid metrics module exists to assist with the creation of PyLag
grid metrics files.
"""

include "constants.pxi"

cimport cython

cimport numpy as np
from libcpp.vector cimport vector

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import numpy as np
from scipy.spatial import Delaunay, cKDTree
import stripy as stripy
from collections import OrderedDict
from netCDF4 import Dataset
import time

from pylag.parameters cimport earth_radius
from pylag.math cimport area_of_a_triangle, area_of_a_spherical_triangle
from pylag.math cimport float_min
from pylag.particle_cpp_wrapper cimport ParticleSmartPtr
from pylag.particle cimport Particle
from pylag.unstructured cimport Grid

from pylag import version
from pylag.unstructured import get_unstructured_grid
from pylag.math import geographic_to_cartesian_coords_python
from pylag.math import cartesian_to_geographic_coords_python


class GridMetricsFileCreator(object):
    """ Grid metrics file creator

    Class to assist with the creation of PyLag grid metrics files

    Parameters
    ----------
    file_name : str, optional
        The name of the grid metrics file that will be created.

    format : str, optional
        The format of the NetCDF file (e.g. NetCDF4). Default: NetCDF4.

    is_global bool, optional
        Flag signifying whether the grid is global or not. Optional, default : False.

    """

    def __init__(self, file_name='./grid_metrics.nc', format="NETCDF4", is_global=False):
        self.file_name = file_name

        self.format = format

        self.is_global = is_global

        # Coordinate dimensions
        self.dims = dict()

        # Coordinate variables
        self.coordinates = dict()

        # Dictionary holding netcdf variable data
        self.vars = dict()

        # Compression options for the netCDF variables.
        self.ncopts = {'zlib': True, 'complevel': 7}

        # Create attribute for the NetCDF4 dataset
        self.ncfile = None

    def create_file(self):
        """ Create the file

        Create a new, skeleton file. Dimensions and variables must be added separately.

        Parameters
        ----------
        N/A

        Returns
        -------
        N/A
        """
        # Create the file
        self.ncfile = Dataset(self.file_name, mode='w', format=self.format)

        # Add global attributes
        self._set_global_attributes()

        # Add universal dimension variables
        self.create_dimension('three', 3)

    def create_dimension(self, name, size):
        """ Add dimension variable

        Parameters
        ----------
        name : str
            Name of the dimension

        size : int
            Name of the string
        """
        self.dims[name] = self.ncfile.createDimension(name, size)

    def create_variable(self, var_name, var_data, dimensions, dtype, fill_value=None, attrs=None):
        """" Add variable

        Parameters
        ----------
        var_name : str
            Name of the variable to add

        var_data : ndarray
            Data array

        dimensions : tuple
            Dimensions of the ndarray

        dtype : str
            Data type (e.g. float)

        fill_value : int, float, optional.
            Fill value to use. Default: None.

        attrs : dict, optional
            Dictionary of attributes. Default: None.
        """
        for dimension in dimensions:
            if dimension not in self.dims.keys():
                raise RuntimeError("Can't create variable `{}': the `{}' coordinate " \
                                   "variable has not yet been created.".format(var_name, dimension))

        if fill_value is not None:
            self.vars[var_name] = self.ncfile.createVariable(var_name, dtype, dimensions, fill_value=fill_value,
                                                             **self.ncopts)
        else:
            self.vars[var_name] = self.ncfile.createVariable(var_name, dtype, dimensions, **self.ncopts)

        if attrs is not None:
            self.vars[var_name].setncatts(attrs)

        self.vars[var_name][:] = var_data.astype(dtype, casting='same_kind')

    def _set_global_attributes(self):
        """ Set global attributes

        Add a set of global attributes to the grid metrics file.
        """

        global_attrs = OrderedDict()

        global_attrs['Conventions'] = "CF-1.7"
        global_attrs['title'] = "PyLag grid metrics file"
        global_attrs['institution'] = 'Plymouth Marine Laboratory (PML)'
        global_attrs['contact'] = 'James R. Clark (jcl@pml.ac.uk)'
        global_attrs['netcdf-version-id'] = 'netCDF-4'
        global_attrs['pylag-version-id'] = version.git_revision

        if self.is_global:
            global_attrs['is_global'] = "True"
        else:
            global_attrs['is_global'] = "False"

        self.ncfile.setncatts(global_attrs)

        global_attrs['comment'] = ""

        return global_attrs

    def close_file(self):
        """ Close the file
        """
        try:
            self.ncfile.close()
        except:
            raise RuntimeError('Problem closing file')

@cython.wraparound(True)
def create_fvcom_grid_metrics_file(fvcom_file_name, obc_file_name, obc_file_delimiter=' ',
                                   grid_metrics_file_name = './grid_metrics.nc'):
    """Create FVCOM grid metrics file

    In FVCOM output files, the grid variables nv and nbe are not ordered in the
    way PyLag expects them to be. Nor do FVCOM output files distinguish between
    open and land boundary nodes - all sides lying along a boundary are simply
    given a -1 flag. This function rectifies these problems by generating a
    separate grid metrics file that can be passed to PyLag.

    Parameters
    ----------
    fvcom_file_name : str
        The path to an FVCOM output file that can be read in a processed

    obc_file_name : str
        The path to the text file containing a list of open boundary nodes

    obc_file_delimiter : str
        The delimiter used in the obc ascii file. To specify a tab delimited
        file, set this equal to '\t'. Default: ' '.

    grid_metrics_file_name : str, optional
        The name of the grid metrics file that will be created

    Note
    ----
    This function only needs to be called once per model grid - the
    grid metrics file generated can be reused by all future simulations.

    """
    # Open the FVCOM dataset for reading
    fvcom_dataset = Dataset(fvcom_file_name, 'r')

    # Read in dimension variables
    n_nodes = fvcom_dataset.dimensions['node'].size
    n_elems = fvcom_dataset.dimensions['nele'].size
    n_siglev = fvcom_dataset.dimensions['siglev'].size
    n_siglay = fvcom_dataset.dimensions['siglay'].size

    print('Creating FVCOM grid metrics file {}'.format(grid_metrics_file_name))

    # Instantiate file creator
    gm_file_creator = GridMetricsFileCreator(file_name=grid_metrics_file_name)

    # Create skeleton file
    gm_file_creator.create_file()

    # Add dimension variables
    gm_file_creator.create_dimension('node', n_nodes)
    gm_file_creator.create_dimension('element', n_elems)
    gm_file_creator.create_dimension('siglev', n_siglev)
    gm_file_creator.create_dimension('siglay', n_siglay)

    # Add grid coordinate variables
    # -----------------------------
    for fvcom_var_name, var_name in zip(['x', 'y', 'xc', 'yc', 'lat', 'lon', 'latc', 'lonc', 'siglev', 'siglay', 'h'],
                                        ['x', 'y', 'xc', 'yc', 'latitude', 'longitude', 'latitude_c', 'longitude_c', 'siglev', 'siglay', 'h']):
        nc_var = fvcom_dataset.variables[fvcom_var_name]

        var_data = nc_var[:]

        dtype = nc_var.dtype.name

        dimensions = nc_var.dimensions
        if 'nele' in dimensions:
            dimensions = list(dimensions)
            dimensions[dimensions.index('nele')] = 'element'
            dimensions = tuple(dimensions)

        # Form dictionary of attributes
        attrs = {}
        for attr_name in nc_var.ncattrs():
            attrs[attr_name] = nc_var.getncattr(attr_name)

        gm_file_creator.create_variable(var_name, var_data, dimensions, dtype, attrs=attrs)

    # Add modified nv array
    # ---------------------
    nv_var = fvcom_dataset.variables['nv']
    nv_data = np.asarray(nv_var[:] - 1, dtype=DTYPE_INT)
    dtype = DTYPE_INT
    dimensions = list(nv_var.dimensions)
    dimensions[dimensions.index('nele')] = 'element'
    dimensions = tuple(dimensions)
    attrs = {}
    for attr_name in nv_var.ncattrs():
        attrs[attr_name] = nv_var.getncattr(attr_name)
    gm_file_creator.create_variable('nv', nv_data, dimensions, dtype, attrs=attrs)

    # Add modified nbe array
    # ----------------------
    nbe_var = fvcom_dataset.variables['nbe']
    nbe_data = np.asarray(nbe_var[:] - 1, dtype=DTYPE_INT)
    nbe_data = nbe_data.T
    nv_data = nv_data.T
    sort_adjacency_array(nv_data, nbe_data)
    nbe_data = nbe_data.T
    nv_data = nv_data.T

    # Add open boundary flags
    open_boundary_nodes = get_fvcom_open_boundary_nodes(obc_file_name, obc_file_delimiter)
    nbe_data = add_fvcom_open_boundary_flags(nv_data, nbe_data, open_boundary_nodes)


    # Add variable
    dtype = DTYPE_INT
    dimensions = list(nv_var.dimensions)
    dimensions[dimensions.index('nele')] = 'element'
    dimensions = tuple(dimensions)
    attrs = {}
    for attr_name in nbe_var.ncattrs():
        attrs[attr_name] = nbe_var.getncattr(attr_name)
    gm_file_creator.create_variable('nbe', nbe_data, dimensions, dtype, attrs=attrs)

    # Create land-sea mask
    land_sea_mask_elements = np.zeros(n_elems, dtype=DTYPE_INT)
    for i in range(n_elems):
        # Mask elements with two land boundaries
        if np.count_nonzero(nbe_data[:, i] == -1) == 2:
            land_sea_mask_elements[i] = 1

    # Land sea mask attributes
    mask_attrs = {'standard_name': 'sea_binary_mask',
                  'units': '1',
                  'long_name': 'Land-sea mask: sea = 0, land = 1'}

    gm_file_creator.create_variable('mask_c', land_sea_mask_elements, ('element',), DTYPE_INT, attrs=mask_attrs)

    # Close FVCOM dataset
    # -------------------
    fvcom_dataset.close()

    # Close grid metrics file creator
    gm_file_creator.close_file()

    return

@cython.wraparound(True)
def create_arakawa_a_grid_metrics_file(file_name, lon_var_name='longitude',lat_var_name='latitude',
                                       depth_var_name='depth', mask_var_name=None, reference_var_name=None,
                                       bathymetry_var_name=None, dim_names=None, is_global=False,
                                       surface_only=False, prng_seed=10, masked_vertices_per_element=0,
                                       grid_metrics_file_name='./grid_metrics.nc'):
    """ Create a Arakawa A-grid metrics file

    This function creates a grid metrics file for data defined on a regular rectilinear
    Arakawa A-grid. The approach taken is to reinterpret the regular grid as a single,
    unstructured grid which can be understood by PyLag.

    The unstructured grid can be created in one of two ways. The first is by using the python
    package `scipy` to create a Delaunay triangulation. The second uses the python package
    `stripy` to create the triangulation. The two methods are tied to different applications.
    The first method should be used when creating a triangulation of a regional grid. The
    second method should be used when creating a triangulation of a global grid. The choice is
    made through the optional argument `is_global`. Two different methods are used
    since they offer significant advantages over each other in the two target applications.

    We use stripy to create a spherical triangulation on the surface of the Earth in the
    global use case. In the version tested (stripy 2.02), stripy requires that the first three
    lat/lon points don't lie on a Great Circle. As this will typically be the case with
    regularly gridded data, we must permute the lat and lon arrays. To achieve this we
    use NumPy to shuffle array indices. These shuffled indices are used to permute data arrays.
    The shuffled indices are also saved as an extra variable in the grid metrics file
    so that the same shuffling can be applied to the time dependent variable arrays when PyLag
    is running. To make the operation reproducible, a fixed seed is used for the PRNG. The seed
    can be specified using the optional argument `prng`.

    If `surface_only` is set to `True`, only 2D surface grid data is extracted and saved.
    Currently, it is assumed the 0 index corresponds to the ocean's surface.

    Parameters
    ----------
    file_name : str
        The path to an file that can be read in a processed

    lon_var_name : str, optional
        The name of the longitude variable. Optional, default : 'longitude'

    lat_var_name : str, optional
        The name of the latitude variable. Optional, default : 'latitude'

    depth_var_name : str, optional
        The name of the depth variable. Optional, default : 'depth'

    mask_var_name : str, optional
        The name of the mask variable which will be used to generate the
        land sea mask. If `None`, the land sea mask is inferred from the
        surface mask of `reference_var_name`, which becomes obligatory if
        `mask_var_name` is None. If the output files contain a time varying
        mask due to changes in sea surface elevation, a land sea mask should
        be provided. Optional, default : None.

    reference_var_name : str, optional
        The name of the reference variable from which to infer the land sea mask
        if `mask_var_name` is None. Must be given if `mask_var_name` is None.
        Optional, default : None.

    bathymetry_var_name : bool, optional
        Bathymetry variable name. If None, the bathymetry is inferred from the depth mask
        of `reference_var_name`. If the output files contain a time varying mask due to
        changes in sea surface elevation, the bathymetry should be provided. Optional,
        default : True.

    dim_names : dict, optional
        Dictionary of dimension names. The dictionary should be used to specify dimension
        names if they are different to the standard names: 'time', 'depth', 'latitude',
        'longitude'. The above strings should be used for the dimension dictionary keys.
        For example, to specify a longitude dimension name of 'lon', pass in the dictionary:
        dim_names = {'longitude': 'lon'}.

    is_global : bool, optional
        Boolean flag signifying whether or not the input grid should be treated as a global
        grid. If `True`, the python package stripy is used to create a global triangulation.
        If `False`, a Delaunay triangulation is created using scipy. To reflect typical
        PyLag use cases, the flag is optional with a default value of False. Optional,
        default : False.

    surface_only : bool, optional
        If False, process depth and bathymetry. If True, only process variables
        specific to the horizontal grid. Set to False if you want to do 3D
        transport modelling, and True if you want to do 2D surface only
        transport modeling. Optional, default : False.

    prng_seed : int, optional
        Seed for the pseudo random number generator.

    masked_vertices_per_element : int, optional
        Number of masked vertices allowed within an element. Such elements may occur
        along the boundary. If masked vertices are allowed, the model will ignore
        these when interpolating within the element. Optional, default : 0.

    grid_metrics_file_name : str, optional
        The name of the grid metrics file that will be created. Optional,
        default : `grid_metrics.nc`.

    Note
    ----
    This function only needs to be called once per model grid - the
    grid metrics file generated can be reused by all future simulations.

    """
    # Seed the PRNG to make indexing permutations reproducible
    np.random.seed(prng_seed)

    if mask_var_name is None and reference_var_name is None:
        raise ValueError('Either the name of the mask variable or the name of a reference '\
                         'masked variable must be given in order to generate the land sea mask '\
                         'as required by PyLag.')

    if mask_var_name and reference_var_name:
        print('Using `mask_var_name` to form the land sea mask. Supplied reference var is unused.')

    if surface_only is False:
        if bathymetry_var_name is None and reference_var_name is None:
            raise ValueError('Either the name of the bathymetry variable or the name of a reference ' \
                             'masked variable must be given in order to compute and save the bathymetry, '\
                             'as required by PyLag when running with input fields.')

    # Process dimension name
    if dim_names is not None:
        time_dim_name = dim_names.get('time', 'time')
        depth_dim_name = dim_names.get('depth', 'depth')
        lon_dim_name = dim_names.get('longitude', 'longitude')
        lat_dim_name = dim_names.get('latitude', 'latitude')
    else:
        time_dim_name = 'time'
        depth_dim_name = 'depth'
        lat_dim_name = 'latitude'
        lon_dim_name = 'longitude'

    # Open the input file for reading
    input_dataset = Dataset(file_name, 'r')

    # Ensure masked variables are indeed masked
    input_dataset.set_auto_maskandscale(True)

    # Read in coordinate variables
    print('Reading the grid:')
    lon_var, lon_attrs_orig = _get_variable(input_dataset, lon_var_name)
    lat_var, lat_attrs_orig = _get_variable(input_dataset, lat_var_name)

    if len(lon_var.shape) != len(lat_var.shape):
        raise ValueError('Lon and lat variables have a different number of dimensions')

    # Filter attributes so that we include just the main ones
    lon_attrs = {}
    lat_attrs = {}
    for attr in ['units', 'standard_name', 'long_name']:
        try:
            lon_attrs[attr] = lon_attrs_orig[attr]
        except KeyError:
            pass
        try:
            lat_attrs[attr] = lat_attrs_orig[attr]
        except KeyError:
            pass

    # Trim the poles if they have been included (we don't want duplicate points). Assumes
    # the poles are the first or last points and that they are given in geographic coordinates.
    # NB Trimming operation is only performed if lon and lat arrays are 1D.
    trim_first_latitude = 0
    trim_last_latitude = 0
    if len(lon_var.shape) == 1:
        # Applying trimming if required
        lat_alpha = float(lat_var[0])
        lat_omega = float(lat_var[-1])
        if lat_alpha == float(-90.0) or lat_alpha == float(90.0):
            print('Trimming first latitude which sits over a pole ({} deg.)'.format(lat_alpha))
            lat_var = lat_var[1:]
            trim_first_latitude = 1
        if lat_omega == float(-90.0) or lat_omega == float(90.0):
            print('Trimming last latitude which sits over a pole ({} deg.)'.format(lat_omega))
            lat_var = lat_var[:-1]
            trim_last_latitude = 1

        # Save original lon and lat sizes
        n_longitude = lon_var.shape[0]
        n_latitude = lat_var.shape[0]
        
        # Form 2D arrays
        lon2d, lat2d = np.meshgrid(lon_var[:], lat_var[:], indexing='ij')

        # Save lons and lats at nodes
        lon_nodes = lon2d.flatten(order='C').astype(DTYPE_FLOAT)
        lat_nodes = lat2d.flatten(order='C').astype(DTYPE_FLOAT)
    
        # Create points array from lon and lat values which will be used for the triangulation
        points = np.array([lon_nodes, lat_nodes], dtype=DTYPE_FLOAT).T

    elif len(lon_var.shape) == 2:
        # Sort axes
        lon2d = sort_axes(lon_var)
        lat2d = sort_axes(lat_var)

        # Save original lon and lat sizes
        n_longitude = lon_var.shape[0]
        n_latitude = lon_var.shape[1]
        
        # Save lons and lats at nodes
        lon_nodes = lon2d.flatten(order='C').astype(DTYPE_FLOAT)
        lat_nodes = lat2d.flatten(order='C').astype(DTYPE_FLOAT)
        
        # Create a regular grid based on lon/lat indices from which to create the triangulation
        xi = np.arange(lon2d.shape[0])
        yi = np.arange(lon2d.shape[1])
        xi2d, yi2d = np.meshgrid(xi, yi, indexing='ij')
        points = np.array([xi2d.flatten(order='C'), yi2d.flatten(order='C')], dtype=DTYPE_FLOAT).T

    else:
        raise ValueError('Lon/lat vars have {} dimensions. Expected one or two.'.format(len(lon_var.shape)))

    # Save the number of nodes
    n_nodes = lon_nodes.shape[0]

    # Save depth
    if surface_only is False:
        depth_var, depth_attrs = _get_variable(input_dataset, depth_var_name)
        depth = depth_var[:]
        n_levels = depth_var.shape[0]

    # Read in the reference variable if needed
    if mask_var_name is None or (surface_only is False and bathymetry_var_name is None):
        ref_var, _ = _get_variable(input_dataset, reference_var_name)
        ref_var = sort_axes(ref_var, time_name=time_dim_name, depth_name=depth_dim_name, lat_name=lat_dim_name,
                            lon_name=lon_dim_name)

        if not np.ma.isMaskedArray(ref_var):
            raise RuntimeError('Reference variable is not a masked array. Cannot generate land-sea mask '/
                               'and/or bathymetry.')

        if len(ref_var.shape) != 4:
            raise ValueError('Reference variable is not 4D ([t, z, y, x]).')

        # Trim latitudes
        if trim_first_latitude == 1:
            ref_var = ref_var[:, :, :, 1:]
        if trim_last_latitude == 1:
            ref_var = ref_var[:, :, :, :-1]

    # Create the Triangulation
    print('\nCreating the triangulation ', end='... ')
    node_indices = np.arange(n_nodes, dtype=DTYPE_INT)

    # Use stipy of scipy depending on whether the grid is global or not.
    if is_global:
        # Permute arrays
        np.random.shuffle(node_indices)
        lon_nodes = lon_nodes[node_indices]
        lat_nodes = lat_nodes[node_indices]

        # Create the triangulation
        tri = stripy.sTriangulation(lons=np.radians(lon_nodes), lats=np.radians(lat_nodes), permute=False)
        print('done')

        # Save simplices
        #   - Flip to reverse ordering, as expected by PyLag
        nv = np.asarray(np.flip(tri.simplices.copy(), axis=1), dtype=DTYPE_INT)

        # Neighbour array
        print('\nIdentifying neighbour simplices:')
        nbe = identify_neighbour_simplices(tri)
    else:
        # Create the Triangulation
        tri = Delaunay(points)
        print('done')

        # Save simplices
        #   - Flip to reverse ordering, as expected by PyLag
        nv = np.asarray(np.flip(tri.simplices.copy(), axis=1), dtype=DTYPE_INT)

        # Save neighbours
        nbe = np.asarray(tri.neighbors, dtype=DTYPE_INT)

    # Save bathymetry
    if surface_only is False:
        print('\nGenerating the bathymetry:')
        if bathymetry_var_name:
            # NB assumes bathymetry is positive up
            bathy_var, _ = _get_variable(input_dataset, bathymetry_var_name)
            bathy = sort_axes(bathy_var, time_name=time_dim_name, depth_name=depth_dim_name, lat_name=lat_dim_name,
                              lon_name=lon_dim_name).squeeze()

            if len(bathy.shape) != 2:
                raise RuntimeError('Bathymetry array is not 2D.')

            # Trim latitudes
            if trim_first_latitude == 1:
                bathy = bathy[:, 1:]
            if trim_last_latitude == 1:
                bathy = bathy[:, :-1]

            # Reshape array
            bathy = bathy.reshape(np.prod(bathy.shape), order='C')

        else:
            # Take first time point
            bathy_ref_var = ref_var[0, :, :, :]

            # Reshape giving (n_levels, n_nodes)
            bathy_ref_var = bathy_ref_var.reshape(n_levels, np.prod(bathy_ref_var.shape[1:]), order='C')

            bathy = np.empty((bathy_ref_var.shape[1]), dtype=DTYPE_FLOAT)
            for i in range(bathy.shape[0]):
                bathy_ref_var_tmp = bathy_ref_var[:, i]
                if np.ma.count(bathy_ref_var_tmp) != 0:
                    index = np.ma.flatnotmasked_edges(bathy_ref_var_tmp)[1]
                    bathy[i] = depth[index]
                else:
                    bathy[i] = 0.0

        # Add standard attributes
        bathy_attrs = {'standard_name': 'depth',
                       'units': 'm',
                       'long_name': 'depth, measured down from the free surface',
                       'axis': 'Z',
                       'positive': 'down'}

        # Permute the bathymetry array
        bathy = bathy[node_indices]

    # Save mask
    print('\nGenerating the land sea mask at element nodes:')
    if mask_var_name:
        mask_var, mask_attrs = _get_variable(input_dataset, mask_var_name)

        # Generate land-sea mask at nodes
        land_sea_mask_nodes = sort_axes(mask_var, time_name=time_dim_name, depth_name=depth_dim_name,
                                        lat_name=lat_dim_name, lon_name=lon_dim_name).squeeze()
        if len(land_sea_mask_nodes.shape) < 2 or len(land_sea_mask_nodes.shape) > 3:
            raise ValueError('Unsupported land sea mask with shape {}'.format(land_sea_mask_nodes.shape))

        # Flip meaning yielding: 1 - masked land point, and 0 sea point.
        land_sea_mask_nodes = 1 - land_sea_mask_nodes

        # Use surface mask only if shape is 3D
        if len(land_sea_mask_nodes.shape) == 3:
            land_sea_mask_nodes = land_sea_mask_nodes[0, :, :]

        # Trim latitudes
        if trim_first_latitude == 1:
            land_sea_mask_nodes = land_sea_mask_nodes[:, 1:]
        if trim_last_latitude == 1:
            land_sea_mask_nodes = land_sea_mask_nodes[:, :-1]

        # Fix up long name to reflect flipping of mask
        mask_attrs['long_name'] = "Land-sea mask: sea = 0 ; land = 1"

    else:
        land_sea_mask_nodes = ref_var.mask[0, 0, :, :]

        # Add standard attributes
        mask_attrs = {'standard_name': 'sea_binary_mask',
                      'units': '1',
                      'long_name': 'Land-sea mask: sea = 0, land = 1'}

    land_sea_mask_nodes = np.asarray(land_sea_mask_nodes.reshape(np.prod(land_sea_mask_nodes.shape), order='C'),
                                     dtype=DTYPE_INT)

    # Permute the land sea mask indices
    land_sea_mask_nodes = land_sea_mask_nodes[node_indices]

    # Save neighbours
    #   - Sort to ensure match with nv
    print('\nSorting the adjacency array ', end='... ')
    sort_adjacency_array(nv, nbe)
    print('done')

    # Save element number
    n_elems = nv.shape[0]

    # Save lons and lats at element centres
    print('\nCalculating lons and lats at element centres ', end='... ')
    if is_global:
        xc, yc = tri.face_midpoints()
        lon_elements = np.degrees(xc)
        lat_elements = np.degrees(yc)
    else:
        lon_elements, lat_elements = compute_element_midpoints_in_geographic_coordinates(nv, lon_nodes, lat_nodes)
    print('done')

    # Save element areas
    print('\nCalculating element areas ', end='... ')
    areas = compute_element_areas(nv, lon_nodes, lat_nodes, coordinate_system='geographic')
    area_attrs = {'standard_name' : 'areas',
                  'units' : 'm^2',
                  'long_name' : 'Element areas'}
    print('done')

    # Generate the land-sea mask at elements
    print('\nGenerating land sea mask at element centres ', end='... ')
    land_sea_mask_elements = np.empty(n_elems, dtype=DTYPE_INT)
    compute_land_sea_element_mask(nv, land_sea_mask_nodes, land_sea_mask_elements, masked_vertices_per_element)
    print('done')

    # Mask elements with two land boundaries
    print('\nMask elements with two land boundaries ', end='... ')
    mask_elements_with_two_land_boundaries(nbe, land_sea_mask_elements)
    print('done')

    # Add standard attributes for the element mask
    element_mask_attrs = {'standard_name': 'sea_binary_mask',
                          'units': '1',
                          'long_name': 'Land-sea mask: sea = 0, land = 1, boundary element = 2'}

    # Transpose nv and nbe arrays to give the dimension ordering expected by PyLag
    nv = nv.T
    nbe = nbe.T

    # Flag open boundaries with -2 flag in the regional case
    if not is_global:
        print('\nFlagging open boundaries ', end='... ')
        nbe[np.asarray(nbe == -1).nonzero()] = -2
        print('done')
    else:
        # In the global case no open boundary neighbours should have been flagged
        if np.count_nonzero(nbe == -1) != 0:
            raise RuntimeError('Neighbour array for global grid contains invalid entries')

    # Create grid metrics file
    # ------------------------
    print('\nCreating grid metrics file {} '.format(grid_metrics_file_name), end='... ')

    # Instantiate file creator
    gm_file_creator = GridMetricsFileCreator(grid_metrics_file_name, is_global=is_global)

    # Create skeleton file
    gm_file_creator.create_file()

    # Add dimension variables
    gm_file_creator.create_dimension('longitude', n_longitude)
    gm_file_creator.create_dimension('latitude', n_latitude)
    gm_file_creator.create_dimension('node', n_nodes)
    gm_file_creator.create_dimension('element', n_elems)

    # Add longitude at nodes
    gm_file_creator.create_variable('longitude', lon_nodes, ('node',), DTYPE_FLOAT, attrs=lon_attrs)

    # Add longitude at element centres
    gm_file_creator.create_variable('longitude_c', lon_elements, ('element',), DTYPE_FLOAT, attrs=lon_attrs)

    # Add latitude at nodes
    gm_file_creator.create_variable('latitude', lat_nodes, ('node',), DTYPE_FLOAT, attrs=lat_attrs)

    # Add latitude at element centres
    gm_file_creator.create_variable('latitude_c', lat_elements, ('element',), DTYPE_FLOAT, attrs=lat_attrs)

    # Flag signifying whether the first latitude point should be trimmed
    trim_attrs = {'long_name': '0 - no, 1 - yes'}
    gm_file_creator.create_variable('trim_first_latitude', np.asarray(trim_first_latitude, dtype=DTYPE_INT), (), DTYPE_INT, attrs=trim_attrs)
    gm_file_creator.create_variable('trim_last_latitude', np.asarray(trim_last_latitude, dtype=DTYPE_INT), (),  DTYPE_INT, attrs=trim_attrs)

    # Vars for 3D runs
    if surface_only is False:
        # Depth dimension variable
        gm_file_creator.create_dimension('depth', n_levels)

        # Depth
        gm_file_creator.create_variable('depth', depth, ('depth',), DTYPE_FLOAT, attrs=depth_attrs)

        # Bathymetry
        gm_file_creator.create_variable('h', bathy, ('node',), DTYPE_FLOAT, attrs=bathy_attrs)

    # Add node index map
    gm_file_creator.create_variable('permutation', node_indices, ('node',), DTYPE_INT,
                                    attrs={'long_name': 'node permutation'})

    # Add simplices
    gm_file_creator.create_variable('nv', nv, ('three', 'element',), DTYPE_INT,
                                    attrs={'long_name': 'nodes surrounding each element'})

    # Add neighbours
    gm_file_creator.create_variable('nbe', nbe, ('three', 'element',), DTYPE_INT,
                                    attrs={'long_name': 'elements surrounding each element'})

    # Add land sea mask - elements
    gm_file_creator.create_variable('mask_c', land_sea_mask_elements, ('element',), DTYPE_INT, attrs=element_mask_attrs)

    # Add land sea mask
    gm_file_creator.create_variable('mask', land_sea_mask_nodes, ('node',), DTYPE_INT, attrs=mask_attrs)

    # Compute element areas
    gm_file_creator.create_variable('area', areas, ('element',), DTYPE_FLOAT, attrs=area_attrs)

    # Close input dataset
    input_dataset.close()

    # Close grid metrics file creator
    gm_file_creator.close_file()

    print('done')

    return

@cython.wraparound(True)
def create_roms_grid_metrics_file(file_name,
                                  process_grid_u=True, process_grid_v=True,
                                  process_grid_rho=True, process_grid_psi=True,
                                  lon_var_name_grid_u='lon_u', lat_var_name_grid_u='lat_u',
                                  lon_var_name_grid_v='lon_v', lat_var_name_grid_v='lat_v',
                                  lon_var_name_grid_rho='lon_rho', lat_var_name_grid_rho='lat_rho',
                                  lon_var_name_grid_psi='lon_psi', lat_var_name_grid_psi='lat_psi',
                                  xi_dim_name_grid_u='xi_u', eta_dim_name_grid_u='eta_u',
                                  xi_dim_name_grid_v='xi_v', eta_dim_name_grid_v='eta_v',
                                  xi_dim_name_grid_rho='xi_rho', eta_dim_name_grid_rho='eta_rho',
                                  xi_dim_name_grid_psi='xi_psi', eta_dim_name_grid_psi='eta_psi',
                                  mask_name_grid_rho='mask_rho',
                                  bathymetry_var_name='h', angles_var_name=None,
                                  grid_metrics_file_name='./grid_metrics.nc', **kwargs):
    """ Create a ROMS metrics file

    This function creates a grid metrics file for data generated by the Regional
    Ocean Model System (ROMS). ROMS uses an Arakawa C-grid. Here, four separate
    unstructured triangular grids are made from the four structured grids that
    make up the Arakawa C-grid. The three unstructured grids have nodes at the
    U-, V-, rho- and psi-points respectively, which form the basis of the Arakawa
    C-grid. In the horizontal, the W grid is the same as the rho grid and a separate
    grid is not constructed for it.

    Default names are given for the latitude and longitude variable names on each
    grid. If the latitude and/or longitude variable names differ from the defaults,
    these should be passed in.

    The keyword arguments that are not explicitly listed are those that help define
    the vertical rho and w grids. They are s_rho, cs_r, s_w, cs_w, hc and vtransform.
    Further details on these can be found in the ROMS manual. If the name of these variables
    are not given using keyword arguments are not given, the model resorts to trying
    standard ROMS variable names.

    Parameters
    ----------
    file_name : str
        The path to an file that can be read in a processed

    process_grid_u : bool, optional
        Process u-grid variables

    process_grid_v : bool, optional
        Process u-grid variables

    process_grid_rho : bool, optional
        Process rho-grid variables

    process_grid_psi : bool, optional
        Process psi-grid variables

    lon_var_name_grid_u : str, optional
        The name of the U-grid longitude variable

    lat_var_name_grid_u : str, optional
        The name of the U-grid latitude variable

    lon_var_name_grid_v : str, optional
        The name of the V-grid latitude variable

    lat_var_name_grid_v : str, optional
        The name of the V-grid latitude variable

    lon_var_name_grid_rho : str, optional
        The name of the rho-grid latitude variable

    lat_var_name_grid_rho : str, optional
        The name of the rho-grid latitude variable

    lon_var_name_grid_psi : str, optional
        The name of the psi-grid latitude variable

    lat_var_name_grid_psi : str, optional
        The name of the psi-grid latitude variable

    xi_dim_name_grid_u : str, optional
        The name of the u-grid xi dimension name.

    xi_dim_name_grid_v : str, optional
        The name of the v-grid xi dimension name.

    xi_dim_name_grid_rho : str, optional
        The name of the rho-grid xi dimension name.

    xi_dim_name_grid_psi : str, optional
        The name of the psi-grid xi dimension name.

    eta_dim_name_grid_u : str, optional
        The name of the u-grid eta dimension name.

    eta_dim_name_grid_v : str, optional
        The name of the v-grid eta dimension name.

    eta_dim_name_grid_rho : str, optional
        The name of the rho-grid eta dimension name.

    eta_dim_name_grid_psi : str, optional
        The name of the psi-grid eta dimension name.

    mask_name_grid_rho : str, optional
        rho-mask variable name.

    bathymetry_var_name : str, optional
        Bathymetry variable name.

    angles_var_name : str, optional
        Angles at rho points

    grid_metrics_file_name : str, optional
        The name of the grid metrics file that will be created

    Note
    ----
    This function only needs to be called once per model grid - the
    grid metrics file generated can be reused by all future simulations.

    """
    # Open the input file for reading
    input_dataset = Dataset(file_name, 'r')

    # Ensure masked variables are indeed masked
    input_dataset.set_auto_maskandscale(True)

    # Set up dictionaries for each grid
    grid_names = ['grid_u', 'grid_v', 'grid_rho', 'grid_psi']
    process_grid = {'grid_u': process_grid_u,
                    'grid_v': process_grid_v,
                    'grid_rho': process_grid_rho,
                    'grid_psi': process_grid_psi}

    lon_var_names = {'grid_u': lon_var_name_grid_u,
                     'grid_v': lon_var_name_grid_v,
                     'grid_rho': lon_var_name_grid_rho,
                     'grid_psi': lon_var_name_grid_psi}

    lat_var_names = {'grid_u': lat_var_name_grid_u,
                     'grid_v': lat_var_name_grid_v,
                     'grid_rho': lat_var_name_grid_rho,
                     'grid_psi': lat_var_name_grid_psi}

    xi_grid_names = {'grid_u': xi_dim_name_grid_u,
                     'grid_v': xi_dim_name_grid_v,
                     'grid_rho': xi_dim_name_grid_rho,
                     'grid_psi': xi_dim_name_grid_psi}

    eta_grid_names = {'grid_u': eta_dim_name_grid_u,
                      'grid_v': eta_dim_name_grid_v,
                      'grid_rho': eta_dim_name_grid_rho,
                      'grid_psi': eta_dim_name_grid_psi}

    # Variable names for those variables that define the vertical grid
    vertical_grid_var_names = {}
    vertical_grid_var_names['s_rho'] = kwargs.pop('s_rho', 's_rho')
    vertical_grid_var_names['cs_r'] = kwargs.pop('cs_r', 'Cs_r')
    vertical_grid_var_names['s_w'] = kwargs.pop('s_w', 's_w')
    vertical_grid_var_names['cs_w'] = kwargs.pop('cs_w', 'Cs_w')
    vertical_grid_var_names['hc'] = kwargs.pop('hc', 'hc')

    vtransform = kwargs.pop('vtransform', None)
    if vtransform is None:
        # If not given, try to read it in the data file
        try:
            vtransform = input_dataset.variables['Vtransform'][0]
        except KeyError:
            print('Vtransform variable not found in dataset. Please provide it as a keyword argument.')
            raise

    # Initialise dictionaries
    nodes = {}
    lon_attrs = {}
    lat_attrs = {}
    lon_nodes = {}
    lat_nodes = {}
    n_latitude = {}
    n_longitude = {}
    tris = {}
    elements = {}
    lon_elements = {}
    lat_elements = {}
    nvs = {}
    nbes = {}
    mask_attrs = {}
    land_sea_mask_nodes = {}
    land_sea_mask_elements = {}
    vertical_grid_vars = {}
    vertical_grid_var_attrs = {}

    # Process bathymetry (defined at rho points)
    bathy_var, bathy_attrs = _get_variable(input_dataset, bathymetry_var_name)
    if len(bathy_var.shape) == 2:
        bathy = sort_axes(bathy_var, lon_name=xi_grid_names['grid_rho'], lat_name=eta_grid_names['grid_rho']).squeeze()
        bathy = bathy.reshape(np.prod(bathy.shape), order='C')
    else:
        raise RuntimeError('Bathymetry array is not 2D.')

    # Remove chunk size from attrs
    try:
        del bathy_attrs['_ChunkSizes']
    except KeyError:
        pass

    # Process angles at rho points
    if angles_var_name is not None:
        angles_var, angles_attrs = _get_variable(input_dataset, angles_var_name)
        angles = sort_axes(angles_var, lon_name=xi_grid_names['grid_rho'], lat_name=eta_grid_names['grid_rho']).squeeze()
        angles = angles.reshape(np.prod(angles.shape), order='C')

    # Loop over all grids
    for grid_name in grid_names:
        if process_grid[grid_name] is False:
            continue

        print('\nProcessing {} variables ...'.format(grid_name))

        # Read in coordinate variables.
        lon_var, lon_attrs[grid_name] = _get_variable(input_dataset, lon_var_names[grid_name])
        lat_var, lat_attrs[grid_name] = _get_variable(input_dataset, lat_var_names[grid_name])

        # Remove chunk size from attrs
        try:
            del lon_attrs[grid_name]['_ChunkSizes']
        except KeyError:
            pass

        # Remove chunk size from attrs
        try:
            del lat_attrs[grid_name]['_ChunkSizes']
        except KeyError:
            pass

        if len(lon_var.shape) != len(lat_var.shape):
            raise RuntimeError('Lon and lat var shapes do not match')

        if len(lon_var.shape) == 1:
            # Regular grid
            # ------------
            lon2d, lat2d = np.meshgrid(lon_var[:], lat_var[:], indexing='ij')

        elif len(lon_var.shape) == 2:
            # Curvilinear grid
            # ----------------
            lon2d = sort_axes(lon_var, lon_name=xi_grid_names[grid_name], lat_name=eta_grid_names[grid_name]).squeeze()
            lat2d = sort_axes(lat_var, lon_name=xi_grid_names[grid_name], lat_name=eta_grid_names[grid_name]).squeeze()
        else:
            raise RuntimeError('Unrecognised lon var shape')

        # Save lon and lat points at nodes
        lon_nodes[grid_name] = lon2d.flatten(order='C')
        lat_nodes[grid_name] = lat2d.flatten(order='C')
        nodes[grid_name] = lon_nodes[grid_name].shape[0]

        # Save original lon and lat sizes
        n_longitude[grid_name] = lon_var.shape[0]
        n_latitude[grid_name] = lat_var.shape[0]

        # Create the Triangulation using local coordinates which helps to ensure a regular grid is made
        xi = range(input_dataset.dimensions[xi_grid_names[grid_name]].size)
        eta = range(input_dataset.dimensions[eta_grid_names[grid_name]].size)
        xi2d, eta2d = np.meshgrid(xi[:], eta[:], indexing='ij')
        points = np.array([xi2d.flatten(order='C'), eta2d.flatten(order='C')]).T
        tris[grid_name] = Delaunay(points)

        # Save simplices
        #   - Flip to reverse ordering, as expected by PyLag
        #   - Transpose to give it the dimension ordering expected by PyLag
        nvs[grid_name] = np.asarray(np.flip(tris[grid_name].simplices.copy(), axis=1), dtype=DTYPE_INT)

        # Save neighbours
        #   - Transpose to give it the dimension ordering expected by PyLag
        #   - Sort to ensure match with nv
        nbes[grid_name] = np.asarray(tris[grid_name].neighbors, dtype=DTYPE_INT)
        sort_adjacency_array(nvs[grid_name], nbes[grid_name])

        # Save element number
        elements[grid_name] = nvs[grid_name].shape[0]

        # Save lon and lat points at element centres
        print('Calculating lons and lats at element centres ', end='... ')
        lon_elements[grid_name], lat_elements[grid_name] = compute_element_midpoints_in_geographic_coordinates(nvs[grid_name],
                                                                                                               lon_nodes[grid_name],
                                                                                                               lat_nodes[grid_name])
        print('done')

        # Transpose to give ordering expected by PyLag
        nvs[grid_name] = nvs[grid_name].T
        nbes[grid_name] = nbes[grid_name].T

        # Flag open boundaries with -2 flag
        nbes[grid_name][np.asarray(nbes[grid_name] == -1).nonzero()] = -2

    # Create psi mask
    print('\nCalculating land sea mask ...')
    mask_grid_rho, _ = _get_variable(input_dataset, mask_name_grid_rho)

    mask_grid_rho = sort_axes(mask_grid_rho,
                              lat_name=eta_grid_names['grid_rho'],
                              lon_name=xi_grid_names['grid_rho']).squeeze()

    mask_grid_rho = np.asarray(mask_grid_rho.flatten(order='C'), dtype=DTYPE_INT)

    land_sea_mask_elements_grid_psi = compute_psi_grid_element_mask(nodes['grid_psi'],
                                                                    elements['grid_psi'],
                                                                    nvs['grid_psi'],
                                                                    nbes['grid_psi'],
                                                                    lon_nodes['grid_psi'],
                                                                    lat_nodes['grid_psi'],
                                                                    lon_elements['grid_psi'],
                                                                    lat_elements['grid_psi'],
                                                                    lon_nodes['grid_rho'],
                                                                    lat_nodes['grid_rho'],
                                                                    mask_grid_rho)

    # Flip meaning yielding: 1 - masked land point, and 0 sea point.
    land_sea_mask_elements_grid_psi = 1 - land_sea_mask_elements_grid_psi

    # Mask attrs
    mask_attrs = {'standard_name': 'psi_grid mask',
                  'units': '1',
                  'long_name': "Land-sea mask: sea = 0 ; land = 1"}

    # Vertical grid vars
    print('\nReading vertical grid vars')
    for key, name in vertical_grid_var_names.items():
        vertical_grid_vars[key], vertical_grid_var_attrs[key] = _get_variable(input_dataset, name)

    # Create grid metrics file
    # ------------------------
    print('\nCreating grid metrics file {}'.format(grid_metrics_file_name))

    # Instantiate file creator
    gm_file_creator = GridMetricsFileCreator(grid_metrics_file_name)

    # Create skeleton file
    gm_file_creator.create_file()

    # Loop over all grids
    for grid_name in grid_names:

        if process_grid[grid_name] is False:
            continue

        # Add dimension variables
        gm_file_creator.create_dimension('longitude_{}'.format(grid_name), n_longitude[grid_name])
        gm_file_creator.create_dimension('latitude_{}'.format(grid_name), n_latitude[grid_name])
        gm_file_creator.create_dimension('node_{}'.format(grid_name), nodes[grid_name])
        gm_file_creator.create_dimension('element_{}'.format(grid_name), elements[grid_name])

        # Add longitude at nodes
        gm_file_creator.create_variable('longitude_{}'.format(grid_name), lon_nodes[grid_name],
                                        ('node_{}'.format(grid_name),), DTYPE_FLOAT, attrs=lon_attrs[grid_name])

        # Add longitude at element centres
        gm_file_creator.create_variable('longitude_c_{}'.format(grid_name), lon_elements[grid_name],
                                        ('element_{}'.format(grid_name),), DTYPE_FLOAT, attrs=lon_attrs[grid_name])

        # Add latitude at nodes
        gm_file_creator.create_variable('latitude_{}'.format(grid_name), lat_nodes[grid_name],
                                        ('node_{}'.format(grid_name),), DTYPE_FLOAT, attrs=lat_attrs[grid_name])

        # Add latitude at element centres
        gm_file_creator.create_variable('latitude_c_{}'.format(grid_name), lat_elements[grid_name],
                                        ('element_{}'.format(grid_name),), DTYPE_FLOAT, attrs=lat_attrs[grid_name])

        # Add simplices
        gm_file_creator.create_variable('nv_{}'.format(grid_name), nvs[grid_name],
                                        ('three', 'element_{}'.format(grid_name),), DTYPE_INT,
                                        attrs={'long_name': 'nodes surrounding each element'})

        # Add neighbours
        gm_file_creator.create_variable('nbe_{}'.format(grid_name), nbes[grid_name],
                                        ('three', 'element_{}'.format(grid_name),), DTYPE_INT,
                                        attrs={'long_name': 'elements surrounding each element'})

    # Add land sea mask - elements
    gm_file_creator.create_variable('mask_c_grid_psi', land_sea_mask_elements_grid_psi,
                                    ('element_grid_psi',), DTYPE_INT, attrs=mask_attrs)

    # Bathymetry
    gm_file_creator.create_variable('h', bathy, ('node_grid_rho',), DTYPE_FLOAT, attrs=bathy_attrs)

    # Angles
    if angles_var_name is not None:
        gm_file_creator.create_variable('angles_grid_rho', angles, ('node_grid_rho',), DTYPE_FLOAT, attrs=angles_attrs)

    # Add dimensions and variables describing the vertical grid. No transforms are done here.
    gm_file_creator.create_dimension('s_rho', input_dataset.dimensions['s_rho'].size)
    gm_file_creator.create_dimension('s_w', input_dataset.dimensions['s_w'].size)

    for key in vertical_grid_var_names.keys():
        var = vertical_grid_vars[key]
        var_data = np.asarray(var[:])
        attrs = vertical_grid_var_attrs[key]
        gm_file_creator.create_variable(key, var_data, var.dimensions, DTYPE_FLOAT, attrs=attrs)

    gm_file_creator.create_variable('vtransform', np.asarray(vtransform, dtype=DTYPE_FLOAT), (), DTYPE_FLOAT,
                                    attrs={'long_name': 'vertical terrain following transformation equation'})

    # Close input dataset
    input_dataset.close()

    # Close grid metrics file creator
    gm_file_creator.close_file()

    return


@cython.wraparound(True)
def sort_axes(nc_var, time_name='time', depth_name='depth', lat_name='latitude',
              lon_name='longitude'):
    """ Sort variables axes

    Variables with the following dimensions are supported:

    2D - [lat, lon] in any order

    3D - [depth, lat, lon] in any order

    4D - [time, depth, lat, lon] in any order

    The function will attempt to reorder variables given common time,
    depth, longitude and latitude names.

    Parameters
    ----------
    nc_var : NetCDF4 variable
        NetCDF variable to sort

    time_name : str, optional
        The name of the time dimension coordinate.

    depth_name : str, optional
        The name of the depth dimension coordinate.

    lat_name : str, optional
        The name of the latitude dimension coordinate.

    lon_name : str, optional
        The name of the longitude dimension coordinate.

    Returns
    -------
    var : NumPy NDArray
        Variable array with sorted axes.
    """
    print("Sorting axes for variable `{}`".format(nc_var.name), end='... ')

    var = nc_var[:]
    dimensions = nc_var.dimensions
    if len(dimensions) == 2:
        lon_index = _get_dimension_index(dimensions, lon_name)

        # Shift axes to give [x, y]
        var = np.moveaxis(var, lon_index, 0)

        print('done')

        return var

    elif len(dimensions) == 3:
        depth_index = _get_dimension_index(dimensions, depth_name)
        lat_index = _get_dimension_index(dimensions, lat_name)
        lon_index = _get_dimension_index(dimensions, lon_name)

        # Shift axes to give [z, x, y]
        var = np.moveaxis(var, depth_index, 0)

        # Update lat/lon indices if needed
        if depth_index > lat_index:
            lat_index += 1
        if depth_index > lon_index:
            lon_index += 1

        var = np.moveaxis(var, lon_index, 1)

        print('done')

        return var

    elif len(dimensions) == 4:
        time_index = _get_dimension_index(dimensions, time_name)
        depth_index = _get_dimension_index(dimensions, depth_name)
        lat_index = _get_dimension_index(dimensions, lat_name)
        lon_index = _get_dimension_index(dimensions, lon_name)

        # Shift t axis
        var = np.moveaxis(var, time_index, 0)

        # Update lat/lon indices if needed
        if time_index > depth_index:
            depth_index += 1
        if time_index > lat_index:
            lat_index += 1
        if time_index > lon_index:
            lon_index += 1

        # Shift depth axis
        var = np.moveaxis(var, depth_index, 1)

        # Update lat/lon indices if needed
        if depth_index > lat_index:
            lat_index += 1
        if depth_index > lon_index:
            lon_index += 1

        var = np.moveaxis(var, lon_index, 2)

        print('done')

        return var

    else:
        raise RuntimeError('Unsupported number of dimensions associated with variable {}'.format(nc_var.name))


def _get_dimension_index(dimensions, name):
    try:
        return dimensions.index(name)
    except ValueError:
        print('\n\nFailed to find dimension index. Dimensions were {}; supplied '
              'names were {}.'.format(dimensions, name))
        raise


def _get_variable(dataset, var_name):
    print("Reading variable `{}` ".format(var_name), end='... ')

    var = None
    try:
        var = dataset.variables[var_name]
        print('done')
    except KeyError:
        print('failed')
        pass

    if var is not None:
        # Form dictionary of attributes
        attrs = {}
        for attr_name in var.ncattrs():
            attrs[attr_name] = var.getncattr(attr_name)

        return var, attrs

    raise RuntimeError("Variable `{}` not found in the supplied dataset")

@cython.wraparound(True)
cpdef identify_neighbour_simplices(stri, iterations=10, verbose=True):
    """ Identify neighbour simplices for all elements in the triangulation

    `stripy.spherical.sTriangulation` includes its own method for identifying
    neighbour simplices. However, while it is fast for small meshes, execution
    time scales with the square of the mesh node number, meaning it becomes
    prohibitively slow to use with large meshes.

    Here we use a K-D Tree to identify nearest neighbours and ultimately the
    desired adjoining neighbours for each element in the element array. It has
    better scaling properties and is the preferred method for large grids.

    Parameters
    ----------
    stri : stripy.spherical.sTriangulation
        Triangulation object.

    iterations : int, optional
        Number of iterations to use when searching for nearest neighbours. The
        number of neighbours picked out on each iteration increases in powers of
        four. On the final iteration, the full element set is checked.  If the
        search is taking a long time, try increasing or decreasing this number
        with `verbose = True`, which will provide progress updates.

    verbose : bool
        Print search progress to screen. Optional, default : True.

    Returns
    -------
    nbe : ndarray
        Array of unsorted neighbour indices with shape (n_simplices, 3).
    """
    cdef DTYPE_INT_t[:, :] simplices
    cdef DTYPE_INT_t[:] simplex_indices
    cdef DTYPE_INT_t[:, :] nbe_view, nbe_tree_view
    cdef DTYPE_INT_t[:] found_view

    cdef DTYPE_INT_t i, j, k, m, n
    cdef DTYPE_INT_t n_simplices_subset
    cdef DTYPE_INT_t simplex_idx, neighbour_idx
    cdef DTYPE_INT_t matching_nodes
    cdef DTYPE_INT_t counter

    # Simplices from Stripy. Sort the array so that we can efficiently scan
    # for neighbours. Save in a Cython memory view for rapid indexing.
    simplices = np.sort(stri.simplices, axis=1).astype(DTYPE_INT)
    n_simplices = simplices.shape[0]

    # Array of k values, where each entry corresponds to the number of nearest
    # neighbours that will be returned through calls to cKDTree.query(). The final
    # entry is the full simplex array which ensures a result is found.
    k_values = [4**n for n in range(1, iterations) if 4**n < n_simplices]
    k_values.append(n_simplices)

    # Compute element centroids in (x, y, z)
    mids = stri.points[simplices].mean(axis=1)
    mids /= np.linalg.norm(mids, axis=1).reshape(-1,1)

    # Form KDTree
    tree = cKDTree(mids)

    # Flag identifying whether all neighbours have been found. One entry per simplex.
    # Initialised to zero which indicates the simplex's neighbours have not yet been
    # found.
    found = np.zeros(n_simplices, dtype=DTYPE_INT)
    found_view = found

    # Neighbour array, initialised to -1. This ensures simplices lying along an
    # open boundary are handled correctly.
    nbe = -1 * np.ones_like(simplices, DTYPE_INT)
    nbe_view = nbe

    # Iteratively find all neighbours for all elements.
    for k in k_values:
        if verbose:
            print('\nSearching for adjoining neighbours with k = {} '.format(k), end='... ')

        simplex_indices = np.asarray(found==0, dtype=DTYPE_INT).nonzero()[0]

        mids_subset = mids[simplex_indices]

        _, nbe_tree = tree.query(mids_subset, k=k, distance_upper_bound=2.0)
        nbe_tree_view = nbe_tree.astype(DTYPE_INT)

        # Optimised inner loops with static typing
        n_simplices_subset = simplex_indices.shape[0]
        for i in range(n_simplices_subset):

            simplex_idx = simplex_indices[i]

            counter = 0
            for j in range(k):
                neighbour_idx = nbe_tree_view[i, j]

                # Count common entries
                m = 0
                n = 0
                matching_nodes = 0
                while m < 3 and n < 3:
                    if simplices[simplex_idx, m] < simplices[neighbour_idx, n]:
                        m += 1
                    elif simplices[simplex_idx, m] > simplices[neighbour_idx, n]:
                        n += 1
                    else:
                        matching_nodes += 1
                        m += 1
                        n += 1

                if matching_nodes == 2:
                    nbe_view[simplex_idx, counter] = neighbour_idx
                    counter += 1

                # Check if all neighbours were found
                if counter == 3:
                    found_view[simplex_idx] = 1
                    break

        if verbose:
            print('found {} %'.format(100*np.count_nonzero(found)/n_simplices))

        if np.count_nonzero(found) == n_simplices:
            return nbe

    return nbe


@cython.wraparound(True)
cpdef sort_adjacency_array(DTYPE_INT_t [:, :] nv, DTYPE_INT_t [:, :] nbe):
    """Sort the adjacency array

    PyLag expects the adjacency array (nbe) to be sorted in a particlular way
    relative to the grid connectivity array (nv). NB The former lists the
    elements surrounding each element; the latter the nodes surrounding each
    element.

    Parameters
    ----------
    nv : 2D ndarray, int
        Nodes surrounding element, shape (n_elems, n_vertices)

    nbe : 2D ndarray, int
        Elements surrounding element, shape (n_elems, n_vertices)
    """
    cdef DTYPE_INT_t [:] side1, side2, side3
    cdef DTYPE_INT_t [:] nv_test
    cdef DTYPE_INT_t index_side1, index_side2, index_side3
    cdef DTYPE_INT_t n_vertices, n_elems
    cdef DTYPE_INT_t elem
    cdef DTYPE_INT_t i, j

    # Initialise memory views with empty arrays
    side1 = np.empty(2, dtype=DTYPE_INT)
    side2 = np.empty(2, dtype=DTYPE_INT)
    side3 = np.empty(2, dtype=DTYPE_INT)

    # Loop over all elems
    n_elems = nv.shape[0]
    n_vertices = nv.shape[1]
    for i in range(n_elems):
        side1[0] = nv[i, 1]
        side1[1] = nv[i, 2]
        side2[0] = nv[i, 2]
        side2[1] = nv[i, 0]
        side3[0] = nv[i, 0]
        side3[1] = nv[i, 1]

        index_side1 = -1
        index_side2 = -1
        index_side3 = -1
        for j in range(n_vertices):
            elem = nbe[i, j]
            if elem != -1:
                nv_test = nv[elem, :]
                if _get_number_of_matching_nodes(nv_test, side1) == 2:
                    index_side1 = elem
                elif _get_number_of_matching_nodes(nv_test, side2) == 2:
                    index_side2 = elem
                elif _get_number_of_matching_nodes(nv_test, side3) == 2:
                    index_side3 = elem
                else:
                    raise Exception('Failed to match side to test element.')

        nbe[i, 0] = index_side1
        nbe[i, 1] = index_side2
        nbe[i, 2] = index_side3


@cython.wraparound(True)
cpdef compute_element_areas(nv, x_nodes, y_nodes, coordinate_system='geographic'):
    cdef DTYPE_INT_t vertex_0, vertex_1, vertex_2

    # Cartesian node coordinates for geographic case
    cdef DTYPE_FLOAT_t p0[3]
    cdef DTYPE_FLOAT_t p1[3]
    cdef DTYPE_FLOAT_t p2[3]
    cdef DTYPE_FLOAT_t[:, :] points_view

    # Cartesian node coordinates for cartesian case
    cdef DTYPE_FLOAT_t x1[2]
    cdef DTYPE_FLOAT_t x2[2]
    cdef DTYPE_FLOAT_t x3[2]

    cdef DTYPE_FLOAT_t[:] areas_view

    cdef DTYPE_INT_t n_elements
    cdef DTYPE_INT_t i, j

    # Number of elements
    n_elements = nv.shape[0]

    # Create array in which to store areas
    areas = np.zeros(n_elements, dtype=DTYPE_FLOAT)
    areas_view = areas

    if coordinate_system == 'geographic':
        # Convert to radians
        lon_nodes_radians = np.radians(x_nodes)
        lat_nodes_radians = np.radians(y_nodes)

        # Convert to Cartesian coordinates
        x, y, z = geographic_to_cartesian_coords_python(lon_nodes_radians, lat_nodes_radians)
        points_view = np.ascontiguousarray(np.column_stack([x, y, z]))

        for i in range(n_elements):
            # Extract nodal cartesian coordinates
            vertex_0 = nv[i, 0]
            vertex_1 = nv[i, 1]
            vertex_2 = nv[i, 2]
            for j in range(3):
                p0[j] = points_view[vertex_0, j]
                p1[j] = points_view[vertex_1, j]
                p2[j] = points_view[vertex_2, j]

            areas_view[i] = area_of_a_spherical_triangle(p0, p1, p2, earth_radius)

    elif coordinate_system == 'cartesian':
        for i in range(n_elements):
            # Extract nodal cartesian coordinates
            vertex_1 = nv[i, 0]
            vertex_2 = nv[i, 1]
            vertex_3 = nv[i, 2]
            x1[0] = x_nodes[vertex_1]
            x1[1] = y_nodes[vertex_1]
            x2[0] = x_nodes[vertex_2]
            x2[1] = y_nodes[vertex_2]
            x3[0] = x_nodes[vertex_3]
            x3[1] = y_nodes[vertex_3]

            areas_view[i] = area_of_a_triangle(x1, x2, x3)
    else:
        raise RuntimeError('Unknown coordinate system {}'.format(coordinate_system))

    return areas


@cython.wraparound(True)
cpdef compute_element_midpoints_in_geographic_coordinates(nv, lon_nodes, lat_nodes):
    # Convert to radians
    lon_nodes_radians = np.radians(lon_nodes)
    lat_nodes_radians = np.radians(lat_nodes)

    # Convert to Cartesian coordinates
    x, y, z = geographic_to_cartesian_coords_python(lon_nodes_radians, lat_nodes_radians)
    points = np.column_stack([x, y, z])

    # Compute mid points in Cartesian coordinates
    mids = points[nv].mean(axis=1)
    mids /= np.linalg.norm(mids, axis=1).reshape(-1,1)

    # Convert back to geographic coordinates
    midlons, midlats = cartesian_to_geographic_coords_python(mids[:,0], mids[:,1], mids[:,2])

    # Convert back to degrees and return
    return np.degrees(midlons), np.degrees(midlats)


@cython.wraparound(True)
cpdef compute_land_sea_element_mask(const DTYPE_INT_t [:,:] nv, const DTYPE_INT_t [:] nodal_mask,
                                    DTYPE_INT_t [:] element_mask, const DTYPE_INT_t masked_vertices_per_element):
    cdef DTYPE_INT_t node
    cdef DTYPE_INT_t n_elements, n_vertices
    cdef DTYPE_INT_t counter
    cdef DTYPE_INT_t i

    if masked_vertices_per_element < 0 or masked_vertices_per_element > 2:
        raise ValueError('Invalid selection for the number of permitted masked vertices forming an element.' \
                         'Options are 0, 1 or 2.')

    n_elements = nv.shape[0]
    n_vertices = nv.shape[1]

    element_mask[:] = 0
    for i in range(n_elements):
        counter = 0
        for j in range(n_vertices):
            node = nv[i, j]
            if nodal_mask[node] == 1:
                counter += 1

        if counter == 0:
            # Sea
            element_mask[i] = SEA
        elif counter > 0 and counter <= masked_vertices_per_element:
            # Boundary element
            element_mask[i] = BOUNDARY_ELEMENT
        else:
            # Land
            element_mask[i] = LAND


cpdef compute_psi_grid_element_mask(n_nodes, n_elements, nv, nbe, lon_grid_psi, lat_grid_psi,
                                    lonc_grid_psi, latc_grid_psi, lon_grid_rho, lat_grid_rho,
                                    mask_grid_rho):
    """ Generate psi grid element mask

    For Arakawa C grids, compute the element mask for the psi grid in which nodes are
    located at cell corners. The psi grid element mask is computed from the rho grid
    mask which gives the location of masked points at cell centres.

    Masked psi grid elements are located by searching for the elements that contain masked
    rho grid points. The search operation uses PyLag functionality. First, an
    UnstructuredGrid object is constructed for the psi grid. Then, the latitude and longitude
    values of masked rho points are passed to host searching algorithms.

    By virtue of the C grid construction, masked rho grid points sit on the edge of psi grid
    elements. For each masked rho point, the two elements that share the edge along which the
    masked rho point sits should be masked. The host element search algorithm will only locate
    one of these. The second element is identified using the barycentric coordinates of the
    masked rho point in the element that was found - the smallest coordinate signifies the
    neighbour element that must also be masked.

    Parameters
    ----------
    n_nodes : int
        The number of nodes on the psi grid.

    n_elements : int
        The number of elements on the psi grid.

    nv : 2D NumPy array, int
        Simplices array for the psi grid with shape [3, n_elements]

    nbe : 2D NumPy array, int
        Neighbour array for the psi grid with shape [3, n_elements]

    lon_grid_psi : 1D NumPy array, float
        1D array of longitudes at nodes on the psi grid.

    lat_grid_psi : 1D NumPy array, float
        1D array of latitudes at nodes on the psi grid.

    lonc_grid_psi : 1D NumPy array, float
        1D array of longitudes at element centres on the psi grid.

    latc_grid_psi : 1D NumPy array, float
        1D array of latitudes at element centres on the psi grid.

    lon_grid_rho : 1D NumPy array, float
        1D array of longitudes at nodes on the rho grid.

    lat_grid_rho : 1D NumPy array, float
        1D array of latitudes at nodes on the rho grid.

    mask_grid_rho : 1D NumPy array, float
        1D array of masked nodes on the rho grid (1 - sea; 0 - land).

    Returns
    -------
    maskc_grid_psi : 1D NumPy array, int
        1D array or masked elements on the psi grid (1 - sea; 0 - land).
    """
    cdef DTYPE_INT_t[:] mask_grid_rho_c
    cdef DTYPE_FLOAT_t[:] lon_grid_rho_c
    cdef DTYPE_FLOAT_t[:] lat_grid_rho_c
    cdef DTYPE_INT_t[:] maskc_grid_psi_c
    cdef Grid grid_psi
    cdef ParticleSmartPtr test_particle
    cdef Particle* test_particle_ptr
    cdef vector[DTYPE_FLOAT_t] phi
    cdef DTYPE_FLOAT_t phi_test
    cdef DTYPE_FLOAT_t lon, lat
    cdef DTYPE_INT_t host, host_nbe
    cdef DTYPE_INT_t flag
    cdef DTYPE_INT_t n_nodes_grid_rho
    cdef DTYPE_INT_t i

    if not (lon_grid_rho.shape[0] == lat_grid_rho.shape[0] == mask_grid_rho.shape[0]):
        raise ValueError('rho grid lon, lat and mask array dimension sizes do not match')

    # Convert to radians and ensure we are working with C contiguous data
    lon_grid_psi_rad = np.ascontiguousarray(np.radians(lon_grid_psi))
    lat_grid_psi_rad = np.ascontiguousarray(np.radians(lat_grid_psi))
    lonc_grid_psi_rad = np.ascontiguousarray(np.radians(lonc_grid_psi))
    latc_grid_psi_rad = np.ascontiguousarray(np.radians(latc_grid_psi))
    lon_grid_rho_rad = np.ascontiguousarray(np.radians(lon_grid_rho))
    lat_grid_rho_rad = np.ascontiguousarray(np.radians(lat_grid_rho))

    # Make sure nv, nbe and mask arrays are contiguous
    nv_cont = np.ascontiguousarray(nv)
    nbe_cont = np.ascontiguousarray(nbe)

    # For speed, create memory views for the rho grid mask, lons and lats
    lon_grid_rho_c = lon_grid_rho_rad
    lat_grid_rho_c = lat_grid_rho_rad
    mask_grid_rho_c = mask_grid_rho

    # Create dummy psi grid masks. These won't be used but are required in the initialisation
    # list for the Grid object we will use for host element searching.
    dummy_mask_elements_grid_psi = np.zeros(n_elements, dtype=DTYPE_INT)
    dummy_mask_nodes_grid_psi = np.zeros(n_nodes, dtype=DTYPE_INT)

    # Create an UnstructuredGeographicGrid object to aid with host element searching
    config = configparser.ConfigParser()
    config.add_section("OCEAN_CIRCULATION_MODEL")
    config.set('OCEAN_CIRCULATION_MODEL', 'coordinate_system', 'geographic')
    grid_psi = get_unstructured_grid(config, b'grid_psi', n_nodes, n_elements, nv_cont, nbe_cont, lon_grid_psi_rad,
                                     lat_grid_psi_rad, lonc_grid_psi_rad, latc_grid_psi_rad,
                                     dummy_mask_elements_grid_psi, dummy_mask_nodes_grid_psi)

    # Create psi grid mask and fill with ones, which indicate sea points. NB the mask is
    # flipped when it is being prepared for PyLag.
    maskc_grid_psi = np.ones(n_elements, dtype=DTYPE_INT)
    maskc_grid_psi_c = maskc_grid_psi

    # Create a test particle which we will use for host element searching on the psi grid
    test_particle = ParticleSmartPtr(host_elements={'grid_psi': 0})
    test_particle_ptr = test_particle.get_ptr()

    # Loop over all rho points
    n_nodes_grid_rho = mask_grid_rho.shape[0]
    for i in range(n_nodes_grid_rho):
        if mask_grid_rho_c[i] == 0:

            lon = lon_grid_rho_c[i]
            lat = lat_grid_rho_c[i]

            # Assign the masked point's lat/lon coordinates to the particle
            test_particle_ptr.set_x1(lon)
            test_particle_ptr.set_x2(lat)

            # Initiate a local search
            flag = grid_psi.find_host_using_local_search(test_particle_ptr)

            # If the search failed, try a global search
            if flag != IN_DOMAIN:
                flag = grid_psi.find_host_using_global_search(test_particle_ptr)

            # Some masked points may be outside of the psi grid domain given the
            # rho grid is larger than the psi grid. Only process points that have
            # been located within the rho grid.
            if flag == IN_DOMAIN:
                host = test_particle_ptr.get_host_horizontal_elem(b'grid_psi')

                # Mask the element
                maskc_grid_psi_c[host] = 0

                # Get the barycentric coordinates
                phi = grid_psi.get_phi(lon, lat, host)

                # From the smallest value of phi, identify the host neighbour that should also be masked
                phi_test = float_min(float_min(phi[0], phi[1]), phi[2])
                if phi[0] == phi_test:
                    host_nbe = nbe[0, host]
                elif phi[1] == phi_test:
                    host_nbe = nbe[1, host]
                else:
                    host_nbe = nbe[2, host]

                # Mask the neighbour element
                maskc_grid_psi_c[host_nbe] = 0

    return maskc_grid_psi


cpdef mask_elements_with_two_land_boundaries(const DTYPE_INT_t[:,:] nbe, DTYPE_INT_t[:] element_mask):
    cdef DTYPE_INT_t i, j
    cdef DTYPE_INT_t n_elements, n_neighbours
    cdef DTYPE_INT_t neighbour
    cdef DTYPE_INT_t counter

    n_elements = nbe.shape[0]
    n_neighbours = nbe.shape[1]
    for i in range(n_elements):
        if element_mask[i] == 0:
            counter = 0
            for j in range(n_neighbours):
                neighbour = nbe[i, j]
                if element_mask[neighbour] == 1:
                    counter += 1

            if counter == 2:
                element_mask[i] = 1


@cython.wraparound(True)
def get_fvcom_open_boundary_nodes(file_name, delimiter=' '):
    """Read fvcom open boundary nodes from file

    Parameters
    ----------
    file_name : str
        Name of file containing a list of the open boundary nodes

    Returns
    -------
    nodes : list, int
        A list of open boundary nodes as read in from file
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()

    # Number of open boundary nodes given on first line
    n_obc_nodes = int(lines.pop(0).strip().split(' ')[-1])
    print('Grid has {} nodes on the open boundary'.format(n_obc_nodes))

    nodes = []
    for line in lines:
        if line.strip():
            entries = line.strip().split(delimiter)

            # There should be exactly three entries in each line. If there aren't,
            # an incorrect delimiter has probably been passed in.
            if len(entries) == 3:
                nodes.append(int(entries[1]))
            else:
                raise ValueError('Failed to correctly parse file {}. Is the supplied '\
                                 'delimiter correct (={})?'.format(file_name, delimiter))

    if n_obc_nodes != len(nodes):
        raise RuntimeError('Error reading open boundary node list file.')

    return nodes


def add_fvcom_open_boundary_flags(nv, nbe, ob_nodes):
    """Add open boundary flags

    For each element, the method checks to see if two of the element's nodes lie
    on the open boundary. If they do, it flags the corresponding neighbour
    element with a -2, rather than a -1 as is the case in FVCOM output files.

    Parameters
    ----------
    nv : 2D ndarray, int
        Nodes surrounding element, shape (3, n_elems)

    nbe : 2D ndarray, int
        Elements surrounding element, shape (3, n_elems)

    ob_nodes : list, int
        List of nodes that lie along the open boundary

    Returns
    -------
    nbe_new : ndarray, int
        Array of neighbouring elements with open boundary flags added.
    """
    n_elems = nv.shape[1]

    nbe_new = nbe.copy()
    for i in range(n_elems):
        nodes = set(nv[:, i]).intersection(ob_nodes)
        if len(nodes) == 2:

            # Element borders the open boundary
            if len(nodes.intersection([nv[1, i], nv[2, i]])) == 2:
                nbe_new[0, i] = -2
            elif len(nodes.intersection([nv[2, i], nv[0, i]])) == 2:
                nbe_new[1, i] = -2
            elif len(nodes.intersection([nv[0, i], nv[1, i]])) == 2:
                nbe_new[2, i] = -2
            else:
                raise RuntimeError('Failed to identify open boundary.')
    return nbe_new


@cython.wraparound(True)
cpdef DTYPE_INT_t _get_number_of_matching_nodes(DTYPE_INT_t [:] array1, DTYPE_INT_t [:] array2):
    cdef DTYPE_INT_t i, j
    cdef DTYPE_INT_t matches

    matches = 0
    for i in range(array1.shape[0]):
        for j in range(array2.shape[0]):
            if array1[i] == array2[j]:
                matches = matches + 1

    return matches


__all__ = ['create_fvcom_grid_metrics_file',
           'create_arakawa_a_grid_metrics_file']
