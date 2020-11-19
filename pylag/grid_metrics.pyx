"""
The grid metrics module exists to assist with the creation of PyLag
grid metrics files.
"""


# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cimport numpy as np

from joblib import Parallel, delayed
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import stripy as stripy
from collections import OrderedDict
from netCDF4 import Dataset
import time


class GridMetricsFileCreator(object):
    """ Grid metrics file creator

    Class to assist with the creation of PyLag grid metrics files

    Parameters
    ----------
    file_name : str, optional
        The name of the grid metrics file that will be created.

    format : str, optional
        The format of the NetCDF file (e.g. NetCDF4). Default: NetCDF4.

    """

    def __init__(self, file_name='./grid_metrics.nc', format="NETCDF4"):
        self.file_name = file_name

        self.format = format

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
        global_attrs['comment'] = ""

        self.ncfile.setncatts(global_attrs)

        return global_attrs

    def close_file(self):
        """ Close the file
        """
        try:
            self.ncfile.close()
        except:
            raise RuntimeError('Problem closing file')


def create_fvcom_grid_metrics_file(fvcom_file_name, obc_file_name, grid_metrics_file_name = './grid_metrics.nc'):
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
    open_boundary_nodes = get_fvcom_open_boundary_nodes(obc_file_name)
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

    # Close FVCOM dataset
    # -------------------
    fvcom_dataset.close()

    # Close grid metrics file creator
    gm_file_creator.close_file()

    return


def create_arakawa_a_grid_metrics_file(file_name, lon_var_name='longitude',lat_var_name='latitude',
                                       depth_var_name='depth', surface_only=False, mask_var_name=None,
                                       reference_var_name=None, bathymetry_var_name=None, stripy_nbe=True,
                                       num_threads=1, prng_seed=10, grid_metrics_file_name='./grid_metrics.nc'):
    """ Create a Arakawa A-grid metrics file

    This function creates a grid metrics file for data defined on a regular rectilinear
    Arakawa A-grid. The approach taken is to reinterpret the regular grid as a single,
    unstructured grid which can be understood by PyLag.

    We use stripy to create a spherical triangulation on the surface of the Earth. This
    may be a full global triangulation or a regional triangulation, depending on the
    input grid. In the version tested (stripy 2.02), stripy requires that the first three
    lat/lon points don't lie on a Great Circle. As this will typically be the case with
    regularly gridded data, we must permute the lat and lon arrays. To achieve this we
    use NumPy to shuffle array indices. These shuffled indices are used to permute data arrays.
    The shuffled indices are also saved as an extra variable in the grid metrics file
    so that the same shuffling can be applied to the time dependent variable arrays when PyLag
    is running. To make the operation reproducible, a fixed seed is used for the PRNG. The seed
    can be specified using the optional argument `prng`.

    If `surface_only` is set to `True`, only 2D surface grid data is extracted and saved.
    Currently, it is assumed the 0 index corresponds to the ocean's surface.

    Certain operations can be computationally expensive on large grids. Threading can be
    used to achieve some speed ups. The number of threads is set via the`num_threads`
    option. However, this will not help with the determination of elements neighbours,
    which is computed internally by stripy. In experiments with large grids (including a
    global 1/12 deg. grid) this operation took several hours to complete. The problem is
    observed to grow with the square of the number of elements.

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

    surface_only : bool, optional
        If False, process depth and bathymetry. If True, only process variables
        specific to the horizontal grid. Set to False if you want to do 3D
        transport modelling, and True if you want to do 2D surface only
        transport modeling. Optional, default : False.

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

    stripy_nbe : bool, optional
        Use `stripy` to compute neighbour simplices if True. If False, create a K-D Tree
        of nearest neighbours and use it to compute neighbour simplices. The former method
        is generally faster for smaller meshes, but suffers from scaling problems with
        larger meshes. Optional, default : True.

    num_threads : int, optional
        The number of threads to use when using threading.

    prng_seed : int, optional
        Seed for the pseudo random number generator.

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
                             'masked variable must be given in order to compute and save the bathymetry,'\
                             'as required by PyLag when running with input fields.')

    # Open the input file for reading
    input_dataset = Dataset(file_name, 'r')

    # Ensure masked variables are indeed masked
    input_dataset.set_auto_maskandscale(True)

    # Read in coordinate variables. Use common names to try and ensure a hit.
    print('Reading the grid:')
    lon_var, lon_attrs_orig = _get_variable(input_dataset, lon_var_name)
    lat_var, lat_attrs_orig = _get_variable(input_dataset, lat_var_name)

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
    lat_alpha = float(lat_var[0])
    lat_omega = float(lat_var[-1])
    trim_first_latitude = 0
    trim_last_latitude = 0
    if lat_alpha == float(-90.0) or lat_alpha == float(90.0):
        print('Trimming first latitude which sits over a pole ({} deg.)'.format(lat_alpha))
        lat_var = lat_var[1:]
        trim_first_latitude = 1
    if lat_omega == float(-90.0) or lat_omega == float(90.0):
        print('Trimming last latitude which sits over a pole ({} deg.)'.format(lat_omega))
        lat_var = lat_var[:-1]
        trim_last_latitude = 1

    # Create points array
    lon2d, lat2d = np.meshgrid(lon_var[:], lat_var[:], indexing='ij')
    points = np.array([lon2d.flatten(order='C'), lat2d.flatten(order='C')], dtype=DTYPE_FLOAT).T

    # Save lon and lat points at nodes
    lon_nodes = points[:, 0]
    lat_nodes = points[:, 1]
    n_nodes = points.shape[0]

    # Save original lon and lat sizes
    n_longitude = lon_var.shape[0]
    n_latitude = lat_var.shape[0]

    # Save depth
    if surface_only is False:
        depth_var, depth_attrs = _get_variable(input_dataset, depth_var_name)
        depth = depth_var[:]
        n_levels = depth_var.shape[0]

    # Read in the reference variable if needed
    if mask_var_name is None or (surface_only is False and bathymetry_var_name is None):
        ref_var, _ = _get_variable(input_dataset, reference_var_name)
        ref_var = sort_axes(ref_var)

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
    print('\nIdentifying neighbour simplices ...')
    if stripy_nbe:
        nbe = np.asarray(tri.neighbour_simplices(), dtype=DTYPE_INT)
    else:
        nbe = identify_neighbour_simplices(tri)
    print('done')

    # Save bathymetry
    if surface_only is False:
        print('\nGenerating the bathymetry:')
        if bathymetry_var_name:
            bathy_var, bathy_attrs = _get_variable(input_dataset, bathymetry_var_name)
            bathy = sort_axes(bathy_var).squeeze()

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
        land_sea_mask_nodes = sort_axes(mask_var).squeeze()
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
    xc, yc = tri.face_midpoints()
    lon_elements = np.degrees(xc)
    lat_elements = np.degrees(yc)
    print('done')

    # Generate the land-sea mask at elements
    print('\nGenerating land sea mask at element centres ', end='... ')
    land_sea_mask_elements = np.empty(n_elems, dtype=DTYPE_INT)
    compute_land_sea_element_mask(nv, land_sea_mask_nodes, land_sea_mask_elements)
    print('done')

    # Transpose nv and nbe arrays to give the dimension ordering expected by PyLag
    nv = nv.T
    nbe = nbe.T

    # Flag open boundaries with -2 flag
    print('\nFlagging open boundaries ', end='... ')
    nbe[np.asarray(nbe == -1).nonzero()] = -2
    print('done')

    # Flag land elements with -1 flag
    print('\nFlag land elements in neighbour array ', end='... ')
    land_elements = np.asarray(land_sea_mask_elements == 1).nonzero()[0]

    # Save a copy of nbe's shape and flatten
    nbe_shp = nbe.shape
    nbe = nbe.flatten()

    if num_threads > 1:
        nbe_split = np.array_split(nbe, num_threads)

        with Parallel(n_jobs=num_threads, backend='threading') as parallel:
            parallel([delayed(flag_land_elements_wrapper, check_pickle=False)(nbe_split[i], land_elements) for i in range(num_threads)])

        # Join the arrays
        nbe = np.concatenate(nbe_split)
    else:
        for element in land_elements:
            nbe[np.asarray(nbe == element).nonzero()] = -1

    # Reshape the original array and return
    nbe = np.asarray(nbe).reshape(nbe_shp)
    print('done')

    # Create grid metrics file
    # ------------------------
    print('\nCreating grid metrics file {} '.format(grid_metrics_file_name), end='... ')

    # Instantiate file creator
    gm_file_creator = GridMetricsFileCreator(grid_metrics_file_name)

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
    gm_file_creator.create_variable('mask', land_sea_mask_elements, ('element',), DTYPE_INT, attrs=mask_attrs)

    # Add land sea mask
    gm_file_creator.create_variable('mask_nodes', land_sea_mask_nodes, ('node',), DTYPE_INT, attrs=mask_attrs)

    # Close input dataset
    input_dataset.close()

    # Close grid metrics file creator
    gm_file_creator.close_file()

    print('done')

    return


def create_roms_grid_metrics_file(file_name,
                                  process_grid_u=True, process_grid_v=True, process_grid_rho=True,
                                  lon_var_name_grid_u='lon_u', lat_var_name_grid_u='lat_u',
                                  lon_var_name_grid_v='lon_v', lat_var_name_grid_v='lat_v',
                                  lon_var_name_grid_rho='lon_rho', lat_var_name_grid_rho='lat_rho',
                                  xi_dim_name_grid_u='xi_u', eta_dim_name_grid_u='eta_rho',
                                  xi_dim_name_grid_v='xi_rho', eta_dim_name_grid_v='eta_v',
                                  xi_dim_name_grid_rho='xi_rho', eta_dim_name_grid_rho='eta_rho',
                                  mask_name_grid_u=None, mask_name_grid_v=None, mask_name_grid_rho='mask_rho',
                                  bathymetry_var_name='h', angles_var_name=None, num_threads=1,
                                  grid_metrics_file_name='./grid_metrics.nc', **kwargs):
    """ Create a ROMS metrics file

    This function creates a grid metrics file for data generated by the Regional
    Ocean Model System (ROMS). ROMS uses an Arakawa C-grid. Here, three separate
    unstructured triangular grids are made from the three structured grids that
    make up the Arakawa C-grid. The three unstructured grids have nodes at the
    U-, V- and T-points respectively, which for the basis of the Arakawa C-grid.

    Default names are given for the latitude and longitude variables names on each
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

    xi_dim_name_grid_u : str, optional
        The name of the u-grid xi dimension name.

    xi_dim_name_grid_v : str, optional
        The name of the v-grid xi dimension name.

    xi_dim_name_grid_rho : str, optional
        The name of the rho-grid xi dimension name.

    eta_dim_name_grid_u : str, optional
        The name of the u-grid eta dimension name.

    eta_dim_name_grid_v : str, optional
        The name of the v-grid eta dimension name.

    eta_dim_name_grid_rho : str, optional
        The name of the rho-grid eta dimension name.

    mask_name_grid_u : str, optional
        u-mask variable name.

    mask_name_grid_v : str, optional
        v-mask variable name.

    mask_name_grid_rho : str, optional
        rho-mask variable name.

    bathymetry_var_name : str, optional
        Bathymetry variable name.

    angles_var_name : str, optional
        Angles at rho points

    num_threads : int, optional
        Number of threads to use

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
    grid_names = ['grid_u', 'grid_v', 'grid_rho']
    process_grid = {'grid_u': process_grid_u,
                    'grid_v': process_grid_v,
                    'grid_rho': process_grid_rho}
    lon_var_names = {'grid_u': lon_var_name_grid_u,
                     'grid_v': lon_var_name_grid_v,
                     'grid_rho': lon_var_name_grid_rho}

    lat_var_names = {'grid_u': lat_var_name_grid_u,
                     'grid_v': lat_var_name_grid_v,
                     'grid_rho': lat_var_name_grid_rho}

    xi_grid_names = {'grid_u': xi_dim_name_grid_u,
                     'grid_v': xi_dim_name_grid_v,
                     'grid_rho': xi_dim_name_grid_rho}

    eta_grid_names = {'grid_u': eta_dim_name_grid_u,
                      'grid_v': eta_dim_name_grid_v,
                      'grid_rho': eta_dim_name_grid_rho}

    mask_var_names = {'grid_u': mask_name_grid_u,
                      'grid_v': mask_name_grid_v,
                      'grid_rho': mask_name_grid_rho}

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

        # Save mask
        if mask_var_names[grid_name] is not None:
            mask_var, _ = _get_variable(input_dataset, mask_var_names[grid_name])

            # Generate land-sea mask at nodes
            land_sea_mask_nodes[grid_name] = sort_axes(mask_var, lat_name=eta_grid_names[grid_name],
                                            lon_name=xi_grid_names[grid_name]).squeeze()
            land_sea_mask_nodes[grid_name] = np.asarray(land_sea_mask_nodes[grid_name], dtype=DTYPE_INT)
            if len(land_sea_mask_nodes[grid_name].shape) < 2 or len(land_sea_mask_nodes[grid_name].shape) > 3:
                raise ValueError('Unsupported land sea mask with shape {}'.format(land_sea_mask_nodes[grid_name].shape))

            # Flip meaning yielding: 1 - masked land point, and 0 sea point.
            land_sea_mask_nodes[grid_name] = 1 - land_sea_mask_nodes[grid_name]

            # Use surface mask only if shape is 3D
            if len(land_sea_mask_nodes[grid_name].shape) == 3:
                land_sea_mask_nodes[grid_name] = land_sea_mask_nodes[grid_name][0, :, :]

            # Fix up long name to reflect flipping of mask
            mask_attrs[grid_name] = {'standard_name': '{} mask'.format(grid_name),
                                     'units': 1,
                                     'long_name': "Land-sea mask: sea = 0 ; land = 1"}

            land_sea_mask_nodes[grid_name] = land_sea_mask_nodes[grid_name].reshape(np.prod(land_sea_mask_nodes[grid_name].shape), order='C')

        else:
            mask_attrs[grid_name] = None
            land_sea_mask_nodes[grid_name] = None

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
        lon_elements[grid_name] = np.empty(elements[grid_name], dtype=DTYPE_FLOAT)
        lat_elements[grid_name] = np.empty(elements[grid_name], dtype=DTYPE_FLOAT)
        compute_element_centre_means(nvs[grid_name], lon_nodes[grid_name], lon_elements[grid_name])
        compute_element_centre_means(nvs[grid_name], lat_nodes[grid_name], lat_elements[grid_name])
        print('done')

        # Generate the land-sea mask at elements
        print('Generating land sea mask at element centres ', end='... ')
        if mask_var_names[grid_name] is not None:
            land_sea_mask_elements[grid_name] = np.empty(elements[grid_name], dtype=DTYPE_INT)
            compute_land_sea_element_mask(nvs[grid_name], land_sea_mask_nodes[grid_name], land_sea_mask_elements[grid_name])
        else:
            land_sea_mask_elements[grid_name] = None
        print('done')

        # Transpose to give ordering expected by PyLag
        nvs[grid_name] = nvs[grid_name].T
        nbes[grid_name] = nbes[grid_name].T

        # Flag open boundaries with -2 flag
        nbes[grid_name][np.asarray(nbes[grid_name] == -1).nonzero()] = -2

        print('Flag land elements in neighbour array ', end='... ')
        if mask_var_names[grid_name] is not None:
            land_elements = np.asarray(land_sea_mask_elements[grid_name] == 1).nonzero()[0]

            # Save a copy of nbe's shape and flatten
            nbe_shp = nbes[grid_name].shape
            nbe = nbes[grid_name].flatten()

            if num_threads > 1:
                nbe_split = np.array_split(nbe, num_threads)

                with Parallel(n_jobs=num_threads, backend='threading') as parallel:
                    parallel([delayed(flag_land_elements_wrapper, check_pickle=False)(nbe_split[i], land_elements) for i in range(num_threads)])

                # Join the arrays
                nbe = np.concatenate(nbe_split)
            else:
                for element in land_elements:
                    nbe[np.asarray(nbe == element).nonzero()] = -1

            # Reshape the original array and return
            nbes[grid_name] = np.asarray(nbe).reshape(nbe_shp)
        print('done')

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
                                        ('node_{}'.format(grid_name),), float, attrs=lon_attrs[grid_name])

        # Add longitude at element centres
        gm_file_creator.create_variable('longitude_c_{}'.format(grid_name), lon_elements[grid_name],
                                        ('element_{}'.format(grid_name),), float, attrs=lon_attrs[grid_name])

        # Add latitude at nodes
        gm_file_creator.create_variable('latitude_{}'.format(grid_name), lat_nodes[grid_name],
                                        ('node_{}'.format(grid_name),), float, attrs=lat_attrs[grid_name])

        # Add latitude at element centres
        gm_file_creator.create_variable('latitude_c_{}'.format(grid_name), lat_elements[grid_name],
                                        ('element_{}'.format(grid_name),), float, attrs=lat_attrs[grid_name])

        # Add simplices
        gm_file_creator.create_variable('nv_{}'.format(grid_name), nvs[grid_name],
                                        ('three', 'element_{}'.format(grid_name),), int,
                                        attrs={'long_name': 'nodes surrounding each element'})

        # Add neighbours
        gm_file_creator.create_variable('nbe_{}'.format(grid_name), nbes[grid_name],
                                        ('three', 'element_{}'.format(grid_name),), int,
                                        attrs={'long_name': 'elements surrounding each element'})

        # Add land sea mask - nodes
        if mask_var_names[grid_name] is not None:
            gm_file_creator.create_variable('mask_{}'.format(grid_name), land_sea_mask_nodes[grid_name],
                                            ('node_{}'.format(grid_name),), int, attrs=mask_attrs[grid_name])

            # Add land sea mask - elements
            gm_file_creator.create_variable('mask_c_{}'.format(grid_name), land_sea_mask_elements[grid_name],
                                            ('element_{}'.format(grid_name),), int, attrs=mask_attrs[grid_name])

    # Bathymetry
    gm_file_creator.create_variable('h', bathy, ('node_grid_rho',), float, attrs=bathy_attrs)

    # Angles
    if angles_var_name is not None:
        gm_file_creator.create_variable('angles_grid_rho', angles, ('node_grid_rho',), float, attrs=angles_attrs)

    # Add dimensions and variables describing the vertical grid. No transforms are done here.
    gm_file_creator.create_dimension('s_rho', input_dataset.dimensions['s_rho'].size)
    gm_file_creator.create_dimension('s_w', input_dataset.dimensions['s_w'].size)

    for key in vertical_grid_var_names.keys():
        var = vertical_grid_vars[key]
        var_data = np.asarray(var[:])
        attrs = vertical_grid_var_attrs[key]
        gm_file_creator.create_variable(key, var_data, var.dimensions, float, attrs=attrs)

    gm_file_creator.create_variable('vtransform', np.asarray(vtransform, dtype=float), (), float,
                                    attrs={'long_name': 'vertical terrain following transformation equation'})

    # Close input dataset
    input_dataset.close()

    # Close grid metrics file creator
    gm_file_creator.close_file()

    return


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
    print("Sorting axes for variable `{}`".format(nc_var.name), end='...')

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
        print('Failed to find dimension index. Dimensions were {}; supplied '
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
    # Number of simplices
    simplices = stri.simplices
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
    found = np.zeros(n_simplices)

    # Neighbour array, initialised to -1. This ensures simplices lying along an
    # open boundary are handled correctly.
    nbe = -1 * np.ones_like(simplices)

    # Iteratively find all neighbours for all elements.
    for k in k_values:
        if verbose:
            print('Searching for adjoining neighbours with k = {}'.format(k))

        simplex_indices = np.asarray(found==0).nonzero()[0]

        mids_subset = mids[simplex_indices]

        _, nbe_tree = tree.query(mids_subset, k=k, distance_upper_bound=2.0)

        n_simplices_subset = simplex_indices.shape[0]
        for i in range(n_simplices_subset):

            simplex_idx = simplex_indices[i]

            counter = 0
            for j in range(k):
                neighbour_idx = nbe_tree[i, j]
                if np.in1d(simplices[simplex_idx, :], simplices[neighbour_idx, :]).sum() == 2:
                    nbe[simplex_idx, counter] = neighbour_idx
                    counter += 1

                # Check if all neighbours were found
                if counter == 3:
                    found[simplex_idx] = 1
                    break

        if verbose:
            print('Found {} %'.format(100*np.count_nonzero(found)/n_simplices))

        if np.count_nonzero(found) == n_simplices:
            return nbe

    return nbe

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


cpdef compute_element_centre_means(const DTYPE_INT_t [:,:] nv, const DTYPE_FLOAT_t [:] nodal_values,
                                   DTYPE_FLOAT_t [:] element_values):
    cdef DTYPE_INT_t n_elements, n_vertices
    cdef DTYPE_INT_t node
    cdef DTYPE_INT_t i, j
    cdef DTYPE_FLOAT_t sum

    n_elements = nv.shape[0]
    n_vertices = nv.shape[1]

    for i in range(n_elements):
        sum = 0.0
        for j in range(n_vertices):
            node = nv[i, j]
            sum += nodal_values[node]
        element_values[i] = sum / n_vertices


cpdef compute_land_sea_element_mask(const DTYPE_INT_t [:,:] nv, const DTYPE_INT_t [:] nodal_mask,
                                    DTYPE_INT_t [:] element_mask):
    cdef DTYPE_INT_t node
    cdef DTYPE_INT_t n_elements, n_vertices
    cdef DTYPE_INT_t i

    n_elements = nv.shape[0]
    n_vertices = nv.shape[1]

    element_mask[:] = 0
    for i in range(n_elements):
        for j in range(n_vertices):
            node = nv[i, j]
            if nodal_mask[node] == 1:
                element_mask[i] = 1


cdef void flag_land_elements(DTYPE_INT_t* nbe, DTYPE_INT_t* land_elements, DTYPE_INT_t n, DTYPE_INT_t m) nogil:

    cdef DTYPE_INT_t i, j

    for i in range(n):
        for j in range(m):
            if nbe[i] == land_elements[j]:
                nbe[i] = -1


cpdef flag_land_elements_wrapper(np.ndarray nbe, np.ndarray land_elements):
    cdef DTYPE_INT_t* nbe_ptr = <DTYPE_INT_t*> nbe.data
    cdef DTYPE_INT_t* land_elements_ptr = <DTYPE_INT_t*> land_elements.data
    cdef DTYPE_INT_t n = nbe.shape[0]
    cdef DTYPE_INT_t m = land_elements.shape[0]

    with nogil:
        flag_land_elements(nbe_ptr, land_elements_ptr, n, m)


def get_fvcom_open_boundary_nodes(file_name):
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
        nodes.append(int(line.strip().split(' ')[1]))

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
