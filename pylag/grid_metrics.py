"""
The grid metrics module exists to assist with the creation of PyLag
grid metrics files.
"""

from __future__ import print_function

import numpy as np
from scipy.spatial import Delaunay
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

        self.vars[var_name][:] = var_data

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
    nv_data = nv_var[:] - 1
    dtype = nv_var.dtype.name
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
    nbe_data = nbe_var[:] - 1
    nbe_data = sort_adjacency_array(nv_data, nbe_data)

    # Add open boundary flags
    open_boundary_nodes = get_fvcom_open_boundary_nodes(obc_file_name)
    nbe_data = add_fvcom_open_boundary_flags(nv_data, nbe_data, open_boundary_nodes)

    # Add variable
    dtype = nbe_var.dtype.name
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


def create_arakawa_a_grid_metrics_file(file_name, has_mask=True, has_bathymetry=True,
                                       grid_metrics_file_name='./grid_metrics.nc'):
    """Create a Arakawa A-grid metrics file

    This function creates a grid metrics file for data defined on an Arakawa
    A-grid. The function is intended to work with regularly gridded, CF
    compliant datasets, which is usually a requirement for datasets submitted
    to public catalogues.

    The approach taken is to reinterpret the regular grid as a single, unstructured
    grid which can be understood by PyLag.

    Parameters
    ----------
    file_name : str
        The path to an file that can be read in a processed

    has_mask : bool
        Flag identifying whether the input file contains a variable mask, which will
        be used to generate the land sea mask. If it doesn't, the land sea mask is
        inferred from the surface mask of one of the variables. If the output files
        contain a time varying mask due to changes in sea surface elevation, a
        land sea mask should be provided. Optional, default True.

    has_bathymetry : bool
        Flag identifying whether the input file contains a variable h corresponding to the
        grid bathymetry. If it doesn't, the bathymetry is inferred from the depth mask
        of one of the variables. If the output files contain a time varying mask due to
        changes in sea surface elevation, the bathymetry should be provided. Optional, default
        True.

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

    # Read in coordinate variables. Use common names to try and ensure a hit.
    lon_var, lon_attrs = _get_variable(input_dataset, ['lon', 'longitude'])
    lat_var, lat_attrs = _get_variable(input_dataset, ['lat', 'latitude'])
    depth_var, depth_attrs = _get_variable(input_dataset, ['depth'])

    # Create points array
    lon2d, lat2d = np.meshgrid(lon_var[:], lat_var[:], indexing='ij')
    points = np.array([lon2d.flatten(order='C'), lat2d.flatten(order='C')]).T

    # Save lon and lat points at nodes
    lon_nodes = points[:, 0]
    lat_nodes = points[:, 1]
    n_nodes = points.shape[0]

    # Save original lon and lat sizes
    n_longitude = lon_var.shape[0]
    n_latitude = lat_var.shape[0]

    # Save depth
    depth = depth_var[:]
    n_levels = depth_var.shape[0]

    # Save bathymetry
    if has_bathymetry:
        bathy_var, bathy_attrs = _get_variable(input_dataset, ['h'])
        bathy = sort_axes(bathy_var).squeeze()
        if len(bathy.shape) == 2:
            bathy = bathy.reshape(np.prod(bathy.shape), order='C')
        else:
            raise RuntimeError('Bathymetry array is not 2D.')
    else:
        # Try to infer bathy from the depth mask for uo
        uo_var, _ = _get_variable(input_dataset, ['uo'])
        uo = sort_axes(uo_var)
        if not np.ma.isMaskedArray(uo):
            raise RuntimeError('Unable to generate bathymetry. Can you provide h?')

        # Read the depth mask and reshape giving (n_levels, n_nodes)
        uo = uo[0, :, :, :]  # Expect a 4D array. Take first time point.
        uo = uo.reshape(n_levels, np.prod(uo.shape[1:]), order='C')

        bathy = np.empty((uo.shape[1]), dtype=float)
        for i in range(bathy.shape[0]):
            uo_tmp = uo[:, i]
            if np.ma.count(uo_tmp) != 0:
                index = np.ma.flatnotmasked_edges(uo_tmp)[1]
                bathy[i] = depth[index]
            else:
                bathy[i] = 0.0

        # Add some standard attributes
        bathy_attrs = {'standard_name': 'depth',
                       'units': 'm',
                       'long_name': 'depth, measured down from the free surface',
                       'axis': 'Z',
                       'positive': 'down'}

    # Save mask
    if has_mask:
        mask_var, mask_attrs = _get_variable(input_dataset, ['mask'])

        # Generate land-sea mask at nodes
        land_sea_mask_nodes = sort_axes(mask_var).squeeze()
        if len(land_sea_mask_nodes.shape) < 2 or len(land_sea_mask_nodes.shape) > 3:
            raise ValueError('Unsupported land sea mask with shape {}'.format(land_sea_mask_nodes.shape))

        # Flip meaning yielding: 1 - masked land point, and 0 sea point.
        land_sea_mask_nodes = 1 - land_sea_mask_nodes

        # Use surface mask only if shape is 3D
        if len(land_sea_mask_nodes.shape) == 3:
            land_sea_mask_nodes = land_sea_mask_nodes[0, :, :]

        # Fix up long name to reflect flipping of mask
        mask_attrs['long_name'] = "Land-sea mask: sea = 0 ; land = 1"

    else:
        # Try to use the surface mask for uo
        uo_var, _ = _get_variable(input_dataset, ['uo'])
        uo = sort_axes(uo_var)
        if not np.ma.isMaskedArray(uo):
            raise RuntimeError('Unable to generate bathymetry. Can you provide h?')

        # Expect a 4D array. Take first time point and top depth level
        land_sea_mask_nodes = uo.mask[0, 0, :, :]

        # Add some standard attributes
        mask_attrs = {'standard_name': 'sea_binary_mask',
                      'units': '1',
                      'long_name': 'Land-sea mask: sea = 0, land = 1'}

    land_sea_mask_nodes = land_sea_mask_nodes.reshape(np.prod(land_sea_mask_nodes.shape), order='C')

    # Create the Triangulation
    tri = Delaunay(points)

    # Save simplices
    #   - Flip to reverse ordering, as expected by PyLag
    #   - Transpose to give it the dimension ordering expected by PyLag
    nv = np.flip(tri.simplices.copy(), axis=1).T
    n_elems = nv.shape[1]

    # Save lon and lat points at element centres
    lon_elements = np.empty(n_elems, dtype=float)
    lat_elements = np.empty(n_elems, dtype=float)
    for i, element in enumerate(range(n_elems)):
        lon_elements[i] = lon_nodes[(nv[:, element])].mean()
        lat_elements[i] = lat_nodes[(nv[:, element])].mean()

    # Save neighbours
    #   - Transpose to give it the dimension ordering expected by PyLag
    #   - Sort to ensure match with nv
    nbe = tri.neighbors.T
    nbe = sort_adjacency_array(nv, nbe)

    # Generate the land-sea mask at elements
    land_sea_mask_elements = np.empty(n_elems, dtype=int)
    for i in range(n_elems):
        element_nodes = nv[:, i]
        land_sea_mask_elements[i] = 1 if np.any(land_sea_mask_nodes[(element_nodes)] == 1) else 0

    # Flag open boundaries with -2 flag
    nbe[np.where(nbe == -1)] = -2

    # Flag land boundaries with -1 flag
    for i, mask in enumerate(land_sea_mask_elements):
        if mask == 1:
            nbe[np.where(nbe == i)] = -1

    # Create grid metrics file
    # ------------------------
    print('Creating grid metrics file {}'.format(grid_metrics_file_name))

    # Instantiate file creator
    gm_file_creator = GridMetricsFileCreator(grid_metrics_file_name)

    # Create skeleton file
    gm_file_creator.create_file()

    # Add dimension variables
    gm_file_creator.create_dimension('longitude', n_longitude)
    gm_file_creator.create_dimension('latitude', n_latitude)
    gm_file_creator.create_dimension('depth', n_levels)
    gm_file_creator.create_dimension('node', n_nodes)
    gm_file_creator.create_dimension('element', n_elems)

    # Add longitude at nodes
    gm_file_creator.create_variable('longitude', lon_nodes, ('node',), float, attrs=lon_attrs)

    # Add longitude at element centres
    gm_file_creator.create_variable('longitude_c', lon_elements, ('element',), float, attrs=lon_attrs)

    # Add latitude at nodes
    gm_file_creator.create_variable('latitude', lat_nodes, ('node',), float, attrs=lat_attrs)

    # Add latitude at element centres
    gm_file_creator.create_variable('latitude_c', lat_elements, ('element',), float, attrs=lat_attrs)

    # Depth
    gm_file_creator.create_variable('depth', depth, ('depth',), float, attrs=depth_attrs)

    # Bathymetry
    gm_file_creator.create_variable('h', bathy, ('node',), float, attrs=bathy_attrs)

    # Add simplices
    gm_file_creator.create_variable('nv', nv, ('three', 'element',), int,
                                    attrs={'long_name': 'nodes surrounding each element'})

    # Add neighbours
    gm_file_creator.create_variable('nbe', nbe, ('three', 'element',), int,
                                    attrs={'long_name': 'elements surrounding each element'})

    # Add land sea mask
    gm_file_creator.create_variable('mask', land_sea_mask_elements, ('element',), int, attrs=mask_attrs)

    # Close input dataset
    input_dataset.close()

    # Close grid metrics file creator
    gm_file_creator.close_file()

    return


def create_roms_grid_metrics_file(file_name,
                                  process_grid_u=True, process_grid_v=True, process_grid_rho=True,
                                  time_var_name='depth', depth_var_name='depth',
                                  time_dim_name='depth', depth_dim_name='depth',
                                  lon_var_name_grid_u='lon_u', lat_var_name_grid_u='lat_u',
                                  lon_var_name_grid_v='lon_v', lat_var_name_grid_v='lat_v',
                                  lon_var_name_grid_rho='lon_rho', lat_var_name_grid_rho='lat_rho',
                                  xi_dim_name_grid_u='xi_u', eta_dim_name_grid_u='eta_rho',
                                  xi_dim_name_grid_v='xi_rho', eta_dim_name_grid_v='eta_v',
                                  xi_dim_name_grid_rho='xi_rho', eta_dim_name_grid_rho='eta_rho',
                                  mask_name_grid_u=None, mask_name_grid_v=None, mask_name_grid_rho=None,
                                  reference_var_name_grid_u=None, reference_var_name_grid_v=None,
                                  reference_var_name_grid_rho=None, grid_metrics_file_name='./grid_metrics.nc'):
    """ Create a ROMS metrics file

    This function creates a grid metrics file for data generated by the Regional
    Ocean Model System (ROMS). ROMS uses an Arakawa C-grid. Here, three separate
    unstructured triangular grids are made from the three structured grids that
    make up the Arakawa C-grid. The three unstructured grids have nodes at the
    U-, V- and T-points respectively, which for the basis of the Arakawa C-grid.

    Default names are given for the latitude and longitude variables names on each
    grid. If the latitude and/or longitude variable names differ from the defaults,
    these should be passed in.

    TODO
    ----
    Understand and implement support for the vertical coordinate system used in
    ROMS.

    Parameters
    ----------
    file_name : str
        The path to an file that can be read in a processed

    time_var_name : str
        Name of the time variable

    depth_var_name : str
        Name of the depth variable

    time_dim_name : str
        Name of the time dimension

    depth_dim_name : str
        Name of the depth dimension

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

    grid_metrics_file_name : str, optional
        The name of the grid metrics file that will be created

    mask_name_grid_u : str, optional
        u-mask variable name.

    mask_name_grid_v : str, optional
        v-mask variable name.

    mask_name_grid_rho : str, optional
        rho-mask variable name.

    reference_var_name_grid_u : str, optional
        Reference u-grid variable, which can be used for extracting the u-mask.

    reference_var_name_grid_v : str, optional
        Reference v-grid variable, which can be used for extracting the u-mask.

    reference_var_name_grid_rho : str, optional
        Reference rho-grid variable, which can be used for extracting the u-mask.

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

    reference_var_names = {'grid_u': reference_var_name_grid_u,
                           'grid_v': reference_var_name_grid_v,
                           'grid_rho': reference_var_name_grid_rho}

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

    # Loop over all grids
    for grid_name in grid_names:
        if process_grid[grid_name] is False:
            continue

        print('Processing {} variables ...'.format(grid_name))

        # Read in coordinate variables.
        lon_var, lon_attrs[grid_name] = _get_variable(input_dataset, lon_var_names[grid_name])
        lat_var, lat_attrs[grid_name] = _get_variable(input_dataset, lat_var_names[grid_name])

        if len(lon_var.shape) != len(lat_var.shape):
            raise RuntimeError('Lon and lat var shapes do not match')

        if len(lon_var.shape) == 1:
            # Regular grid
            # ------------
            lon2d, lat2d = np.meshgrid(lon_var[:], lat_var[:], indexing='ij')

        elif len(lon_var.shape) == 2:
            # Curvilinear grid
            # ----------------

            # Longitude
            if lon_var.dimensions[0] == xi_grid_names[grid_name] and lon_var.dimensions[1] == eta_grid_names[grid_name]:
                lon2d = lon_var[:]
            elif lon_var.dimensions[0] == eta_grid_names[grid_name] and lon_var.dimensions[1] == xi_grid_names[grid_name]:
                lon2d = lon_var[:].T
            else:
                raise RuntimeError('Unexpected dimension name in lon variable')

            # Latitude
            if lat_var.dimensions[0] == xi_grid_names[grid_name] and lat_var.dimensions[1] == eta_grid_names[grid_name]:
                lat2d = lat_var[:]
            elif lat_var.dimensions[0] == eta_grid_names[grid_name] and lat_var.dimensions[1] == xi_grid_names[grid_name]:
                lat2d = lat_var[:].T
            else:
                raise RuntimeError('Unexpected dimension name in lat variable')
        else:
            raise RuntimeError('Unrecognised lon var shape')

        # Save mask
        if mask_var_names[grid_name] is not None:
            raise NotImplementedError('Support for named masks still needs to be implemented')
        else:
            # Try to use the surface mask for reference u-grid variable name
            ref_var, _ = _get_variable(input_dataset, [reference_var_names[grid_name]])
            ref_var = sort_axes(ref_var, time_name=time_dim_name, depth_name=depth_dim_name,
                                lat_name=eta_grid_names[grid_name], lon_name=xi_grid_names[grid_name])
            if not np.ma.isMaskedArray(ref_var):
                raise RuntimeError('Unable to generate mask. Can you provide a mask variable?')

            # Expect a 4D array. Take first time point and top depth level
            land_sea_mask_nodes[grid_name] = ref_var.mask[0, 0, :, :]

            # Add some standard attributes
            mask_attrs[grid_name] = {'standard_name': 'sea_binary_mask',
                                     'units': '1',
                                     'long_name': 'Land-sea mask: sea = 0, land = 1'}

        land_sea_mask_nodes[grid_name] = land_sea_mask_nodes[grid_name].reshape(np.prod(land_sea_mask_nodes[grid_name].shape), order='C')

        points = np.array([lon2d.flatten(order='C'), lat2d.flatten(order='C')]).T

        # Save lon and lat points at nodes
        lon_nodes[grid_name] = points[:, 0]
        lat_nodes[grid_name] = points[:, 1]
        nodes[grid_name] = points.shape[0]

        # Save original lon and lat sizes
        n_longitude[grid_name] = lon_var.shape[0]
        n_latitude[grid_name] = lat_var.shape[0]

        # Create the Triangulation
        tris[grid_name] = Delaunay(points)

        # Save simplices
        #   - Flip to reverse ordering, as expected by PyLag
        #   - Transpose to give it the dimension ordering expected by PyLag
        nvs[grid_name] = np.flip(tris[grid_name].simplices.copy(), axis=1).T
        elements[grid_name] = nvs[grid_name].shape[1]

        # Save lon and lat points at element centres
        lon_elements[grid_name] = np.empty(elements[grid_name], dtype=float)
        lat_elements[grid_name] = np.empty(elements[grid_name], dtype=float)
        for element in range(elements[grid_name]):
            lon_elements[grid_name][element] = lon_nodes[grid_name][(nvs[grid_name][:, element])].mean()
            lat_elements[grid_name][element] = lat_nodes[grid_name][(nvs[grid_name][:, element])].mean()

        # Save neighbours
        #   - Transpose to give it the dimension ordering expected by PyLag
        #   - Sort to ensure match with nv
        nbes[grid_name] = tris[grid_name].neighbors.T
        nbes[grid_name] = sort_adjacency_array(nvs[grid_name], nbes[grid_name])

        # Generate the land-sea mask at elements
        land_sea_mask_elements[grid_name] = np.empty(elements[grid_name], dtype=int)
        for i in range(elements[grid_name]):
            element_nodes = nvs[grid_name][:, i]
            land_sea_mask_elements[grid_name][i] = 1 if np.any(land_sea_mask_nodes[grid_name][element_nodes] == 1) else 0

        # Flag open boundaries with -2 flag
        nbes[grid_name][np.where(nbes[grid_name] == -1)] = -2

        # Flag land boundaries with -1 flag
        mask_indices = np.where(land_sea_mask_elements[grid_name] == 1)
        for index in mask_indices:
            nbes[grid_name][np.where(nbes[grid_name] == index)] = -1

    # Create grid metrics file
    # ------------------------
    print('Creating grid metrics file {}'.format(grid_metrics_file_name))

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
        gm_file_creator.create_dimension('depth_{}'.format(grid_name), 1) # TODO add depth data
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

        # Depth
        #gm_file_creator.create_variable('depth', depth, ('depth',), float, attrs=depth_attrs)

        # Bathymetry
        #gm_file_creator.create_variable('h', bathy, ('node',), float, attrs=bathy_attrs)

        # Add simplices
        gm_file_creator.create_variable('nv_{}'.format(grid_name), nvs[grid_name],
                                        ('three', 'element_{}'.format(grid_name),), int,
                                        attrs={'long_name': 'nodes surrounding each element'})

        # Add neighbours
        gm_file_creator.create_variable('nbe_{}'.format(grid_name), nbes[grid_name],
                                        ('three', 'element_{}'.format(grid_name),), int,
                                        attrs={'long_name': 'elements surrounding each element'})

        # Add land sea mask
        gm_file_creator.create_variable('mask_{}'.format(grid_name), land_sea_mask_elements[grid_name],
                                        ('element_{}'.format(grid_name),), int, attrs=mask_attrs[grid_name])

    # Close input dataset
    input_dataset.close()

    # Close grid metrics file creator
    gm_file_creator.close_file()

    return


def sort_axes(nc_var, time_name='depth', depth_name='depth', lat_name='latitude',
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
        The name of the latiitude dimension coordinate.

    lon_name : str, optional
        The name of the longitude dimension coordinate.

    Returns
    -------
    var : NumPy NDArray
        Variable array with sorted axes.
    """
    print('Sorting axes for variable {}'.format(nc_var.name))
    var = nc_var[:]
    dimensions = nc_var.dimensions
    if len(dimensions) == 2:
        lon_index = _get_dimension_index(dimensions, lat_name)

        # Shift axes to give [x, y]
        var = np.moveaxis(var, lon_index, 0)

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


def _get_variable(dataset, var_names):
    if type(var_names) == str:
        var_names = [var_names]

    var = None
    for var_name in var_names:
        try:
            var = dataset.variables[var_name]
        except KeyError:
            pass

    if var is not None:
        # Form dictionary of attributes
        attrs = {}
        for attr_name in var.ncattrs():
            attrs[attr_name] = var.getncattr(attr_name)

        return var, attrs

    raise RuntimeError('Variable not found in dataset')


def sort_adjacency_array(nv, nbe):
    """Sort the adjacency array

    PyLag expects the adjacency array (nbe) to be sorted in a particlular way
    relative to the grid connectivity array (nv). NB The former lists the
    elements surrounding each element; the latter the nodes surrounding each
    element.

    Parameters
    ----------
    nv : 2D ndarray, int
        Nodes surrounding element, shape (3, n_elems)

    nbe : 2D ndarray, int
        Elements surrounding element, shape (3, n_elems)

    Returns
    -------
    nbe_sorted: 2D ndarray, int
        The new nbe array
    """
    n_elems = nv.shape[1]

    # Our new to-be-sorted nbe array
    nbe_sorted = np.zeros([3, n_elems], dtype=np.int32) - 1

    # Loop over all elems
    for i in range(n_elems):
        side1, side2, side3 = _get_empty_arrays()

        side1[0] = nv[1, i]
        side1[1] = nv[2, i]
        side2[0] = nv[2, i]
        side2[1] = nv[0, i]
        side3[0] = nv[0, i]
        side3[1] = nv[1, i]

        index_side1 = -1
        index_side2 = -1
        index_side3 = -1
        for j in range(3):
            elem = nbe[j, i]
            if elem != -1:
                nv_test = nv[:, elem]
                if _get_number_of_matching_nodes(nv_test, side1) == 2:
                    index_side1 = elem
                elif _get_number_of_matching_nodes(nv_test, side2) == 2:
                    index_side2 = elem
                elif _get_number_of_matching_nodes(nv_test, side3) == 2:
                    index_side3 = elem
                else:
                    raise Exception('Failed to match side to test element.')

        nbe_sorted[0, i] = index_side1
        nbe_sorted[1, i] = index_side2
        nbe_sorted[2, i] = index_side3

    return nbe_sorted


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


def _get_empty_arrays():
    side1 = np.empty(2)
    side2 = np.empty(2)
    side3 = np.empty(2)
    return side1, side2, side3


def _get_number_of_matching_nodes(array1, array2):
    match = 0
    for a1 in array1:
        for a2 in array2:
            if a1 == a2: match = match + 1

    return match


__all__ = ['create_fvcom_grid_metrics_file',
           'create_arakawa_a_grid_metrics_file']
