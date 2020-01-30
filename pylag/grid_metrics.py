from __future__ import print_function

import numpy as np
from collections import OrderedDict
from netCDF4 import Dataset


class GridMetricsFileCreator(object):
    """ Grid metrics file creator

    Class to assist with the creation of PyLag grid metrics files

    Parameters:
    -----------
    file_name : str, optional
        The name of the grid metrics file that will be created.

    Returns:
    --------
    N/A
    """

    def __init__(self, file_name='./grid_metrics.nc', format="NETCDF4_CLASSIC"):
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

        Parameters:
        -----------
        N/A

        Returns:
        --------
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

        Parameters:
        -----------
        name : str
            Name of the dimension

        size : int
            Name of the string
        """
        self.dims[name] = self.ncfile.createDimension(name, size)

    def create_variable(self, var_name, var_data, dimensions, dtype, fill_value=None, attrs=None):
        """" Add variable

        Parameters:
        -----------
        var_name : str
            Name of the variable to add

        var_data : ndarray
            Data array

        dimensions : tuple
            Dimensions of the ndarray

        dtype : str
            Data type (e.g. float)

        fill_value : int, float ...
            Fill value to use

        attrs : dict
            Dictionary of attributes.
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

    NB This function only needs to be called once per FVCOM model grid - the
    grid metrics file generated can be reused by all future simulations.

    Parameters:
    -----------
    fvcom_file_name : str
        The path to an FVCOM output file that can be read in a processed

    obc_file_name : str
        The path to the text file containing a list of open boundary nodes

    grid_metrics_file_name : str, optional
        The name of the grid metrics file that will be created
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
    gm_file_creator = GridMetricsFileCreator()

    # Create skeleton file
    gm_file_creator.create_file()

    # Add dimension variables
    gm_file_creator.create_dimension('node', n_nodes)
    gm_file_creator.create_dimension('nele', n_elems)
    gm_file_creator.create_dimension('siglev', n_siglev)
    gm_file_creator.create_dimension('siglay', n_siglay)

    # Add grid coordinate variables
    # -----------------------------
    for var_name in ['x', 'y', 'xc', 'yc', 'lat', 'lon', 'latc', 'lonc', 'siglev', 'siglay', 'h']:
        nc_var = fvcom_dataset.variables[var_name]

        var_data = nc_var[:]

        dtype = nc_var.dtype.name

        dimensions = nc_var.dimensions

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
    dimensions = nv_var.dimensions
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
    open_boundary_nodes = get_open_boundary_nodes(obc_file_name)
    nbe_data = add_open_boundary_flags(nv_data, nbe_data, open_boundary_nodes)

    # Add variable
    dtype = nbe_var.dtype.name
    dimensions = nbe_var.dimensions
    attrs = {}
    for attr_name in nbe_var.ncattrs():
        attrs[attr_name] = nbe_var.getncattr(attr_name)
    gm_file_creator.create_variable('nbe', nbe_data, dimensions, dtype, attrs=attrs)

    # Close FVCOM dataset
    fvcom_dataset.close()

    # Close grid metrics file creator
    gm_file_creator.close_file()

    return


def sort_adjacency_array(nv, nbe):
    """Sort the adjacency array

    PyLag expects the adjacency array (nbe) to be sorted in a particlular way
    relative to the grid connectivity array (nv). NB The former lists the
    elements surrounding each element; the latter the nodes surrounding each
    element.

    Parameters:
    -----------
    nv : 2D ndarray, int
        Nodes surrounding element, shape (3, n_elems)

    nbe : 2D ndarray, int
        Elements surrounding element, shape (3, n_elems)

    Returns:
    --------
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


def sort_interpolants(a1u, a2u, nbe, nbe_sorted):
    """Sort interpolant arrays

    PyLag expects the arrays containing interpolation coefficients (a1u and a2u)
    to be sorted in the same way as the sorted nbe array (see above). This
    function matches entries in a{1,2}u to nbe_sorted, given that a{1,2}u is
    currently matched to the array nbe.

    Parameters:
    -----------
    a1u : 2D ndarray
        FVCOM interpolation coefficients, shape (4, n_elems)

    a2u : 2D ndarray
        FVCOM interpolation coefficients, shape (4, n_elems)

    nbe : 2D ndarray, int
        Elements surrounding each element, shape (3, n_elems)

    nbe_sorted : 2D ndarray, int
        Sorted nbe array, shape (3, n_elems)
    """

    n_elems = a1u.shape[1]

    if n_elems != a2u.shape[1] or n_elems != nbe.shape[1] or n_elems != nbe_sorted.shape[1]:
        raise ValueError('Array dimensions do not match')

    a1u_sorted = np.empty_like(a1u)
    a2u_sorted = np.empty_like(a2u)

    # Host element index is 0
    a1u_sorted[0, :] = a1u[0, :]
    a2u_sorted[0, :] = a2u[0, :]

    for j in range(n_elems):
        for i in range(3):
            if nbe_sorted[i, j] == nbe[0, j]:
                a1u_sorted[i + 1, j] = a1u[1, j]
                a2u_sorted[i + 1, j] = a2u[1, j]
            elif nbe_sorted[i, j] == nbe[1, j]:
                a1u_sorted[i + 1, j] = a1u[2, j]
                a2u_sorted[i + 1, j] = a2u[2, j]
            elif nbe_sorted[i, j] == nbe[2, j]:
                a1u_sorted[i + 1, j] = a1u[3, j]
                a2u_sorted[i + 1, j] = a2u[3, j]
            else:
                raise ValueError('Failed to match entry in nbe and nbe_sorted.')

    return a1u_sorted, a2u_sorted


def get_open_boundary_nodes(file_name):
    """Read open boundary nodes from file

    Parameters:
    -----------
    file_name : str
        Name of file containing a list of the open boundary nodes
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


def add_open_boundary_flags(nv, nbe, ob_nodes):
    """Add open boundary flags

    For each element, the method checks to see if two of the element's nodes lie
    on the open boundary. If they do, it flags the corresponding neighbour
    element with a -2, rather than a -1 as is the case in FVCOM output files.

    Parameters:
    -----------
    nv : 2D ndarray, int
        Nodes surrounding element, shape (3, n_elems)

    nbe : 2D ndarray, int
        Elements surrounding element, shape (3, n_elems)

    ob_nodes : list, int
        List of nodes that lie along the open boundary
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
