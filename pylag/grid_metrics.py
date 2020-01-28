from __future__ import print_function

import numpy as np
import os
from netCDF4 import Dataset


def create_fvcom_grid_metrics_file(ncin_file_name, obc_file_name,
                                   ncout_file_name='./grid_metrics.nc'):
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
    ncin_file_name : str
        The path to an FVCOM output file that can be read in a processed
    obc_file_name : str
        The path to the text file containing a list of open boundary nodes
    ncout_file_name : str, optional
        The name of the grid metrics file that will be created
    """
    print('Creating FVCOM grid metrics file {}'.format(ncout_file_name))

    ob_nodes = get_open_boundary_nodes(obc_file_name)

    # Make new file with global attributes and copied variables
    var_to_copy = ['nv', 'nbe', 'x', 'y', 'xc', 'yc', 'lat', 'lon', 'latc',
                   'lonc', 'siglev', 'siglay', 'h', 'a1u', 'a2u']
    os.system("ncks -O -v " + ",".join(var_to_copy) + " " + ncin_file_name + " " + ncout_file_name)

    # Update nv, nbe, a1u and a2u
    # ---------------------------

    ds_in = Dataset(ncin_file_name, 'r')
    nv = ds_in.variables['nv'][:] - 1  # -1 for zero based numbering
    nbe = ds_in.variables['nbe'][:] - 1  # -1 for zero based numbering
    a1u = ds_in.variables['a1u'][:]
    a2u = ds_in.variables['a2u'][:]

    # Sort the adjacency array
    nbe_sorted = sort_adjacency_array(nv, nbe)

    # Sort interpolation coefficients to match nbe_sorted
    a1u_sorted, a2u_sorted = sort_interpolants(a1u, a2u, nbe, nbe_sorted)

    # Flag open boundaries
    nbe_sorted_with_ob_flags = add_open_boundary_flags(nv, nbe_sorted, ob_nodes)

    # Write updates to the nv, nbe, a1u and a2u variables to file
    ds_out = Dataset(ncout_file_name, 'a')
    ds_out.variables['nv'][:] = nv[:]
    ds_out.variables['nbe'][:] = nbe_sorted_with_ob_flags[:]
    ds_out.variables['a1u'][:] = a1u_sorted[:]
    ds_out.variables['a2u'][:] = a2u_sorted[:]

    # Close files
    ds_in.close()
    ds_out.close()


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