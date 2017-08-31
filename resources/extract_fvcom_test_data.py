""" Script used to extract and create test data using an FVCOM output file

This script is here for reference purposes only. All grid and variable values
are specified in MockFVCOMMediator.
"""


import numpy as np
from netCDF4 import Dataset

# Dimensions
three_dimname = 'three'
nthree = 3
four_dimname = 'four'
nfour = 4
elems_dimname = 'nele'
nelems = 4
nodes_dimname = 'node'
nnodes = 6
siglev_dimname = 'siglev'
nsiglev = 4
siglay_dimname = 'siglay'
nsiglay = 3
time_dimname = 'time'
ntime = None

print 'Creating test data input file.'
ncfile = Dataset('fvcom_data_test.nc', mode='w', format='NETCDF4_CLASSIC')

ncfile.title = 'FVCOM test data.'
ncfile.createDimension(elems_dimname, nelems)
ncfile.createDimension(nodes_dimname, nnodes)
ncfile.createDimension(siglev_dimname, nsiglev)
ncfile.createDimension(siglay_dimname, nsiglay)
ncfile.createDimension(time_dimname, ntime)
ncfile.createDimension(three_dimname, nthree)
ncfile.createDimension(four_dimname, nfour)

# Extract variable attribute data
vars_to_extract = ['x', 'y', 'xc', 'yc', 'lat', 'lon', 'latc', 'lonc', 'siglay', 'siglev', 'h',
                   'nv', 'nbe', 'time', 'zeta', 'a1u', 'a2u', 'u', 'v', 'ww', 'kh',
                   'viscofh', 'wet_cells'] 
ds = Dataset('irish_sea_v20_0001.nc','r')

# Create variables with variable attributes
test_vars = {}
for var_name in vars_to_extract:
    var = ds.variables[var_name]
    test_vars[var_name] = ncfile.createVariable(var_name, var.dtype, var.dimensions)
    for attr in var.ncattrs():
        value = var.getncattr(attr)
        test_vars[var_name].setncattr(attr, value)

# Central element to extract data for
central_element = 3
central_element_python_idx = central_element - 1 # For python 0-based indexing

# Elements to extract data for (4 in total)
neighbour_elements = ds.variables['nbe'][:, central_element_python_idx].squeeze().tolist()
all_elements = [central_element] + neighbour_elements
print '\nElements:'
print 'Central element: {}'.format(central_element)
print 'Neighbour elements: {}'.format(neighbour_elements)
print 'Extracting data for elements: {}'.format(all_elements)

# Nodes to extract data for (6 in total)
central_nodes = ds.variables['nv'][:, central_element_python_idx].squeeze().tolist()
neighbour_nodes = []
for neighbour_element in neighbour_elements:
    neighbour_element_python_idx = neighbour_element - 1 # For python 0-based indexing
    neighbour_nodes = neighbour_nodes + ds.variables['nv'][:, neighbour_element_python_idx].squeeze().tolist()
all_nodes = np.unique(central_nodes + neighbour_nodes).tolist()
print '\nNodes:'
print 'Central nodes: {}'.format(central_nodes)
print 'Neighbour nodes: {}'.format(neighbour_nodes)
print 'Extracting data for nodes: {}'.format(all_nodes)

# Creating mappings for element indices
node_map = {}
for idx, node in enumerate(all_nodes):
    test_node = idx + 1 # For 1-based numbering
    node_map[node] = test_node

# Fill nv using node mapping
for idx, element in enumerate(all_elements):
    element_python_idx = element - 1
    nodes = ds.variables['nv'][:, element_python_idx].squeeze().tolist()

    test_nodes = []    
    for node in nodes:
        test_nodes.append(node_map[node])
    test_vars['nv'][:,idx] = test_nodes

# Creating mapping for element indices
element_map = {}
for idx, element in enumerate(all_elements):
    test_element = idx + 1 # For 1-based numbering
    element_map[element] = test_element

# Fill nbe using element mapping. Replace indices for elements that will not
# be extracted with zeros which indicate boundary elements
for idx, element in enumerate(all_elements):
    element_python_idx = element - 1
    neighbour_elements = ds.variables['nbe'][:, element_python_idx].squeeze().tolist()

    test_elements = []
    for neighbour_element in neighbour_elements:
        if neighbour_element in all_elements:
            test_elements.append(element_map[neighbour_element])
        else:
            test_elements.append(0)
    test_vars['nbe'][:,idx] = test_elements

# Extract x, y, xc and yc variables from the original file
test_vars['x'][:] = [ds.variables['x'][i-1] for i in all_nodes]
test_vars['y'][:] = [ds.variables['y'][i-1] for i in all_nodes]
test_vars['xc'][:] = [ds.variables['xc'][i-1] for i in all_elements]
test_vars['yc'][:] = [ds.variables['yc'][i-1] for i in all_elements]

# Extract lat, lon, latc and lonc variables from the original file
test_vars['lon'][:] = [ds.variables['x'][i-1] for i in all_nodes]
test_vars['lat'][:] = [ds.variables['y'][i-1] for i in all_nodes]
test_vars['lonc'][:] = [ds.variables['xc'][i-1] for i in all_elements]
test_vars['latc'][:] = [ds.variables['yc'][i-1] for i in all_elements]

# Extract a1u and a2u interpolants from the original file
for idx, element in enumerate(all_elements):
    element_python_idx = element - 1
    a1u = ds.variables['a1u'][:, element_python_idx].squeeze().tolist()
    a2u = ds.variables['a2u'][:, element_python_idx].squeeze().tolist()
    
    test_element_python_idx = element_map[element] - 1
    test_vars['a1u'][:,test_element_python_idx] = a1u
    test_vars['a2u'][:,test_element_python_idx] = a2u

# Add data for two time points
test_vars['time'][:] = ds.variables['time'][:2].tolist()

# Siglevs (in sigma coordinates) are imposed
siglevs = [0.0, -0.2, -0.8, -1.0]
for i, siglev in enumerate(siglevs):
    test_vars['siglev'][i, :] = siglev

# Siglays (in sigma coordinates) are imposed and chosen to match siglevs
siglays = [-0.1, -0.5, -0.9]
for i, siglay in enumerate(siglays):
    test_vars['siglay'][i, :] = siglay

# h is imposed
test_vars['h'][:] = [10.0, 11.0, 10.0, 11.0, 11.0, 10.0]  # Host element h = 11.0

# zeta is imposed
test_vars['zeta'][0,:] = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0] # Host element zeta = 1.0
test_vars['zeta'][1,:] = [0.0, 2.0, 0.0, 2.0, 2.0, 0.0] # Host element zeta = 2.0

# u/v are imposed, equal across elements, decreasing with depth and increasing in time
uvw_t0 = [2.0, 1.0, 0.0]
uvw_t1 = [4.0, 2.0, 0.0]
for i, uvw in enumerate(uvw_t0): 
    test_vars['u'][0, i, :] = uvw
    test_vars['v'][0, i, :] = uvw
    test_vars['ww'][0, i, :] = uvw
for i, uvw in enumerate(uvw_t1): 
    test_vars['u'][1, i, :] = uvw
    test_vars['v'][1, i, :] = uvw
    test_vars['ww'][1, i, :] = uvw

# kh is imposed, equal across nodes, variable with depth and increasing in time
kh_t0 = [0.0, 0.01, 0.01, 0.0]
kh_t1 = [0.0, 0.1, 0.1, 0.0]
for i, kh in enumerate(kh_t0): 
    test_vars['kh'][0, i, :] = kh
for i, kh in enumerate(kh_t1): 
    test_vars['kh'][1, i, :] = kh

# viscofh is imposed, equal across nodes, variable with depth and increasing in time
viscofh_t0 = [0.01, 0.01, 0.0]
viscofh_t1 = [0.1, 0.1, 0.0]
for i, viscofh in enumerate(viscofh_t0): 
    test_vars['viscofh'][0, i, :] = viscofh
for i, viscofh in enumerate(viscofh_t1): 
    test_vars['viscofh'][1, i, :] = viscofh

# wet_cells is imposed, equal across elements and through time
test_vars['wet_cells'][0,:] = [1, 1, 1, 1]
test_vars['wet_cells'][1,:] = [1, 1, 1, 1]

ds.close()
ncfile.close()

