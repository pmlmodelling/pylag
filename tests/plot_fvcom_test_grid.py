from matplotlib import pyplot as plt
from matplotlib.tri.triangulation import Triangulation

from fvcom_data_reader_test import MockFVCOMMediator

mediator = MockFVCOMMediator()

# Read in the grid's dimensions
n_nodes = mediator.get_dimension_variable('node')
n_elems = mediator.get_dimension_variable('nele')

# Grid connectivity/adjacency
nv = mediator.get_grid_variable('nv', (3, n_elems), int)

# Cartesian coordinates
x_nodes = mediator.get_grid_variable('x', (n_nodes), float)
y_nodes = mediator.get_grid_variable('y', (n_nodes), float)
x_centroids = mediator.get_grid_variable('xc', (n_elems), float)
y_centroids = mediator.get_grid_variable('yc', (n_elems), float)

triangles = nv.transpose()
tri = Triangulation(x_nodes, y_nodes, triangles)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.triplot(tri)

for idx, (xc, yc) in enumerate(zip(x_centroids, y_centroids)):
    ax.scatter(xc, yc)
    ax.annotate('xc_{}'.format(idx), xy=(xc, yc), xytext=(xc, yc))

# Plot
for idx, (x, y) in enumerate(zip(x_nodes, y_nodes)):
    ax.scatter(x, y)
    ax.annotate('xn_{}'.format(idx), xy=(x, y), xytext=(x, y))

print 'Centroids'
print '%.10f' %x_centroids[0], '%.10f' %y_centroids[0]
print '%.10f' %x_centroids[1], '%.10f' %y_centroids[1]
print '%.10f' %x_centroids[2], '%.10f' %y_centroids[2]
print '%.10f' %x_centroids[3], '%.10f' %y_centroids[3]

plt.show()
