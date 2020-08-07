from __future__ import print_function

from matplotlib import pyplot as plt
from matplotlib.tri.triangulation import Triangulation

from test_roms_data_reader import MockROMSMediator

plt.ion()

mediator = MockROMSMediator()

grid_names = ['grid_rho', 'grid_u', 'grid_v']
for grid_name in grid_names:

    # Read in the grid's dimensions
    n_nodes = mediator.get_dimension_variable('node_{}'.format(grid_name))
    n_elems = mediator.get_dimension_variable('element_{}'.format(grid_name))

    # Grid connectivity/adjacency
    nv = mediator.get_grid_variable('nv_{}'.format(grid_name), (3, n_elems), int)

    # Cartesian coordinates
    x_nodes = mediator.get_grid_variable('longitude_{}'.format(grid_name), (n_nodes), float)
    y_nodes = mediator.get_grid_variable('latitude_{}'.format(grid_name), (n_nodes), float)
    x_centroids = mediator.get_grid_variable('longitude_c_{}'.format(grid_name), (n_elems), float)
    y_centroids = mediator.get_grid_variable('latitude_c_{}'.format(grid_name), (n_elems), float)

    triangles = nv.transpose()
    tri = Triangulation(x_nodes, y_nodes, triangles)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.triplot(tri)
    ax.set_title('{}'.format(grid_name))

    for idx, (xc, yc) in enumerate(zip(x_centroids, y_centroids)):
        ax.scatter(xc, yc)
        ax.annotate('xc_{}'.format(idx), xy=(xc, yc), xytext=(xc, yc))
        print('%.10f' %xc, '%.10f' %yc)

    # Plot
    for idx, (x, y) in enumerate(zip(x_nodes, y_nodes)):
        ax.scatter(x, y)
        ax.annotate('xn_{}'.format(idx), xy=(x, y), xytext=(x, y))

    plt.savefig('roms_{}_test_grid.png')

