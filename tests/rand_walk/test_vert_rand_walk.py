"""
TODO.

Author:
-------
James Clark (PML)
"""

import numpy as np
from matplotlib import pyplot as plt
from ConfigParser import SafeConfigParser

from pylag.analytic_data_reader import TestDiffusivityDataReader
from pylag.boundary_conditions import get_vert_boundary_condition_calculator

from pylag import cwrappers
import pylag.random as random

# Timings
time_start = 0.0 # seconds
time_end = 3600.0 # seconds
time_step = 1.0 # seconds
time = np.arange(time_start,time_end,time_step)
n_time = len(time)

# Number of particle to use
n_particles = 4000

# Config - needed for creation of the numerical integrator
config = SafeConfigParser()
config.add_section("SIMULATION")
config.set("SIMULATION", "time_step", str(time_step))
config.set("SIMULATION", "vertical_random_walk_model", "AR0")
config.set("SIMULATION", "vert_bound_cond", "reflecting")

# Create data reader
data_reader = TestDiffusivityDataReader(config)

# Create test object
test_vrwm = cwrappers.TestVerticalRandomWalk(config)

# Create vertical boundary condition calculator
test_vert_bc_calculator = get_vert_boundary_condition_calculator(config)

# z min and max values
zmin = data_reader.get_zmin(0.0,0.0,0.0)
zmax = data_reader.get_zmax(0.0,0.0,0.0)

# Initial z positions - uniformly distributed
z_positions = []
for i in xrange(n_particles):
    z_positions.append(random.uniform(zmin, zmax)) # Particles uniformly distributed in the first instance

# x/y
x_pos = 0.0
y_pos = 0.0

# Create array in which to store particle depths
particle_depths = np.empty((n_particles, n_time), dtype=float)

# The random walk
for t_idx, t in enumerate(time):
    for z_idx, z_pos in enumerate(z_positions):
        particle_depths[z_idx, t_idx] = z_pos
        
        # Compute the new position and apply boundary conditions as necessary
        zpos_new = test_vrwm.random_walk(data_reader, t, x_pos, y_pos, z_pos)
        if zpos_new < zmin or zpos_new > zmax:
            zpos_new = test_vert_bc_calculator.apply(zpos_new, zmin, zmax)
        z_positions[z_idx] =  zpos_new

# Bin edges used for concentration calculation
bin_edges = range(int(zmax-zmin) + 1)

# Compute particle concentrations
concentration = np.empty((len(bin_edges)-1, n_time))
for i in xrange(n_time):
    hist, bins = np.histogram(particle_depths[:,i], bins=bin_edges)
    concentration[:,i] = hist

# Grids for plotting using pcolormesh
time_out = np.arange(time_start-time_step/2., time_end + time_step/2., time_step)/3600.0
depth_out = np.array(bin_edges)
time_grid, depth_grid = np.meshgrid(time_out, depth_out)

# Plot
plt.pcolormesh(time_grid, depth_grid, concentration)
plt.colorbar()
plt.xlabel('Time (h)')
plt.ylabel('Height above sea bed (m)')
plt.show()
