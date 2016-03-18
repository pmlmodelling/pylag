"""
TODO.

Author:
-------
James Clark (PML)
"""

import numpy as np
from matplotlib import pyplot as plt
from ConfigParser import SafeConfigParser

from pylag.particle import Particle
from pylag.analytic_data_reader import TestDiffusivityDataReader
from pylag.random_walk import NaiveVerticalRandomWalk, AR0VerticalRandomWalk
from pylag.delta import Delta
import pylag.random as random

def get_vertical_random_walk_model(model_name, config):
    if model_name == "naive":
        return NaiveVerticalRandomWalk(config)
    elif model_name == "ar0":
        return AR0VerticalRandomWalk(config)
    else:
        raise ValueError('Unrecognised vertical random walk model: {}.'.format(model_name))

# Name of model to test
model_name = "naive"

# Define a range of z values (measured as height above the sea bed)
zmin = 0.0
zmax = 40.0

# Timings
n_times = 60*60 # 60 * 60 * time_step gives 6 hours
time_start = 3.0
time_step = 6.0
times = np.arange(time_start,time_step*n_times,time_step)

# Initial properties (other than z!) of test particles
group_id = 0
x_0 = 0.0
y_0 = 0.0

# Create a set of test particles with the listed properties
n_particles = 4000
test_particles = []
for i in xrange(n_particles):
    z_0 = random.uniform(zmin, zmax) # Particles uniformly distributed in the first instance
    test_particles.append(Particle(group_id, x_0, y_0, z_0))

# Config - needed for creation of data reader and random walk objects
config = SafeConfigParser()
config.add_section("SIMULATION")
config.add_section("OCEAN_CIRCULATION_MODEL")
config.set("SIMULATION", "time_step", str(time_step))
config.set("OCEAN_CIRCULATION_MODEL", "vertical_coordinate_system", 'cartesian')
config.set("OCEAN_CIRCULATION_MODEL", "zmin", str(zmin))
config.set("OCEAN_CIRCULATION_MODEL", "zmax", str(zmax))

# Create data reader
data_reader = TestDiffusivityDataReader(config)

# Vertical random walk model
vrwm = get_vertical_random_walk_model(model_name, config)

# Object for storing position deltas
delta_X = Delta()

# Depth array
particle_depths = np.empty((n_particles, n_times), dtype=float)

# The random walk
for p_idx, particle in enumerate(test_particles):
    print 'Updating particle {}'.format(p_idx)
    for t_idx, time in enumerate(times):
        delta_X.reset()

        vrwm.random_walk(time, particle, data_reader, delta_X)
        
        # Apply reflecting boundary conditions
        zpos = particle.zpos + delta_X.z
        if zpos < zmin:
            zpos = zmin + zmin - zpos
        elif zpos > zmax:
            zpos = zmax + zmax - zpos
        
        particle.zpos = zpos
        
        particle_depths[p_idx, t_idx] = particle.zpos

# Bin edges used for concentration calculation
bin_edges = range(int(zmax-zmin) + 1)

# Compute particle concentrations
concentration = np.empty((len(bin_edges)-1, n_times))
for i in xrange(n_times):
    hist, bins = np.histogram(particle_depths[:,i], bins=bin_edges)
    concentration[:,i] = hist

# Grids for plotting using pcolormesh
time_out = np.arange(time_start-time_step/2,time_step*(n_times+1),time_step)/3600.0
depth_out = np.array(bin_edges)
time_grid, depth_grid = np.meshgrid(time_out, depth_out)

# Plot
plt.pcolormesh(time_grid, depth_grid, concentration)
plt.colorbar()
plt.xlabel('Time (h)')
plt.ylabel('Height above sea bed (m)')
plt.show()
