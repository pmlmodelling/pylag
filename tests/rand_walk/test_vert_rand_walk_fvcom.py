import numpy as np
from matplotlib import pyplot as plt
from ConfigParser import SafeConfigParser

from pylag.particle import Particle
from pylag.fvcom_data_reader import FVCOMDataReader
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
model_name = "ar0"

# FVCOM z min and max values in sigma coords
zmin = -1.0
zmax = 0.0

# Timings
n_times = 60*10# 60 * 60 * time_step gives 6 hours
time_start = 3.0 # in seconds
time_step = 6.0 # in seconds
times = np.arange(time_start,time_start+time_step*n_times,time_step)

# Configuration
config = SafeConfigParser()
config.add_section("SIMULATION")
config.add_section("OCEAN_CIRCULATION_MODEL")
config.set('SIMULATION', 'start_datetime', '2013-01-06 00:00:00')
config.set("SIMULATION", "time_step", str(time_step))
config.set('OCEAN_CIRCULATION_MODEL', 'data_dir', '../../resources/')
config.set('OCEAN_CIRCULATION_MODEL', 'grid_metrics_file', 'irish_sea_v20_grid_metrics.nc')
config.set('OCEAN_CIRCULATION_MODEL', 'data_file_stem', 'irish_sea_v20_0001')
config.set('OCEAN_CIRCULATION_MODEL', 'rounding_interval', '3600')
config.set('OCEAN_CIRCULATION_MODEL', 'zmin', str(zmin))
config.set('OCEAN_CIRCULATION_MODEL', 'zmax', str(zmax))
data_reader = FVCOMDataReader(config)

# Initial properties (other than z!) of test particles
group_id = 0
x_0 = 365751.7
y_0 = 5323568.0
host_0 = data_reader.find_host_using_global_search(x_0, y_0)

# Compute the diffusivity profile
zgrid = np.linspace(zmin, zmax, 100)
k = []
for z in zgrid:
    k.append(data_reader.get_vertical_eddy_diffusivity(time_start, x_0, y_0, z, host_0))

# Initialise the RNG
#random.seed()

# Create a set of test particles with the listed properties
n_particles = 4000
test_particles = []
for i in xrange(n_particles):
    # Particles uniformly distributed in the first instance
    z_0 = random.uniform(zmin, zmax)

    test_particles.append(Particle(group_id, x_0, y_0, z_0, host_0))

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

        # Always passing in the start time means we use a constant diffusivity profile
        vrwm.random_walk(time_start, particle, data_reader, delta_X)
        
        # Apply reflecting boundary conditions
        zpos = particle.zpos + delta_X.z
        if zpos < zmin:
            zpos = zmin + zmin - zpos
        elif zpos > zmax:
            zpos = zmax + zmax - zpos
        
        particle.zpos = zpos
        
        # Save the particle's position; first convert to cartesian coords
        particle_depths[p_idx, t_idx] = particle.zpos

# Bin edges used for concentration calculation
bin_edges = np.arange(zmin,zmax+0.025,0.01)

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
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
line = ax1.plot(k, zgrid)
ax1.set_ylim([zmin,zmax])
ax1.set_xlabel('K')
ax1.set_ylabel('Depth (sigma)')

pcol = ax2.pcolormesh(time_grid, depth_grid, concentration)
ax2.set_xlim([time_out.min(),time_out.max()])
ax2.set_ylim([zmin,zmax])
f.colorbar(pcol)
ax2.set_xlabel('Time (h)')

plt.savefig('{}.png'.format(model_name))
plt.show()
