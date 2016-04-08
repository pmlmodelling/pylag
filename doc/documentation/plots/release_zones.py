import numpy as np
from matplotlib import pyplot as plt

from pylagtools import create_release_zone

group_id = 1                 # Particle group ID
n_particles = 100            # Number of particles per release zone
radius = 10.0                # Release zone radius
centre = np.array([0.0,0.0]) # (x,y) coordinates of the release zone's centre
depth = 0.0                  # Depth of particles

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))

release_zone_reg = create_release_zone(group_id, radius, centre, n_particles, 
        depth, random=False)
eastings_reg, northings_reg, depth_reg = release_zone_reg.get_coords()
ax1.scatter(eastings_reg, northings_reg, c='b', marker='o')
ax1.add_patch(plt.Circle(centre, radius=radius, color='k', alpha=0.2))
ax1.set_title('Regular (n = {})'.format(
        release_zone_reg.get_number_of_particles()))
ax1.set_xlim(-radius,radius)
ax1.set_ylim(-radius,radius)

release_zone_rand = create_release_zone(group_id, radius, centre, n_particles, 
        depth, random=True)
eastings_rand, northings_rand, depth_rand = release_zone_rand.get_coords()
ax2.scatter(eastings_rand, northings_rand, c='r', marker='o')
ax2.add_patch(plt.Circle(centre, radius=radius, color='k', alpha=0.2))
ax2.set_title('Uniform random (n = {})'.format(
        release_zone_rand.get_number_of_particles()))
ax2.set_xlim(-radius,radius)
ax2.set_ylim(-radius,radius)

plt.show()