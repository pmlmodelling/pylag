"""
Plot the analytic diffusivity profile used for testing random walk/displacement
models.

Author:
-------
James Clark (PML)
"""

import numpy as np
import matplotlib.pyplot as plt
from ConfigParser import SafeConfigParser

from pylag.analytic_data_reader import TestDiffusivityDataReader
from pylag.cwrappers import get_vertical_eddy_diffusivity

# (Unused) variables passed in as function arguments. In real applications
# these would be used for interpolating a gridded diffusivity field to a
# given point in space/time.
t = 0.0
xpos = 0.0
ypos = 0.0
host = 0
zlayer = 0

# Define a range of z values (measured as height above the sea bed)
zmin = 0.0
zmax = 40.0

# Resolution in z
dz = 0.1

# Config - needed for creation of NaiveVerticalRandomWalk
config = SafeConfigParser()
config.add_section("OCEAN_CIRCULATION_MODEL")

# Diffusivity data reader
data_reader = TestDiffusivityDataReader(config)

# Depth array
z = np.arange(zmin, zmax, dz, dtype=float)

# Diffusivity array
K = np.empty_like(z)

# Compute K
for i, zpos in enumerate(z):
    K[i] = get_vertical_eddy_diffusivity(data_reader, t, xpos, ypos, zpos, host)

# Plot
plt.figure()
plt.plot(K, z, 'b')
plt.xlabel(r'K (m$^{2}$ s$^{-1}$)')
plt.ylabel('Height above sea bed (m)')
plt.show()
