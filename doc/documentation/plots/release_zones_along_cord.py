import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap

from PyFVCOM.ll2utm import UTMtoLL
from pylagtools import create_release_zones_along_cord
    
group_id = 1                 # Particle group ID
n_particles = 100            # Number of particles per release zone
radius = 400.0               # Release zone radius (m)
depth = 0.0                  # Depth of particles

# x,y position vector of LHS of Tamar Estuary (UTM WGS-84 / 30N)
r1=np.array([415409.776,5574398.16])

# x,y position vector of RHS of Tamar Estuary (UTM WGS-84 / 30N)
r2=np.array([419953.239,5574501.067])

# Create release zones
release_zones = create_release_zones_along_cord(r1, r2, group_id, radius, 
        n_particles, depth, random=True)

fig, ax = plt.subplots()
m = Basemap(llcrnrlon=-4.22,
            llcrnrlat=50.30,
            urcrnrlon=-4.10, 
            urcrnrlat=50.40,
            projection="tmerc",
            rsphere=(6378137.00,6356752.3142), 
            resolution='f',
            area_thresh=0.1,
            lon_0=-4.16,
            lat_0=50.35)
m.drawcoastlines()
m.fillcontinents()

for zone in release_zones:
    eastings = zone.get_eastings()
    northings = zone.get_northings()
    lats, lons = UTMtoLL(23, northings, eastings, "30N")
    x,y = m(lons, lats)

    # Add scatter plot of locations
    m.scatter(x, y, marker="x", color='r')

plt.show()