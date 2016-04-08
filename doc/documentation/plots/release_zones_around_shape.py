from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import shapefile

from PyFVCOM.ll2utm import UTMtoLL
from pylagtools import create_release_zones_around_shape

start = (-4.0, 50.3)   # Lon/Lat coordinates for the approximate location of the
                       # first release zone to be created
target_length = 2.0e5  # Target length of path along which to create release
                       # zones (m)
group_id = 1           # Group ID
radius = 2.0e3         # Release zone radius (m)
n_particles = 1000     # No. of particles per release zone

# Smple shapefile covering mainland UK
shp = shapefile.Reader('./resources/ref_shapefile')
shape_obj = shp.shape(0)

# Create release zones
release_zones = create_release_zones_around_shape(shape_obj, start, 
        target_length, group_id, radius, n_particles, check_overlaps=True)

# Plot particle positions
fig, ax = plt.subplots()
m = Basemap(llcrnrlon=-4.5,
            llcrnrlat=50.1,
            urcrnrlon=-4.0, 
            urcrnrlat=50.4,
            projection="tmerc",
            rsphere=(6378137.00,6356752.3142), 
            resolution='f',
            area_thresh=0.1,
            lon_0=-4.0,
            lat_0=50.0)
m.drawcoastlines()
m.fillcontinents()

for zone in release_zones:
    eastings = zone.get_eastings()
    northings = zone.get_northings()
    lats, lons = UTMtoLL(23, northings, eastings, "30N")
    x,y = m(lons, lats)

    # Add scatter plot of locations
    m.scatter(x, y, marker="x", color='r')

plt.title('No. of release zones = {}'.format(len(release_zones)))

plt.show()