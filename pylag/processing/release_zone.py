"""
A set of classes and functions to help with creating particle release zones.
"""

from __future__ import division, print_function

import numpy as np
import numbers
from scipy.spatial import ConvexHull
from typing import Optional

from pylag.processing.coordinate import utm_from_lonlat, get_epsg_code

have_shapely = True
try:
    import shapely.geometry
except ImportError:
    have_shapely = False


class ReleaseZone(object):
    """ Release zone

    A release zone is a circular region in cartesian space containing a set of
    particles.

    Parameters
    ----------
    group_id : int
        Group ID associated with the release zone

    radius : float
        The radius of the release zone in m.

    centre : array_like
        Two element array giving the coordinates of the centre of the
        release zone.
    """
    def __init__(self, group_id=1, radius=100.0, centre=[0.0, 0.0]):
        self.__group_id = group_id
        self.__radius = radius
        self.__centre = centre
        self.__particle_set = []

    def create_particle_set(self, n_particles=100, z=0.0, random=True):
        """ Create a new particle set

        Create a new particle set (`n=n_particles`). The spatial coordinates
        of each particle are computed. If random is true, exactly n, random,
        uniformly distributed particles will be created. If false, particles
        are uniformly distributed on a cartesian grid, which is then filtered
        for  the area encompassed by the release zone, which is determined
        from the radius. The latter algorithm yields a particle set with
        `n <= n_particles` where `|n - n_particles| / n -> 1` for large n.
        The former guarantees that positions for exactly n particles are
        created. However, for small n the particle distribution with be
        patchy. All particles are created are given the same depth
        coordinates.

        Parameters
        ----------
        n_particles : int, optional
            The number of particles to be created and added to the
            release zone.

        z : float, optional
            The depth of the particles.

        random : bool, optional
            If True (default) create particle positions at random. This
            guarantees that n_particles will be added to the release zone. If
            False, particles are regularly spaced on a Cartesian grid, which
            is then filtered for the area of the circle.

        Returns
        -------
        N/A
        """

        # First delete the current particle set, if it exists
        if len(self.__particle_set) != 0:
            self.__particle_set = []

        # If random, create a set of n randomly distributed particles.
        if random:
            radii = self.__radius * np.sqrt(np.random.uniform(0.0,
                                                              1.0,
                                                              n_particles))
            angles = np.random.uniform(0.0, 2.0 * np.pi, n_particles)
            for r, theta in zip(radii, angles):
                x = r * np.cos(theta) + self.__centre[0]
                y = r * np.sin(theta) + self.__centre[1]
                self.add_particle(x, y, z)
            return

        # Filtered cartesian grid. Assume each particle sits at the centre
        # of a square of length delta, which is then equivalent to the
        # particle separation.
        delta = np.sqrt(self.get_area() / float(n_particles))
        n_coords_xy = int(2.0 * self.__radius / delta)

        # Form a regular square grid of particle positions centered on centre
        x_vals = np.linspace(self.__centre[0] - self.__radius,
                             self.__centre[0] + self.__radius, n_coords_xy)
        y_vals = np.linspace(self.__centre[1] - self.__radius,
                             self.__centre[1] + self.__radius, n_coords_xy)
        x_coords, y_coords = np.meshgrid(x_vals, y_vals)

        # Now filter for points lying inside the arc of C (handled by
        # add_particle)
        for x, y in zip(x_coords.flatten(), y_coords.flatten()):
            try:
                self.add_particle(x, y, z)
            except ValueError:
                pass
        return

    def get_group_id(self):
        """ Return the group ID

        Returns
        -------
         : int
             The group ID.
        """
        return self.__group_id

    def set_group_id(self, id):
        """ Set the group ID

        Parameters
        ----------
        id : int
            The group ID.

        Returns
        -------
         : None
        """
        self.__group_id = id

    def get_radius(self):
        """ Get the radius

        Returns
        -------
         : float
            The radius of the relase zone

        """
        return self.__radius

    def get_area(self):
        """ Get the area

        Returns
        -------
         : float
            The area of the release zone

        """
        return np.pi * self.__radius * self.__radius

    def get_centre(self):
        """ Get the central coordinates

        Returns
        -------
         : array_list
            Array of central coordinates [x, y].

        """
        return self.__centre

    def get_particle_set(self):
        """ Get the particle set

        Returns
        -------
         : list[tuple]
             List of tuples of particle coordinates
        """
        return self.__particle_set

    def add_particle(self, x, y, z):
        """ Add a particle to the release zone

        Parameters
        ----------
        x : float
            x-coordinate

        y : float
            y-coordinate

        z : float
            z-coordinate

        Returns
        -------
         : None
        """
        if np.sqrt((x-self.__centre[0]) * (x-self.__centre[0]) + (y-self.__centre[1]) * (y-self.__centre[1])) <= self.__radius:
            self.__particle_set.append((x, y, z))
            return
        raise ValueError('Particle coordinates lie outside of the '
                         'release zone')

    def get_number_of_particles(self):
        """ Get the total number of particles

        Returns
        -------
         : int
            The total number of particles
        """
        return np.shape(self.__particle_set)[0]

    def get_coords(self):
        """ Get particle coordinates

        Returns
        -------
         : array_like
            Eastings

         : array_like
            Northings

         : array_like
            Depths
        """
        return self.get_eastings(), self.get_northings(), self.get_depths()

    def get_eastings(self):
        """

        Returns
        -------
         : array_like
            Eastings
        """
        return [particle_coords[0] for particle_coords in self.__particle_set]

    def get_northings(self):
        """ Get northings

        Returns
        -------
         : array_like
            Northings
        """
        return [particle_coords[1] for particle_coords in self.__particle_set]

    def get_depths(self):
        """ Get depths

        Returns
        -------
         : array_like
            Depths
        """
        return [particle_coords[2] for particle_coords in self.__particle_set]

    def get_zone_polygon(self):
        """ Make a polygon of the points in the zone (based on its convex hull)

        Returns
        -------
        poly : shapely.geometry.Polygon
            Polygon
        """

        if not have_shapely:
            raise ImportError('Cannot create a polygon for this release '
                              'zone as we do not have shapely installed.')

        points = np.asarray([np.asarray(i) for i in self.get_particle_set()])
        qhull = ConvexHull(points[:, :-1])  # skip depths for the convex hull
        x = points[qhull.vertices, 0]
        y = points[qhull.vertices, 1]
        # Add depths back in when making the polygon.
        z = points[qhull.vertices, 2]
        poly = shapely.geometry.Polygon(np.asarray((x, y, z)).T)

        return poly


def create_release_zone(group_id=1, radius=100.0, centre=[0.0, 0.0],
                        n_particles=100, depth=0.0, random=True):
    """ Create a new release zone

    Parameters
    ----------
    group_id : integer, optional
        Group identifier.

    radius : float, optional
        Radius of the circle in meters.

    centre : ndarray [float, float], optional
        x, y coordinates of the circle centre in meters.

    n_particles : integer, optional
        The number of particles.

    depth : float, optional
        Zone depth in m (default 0.0m).

    random : boolean, optional
        Assign x/y positions randomly (default).

    Returns
    -------
    release_zone : ReleaseZone
       ReleaseZone object.

    """
    if ~isinstance(radius, numbers.Real):
        radius = float(radius)

    if ~isinstance(depth, numbers.Real):
        depth = float(depth)

    # Create a new release zone given its radius and centre
    release_zone = ReleaseZone(group_id, radius, centre)

    # Create a new particle set of n_particles at the given depth
    release_zone.create_particle_set(n_particles, depth, random)

    return release_zone


def create_release_zones_along_cord(r1, r2, group_id=1, radius=100.0,
                                    n_particles=100, depth=0.0, random=True,
                                    verbose=False):
    """ Generate a set of release zones along a cord

    Return particle positions along a line `r3`, defined by the
    position vectors `r1` and `r2`. Particles are packed into
    circlular zones of radius radius, running along `r3`. Positions
    for approximately n particles (`= n` if random is `True`) are returned
    per zone. If `2*radius` is `> |r3|`, no zones are created.

    Parameters
    ----------
    r1 : ndarray [float, float]
        Two component position vector in cartesian coordinates (x,y).

    r2 : ndarray [float, float]
        Two component position vector in cartesian coordinates (x,y).

    group_id : integer, optional
        Group id for the 1st release zone created along the cord.

    radius : float, optional
        Zone radius in m.

    n_particles : integer, optional
        Number of particles per zone.

    depth : float, optional
        Zone depth in m.

    random : boolean, optional
        If true create a random uniform distribution of particles within
        each zone (default).

    verbose : boolean, optional
        Hide warnings and progress details (default).

    Returns
    -------
    zones : object, iterable
        List of release zone objects along the cord.
    """

    # Use the line vector running between the position vectors r1 and r2
    # to calculate the no. of release zones.
    r3 = r2 - r1
    r3_length = np.sqrt((r3*r3).sum())
    r3_unit_vector = r3 / r3_length
    n_zones, buffer_zone = divmod(r3_length, 2.0*radius)

    if verbose:
        print(f"Cord length is {r3_length} m. A radius of {radius} m thus "
              f"yields {n_zones} release zones.")

    if n_zones == 0:
        print("WARNING: zero release zones have been created. Try "
              "reducing the zone radius.")
        return None

    # Move along in the direction of r3 generating release zones
    # every (2.0*radius) m.
    release_zones = []
    for n in np.arange(n_zones, dtype=int):
        r3_prime = (2.0 * float(n) * radius + radius + buffer_zone/2.0)*r3_unit_vector
        centre = r1 + r3_prime
        release_zone = create_release_zone(group_id, radius, centre, n_particles, depth, random)
        if verbose:
            n_particles_in_release_zone = release_zone.get_number_of_particles()
            print(f"Zone {n} (group_id = {group_id}) contains {n_particles_in_release_zone} particles.")
        release_zones.append(release_zone)
        group_id += 1

    return release_zones


# def create_release_zones_around_shape(shape_obj, start, target_length, group_id,
#                                      radius, n_particles, depth=0.0, random=True,
#                                      epsg_code=None, maximum_vertex_separation=None,
#                                      return_end_zone=True, verbose=False, ax=None):
#    """ Create a set of adjacent release zones around an arbitrary polygon.
#
#    Parameters
#    ----------
#    shape_obj : _Shape from module shapefile, np.ndarray
#        Shapefile describing a given landmass or an array of points (n, 2) as (x, y).
#
#    start : tuple(float,float)
#        Approximate starting point (lon,lat) in degrees
#
#    target_length : float
#        Distance along which to position release zones
#
#    group_id : integer
#        Group identifier.
#
#    radius : float
#        Zone radius in m.
#
#    n_particles : integer
#        Number of particles per zone.
#
#    depth : float, optional
#        Zone depth in m.
#
#    random : boolean, optional
#        If true create a random uniform distribution of particles within each
#        zone (default).
#
#    epsg_code : str, optional
#        EPSG code to which to force all coordinates. This is useful if you've
#        got a large model domain.
#
#    maximum_vertex_separation : float, optional
#        Skip sections of the shapefile which have a length in excess of this
#        value. Helps skip parts of a domain which are very simple.
#
#    return_end_zone : bool, optional
#        If False, do not return the last zone. Defaults to True (do return it). This is useful when chaining release
#        zones along a shapefile with a fixed distance and we want to use the end of one chain as the start of another.
#
#    verbose : boolean, optional
#        Hide warnings and progress details (default).
#
#    """
#    raise PyLagRuntimeError('Function has not been tested.')
#
#    # Use all points from the shapefile.
#    if hasattr(shape_obj, 'points'):
#        points = np.array(shape_obj.points)
#    else:
#        points = np.array(shape_obj)
#
#    # Use shapely to do all the heavy lifting. Work in cartesian coordinates so distances make sense.
#    xx, yy, _ = utm_from_lonlat(points[:, 0], points[:, 1], epsg_code=epsg_code)
#    # Calculate the section lengths so we can drop points which are separated by more than that distance.
#    close_enough = [True] * len(xx)
#    if maximum_vertex_separation is not None:
#        close_enough = [True] + [np.hypot(i[0], i[1]) <= maximum_vertex_separation for i in zip(xx[:-1] - xx[1:], yy[:-1] - yy[1:])]
#
#        # Reorder the coordinates so we start from the last point after the first jump in distance. This should make
#        # the polygon more sensibly stored (i.e. the beginning will be after the first long stretch.).
#        big_sections = np.argwhere(~np.asarray(close_enough))
#        if np.any(big_sections):
#            if is_clockwise_ordered(points):
#                reorder_idx = big_sections[-1][0] + 1
#            else:
#                reorder_idx = big_sections[0][0] - 1
#
#            xx = np.array(xx[reorder_idx:].tolist() + xx[:reorder_idx].tolist())
#            yy = np.array(yy[reorder_idx:].tolist() + yy[:reorder_idx].tolist())
#            # Redo the close_enough no we've reordered things so the indexing works OK.
#            close_enough = [True] + [np.hypot(i[0], i[1]) <= maximum_vertex_separation for i in zip(xx[:-1] - xx[1:], yy[:-1] - yy[1:])]
#
#    xx = xx[close_enough]
#    yy = yy[close_enough]
#
#    # Split our line into sections of length target_length. Then, on each of those, further split them into sections
#    # radius * 2 long. On each of those sections, create release zones. Return the lot as a single list.
#
#    # For reasons I can't fathom, using shapely.ops.split() doesn't work with the polygon line. The issue is that
#    # "not polygon_line.relate_patten(splitter, '0********')" returns True, which means shapely.ops.split() just
#    # returns the whole line as a list (which is then bundled into a GeometryCollection). This is not what I want!
#    # So, I'll manually split the line after we've put zones all the way along it. This isn't as neat :(
#    line = shapely.geometry.LineString(np.array((xx, yy)).T)
#    # Create release zones at radius * 2 distance along the current line.
#    zone_centres = [line.interpolate(i) for i in np.arange(1, line.length, radius * 2)]
#    release_zones = [create_release_zone(group_id, radius, i.coords[0], n_particles, depth, random) for i in zone_centres]
#
#    # Now, we'll split these into groups of target_length and increment each group ID.
#    zone_separations = [0] + [get_length(i[0].get_centre(), i[1].get_centre()) for i in zip(release_zones[:-1], release_zones[1:])]
#    # Find the indices where we've exceeded the target_length.
#    split_indices = np.argwhere(np.diff(np.mod(np.cumsum(zone_separations), target_length)) < 0).ravel()
#    split_indices = np.append(split_indices, -1)
#    split_indices = np.append([0], split_indices)
#
#    new_zones = []
#    for start_index, end_index in zip(split_indices[:-1], split_indices[1:]):
#        # Get a set of release zones and increment their group ID.
#        current_zones = release_zones[start_index:end_index]
#        _ = [i.set_group_id(group_id) for i in current_zones]
#        n_points = len(current_zones * n_particles)
#        if verbose:
#            print(f'Group {group_id} contains {n_points} particles.')
#
#        group_id += 1
#        new_zones += current_zones
#
#        # Plot the release zones.
#        if ax is not None:
#            ax.plot(*current_zones[0].get_centre(), 'ro', zorder=10)
#            ax.text(*current_zones[0].get_centre(), f'Group {current_zones[0].get_group_id()} start', zorder=1000)
#
#    release_zones = new_zones
#
#    if not return_end_zone:
#        release_zones = release_zones[:-1]
#
#    return release_zones


def create_release_zones_around_shape_section(
        polygon: shapely.geometry.Polygon,
        start_point: shapely.geometry.Point,
        target_length: Optional[float] = None,
        release_zone_radius: Optional[float] = 100.0,
        n_particles: Optional[int] = 100,
        group_id: Optional[int] = 0,
        depth: Optional[float] = 0.0, epsg_code: Optional[str] = None,
        random: Optional[bool] = True,
        check_overlaps: Optional[bool] = False,
        overlap_tol: Optional[float] = 0.0001,
        verbose: Optional[bool] = False):
    """ Create a set of adjacent release zones around shape section

    This function is distinct from the function
    `create_release_zones_around_shape` in the sense that it a) only
    creates release zones around a specified length of the polygon, and
    b) gives each release zone a separate individual ID tag.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon describing a given landmass or object.
        All points in lon/lat coordinates.

    start_point : shapely.geometry.Point
        Approximate start point for release zones in lon/lat
        coordinates. The actual start point will be the nearest
        vertex on the polygon to the given coordinates.

    target_length : float, optional
        Distance along which to position release zones in m. Optional,
        defaults to None. If None, the release zones will be created
        around the full perimeter of the polygon.

    release_zone_radius : float, optional
        Radius of each circular release zone in m. Optional, defaults
        to 100.0 m.

    n_particles : integer, optional
        Number of particles per release zone. Optional, defaults to 100.

    group_id : integer, optional
        ID of the first release zone created. All subsequent release zones
        will have an ID of group_id + 1, group_id + 2, etc. Optional,
        defaults to 0.

    depth : float, optional
        Zone depth in m. Defaults to a depth of 0.0 m.

    epsg_code : str, optional
        EPSG code which should be used to covert to UTM coordiantes. If
        not given, the EPSG code will be inferred from `start_point`.

    random : boolean, optional
        If true create a random uniform distribution of particles within each
        zone. Optional, defaults to True.

    check_overlaps : boolean, optional
        If true check for overlaps between non-adjacent release zones that may
        result from jagged features. NOT YET IMPLEMENTED.

    overlap_tol : float, optional
        Overlap tolerance. This can be increased to permit small overlaps
        between adjacent release zones. Optional, defaults to 0.0001.

    verbose : boolean, optional
        Hide warnings and progress details. Optional, default False.

    TODO
    ----
    1) Allow users to provide a set of finish coordinates as an
    alternative to a length, which can then be used to determine the
    location of the last release zone to be created.
    """
    # Extract the points from the polygon.
    points = np.array(polygon.exterior.xy).T

    # Total number of points in this part
    n_points = points.shape[0]

    # Establish whether or not the shapefile is ordered in a clockwise
    # or anticlockwise manner
    clockwise_ordering = _is_clockwise_ordered(points)

    # Set the epsg_code from the start coordinates?
    if epsg_code is None:
        epsg_code = get_epsg_code(start_point.x, start_point.y)

    # Find starting location
    start_idx = _find_start_index(points, start_point.x, start_point.y,
                                  epsg_code=epsg_code)

    # Form the first release zone centred on the point for start_idx
    # --------------------------------------------------------------
    release_zones = []
    idx = start_idx - n_points if clockwise_ordering else start_idx
    x, y, _ = utm_from_lonlat(points[idx, 0], points[idx, 1],
                              epsg_code=epsg_code)
 
    # Coordinates of zone centre
    centre_ref = np.array([x[0], y[0]])
    release_zone = create_release_zone(group_id, release_zone_radius,
                                       centre_ref, n_particles, depth, random)
    if verbose:
        n_particles_in_release_zone = release_zone.get_number_of_particles()
        print(f"Zone (group_id = {group_id}) contains "
              f"{n_particles_in_release_zone} particles.")
    release_zones.append(release_zone)

    # Now step through shape vertices creating new release zones en route
    # -------------------------------------------------------------------
    # Target separation of adjacent release zones (i.e. touching circles)
    target_separation = 2.0 * release_zone_radius

    # Cumulative distance travelled around polygon (exit when >target_length)
    distance_travelled = 0.0

    # Coordinates of the last vertex, used for length calculation
    last_vertex = centre_ref

    # Update group id
    group_id += 1

    # Update the current index
    idx = _update_idx(idx, clockwise_ordering)

    # Loop until we've either run out of points or we've exceeded the
    # target length.
    counter = 1
    while True:
        if target_length is None:
            if counter > n_points:
                # Full perimeter has been traversed. Break.
                break
        else:
            if counter > n_points or distance_travelled > target_length:
                break

        x, y, _ = utm_from_lonlat(points[idx, 0], points[idx, 1],
                                  epsg_code=epsg_code)
        current_separation = _get_length(centre_ref, np.array([x[0], y[0]]))
        if current_separation >= target_separation:
            # Track back along the last cord to find the point
            # giving a release zone separation of 2*radius.
            #
            # Approach:
            # At present we know position vectors r1 (=centre_ref, coords for
            # centre of the last release zone), r2 (=last_vertex, coords for
            # the last vertex in the polygon, and r3 (the current vertex in
            # the polygon [x, y]). We also know that |r4-r1| must equal
            # target_separation (i.e. 2x the radius of a relase zone). The
            # task is then to find r4, which lies on the line joining r2 and
            # r3. This is managed by the method find_release_zone_location().
            r4 = _find_release_zone_location(centre_ref, last_vertex,
                                             np.array([x[0], y[0]]),
                                             target_separation)

            # Create location
            release_zone = create_release_zone(group_id, release_zone_radius,
                                               r4, n_particles, depth, random)
            if verbose:
                n = release_zone.get_number_of_particles()
                print(f"Zone (group_id = {group_id}) contains {n} particles.")

            # Check if the new release zone overlaps with non-adjacent zones
            if check_overlaps:
                for zone_test in release_zones:
                    centre_test = zone_test.get_centre()
                    zone_separation = _get_length(r4, centre_test)
                    offset = (target_separation - zone_separation)
                    if offset / target_separation > overlap_tol:
                        print(f"WARNING: Area overlap detected between "
                              f"release zones {zone_test.get_group_id()} "
                              f"and {release_zone.get_group_id()}. "
                              f"Target separation = {target_separation}. "
                              f"Actual separation = {zone_separation}.")

            # Append the new release zone to the current set.
            release_zones.append(release_zone)

            # Update references and counters
            centre_ref = r4
            group_id += 1
        else:
            # Update counters
            distance_travelled += _get_length(last_vertex,
                                              np.array([x[0], y[0]]))
            last_vertex = np.array([x[0], y[0]])
            idx = _update_idx(idx, clockwise_ordering)
            counter += 1

    return release_zones


def _is_clockwise_ordered(points: np.ndarray) -> bool:
    """ Check to see if a set of points are clockwise ordered

    Establish the index of the left most point in x, then check for
    rising or falling y.

    Parameters
    ----------
    points: ndarray
        2D array (n,2) containing lon/lat coordinates for n locations.
        Ordering is critical, and must adhere to points[:, 0] ->
        lon values, points[:, 1] -> lat values.

    Returns
    -------
    : boolean
       True if clockwise ordered.

    """

    is_clockwise = False
    if have_shapely:
        # Use shapely to figure this out more robustly.
        poly = shapely.geometry.Polygon(points)
        # Make a clockwise ordered polygon from our points and compare against
        # what we've been given. If they're the same, then we've got a
        # clockwise one, otherwise, we've got anti-clockwise.
        ordered_poly = shapely.geometry.polygon.orient(poly, sign=1.0)
        ordered_points = np.array(ordered_poly.exterior.xy).T
        if not np.all(ordered_points == points):
            is_clockwise = True
    else:
        idx = np.argmin(points[:, 0])
        # This fails if the minimum occurs as the last point in the array.
        # If that's the case, reverse the check (i.e. search for smaller
        # latitudes at the index beforehand).
        if idx + 1 == np.shape(points)[0]:
            if points[idx - 1, 1] < points[idx, 1]:
                is_clockwise = True
        else:
            if points[idx + 1, 1] > points[idx - 1, 1]:
                is_clockwise = True

    return is_clockwise


def _find_start_index(points: np.ndarray, lon: float, lat: float,
                      tolerance: Optional[float] = None,
                      epsg_code: Optional[str] = None) -> int:
    """ Find start index for release zone creation.

    Parameters
    ----------
    points: ndarray
        2D array (n,2) containing lon/lat coordinates for n locations. Ordering
        is criticial, and must adhere to points[:,0] -> lon values, points[:,1]
        -> lat values.

    lon: float
        Target longitude

    lat: float
        Target latitude

    tolerance: float, optional
        Raise ValueError if the target lat/lon values lie beyond this distance
        (in m) from the shape_obj.

    epsg_code : str, optional
        Give a EPSG code to use when transforming lat and lon coordinates to m.
        If not provided, the supplied lat and lon values are used to infer the
        EPSG code. Useful for large domains which spread over multiple UTM
        zones.

    Returns
    -------
    start_idx: integer
        Start index in array points[start_idx,:].

    """
    if epsg_code is None:
        epsg_code = get_epsg_code(lon, lat)

    x_start, y_start, _ = utm_from_lonlat(lon, lat, epsg_code=epsg_code)

    x_points, y_points, _ = utm_from_lonlat(points[:, 0], points[:, 1],
                                            epsg_code=epsg_code)

    # Find the position on the boundary closest to the supplied lat/lon values.
    distances = np.hypot(x_points - x_start[0], y_points - y_start[0])
    distance_min = np.min(distances)
    if tolerance is not None:
        if distance_min < tolerance:
            start_idx = np.argmin(distances)
        else:
            raise ValueError(f"Supplied lat/lon values are further away that "
                             f"{tolerance} m to the nearest vertex of the "
                             f"supplied shape.")
    else:
        start_idx = np.argmin(distances)

    return start_idx


def _get_length(r1: np.ndarray, r2: np.ndarray) -> float:
    """ Return the length of the line vector joining r1 and r2.

    Parameters
    ----------
    r1: ndarray (float, float)
        Position vector [x,y] for point 1 in m.

    r2: ndarray (float, float)
        Position vector [x,y] for point 2 in m.

    Returns
    -------
    length: float
       Length in m.

    """
    if ~isinstance(r1, np.ndarray):
        r1 = np.array(r1)
    if ~isinstance(r2, np.ndarray):
        r2 = np.array(r2)

    r12 = r2 - r1

    return np.sqrt((r12*r12).sum())


def _update_idx(idx: int, clockwise_ordering: bool) -> int:
    """ Update idx, depending on the value of clockwise_ordering.

    Parameters
    ----------
    idx: int
        Current index.

    clockwise_ordering: boolean
        Is clockwise ordered?

    Returns
    -------
    idx: integer
        New index.

    """
    return idx + 1 if clockwise_ordering else idx - 1


def _find_release_zone_location(r1, r2, r3, r14_length) -> np.ndarray:
    """ Find release zone location

    Find the position vector r4 that sits on the line joining the position
    vectors r2 and r3, at a distance r14_length from the position vector r1.
    First form three equations with three unknowns (r4[0], r4[1] and |r24|).
    Solve for |r24|, then solve for r4 (=(|r24|/|r23|)*r23).

    Parameters
    ----------
    r1, r2, r3: ndarray, 2D
        Position vectors r1, r2, and r3.

    r14_length: float
        Length of the vector joining the position vectors r1 and r4.

    Returns:
    --------
    r4: ndarray, 2D
        The position vector r4.

    """
    if ~isinstance(r1, np.ndarray):
        r1 = np.array(r1)
    if ~isinstance(r2, np.ndarray):
        r2 = np.array(r2)
    if ~isinstance(r3, np.ndarray):
        r3 = np.array(r3)

    # Vector running from r1 to r2
    r12 = r2 - r1

    # Vector running from r2 to r3
    r23 = r3 - r2
    r23_length = np.sqrt((r23 * r23).sum())

    # Intermediate variables
    A = r23[0] / r23_length
    B = r23[1] / r23_length
    C = A * A + B * B
    D = 2.0 * (A * r12[0] + B * r12[1])
    E = r12[0] * r12[0] + r12[1] * r12[1] - (r14_length * r14_length)
    F = D / C
    G = E / C

    # Now use the quadratic formula to find r24_length
    r24_length_a = 0.5 * (-F + np.sqrt(F * F - 4.0 * G))
    r24_length_b = 0.5 * (-F - np.sqrt(F * F - 4.0 * G))

    # Try to find the correct root
    r24_length_a_is_valid = False
    r24_length_b_is_valid = False
    if 0.0 <= r24_length_a <= r23_length:
        r24_length_a_is_valid = True
    if 0.0 <= r24_length_b <= r23_length:
        r24_length_b_is_valid = True

    # We might end up in the situation with two valid roots being equidistant
    # from r1 but in opposite directions. In that situation, we want to pick
    # the one going towards the end point (r3).
    if r24_length_a_is_valid and r24_length_b_is_valid:
        res1 = r2 + (r24_length_a / r23_length) * r23
        res2 = r2 + (r24_length_b / r23_length) * r23
        res1r3 = np.hypot(res1[0] - r3[0], res1[1] - r3[1])
        res2r3 = np.hypot(res2[0] - r3[0], res2[1] - r3[1])
        if res1r3 > res2r3:
            r24_length_a_is_valid = False
            r24_length_b_is_valid = True
        else:
            r24_length_a_is_valid = True
            r24_length_b_is_valid = False

    # Error checking
    if r24_length_a_is_valid and r24_length_b_is_valid:
        raise ValueError(f"Two apparently valid roots identified: a) "
                         f"{r24_length_a[0]:f} and b) {r24_length_b[0]:f}.")
    if not r24_length_a_is_valid and not r24_length_b_is_valid:
        raise ValueError(f"No valid roots found: a) {r24_length_a[0]:f} "
                         f"and b) {r24_length_b[0]:f}.")

    # Set r24_length equal to the valid root
    if r24_length_a_is_valid:
        r24_length = r24_length_a
    elif r24_length_b_is_valid:
        r24_length = r24_length_b

    return r2 + (r24_length / r23_length) * r23
