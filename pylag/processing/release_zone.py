"""
A set of classes and functions to help with creating particle release zones.
"""

from __future__ import division, print_function

import numpy as np
from scipy.spatial import ConvexHull
from typing import Optional

from pylag.exceptions import PyLagAttributeError
from pylag.processing.coordinate import utm_from_lonlat
from pylag.processing.coordinate import lonlat_from_utm
from pylag.processing.coordinate import get_epsg_code

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
    group_id : int, optional
        Group ID associated with the release zone. Optional,
        defaults to 1.

    radius : float, optional
        The radius of the release zone in m. Optional, defaults to 100.0 m.

    centre : array_like, optional
        Two element array giving the coordinates of the centre of the
        release zone. Optional, defaults to [0.0, 0.0].
    
    coordinate_system : str, optional
        Coordinate system used to interpret the given `centre` coordinates.
        The options are 'geographic' or 'cartesian' (default). If 'geographic'
        is given, the coordinates are assumed to be in lon/lat. If 'cartesian'
        is given, the coordinates are assumed to be in x/y.
    
    epsg_code : str, optional
        EPSG code which should be used to covert to UTM coordiantes. If
        not given, the EPSG code will be inferred from `centre`. If working
        in cartesian coordinates, this argument is ignored.
    """
    def __init__(self, group_id: Optional[int] = 1,
                 radius: Optional[float] = 100.0,
                 centre = [0.0, 0.0],
                 coordinate_system: Optional[str]='cartesian',
                 epsg_code: Optional[str]=None):
        self.__group_id = group_id
        self.__radius = radius
        self.__coordinate_system = coordinate_system
        self.__epsg_code = epsg_code

        if self.__coordinate_system == 'geographic':
            eastings, northings, epsg_code_centre = utm_from_lonlat(
                centre[0], centre[1])

            self.__centre = [eastings[0], northings[0]]

            if epsg_code is None:
                self.__epsg_code = epsg_code_centre

        elif self.__coordinate_system == 'cartesian':
            self.__centre = [centre[0], centre[1]]
            self.__epsg_code = None
        
        else:
            raise ValueError(f"Unrecognised coordinate system "
                             f"{self.__coordinate_system}. "
                             f"Options are 'geographic' or 'cartesian'.")

        # The particle set is a list of tuples of the form (x, y, z)
        # where x, y and z are the particle coordinates. Initially, the
        # particle set is empty. Particles are added to the set using
        # the add_particle method. 
        self.__particle_set = []

    def create_particle_set(self, n_particles: Optional[int]=100,
                            z: Optional[float]=0.0,
                            random: Optional[bool]=True):
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
        patchy. All particles created are given the same depth
        coordinates.

        Parameters
        ----------
        n_particles : int, optional
            The number of particles to be created and added to the
            release zone. Defaults to 100.

        z : float, optional
            The depth of the particles. Defaults to 0.0 m.

        random : bool, optional
            If True (default) create particle positions at random. This
            guarantees that n_particles will be added to the release zone. If
            False, particles are regularly spaced on a Cartesian grid, which
            is then filtered for the area of the circle. Optional, defaults to
            True.

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
        delta = np.sqrt(self.area / float(n_particles))
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

    @property
    def group_id(self):
        return self.__group_id
    
    @group_id.setter
    def group_id(self, value: Optional[int]):
        self.__group_id = value

    @property
    def radius(self):
        return self.__radius
    
    @radius.setter
    def radius(self, value: Optional[float]):
        raise PyLagAttributeError("Radius is immutable.")

    @property
    def area(self):
        return np.pi * self.__radius * self.__radius

    @property
    def centre(self):
        return [x for x in self.__centre]

    @centre.setter
    def centre(self, value: Optional[list]):
        raise PyLagAttributeError("Centre is immutable.")

    @property
    def coordinate_system(self):
        return self.__coordinate_system

    @coordinate_system.setter
    def coordinate_system(self, value: Optional[str]):
        raise PyLagAttributeError("Coordinate system is immutable.")

    @property
    def epsg_code(self):
        if self.__coordinate_system == 'geographic':
            return self.__epsg_code
        else:
            raise PyLagAttributeError("No EPSG code available for "
                                      "release zones defined in "
                                      "cartesian coordinates.")

    @epsg_code.setter
    def epsg_code(self, value: Optional[str]):
        raise PyLagAttributeError("EPSG code is immutable.")

    @property
    def particle_set(self):
        return [p for p in self.__particle_set]

    @particle_set.setter
    def particle_set(self, value: Optional[list]):
        raise PyLagAttributeError("Particle set is immutable.")

    # Methods used to get the coordinates of particles in the release zone
    # ---------------------------------------------------------------------

    @property
    def x_coordinates(self):
        if self.__coordinate_system == 'cartesian':
            return [coords[0] for coords in self.__particle_set]
        else:
            x = [coords[0] for coords in self.__particle_set]
            y = [coords[1] for coords in self.__particle_set]
            lons, _ = lonlat_from_utm(x, y, self.__epsg_code)
            
            return lons

    @x_coordinates.setter
    def x_coordinates(self, value: Optional[list]):
        raise PyLagAttributeError("X coordinates are immutable.")

    @property
    def y_coordinates(self):
        if self.__coordinate_system == 'cartesian':
            return [coords[1] for coords in self.__particle_set]
        else:
            x = [coords[0] for coords in self.__particle_set]
            y = [coords[1] for coords in self.__particle_set]
            _, lats = lonlat_from_utm(x, y, self.__epsg_code)
            
            return lats

    @y_coordinates.setter
    def y_coordinates(self, value: Optional[list]):
        raise PyLagAttributeError("Y coordinates are immutable.")

    @property
    def z_coordinates(self):
        return [coords[2] for coords in self.__particle_set]

    @z_coordinates.setter
    def z_coordinates(self, value: Optional[list]):
        raise PyLagAttributeError("Z coordinates are immutable.")

    def get_coordinates(self):
        """ Get particle coordinates

        Returns
        -------
         : array_like
            Particle x-coordinates

         : array_like
            Particle y-coordinates

         : array_like
            Particle z-coordinates
        """
        return self.x_coordinates, self.y_coordinates, self.z_coordinates

    def get_centre_utm_coordinates(self):
        """ Get UTM transformed centre coordinates
        """
        if self.coordinate_system == 'geographic':
            # Transform centre coordinates to UTM
            easting, northing, _ = utm_from_lonlat(self.__centre[0],
                                                   self.__centre[1],
                                                   self.__epsg_code)
            return [easting[0], northing[0]]
        else:
            raise PyLagAttributeError("Cannot return UTM coordinates for "
                                      "release zones defined in cartesian "
                                      "coordinates.")

    def get_utm_coordinates(self):
        """ Get particle UTM coordinates
        
        Return UTM coordiantes for all particles in the set. This
        method will raise an exception if the release zone is defined
        in cartesian coordinates. For release zones defined in
        geographic coordinates, convert these coordinates to UTM
        coordiantes and return. The returned coordinates are in the
        form (eastings, northings, depths). The EPSG code used to
        transform from geographic to UTM coordinates is also returned.

        Parameters
        ----------
        N/A

        Returns
        -------
        eastings : array_like
            Eastings in m.
        
        northings : array_like
            Northings in m.
        
        depths : array_like
            Depths in m.
        
        epsg_code : str
            EPSG code used to transform from geographic to UTM
            coordinates.
        """
        if self.__coordinate_system == "geographic":
            eastings = [coords[0] for coords in self.__particle_set]
            northings = [coords[1] for coords in self.__particle_set]
            depths = [coords[2] for coords in self.__particle_set]

            return eastings, northings, depths, self.__epsg_code
        else:
            raise PyLagAttributeError("Cannot return UTM coordinates for "
                                      "release zones defined in cartesian "
                                      "coordinates.")

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
        delta_x = x - self.__centre[0]
        delta_y = y - self.__centre[1]
        if np.sqrt(delta_x * delta_x + delta_y * delta_y) <= self.__radius:
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


def create_release_zone(group_id: Optional[int] = 1,
                        radius: Optional[float] = 100.0,
                        centre = [0.0, 0.0],
                        coordinate_system: Optional[str] = 'cartesian',
                        epsg_code: Optional[str] = None,
                        n_particles: Optional[int] = 100,
                        depth: Optional[float] = 0.0,
                        random: Optional[bool] = True) -> ReleaseZone:
    """ Create a new release zone

    Parameters
    ----------
    group_id : integer, optional
        Group identifier. Optional, defaults to 1.

    radius : float, optional
        Radius of the circle in meters. Optional, defaults to 100.0 m.

    centre : ndarray [float, float], optional
        x, y coordinates of the circle centre in meters. Optional,
        defaults to [0.0, 0.0].

    coordinate_system : str, optional
        Coordinate system used to interpret the given `centre` coordinates.
        The options are 'geographic' or 'cartesian' (default). If 'geographic'
        is given, the coordinates are assumed to be in lon/lat. If 'cartesian'
        is given, the coordinates are assumed to be in x/y.

    epsg_code : str, optional
        EPSG code which should be used to covert to UTM coordiantes. If
        not given, the EPSG code will be inferred from `centre`. If working
        in cartesian coordinates, this argument is ignored.

    n_particles : integer, optional
        The number of particles. Optional, defaults to 100.
 
    depth : float, optional
        Zone depth in m. Optional, defaults to 0.0 m.

    random : boolean, optional
        Assign x/y positions randomly. Optional, defaults to True.

    Returns
    -------
    release_zone : ReleaseZone
       ReleaseZone object.

    """
    # Create a new release zone given its radius and centre
    release_zone = ReleaseZone(group_id=group_id,
                               radius=radius,
                               centre=centre,
                               coordinate_system=coordinate_system,
                               epsg_code=epsg_code)

    # Create a new particle set of n_particles at the given depth
    release_zone.create_particle_set(n_particles=n_particles,
                                     z=depth,
                                     random=random)

    return release_zone


def create_release_zones_along_cord(
        start_point, end_point,
        coordinate_system: Optional[str] = 'cartesian',
        epsg_code: Optional[str] = None,
        group_id: Optional[int] = 1,
        radius: Optional[float] = 100.0,
        n_particles: Optional[int] = 100,
        depth: Optional[float] = 0.0,
        random: Optional[bool] = True,
        verbose: Optional[bool] = False) -> list:
    """ Generate a set of release zones along a cord

    Return particle positions along a line `r`, defined by the
    position vectors `start_point` and `end_point`.
    `start_point` and `end_point` may be defined in Cartesian or geographic
    coordinates. Optionally, a epsg_code code can be provided.
    If provided, this will be used to convert geographic coordinates into
    UTM coordinates so distances can be calculated.Particles are packed into
    circlular zones of radius `radius`, running along `r`. Positions
    for approximately `n` particles (`= n` if random is `True`) are
    returned per zone. If `2*radius` is `> |r|`, no zones are created.

    Parameters
    ----------
    start_point : array_like [float, float]
        Two component position vector in cartesian or geographic
        coordinates (x,y or lon/lat) that defines the start of the
        cord.

    end_point : array_like [float, float]
        Two component position vector in cartesian coordinates or
        geographic coordinates (x,y or lon/lat) that defines the
        end of the cord.

    coordinate_system : str, optional
        Coordinate system used to interpret the given `r1` and `r2`
        coordinates. The options are 'geographic' or 'cartesian'.

    epsg_code : str, optional
        EPSG code which should be used to covert to UTM coordiantes. If
        not given, the EPSG code will be inferred from `start_point`.
        If working in cartesian coordinates, this argument is ignored.

    group_id : integer, optional
        Group id for the 1st release zone created along the cord.

    radius : float, optional
        The radius of each zone m.

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
    zones : list
        List of release zone objects along the cord.
    """
    if coordinate_system not in ['geographic', 'cartesian']:
        raise ValueError(f"Unrecognised coordinate system "
                         f"{coordinate_system}. Options are "
                         f"'geographic' or 'cartesian'.")

    if coordinate_system == 'geographic':
        if epsg_code is None:
            epsg_code = get_epsg_code(start_point[0], start_point[1])

        x1, y1, _ = utm_from_lonlat(start_point[0],
                                    start_point[1],
                                    epsg_code=epsg_code)
        
        x2, y2, _ = utm_from_lonlat(end_point[0],
                                    end_point[1],
                                    epsg_code)

        r1 = np.array([x1[0], y1[0]], dtype=float)
        r2 = np.array([x2[0], y2[0]], dtype=float)
    else:
        r1 = np.array(start_point, dtype=float)
        r2 = np.array(end_point, dtype=float)

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
        r3_prime = (2.0 * float(n) * radius +
                    radius + buffer_zone/2.0) * r3_unit_vector
        centre_xy = r1 + r3_prime

        if coordinate_system == 'geographic':
            centre_lon, centre_lat = lonlat_from_utm(centre_xy[0],
                                                     centre_xy[1],
                                                     epsg_code)
            centre = np.array([centre_lon[0], centre_lat[0]])
        else:
            centre = np.array(centre_xy[0], centre_xy[1])

        release_zone = create_release_zone(
            group_id=group_id,
            radius=radius,
            centre=centre,
            coordinate_system=coordinate_system,
            epsg_code=epsg_code,
            n_particles=n_particles,
            depth=depth,
            random=random)

        if verbose:
            particles = release_zone.get_number_of_particles()
            print(f"Zone {n} (group_id = {group_id}) contains {particles} particles.")
        release_zones.append(release_zone)
        group_id += 1

    return release_zones


def create_release_zones_around_shape(
        polygon: shapely.geometry.Polygon,
        start_point: shapely.geometry.Point,
        coordinate_system: Optional[str] = 'cartesian',
        epsg_code: Optional[str] = None,
        target_length: Optional[float] = None,
        release_zone_radius: Optional[float] = 100.0,
        n_particles: Optional[int] = 100,
        group_id: Optional[int] = 0,
        depth: Optional[float] = 0.0,
        random: Optional[bool] = True,
        check_overlaps: Optional[bool] = False,
        overlap_tol: Optional[float] = 0.0001,
        verbose: Optional[bool] = False) -> list:
    """ Create a set of adjacent release zones around a shape

    This function will create a set of release zones around the
    perimeter of a polygon. The release zones are created by stepping
    around the perimeter of the polygon, starting at the point
    `start_point`. `start_point` does not need to be exactly on the
    perimeter of the polygon - the method will find the nearest
    vertex and use this as the actual start point. The release zones
    are created such that they are adjacent to one another,
    and each release zone is given a unique groud ID.

    The coordinates used to defined the polygon and the start point
    should be the same, and consistent with the `coordinate_system`
    argument. Valid options for `coordinate_system` are
    'geographic' or 'cartesian'. If 'geographic' is given, the
    coordinates are assumed to be in lon/lat. To compute distances,
    lon/lat coordinates are converted to UTM coordinates using the
    EPSG code given by `epsg_code`. If `epsg_code` is not given, the
    EPSG code is inferred from the start point coordinates. If
    'cartesian' is given, the coordinates are assumed to be in x/y
    and not further coordinate transformations are performed.
    
    The argument `target_length` can be used to specify the length of the
    polygon section around which release zones should be created. If None,
    the release zones will be created around the full perimeter of the
    polygon. The argument `release_zone_radius` specifies the radius
    of each release zone.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon describing a given landmass or object.
        All points in lon/lat coordinates.

    start_point : shapely.geometry.Point
        Approximate start point for release zones in lon/lat
        coordinates. The actual start point will be the nearest
        vertex on the polygon to the given coordinates.

    coordinate_system : str, optional
        Coordinate system used to interpret the coordinates of
        the `polygon` and `start_point` objects. The options are
        'geographic' or 'cartesian'. Optional, defaults to 'cartesian'.

    epsg_code : str, optional
        EPSG code which should be used to covert to UTM coordiantes. If
        not given, the EPSG code will be inferred from `start_point`.

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

    Returns
    -------
    release_zones : list
        List of release zone objects.

    TODO
    ----
    1) Allow users to provide a set of finish coordinates as an
    alternative to a length, which can then be used to determine the
    location of the last release zone to be created.
    """
    # Check the given coordinate system is valid
    if coordinate_system not in ['geographic', 'cartesian']:
        raise ValueError(f"Unrecognised coordinate system "
                         f"{coordinate_system}. Options are "
                         f"'geographic' or 'cartesian'.")

    # Extract the points that make up the polygon.
    points = np.array(polygon.exterior.xy).T

    # Create a second 'working' copy of of the points array. If the original
    # points array is specified in geographic coordinates, transform these
    # to UTM coordinates.
    if coordinate_system == 'geographic':
        if epsg_code is None:
            epsg_code = get_epsg_code(start_point.x, start_point.y)
        
        # Convert start_point to UTM coordinates
        start_point_x, start_point_y, _ = utm_from_lonlat(start_point.x,
                                                          start_point.y,
                                                          epsg_code=epsg_code)
        
        # Convert points to UTM coordinates
        x, y, _ = utm_from_lonlat(points[:, 0], points[:, 1],
                                  epsg_code=epsg_code)
        points_xy = np.array([x, y]).T
    else:
        start_point_x = start_point.x
        start_point_y = start_point.y
        points_xy = points.copy()

    # Total number of points
    n_points = points_xy.shape[0]

    # Establish whether or not the shapefile is ordered in a clockwise
    # or anticlockwise manner
    clockwise_ordering = _is_clockwise_ordered(points_xy)

    # Find starting location using the working points array
    start_idx = _find_start_index(points_xy, start_point_x, start_point_y)

    # Form the first release zone centred on the point for start_idx
    # --------------------------------------------------------------
    release_zones = []
    idx = start_idx - n_points if clockwise_ordering else start_idx
 
    # Coordinates of zone centre (NB from the original points array)
    centre_ref = np.array([points[idx, 0], points[idx, 1]])
    centre_ref_xy = np.array([points_xy[idx, 0], points_xy[idx, 1]])
    release_zone = create_release_zone(group_id=group_id,
                                       radius=release_zone_radius,
                                       coordinate_system=coordinate_system,
                                       centre=centre_ref,
                                       n_particles=n_particles,
                                       depth = depth,
                                       random = random)
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
    last_vertex = centre_ref_xy

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

        current_vertex = np.array([points_xy[idx, 0], points_xy[idx, 1]])

        # Compute current separation
        current_separation = _get_length(centre_ref_xy, current_vertex)

        if current_separation >= target_separation:
            # Track back along the last cord to find the point
            # giving a release zone separation of 2*radius.
            centre_xy = _find_release_zone_location(centre_ref_xy, last_vertex,
                                                    current_vertex,
                                                    target_separation)
            
            # Convert back to lon/lat if necessary
            if coordinate_system == 'geographic':
                centre_lon, centre_lat = lonlat_from_utm(
                    centre_xy[0], centre_xy[1], epsg_code)
                centre = np.array([centre_lon[0], centre_lat[0]])
            else:
                centre = np.array(centre_xy[0], centre_xy[1])

            # Create the release zone
            release_zone = create_release_zone(
                group_id=group_id,
                radius=release_zone_radius,
                centre=centre,
                coordinate_system=coordinate_system,
                epsg_code=epsg_code,
                n_particles=n_particles,
                depth=depth,
                random=random)

            if verbose:
                n = release_zone.get_number_of_particles()
                print(f"Zone (group_id = {group_id}) contains {n} particles.")

            # Check if the new release zone overlaps with non-adjacent zones
            if check_overlaps:
                for zone_test in release_zones:
                    # Get centre
                    centre_test = zone_test.centre
                    if coordinate_system == 'geographic':
                        easting, northing, _ = utm_from_lonlat(
                            centre_test[0], centre_test[1], epsg_code)
                        centre_test = np.array([easting[0], northing[0]])

                    # Computer separation
                    zone_separation = _get_length(centre_xy, centre_test)
                    offset = (target_separation - zone_separation)
                    if offset / target_separation > overlap_tol:
                        print(f"WARNING: Area overlap detected between "
                              f"release zones {zone_test.group_id} "
                              f"and {release_zone.group_id}. "
                              f"Target separation = {target_separation}. "
                              f"Actual separation = {zone_separation}.")

            # Append the new release zone to the current set.
            release_zones.append(release_zone)

            # Update references and counters
            centre_ref_xy = centre_xy
            group_id += 1
        else:
            # Update counters
            distance_travelled += _get_length(last_vertex, current_vertex)
            last_vertex = current_vertex
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


def _find_start_index(points: np.ndarray, x: float, y: float,
                      tolerance: Optional[float] = None) -> int:
    """ Find start index for release zone creation.

    Parameters
    ----------
    points: ndarray
        2D array (n,2) containing x/y coordinates for n locations. Ordering
        is criticial, and must adhere to points[:,0] -> x values, points[:,1]
        -> y values.

    x: float
        Target x

    y: float
        Target y

    tolerance: float, optional
        Raise ValueError if the target x/y values lie beyond this distance
        (in m) from the shape_obj.

    Returns
    -------
    start_idx: integer
        Start index in array points[start_idx,:].

    """
    x_points = points[:, 0]
    y_points = points[:, 1]

    # Find the position on the boundary closest to the supplied x/y values.
    distances = np.hypot(x_points - x, y_points - y)
    distance_min = np.min(distances)
    if tolerance is not None:
        if distance_min < tolerance:
            start_idx = np.argmin(distances)
        else:
            raise ValueError(f"Supplied x/y values are further away than "
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


def _find_release_zone_location(r1: np.ndarray,
                                r2: np.ndarray,
                                r3: np.ndarray,
                                r14_length: float) -> np.ndarray:
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
