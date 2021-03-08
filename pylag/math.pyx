"""
Module implementing several basic mathematical functions

Note
----
math is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""
include "constants.pxi"

import numpy as np

from libc.math cimport sin, cos, asin, acos, atan2, sqrt, abs

from pylag.parameters cimport pi, earth_radius, radians_to_deg


def get_intersection_point_wrapper(const vector[DTYPE_FLOAT_t] &x1,
                                   const vector[DTYPE_FLOAT_t] &x2,
                                   const vector[DTYPE_FLOAT_t] &x3,
                                   const vector[DTYPE_FLOAT_t] &x4):
    cdef DTYPE_FLOAT_t _x1[2]
    cdef DTYPE_FLOAT_t _x2[2]
    cdef DTYPE_FLOAT_t _x3[2]
    cdef DTYPE_FLOAT_t _x4[2]
    cdef DTYPE_FLOAT_t _xi[2]

    cdef vector[DTYPE_FLOAT_t] xi = vector[DTYPE_FLOAT_t](2, -999.)

    cdef int i

    if x1.size() != 2 or x2.size() != 2 or x3.size() != 2 or x4.size() != 2:
        raise ValueError('Input arrays should be of length 2.')

    for i in range(2):
        _x1[i] = x1[i]
        _x2[i] = x2[i]
        _x3[i] = x3[i]
        _x4[i] = x4[i]

    if get_intersection_point(_x1, _x2, _x3, _x4, _xi) == 1:
        for i in range(2):
            xi[i] = _xi[i]
        return xi
    else:
        raise ValueError('Lines do not intersect')


cdef DTYPE_INT_t get_intersection_point(const DTYPE_FLOAT_t x1[2],
                                        const DTYPE_FLOAT_t x2[2],
                                        const DTYPE_FLOAT_t x3[2],
                                        const DTYPE_FLOAT_t x4[2],
                                        DTYPE_FLOAT_t xi[2]) except INT_ERR:
    """ Determine the intersection point of two line segments
    
    Determine the intersection point of two line segments. The first is defined
    by the two dimensional position vectors x1 and x2; the second by the two
    dimensional position vectors x3 and x4. The intersection point is found by
    equating the two lines' parametric forms (see Press et al. 2007, p. 1117).
    
    Parameters:
    -----------
    x1, x2, x3, x4 : C array, [float, float]
        Position vectors for the end points of the two lines x1x2 and x3x4.
    
    xi : C array, [float, float]
        x and y coordinates of the intersection point.
    
    References:
    -----------
    Press et al. 2007. Numerical Recipes (Third Edition).
    """
    cdef DTYPE_FLOAT_t r[2]
    cdef DTYPE_FLOAT_t s[2]
    cdef DTYPE_FLOAT_t x13[2]

    cdef DTYPE_FLOAT_t denom, t, u
    cdef DTYPE_INT_t i
    
    for i in xrange(2):
        r[i] = x2[i] - x1[i]
        s[i] = x4[i] - x3[i]
        x13[i] = x3[i] - x1[i]

    denom = det_second_order(r, s)

    if denom == 0.0:
        return 0

    t = det_second_order(x13, s) / denom
    u = det_second_order(x13, r) / denom
    
    if (0.0 <= u <= 1.0) and (0.0 <= t <= 1.0):
        for i in xrange(2):
            xi[i] = x1[i] + t * r[i]
    else:
        return 0

    return 1


def get_intersection_point_in_geographic_coordinates_wrapper(const vector[DTYPE_FLOAT_t] &x1,
                                                             const vector[DTYPE_FLOAT_t] &x2,
                                                             const vector[DTYPE_FLOAT_t] &x3,
                                                             const vector[DTYPE_FLOAT_t] &x4,
                                                             in_degrees = True):
    cdef DTYPE_FLOAT_t _x1[2]
    cdef DTYPE_FLOAT_t _x2[2]
    cdef DTYPE_FLOAT_t _x3[2]
    cdef DTYPE_FLOAT_t _x4[2]
    cdef DTYPE_FLOAT_t _xi[2]

    cdef vector[DTYPE_FLOAT_t] xi = vector[DTYPE_FLOAT_t](2, -999.)

    cdef int i

    if x1.size() != 2 or x2.size() != 2 or x3.size() != 2 or x4.size() != 2:
        raise ValueError('Input arrays should be of length 2.')

    for i in range(2):
        _x1[i] = x1[i]
        _x2[i] = x2[i]
        _x3[i] = x3[i]
        _x4[i] = x4[i]

    if get_intersection_point_in_geographic_coordinates(_x1, _x2, _x3, _x4, _xi) == 1:
        if in_degrees:
            xi[0] = radians_to_deg * _xi[0]
            xi[1] = radians_to_deg * _xi[1]
        else:
            xi[0] = _xi[0]
            xi[1] = _xi[1]
        return xi
    else:
        raise ValueError('Lines do not intersect')


cdef DTYPE_INT_t get_intersection_point_in_geographic_coordinates(const DTYPE_FLOAT_t x1[2],
                                                                  const DTYPE_FLOAT_t x2[2],
                                                                  const DTYPE_FLOAT_t x3[2],
                                                                  const DTYPE_FLOAT_t x4[2],
                                                                  DTYPE_FLOAT_t xi[2]) except INT_ERR:
    """ Determine the intersection point of two line segments in geographic coordinates

    Determine the intersection point of two line segments in geographic
    coordinates. The first line is defined by the two dimensional position
    vectors x1 and x2; the second by the two dimensional position vectors
    x3 and x4.

    Parameters:
    -----------
    x1, x2, x3, x4 : C array, [float, float]
        Position vectors for the end points of the two lines x1x2 and x3x4 in radians.

    xi : C array, [float, float]
        x and y coordinates of the intersection point in radians.
    """
    cdef DTYPE_FLOAT_t x1_cart[3]
    cdef DTYPE_FLOAT_t x2_cart[3]
    cdef DTYPE_FLOAT_t x3_cart[3]
    cdef DTYPE_FLOAT_t x4_cart[3]
    cdef DTYPE_FLOAT_t n1[3]
    cdef DTYPE_FLOAT_t n2[3]
    cdef DTYPE_FLOAT_t l[3]
    cdef DTYPE_FLOAT_t intersection_1[3]
    cdef DTYPE_FLOAT_t intersection_2[3]
    cdef DTYPE_INT_t intersection_1_is_valid, intersection_2_is_valid

    # Convert from geographic to cartesian coordinates
    geographic_to_cartesian_coords(x1[0], x1[1], 1, x1_cart)
    geographic_to_cartesian_coords(x2[0], x2[1], 1, x2_cart)
    geographic_to_cartesian_coords(x3[0], x3[1], 1, x3_cart)
    geographic_to_cartesian_coords(x4[0], x4[1], 1, x4_cart)

    # Compute plane normals for the two arcs
    vector_product(x1_cart, x2_cart, n1)
    vector_product(x3_cart, x4_cart, n2)

    # Compute vector running through the two great circle intersection points
    vector_product(n1, n2, l)

    # Get the two intersection points
    unit_vector(l, intersection_1)
    for i in xrange(3):
        intersection_2[i] = -intersection_1[i]

    # Check if either of the two intersections are valid
    if intersection_is_within_arc_segments(x1_cart, x2_cart, x3_cart, x4_cart, intersection_1) == 1:
        cartesian_to_geographic_coords(intersection_1, xi)
        return 1

    if intersection_is_within_arc_segments(x1_cart, x2_cart, x3_cart, x4_cart, intersection_2) == 1:
        cartesian_to_geographic_coords(intersection_2, xi)
        return 1

    # Intersection was not found!
    return 0


cdef DTYPE_INT_t great_circle_arc_segments_intersect(const DTYPE_FLOAT_t x1[3],
                                                     const DTYPE_FLOAT_t x2[3],
                                                     const DTYPE_FLOAT_t x3[3],
                                                     const DTYPE_FLOAT_t x4[3]) except INT_ERR:
    """ Determine whether two arc segments intersect

    Determine whether two arc segments on the surface of a sphere intersect. The first is
    defined by the two dimensional position vectors x1 and x2; the second by the two
    dimensional position vectors x3 and x4. Position vectors should be in geographic
    coordinates converted into radians.

    Parameters
    ----------
    x1, x2, x3, x4 : vector[float, float]
        Position vectors for the end points of the arcs x1x2 and x3x4. Position vectors
        should be given in geographic coordinates converted into radians.

    Returns
    -------
     : int
        Integer flag identifying whether or not the two arc segments intersect. 0 for
        False, 1 for True.
    """
    cdef DTYPE_FLOAT_t x1_cart[3]
    cdef DTYPE_FLOAT_t x2_cart[3]
    cdef DTYPE_FLOAT_t x3_cart[3]
    cdef DTYPE_FLOAT_t x4_cart[3]
    cdef DTYPE_FLOAT_t n1[3]
    cdef DTYPE_FLOAT_t n2[3]
    cdef DTYPE_FLOAT_t l[3]
    cdef DTYPE_FLOAT_t intersection_1[3]
    cdef DTYPE_FLOAT_t intersection_2[3]
    cdef DTYPE_INT_t intersection_1_is_valid, intersection_2_is_valid

    # Convert from geographic to cartesian coordinates
    geographic_to_cartesian_coords(x1[0], x1[1], 1, x1_cart)
    geographic_to_cartesian_coords(x2[0], x2[1], 1, x2_cart)
    geographic_to_cartesian_coords(x3[0], x3[1], 1, x3_cart)
    geographic_to_cartesian_coords(x4[0], x4[1], 1, x4_cart)

    # Compute plane normals for the two arcs
    vector_product(x1_cart, x2_cart, n1)
    vector_product(x3_cart, x4_cart, n2)

    # Compute vector running through the two great circle intersection points
    vector_product(n1, n2, l)

    # Get the two intersection points
    unit_vector(l, intersection_1)
    for i in xrange(3):
        intersection_2[i] = -intersection_1[i]

    # Check if either of the two intersections are valid
    intersection_1_is_valid = intersection_is_within_arc_segments(x1_cart, x2_cart, x3_cart, x4_cart, intersection_1)
    intersection_2_is_valid = intersection_is_within_arc_segments(x1_cart, x2_cart, x3_cart, x4_cart, intersection_2)

    if intersection_1_is_valid == 0 and intersection_2_is_valid == 0:
        return 0

    return 1


cdef DTYPE_INT_t intersection_is_within_arc_segments(const DTYPE_FLOAT_t x1[3],
                                                     const DTYPE_FLOAT_t x2[3],
                                                     const DTYPE_FLOAT_t x3[3],
                                                     const DTYPE_FLOAT_t x4[3],
                                                     const DTYPE_FLOAT_t xi[3]) except INT_ERR:
    """ Determine whether the intersection lies on the two arc segments

    Parameters
    ----------
    x1, x2, x3, x4 : vector[float, float, float]
        Position vectors for the end points of the arcs x1x2 and x3x4. Position vectors
        should be given in cartesian coordinates.

    xi : vector[float, float, float]
        Position vector of the intersection point under test

    Returns
    -------
     : int
        Integer flag identifying whether or not the intersection point lies within the two arc
        sedments. 0 False, 1 True.
    """
    cdef DTYPE_INT_t is_within_segment_1, is_within_segment_2

    if intersection_is_within_arc_segment(x1, x2, xi) == 0:
        return 0

    if intersection_is_within_arc_segment(x3, x4, xi) == 0:
        return 0

    return 1


cdef DTYPE_INT_t intersection_is_within_arc_segment(const DTYPE_FLOAT_t x1[3],
                                                    const DTYPE_FLOAT_t x2[3],
                                                    const DTYPE_FLOAT_t xi[3]) except INT_ERR:
    """ Determine whether the intersection lies on the arc segment

    Method based an determining whether the angle formed between the two end points
    is equal to the sum of the two angles formed between the intersection point and
    the end points.

    Parameters
    ----------
    x1, x2 : C array, [float, float, float]
        Position vectors for the end points of the arc x1x2. Position vectors
        should be given in cartesian coordinates.

    xi : C array, [float, float, float]
        Position vector of the intersection point under test

    Returns
    -------
     : int
        Integer flag identifying whether or not the intersection point lies within the arc.
        1 is True, 0 False.
    """
    cdef DTYPE_FLOAT_t theta_arc, theta_1, theta_2
    cdef DTYPE_FLOAT_t difference

    # Determine the angle between the two arc end points
    theta_arc = angle_between_two_vectors(x1, x2)

    # Avoid potential numerical issues associated with small angles
    if theta_arc < EPSILON:
        return 0

    # Determine the angles between the arc end points and the intersection point
    theta_1 = angle_between_two_vectors(x1, xi)
    theta_2 = angle_between_two_vectors(x2, xi)

    # Compute the difference
    difference = abs(theta_arc - theta_1 - theta_2)

    if difference > EPSILON:
        return 0

    return 1


cdef DTYPE_FLOAT_t angle_between_two_vectors(const DTYPE_FLOAT_t a[3],
                                             const DTYPE_FLOAT_t b[3]) except FLOAT_ERR:
    """ Determine the angle between two unit vectors
    """
    cdef DTYPE_FLOAT_t x

    # Compute the inner product
    x = inner_product_three(a, b)

    # Defend against numerical issues with acos
    x = float_max(x, -1.0)
    x = float_min(x, 1.0)

    return acos(x)

cdef void unit_vector(const DTYPE_FLOAT_t a[3], DTYPE_FLOAT_t a_unit[3]) except +:
    cdef DTYPE_FLOAT_t norm
    cdef size_t i
    """ Compute unit vector
    """
    norm = euclidian_norm(a)

    for i in xrange(3):
        a_unit[i] = a[i] / norm


cdef void vector_product(const DTYPE_FLOAT_t a[3], const DTYPE_FLOAT_t b[3], DTYPE_FLOAT_t c[3]) except +:
    """ Compute vector product
    """
    c[0] = a[1] * b[2] - b[1] * a[2]
    c[1] = -a[0] * b[2] + b[0] * a[2]
    c[2] = a[0] * b[1] - b[0] * a[1]

cdef void rotate_x(const DTYPE_FLOAT_t p[3], const DTYPE_FLOAT_t &angle, DTYPE_FLOAT_t p_rot[3]) except +:
    """ Rotate the given point about the x-axis

    Parameters
    ----------
    p : C array, float
        Three vector giving the point's position in cartesian coordinates.

    angle : float
        The angle in radians through which to rotate the point anticlockwise about the x-axis.

    p_rot : C array, float
        Three vector giving the point's rotated position in cartesian coordinates.

    """
    cdef DTYPE_FLOAT_t cos_angle, sin_angle

    cos_angle = cos(angle)
    sin_angle = sin(angle)

    p_rot[0] = p[0]
    p_rot[1] = cos_angle * p[1] + sin_angle * p[2]
    p_rot[2] = -sin_angle * p[1] + cos_angle * p[2]


cdef void rotate_y(const DTYPE_FLOAT_t p[3], const DTYPE_FLOAT_t &angle, DTYPE_FLOAT_t p_rot[3]) except +:
    """ Rotate the given point about the y-axis

    Parameters
    ----------
    p : C array, float
        Three vector giving the point's position in cartesian coordinates.

    angle : float
        The angle in radians through which to rotate the point anticlockwise about the y-axis.

    p_rot : C array, float
        Three vector giving the point's rotated position in cartesian coordinates.

    """
    cdef DTYPE_FLOAT_t cos_angle, sin_angle

    cos_angle = cos(angle)
    sin_angle = sin(angle)

    p_rot[0] = cos_angle * p[0] - sin_angle * p[2]
    p_rot[1] = p[1]
    p_rot[2] = sin_angle * p[0] + cos_angle * p[2]


cdef void rotate_z(const DTYPE_FLOAT_t p[3], const DTYPE_FLOAT_t &angle, DTYPE_FLOAT_t p_rot[3]) except +:
    """ Rotate the given point about the z-axis

    Parameters
    ----------
    p : C array, float
        Three vector giving the point's position in cartesian coordinates.

    angle : float
        The angle in radians through which to rotate the point anticlockwise about the z-axis.

    p_rot : C array, float
        Three vector giving the point's rotated position in cartesian coordinates.

    """
    cdef DTYPE_FLOAT_t cos_angle, sin_angle

    cos_angle = cos(angle)
    sin_angle = sin(angle)

    p_rot[0] = cos_angle * p[0] + sin_angle * p[1]
    p_rot[1] = -sin_angle * p[0] + cos_angle * p[1]
    p_rot[2] = p[2]


cdef void rotate_axes(const DTYPE_FLOAT_t p[3], const DTYPE_FLOAT_t &lon_rad,
                      const DTYPE_FLOAT_t &lat_rad, DTYPE_FLOAT_t p_new[3]) except +:
    """ Rotate coordinates axes

    Perform a series of coordinate rotations that rotate the cartesian axes
    so that the positive z-axis forms an outward normal through the given
    geographic coordinates, while the x- and y- axes are locally aligned with
    lines of constant longitude and latitude respectively.

    Parameters
    ----------
    p : C array, float
        Three vector giving the point's position in cartesian coordinates.

    lon_rad : float
        Longitude in radians through which the axes will be rotated.

    lat_rad : float
        Latitude in radians through which the axes will be rotates.

    p_new : C array, float
        Three vector giving the point's rotated position in cartesian coordinates.
    """
    cdef DTYPE_FLOAT_t p_rot_x[3]
    cdef DTYPE_FLOAT_t p_rot_y[3]
    cdef DTYPE_FLOAT_t p_rot_z[3]

    # First perform two rotations which correctly align the x-, y- and z-axes at (0, 0)
    rotate_z(p, pi/2.0, p_rot_z)
    rotate_x(p_rot_z, pi/2.0, p_rot_x)

    # Now rotate so that the z-axis forms an outward normal through the specified coordinates
    rotate_y(p_rot_x, lon_rad, p_rot_y)
    rotate_x(p_rot_y, -lat_rad, p_new)


cdef void reverse_rotate_axes(const DTYPE_FLOAT_t p[3], const DTYPE_FLOAT_t &lon_rad,
                              const DTYPE_FLOAT_t &lat_rad, DTYPE_FLOAT_t p_new[3]) except +:
    """ Reverse rotate coordinates axes

    Perform a series of coordinate rotations that undo the rotations performed
    by the function rotate_axes.

    Parameters
    ----------
    p : C array, float
        Three vector giving the point's position in cartesian coordinates.

    lon_rad : float
        Longitude in radians through which the axes will be rotated.

    lat_rad : float
        Latitude in radians through which the axes will be rotates.

    p_new : C array, float
        Three vector giving the point's rotated position in cartesian coordinates.
    """
    cdef DTYPE_FLOAT_t p_rot_x_1[3]
    cdef DTYPE_FLOAT_t p_rot_x_2[3]
    cdef DTYPE_FLOAT_t p_rot_y[3]

    rotate_x(p, lat_rad, p_rot_x_1)
    rotate_y(p_rot_x_1, -lon_rad, p_rot_y)
    rotate_x(p_rot_y, -pi/2.0, p_rot_x_2)
    rotate_z(p_rot_x_2, -pi/2.0, p_new)


cpdef DTYPE_FLOAT_t haversine(const DTYPE_FLOAT_t &lon1_rad,
                              const DTYPE_FLOAT_t &lat1_rad,
                              const DTYPE_FLOAT_t &lon2_rad,
                              const DTYPE_FLOAT_t &lat2_rad) except FLOAT_ERR:
    """ Calculate the great circle distance between two points on the unit sphere

    Parameters
    ----------
    lon1_rad, lat1_rad : float
        Longitude and latitude of point 1 in radians

    lon2_rad, lat2_rad : float
        Longitude and latitude of point 2 in radians

    Returns
    -------
     : float
        The distance in meters
    """
    cdef DTYPE_FLOAT_t dlon, dlat
    cdef DTYPE_FLOAT_t a

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat/2.)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2.)**2

    return 2.0 * asin(sqrt(a))


cdef geographic_to_cartesian_coords(const DTYPE_FLOAT_t &lon_rad,
                                    const DTYPE_FLOAT_t &lat_rad,
                                    const DTYPE_FLOAT_t &r,
                                    DTYPE_FLOAT_t coords_cart[3]):
    """ Convert geographic to cartesian coordinates

    Parameters
    ----------
    lon_rad : float
        Longitude radians

    lat_rad : float
        Latitude in radians

    r : float
        Radius

     coords_cart: C array, float
         Vector of cartesian coordinates (x, y, z)
    """
    coords_cart[0] = r * cos(lon_rad) * cos(lat_rad)
    coords_cart[1] = r * sin(lon_rad) * cos(lat_rad)
    coords_cart[2] = r * sin(lat_rad)


cdef cartesian_to_geographic_coords(const DTYPE_FLOAT_t coords_cart[3], DTYPE_FLOAT_t coords_geog[2]):
    """ Convert cartesian to geographic coordinates

    Cartesian coordinates should be a unit vector. No check is made to verify this is so.

    Parameters
    ----------
    coords_cart : C array, float
        Cartesian coordinates.

    coords_geog: C array, float
        [lon, lat] coordinates.
    """
    coords_geog[0] = atan2(coords_cart[1], coords_cart[0])
    coords_geog[1] = asin(coords_cart[2])


def geographic_to_cartesian_coords_python(lon_rad, lat_rad):
    """ Convert geographic to cartesian coordinates on a unit sphere

    Pure python implementation of the conversion from geographic to
    cartesian coordinates which acts on numpy arrays.

    Parameters
    ----------
    lon_rad : NumPy 1D array
        Longitude radians

    lat_rad : NumPy 1D array
        Latitude in radians

    Returns
    -------
     x, y, z : NumPy 1D arrays
         Cartesian coordinates.
    """

    x = np.cos(lon_rad) * np.cos(lat_rad)
    y = np.sin(lon_rad) * np.cos(lat_rad)
    z = np.sin(lat_rad)

    return x, y, z


def cartesian_to_geographic_coords_python(x, y, z):
    """ Convert cartesian to geographic coordinates

    (x, y, z) should be unit vectors.

    Parameters
    ----------
    (x,y,z) : float
        (x,y,z) coordinates

    Returns
    -------
     lon, lat: float
         Longitude and latitude in radians
    """
    lon = np.arctan2(y, x)
    lat = np.arcsin(z)

    return lon, lat
