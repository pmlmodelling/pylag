"""
Module implementing several basic mathematical functions

Note
----
math is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""
include "constants.pxi"

import numpy as np

from libc.math cimport sin, cos, asin, acos, sqrt, abs

from pylag.parameters cimport pi, earth_radius


cdef class Intersection:
    """ Simple class describing the intersection point of two lines

    The class includes attributes for the coordinates of the end points of the line
    and the coordinates of the intersection point itself.
    """

    def __init__(self, x1=-999., y1=-999., x2=-999., y2=-999., xi=-999., yi=-999.):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.xi = xi
        self.yi = yi

    @property
    def x1_py(self):
        return self.x1

    @x1_py.setter
    def x1_py(self, value):
        self.x1 = value

    @property
    def y1_py(self):
        return self.y1

    @y1_py.setter
    def y1_py(self, value):
        self.y1 = value

    @property
    def x2_py(self):
        return self.x2

    @x2_py.setter
    def x2_py(self, value):
        self.x2 = value

    @property
    def y2_py(self):
        return self.y2

    @y2_py.setter
    def y2_py(self, value):
        self.y2 = value

    @property
    def xi_py(self):
        return self.xi

    @xi_py.setter
    def xi_py(self, value):
        self.xi = value

    @property
    def yi_py(self):
        return self.yi

    @yi_py.setter
    def yi_py(self, value):
        self.yi = value

def get_intersection_point_wrapper(const vector[DTYPE_FLOAT_t] &x1,
                                    const vector[DTYPE_FLOAT_t] &x2,
                                    const vector[DTYPE_FLOAT_t] &x3,
                                    const vector[DTYPE_FLOAT_t] &x4):
    cdef vector[DTYPE_FLOAT_t] xi = vector[DTYPE_FLOAT_t](2, -999.)

    if get_intersection_point(x1, x2, x3, x4, xi) == 1:
        return xi
    else:
        raise ValueError('Lines do not intersect')


cdef DTYPE_INT_t get_intersection_point(const vector[DTYPE_FLOAT_t] &x1,
                                    const vector[DTYPE_FLOAT_t] &x2,
                                    const vector[DTYPE_FLOAT_t] &x3,
                                    const vector[DTYPE_FLOAT_t] &x4,
                                    vector[DTYPE_FLOAT_t] &xi) except INT_ERR:
    """ Determine the intersection point of two line segments
    
    Determine the intersection point of two line segments. The first is defined
    by the two dimensional position vectors x1 and x2; the second by the two
    dimensional position vectors x3 and x4. The intersection point is found by
    equating the two lines' parametric forms (see Press et al. 2007, p. 1117).
    
    Parameters:
    -----------
    x1, x2, x3, x4 : vector[float, float]
        Position vectors for the end points of the two lines x1x2 and x3x4.
    
    Returns:
    --------
    xi : vector[float, float]
        x and y coordinates of the intersection point.
    
    References:
    -----------
    Press et al. 2007. Numerical Recipes (Third Edition).
    """
    cdef vector[DTYPE_FLOAT_t] r = vector[DTYPE_FLOAT_t](2, -999.)
    cdef vector[DTYPE_FLOAT_t] s = vector[DTYPE_FLOAT_t](2, -999.)
    cdef vector[DTYPE_FLOAT_t] x13 = vector[DTYPE_FLOAT_t](2, -999.)

    cdef DTYPE_FLOAT_t denom, t, u
    cdef DTYPE_INT_t i
    
    for i in xrange(2):
        r[i] = x2[i] - x1[i]
        s[i] = x4[i] - x3[i]
        x13[i] = x3[i] - x1[i]

    denom = det_second_order(r,s)

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

cpdef DTYPE_INT_t great_circle_arc_segments_intersect(const vector[DTYPE_FLOAT_t] &x1,
                                                      const vector[DTYPE_FLOAT_t] &x2,
                                                      const vector[DTYPE_FLOAT_t] &x3,
                                                      const vector[DTYPE_FLOAT_t] &x4) except INT_ERR:
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
    cdef vector[DTYPE_FLOAT_t] x1_cart, x2_cart, x3_cart, x4_cart
    cdef vector[DTYPE_FLOAT_t] n1, n2
    cdef vector[DTYPE_FLOAT_t] l
    cdef vector[DTYPE_FLOAT_t] intersection_1 = vector[DTYPE_FLOAT_t](3, 999.)
    cdef vector[DTYPE_FLOAT_t] intersection_2 = vector[DTYPE_FLOAT_t](3, 999.)
    cdef DTYPE_INT_t intersection_1_is_valid, intersection_2_is_valid

    # Convert from geographic to cartesian coordinates
    x1_cart = geographic_to_cartesian_coords(x1[0], x1[1], 1)
    x2_cart = geographic_to_cartesian_coords(x2[0], x2[1], 1)
    x3_cart = geographic_to_cartesian_coords(x3[0], x3[1], 1)
    x4_cart = geographic_to_cartesian_coords(x4[0], x4[1], 1)

    # Compute plane normals for the two arcs
    n1 = vector_product(x1_cart, x2_cart)
    n2 = vector_product(x3_cart, x4_cart)

    # Compute vector running through the two great circle intersection points
    l = vector_product(n1, n2)

    # Get the two intersection points
    intersection_1 = unit_vector(l)
    for i in xrange(3):
        intersection_2[i] = -intersection_1[i]

    # Check if either of the two intersections are valid
    intersection_1_is_valid = intersection_is_within_arc_segments(x1_cart, x2_cart, x3_cart, x4_cart, intersection_1)
    intersection_2_is_valid = intersection_is_within_arc_segments(x1_cart, x2_cart, x3_cart, x4_cart, intersection_2)

    if intersection_1_is_valid == 0 and intersection_2_is_valid == 0:
        return 0

    return 1

cpdef DTYPE_INT_t intersection_is_within_arc_segments(const vector[DTYPE_FLOAT_t] &x1,
                                                      const vector[DTYPE_FLOAT_t] &x2,
                                                      const vector[DTYPE_FLOAT_t] &x3,
                                                      const vector[DTYPE_FLOAT_t] &x4,
                                                      const vector[DTYPE_FLOAT_t] &xi) except INT_ERR:
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

    is_within_segment_1 = intersection_is_within_arc_segment(x1, x2, xi)
    is_within_segment_2 = intersection_is_within_arc_segment(x3, x4, xi)

    if is_within_segment_1 == 1 and is_within_segment_2 == 1:
        return 1

    return 0

cpdef DTYPE_INT_t intersection_is_within_arc_segment(const vector[DTYPE_FLOAT_t] &x1,
                                                     const vector[DTYPE_FLOAT_t] &x2,
                                                     const vector[DTYPE_FLOAT_t] &xi) except INT_ERR:
    """ Determine whether the intersection lies on the arc segment

    Method based an determining whether the angle formed between the two end points
    is equal to the sum of the two angles formed between the intersection point and
    the end points.

    Parameters
    ----------
    x1, x2 : vector[float, float, float]
        Position vectors for the end points of the arc x1x2. Position vectors
        should be given in cartesian coordinates.

    xi : vector[float, float, float]
        Position vector of the intersection point under test

    Returns
    -------
     : int
        Integer flag identifying whether or not the intersection point lies within the arc
    """
    cdef DTYPE_FLOAT_t theta_arc, theta_1, theta_2
    cdef DTYPE_FLOAT_t difference

    # Determine the angle between the two arc end points
    theta_arc = angle_between_two_vectors(x1, x2)

    # Determine the angles between the arc end points and the intersection point
    theta_1 = angle_between_two_vectors(x1, xi)
    theta_2 = angle_between_two_vectors(x2, xi)

    # Compute the difference
    difference = abs(theta_arc - theta_1 - theta_2)

    if difference > EPSILON:
        return 0

    return 1


cpdef DTYPE_FLOAT_t angle_between_two_vectors(const vector[DTYPE_FLOAT_t] &a,
                                              const vector[DTYPE_FLOAT_t] &b) except FLOAT_ERR:
    """ Determine the angle between two vectors
    """
    return acos(inner_product_three(a, b) / (euclidian_norm(a) * euclidian_norm(b)))


cpdef vector[DTYPE_FLOAT_t] unit_vector(const vector[DTYPE_FLOAT_t] &a) except *:
    cdef vector[DTYPE_FLOAT_t] a_unit = vector[DTYPE_FLOAT_t](3, 999.)
    cdef DTYPE_FLOAT_t norm
    cdef DTYPE_INT_t i
    """ Compute unit vector
    """
    norm = euclidian_norm(a)

    for i in xrange(3):
        a_unit[i] = a[i] / norm

    return a_unit


cpdef vector[DTYPE_FLOAT_t] vector_product(const vector[DTYPE_FLOAT_t] &a,
                                           const vector[DTYPE_FLOAT_t] &b) except *:
    """ Compute vector product
    """
    cdef vector[DTYPE_FLOAT_t] c = vector[DTYPE_FLOAT_t](3, 999.)

    c[0] = a[1] * b[2] - b[1] * a[2]
    c[1] = -a[0] * b[2] + b[0] * a[2]
    c[2] = a[0] * b[1] - b[0] * a[1]

    return c


cpdef vector[DTYPE_FLOAT_t] rotate_x(const vector[DTYPE_FLOAT_t] &p, const DTYPE_FLOAT_t &angle):
    """ Rotate the given point about the x-axis

    Parameters
    ----------
    p : vector[float]
        Three vector giving the point's position in cartesian coordinates.

    angle : float
        The angle in radians through which to rotate the point anticlockwise about the x-axis.

    Returns
    -------
     : vector[float]
         The rotated position vector.
    """
    cdef DTYPE_FLOAT_t cos_angle, sin_angle
    cdef vector[DTYPE_FLOAT_t] p_rot = vector[DTYPE_FLOAT_t](3, -999.)

    cos_angle = cos(angle)
    sin_angle = sin(angle)

    p_rot[0] = p[0]
    p_rot[1] = cos_angle * p[1] + sin_angle * p[2]
    p_rot[2] = -sin_angle * p[1] + cos_angle * p[2]

    return p_rot


cpdef vector[DTYPE_FLOAT_t] rotate_y(const vector[DTYPE_FLOAT_t] &p, const DTYPE_FLOAT_t &angle):
    """ Rotate the given point about the y-axis

    Parameters
    ----------
    p : vector[float]
        Three vector giving the point's position in cartesian coordinates.

    angle : float
        The angle in radians through which to rotate the point anticlockwise about the y-axis.

    Returns
    -------
     : vector[float]
         The rotated position vector.
    """
    cdef DTYPE_FLOAT_t cos_angle, sin_angle
    cdef vector[DTYPE_FLOAT_t] p_rot = vector[DTYPE_FLOAT_t](3, -999.)

    cos_angle = cos(angle)
    sin_angle = sin(angle)

    p_rot[0] = cos_angle * p[0] - sin_angle * p[2]
    p_rot[1] = p[1]
    p_rot[2] = sin_angle * p[0] + cos_angle * p[2]

    return p_rot


cpdef vector[DTYPE_FLOAT_t] rotate_z(const vector[DTYPE_FLOAT_t] &p, const DTYPE_FLOAT_t &angle):
    """ Rotate the given point about the z-axis

    Parameters
    ----------
    p : vector[float]
        Three vector giving the point's position in cartesian coordinates.

    angle : float
        The angle in radians through which to rotate the point anticlockwise about the z-axis.

    Returns
    -------
     : vector[float]
         The rotated position vector.
    """
    cdef DTYPE_FLOAT_t cos_angle, sin_angle
    cdef vector[DTYPE_FLOAT_t] p_rot = vector[DTYPE_FLOAT_t](3, -999.)

    cos_angle = cos(angle)
    sin_angle = sin(angle)

    p_rot[0] = cos_angle * p[0] + sin_angle * p[1]
    p_rot[1] = -sin_angle * p[0] + cos_angle * p[1]
    p_rot[2] = p[2]

    return p_rot


cpdef vector[DTYPE_FLOAT_t] rotate_axes(const vector[DTYPE_FLOAT_t] &p,
                                        const DTYPE_FLOAT_t &lon_rad,
                                        const DTYPE_FLOAT_t &lat_rad):
    """ Rotate coordinates axes

    Perform a series of coordinate rotations that rotate the cartesian axes
    so that the positive z-axis forms an outward normal through the given
    geographic coordinates, while the x- and y- axes are locally aligned with
    lines of constant longitude and latitude respectively.

    Parameters
    ----------
    p : vector[float]
        Three vector giving the point's position in cartesian coordinates.

    lon_rad : float
        Longitude in radians through which the axes will be rotated.

    lat_rad : float
        Latitude in radians through which the axes will be rotates.
    """
    cdef vector[DTYPE_FLOAT_t] p_new

    # First perform two rotations which correctly align the x-, y- and z-axes at (0, 0)
    p_new = rotate_z(p, pi/2.0)
    p_new = rotate_x(p_new, pi/2.0)

    # Now rotate so that the z-axis forms an outward normal through the specified coordinates
    p_new = rotate_y(p_new, lon_rad)
    p_new = rotate_x(p_new, -lat_rad)

    return p_new


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


cpdef vector[DTYPE_FLOAT_t] geographic_to_cartesian_coords(const DTYPE_FLOAT_t &lon_rad,
                                                           const DTYPE_FLOAT_t &lat_rad,
                                                           const DTYPE_FLOAT_t &r):
    """ Convert geographic to cartesian coordinates

    Parameters
    ----------
    lon_rad : float
        Longitude radians

    lat_rad : float
        Latitude in radians

    r : float
        Radius

    Returns
    -------
     coords: vector[float]
         Vector of cartesian coordinates (x, y, z)
    """
    cdef vector[DTYPE_FLOAT_t] coords = vector[DTYPE_FLOAT_t](3, -999.)

    coords[0] = r * cos(lon_rad) * cos(lat_rad)
    coords[1] = r * sin(lon_rad) * cos(lat_rad)
    coords[2] = r * sin(lat_rad)

    return coords
