"""
Module implementing several basic mathematical functions

Note
----
math is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""
import numpy as np

from libc.math cimport sin, cos

from pylag.parameters cimport pi


def det_second_order_wrapper(p1, p2):
    cdef vector[DTYPE_FLOAT_t] _p1 = vector[DTYPE_FLOAT_t](2, -999)
    cdef vector[DTYPE_FLOAT_t] _p2 = vector[DTYPE_FLOAT_t](2, -999)

    _p1 = p1[:]
    _p2 = p2[:]

    return det_second_order(_p1, _p2)


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

cdef get_intersection_point(DTYPE_FLOAT_t x1[2], DTYPE_FLOAT_t x2[2],
        DTYPE_FLOAT_t x3[2], DTYPE_FLOAT_t x4[2], DTYPE_FLOAT_t xi[2]):
    """ Determine the intersection point of two line segments
    
    Determine the intersection point of two line segments. The first is defined
    by the two dimensional position vectors x1 and x2; the second by the two
    dimensional position vectors x3 and x4. The intersection point is found by
    equating the two lines' parametric forms (see Press et al. 2007, p. 1117).
    
    Parameters:
    -----------
    x1, x2, x3, x4 : [float, float]
        Position vectors for the end points of the two lines x1x2 and x3x4.
    
    Returns:
    --------
    xi : list [float, float]
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
        raise ValueError('Lines do not interesct')

    t = det_second_order(x13, s) / denom
    u = det_second_order(x13, r) / denom
    
    if (0.0 <= u <= 1.0) and (0.0 <= t <= 1.0):
        for i in xrange(2):
            xi[i] = x1[i] + t * r[i]
    else:
        raise ValueError('Line segments do not intersect.')


cpdef vector[DTYPE_FLOAT_t] rotate_x(const vector[DTYPE_FLOAT_t] &p, const DTYPE_FLOAT_t &angle):
    """ Rotate the given point about the x-axis

    Parameters
    ----------
    p : vector[float]
        Three vector giving the point's position in cartesian coordinates.

    angle : float
        The angle in radians through which to rotate the point about the x-axis.

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
        The angle in radians through which to rotate the point about the y-axis.

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
        The angle in radians through which to rotate the point about the z-axis.

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

    # First perform three rotations which correctly align the x-, y- and z-axes at (0, 0)
    p_new = rotate_z(p, pi/2.0)
    p_new = rotate_x(p_new, pi/2.0)

    # Now rotate so that the z-axis forms an outward normal through the specified coordinates
    p_new = rotate_y(p_new, lon_rad)
    p_new = rotate_x(p_new, -lat_rad)

    return p_new

cdef vector[DTYPE_FLOAT_t] geographic_to_cartesian_coords(const DTYPE_FLOAT_t &lon_rad,
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
