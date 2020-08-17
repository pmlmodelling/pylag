"""
Module implementing several basic mathematical functions

Note
----
math is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""
import numpy as np

from libc.math cimport sin, cos

# Degrees -> radians conversion factor
deg_to_radians = np.radians(1)

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
    cdef DTYPE_FLOAT_t r[2]
    cdef DTYPE_FLOAT_t s[2]
    cdef DTYPE_FLOAT_t x13[2]

    cdef DTYPE_FLOAT_t denom, t, u
    cdef DTYPE_INT_t i
    
    for i in xrange(2):
        r[i] = x2[i] - x1[i]
        s[i] = x4[i] - x3[i]
        x13[i] = x3[i] - x1[i]

    denom = det(r,s)

    if denom == 0.0:
        raise ValueError('Lines do not interesct')

    t = det(x13, s) / denom
    u = det(x13, r) / denom
    
    if (0.0 <= u <= 1.0) and (0.0 <= t <= 1.0):
        for i in xrange(2):
            xi[i] = x1[i] + t * r[i]
    else:
        raise ValueError('Line segments do not intersect.')


cdef vector[DTYPE_FLOAT_t] geographic_to_cartesian_coords(DTYPE_FLOAT_t longitude,
                                                          DTYPE_FLOAT_t latitude,
                                                          DTYPE_FLOAT_t r):
    """ Convert geographic to cartesian coordinates

    Parameters
    ----------
    longitude : float
        Longitude in deg E

    latitude : float
        Latitude in deg N

    r : float
        Radius

    Returns
    -------
     coords: vector[float]
         Vector of cartesian coordinates (x, y, z)
    """
    cdef DTYPE_FLOAT_t lon_rad, lat_rad

    cdef vector[DTYPE_FLOAT_t] coords = vector[DTYPE_FLOAT_t](3, -999.)

    lon_rad = longitude * deg_to_radians
    lat_rad = latitude * deg_to_radians

    coords[0] = r * cos(lon_rad) * cos(lat_rad)
    coords[1] = r * sin(lon_rad) * cos(lat_rad)
    coords[2] = r * sin(lat_rad)

    return coords
