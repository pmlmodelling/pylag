# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

cdef class Intersection:
    """ Simple class describing the intersection point of two lines

    The class includes attributes for the coordinates of the end points of the line
    and the coordinates of the intersection point itself.
    """

    def __init__(self, x1=-999., y1=-999., x2=-999., y2=-999., xi=-999., yi=-999.):
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        self._xi = xi
        self._yi = yi

    @property
    def x1(self):
        return self._x1

    @x1.setter
    def x1(self, value):
        self._x1 = value

    @property
    def y1(self):
        return self._y1

    @y1.setter
    def y1(self, value):
        self._y1 = value

    @property
    def x2(self):
        return self._x2

    @x2.setter
    def x2(self, value):
        self._x2 = value

    @property
    def y2(self):
        return self._y2

    @y2.setter
    def y2(self, value):
        self._y2 = value

    @property
    def xi(self):
        return self._xi

    @xi.setter
    def xi(self, value):
        self._xi = value

    @property
    def yi(self):
        return self._yi

    @yi.setter
    def yi(self, value):
        self._yi = value

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
