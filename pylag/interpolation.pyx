include "constants.pxi"

# Data types used for constructing C data structures
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef get_barycentric_coords(DTYPE_FLOAT_t x, DTYPE_FLOAT_t y,
        DTYPE_FLOAT_t x_tri[3], DTYPE_FLOAT_t y_tri[3], DTYPE_FLOAT_t phi[3]):
    """ Get barycentric coordinates.
    
    Compute and return barycentric coordinates for the point (x,y) within the
    triangle defined by x/y coordinates stored in the arrays x_tri and y_tri.
     
    Parameters:
    -----------
    x : float
        x-position.
    
    y : float
        y-position.
    
    x_tri : C array, float
        Triangle x coordinates.
        
    y_tri : C array, float
        Triangle y coordinates.
    
    phi : C array, float
        Barycentric coordinates.
    """

    cdef DTYPE_FLOAT_t a1, a2, a3, a4, det

    a1 = x_tri[1] - x_tri[0]
    a2 = y_tri[2] - y_tri[0]
    a3 = y_tri[1] - y_tri[0]
    a4 = x_tri[2] - x_tri[0]

    # Determinant
    det = a1 * a2 - a3 * a4

    # Transformation to barycentric coordinates
    phi[2] = (a1*(y - y_tri[0]) - a3*(x - x_tri[0]))/det
    phi[1] = (a2*(x - x_tri[2]) - a4*(y - y_tri[2]))/det
    phi[0] = 1.0 - phi[1] - phi[2]

cdef get_barycentric_gradients(DTYPE_FLOAT_t x_tri[3], DTYPE_FLOAT_t y_tri[3],
        DTYPE_FLOAT_t dphi_dx[3], DTYPE_FLOAT_t dphi_dy[3]):
    """ Compute barycentric coordinate gradients with respect to x and y

    Compute and return dphi_i/dx and dphi_i/dy - the gradient in the element's
    barycentric coordinates. In all cases phi_i is linear in both x and y 
    meaning the gradient is constant within the element.

    Parameters:
    -----------
    x_tri : C array, float
        Triangle x coordinates.

    y_tri : C array, float
        Triangle y coordinates.

    dphi_dx : C array, float
        The gradient in phi_i with respect to x.

    dphi_dy : C array, float
        The gradient in phi_i with respect to y.
    """

    cdef DTYPE_FLOAT_t a1, a2, a3, a4, den

    a1 = x_tri[1] - x_tri[0]
    a2 = y_tri[2] - y_tri[0]
    a3 = y_tri[1] - y_tri[0]
    a4 = x_tri[2] - x_tri[0]

    # Denominator
    den = a1 * a2 - a3 * a4

    dphi_dx[0] = (y_tri[1] - y_tri[2])/den
    dphi_dx[1] = (y_tri[2] - y_tri[0])/den
    dphi_dx[2] = (y_tri[0] - y_tri[1])/den

    dphi_dy[0] = (x_tri[2] - x_tri[1])/den
    dphi_dy[1] = (x_tri[0] - x_tri[2])/den
    dphi_dy[2] = (x_tri[1] - x_tri[0])/den

cdef DTYPE_FLOAT_t shepard_interpolation(DTYPE_FLOAT_t x,
        DTYPE_FLOAT_t y, vector[DTYPE_FLOAT_t] xpts, vector[DTYPE_FLOAT_t] ypts,
        vector[DTYPE_FLOAT_t] vals) except FLOAT_ERR:
    """Shepard interpolation.

    """
    # Euclidian distance between the point and a reference point
    cdef DTYPE_FLOAT_t r

    # Weighting applied to a given point
    cdef DTYPE_FLOAT_t w

    # Summed quantities
    cdef DTYPE_FLOAT_t sum
    cdef DTYPE_FLOAT_t sumw

    # For looping
    cdef DTYPE_INT_t i, npts

    # Don't like this much. Would be better to use a cython equivalent to 
    # `zip'. The boost C++ libraries provide something like this, but using
    # it would build in a new dependency.
    if xpts.size() == ypts.size() == vals.size():
        n_pts = xpts.size()
    else:
        raise ValueError('Array lengths do not match.')

    # Loop over all reference points
    sum = 0.0
    sumw = 0.0
    for i in xrange(n_pts):
        r = get_euclidian_distance(x, y, xpts[i], ypts[i])
        if r == 0.0: return vals[i]
        w = 1.0/(r*r) # hardoced p value of -2
        sum = sum + w
        sumw = sumw + w*vals[i]

    return sumw/sum

cdef DTYPE_FLOAT_t get_linear_fraction_safe(DTYPE_FLOAT_t var, DTYPE_FLOAT_t var1,
        DTYPE_FLOAT_t var2) except FLOAT_ERR:
    """Compute the fractional linear distance of a point between two numbers.
    
    The function is deemed safe as it raises an exception if `var' does not lie
    between `var1' and`var2'. Clients should call `get_linear_fraction' if this
    behaviour is not desired.
    
    """
    cdef DTYPE_FLOAT_t frac
    
    frac = get_linear_fraction(var, var1, var2)
    
    if frac >= 0.0 and frac <= 1.0:
        return frac
    else:
        raise ValueError('{} does not lie between {} and {}.'.format(var, var1, var2))
