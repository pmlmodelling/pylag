import numpy as np
import numpy.testing as test

#import pylag.interpolation as interp
from pylag.data_types_python import DTYPE_FLOAT, DTYPE_INT
from pylag import interpolation as interp
from pylag import cwrappers

def test_get_barycentric_coords():
    xpos = 1.0
    ypos = 1.0
    x_tri = np.array([0.0, 2.0, 0.0], dtype=DTYPE_FLOAT)
    y_tri = np.array([0.0, 0.0, 2.0], dtype=DTYPE_FLOAT)
    phi = cwrappers.get_barycentric_coords(xpos, ypos, x_tri, y_tri)
    test.assert_array_almost_equal(phi, [0.5, 0.5, 0.0])

def test_shephard_interpolation():
    xpts = np.array([-2.0, -1.0, 1.0, 2.0], dtype=DTYPE_FLOAT)
    ypts = np.array([-2.0, -1.0, 1.0, 2.0], dtype=DTYPE_FLOAT)
    vals = np.array([0.0, 0.0, 1.0, 0.0], dtype=DTYPE_FLOAT)

    val = cwrappers.shepard_interpolation(1.0, 1.0, xpts, ypts, vals)
    test.assert_almost_equal(val, 1.0)
    
    val = cwrappers.shepard_interpolation(0.0, 0.0, xpts, ypts, vals)
    test.assert_almost_equal(val, 0.4)

def test_get_euclidian_distance():
    r1 = interp.get_euclidian_distance(0.0, 0.0, 3.0, 4.0)
    r2 = interp.get_euclidian_distance(0.0, 0.0, -3.0, -4.0)
    test.assert_almost_equal(r1, 5.0)
    test.assert_almost_equal(r2, 5.0)

def test_get_linear_fraction():
    var = 1.5
    var1 = 1.0
    var2 = 2.0
    val = interp.get_linear_fraction(var, var1, var2)
    test.assert_almost_equal(val, 0.5)

def test_linear_interpolation():
    fraction = 0.5
    val_last = 1.0
    val_next = 2.0
    val = interp.linear_interp(fraction, val_last, val_next)
    test.assert_almost_equal(val, 1.5)

def test_interpolate_within_element():
    var = np.array([0.0, 1.0, 2.0])

    phi = np.array([0.5, 0.5, 0.])
    val = cwrappers.interpolate_within_element(var, phi)
    test.assert_almost_equal(val, 1.5)

    phi = np.array([0., 0., 1.])
    val = cwrappers.interpolate_within_element(var, phi)
    test.assert_almost_equal(val, 0.0)
