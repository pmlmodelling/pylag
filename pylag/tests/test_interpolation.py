import numpy as np
import numpy.testing as test

from pylag.data_types_python import DTYPE_FLOAT
from pylag import interpolation as interp


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

    var = 1.0
    val = interp.get_linear_fraction(var, var1, var2)
    test.assert_almost_equal(val, 0.0)

    var = 2.0
    val = interp.get_linear_fraction(var, var1, var2)
    test.assert_almost_equal(val, 1.0)

def test_linear_interpolation():
    fraction = 0.5
    val_last = 1.0
    val_next = 2.0
    val = interp.linear_interp(fraction, val_last, val_next)
    test.assert_almost_equal(val, 1.5)

    fraction = 0.0
    val = interp.linear_interp(fraction, val_last, val_next)
    test.assert_almost_equal(val, 1.0)

    fraction = 1.0
    val = interp.linear_interp(fraction, val_last, val_next)
    test.assert_almost_equal(val, 2.0)

    fraction = 0.0-1e-10
    val = interp.linear_interp(fraction, val_last, val_next)
    test.assert_almost_equal(val, 1.0)

    fraction = 1.0+1e-10
    val = interp.linear_interp(fraction, val_last, val_next)
    test.assert_almost_equal(val, 2.0)


def test_interpolate_within_element():
    var = np.array([0.0, 1.0, 2.0])

    phi = np.array([0.0, 0.5, 0.5])
    val = interp.interpolate_within_element(var, phi)
    test.assert_almost_equal(val, 1.5)

    phi = np.array([0., 0., 1.])
    val = interp.interpolate_within_element(var, phi)
    test.assert_almost_equal(val, 2.0)
