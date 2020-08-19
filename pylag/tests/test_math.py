from unittest import TestCase
import numpy as np
import numpy.testing as test

from pylag.data_types_python import DTYPE_FLOAT

from pylag import math


def test_det_second_order():
    a = [1.0, 2.0]
    b = [3.0, 4.0]
    det = math.det_second_order(a, b)
    test.assert_array_almost_equal(det, -2.0)


def test_det_third_order():
    a = [1.0, 2.0, -1.0]
    b = [3.0, 6.0, 0.0]
    c = [0.0, 4.0, 2.0]
    det = math.det_third_order(a, b, c)
    test.assert_array_almost_equal(det, -12.0)


def test_inner_product():
    a = [1.0, 2.0]
    b = [3.0, 4.0]
    c = math.inner_product(a, b)
    test.assert_array_almost_equal(c, 11.0)


def test_rotate_x():
    p = [0., 10., 0.]
    angle = -np.pi/2
    p_rot = math.rotate_x(p, angle)
    test.assert_array_almost_equal(p_rot, [0., 0., 10.])


def test_rotate_y():
    p = [10., 0., 0.]
    angle = np.pi/2
    p_rot = math.rotate_y(p, angle)
    test.assert_array_almost_equal(p_rot, [0., 0., 10.])


def test_rotate_z():
    p = [0., 10., 0.]
    angle = np.pi/2
    p_rot = math.rotate_z(p, angle)
    test.assert_array_almost_equal(p_rot, [10., 0., 0.])


def test_rotate_axes():
    p = [0., 10., 0.]
    lon_rad = np.pi/2
    lat_rad = 0.0
    p_rot = math.rotate_axes(p, lon_rad, lat_rad)
    test.assert_array_almost_equal(p_rot, [0., 0., 10.])


def test_geographic_to_cartesian_coords():
    lon = np.pi/2
    lat = 0.0
    r = 10.0

    coords = math.geographic_to_cartesian_coords(lon, lat, r)
    test.assert_array_almost_equal(coords, [0., 10., 0.])


def test_get_intersection_point_for_perpendicular_lines():
    x1 = np.array([0.0, 2.0], dtype=DTYPE_FLOAT)
    x2 = np.array([0.0, 0.0], dtype=DTYPE_FLOAT)
    x3 = np.array([2.0, 1.0], dtype=DTYPE_FLOAT)
    x4 = np.array([-1.0, 1.0], dtype=DTYPE_FLOAT)
    xi = math.get_intersection_point(x1, x2, x3, x4)
    test.assert_array_almost_equal(xi, [0.0, 1.0])


def test_get_intersection_point_for_angled_lines():
    x1 = np.array([-1.0, -1.0], dtype=DTYPE_FLOAT)
    x2 = np.array([1.0, 1.0], dtype=DTYPE_FLOAT)
    x3 = np.array([0.0, -1.0], dtype=DTYPE_FLOAT)
    x4 = np.array([0.0, 1.0], dtype=DTYPE_FLOAT)
    xi = math.get_intersection_point(x1, x2, x3, x4)
    test.assert_array_almost_equal(xi, [0.0, 0.0])


def test_cartesian_to_sigma_coords():
    h = -50.0
    zeta = 2.0

    z = 2.0
    sigma = math.cartesian_to_sigma_coords(z, h, zeta)
    test.assert_almost_equal(sigma, 0.0)

    z = -24.0
    sigma = math.cartesian_to_sigma_coords(z, h, zeta)
    test.assert_almost_equal(sigma, -0.5)

    z = -50.0
    sigma = math.cartesian_to_sigma_coords(z, h, zeta)
    test.assert_almost_equal(sigma, -1.0)   

