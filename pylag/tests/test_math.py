from unittest import TestCase
import numpy as np
import numpy.testing as test

from pylag.data_types_python import DTYPE_FLOAT

from pylag import math


def test_det_second_order():
    a = [1.0, 2.0]
    b = [3.0, 4.0]
    det = math.det_second_order_wrapper(a, b)
    test.assert_array_almost_equal(det, -2.0)


def test_det_third_order():
    a = [1.0, 2.0, -1.0]
    b = [3.0, 6.0, 0.0]
    c = [0.0, 4.0, 2.0]
    det = math.det_third_order_wrapper(a, b, c)
    test.assert_array_almost_equal(det, -12.0)


def test_euclidian_norm():
    a = [3.0, 4.0, 0.0]
    norm = math.euclidian_norm_wrapper(a)
    test.assert_almost_equal(norm, 5.)


def test_angle_between_two_vectors():
    a = [0.0, 0.0, 1.0]
    b = [0.0, 1.0, 0.0]
    angle = math.angle_between_two_vectors_wrapper(a, b)
    test.assert_almost_equal(angle, np.pi/2.)


def test_unit_vector():
    a = [1., 1., 1.]
    a_unit = math.unit_vector_wrapper(a)
    test.assert_array_almost_equal(a_unit, [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)])


def test_inner_product_two():
    a = [1.0, 2.0]
    b = [3.0, 4.0]
    c = math.inner_product_two_wrapper(a, b)
    test.assert_almost_equal(c, 11.0)


def test_inner_product_three():
    a = [1.0, 2.0, 3.0]
    b = [3.0, 4.0, 1.0]
    c = math.inner_product_three_wrapper(a, b)
    test.assert_almost_equal(c, 14.0)


def test_vector_product():
    a = [1., 1., 0.]
    b = [3., 0., 0.]
    c = math.vector_product_wrapper(a, b)
    test.assert_array_almost_equal(c, [0., 0., -3.])


def test_area_of_a_triangle():
    a = [0., 0.]
    b = [1., 0.]
    c = [0., 1.]

    area = math.area_of_a_triangle_wrapper(a, b, c)

    test.assert_almost_equal(area, 0.5)


def test_area_of_a_spherical_triangle():
    a = [1., 0., 0.]
    b = [0., 1., 0.]
    c = [0., 0., 1.]
    r = 1.0

    area = math.area_of_a_spherical_triangle_wrapper(a, b, c, r)

    test.assert_almost_equal(area, np.pi/2.)


def test_great_circle_arc_segments_intersect():
    a = [np.radians(89.), np.radians(0.)]
    b = [np.radians(91.), np.radians(0.)]
    c = [np.radians(90.), np.radians(-1.)]
    d = [np.radians(90.), np.radians(1.)]
    intersection_is_valid = math.great_circle_arc_segments_intersect_wrapper(a, b, c, d)
    test.assert_equal(intersection_is_valid, 1)


def test_rotate_x():
    p = [0., 10., 0.]
    angle = -np.pi/2
    p_rot = math.rotate_x_wrapper(p, angle)
    test.assert_array_almost_equal(p_rot, [0., 0., 10.])


def test_rotate_y():
    p = [10., 0., 0.]
    angle = np.pi/2
    p_rot = math.rotate_y_wrapper(p, angle)
    test.assert_array_almost_equal(p_rot, [0., 0., 10.])


def test_rotate_z():
    p = [0., 10., 0.]
    angle = np.pi/2
    p_rot = math.rotate_z_wrapper(p, angle)
    test.assert_array_almost_equal(p_rot, [10., 0., 0.])


def test_rotate_axes():
    p = [0., 10., 0.]
    lon_rad = np.pi/2
    lat_rad = 0.0
    p_rot = math.rotate_axes_wrapper(p, lon_rad, lat_rad)
    test.assert_array_almost_equal(p_rot, [0., 0., 10.])


def test_haversine():
    lon1 = np.radians(0)
    lat1 = np.radians(0)
    lon2 = np.radians(180)
    lat2 = np.radians(0)

    distance = math.haversine(lon1, lat1, lon2, lat2)

    test.assert_almost_equal(distance, np.pi)


def test_geographic_to_cartesian_coords():
    lon = np.pi/2
    lat = 0.0
    r = 10.0

    coords = math.geographic_to_cartesian_coords_wrapper(lon, lat, r)
    test.assert_array_almost_equal(coords, [0., 10., 0.])


def test_cartesian_to_geographic_coords():
    coords_cart = [0., 1., 0.]

    coords_geog = math.cartesian_to_geographic_coords_wrapper(coords_cart)

    test.assert_array_almost_equal(coords_geog, [np.pi/2., 0.0])


def test_get_intersection_point_for_perpendicular_lines():
    x1 = np.array([0.0, 2.0], dtype=DTYPE_FLOAT)
    x2 = np.array([0.0, 0.0], dtype=DTYPE_FLOAT)
    x3 = np.array([2.0, 1.0], dtype=DTYPE_FLOAT)
    x4 = np.array([-1.0, 1.0], dtype=DTYPE_FLOAT)
    xi = math.get_intersection_point_wrapper(x1, x2, x3, x4)
    test.assert_array_almost_equal(xi, [0.0, 1.0])


def test_get_intersection_point_in_geographic_coordinates():
    x1 = np.radians([0.0, 2.0]).astype(DTYPE_FLOAT)
    x2 = np.radians([0.0, -1.0]).astype(DTYPE_FLOAT)
    x3 = np.radians([2.0, 0.0]).astype(DTYPE_FLOAT)
    x4 = np.radians([-1.0, 0.0]).astype(DTYPE_FLOAT)
    xi = math.get_intersection_point_in_geographic_coordinates_wrapper(x1, x2, x3, x4)
    test.assert_array_almost_equal(xi, [0.0, 0.0])


def test_get_intersection_point_for_angled_lines():
    x1 = np.array([-1.0, -1.0], dtype=DTYPE_FLOAT)
    x2 = np.array([1.0, 1.0], dtype=DTYPE_FLOAT)
    x3 = np.array([0.0, -1.0], dtype=DTYPE_FLOAT)
    x4 = np.array([0.0, 1.0], dtype=DTYPE_FLOAT)
    xi = math.get_intersection_point_wrapper(x1, x2, x3, x4)
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

