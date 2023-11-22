from unittest import TestCase

import numpy as np
import numpy.testing as test

from pylag.processing.release_zone import ReleaseZone
from pylag.processing.release_zone import _is_clockwise_ordered as is_clockwise_ordered
from pylag.processing.release_zone import _find_start_index as find_start_index
from pylag.processing.release_zone import _get_length as get_length
from pylag.processing.release_zone import _find_release_zone_location as find_release_zone_location


class ReleaseZoneCreatorTest(TestCase):

    def setUp(self):
        self.group_id = 1
        self.radius = 2.0
        self.centre = [0.0, 0.0]
        self.epsg_code = '32630'

    def tearDown(self):
        del(self.group_id)
        del(self.radius)
        del(self.centre)

    def test_get_group_id(self):
        release_zone = ReleaseZone(self.group_id, self.radius, self.centre)
        group_id = release_zone.group_id
        test.assert_equal(self.group_id, group_id)

    def test_get_radius(self):
        release_zone = ReleaseZone(self.group_id, self.radius, self.centre)
        radius = release_zone.radius
        test.assert_almost_equal(self.radius, radius)

    def test_get_area(self):
        release_zone = ReleaseZone(self.group_id, self.radius, self.centre)
        area = release_zone.area
        test.assert_almost_equal(np.pi*self.radius*self.radius, area)

    def test_get_centre(self):
        release_zone = ReleaseZone(self.group_id, self.radius, self.centre)
        centre = release_zone.centre
        test.assert_almost_equal(self.centre[0], centre[0])
        test.assert_almost_equal(self.centre[1], centre[1])

    def test_add_particle_with_valid_coordinates(self):
        x = self.centre[0] + self.radius/2.0
        y = self.centre[1]
        z = 0.0
        release_zone = ReleaseZone(self.group_id, self.radius, self.centre)
        release_zone.add_particle(x, y, z)
        particle_number = release_zone.get_number_of_particles()
        test.assert_equal(1, particle_number)

    def test_add_particle_with_invalid_coordinates(self):
        x = self.centre[0] + self.radius*2.0
        y = self.centre[1]
        z = 0.0
        release_zone = ReleaseZone(self.group_id, self.radius, self.centre)
        try:
            release_zone.add_particle(x, y, z)
        except ValueError:
            pass
        particle_number = release_zone.get_number_of_particles()
        test.assert_equal(0, particle_number)

    def test_get_coords(self):
        x = self.centre[0]
        y = self.centre[1]
        z = 0.0
        release_zone = ReleaseZone(self.group_id, self.radius, self.centre)
        release_zone.add_particle(x, y, z)
        x_coords, y_coords, z_coords = release_zone.get_coordinates()
        test.assert_array_almost_equal(x, x_coords)
        test.assert_array_almost_equal(y, y_coords)
        test.assert_array_almost_equal(z, z_coords)

    def test_is_clockwise_ordered_is_true(self):
        points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        clockwise_ordered = is_clockwise_ordered(points)
        test.assert_equal(True, clockwise_ordered)

    def test_is_clockwise_ordered_is_false(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        clockwise_ordered = is_clockwise_ordered(points)
        test.assert_equal(False, clockwise_ordered)

    def test_find_start_index(self):
        lon_start = 0.1
        lat_start = 1.1
        points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        start_idx = find_start_index(points, lon_start, lat_start)
        test.assert_equal(1, start_idx)

    def test_get_length(self):
        r1 = np.array([0.0,0.0])
        r2 = np.array([3.0,4.0])
        length = get_length(r1, r2)
        test.assert_almost_equal(5.0, length) 

    def test_find_release_zone_location(self):
        r1 = np.array([0.0, 0.0])
        r2 = np.array([2.0, 2.0])
        r3 = np.array([10.0, 10.0])
        r14_modulus = 8.0
        r4 = np.array([np.sqrt(32.0),np.sqrt(32.0)])
        r4_test = find_release_zone_location(r1, r2, r3, r14_modulus)
        test.assert_array_almost_equal(r4, r4_test)

