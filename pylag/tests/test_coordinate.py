""" Unit tests for pylag.processing.coordinate.py """

import unittest
from unittest import TestCase
import numpy.testing as test
from collections import namedtuple
import numpy as np

import pylag.processing.coordinate as coordinate
from pylag.exceptions import PyLagOutOfBoundsError


# Helpers
Coordinates = namedtuple('Coordinates', ('lon', 'lat', 'easting', 'northing', 'zone'))


class FileReader_test(TestCase):

    def _back_forth(self, lat_arg, lon_arg, zone_arg=False):
        """ Simple back and forth test of the functions with a given lat/lon pair

        Parameters
        ----------
        lat_arg : float
            Latitude in degrees N.

        lon_arg : float
            Longitude in degrees E.

        zone_arg : float
            Zone identifier (e.g. 30N)

        Returns
        -------
         : Coordinates
             A Coordinates namedtuple.
        """
        easting, northing, zone = coordinate.utm_from_lonlat(lon_arg, lat_arg, zone_arg)
        lon, lat = coordinate.lonlat_from_utm(easting, northing, zone)

        return Coordinates(lon=lon, lat=lat, easting=easting, northing=northing, zone=zone)

    def test_convert_latlon_coords(self):
        lat, lon = 50, -5
        coords = self._back_forth(lat, lon)

        test.assert_equal(lat, coords.lat)
        test.assert_equal(lon, coords.lon)
        test.assert_equal('30N', coords.zone)

    def test_to_list_with_a_float(self):
        value = 10.0
        x = coordinate.to_list(value)
        self.assertIsInstance(x, list)
        self.assertEqual(x, [10.0])

    def test_to_list_with_an_int(self):
        value = 10
        x = coordinate.to_list(value)
        self.assertIsInstance(x, list)
        self.assertEqual(x, [10])

    def test_to_list_with_a_string(self):
        value = '10'
        x = coordinate.to_list(value)
        self.assertIsInstance(x, list)
        self.assertEqual(x, ['10'])

    def test_to_list_with_a_list(self):
        value = [1.0, 2.0]
        x = coordinate.to_list(value)
        self.assertIsInstance(x, list)
        self.assertEqual(x, [1.0, 2.0])

    def test_to_list_with_a_tuple(self):
        value = (1.0, 2.0)
        x = coordinate.to_list(value)
        self.assertIsInstance(x, list)
        self.assertEqual(x, [1.0, 2.0])

    def test_to_list_with_a_ndarray(self):
        value = np.array([1.0, 2.0])
        x = coordinate.to_list(value)
        self.assertIsInstance(x, list)
        self.assertEqual(x, [1.0, 2.0])

    def test_get_zone_number_with_longitude_and_latitude_for_30U(self):
        latitude = 50.0
        longitude = -4.0

        zone_number = coordinate.get_zone_number(longitude, latitude)

        self.assertEqual(zone_number, 30)

    def test_get_zone_letter_with_latitude_for_30U(self):
        latitude = 50.0

        zone_letter = coordinate.get_zone_letter(latitude)

        self.assertEqual(zone_letter, 'U')

    def test_get_zone_letter_with_invalid_latitude(self):
        latitude = 89.0

        self.assertRaises(PyLagOutOfBoundsError, coordinate.get_zone_letter, latitude)

    def test_check_valid_zone_with_valid_number_and_valid_letter(self):
        zone_number = 30
        zone_letter = 'N'

        coordinate.check_valid_zone(zone_number, zone_letter)

    def test_check_valid_zone_with_valid_number_and_invalid_letter(self):
        zone_number = 30
        zone_letter = 'A'

        self.assertRaises(PyLagOutOfBoundsError, coordinate.check_valid_zone, zone_number, zone_letter)

    def test_check_valid_zone_with_invalid_number(self):
        zone_number = 80
        zone_letter = None

        self.assertRaises(PyLagOutOfBoundsError, coordinate.check_valid_zone, zone_number, zone_letter)
