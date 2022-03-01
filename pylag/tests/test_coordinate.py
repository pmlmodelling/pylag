""" Unit tests for pylag.processing.coordinate.py """

import unittest
from unittest import TestCase
import numpy.testing as test
from collections import namedtuple
import numpy as np

import pylag.processing.coordinate as coordinate
from pylag.exceptions import PyLagOutOfBoundsError


# Helpers
Coordinates = namedtuple('Coordinates', ('lon', 'lat', 'easting', 'northing', 'epsg'))


class FileReader_test(TestCase):

    def _back_forth(self, lon_arg, lat_arg, epsg_code: int):
        """ Simple back and forth test of the functions with a given lat/lon pair

        Parameters
        ----------
        lon_arg : float
            Longitude in degrees E.

        lat_arg : float
            Latitude in degrees N.

        epsg_code : int
            EPSG code.

        Returns
        -------
         : Coordinates
             A Coordinates namedtuple.
        """
        easting, northing = coordinate.utm_from_lonlat(lon_arg, lat_arg, epsg_code)
        lon, lat = coordinate.lonlat_from_utm(easting, northing, epsg_code)

        return Coordinates(lon=lon, lat=lat, easting=easting, northing=northing, epsg=epsg_code)

    def test_convert_latlon_coords(self):
        # Test values (Easting: 428333.55, Northing: 5539109.82)
        lon, lat = -4., 50.
        epsg_code = 32630

        coords = self._back_forth(lon, lat, epsg_code)

        test.assert_almost_equal(coords.lon[0], lon)
        test.assert_almost_equal(coords.lat[0], lat)
        test.assert_almost_equal(coords.easting[0], 428333.55, decimal=2)
        test.assert_almost_equal(coords.northing[0], 5539109.82, decimal=2)

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
