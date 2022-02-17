""" Unit tests for pylag.processing.coordinate.py """

from unittest import TestCase
import numpy.testing as test
from collections import namedtuple


from pylag.processing.coordinate import utm_from_lonlat, lonlat_from_utm

# Helpers
Coordinates = namedtuple('Coordinates', ('lon', 'lat', 'easting', 'northing', 'zone'))


def _back_forth(lat_arg, lon_arg, zone_arg=False):
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
    easting, northing, zone = utm_from_lonlat(lon_arg, lat_arg, zone_arg)
    lon, lat = lonlat_from_utm(easting, northing, zone)

    return Coordinates(lon=lon, lat=lat, easting=easting, northing=northing, zone=zone)


def test_convert_latlon_coords():
    lat, lon = 50, -5
    coords = _back_forth(lat, lon)

    test.assert_equal(lat, coords.lat)
    test.assert_equal(lon, coords.lon)
    test.assert_equal('30N', coords.zone)

