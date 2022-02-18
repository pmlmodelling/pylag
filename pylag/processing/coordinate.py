"""
The version of coordinate.py provided here is taken directly from
the PyFVCOM library, which can be accessed via GitHub. It is copied
here in order to minimise PyLag dependencies. The DOI for PyFVCOM is
DOI: https://doi.org/10.5281/zenodo.1422462.

Utilities to convert from cartesian UTM coordinates to spherical latitude and
longitudes.

This function uses functions lifted from the utm library. For those that are
interested, the reason we don't just use the utm utility is because it
refuses to do coordinate conversions outside a single zone.

"""

from __future__ import division

import multiprocessing
import numpy as np

import pyproj

from pylag.exceptions import PyLagRuntimeError, PyLagTypeError, PyLagOutOfBoundsError


def to_list(x):
    """ Try to convert `x` to a list

    Support for strings, ints, floats, tuples and ndarrays is included.

    Parameters
    ----------
    x : object
        The object to convert.

    Returns
    -------
     : list
        The processed object transformed to a list.

    Raises
    ------
    PyLagTypeError is the object is not supported.
    """

    if isinstance(x, str) or isinstance(x, (int, float)):
        return [x]
    elif isinstance(x, tuple):
        return list(x)
    elif isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, list):
        return x

    x_type = type(x)

    raise PyLagTypeError('Unexpected type {x_type!r}')


def __test(inLat, inLong, inZone=False):
    """ Simple back and forth test of the functions with a given lat/long pair.

    Parameters
    ----------
    inLat, inLong : float
        Input latitude and longitude pair.

    """
    e, n, z = utm_from_lonlat(inLong, inLat, inZone)
    lon, lat = lonlat_from_utm(e, n, z)

    return z, e, n, lon, lat


def __convert(args):
    """
    Child function to create a projection object and convert coordinates (optionally inverse).

    This function can therefore do UTM -> lat/lon as well as lat/lon -> UTM transformations.

    Parameters
    ----------
    args : tuple
        Arguments:
            in_x, in_y : float
            zone, ellipsoid, datum : str
            inverse : bool

    Returns
    -------
    out_x, out_y : float
        Coordinates following conversion.

    """
    a, b, zone, ellipsoid, datum, inverse = args
    projection = pyproj.Proj(f"+proj=utm +zone={zone}, +ellps={ellipsoid} +datum={datum} +units=m +no_defs")
    c, d = projection(a, b, inverse=inverse)

    return c, d


def get_zone_number(longitude, latitude):
    """
    Calculate the UTM zone number from the given coordinate. Shamelessly lifted
    from the utm.lonlat_to_zone_number function.

    Parameters
    ----------
    longitude : float
        Longitude in degrees East.

    latitude : float
        Latitude in degrees North.

    Returns
    -------
    zone_number : int
        Zone number

    """

    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude <= 9:
            return 31
        elif longitude <= 21:
            return 33
        elif longitude <= 33:
            return 35
        elif longitude <= 42:
            return 37

    return int(((longitude + 180) % 360) / 6) + 1


def get_zone_letter(latitude):
    """
    Calculate the UTM zone letter from the given latitude. Shamelessly lifted
    from the utm.latitude_to_zone_letter function.

    Parameters
    ----------
    latitude : float
        Latitude

    Returns
    -------
    zone_letter : str
        Zone letter

    Raises
    ------
    PyLagOutOfBoundsError - For a latitude that is < -80 or > 84
    """
    ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"

    if -80 <= latitude <= 84:
        return ZONE_LETTERS[int(latitude + 80) >> 3]

    raise PyLagOutOfBoundsError(f'Latitude of {latitude!r} out of range for utm zones')


def utm_from_lonlat(longitude, latitude, zone_number, northern, ellipsoid='WGS84',
                    datum='WGS84', parallel=False):
    """ Converts lats and lons to UTM for the specified zone if given.

    East Longitudes are positive, west longitudes are negative. North latitudes
    are positive, south latitudes are negative. Lat and Lon are in decimal
    degrees.

    Be careful when using this function for coordinates that span multiple zones
    as the transformation is specific to the specified zone.

    Parameters
    ----------
    longitude, latitude : float, tuple, list, np.ndarray
        Longitudes and latitudes.
    zone_number : int
        Zone number (e.g. 30).
    northern : bool
        Flag signifying whether the zone is in the northern or southern hemisphere.
    ellipsoid : str, optional
        Give an ellipsoid for the conversion. Defaults to WGS84.
    datum : str, optional
        Give a datum for the conversion. Defaults to WGS84.
    parallel : bool, optional
        Optionally enable the parallel processing (sometimes this is faster
        for a very large number of positions). Defaults to False.

    Returns
    -------
    eastings, northings : np.ndarray
        Eastings and northings for the given longitude, latitudes and zone.
    """
    lon = to_list(longitude)
    lat = to_list(latitude)

    zone = f'{zone_number}'
    if not northern:
        zone = f'{zone}S'

    inverse = False

    n_positions = len(lon)
    if n_positions != len(lat):
        raise PyLagRuntimeError('Lat and lon array sizes do not match')

    # Do this in parallel unless we only have a single position or we've been
    # told not to.
    if n_positions > 1 and parallel:
        pool = multiprocessing.Pool()
        arguments = zip(lon, lat,
                        [zone] * n_positions,
                        [ellipsoid] * n_positions,
                        [datum] * n_positions,
                        [inverse] * n_positions)
        results = pool.map(__convert, arguments)
        results = np.asarray(results)
        eastings, northings = results[:, 0], results[:, 1]
        pool.close()
    elif n_positions > 1 and not parallel:
        eastings, northings = [], []
        for pos in zip(lon, lat,
                       [zone] * n_positions,
                       [ellipsoid] * n_positions,
                       [datum] * n_positions,
                       [inverse] * n_positions):
            result = __convert(pos)
            eastings.append(result[0])
            northings.append(result[1])
    else:
        eastings, northings = __convert((lon[0], lat[0], zone, ellipsoid, datum, inverse))

        eastings = to_list(eastings)
        northings = to_list(northings)

    return np.asarray(eastings), np.asarray(northings), np.asarray(zone)


def lonlat_from_utm(eastings, northings, zone, ellipsoid='WGS84', datum='WGS84', parallel=False):
    """
    Converts UTM coordinates to lat/long.

    East Longitudes are positive, west longitudes are negative. North latitudes
    are positive, south latitudes are negative. Lat and Long are in decimal
    degrees.

    Parameters
    ----------
    eastings, northings : float, tuple, list, np.ndarray
        Eastings and northings.
    zone : str, tuple, list, np.ndarray
        Zone number and letter (e.g. 30N).
    ellipsoid : str, optional
        Give an ellipsoid for the conversion. Defaults to WGS84.
    datum : str, optional
        Give a datum for the conversion. Defaults to WGS84.
    parallel : bool, optional
        Optionally enable parallel processing (sometimes this is faster
        for a large number of positions). Defaults to False.

    Returns
    -------
    lon, lat : float, np.ndarray
        Longitude and latitudes for the given eastings and northings.

    """

    eastings = to_list(eastings)
    northings = to_list(northings)
    zone = to_list(zone)

    inverse = True

    npos = len(eastings)
    if npos != len(northings):
        raise ValueError('Supplied eastings and northings are not the same size.')

    # If we've been given a single zone and multiple coordinates, make a
    # list of zones so we can do things easily in parallel.
    if len(zone) != npos:
        zone = zone * npos

    # Do this in parallel unless we only have a small number of positions or
    # we've been told not to.
    if npos > 1 and parallel:
        pool = multiprocessing.Pool()
        arguments = zip(eastings, northings, zone,
                        [ellipsoid] * npos,
                        [datum] * npos,
                        [inverse] * npos)
        results = pool.map(__convert, arguments)
        results = np.asarray(results)
        lon, lat = results[:, 0], results[:, 1]
        pool.close()
    elif npos > 1 and not parallel:
        lon, lat = [], []
        for pos in zip(eastings, northings, zone, [ellipsoid] * npos, [datum] * npos, [inverse] * npos):
            result = __convert(pos)
            lon.append(result[0])
            lat.append(result[1])
    else:
        # The eastings, northings and zone will all be lists here, For
        # cross-python2/python3 support, we can't just * them, so assume
        # the first value in the list is what we want.
        lon, lat = __convert((eastings[0], northings[0], zone[0], ellipsoid, datum, inverse))

    return np.asarray(lon), np.asarray(lat)


def british_national_grid_to_lonlat(eastings, northings):
    """
    Converts British National Grid coordinates to latitude and longitude on the WGS84 spheroid.

    Parameters
    ----------
    eastings : ndarray
        Array of eastings (in metres)
    northings : ndarray
        Array of northings (in metres)

    Returns
    -------
    lon : ndarray
        Array of converted longitudes (decimal degrees)
    lat : ndarray
        Array of converted latitudes (decimal degrees)

    """

    crs_british = pyproj.Proj(init='EPSG:27700')
    crs_wgs84 = pyproj.Proj(init='EPSG:4326')
    lon, lat = pyproj.transform(crs_british, crs_wgs84, eastings, northings)

    return lon, lat


def lonlat_decimal_from_degminsec(lon_degminsec, lat_degminsec):
    """
    Converts from lon lat in degrees minutes and seconds to decimal degrees

    Parameters
    ----------
    lon_degminsec : Mx3 np.ndarray
        Array of longitude degrees, minutes and seconds in 3 separate columns (for M positions). East and West is
        determined by the sign of the leading non-zero number (e.g. [-4, 20, 16] or [0, -3, 10])
    lat_degminsec : Mx3 np.ndarray
        Array of latitude degrees, minutes and seconds in 3 seperate columns (for M positions). North and South are
        determined by the sign of the leading number.

    Returns
    -------
    lon : np.ndarray
        Array of converted decimal longitudes.
    lat : np.ndarray
        Array of converted decimal latitudes.

    """
    right_dims = (np.ndim(lon_degminsec) == 2 and np.ndim(lat_degminsec) == 2)
    right_columns = (np.shape(lon_degminsec)[1] == 3 and np.shape(lat_degminsec)[1] == 3)
    if not right_dims or not right_columns:
        raise ValueError('Inputs are the wrong shape: incorrect dimensions (2) or number of columns (3).')

    lon_sign = np.sign(lon_degminsec)
    lat_sign = np.sign(lat_degminsec)
    # Zeros have a sign of 0, but for our arithmetic, we need only -1 and 1, so replace 0 with 1 in the sign arrays.
    lon_sign[lon_sign == 0] = 1
    lat_sign[lat_sign == 0] = 1
    # Since we've got only -1 and 1 in the arrays now, we can just find the minimum along each row which will tell us
    # if we've got negative anything for a given set of coordinates. Since we're doing all the arithmetic on positive
    # only numbers, we can then just flip the sign for the correct hemisphere at the end.
    lon_sign = np.min(lon_sign, axis=1)
    lat_sign = np.min(lat_sign, axis=1)

    lon_adj = np.abs(lon_degminsec)
    lat_adj = np.abs(lat_degminsec)

    lon = lon_adj[:, 0] + lon_adj[:, 1] / 60 + lon_adj[:, 2] / 3600
    lat = lat_adj[:, 0] + lat_adj[:, 1] / 60 + lat_adj[:, 2] / 3600

    lon = lon_sign * lon
    lat = lat_sign * lat

    return lon, lat


def lonlat_decimal_from_degminsec_wco(lon_degminsec, lat_degminsec):
    """
    Converts from lon lat in degrees minutes and seconds to decimal degrees for the Western Channel Observatory
    format (DDD.MMSSS) rather than actual degrees, minutes seconds.

    Parameters
    ----------
    lon_degminsec : Mx3 np.ndarray
        Array of longitude degrees, minutes and seconds in 3 separate columns (for M positions). East and West is
        determined by the sign of the leading non-zero number (e.g. [-4, 20, 16] or [0, -3, 10])
    lat_degminsec : Mx3 np.ndarray
        Array of latitude degrees, minutes and seconds in 3 seperate columns (for M positions). North and South are
        determined by the sign of the leading number.

    Returns
    -------
    lon : np.ndarray
        Array of converted decimal longitudes.
    lat : np.ndarray
        Array of converted decimal latitudes.

    """
    right_dims = (np.ndim(lon_degminsec) == 2 and np.ndim(lat_degminsec) == 2)
    right_columns = (np.shape(lon_degminsec)[1] == 3 and np.shape(lat_degminsec)[1] == 3)
    if not right_dims or not right_columns:
        raise ValueError('Inputs are the wrong shape: incorrect dimensions (2) or number of columns (3).')

    lon_sign = np.sign(lon_degminsec)
    lat_sign = np.sign(lat_degminsec)
    # Zeros have a sign of 0, but for our arithmetic, we need only -1 and 1, so replace 0 with 1 in the sign arrays.
    lon_sign[lon_sign == 0] = 1
    lat_sign[lat_sign == 0] = 1
    # Since we've got only -1 and 1 in the arrays now, we can just find the minimum along each row which will tell us
    # if we've got negative anything for a given set of coordinates. Since we're doing all the arithmetic on positive
    # only numbers, we can then just flip the sign for the correct hemisphere at the end.
    lon_sign = np.min(lon_sign, axis=1)
    lat_sign = np.min(lat_sign, axis=1)

    lon_adj = np.abs(lon_degminsec)
    lat_adj = np.abs(lat_degminsec)

    lon = lon_adj[:, 0] + lon_adj[:, 1] / 60 + lon_adj[:, 2] / 6000
    lat = lat_adj[:, 0] + lat_adj[:, 1] / 60 + lat_adj[:, 2] / 6000

    lon = lon_sign * lon
    lat = lat_sign * lat

    return lon, lat


def check_valid_zone(zone_number, zone_letter=None):
    """ Check zone is valid

    Implementation copied from the utm package. See https://github.com/Turbo87/utm.

    Parameters
    ----------
    zone_number : int
        The UTM zone number.

    zone_letter : str, optional
        The UTM zone letter. If provided, this should be a single character in length
        and between C and X.
    """
    if not 1 <= zone_number <= 60:
        raise PyLagOutOfBoundsError('Zone number out of range (must be between 1 and 60)')

    if zone_letter is not None:
        zone_letter = zone_letter.upper()

        if not 'C' <= zone_letter <= 'X' or zone_letter in ['I', 'O']:
            raise PyLagOutOfBoundsError('Zone letter out of range (must be between C and X)')


if __name__ == '__main__':

    print('\nTest with NumPy single values')
    latTest, lonTest = 50, -5
    z, e, n, outLat, outLong = __test(latTest, lonTest)
    print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
        latTest, lonTest, outLat, outLong))
    print("Intermediate UTM: {}, {}".format(e, n))

    print('\nTest with lists')
    latTest, lonTest = [50, 55], [-5, -20]
    z, e, n, outLat, outLong = __test(latTest, lonTest)
    for c in range(len(latTest)):
        print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
            latTest[c], lonTest[c], outLat[c], outLong[c]))
        print("Intermediate UTM: {}, {}".format(e[c], n[c]))

    print('\nTest with NumPy arrays')
    latTest, lonTest = np.asarray([50, 55]), np.asarray([-5, -20])
    z, e, n, outLat, outLong = __test(latTest, lonTest)
    for c in range(len(latTest)):
        print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
            latTest[c], lonTest[c], outLat[c], outLong[c]))
        print("Intermediate UTM: {}, {}".format(e[c], n[c]))

    print('\nTest with NumPy arrays but a single zone')
    latTest, lonTest = np.asarray([50, 55]), np.asarray([-5, -20])
    z, e, n, outLat, outLong = __test(latTest, lonTest, inZone=30)
    for c in range(len(latTest)):
        print("Input (lat/long): {}, {}\nOutput (lat/long): {} {}".format(
            latTest[c], lonTest[c], outLat[c], outLong[c]))
        print("Intermediate UTM: {}, {}".format(e[c], n[c]))
