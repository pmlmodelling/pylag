"""
Library of standard PyLag variables.

This is a hardcoded library of standard variable names and units
which is intended to help with the consistent writing of data to
file.

It includes maps for different types of input data, which map
PyLag names (e.g. "temperature") to those used in different types
of input data (e.g. "temp" in FVCOM).
"""
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_python import INT_INVALID, FLOAT_INVALID

from pylag.exceptions import PyLagValueError


def get_data_type(var_name):
    """ Get variable data type

    Parameters
    ----------
    var_name : str
        The variable name

    Returns
    -------
        : Python data type
        The variable data type

    """
    return _variable_data_types[var_name]


def get_units(var_name):
    """ Get the variable's units

    Parameters
    ----------
    var_name : str
        The variable name

    Returns
    -------
     : str
         It's units

    """
    return _variable_units[var_name]


def get_long_name(var_name):
    """ Get the variable long name

    Parameters
    ----------
    var_name : str
        The variable's name

    Returns
    -------
     : str
         The variable's long name

    """
    return _variable_long_names[var_name]


def get_invalid_value(var_name):
    """ Get value used for invalid entries

    Parameters
    ----------
    var_name : str
        The variable's name

    Returns
    -------
     : int, float
         Value of invalid values

    """
    return _variable_invalid_values[var_name]


def get_coordinate_variable_name(coordinate_system, variable_name):
    """ Get coordinate variable name

    Parameters
    ----------
    coordinate_system : str
        The coordinate system (i.e. `cartesian` or `geographic`)

    variable_name : str
        The variable name (i.e. `x1`, `x2` or `x3`)

    Returns
    -------
     : str
         The coordinate variable name (e.g. `longitude`)

    """
    if coordinate_system == "cartesian":
        return cartesian_coordinate_variable_names[variable_name]
    elif coordinate_system == "geographic":
        return geographic_coordinate_variable_names[variable_name]
    else:
        raise PyLagValueError(f"Unsupported model coordinate "
                              f"system `{coordinate_system}'")


# Dictionaries holding variable attributes
_variable_data_types = {}
_variable_units = {}
_variable_long_names = {}
_variable_invalid_values = {}


# Particle group ID
_variable_data_types['group_id'] = DTYPE_INT
_variable_units['group_id'] = 'NA'
_variable_long_names['group_id'] = 'Particle group ID'

# Cartesian spatial coordinates
# -----------------------------
cartesian_coordinate_variable_names = {'x1': 'x',
                                       'x2': 'y',
                                       'x3': 'z'}

# Particle x-position
_variable_data_types['x'] = DTYPE_FLOAT
_variable_units['x'] = 'm'
_variable_long_names['x'] = 'x'

# Particle y-position
_variable_data_types['y'] = DTYPE_FLOAT
_variable_units['y'] = 'm'
_variable_long_names['y'] = 'y'

# Particle z-position
_variable_data_types['z'] = DTYPE_FLOAT
_variable_units['z'] = 'm'
_variable_long_names['z'] = 'z'


# Geographic spatial coordinates
# -----------------------------
geographic_coordinate_variable_names = {'x1': 'longitude',
                                        'x2': 'latitude',
                                        'x3': 'depth'}

# Particle longitude
_variable_data_types['longitude'] = DTYPE_FLOAT
_variable_units['longitude'] = 'degrees_east'
_variable_long_names['longitude'] = 'longitude'

# Particle latitude
_variable_data_types['latitude'] = DTYPE_FLOAT
_variable_units['latitude'] = 'degrees_north'
_variable_long_names['latitude'] = 'latitude'

# Particle z-position
_variable_data_types['depth'] = DTYPE_FLOAT
_variable_units['depth'] = 'm'
_variable_long_names['depth'] = 'depth'


# Extra grid variables
# --------------------

# h
_variable_data_types['h'] = DTYPE_FLOAT
_variable_units['h'] = 'meters (m)'
_variable_long_names['h'] = 'Water depth'
_variable_invalid_values['h'] = FLOAT_INVALID

# zeta
_variable_data_types['zeta'] = DTYPE_FLOAT
_variable_units['zeta'] = 'meters (m)'
_variable_long_names['zeta'] = 'Sea surface elevation'
_variable_invalid_values['zeta'] = FLOAT_INVALID


# Status variables
# ---------------
_variable_data_types['age'] = DTYPE_FLOAT
_variable_units['age'] = 'days (d)'
_variable_long_names['age'] = 'Particle age'
_variable_invalid_values['age'] = FLOAT_INVALID


# Environmental variables
# -----------------------

# thetao
_variable_data_types['thetao'] = DTYPE_FLOAT
_variable_units['thetao'] = 'degC'
_variable_long_names['thetao'] = 'Sea Water Potential Temperature'
_variable_invalid_values['thetao'] = FLOAT_INVALID

# so
_variable_data_types['so'] = DTYPE_FLOAT
_variable_units['so'] = 'psu'
_variable_long_names['so'] = 'Sea Water Salinity'
_variable_invalid_values['so'] = FLOAT_INVALID

# rsdo
_variable_data_types['rsdo'] = DTYPE_FLOAT
_variable_units['rsdo'] = 'W m-2'
_variable_long_names['rsdo'] = 'Downwelling Shortwave Radiation in Sea Water '
_variable_invalid_values['rsdo'] = FLOAT_INVALID

# Standard name mappings
standard_variable_names = {'thetao': 'thetao', 'so': 'so'}

# FVCOM name mappings
fvcom_variable_names = {'thetao': 'temp', 'so': 'salinity'}

# GOTM name mappings
gotm_variable_names = {'thetao': 'temp', 'so': 'salt', 'rsdo': 'rad'}


__all__ = ['get_data_type',
           'get_units',
           'get_long_name',
           'get_invalid_value',
           'get_coordinate_variable_name']
