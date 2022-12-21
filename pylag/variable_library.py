"""
Library of standard PyLag variables.

This is a hardcoded library of standard variable names and units
which is intended to help with the consistent writing of data to
file.

It includes maps for different types of input data, which map
PyLag names (e.g. "temperature") to those used in different types
of input data (e.g. "temp" in FVCOM).
"""
import numpy as np

from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_python import INT_INVALID, FLOAT_INVALID

from pylag.exceptions import PyLagValueError


def get_data_type(var_name, precision='s'):
    """ Get variable data type

    Parameters
    ----------
    var_name : str
        The variable name

    precision : str
        Pass in `s` for single or `d` for double precision. Optional,
        default: `s`.

    Returns
    -------
        : Python data type
        The variable data type
    """
    data_type_str = _variable_data_types[var_name]

    if data_type_str not in ['INT', 'REAL']:
        raise PyLagValueError(f'Unsupported data type `{data_type_str} for '
                              f'variable {var_name}.')

    if precision == 's':
        if data_type_str == 'INT':
            return np.int32
        return np.float32
    elif precision == 'd':
        if data_type_str == 'INT':
            return np.int64
        return np.float64
    else:
        raise PyLagValueError(f'Unknown precision flag `{precision}')


def get_integer_type(precision="s"):
    """ Return intger data type of the specified precision

    Parameters
    ----------
    precision : str
        Pass in `s` for single or `d` for double precision. Optional,
        default: `s`.

    Returns
    -------
        : Python data type
        The variable data type
    """
    if precision == "s":
        return np.int32
    elif precision == "d":
        return np.int64
    else:
        raise PyLagValueError(f'Unknown precision flag `{precision}')


def get_real_type(precision="s"):
    """ Return real data type of the specified precision

    Parameters
    ----------
    precision : str
        Pass in `s` for single or `d` for double precision. Optional,
        default: `s`.

    Returns
    -------
        : Python data type
        The variable data type
    """
    if precision == "s":
        return np.int32
    elif precision == "d":
        return np.int64
    else:
        raise PyLagValueError(f'Unknown precision flag `{precision}')


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


def get_invalid_value(dtype):
    """ Get value used for invalid entries

    Parameters
    ----------
    dtype : np.dtype
        The data type.

    Returns
    -------
     : int, float
         Value of invalid values

    """
    if np.issubdtype(dtype, np.integer):
        return INT_INVALID

    return FLOAT_INVALID


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


# Particle group ID
_variable_data_types['group_id'] = 'INT'
_variable_units['group_id'] = 'NA'
_variable_long_names['group_id'] = 'Particle group ID'

# Cartesian spatial coordinates
# -----------------------------
cartesian_coordinate_variable_names = {'x1': 'x',
                                       'x2': 'y',
                                       'x3': 'z'}

# Particle x-position
_variable_data_types['x'] = 'REAL'
_variable_units['x'] = 'm'
_variable_long_names['x'] = 'x'

# Particle y-position
_variable_data_types['y'] = 'REAL'
_variable_units['y'] = 'm'
_variable_long_names['y'] = 'y'

# Particle z-position
_variable_data_types['z'] = 'REAL'
_variable_units['z'] = 'm'
_variable_long_names['z'] = 'z'


# Geographic spatial coordinates
# -----------------------------
geographic_coordinate_variable_names = {'x1': 'longitude',
                                        'x2': 'latitude',
                                        'x3': 'depth'}

# Particle longitude
_variable_data_types['longitude'] = 'REAL'
_variable_units['longitude'] = 'degrees_east'
_variable_long_names['longitude'] = 'longitude'

# Particle latitude
_variable_data_types['latitude'] = 'REAL'
_variable_units['latitude'] = 'degrees_north'
_variable_long_names['latitude'] = 'latitude'

# Particle z-position
_variable_data_types['depth'] = 'REAL'
_variable_units['depth'] = 'm'
_variable_long_names['depth'] = 'depth'


# Extra grid variables
# --------------------

# h
_variable_data_types['h'] = 'REAL'
_variable_units['h'] = 'meters (m)'
_variable_long_names['h'] = 'Water depth'

# zeta
_variable_data_types['zeta'] = 'REAL'
_variable_units['zeta'] = 'meters (m)'
_variable_long_names['zeta'] = 'Sea surface elevation'

# Number of land boundary encounters
_variable_data_types['land_boundary_encounters'] = 'INT'
_variable_units['land_boundary_encounters'] = 'None'
_variable_long_names['land_boundary_encounters'] = \
        'Number of land boundary encounters'

# Status variables
# ---------------
_variable_data_types['error_status'] = 'INT'
_variable_units['error_status'] = 'None'
_variable_long_names['error_status'] = 'Status flag (1 - error state; 0 - ok)'

_variable_data_types['in_domain'] = 'INT'
_variable_units['in_domain'] = 'None'
_variable_long_names['in_domain'] = 'In domain flag (1 - yes; 0 - no)'

_variable_data_types['is_beached'] = 'INT'
_variable_units['is_beached'] = 'None'
_variable_long_names['is_beached'] = 'Is beached (1 - yes; 0 - no)'

# Bio model variables
# -------------------

# Particle age
_variable_data_types['age'] = 'REAL'
_variable_units['age'] = 'days (d)'
_variable_long_names['age'] = 'Particle age'

# Alive/dead status
_variable_data_types['is_alive'] = 'INT'
_variable_units['is_alive'] = 'None'
_variable_long_names['is_alive'] = 'Is alive flag (1 - yes; 0 - no)'


# Environmental variables
# -----------------------

# thetao
_variable_data_types['thetao'] = 'REAL'
_variable_units['thetao'] = 'degC'
_variable_long_names['thetao'] = 'Sea Water Potential Temperature'

# so
_variable_data_types['so'] = 'REAL'
_variable_units['so'] = 'psu'
_variable_long_names['so'] = 'Sea Water Salinity'

# rsdo
_variable_data_types['rsdo'] = 'REAL'
_variable_units['rsdo'] = 'W m-2'
_variable_long_names['rsdo'] = 'Downwelling Shortwave Radiation in Sea Water '

# Standard name mappings
standard_variable_names = {'thetao': 'thetao', 'so': 'so'}

# FVCOM name mappings
fvcom_variable_names = {'thetao': 'temp', 'so': 'salinity'}

# GOTM name mappings
gotm_variable_names = {'thetao': 'temp', 'so': 'salt', 'rsdo': 'rad'}


__all__ = ['get_data_type',
           'get_integer_type',
           'get_real_type',
           'get_units',
           'get_long_name',
           'get_invalid_value',
           'get_coordinate_variable_name']
