""" Library of standard PyLag variables

This is a hardcoded library of standard variable names and units
which is intended to help with the consistent writing of data to
file.

It includes maps for different types of input data, which map
PyLag names (e.g. "temperature") to those used in different types
of input data (e.g. "temp" in FVCOM).
"""
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT


def get_data_type(var_name):
    return _variable_data_types[var_name]


def get_units(var_name):
    return _variable_units[var_name]


def get_long_name(var_name):
    return _variable_long_names[var_name]


def get_coordinate_variable_name(coordinate_system, variable_name):
    if coordinate_system == "cartesian":
        return cartesian_coordinate_variable_names[variable_name]
    elif coordinate_system == "spherical":
        return spherical_coordinate_variable_names[variable_name]
    else:
        raise ValueError("Unsupported model coordinate system `{}'".format(coordinate_system))


# Dictionaries holding variable attributes
_variable_data_types = {}
_variable_units = {}
_variable_long_names = {}


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


# Spherical spatial coordinates
# -----------------------------
spherical_coordinate_variable_names = {'x1': 'longitude',
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


# Environmental variables
# -----------------------

# thetao
_variable_data_types['thetao'] = DTYPE_FLOAT
_variable_units['thetao'] = 'degC'
_variable_long_names['thetao'] = 'Sea Water Potential Temperature'

# so
_variable_data_types['so'] = DTYPE_FLOAT
_variable_units['so'] = 'psu'
_variable_long_names['so'] = 'Sea Water Salinity'

# rsdo
_variable_data_types['rsdo'] = DTYPE_FLOAT
_variable_units['rsdo'] = 'W m-2'
_variable_long_names['rsdo'] = 'Downwelling Shortwave Radiation in Sea Water '

# FVCOM name mappings
fvcom_variable_names = {'thetao': 'temp', 'so': 'salinity'}

# GOTM name mappings
gotm_variable_names = {'thetao': 'temp', 'so': 'salt', 'rsdo': 'rad'}

