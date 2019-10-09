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


# Dictionaries holding variable attributes
_variable_data_types = {}
_variable_units = {}
_variable_long_names = {}


# Particle group ID
_variable_data_types['group_id'] = DTYPE_INT
_variable_units['group_id'] = 'NA'
_variable_long_names['group_id'] = 'Particle group ID'

# Particle x-position
_variable_data_types['xpos'] = DTYPE_FLOAT
_variable_units['xpos'] = 'Meters (m)'
_variable_long_names['xpos'] = 'Particle x-position'

# Particle y-position
_variable_data_types['ypos'] = DTYPE_FLOAT
_variable_units['ypos'] = 'Meters (m)'
_variable_long_names['ypos'] = 'Particle y-position'

# Particle z-position
_variable_data_types['zpos'] = DTYPE_FLOAT
_variable_units['zpos'] = 'Meters (m)'
_variable_long_names['zpos'] = 'Particle z-position'

# Tracer names
# ------------
#
# Mapping provided of the form PyLag_standard_name -> FVCOM_standard_name.

# FVCOM
fvcom_variable_names = {'thetao': 'temp', 'so': 'salinity'}
