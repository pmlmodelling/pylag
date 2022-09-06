"""
Module in which mathematical or physical constants are defined
"""
import numpy as np

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# Earth's radius in m
cdef DTYPE_FLOAT_t earth_radius

# Unit conversions
cdef DTYPE_FLOAT_t deg_to_radians
cdef DTYPE_FLOAT_t radians_to_deg

# Pi
cdef DTYPE_FLOAT_t pi

# Seconds per day
cdef DTYPE_FLOAT_t seconds_per_day

# Cordinate system flags
cdef DTYPE_INT_t cartesian
cdef DTYPE_INT_t geographic


