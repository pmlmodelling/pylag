"""
Module in which mathematical or physical constants are defined
"""
import numpy as np

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# Earth's radius in m
cpdef DTYPE_FLOAT_t earth_radius = 6371000.

# Unit conversions
cpdef DTYPE_FLOAT_t deg_to_radians = np.radians(1)

# Pi
cpdef DTYPE_FLOAT_t pi = np.pi
