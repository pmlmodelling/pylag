include "constants.pxi"

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT

# PyLag cimports
from pylag.particle cimport Particle
from pylag.data_reader cimport DataReader


# Base class for Stoke's Drift velocity calculators
cdef class StokesDriftCalculator:

    cdef void get_velocity(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t stokes_drift_velocity[2]) except *