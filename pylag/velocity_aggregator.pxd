include "constants.pxi"

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from pylag.data_reader cimport DataReader
from pylag.particle cimport Particle

# Velocity aggregator
cdef class VelocityAggregator:

    cdef bint _apply_ocean_velocity_term

    cdef bint _apply_stokes_drift_term

    cdef bint _apply_sail_effect_term

    cdef bint _apply_buoyancy_term

    cdef bint _apply_behaviour_term

    cdef void get_velocity(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t velocity[3]) except +

