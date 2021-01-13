include "constants.pxi"

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from pylag.particle cimport Particle
from pylag.data_reader cimport DataReader
from pylag.mortality cimport MortalityCalculator


# Base class for PositionModifier objects
cdef class BioModel:
    cdef MortalityCalculator mortality_calculator

    cdef void set_initial_particle_properties(self, Particle *particle) except *

    cdef void update(self, DataReader data_reader, DTYPE_FLOAT_t time,
                     Particle *particle) except *
