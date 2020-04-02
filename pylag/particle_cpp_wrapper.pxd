# Data types
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from particle cimport Particle


cdef class ParticleSmartPtr:
    cdef Particle* _particle

    cdef Particle* get_ptr(self)

cdef ParticleSmartPtr copy(ParticleSmartPtr)

cdef to_string(Particle* particle)
