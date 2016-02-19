from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle import Particle
from particle cimport Particle
from data_reader import DataReader
from data_reader cimport DataReader

# Random walk base class
cdef class RandomWalk:
    cdef DTYPE_FLOAT_t _time_step

    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader)


# Vertical random walks
cdef class VerticalRandomWalk(RandomWalk):
    # Grid boundary limits
    cdef DTYPE_FLOAT_t _zmin
    cdef DTYPE_FLOAT_t _zmax
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader)

cdef class NaiveVerticalRandomWalk(VerticalRandomWalk):
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader)

cdef class AR0VerticalRandomWalk(VerticalRandomWalk):
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader)

cdef class AR0VerticalRandomWalkWithSpline(VerticalRandomWalk):
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader)


# Horizontal random walks
cdef class HorizontalRandomWalk(RandomWalk):
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader)
    
cdef class NaiveHorizontalRandomWalk(HorizontalRandomWalk):
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader)

cdef class AR0HorizontalRandomWalk(HorizontalRandomWalk):
    cpdef random_walk(self, DTYPE_FLOAT_t time, Particle particle, DataReader data_reader)
