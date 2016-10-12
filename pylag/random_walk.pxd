from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle cimport Particle
from data_reader import DataReader
from data_reader cimport DataReader
from delta cimport Delta

# Random walk base class
cdef class RandomWalk:
    cdef DTYPE_FLOAT_t _time_step

    cdef random_walk(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)


# Vertical random walks
cdef class VerticalRandomWalk(RandomWalk):
    # Grid boundary limits
    cdef DTYPE_FLOAT_t _zmin
    cdef DTYPE_FLOAT_t _zmax

    cdef random_walk(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)

cdef class NaiveVerticalRandomWalk(VerticalRandomWalk):
    cdef random_walk(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)

cdef class AR0VerticalRandomWalk(VerticalRandomWalk):
    cdef random_walk(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)

cdef class AR0VerticalRandomWalkWithSpline(VerticalRandomWalk):
    cdef random_walk(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)


# Horizontal random walks
cdef class HorizontalRandomWalk(RandomWalk):
    cdef random_walk(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)

cdef class ConstantHorizontalRandomWalk(HorizontalRandomWalk):
    cdef DTYPE_FLOAT_t _kh

    cdef random_walk(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)
    
cdef class NaiveHorizontalRandomWalk(HorizontalRandomWalk):
    cdef random_walk(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)

cdef class AR0HorizontalRandomWalk(HorizontalRandomWalk):
    cdef random_walk(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)
