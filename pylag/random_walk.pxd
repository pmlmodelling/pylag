from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle cimport Particle
from pylag.data_reader cimport DataReader
from pylag.delta cimport Delta
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator

# Random walk base class
cdef class RandomWalk:
    cdef DTYPE_FLOAT_t _time_step

    cdef random_walk(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)


# Vertical random walks
cdef class VerticalRandomWalk(RandomWalk):
    cdef VertBoundaryConditionCalculator _vert_bc_calculator

    cdef random_walk(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)

# Horizontal random walks
cdef class HorizontalRandomWalk(RandomWalk):

    cdef random_walk(self, DTYPE_FLOAT_t time, Particle *particle, DataReader data_reader, Delta *delta_X)
