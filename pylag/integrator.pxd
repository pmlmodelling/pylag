from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle import Particle
from particle cimport Particle
from data_reader import DataReader
from data_reader cimport DataReader
from delta cimport Delta
from pylag.boundary_conditions cimport HorizBoundaryConditionCalculator

cdef class NumIntegrator:
    cdef DTYPE_INT_t advect(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X)

cdef class RK4Integrator2D(NumIntegrator):
    cdef DTYPE_FLOAT_t _time_step
    cdef HorizBoundaryConditionCalculator horiz_bc_calculator

    cdef DTYPE_INT_t advect(self, DTYPE_FLOAT_t time, Particle *particle, 
            DataReader data_reader, Delta *delta_X)
    
cdef class RK4Integrator3D(NumIntegrator):
    cdef DTYPE_FLOAT_t _time_step
    cdef HorizBoundaryConditionCalculator horiz_bc_calculator

    # Grid boundary limits
    cdef DTYPE_FLOAT_t _zmin
    cdef DTYPE_FLOAT_t _zmax

    cdef DTYPE_INT_t advect(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X)
