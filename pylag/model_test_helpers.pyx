include "constants.pxi"

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

from pylag.data_reader cimport DataReader
from pylag.particle cimport Particle

cdef class TestOPTModelDataReader(DataReader):
    """ Data reader for testing OPTModel objects

    The test data reader defines a grid made up of a unit
    cube with one corner positioned at the origin, and
    three sides that run along the positive coordinate axes.

    Points within the cube lie within the domain, while anything
    outside of the cube is outside of the domain.
    """
    cdef DTYPE_FLOAT_t _xmin, _ymin, _zmin
    cdef DTYPE_FLOAT_t _xmax, _ymax, _zmax
    
    def __init__(self):
        self._xmin = 0.0
        self._ymin = 0.0
        self._zmin = 0.0
        self._xmax = 1.0
        self._ymax = 1.0
        self._zmax = 1.0

    cdef DTYPE_INT_t find_host(self, Particle *particle_old,
                               Particle *particle_new) except INT_ERR:
        if particle_new.get_x1() < self._xmin or particle_new.get_x1() > self._xmax:
            return BDY_ERROR
        elif particle_new.get_x2() < self._ymin or particle_new.get_x2() > self._ymax:
            return  BDY_ERROR
        else:
            return IN_DOMAIN

    cdef DTYPE_INT_t find_host_using_global_search(self,
                                                   Particle *particle) except INT_ERR:
        if particle.get_x1() < self._xmin or particle.get_x1() > self._xmax:
            return BDY_ERROR
        elif particle.get_x2() < self._ymin or particle.get_x2() > self._ymax:
            return  BDY_ERROR
        else:
            return IN_DOMAIN

    cdef DTYPE_INT_t find_host_using_local_search(self,
                                                  Particle *particle_old) except INT_ERR:
        if particle_old.get_x1() < self._xmin or particle_old.get_x1() > self._xmax:
            return BDY_ERROR
        elif particle_old.get_x2() < self._ymin or particle_old.get_x2() > self._ymax:
            return  BDY_ERROR
        else:
            return IN_DOMAIN

    cdef set_local_coordinates(self, Particle *particle):
        pass

    cdef DTYPE_INT_t set_vertical_grid_vars(self, DTYPE_FLOAT_t time,
                                            Particle *particle) except INT_ERR:
        pass

    cpdef DTYPE_FLOAT_t get_xmin(self) except FLOAT_ERR:
        return self._xmin

    cpdef DTYPE_FLOAT_t get_ymin(self) except FLOAT_ERR:
        return self._ymin

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        return self._zmin

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle) except FLOAT_ERR:
        return self._zmax

    cdef DTYPE_INT_t is_wet(self, DTYPE_FLOAT_t time, Particle *particle) except INT_ERR:
        return 1
