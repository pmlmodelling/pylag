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
    def __init__(self):
        self._xmin = 0.0
        self._ymin = 0.0
        self._zmin = 0.0
        self._xmax = 1.0
        self._ymax = 1.0
        self._zmax = 1.0

    cpdef DTYPE_FLOAT_t get_xmin(self) except FLOAT_ERR:
        return self._xmin

    cpdef DTYPE_FLOAT_t get_ymin(self) except FLOAT_ERR:
        return self._ymin

    cdef DTYPE_FLOAT_t get_zmin(self, DTYPE_FLOAT_t time, Particle *particle):
        return self._zmin

    cdef DTYPE_FLOAT_t get_zmax(self, DTYPE_FLOAT_t time, Particle *particle):
        return self._zmax





