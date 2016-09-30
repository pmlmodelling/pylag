import numpy as np
import logging

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef class Delta:
    """ Object for storing changes in a particle's position.
    
    Objects of type Delta store and provide access to position delta values
    that are a result, for example, of advection or random displacement. The
    purpose of Delta objects is to allow position deltas to first be accumulated
    before being used to update a given particle's position.
    """
    def __init__(self, DTYPE_FLOAT_t x=0., DTYPE_FLOAT_t y=0., 
            DTYPE_FLOAT_t z=0.):
        self._delta_x = x
        self._delta_y = y
        self._delta_z = z
    
    # delta x
    property x:
        def __get__(self):
            return self._delta_x
        def __set__(self, DTYPE_FLOAT_t value):
            self._delta_x = value
            
    # delta y
    property y:
        def __get__(self):
            return self._delta_y
        def __set__(self, DTYPE_FLOAT_t value):
            self._delta_y = value
            
    # delta z
    property z:
        def __get__(self):
            return self._delta_z
        def __set__(self, DTYPE_FLOAT_t value):
            self._delta_z = value

    cpdef reset(self):
        """
        Reset stored delta values to zero.
        """
        self._delta_x = 0.0
        self._delta_y = 0.0
        self._delta_z = 0.0
