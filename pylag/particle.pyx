# cython: profile=True
# cython: linetrace=True

import numpy as np
import logging

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef class Particle:
    def __init__(self, DTYPE_INT_t group_id=-999, DTYPE_FLOAT_t xpos=-999., 
            DTYPE_FLOAT_t ypos=-999., DTYPE_FLOAT_t zpos=-999., DTYPE_INT_t host=-999, 
            DTYPE_INT_t in_domain=-999):
        self._group_id = group_id
        
        self._xpos = xpos
        self._ypos = ypos
        self._zpos = zpos
        
        self._host_horizontal_elem = host

        self._in_domain = in_domain

    # Group ID
    property group_id:
        def __get__(self):
            return self._group_id
        def __set__(self, DTYPE_INT_t value):
            self._group_id = value
    
    # x position
    property xpos:
        def __get__(self):
            return self._xpos
        def __set__(self, DTYPE_FLOAT_t value):
            self._xpos = value
            
    # y position
    property ypos:
        def __get__(self):
            return self._ypos
        def __set__(self, DTYPE_FLOAT_t value):
            self._ypos = value
            
    # z position
    property zpos:
        def __get__(self):
            return self._zpos
        def __set__(self, DTYPE_FLOAT_t value):
            self._zpos = value

    # Host horizontal element
    property host_horizontal_elem:
        def __get__(self):
            return self._host_horizontal_elem
        def __set__(self, DTYPE_INT_t value):
            self._host_horizontal_elem = value

    # Is the particle in the domain?
    property in_domain:
        def __get__(self):
            return self._in_domain
        def __set__(self, DTYPE_INT_t value):
            self._in_domain = value
