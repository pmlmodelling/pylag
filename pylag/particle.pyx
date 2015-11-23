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

cdef class Particle(object):
    cdef DTYPE_INT_t _group_id
    cdef DTYPE_FLOAT_t _xpos
    cdef DTYPE_FLOAT_t _ypos
    cdef DTYPE_FLOAT_t _zpos
    
    cdef DTYPE_INT_t _in_domain
    
    cdef DTYPE_FLOAT_t _host_horizontal_elem
    
    cdef DTYPE_FLOAT_t _h
    
    cdef DTYPE_FLOAT_t _zeta

    def __init__(self, DTYPE_INT_t group_id, DTYPE_FLOAT_t xpos, 
            DTYPE_FLOAT_t ypos, DTYPE_FLOAT_t zpos, DTYPE_INT_t host=-999, 
            DTYPE_FLOAT_t h=-999., DTYPE_FLOAT_t zeta=-999.):
        self._group_id = group_id
        
        self._xpos = xpos
        self._ypos = ypos
        self._zpos = zpos
        
        self._in_domain = False
        
        self._host_horizontal_elem = host
        
        self._h = h
        
        self._zeta = zeta

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

    # IS the particle in the domain?
    property in_domain:
        def __get__(self):
            return self._in_domain
        def __set__(self, DTYPE_INT_t value):
            self._in_domain = value

    # Host horizontal element
    property host_horizontal_elem:
        def __get__(self):
            return self._host_horizontal_elem
        def __set__(self, DTYPE_INT_t value):
            self._host_horizontal_elem = value

    # Bathymetry at the particle's current position
    property h:
        def __get__(self):
            return self._h
        def __set__(self, DTYPE_FLOAT_t value):
            self._h = value

    # Sea surface elevation at the particle's current position
    property zeta:
        def __get__(self):
            return self._zeta
        def __set__(self, DTYPE_FLOAT_t value):
            self._zeta = value

def get_particle_seed(config=None):
    global _seed
    
    # Return a copy of the particle seed if it has already been created
    if _seed:
        return _seed
    
    # Input file containing particle initial positions
    file_name = config.get('GENERAL', 'initial_positions_file')
    with open(file_name, 'r') as f:
        lines=f.readlines()

    data=[]
    for line in lines:
        data.append(line.rstrip('\r\n').split(' '))    
        
    # The first entry is the number of particles
    n_particles = _get_entry(data.pop(0)[0], int)

    # Create seed particle set
    _seed = []
    for row in data:
        group_id = _get_entry(row[0], int)
        xpos = _get_entry(row[1], float)
        ypos = _get_entry(row[2], float)
        zpos = _get_entry(row[3], float)
        particle = Particle(group_id, xpos, ypos, zpos)
        _seed.append(particle)

    logger = logging.getLogger(__name__)
    if n_particles == len(_seed):
        logger.info('Particle seed contains {} particles.'.format(n_particles))
        return np.copy(_seed)
    else:
        logger.warning('Error reading particle initial positions from file. '\
            'The number of particles specified in the file is {}. The actual number found '\
            'while parsing the file was {}.'.format(n_particles, len(_seed)))
        raise ValueError('Error reading particle initial positions file. See log ' \
            'file for more information.')

def _get_entry(value, type):
    return type(value)

# Initial particle set with particle starting positions initalised from file
_seed = None
