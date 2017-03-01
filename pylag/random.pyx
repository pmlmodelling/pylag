"""
This Cython module has the purpose of providing clients with rapid access to 
pseudo random numbers generated the Mersenne Twister pseudo RNG.
"""
import os
import time

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cimport crandom

cdef class Pymt19937:
    """Cython wrapper class for mt19937

    """
    cdef crandom.mt19937 c_mt19937
    def __cinit__(self, DTYPE_INT_t seed):
        self.c_mt19937 = crandom.mt19937(seed)
    
    cdef crandom.mt19937 get(self):
        return self.c_mt19937

# PRNG generator
cdef Pymt19937 generator

# Seed for the PRNG
_seed = None

def get_seed():
    """Return the value of the seed used with the PRNG.

    """
    return _seed

def seed(seed=None):
    """
    Seed the random number generator. If seed is None, use a combination of the
    system time and processor ID to set the random seed. The approach ensures
    each worker uses a unique seed during parallel simulations. Algorithm
    adapted from http://goo.gl/BVxgFl.
    
    Parameters:
    -----------
    seed: long, optional
        The seed to be used.
    """
    global _seed, generator
    
    if seed is None:
        # Initialise the PRNG. Use the pid to ensure each worker uses a unique seed
        pid = os.getpid()
        s = time.time() * 256
        _seed = long(abs(((s*181)*((pid-83)*359))%104729))
    else:
        _seed = long(seed)

    # Set the seed for the RNG
    generator = Pymt19937(_seed)
        
cpdef DTYPE_FLOAT_t gauss(DTYPE_FLOAT_t mean = 0.0, DTYPE_FLOAT_t std = 1.0):
    """
    Generate a random Gaussian variate. The Gaussian distribution has a standard
    deviation of std, and a mean of 0.0.
    
    Parameters:
    -----------
    std: float, optional
        Standard deviation of the Gaussian distribution.
        
    Returns:
    --------
    variate: float
        Random Gaussian variate
    """
    global generator

    cdef crandom.normal_distribution[DTYPE_FLOAT_t] dist = crandom.normal_distribution[DTYPE_FLOAT_t](mean, std)
    return dist(generator.c_mt19937)

cpdef DTYPE_FLOAT_t uniform(DTYPE_FLOAT_t a = -1.0, DTYPE_FLOAT_t b = 1.0):
    """
    Generate a random variate within the range [a, b].
    
    Parameters:
    -----------
    a: float, optional
        Lower limit
    b: float, optional
        Upper limit
        
    Returns:
    --------
    variate: float
        Random variate
    """
    global generator

    cdef crandom.uniform_real_distribution[DTYPE_FLOAT_t] dist = crandom.uniform_real_distribution[DTYPE_FLOAT_t](a, b)
    return dist(generator.c_mt19937)
