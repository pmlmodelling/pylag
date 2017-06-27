include "constants.pxi"

from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle cimport Particle
from pylag.data_reader cimport DataReader
from pylag.delta cimport Delta
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator

# Objects of type NumMethod
# -------------------------
#
# Different types of NumMethod object encode different approaches to combining
# the effects of advection and diffusion. Types that represent just the
# contribution of advection or just the contribution of diffusion are also
# included.

# Base class for NumMethod objects
cdef class NumMethod:

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X) except INT_ERR:
        pass


# Advection only numerical method
cdef class AdvNumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle, Delta *delta_X) except INT_ERR:
        pass


# Diffusion only numerical method
cdef class DiffNumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle, Delta *delta_X) except INT_ERR:
        pass

# Advection-diffusion numerical method that combines the two without using
# operator splitting
cdef class AdvDiffNumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle, Delta *delta_X) except INT_ERR:
        pass


# Advection-diffusion numerical method that uses a form of operator splitting - 
# first the advection  step is computed, then the diffusion step.
cdef class OS0NumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle, Delta *delta_X) except INT_ERR:
        pass


# Advection-diffusion numerical method that uses a form or operator splitting 
# known using Strang Splitting
cdef class OS1NumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

    cdef DTYPE_INT_t step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle, Delta *delta_X) except INT_ERR:
        pass

def get_num_method(config):
    if not config.has_option("NUMERICS", "num_method"):
        raise ValueError("Failed to find the option `num_method' in the "\
                "supplied configuration file. This option is deemed mandatory "\
                "since without it particle positions would not be updated.")
    
    # Return the specified numerical integrator.
    if config.get("NUMERICS", "num_method") == "adv_only":
        return AdvNumMethod(config)
    elif config.get("NUMERICS", "num_method") == "diff_only":
        return DiffNumMethod(config)
    elif config.get("NUMERICS", "num_method") == "adv_diff":
        return AdvDiffNumMethod(config)
    elif config.get("NUMERICS", "num_method") == "operator_split_0":
        return OS0NumMethod(config)
    elif config.get("NUMERICS", "num_method") == "operator_split_1":
        return OS1NumMethod(config)
    else:
        raise ValueError('Unsupported numerical method specified.')
    

# Objects of type DetItMethod
# ---------------------------

# Base class for DetItMethod objects
cdef class DetItMethod:
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class RK4_2D_DetItMethod(DetItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class RK4_3D_DetItMethod(DetItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

def get_deterministic_iterative_method(config):
    if not config.has_option("NUMERICS", "det_iterative_method"):
        raise ValueError("Failed to find the option `det_iterative_method' in the "\
                "supplied configuration file.")
    
    # Return the specified numerical integrator.
    if config.get("NUMERICS", "det_iterative_method") == "RK4_2D":
        return RK4_2D_DetItMethod(config)
    elif config.get("NUMERICS", "det_iterative_method") == "RK4_3D":
        return RK4_3D_DetItMethod(config)
    else:
        raise ValueError('Unsupported deterministic iterative method.')

# Objects of type StocItMethod
# ---------------------------
                
# Base class for StocItMethod objects
cdef class StocItMethod:
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class Naive_1D_StocItMethod(StocItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class Euler_1D_StocItMethod(StocItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class Visser_1D_StocItMethod(StocItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class Milstein_1D_StocItMethod(StocItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class Milstein_2D_StocItMethod(StocItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class Milstein_3D_StocItMethod(StocItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

def get_stochastic_iterative_method(config):
    if not config.has_option("NUMERICS", "stoc_iterative_method"):
        raise ValueError("Failed to find the option `stoc_iterative_method' in the "\
                "supplied configuration file.")
    
    # Return the specified numerical integrator.
    if config.get("NUMERICS", "stoc_iterative_method") == "naive_1D":
        return Naive_1D_StocItMethod(config)
    elif config.get("NUMERICS", "stoc_iterative_method") == "euler_1D":
        return Euler_1D_StocItMethod(config)
    elif config.get("NUMERICS", "stoc_iterative_method") == "visser_1D":
        return Visser_1D_StocItMethod(config)
    elif config.get("NUMERICS", "stoc_iterative_method") == "milstein_1D":
        return Milstein_1D_StocItMethod(config)
    elif config.get("NUMERICS", "stoc_iterative_method") == "milstein_2D":
        return Milstein_2D_StocItMethod(config)
    elif config.get("NUMERICS", "stoc_iterative_method") == "milstein_3D":
        return Milstein_3D_StocItMethod(config)
    else:
        raise ValueError('Unsupported stochastic iterative method.')

# Objects of type DetStocItMethod
# ---------------------------

# Base class for DetStocItMethod objects
cdef class DetStocItMethod:
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

cdef class Milstein_3D_DetStocItMethod(DetStocItMethod):
    cdef DTYPE_INT_t step(self, DTYPE_FLOAT_t time, Particle *particle,
            DataReader data_reader, Delta *delta_X) except INT_ERR:
        pass

def get_deterministic_stochastic_iterative_method(config):
    if not config.has_option("NUMERICS", "det_stoc_iterative_method"):
        raise ValueError("Failed to find the option `det_stoc_iterative_method' in the "\
                "supplied configuration file.")
    
    # Return the specified iterative method
    if config.get("NUMERICS", "det_stoc_iterative_method") == "milstein_3D":
        return Milstein_3D_DetStocItMethod(config)
    else:
        raise ValueError('Unsupported deterministic-stochastic iterative method.')