from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from particle cimport Particle
from pylag.data_reader cimport DataReader
from pylag.delta cimport Delta
from pylag.boundary_conditions cimport VertBoundaryConditionCalculator

# Base class for NumMethod objects
cdef class NumMethod:

    cdef step(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, Delta *delta_X):
        pass


# Advection only numerical method
cdef class AdvNumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

    cdef step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle, Delta *delta_X):
        pass


# Diffusion only numerical method
cdef class DiffNumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

    cdef step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle, Delta *delta_X):
        pass

# Advection-diffusion NumMethod that combines the two without using operator splitting
cdef class AdvDiffNumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

    cdef step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle, Delta *delta_X):
        pass


# Advection-diffusion NumMethod using operator splitting - first the advection 
# step is computed, then the diffusion step.
cdef class OS0NumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

    cdef step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle, Delta *delta_X):
        pass


# Advection-diffusion NumMethod using Strang Splitting
cdef class OS1NumMethod(NumMethod):
    cdef DTYPE_FLOAT_t _time_step

    def __init__(self, config):
        self._time_step = config.getfloat('NUMERICS', 'time_step')

    cdef step(self, DataReader data_reader, DTYPE_FLOAT_t time, 
            Particle *particle, Delta *delta_X):
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