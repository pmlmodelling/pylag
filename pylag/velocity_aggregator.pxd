include "constants.pxi"

from libcpp.string cimport string

from pylag.data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

# PyLag cimports
from pylag.data_reader cimport DataReader
from pylag.particle cimport Particle
from pylag.windage cimport WindageCalculator
from pylag.stokes_drift cimport StokesDriftCalculator

# Velocity aggregator
cdef class VelocityAggregator:

    # Flags
    cdef bint _apply_ocean_velocity_term

    cdef bint _apply_stokes_drift_term

    cdef bint _apply_windage_term

    cdef bint _apply_settling_term

    cdef bint _apply_behaviour_term

    # Calculators
    cdef WindageCalculator _windage_calculator
    cdef StokesDriftCalculator _stokes_drift_calculator

    # Parameter names
    cdef string _settling_velocity_variable_name

    # Methods
    cdef void get_velocity(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t velocity[3]) except +

