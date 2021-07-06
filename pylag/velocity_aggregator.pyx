"""
A velocity aggregator pulls together and combines different
terms that together yield a particle's velocity. These include:

1) The effect of ocean currents
2) The effect of Stokes Drift (under development)
3) The effect of direct wind forcing, i.e. sail effects (under development)
4) The effect of buoyancy forcing, i.e. settling or rising (under development)
5) The effect of movement (under development)

Only a subest of the above effects may be used at any one time, with the
choice determined using the run configuration file.

Note
----
velocity_aggregator is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

include "constants.pxi"

import numpy as np

from pylag.settling import parameter_names as settling_parameter_names

from pylag.particle_cpp_wrapper cimport ParticleSmartPtr


# Velocity aggregator
cdef class VelocityAggregator:

    def __init__(self, config):
        # TODO set values for these from the config
        self._apply_ocean_velocity_term = True

        # Support for these has not been included yet
        self._apply_stokes_drift_term = False
        self._apply_sail_effect_term = False
        self._apply_buoyancy_term = False
        self._apply_behaviour_term = False

        # Parameter names
        self._settling_velocity_parameter_name = settling_parameter_names['settling_velocity']

    cdef void get_velocity(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t velocity[3]) except +:
        """ Get the velocity
        """
        cdef DTYPE_FLOAT_t velocity_tmp[3]

        cdef DTYPE_INT_t i

        if self._apply_ocean_velocity_term == True:
            data_reader.get_velocity(time, particle, velocity_tmp)
            for i in range(3):
                velocity[i] = velocity_tmp[i]

        if self._apply_buoyancy_term == True:
            velocity[2] += particle.get_bio_parameter(self._settling_velocity_parameter_name)

        return

