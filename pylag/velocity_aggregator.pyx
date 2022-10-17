"""
A velocity aggregator pulls together and combines different
terms that together yield a particle's velocity. These include:

1) The effect of ocean currents
2) The effect of Stokes Drift
3) The effect of direct wind forcing, i.e. sail effects
4) The effect of buoyancy, i.e. settling or rising
5) The effect of movement (under development)

Any combination of the above effects may be used at any one time, with the
choice determined using the run configuration file.

Note
----
velocity_aggregator is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

include "constants.pxi"

import numpy as np

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.windage import get_windage_calculator
from pylag.stokes_drift import get_stokes_drift_calculator
from pylag.settling import variable_names as settling_variable_names

from pylag.particle_cpp_wrapper cimport ParticleSmartPtr


# Velocity aggregator
cdef class VelocityAggregator:

    def __init__(self, config):
        # Ocean transport
        try:
            ocean_product_name = config.get("OCEAN_CIRCULATION_MODEL", "name").strip().lower()
        except (configparser.NoSectionError, configparser.NoOptionError):
            self._apply_ocean_velocity_term = False
        else:
            if ocean_product_name == "none":
                self._apply_ocean_velocity_term = False
            else:
                self._apply_ocean_velocity_term = True

        # Windage
        self._windage_calculator = \
                get_windage_calculator(config)
        if self._windage_calculator is not None:
            self._apply_windage_term = True
        else:
            self._apply_windage_term = False

        # Stoke's drift
        self._stokes_drift_calculator = \
                get_stokes_drift_calculator(config)
        if self._stokes_drift_calculator is not None:
            self._apply_stokes_drift_term = True
        else:
            self._apply_stokes_drift_term = False

        # Settling
        try:
            settling = config.get("SETTLING",
                    "settling_velocity_calculator").strip().lower()
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            self._apply_settling_term = False
        else:
            if settling == "none":
                self._apply_settling_term = False
            else:
                self._apply_settling_term = True

        # Parameter names
        self._settling_velocity_variable_name = \
                settling_variable_names['settling_velocity']

        # TODO Support for this has not been included yet
        self._apply_behaviour_term = False

    cdef void get_velocity(self, DataReader data_reader, DTYPE_FLOAT_t time,
            Particle *particle, DTYPE_FLOAT_t velocity[3]) except +:
        """ Get the velocity
        """
        cdef DTYPE_FLOAT_t ocean_velocity[3]
        cdef DTYPE_FLOAT_t windage_velocity[2]
        cdef DTYPE_FLOAT_t stokes_drift_velocity[2]

        cdef DTYPE_INT_t i

        # Ensure we are working with a zeroed array
        for i in range(3):
            velocity[i] = 0.0

        # Ocean transport (three component)
        if self._apply_ocean_velocity_term == True:
            data_reader.get_velocity(time, particle, ocean_velocity)
            for i in range(3):
                velocity[i] += ocean_velocity[i]

        # Windage (two component)
        if self._apply_windage_term == True:
            self._windage_calculator.get_velocity(data_reader,
                    time, particle, windage_velocity)
            for i in range(2):
                velocity[i] += windage_velocity[i]

        # Stoke's drift (Two component)
        if self._apply_stokes_drift_term == True:
            self._stokes_drift_calculator.get_velocity(data_reader,
                    time, particle, stokes_drift_velocity)
            for i in range(2):
                velocity[i] += stokes_drift_velocity[i]

        # Settling
        if self._apply_settling_term == True:
            # Settling velocity is positive down
            velocity[2] -= particle.get_diagnostic_variable(
                    self._settling_velocity_variable_name)

        return
