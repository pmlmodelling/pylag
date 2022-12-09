"""
WindageCalculators are used to determine the contribution of windage to a
particle's velocity.

Note
----
windage is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

include "constants.pxi"

import numpy as np

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT

from pylag.exceptions import PyLagValueError

from pylag.particle_cpp_wrapper cimport ParticleSmartPtr


cdef class WindageCalculator:
    """ Abstract base class for PyLag windage calculators
    """
    def get_velocity_wrapper(self, DataReader data_reader,
                             DTYPE_FLOAT_t time,
                             ParticleSmartPtr particle):
        """ Python friendly wrapper for get_velocity()

        The get_velocity() method must be implemented in the derived class.

        Parameters
        ----------
        data_reader : pylag.data_reader.DataReader
            A concrete PyLag data reader which inherits from the base class
            `pylag.data_reader.DataReader`.

        time : float
            The time the crossing occurred.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            A ParticleSmartPtr.

        Returns
        -------
        windage_velocity : NumPy array
            Two component array giving the velocity due to Stoke's drift.
        """
        cdef DTYPE_FLOAT_t windage_velocity_c[2]
        cdef DTYPE_INT_t i

        self.get_velocity(data_reader, time, particle.get_ptr(),
                windage_velocity_c)

        windage_velocity = np.empty(2, dtype=DTYPE_FLOAT)
        for i in range(2):
                windage_velocity[i] = windage_velocity_c[i]

        return windage_velocity

    cdef void get_velocity(self, DataReader data_reader,
            DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t windage_velocity[2]) except *:
        """ Compute windage velocity

        Parameters
        ----------
        data_reader : DataReader
            DataReader object used for calculating point velocities.

        time : float
            The current time.

        particle : C pointer
            C pointer to a Particle struct

        windage_velocity : C array
            C array of length two.
        """
        raise NotImplementedError


cdef class FixedDragWindageCalculator(WindageCalculator):
    """ Velocity computed from drag coefficient and the ten meter wind speed

    Parameters
    ----------
    config : ConfigParser
        Configuration object.

    Attributes
    ----------
    _config : ConfigParser
        Configuration object.

    _drag_coefficient : float
        Drag coefficient.
    """
    cdef object _config
    cdef DTYPE_FLOAT_t _drag_coefficient

    def __init__(self, config):
        self._config = config
        self._drag_coefficient = config.getfloat('FIXED_DRAG_WINDAGE_CALCULATOR',
                                                 'drag_coefficient')

    cdef void get_velocity(self, DataReader data_reader,
            DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t windage_velocity[2]) except *:
        """ Return the windage velocity

        Parameters
        ----------
        data_reader : DataReader
            DataReader object used for calculating point velocities.

        time : float
            The current time.

        particle : C pointer
            C pointer to a Particle struct

        windage_velocity : C array
            The windage velocity.
        """
        cdef DTYPE_FLOAT_t wind_velocity[2]
        cdef DTYPE_INT_t i

        data_reader.get_ten_meter_wind_velocity(time, particle, wind_velocity)

        for i in range(2):
            windage_velocity[i] = self._drag_coefficient * wind_velocity[i]

def get_windage_calculator(config):
    """ Factory method for windage calculators

    Parameters
    ----------
    config : ConfigParser
        PyLag configuraton object

    Returns
    -------
     : WindageCalculator
         A windage calculator
    """
    try:
        windage = config.get("WINDAGE", "windage_calculator").strip().lower()
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        return None
    else:
        if windage == "fixed_drag":
            return FixedDragWindageCalculator(config)
        elif windage == "none":
            return None
        else:
            raise PyLagValueError('Unsupported windage calculator.')
