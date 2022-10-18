"""
StokesDriftCalculators are used to determine the contribution of Stoke's Drift
to a particle's velocity.

Note
----
stokes_drift is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

include "constants.pxi"

import numpy as np

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.exceptions import PyLagValueError

from pylag.particle_cpp_wrapper cimport ParticleSmartPtr


cdef class StokesDriftCalculator:
    """ Abstract base class for PyLag Stoke's Drift calculators
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
        stokes_drift_velocity : NumPy array
            Two component array giving the velocity due to Stoke's drift.
        """
        cdef DTYPE_FLOAT_t stokes_drift_velocity_c[2]
        cdef DTYPE_INT_t i

        self.get_velocity(data_reader, time, particle.get_ptr(),
                stokes_drift_velocity_c)

        stokes_drift_velocity = np.empty(2, dtype=DTYPE_FLOAT)
        for i in range(2):
                stokes_drift_velocity[i] = stokes_drift_velocity_c[i]

        return stokes_drift_velocity

    cdef void get_velocity(self, DataReader data_reader,
            DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t stokes_drift[2]) except *:
        """

        Parameters
        ----------
        data_reader : DataReader
            DataReader object used for calculating point velocities.

        time : float
            The current time.

        particle : C pointer
            C pointer to a Particle struct

        stokes_drift : C array
            C array of length two.
        """
        raise NotImplementedError


cdef class SurfaceStokesDriftCalculator(StokesDriftCalculator):
    """ Surface Stoke's drift is as given in the input file

    Parameters
    ----------
    config : ConfigParser
        Configuration object.

    Attributes
    ----------
    _config : ConfigParser
        Configuration object.
    """
    cdef object _config

    def __init__(self, config):
        self._config = config

    cdef void get_velocity(self, DataReader data_reader,
            DTYPE_FLOAT_t time, Particle *particle,
            DTYPE_FLOAT_t stokes_drift[2]) except *:
        """ Return surface Stoke's drift velocity

        Parameters
        ----------
        data_reader : DataReader
            DataReader object used for calculating point velocities.

        time : float
            The current time.

        particle : C pointer
            C pointer to a Particle struct

        stokes_drift : C array
            The Stoke's drift velocity.
        """
        data_reader.get_surface_stokes_drift_velocity(time, particle,
                stokes_drift)

        return

def get_stokes_drift_calculator(config):
    """ Factory method for Stoke's Drift calculators

    Parameters
    ----------
    config : ConfigParser
        PyLag configuraton object

    Returns
    -------
     : StokesDriftCalculator
         A Stoke's Drift calculator
    """
    try:
        stokes_drift = config.get("STOKES_DRIFT",
                                  "stokes_drift_calculator").strip().lower()
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        return None
    else:
        if stokes_drift == "surface":
            return SurfaceStokesDriftCalculator(config)
        elif stokes_drift == "none":
            return None
        else:
            raise PyLagValueError("Unsupported Stoke's Drift calculator.")