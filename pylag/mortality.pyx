"""
Bio model objects can be used to manage the configuration and update of
biological particles.

Note
----
bio_model is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

include "constants.pxi"

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.numerics import get_bio_time_step

from pylag.particle_cpp_wrapper cimport ParticleSmartPtr
from pylag.parameters cimport seconds_per_day
from pylag.random cimport uniform


cdef class MortalityCalculator:
    """ Abstract base class for PyLag mortality calculator
    """
    def apply_wrapper(self, DataReader data_reader,
                      DTYPE_FLOAT_t time,
                      ParticleSmartPtr particle):
        """ Python friendly wrapper for apply()

        The apply() method must be implemented in the derived class.

        Parameters
        ----------
        data_reader : pylag.data_reader.DataReader
            A concrete PyLag data reader which inherits from the base class
            `pylag.data_reader.DataReader`.

        time : float
            The time the crossing occurred.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            A ParticleSmartPtr.
        """
        return self.apply(data_reader, time, particle.get_ptr())

    cdef void apply(self, DataReader data_reader, DTYPE_FLOAT_t time,
                    Particle *particle) except *:
        """ Determine whether a particle dies and updates its status

        Parameters
        ----------
        data_reader : DataReader
            DataReader object used for calculating point velocities.

        time : float
            The current time.

        particle : C pointer
            C pointer to a Particle struct
        """
        raise RuntimeError


cdef class FixedTimeMortalityCalculator(MortalityCalculator):
    """ Kill particles once they reach a given age

    The age of particles at death should be specified in the run config file.

    Parameters
    ----------
    config : ConfigParser
        Configuration object

    Attributes
    ----------
    age_of_death : float
        The age of the particle at death in seconds.
    """
    cdef DTYPE_FLOAT_t age_of_death

    def __init__(self, config):
        # Set time step
        self.age_of_death = config.getfloat('BIO_MODEL', 'age_of_death_in_days') * seconds_per_day


    cdef void apply(self, DataReader data_reader, DTYPE_FLOAT_t time,
                    Particle *particle) except *:
        """ Determine whether a particle dies and updates its status

        Parameters
        ----------
        data_reader : DataReader
            DataReader object used for calculating point velocities.

        time : float
            The current time.

        particle : C pointer
            C pointer to a Particle struct
        """
        if particle.get_age() >= self.age_of_death:
            particle.set_is_alive(False)

        return


cdef class ProbabilisticMortalityCalculator(MortalityCalculator):
    """ Kill individuals randomly given a specified death rate

    The probability of an individual dying is:

    P_{D} = \mu_{D} \Delta t

    where \mu_{D} is the death rate and \Delta t is the model time
    step for biological processes.

    Parameters
    ----------
    config : ConfigParser
        Configuration object

    Attributes
    ----------
    test_value : float
        The value against which to test the condition, equal to the death rate multiplied
        by the time step for biological processes.
    """
    cdef DTYPE_FLOAT_t test_value

    def __init__(self, config):
        # The death rate. Convert config death rate to per second.
        death_rate = config.getfloat('BIO_MODEL', 'death_rate_per_day') / seconds_per_day

        # The time step for biolgical processes.
        time_step = get_bio_time_step(config)

        # Save test value
        self.test_value = death_rate * time_step

    cdef void apply(self, DataReader data_reader, DTYPE_FLOAT_t time,
                    Particle *particle) except *:
        """ Determine whether a particle dies and updates its status

        Parameters
        ----------
        data_reader : DataReader
            DataReader object used for calculating point velocities.

        time : float
            The current time.

        particle : C pointer
            C pointer to a Particle struct
        """
        cdef DTYPE_FLOAT_t p

        p = uniform(0.0, 1.0)

        if p < self.test_value:
            particle.set_is_alive(False)

        return


def get_mortality_calculator(config):
    """ Factory method for mortality calculators

    Parameters
    ----------
    config : ConfigParser
        PyLag configuraton object

    Returns
    -------
     : MortalityCalculator
         A mortality calculator
    """
    try:
        mortality = config.get("BIO_MODEL", "mortality").strip().lower()
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        return None
    else:
        if mortality == "fixed_time":
            return FixedTimeMortalityCalculator(config)
        elif mortality == "probabilistic":
            return ProbabilisticMortalityCalculator(config)
        elif mortality == "none":
            return None
        else:
            raise ValueError('Unsupported mortality calculator.')
