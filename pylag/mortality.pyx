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
cimport pylag.random as random


cdef class MortalityCalculator:
    """ Abstract base class for PyLag mortality calculator
    """
    def set_initial_particle_properties_wrapper(self, ParticleSmartPtr particle):
        """ Python friendly wrapper for apply()

        Parameters
        ----------
        particle : C pointer
            C pointer to a Particle struct
        """
        return self.set_initial_particle_properties(particle.get_ptr())

    cdef void set_initial_particle_properties(self, Particle *particle) except *:
        """ Set initial particle properties

        Parameters
        ----------
        particle : C pointer
            C pointer to a Particle struct
        """
        raise RuntimeError

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

    Three different methods are used to initialise the parameter giving the age
    at which an individual dies. These are: a) A single fixed value in days, 2)
    2) A value in days drawn from a uniform distribution with given limits, and
    3) A value in days drawn from a gaussian distribution with given mean and
    standard deviation.

    Parameters
    ----------
    config : ConfigParser
        Configuration object

    Attributes
    ----------
    age_of_death_parameter_name : str
        The age of death parameter name.

    initialisation_method : str
        The method used for initialising particle parameters.

    age_of_death : float
        The age at which an individual dies.

    minimum_bound : float
        The minimum bound used for the uniform random number initialisation method.

    maximum_bound : float
        The maximum bound used for the uniform random number initialisation method.

    mean : float
        The mean used for the guassian random number initialisation method.

    standard_deviation : float
        The standard deviation used for the guassian random number initialisation method.
    """
    cdef string age_of_death_parameter_name
    cdef object initialisation_method
    cdef DTYPE_FLOAT_t age_of_death
    cdef DTYPE_FLOAT_t minimum_bound
    cdef DTYPE_FLOAT_t maximum_bound
    cdef DTYPE_FLOAT_t mean
    cdef DTYPE_FLOAT_t standard_deviation

    def __init__(self, config):
        self.age_of_death_parameter_name = b'age_of_death_in_seconds'

        self.initialisation_method = config.get('FIXED_TIME_MORTALITY_CALCULATOR', 'initialisation_method')

        if self.initialisation_method == "common_value":
            self.age_of_death = config.getfloat('FIXED_TIME_MORTALITY_CALCULATOR', 'common_value') * seconds_per_day
        elif self.initialisation_method == "uniform_random":
            self.minimum_bound = config.getfloat('FIXED_TIME_MORTALITY_CALCULATOR', 'minimum_bound') * seconds_per_day
            self.maximum_bound = config.getfloat('FIXED_TIME_MORTALITY_CALCULATOR', 'maximum_bound') * seconds_per_day
        elif self.initialisation_method == "gaussian_random":
            self.mean = config.getfloat('FIXED_TIME_MORTALITY_CALCULATOR', 'mean') * seconds_per_day
            self.standard_deviation = config.getfloat('FIXED_TIME_MORTALITY_CALCULATOR', 'standard_deviation') * seconds_per_day
        else:
            raise ValueError("Unsupported initialisation method `{}`".format(self.initialisation_method))

    cdef void set_initial_particle_properties(self, Particle *particle) except *:
        """ Set initial particle properties

        Parameters
        ----------
        particle : C pointer
            C pointer to a Particle struct
        """
        if self.initialisation_method == "common_value":
            particle.set_parameter(self.age_of_death_parameter_name, self.age_of_death)
        elif self.initialisation_method == "uniform_random":
            particle.set_parameter(self.age_of_death_parameter_name, random.uniform(self.minimum_bound, self.maximum_bound))
        elif self.initialisation_method == "gaussian_random":
            particle.set_parameter(self.age_of_death_parameter_name, random.gauss(self.mean, self.standard_deviation))
        else:
            raise ValueError("Unsupported initialisation method `{}`".format(self.initialisation_method))

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
        if particle.get_age() >= particle.get_parameter(self.age_of_death_parameter_name):
            particle.set_is_alive(False)


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
    death_rate_parameter_name : str
        The name of the death rate parameter name.

    test_value : float
        The value against which to test the condition, equal to the death rate multiplied
        by the time step for biological processes.
    """
    cdef string death_rate_parameter_name
    cdef DTYPE_FLOAT_t death_rate
    cdef DTYPE_FLOAT_t time_step

    def __init__(self, config):
        # Death rate parameter name
        self.death_rate_parameter_name = b'death_rate_per_second'

        # The death rate. Convert config death rate to per second.
        self.death_rate = config.getfloat('PROBABILISTIC_MORTALITY_CALCULATOR', 'death_rate_per_day') / seconds_per_day

        # The time step for biolgical processes.
        self.time_step = get_bio_time_step(config)

    cdef void set_initial_particle_properties(self, Particle *particle) except *:
        """ Set initial particle properties

        Parameters
        ----------
        particle : C pointer
            C pointer to a Particle struct
        """
        particle.set_parameter(self.death_rate_parameter_name, self.death_rate)

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

        p = random.uniform(0.0, 1.0)

        if p < particle.get_parameter(self.death_rate_parameter_name) * self.time_step:
            particle.set_is_alive(False)


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
        mortality = config.get("BIO_MODEL", "mortality_calculator").strip().lower()
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
