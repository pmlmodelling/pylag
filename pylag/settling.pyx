"""
Module for settling velocity calculators.

Note
----
settling is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag.particle_cpp_wrapper cimport ParticleSmartPtr
cimport pylag.random as random


# Base class
cdef class SettlingVelocityCalculator:
    """ Abstract base class for PyLag settling velocity calculators

    SettlingVelocityCalculators set the settling velocity of particles.

    For efficiency reasons, SettlingVelocityCalculator has been implemented
    in Cython and only part of its API is exposed in Python. In order to make
    it possible to use SettlingVelocityCalculator objects in Python, a set of
    Python wrappers have been added to the SettlingVelocityCalculator base class.
    These are documented here.
    """
    def init_particle_settling_velocity_wrapper(self, DataReader data_reader, DTYPE_FLOAT_t time,
                    ParticleSmartPtr particle):
        """ Python friendly wrapper for init_particle_settling_velocity()

        Parameters
        ----------
        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            PyLag ParticleSmartPtr.
        """
        self.init_particle_settling_velocity(particle.get_ptr())

    cdef void init_particle_settling_velocity(self, Particle *particle) except *:
        raise NotImplementedError

    def set_particle_settling_velocity_wrapper(self, DataReader data_reader, DTYPE_FLOAT_t time,
                    ParticleSmartPtr particle):
        """ Python friendly wrapper for set_particle_settling_velocity()

        Parameters
        ----------
        data_reader : pylag.data_reader.DataReader
            A concrete PyLag data reader which inherits from the base class
            `pylag.data_reader.DataReader`.

        time : float
            The current time.

        particle : pylag.particle_cpp_wrapper.ParticleSmartPtr
            PyLag ParticleSmartPtr.
        """
        self.set_particle_settling_velocity(data_reader, time, particle.get_ptr())

    cdef void set_particle_settling_velocity(self, DataReader data_reader, DTYPE_FLOAT_t time,
                    Particle *particle) except *:
        raise NotImplementedError


cdef class ConstantSettlingVelocityCalculator(SettlingVelocityCalculator):
    """ Constant settling velocity calculator

    A constant settling velocity calculator assigns a constant settling
    velocity to a particle. The value does not change with time. The
    settling velocity can be initialised in one of two ways: either as
    a fixed value which is specified in the run configuration file, or
    as a random value bounded by limits given in the run configuration
    file.

    Parameters
    ----------
    config : ConfigParser
        Configuration object

    Attributes
    ----------
    config : ConfigParser
        Configuration object

    _settling_velocity_parameters_name : str
        The name of the parameter with which the settling velocity is associated.

    _w_settling_fixed : float
        Fixed settling velocity. Used if `initialisation_method` is set to `fixed_value`.
        Set using the configuration option `settling_velocity`.

    _w_settling_min : float
        Minimum settling velocity. Used if `initialisation_method` is set to `uniform_random`.
        Set using the configuration option `min_settling_velocity`.

    _w_settling_max : float
        Maximum settling velocity. Used if `initialisation_method` is set to `uniform_random`.
        Set using the configuration option `max_settling_velocity`.
    """
    # Settling velocity parameter name
    cdef string _settling_velocity_parameter_name

    # Settling velocities, used for initialisation purposes
    cdef DTYPE_FLOAT_t _w_settling_fixed
    cdef DTYPE_FLOAT_t _w_settling_min
    cdef DTYPE_FLOAT_t _w_settling_max

    def __init__(self, config):
        self._config = config

        # Settling velocity parameter name
        self._settling_velocity_parameter_name = parameter_names['settling_velocity']

        if self.config.get("CONSTANT_SETTLING_VELOCITY_CALCULATOR", "initialisation_method") == "fixed_value":
            self._w_sink_fixed = self.config.getfloat("CONSTANT_SETTLING_VELOCITY_CALCULATOR", "settling_velocity")
            self._w_sink_min = -999.
            self._w_sink_max = -999.
        elif self.config.get("CONSTANT_SETTLING_VELOCITY_CALCULATOR", "initialisation_method") == "uniform_random":
            self._w_sink_fixed = -999.
            self._w_sink_min = self.config.getfloat("CONSTANT_SETTLING_VELOCITY_CALCULATOR", "min_settling_velocity")
            self._w_sink_max = self.config.getfloat("CONSTANT_SETTLING_VELOCITY_CALCULATOR", "max_settling_velocity")
        else:
            raise ValueError("Unsupported settling velocity initialisation "\
                "method `{}'.".format(self.config.get("CONSTANT_SETTLING_VELOCITY_CALCULATOR", "initialisation_method")))

    cdef void init_particle_settling_velocity(self, Particle *particle) except *:
        """ Initialise the particle settling velocity

        Parameters
        ----------
        particle : C pointer
            C pointer to a particle struct.
        """
        cdef DTYPE_FLOAT_t w_settling = -999.

        if self.config.get("CONSTANT_SETTLING_VELOCITY_CALCULATOR", "initialisation_method") == "fixed_value":
            w_settling = self._w_sink_fixed
        elif self.config.get("CONSTANT_SETTLING_VELOCITY_CALCULATOR", "initialisation_method") == "uniform_random":
            w_settling = random.uniform(self._w_sink_min, self._w_sink_max)
        else:
            raise ValueError("Unsupported settling velocity initialisation "\
                "method `{}'.".format(self.config.get("CONSTANT_SETTLING_VELOCITY_CALCULATOR", "initialisation_method")))

        # Set the particle's settling velocity
        particle.set_bio_parameter(self.settling_velocity_parameter_name, w_settling)

    cdef void set_particle_settling_velocity(self, DataReader data_reader, DTYPE_FLOAT_t time,
                    Particle *particle) except *:
        """ Set the settling velocity

        As the velocity is fixed, do nothing.

        Parameters
        ----------
        data_reader : DataReader
            DataReader object used for calculating point velocities
            and/or diffusivities.

        time : float
            The current time.

        particle : C pointer
            C pointer to a particle struct.
        """
        pass


def get_settling_velocity_calculator(config):
    """ Factory method for settling velocity calculators

    Parameters
    ----------
    config : ConfigParser
        PyLag configuraton object

    Returns
    -------
     : SettlingVelocityCalculator
         A settling velocity calculator
    """
    try:
        settling = config.get("SETTLING", "settling_velocity_calculator").strip().lower()
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        return None
    else:
        if settling == "none":
            return None
        elif settling == "constant":
            return ConstantSettlingVelocityCalculator(config)
        else:
            raise ValueError('Unsupported settling velocity calculator.')


# Settling parameter names
# ----------------------------
parameter_names = {'settling_velocity' : b'settling_velocity_in_meters_per_second'}
