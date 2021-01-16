"""
Bio model objects can be used to manage the configuration and update of
biological particles.

Note
----
bio_model is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

include "constants.pxi"

from pylag.mortality import get_mortality_calculator


cdef class BioModel:
    """ A bio model object

    Attributes
    ----------
    mortality_calculator : MortalityCalculator
        Mortality calculator.
    """
    def __init__(self, config):
        self.mortality_calculator = get_mortality_calculator(config)

    cdef void set_initial_particle_properties(self, Particle *particle) except *:
        """ Initialise particle properties

        Parameters
        ----------
        particle : C pointer
            C pointer to a Particle struct
        """
        # Make the particle alive
        particle.set_is_alive(True)

        # Initialise mortality parameters
        if self.mortality_calculator:
            self.mortality_calculator.set_initial_particle_properties(particle)

    cdef void update(self, DataReader data_reader, DTYPE_FLOAT_t time,
                     Particle *particle) except *:
        """ Update particle properties

        Parameters
        ----------
        data_reader : DataReader
            DataReader object used for calculating point velocities.

        time : float
            The current time.

        particle : C pointer
            C pointer to a Particle struct
        """
        # Mortality
        if self.mortality_calculator:
            self.mortality_calculator.apply(data_reader, time, particle)
