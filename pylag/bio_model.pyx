"""
Bio model objects can be used to manage the configuration and update of
biological particles.

Note
----
bio_model is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

include "constants.pxi"


cdef class BioModel:
    """ A bio model object
    """
    def __init__(self):
        pass

    cdef void set_initial_particle_properties(self, Particle *particle) except *:
        """ Initialise particle properties

        Parameters
        ----------
        particle : C pointer
            C pointer to a Particle struct
        """
        # Make the particle alive
        particle.set_is_alive(True)

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
        pass
