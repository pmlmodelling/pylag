"""
Position modifiers which manage the application of deltas to particle positions.
Included within a separate module in order to support multiple coordinate systems.

Note
----
position_modifier is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

include "constants.pxi"

from libc.math cimport cos

from pylag.parameters cimport deg_to_radians, earth_radius


cdef class PositionModifier:
    """ An abstract base class for position modifiers

    The following method(s) should be implemented in the derived class:

    * :meth: `update_position`
    """

    cdef void update_position(self, Particle *particle, Delta *delta_X) except *:
        raise NotImplementedError

cdef class CartesianPositionModifier(PositionModifier):
    """ Update particle positions within a cartesian coordinate system

    """
    def __init__(self):
        pass

    cdef void update_position(self, Particle *particle, Delta *delta_X) except *:
        """ Update the particle's position

        Parameters
        -----------
        particle : C pointer
            C pointer to a Particle struct

        delta_X : C pointer
            C pointer to a Delta struct
        """
        cdef DTYPE_FLOAT_t x1, x2, x3

        x1 = particle.get_x1() + delta_X.x1
        x2 = particle.get_x2() + delta_X.x2
        x3 = particle.get_x3() + delta_X.x3

        particle.set_x1(x1)
        particle.set_x2(x2)
        particle.set_x3(x3)


cdef class GeographicPositionModifier(PositionModifier):
    """ Update particle positions within a geographic coordinate system

    """
    cdef DTYPE_FLOAT_t deg_to_rad
    cdef DTYPE_FLOAT_t multiplier

    def __init__(self):
        self.multiplier = earth_radius

    cdef void update_position(self, Particle *particle, Delta *delta_X) except *:
        """ Update the particle's position

        Parameters
        -----------
        particle : C pointer
            C pointer to a Particle struct

        delta_X : C pointer
            C pointer to a Delta struct
        """
        cdef DTYPE_FLOAT_t x1, x2, x3

        x1 = particle.get_x1() + delta_X.x1 / (self.multiplier * cos(particle.get_x2()))
        x2 = particle.get_x2() + delta_X.x2 / self.multiplier
        x3 = particle.get_x3() + delta_X.x3

        particle.set_x1(x1)
        particle.set_x2(x2)
        particle.set_x3(x3)


def get_position_modifier(config):
    """ Factory method for constructing PositionModifier objects

    Parameters
    ----------
    config : ConfigParser
        Object of type ConfigParser.
    """
    if not config.has_option("OCEAN_CIRCULATION_MODEL", "coordinate_system"):
        raise ValueError("Failed to find the option `coordinate_system' in the "\
                "supplied configuration file.")

    # Return the specified numerical integrator.
    if config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system") == "cartesian":
        return CartesianPositionModifier()
    elif config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system") == "geographic":
        return GeographicPositionModifier()
    else:
        raise ValueError("Unsupported coordinate system specified")

