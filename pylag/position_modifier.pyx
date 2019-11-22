include "constants.pxi"

from libc.math cimport cos

from pylag import parameters


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
    cdef void update_position(self, Particle *particle, Delta *delta_X) except *:
        """ Update the particle's position

        Parameters:
        -----------
        particle : C pointer
            C pointer to a Particle struct

        delta_X : C pointer
            C pointer to a Delta struct
        """
        particle.x1 += delta_X.x1
        particle.x2 += delta_X.x2
        particle.x3 += delta_X.x3


cdef class SphericalPositionModifier(PositionModifier):
    """ Update particle positions within a spherical polar coordinate system

    """
    cdef DTYPE_FLOAT_t deg_to_rad
    cdef DTYPE_FLOAT_t multiplier

    def __init__(self):
        self.deg_to_rad = parameters.deg_to_rad
        self.multiplier = self.deg_to_rad * parameters.earths_radius

    cdef void update_position(self, Particle *particle, Delta *delta_X) except *:
        """ Update the particle's position

        Parameters:
        -----------
        particle : C pointer
            C pointer to a Particle struct

        delta_X : C pointer
            C pointer to a Delta struct
        """
        particle.x1 = particle.x1 + delta_X.x1 / (self.multiplier * cos(self.deg_to_rad * particle.x2))
        particle.x2 = particle.x2 + delta_X.x2 / self.multiplier
        particle.x3 = particle.x3 + delta_X.x3


def get_position_modifier(config):
    """ Factory method for constructing PositionModifier objects

    Parameters:
    -----------
    config : ConfigParser
        Object of type ConfigParser.
    """
    if not config.has_option("OCEAN_CIRCULATION_MODEL", "coordinate_system"):
        raise ValueError("Failed to find the option `coordinate_system' in the "\
                "supplied configuration file.")

    # Return the specified numerical integrator.
    if config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system") == "cartesian":
        return CartesianPositionModifier(config)
    elif config.get("OCEAN_CIRCULATION_MODEL", "coordinate_system") == "spherical":
        return SphericalPositionModifier(config)
    else:
        raise ValueError("Unsupported coordinate system specified")

