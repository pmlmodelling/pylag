"""
Module containing a small container for storing changes (deltas) in a particle's position.
"""

cdef reset(Delta *delta):
    """ Reset stored delta values to zero

    """
    delta.x1 = 0.0
    delta.x2 = 0.0
    delta.x3 = 0.0
