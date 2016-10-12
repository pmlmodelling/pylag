cdef reset(Delta *delta):
    """ Reset stored delta values to zero

    """
    delta.x = 0.0
    delta.y = 0.0
    delta.z = 0.0
