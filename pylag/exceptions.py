""" PyLag exceptions

The intention here is to help callers distinguish between expected
PyLag exceptions and regular python exceptions, which indicate a
bug in PyLag's API. See Item 87 of Effective Python by Brett
Slatkin (2020) for a full description.
"""


class PyLagException(Exception):
    pass


class PyLagRuntimeError(PyLagException):
    pass


class PyLagValueError(PyLagException):
    pass


class PyLagTypeError(PyLagException):
    pass


class PyLagOutOfBoundsError(PyLagException):
    pass
