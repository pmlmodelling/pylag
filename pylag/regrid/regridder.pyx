"""
Regridder objects can be used to regrid input data.

Note
----
regridder is implemented in Cython. Only a small portion of the
API is exposed in Python with accompanying documentation.
"""

include "constants.pxi"

# Python imports
from pylag.arakawa_a_data_reader import ArakawaADataReader
from pylag.fvcom_data_reader import FVCOMDataReader
from pylag.roms_data_reader import ROMSDataReader
from pylag.mediator import SerialMediator

# Cython imports
from pylag.data_reader cimport DataReader


cdef class Regridder:
    """ A Regridder object

    An object of type Regridder can be used to interpolate data onto
    a new given grid. Regridder's leverage PyLag's interpolation
    routines to facilitate the task.

    Parameters
    ----------
    config : configparser.ConfigParser
        PyLag configuration object

    datetime_start : datetime.datetime
        The earliest date and time at which regridded data is desired.
        Requested here as PyLag will first check that the given date and
        time is covered by the input data.

    datetime_end : datetime.datetime
        The latest date and time at which regridded data is desired.
        Requested here as PyLag will first check that the given date and
        time is covered by the input data.

    Attributes
    ----------
    datetime_start : datetime.datetime
        The earliest date and time at which regridded data is desired.
        Requested here as PyLag will first check that the given date and
        time is covered by the input data.

    datetime_end : datetime.datetime
        The latest date and time at which regridded data is desired.
        Requested here as PyLag will first check that the given date and
        time is covered by the input data.

    data_reader : pylag.data_reader.DataReader
        A PyLag DataReader object.
    """
    cdef object datetime_start
    cdef object datetime_end

    cdef DataReader data_reader

    def __init__(self, config, datetime_start, datetime_end):

        # Save reference times
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end

        # Intialise the data reader
        if config.get("OCEAN_CIRCULATION_MODEL", "name") == "ArakawaA":
            mediator = SerialMediator(config, datetime_start, datetime_end)
            self.data_reader = ArakawaADataReader(config, mediator)
        elif config.get("OCEAN_CIRCULATION_MODEL", "name") == "FVCOM":
            mediator = SerialMediator(config, datetime_start, datetime_end)
            self.data_reader = FVCOMDataReader(config, mediator)
        elif config.get("OCEAN_CIRCULATION_MODEL", "name") == "ROMS":
            mediator = SerialMediator(config, datetime_start, datetime_end)
            self.data_reader = ROMSDataReader(config, mediator)
        else:
            raise ValueError('Unsupported ocean circulation model.')