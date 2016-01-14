# cython: profile=True
# cython: linetrace=True

import numpy as np
import copy
import datetime

# Cython imports
cimport numpy as np
np.import_array()

# Data types used for constructing C data structures
from pylag.data_types_python import DTYPE_INT, DTYPE_FLOAT
from data_types_cython cimport DTYPE_INT_t, DTYPE_FLOAT_t

cdef class TimeManager(object):
    cdef object _datetime_start
    cdef object _datetime_end
    
    cdef DTYPE_FLOAT_t _time_start
    cdef DTYPE_FLOAT_t _time_end
    cdef DTYPE_FLOAT_t _time

    cdef DTYPE_FLOAT_t _time_step
    cdef DTYPE_FLOAT_t _output_frequency
    
    def __init__(self, config):
        self._initialise_time_vars(config)
        
    def _initialise_time_vars(self, config):
        datetime_start = config.get("SIMULATION", "start_datetime")
        self._datetime_start = datetime.datetime.strptime(datetime_start, "%Y-%m-%d %H:%M:%S")
        datetime_end = config.get("SIMULATION", "end_datetime")
        self._datetime_end = datetime.datetime.strptime(datetime_end, "%Y-%m-%d %H:%M:%S")
        
        # Convert time counters to seconds (of type int)
        self._time_start = 0.0
        self._time_end = (self._datetime_end - self._datetime_start).total_seconds()

        # Initialise the current time
        self._time = self._time_start
        
        self._time_step = config.getfloat("SIMULATION", "time_step")
        self._output_frequency = config.getfloat("SIMULATION", "output_frequency")

    def update_current_time(self):
        self._time = self._time + self._time_step
        
    def write_output_to_file(self):
        cdef DTYPE_FLOAT_t time_diff
        
        time_diff = self._time - self._time_start
        if <int>time_diff % <int>self._output_frequency == 0:
            return 1
        return 0
    
    # Properties:

    # Integration start datetime
    property datetime_start:
        def __get__(self):
            return self._datetime_start
    
    # Integration end datetime
    property datetime_end:
        def __get__(self):
            return self._datetime_end
        
    # Current time (seconds elapsed since start)
    property time:
        def __get__(self):
            return self._time
        
    # Integration time step (seconds)
    property time_step:
        def __get__(self):
            return self._time_step
 
     # Integration end time (seconds)
    property time_end:
        def __get__(self):
            return self._time_end

