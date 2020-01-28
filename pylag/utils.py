from __future__ import print_function

import datetime

def round_time(datetime_raw, rounding_interval=3600):
    """Apply rounding to datetime objects
    
    Rounding is sometimes required when simulation times are written to file 
    with limited precision.

    Parameters:
    -----------
    datetime_raw: List, Datetime
        List of datetime objects to which rounding should be applied

    rounding_interval: int, optional
        No. of seconds to round to (default 3600, or one hour)
        
    Returns:
    --------
    datetime_rounded: List, Datetime
        List of rounded datetime objects
    """
    datetime_rounded = []
    for dt in datetime_raw:
        seconds = (dt - dt.min).seconds
        rounding = (seconds + rounding_interval/2) // rounding_interval * rounding_interval
        datetime_rounded.append(dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond))
    return datetime_rounded
