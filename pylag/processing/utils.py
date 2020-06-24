from __future__ import division, print_function

import numpy as np
import datetime
import glob
from natsort import natsorted, ns


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
    return np.array(datetime_rounded)


def get_time_index(dates, ref_date, tol=60):
    """ Get array index that best matches ref_date

    Parameters
    ----------
    dates : ndarray, Datetime

    ref_date : Datetime
    """
    for idx, date in enumerate(dates):
        if abs((date - ref_date).total_seconds()) < tol:
            # i.e. to within a minute
            return idx
    raise RuntimeError('Date not found:'.format(ref_date))


def get_grid_bands(array_in):
    """ Return grid bands for data on a regular 1D grid

    Parameters:
    -----------
    array : 1D NumPy array
        1D array of regularly spaced data

    Returns:
    --------
     : 1D NumPy Array
        1D array with size len(array) + 1, corresponding to the grid edges
    """
    if len(array_in.shape) != 1:
        raise ValueError('Expected 1D array')

    n_data_points = array_in.shape[0]

    dx = array_in[1] - array_in[0]

    array_out = np.empty((n_data_points + 1), dtype=array_in.dtype)

    array_out[:-1] = array_in - dx/2.
    array_out[-1] = array_in[-1] + dx/2.

    return array_out


def get_file_list(data_dir, file_stem):
    """ Get sorted file list

    Parameters:
    -----------
    data_dir : str
        The directory containing the files.

    file_stem : str
        The part of the file name that is common to all output files. For
        example, if ensemble outputs are stored in the files `pylag_1.nc',
        `pylag_2.nc' etc, then the file_stem could be `pylag_'.
    """
    file_names = natsorted(glob.glob("{}/{}*.nc".format(data_dir, file_stem)), alg=ns.D)

    if len(file_names) == 0:
        raise ValueError("Failed to find any output files matching the "
                         "string `{}' in the directory `{}'.".format(file_stem,
                         data_dir))

    return file_names
