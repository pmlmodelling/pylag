"""
Set module level variables controlling floating point and integral types.
These values are used throughout PyLag's codebase in order to ensure
there is consistent mapping between Python and C types. The corresponding
module level names are `DTYPE_FLOAT` and `DTYPE_INT`.

See also
--------
data_types_cython : The corresponding C types used with Cython.
"""

import numpy as np

DTYPE_FLOAT = np.float64
DTYPE_INT = np.int64

FLOAT_INVALID = np.ma.default_fill_value(np.dtype(DTYPE_FLOAT))
INT_INVALID = -999
