"""
Set module level variables controlling floating point and integral types.
These values are used throughout PyLag's codebase in order to ensure
there is consistent mapping between Python and C types.

See also
--------
data_types_python : The corresponding python types used with Python.
"""

import numpy as np

# Cython imports
cimport numpy as np

ctypedef np.float64_t DTYPE_FLOAT_t
ctypedef np.int64_t DTYPE_INT_t
