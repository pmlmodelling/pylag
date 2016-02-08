import numpy as np

import numpy.testing as test

import pylag.unstruct_grid_tools as ugt

def test_sort_interpolants():
    nbe = np.array([[1,2,3]]).transpose()
    nbe_sorted = np.array([[2,3,1]]).transpose()

    a1u = np.array([[10,11,12,13]]).transpose()
    a2u = np.array([[10,11,12,13]]).transpose()

    a1u_sorted, a2u_sorted = ugt.sort_interpolants(a1u, a2u, nbe, nbe_sorted)

    test.assert_array_equal(a1u_sorted, np.array([[10, 12, 13, 11]]).transpose())
    test.assert_array_equal(a2u_sorted, np.array([[10, 12, 13, 11]]).transpose())
