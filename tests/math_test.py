from unittest import TestCase
import numpy as np
import numpy.testing as test

from pylag.cwrappers import det_wrapper, inner_product_wrapper

def test_det():
    a = [1.0, 2.0]
    b = [3.0, 4.0]
    c = det_wrapper(a, b)
    test.assert_array_almost_equal(c, -2.0)

def test_inner_product():
    a = [1.0, 2.0]
    b = [3.0, 4.0]
    c = inner_product_wrapper(a, b)
    test.assert_array_almost_equal(c, 11.0)

