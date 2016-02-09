from unittest import TestCase
import numpy as np
import numpy.testing as test

from pylag.configuration import get_config
from pylag.fvcom_data_reader import FVCOMDataReader
        
class FVCOMDataReader_test(TestCase):

    def setUp(self):
        self.config = get_config('../resources/pylag.cfg')
        self.data_reader = FVCOMDataReader(self.config)

    def tearDown(self):
        del(self.data_reader)

    def test_find_host_using_local_search(self):
        xpos = -39759.113 # x-centre of grid cell 500
        ypos = 5717990.0 # y-centre of grid cell 500
        guess = 689 # Known neighbour
        host = self.data_reader.find_host_using_local_search(xpos, ypos, guess)
        test.assert_equal(host, 500)
    
    def test_find_host_using_global_search(self):
        xpos = -39759.113 # x-centre of grid cell 500
        ypos = 5717990.0 # y-centre of grid cell 500
        host = self.data_reader.find_host_using_global_search(xpos, ypos)
        test.assert_equal(host, 500)
