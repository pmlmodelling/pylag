from unittest import TestCase
import numpy.testing as test
import numpy as np
from ConfigParser import SafeConfigParser

from pylag.model import OPTModel

from pylag.tests.model_test_helpers import TestOPTModelDataReader

class OPTModel_test(TestCase):

    def setUp(self):
        # Create config
        config = SafeConfigParser()
        config.add_section("GENERAL")
        config.set('GENERAL', 'log_level', 'info')
        config.add_section("SIMULATION")
        config.set('SIMULATION', 'depth_coordinates', 'depth_below_surface')
        config.add_section("NUMERICS")
        config.set('NUMERICS', 'num_method', 'test')
        
        # Create test data reader
        data_reader = TestOPTModelDataReader()
        
        # Create data reader
        self.model = OPTModel(config, data_reader)

    def tearDown(self):
        del(self.model)

    def test_all_particles_lie_outside_of_the_model_domain(self):
        group_ids = np.array([1,1])
        x_positions = np.array([-1., -1.])
        y_positions = np.array([-1., -1.])
        z_positions = np.array([-1., -1.])
        time = 0.0
        self.model.set_particle_data(group_ids, x_positions, y_positions, z_positions)
        self.assertRaises(RuntimeError, self.model.seed, time)
