from unittest import TestCase
import numpy as np
import numpy.testing as test

try:
    import configparser
except ImportError:
    import ConfigParser as configparser


from pylag.exceptions import PyLagValueError
from pylag import stokes_drift

from pylag.mock import MockWavesDataReader
from pylag.particle_cpp_wrapper import ParticleSmartPtr


class StokesDrift_test(TestCase):

    def test_get_stokes_drift_calculator_returns_none_when_one_is_not_specified_in_the_run_config(self):
        # No section of option
        config = configparser.ConfigParser()
        sdc = stokes_drift.get_stokes_drift_calculator(config)
        assert sdc is None

        # Has the section, but no option
        config.add_section('STOKES_DRIFT')
        sdc = stokes_drift.get_stokes_drift_calculator(config)
        assert sdc is None

    def test_get_stokes_drift_calculator(self):
        config = configparser.ConfigParser()
        config.add_section('STOKES_DRIFT')
        config.set('STOKES_DRIFT', 'stokes_drift_calculator', 'surface')
        sdc = stokes_drift.get_stokes_drift_calculator(config)
        assert isinstance(sdc, stokes_drift.SurfaceStokesDriftCalculator)

    def test_get_invalid_stokes_drift_calculator(self):
        config = configparser.ConfigParser()
        config.add_section('STOKES_DRIFT')
        config.set('STOKES_DRIFT', 'stokes_drift_calculator', 'does_not_exist')

        self.assertRaises(PyLagValueError,
                          stokes_drift.get_stokes_drift_calculator,
                          config)

    def test_get_surface_stokes_drift_velocity(self):
        config = configparser.ConfigParser()
        config.add_section('STOKES_DRIFT')
        config.set('STOKES_DRIFT', 'stokes_drift_calculator', 'surface')
        sdc = stokes_drift.get_stokes_drift_calculator(config)

        # Create test data reader
        stokes_drift_u_component = 0.1
        stokes_drift_v_component = 0.2
        stokes_drift_ref = np.array([stokes_drift_u_component,
                                     stokes_drift_v_component])
        test_data_reader = MockWavesDataReader(stokes_drift_u_component,
                                               stokes_drift_v_component)
        test_time = 0.0
        test_particle = ParticleSmartPtr()

        stokes_drift_val = sdc.get_velocity_wrapper(test_data_reader,
                                                    test_time,
                                                    test_particle)

        test.assert_array_almost_equal(stokes_drift_val, stokes_drift_ref)
