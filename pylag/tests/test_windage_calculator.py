from unittest import TestCase
import numpy as np
import numpy.testing as test

try:
    import configparser
except ImportError:
    import ConfigParser as configparser


from pylag.exceptions import PyLagValueError
from pylag import windage

from pylag.mock import MockAtmosphereDataReader
from pylag.particle_cpp_wrapper import ParticleSmartPtr


class Windage_test(TestCase):

    def test_get_windage_calculator_returns_none_when_one_is_not_specified_in_the_run_config(self):
        # No section of option
        config = configparser.ConfigParser()
        windage_calculator = windage.get_windage_calculator(config)
        assert windage_calculator is None

        # Has the section, but no option
        config.add_section('WINDAGE')
        windage_calculator = windage.get_windage_calculator(config)
        assert windage_calculator is None

    def test_get_windage_calculator(self):
        config = configparser.ConfigParser()
        config.add_section('WINDAGE')
        config.set('WINDAGE', 'windage_calculator', 'zero_deflection')
        config.add_section("ZERO_DEFLECTION_WINDAGE_CALCULATOR")
        config.set('ZERO_DEFLECTION_WINDAGE_CALCULATOR', 'wind_factor',
                   '0.5')
        windage_calculator = windage.get_windage_calculator(config)
        assert isinstance(windage_calculator,
                          windage.ZeroDeflectionWindageCalculator)

    def test_get_invalid_windage_calculator(self):
        config = configparser.ConfigParser()
        config.add_section('WINDAGE')
        config.set('WINDAGE', 'windage_calculator', 'does_not_exist')

        self.assertRaises(PyLagValueError,
                          windage.get_windage_calculator,
                          config)

    def test_get_fixed_drag_windage_velocity(self):
        """ Test drag coefficient calculation

        Given:

        u10 = 4.0 m/s
        v10 = 3.0 m/s
        wspd = 5.0 m/s

        and with a drag coefficient of 0.5, it should return:

        u10 = 2.0
        v10 = 1.5
        """
        config = configparser.ConfigParser()
        config.add_section('WINDAGE')
        config.set('WINDAGE', 'windage_calculator', 'zero_deflection')
        config.add_section("ZERO_DEFLECTION_WINDAGE_CALCULATOR")
        config.set('ZERO_DEFLECTION_WINDAGE_CALCULATOR', 'wind_factor',
                   '0.5')

        windage_calculator = windage.get_windage_calculator(config)

        # Create test data reader
        test_data_reader = MockAtmosphereDataReader(u10=4.0, v10=3.0)
        test_time = 0.0
        test_particle = ParticleSmartPtr()
        windage_val = windage_calculator.get_velocity_wrapper(test_data_reader,
                                                              test_time,
                                                              test_particle)

        test.assert_array_almost_equal(windage_val, [2.0, 1.5])
