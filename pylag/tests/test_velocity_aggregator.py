from unittest import TestCase
import numpy as np
import numpy.testing as test

try:
    import configparser
except ImportError:
    import ConfigParser as configparser


from pylag.exceptions import PyLagValueError
from pylag.velocity_aggregator import VelocityAggregator

from pylag.mock import MockOceanDataReader
from pylag.mock import MockWavesDataReader
from pylag.mock import MockAtmosphereDataReader
from pylag.composite_data_reader import CompositeDataReader

from pylag.particle_cpp_wrapper import ParticleSmartPtr


class VelocityAggregator_test(TestCase):

    def test_get_ocean_only_velocity(self):
        config = configparser.ConfigParser()
        config.add_section('OCEAN_DATA')
        config.set('OCEAN_DATA', 'name', 'test')

        vel_agg = VelocityAggregator(config)

        # Create test data reader
        u = 1.
        v = 2.
        w = 3.
        vel_ref = np.array([u, v, w])
        test_data_reader = MockOceanDataReader(u, v, w)

        # Get velocity
        test_time = 0.0
        test_particle = ParticleSmartPtr()
        vel = vel_agg.get_velocity_wrapper(test_data_reader, test_time,
                                           test_particle)

        test.assert_array_almost_equal(vel, vel_ref)

    def test_get_windage_only_velocity(self):
        config = configparser.ConfigParser()
        config.add_section('WINDAGE')
        config.set('WINDAGE', 'name', 'test')
        config.set('WINDAGE', 'windage_calculator', 'fixed_drag')
        config.add_section("FIXED_DRAG_WINDAGE_CALCULATOR")
        config.set('FIXED_DRAG_WINDAGE_CALCULATOR', 'drag_coefficient',
                   '1.0')

        vel_agg = VelocityAggregator(config)

        # Create test data reader
        test_data_reader = MockAtmosphereDataReader(u10=4.0, v10=3.0)

        # Get velocity
        test_time = 0.0
        test_particle = ParticleSmartPtr()
        vel = vel_agg.get_velocity_wrapper(test_data_reader,
                                            test_time,
                                            test_particle)

        test.assert_array_almost_equal(vel, [4.0, 3.0, 0.0])

    def test_get_stokes_drift_only_velocity(self):
        config = configparser.ConfigParser()
        config.add_section('STOKES_DRIFT')
        config.set('STOKES_DRIFT', 'stokes_drift_calculator', 'surface')

        vel_agg = VelocityAggregator(config)

        # Create test data reader
        stokes_drift_u_component = 0.1
        stokes_drift_v_component = 0.2
        test_data_reader = MockWavesDataReader(stokes_drift_u_component,
                                               stokes_drift_v_component)

        # Get velocity
        test_time = 0.0
        test_particle = ParticleSmartPtr()
        vel = vel_agg.get_velocity_wrapper(test_data_reader,
                                           test_time,
                                           test_particle)

        test.assert_array_almost_equal(vel, [0.1, 0.2, 0.0])

    def test_get_ocean_and_windage_velocity(self):
        config = configparser.ConfigParser()
        config.add_section('OCEAN_DATA')
        config.set('OCEAN_DATA', 'name', 'test')
        config.add_section('WINDAGE')
        config.set('WINDAGE', 'name', 'test')
        config.set('WINDAGE', 'windage_calculator', 'fixed_drag')
        config.add_section("FIXED_DRAG_WINDAGE_CALCULATOR")
        config.set('FIXED_DRAG_WINDAGE_CALCULATOR', 'drag_coefficient',
                   '1.0')

        vel_agg = VelocityAggregator(config)

        # Create ocean data reader
        u = 1.
        v = 2.
        w = 3.
        ocean_data_reader = MockOceanDataReader(u, v, w)

        # Atmosphere data reader
        atmos_data_reader = MockAtmosphereDataReader(u10=4.0, v10=3.0)

        # No waves data reader
        waves_data_reader = None

        test_data_reader = CompositeDataReader(config,
                                               ocean_data_reader,
                                               atmos_data_reader,
                                               waves_data_reader)

        # Get velocity
        test_time = 0.0
        test_particle = ParticleSmartPtr()
        vel = vel_agg.get_velocity_wrapper(test_data_reader,
                                           test_time,
                                           test_particle)

        test.assert_array_almost_equal(vel, [5., 5., 3.0])

    def test_get_ocean_and_stokes_drift_velocity(self):
        config = configparser.ConfigParser()
        config.add_section('OCEAN_DATA')
        config.set('OCEAN_DATA', 'name', 'test')
        config.add_section('STOKES_DRIFT')
        config.set('STOKES_DRIFT', 'stokes_drift_calculator', 'surface')

        vel_agg = VelocityAggregator(config)

        # Create ocean data reader
        u = 1.
        v = 2.
        w = 3.
        ocean_data_reader = MockOceanDataReader(u, v, w)

        # No atmosphere data reader
        atmos_data_reader = None

        # Create waves data reader
        stokes_drift_u_component = 0.1
        stokes_drift_v_component = 0.2
        waves_data_reader = MockWavesDataReader(stokes_drift_u_component,
                                                stokes_drift_v_component)

        test_data_reader = CompositeDataReader(config,
                                               ocean_data_reader,
                                               atmos_data_reader,
                                               waves_data_reader)

        # Get velocity
        test_time = 0.0
        test_particle = ParticleSmartPtr()
        vel = vel_agg.get_velocity_wrapper(test_data_reader,
                                           test_time,
                                           test_particle)

        test.assert_array_almost_equal(vel, [1.1, 2.2, 3.0])
