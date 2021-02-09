from unittest import TestCase

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag import boundary_conditions as bc


class BoundaryConditions_test(TestCase):

    def test_get_horiz_boundary_condition_calculator_returns_none_when_one_is_not_specified_in_the_run_config(self):
        # No section of option
        config = configparser.ConfigParser()
        hbc = bc.get_horiz_boundary_condition_calculator(config)
        assert hbc is None

        # Has the section, but no option
        config.add_section('BOUNDARY_CONDITIONS')
        hbc = bc.get_horiz_boundary_condition_calculator(config)
        assert hbc is None

    def test_get_reflecting_horiz_boundary_condition_calculator(self):
        config = configparser.ConfigParser()
        config.add_section('BOUNDARY_CONDITIONS')
        config.set('BOUNDARY_CONDITIONS', 'horiz_bound_cond', 'reflecting')
        config.add_section('OCEAN_CIRCULATION_MODEL')
        config.set('OCEAN_CIRCULATION_MODEL', 'coordinate_system', 'cartesian')
        hbc = bc.get_horiz_boundary_condition_calculator(config)
        assert isinstance(hbc, bc.RefHorizCartesianBoundaryConditionCalculator)

    def test_get_invalid_horiz_boundary_condition_calculator(self):
        config = configparser.ConfigParser()
        config.add_section('OCEAN_CIRCULATION_MODEL')
        config.set('OCEAN_CIRCULATION_MODEL', 'coordinate_system', 'cartesian')
        config.add_section('BOUNDARY_CONDITIONS')
        config.set('BOUNDARY_CONDITIONS', 'horiz_bound_cond', 'does_not_exist')

        self.assertRaises(ValueError, bc.get_horiz_boundary_condition_calculator, config)

    def test_get_vert_boundary_condition_calculator_returns_none_when_one_is_not_specified_in_the_run_config(self):
        # No section of option
        config = configparser.ConfigParser()
        vbc = bc.get_vert_boundary_condition_calculator(config)
        assert vbc is None

        # Has the section, but no option
        config.add_section('BOUNDARY_CONDITIONS')
        vbc = bc.get_vert_boundary_condition_calculator(config)
        assert vbc is None

    def test_get_reflecting_vert_boundary_condition_calculator(self):
        config = configparser.ConfigParser()
        config.add_section('BOUNDARY_CONDITIONS')
        config.set('BOUNDARY_CONDITIONS', 'vert_bound_cond', 'reflecting')
        vbc = bc.get_vert_boundary_condition_calculator(config)
        assert isinstance(vbc, bc.RefVertBoundaryConditionCalculator)

    def test_get_invalid_vert_boundary_condition_calculator(self):
        config = configparser.ConfigParser()
        config.add_section('BOUNDARY_CONDITIONS')
        config.set('BOUNDARY_CONDITIONS', 'vert_bound_cond', 'does_not_exist')

        self.assertRaises(ValueError, bc.get_vert_boundary_condition_calculator, config)
