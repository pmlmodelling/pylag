from nose.tools import raises

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from pylag import boundary_conditions as bc


def test_get_horiz_boundary_condition_calculator_returns_none_when_one_is_not_specified_in_the_run_config():
    # No section of option
    config = configparser.SafeConfigParser()
    hbc = bc.get_horiz_boundary_condition_calculator(config)
    assert hbc is None

    # Has the section, but no option
    config.add_section('BOUNDARY_CONDITIONS')
    hbc = bc.get_horiz_boundary_condition_calculator(config)
    assert hbc is None

def test_get_reflecting_horiz_boundary_condition_calculator():
    config = configparser.SafeConfigParser()
    config.add_section('BOUNDARY_CONDITIONS')
    config.set('BOUNDARY_CONDITIONS', 'horiz_bound_cond', 'reflecting')
    hbc = bc.get_horiz_boundary_condition_calculator(config)
    assert isinstance(hbc, bc.RefHorizBoundaryConditionCalculator)

@raises(ValueError)
def test_get_invalid_horiz_boundary_condition_calculator():
    config = configparser.SafeConfigParser()
    config.add_section('BOUNDARY_CONDITIONS')
    config.set('BOUNDARY_CONDITIONS', 'horiz_bound_cond', 'does_not_exist')
    hbc = bc.get_horiz_boundary_condition_calculator(config)

def test_get_vert_boundary_condition_calculator_returns_none_when_one_is_not_specified_in_the_run_config():
    # No section of option
    config = configparser.SafeConfigParser()
    vbc = bc.get_vert_boundary_condition_calculator(config)
    assert vbc is None

    # Has the section, but no option
    config.add_section('BOUNDARY_CONDITIONS')
    vbc = bc.get_vert_boundary_condition_calculator(config)
    assert vbc is None

def test_get_reflecting_vert_boundary_condition_calculator():
    config = configparser.SafeConfigParser()
    config.add_section('BOUNDARY_CONDITIONS')
    config.set('BOUNDARY_CONDITIONS', 'vert_bound_cond', 'reflecting')
    vbc = bc.get_vert_boundary_condition_calculator(config)
    assert isinstance(vbc, bc.RefVertBoundaryConditionCalculator)

@raises(ValueError)
def test_get_invalid_vert_boundary_condition_calculator():
    config = configparser.SafeConfigParser()
    config.add_section('BOUNDARY_CONDITIONS')
    config.set('BOUNDARY_CONDITIONS', 'vert_bound_cond', 'does_not_exist')
    vbc = bc.get_vert_boundary_condition_calculator(config)

