from nose.tools import raises

from ConfigParser import SafeConfigParser

from pylag.numerics import get_num_method

def test_set_valid_OSONumMethod_advection_and_diffusion_time_steps():
    # Create config
    config = SafeConfigParser()

    config.add_section("SIMULATION")
    config.set('SIMULATION', 'horiz_bound_cond', 'None')
    config.set('SIMULATION', 'vert_bound_cond', 'None')

    config.add_section("NUMERICS")
    config.set('NUMERICS', 'num_method', 'operator_split_0')
    config.set('NUMERICS', 'adv_iterative_method', 'Adv_RK4_3D')
    config.set('NUMERICS', 'diff_iterative_method', 'Diff_Milstein_3D')

    # Valid time steps
    config.set('NUMERICS', 'time_step_adv', '100.0')
    config.set('NUMERICS', 'time_step_diff', '5.0')
    
    num_method = get_num_method(config)

@raises(ValueError)
def test_set_invalid_OSONumMethod_diffusion_time_step_that_is_greater_than_the_advection_time_step():
    # Create config
    config = SafeConfigParser()

    config.add_section("SIMULATION")
    config.set('SIMULATION', 'horiz_bound_cond', 'None')
    config.set('SIMULATION', 'vert_bound_cond', 'None')

    config.add_section("NUMERICS")
    config.set('NUMERICS', 'num_method', 'operator_split_0')
    config.set('NUMERICS', 'adv_iterative_method', 'Adv_RK4_3D')
    config.set('NUMERICS', 'diff_iterative_method', 'Diff_Milstein_3D')

    # Valid time steps
    config.set('NUMERICS', 'time_step_adv', '100.0')
    config.set('NUMERICS', 'time_step_diff', '200.0')

    num_method = get_num_method(config)

@raises(ValueError)
def test_set_invalid_OS0NumMethod_diffusion_time_step_that_is_not_an_exact_multiple_of_the_advection_time_step():
    # Create config
    config = SafeConfigParser()

    config.add_section("SIMULATION")
    config.set('SIMULATION', 'horiz_bound_cond', 'None')
    config.set('SIMULATION', 'vert_bound_cond', 'None')

    config.add_section("NUMERICS")
    config.set('NUMERICS', 'num_method', 'operator_split_0')
    config.set('NUMERICS', 'adv_iterative_method', 'Adv_RK4_3D')
    config.set('NUMERICS', 'diff_iterative_method', 'Diff_Milstein_3D')

    # Valid time steps
    config.set('NUMERICS', 'time_step_adv', '100.0')
    config.set('NUMERICS', 'time_step_diff', '6.0')

    num_method = get_num_method(config)

def test_set_valid_OS1NumMethod_advection_and_diffusion_time_steps():
    # Create config
    config = SafeConfigParser()

    config.add_section("SIMULATION")
    config.set('SIMULATION', 'horiz_bound_cond', 'None')
    config.set('SIMULATION', 'vert_bound_cond', 'None')

    config.add_section("NUMERICS")
    config.set('NUMERICS', 'num_method', 'operator_split_0')
    config.set('NUMERICS', 'adv_iterative_method', 'Adv_RK4_3D')
    config.set('NUMERICS', 'diff_iterative_method', 'Diff_Milstein_3D')

    # Valid time steps
    config.set('NUMERICS', 'time_step_adv', '10.0')
    config.set('NUMERICS', 'time_step_diff', '5.0')
    
    num_method = get_num_method(config)

@raises(ValueError)
def test_set_invalid_OS1NumMethod_diffusion_time_step_that_is_not_equal_to_half_the_advection_time_step():
    # Create config
    config = SafeConfigParser()

    config.add_section("SIMULATION")
    config.set('SIMULATION', 'horiz_bound_cond', 'None')
    config.set('SIMULATION', 'vert_bound_cond', 'None')

    config.add_section("NUMERICS")
    config.set('NUMERICS', 'num_method', 'operator_split_0')
    config.set('NUMERICS', 'adv_iterative_method', 'Adv_RK4_3D')
    config.set('NUMERICS', 'diff_iterative_method', 'Diff_Milstein_3D')

    # Valid time steps
    config.set('NUMERICS', 'time_step_adv', '10.0')
    config.set('NUMERICS', 'time_step_diff', '6.0')

    num_method = get_num_method(config)