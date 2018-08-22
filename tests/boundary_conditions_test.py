from unittest import TestCase
import numpy.testing as test

# Test data readers
from pylag.tests.boundary_conditions import TestHorizBCDataReader

# Test calculators
from pylag.boundary_conditions import RefHorizBoundaryConditionCalculator
from pylag.boundary_conditions import RefVertBoundaryConditionCalculator

# Test particles
from pylag.particle import ParticleSmartPtr


def test_apply_reflecting_horizontal_boundary_condition():
    
    data_reader = TestHorizBCDataReader()
    
    horiz_bc_calculator = RefHorizBoundaryConditionCalculator()
    
    particle_old = ParticleSmartPtr(xpos=0.0, ypos=-1.0, host=0)
    particle_new = ParticleSmartPtr(xpos=0.0, ypos=1.0, host=0)
    
    horiz_bc_calculator.apply_wrapper(data_reader, particle_old, particle_new)

    test.assert_almost_equal(particle_new.xpos, 1.0)
    test.assert_almost_equal(particle_new.ypos, 0.0)

class RefVertBoundaryConditionCalculator_test(TestCase):

    def test_apply_reflecting_boundary_condition_for_small_excursion(self):
        vert_bc_calculator = RefVertBoundaryConditionCalculator()
        zpos = 0.4; zmin = -1.0; zmax = 0.0
        zpos_new = vert_bc_calculator.apply(zpos, zmin, zmax)
        test.assert_almost_equal(zpos_new, -0.4)

    def test_apply_reflecting_boundary_condition_for_small_excursion(self):
        vert_bc_calculator = RefVertBoundaryConditionCalculator()
        zpos = 1.4; zmin = -1.0; zmax = 0.0
        zpos_new = vert_bc_calculator.apply(zpos, zmin, zmax)
        test.assert_almost_equal(zpos_new, -0.6)

