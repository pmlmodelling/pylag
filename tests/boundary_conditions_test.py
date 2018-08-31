from unittest import TestCase
import numpy.testing as test

# Test calculators
from pylag.boundary_conditions import RefVertBoundaryConditionCalculator

# Test particles
from pylag.particle import ParticleSmartPtr


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

