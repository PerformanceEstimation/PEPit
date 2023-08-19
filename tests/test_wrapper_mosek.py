import unittest
import numpy as np
import mosek as mosek

import PEPit.wrappers.mosek_wrapper as wrap
from PEPit.pep import PEP
from PEPit import MOSEK
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.function import Function
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.primitive_steps import inexact_gradient_step


class TestWrapperMOSEK(unittest.TestCase):

    def setUp(self):
        # Smooth strongly convex gradient descent set up
        self.L = 1.
        self.mu = 0.1
        self.gamma = 1 / self.L

        # Instantiate PEP
        self.problem = PEP()

        # Declare a strongly convex smooth function
        self.func = self.problem.declare_function(SmoothStronglyConvexFunction, mu=self.mu, L=self.L)

        # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
        self.xs = self.func.stationary_point()

        # Then define the starting point x0 of the algorithm
        self.x0 = self.problem.set_initial_point()

        # Set the initial constraint that is the distance between x0 and x^*
        self.problem.set_initial_condition((self.x0 - self.xs) ** 2 <= 1)

        # Run n steps of the GD method
        self.x1 = self.x0 - self.gamma * self.func.gradient(self.x0)

        # Set the performance metric to the function values accuracy
        self.problem.set_performance_metric((self.x1 - self.xs) ** 2)

        # Compute theoretical rate of the above problem
        self.theoretical_tau = max((1 - self.mu * self.gamma) ** 2, (1 - self.L * self.gamma) ** 2)

    def test_dimension_reduction(self):

        # Compute pepit_tau very basically
        pepit_tau = self.problem.solve(verbose=0)

        # Return the full problem and verify the problem value is still pepit_tau
        prob = self.problem.solve(verbose=0, return_full_problem=True, dimension_reduction_heuristic=None, solver=MOSEK)
        self.assertAlmostEqual(prob.getprimalobj(mosek.soltype.itr), pepit_tau, delta=10 ** -2)

        # Return the full dimension reduction problem
        # and verify that its value is not pepit_tau anymore but the heuristic value
        prob2 = self.problem.solve(verbose=0, return_full_problem=True, dimension_reduction_heuristic="trace", solver=MOSEK)
        self.assertAlmostEqual(prob2.getprimalobj(mosek.soltype.itr), .5 + self.mu ** 2, delta=10 ** -2)

        # Verify that, even with dimension reduction (using trace heuristic),
        # the solve method returns the worst-case performance, not the chosen heuristic value.
        pepit_tau2 = self.problem.solve(verbose=0, dimension_reduction_heuristic="trace", solver=MOSEK)
        self.assertAlmostEqual(pepit_tau2, pepit_tau, delta=10 ** -2)

        # Verify that, even with dimension reduction (using 2 steps of local regularization of the log det heuristic),
        # the solve method returns the worst-case performance, not the chosen heuristic value.
        pepit_tau3 = self.problem.solve(verbose=0, dimension_reduction_heuristic="logdet2", solver=MOSEK)
        self.assertAlmostEqual(pepit_tau3, pepit_tau, delta=10 ** -2)
