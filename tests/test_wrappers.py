import unittest

from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction


class TestWrapperCVXPY(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrapper = "cvxpy"

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

        # Set an expression lowering the norm of x1-xs.
        expr = Expression()
        self.problem.add_psd_matrix([[(self.x1 - self.xs) ** 2, expr], [expr, 1]])

        # Set the performance metric to the function values accuracy
        # Since we maximize expr, its value will exactly be the one of the norm of x1-xs.
        self.problem.set_performance_metric(expr)

        # Compute theoretical rate of the above problem
        self.theoretical_tau = max((1 - self.mu * self.gamma) ** 2, (1 - self.L * self.gamma) ** 2)

    def test_dimension_reduction(self):

        # Compute pepit_tau very basically.
        pepit_tau = self.problem.solve(verbose=0, wrapper=self.wrapper)

        # Compute pepit_tau very basically with dimension_reduction_heuristic off and verify all is fine.
        pepit_tau2 = self.problem.solve(verbose=0, dimension_reduction_heuristic=None, wrapper=self.wrapper)
        self.assertAlmostEqual(pepit_tau2, pepit_tau, delta=10 ** -2)

        # Verify that, even with dimension reduction (using trace heuristic),
        # the solve method returns the worst-case performance, not the chosen heuristic value.
        pepit_tau3 = self.problem.solve(verbose=0, dimension_reduction_heuristic="trace", wrapper=self.wrapper)
        self.assertAlmostEqual(pepit_tau3, pepit_tau, delta=10 ** -2)

        # Verify that, even with dimension reduction (using 2 steps of local regularization of the log det heuristic),
        # the solve method returns the worst-case performance, not the chosen heuristic value.
        pepit_tau4 = self.problem.solve(verbose=0, dimension_reduction_heuristic="logdet2", wrapper=self.wrapper)
        self.assertAlmostEqual(pepit_tau4, pepit_tau, delta=10 ** -2)


class TestWrapperMOSEK(TestWrapperCVXPY):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrapper = "mosek"
