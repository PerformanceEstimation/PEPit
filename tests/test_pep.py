import unittest
import numpy as np

from PEPit.pep import PEP
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.function import Function
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction


class TestPEP(unittest.TestCase):

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

    def test_is_instance(self):

        self.assertIsInstance(self.problem, PEP)
        self.assertEqual(len(self.problem.list_of_functions), 1)
        self.assertEqual(len(self.problem.list_of_points), 1)
        self.assertEqual(len(self.problem.list_of_constraints), 1)
        self.assertEqual(len(self.problem.list_of_performance_metrics), 1)
        self.assertEqual(len(self.func.list_of_constraints), 0)

        pepit_tau = self.problem.solve(verbose=0)
        self.assertEqual(len(self.func.list_of_constraints), 2)
        self.assertEqual(Point.counter, 3)
        self.assertEqual(Expression.counter, 2)
        self.assertEqual(Function.counter, 1)

    def test_eval_points_and_function_values(self):

        self.problem.solve(verbose=0)

        for triplet in self.func.list_of_points:

            point, gradient, function_value = triplet

            self.assertIsInstance(point.eval(), np.ndarray)
            self.assertIsInstance(gradient.eval(), np.ndarray)
            self.assertIsInstance(function_value.eval(), float)

    def test_eval_constraint_dual_values(self):

        pepit_tau = self.problem.solve(verbose=0)
        self.assertAlmostEqual(pepit_tau, self.theoretical_tau, delta=self.theoretical_tau * 10 ** -3)

        for condition in self.problem.list_of_constraints:
            self.assertIsInstance(condition._dual_variable_value, float)
            self.assertAlmostEqual(condition._dual_variable_value, pepit_tau, delta=pepit_tau * 10 ** -3)

        for constraint in self.func.list_of_constraints:
            self.assertIsInstance(constraint._dual_variable_value, float)
            self.assertAlmostEqual(constraint._dual_variable_value,
                                   2 * self.gamma * max(abs(1 - self.mu * self.gamma), abs(1 - self.L * self.gamma)),
                                   delta=2 * self.gamma * 10 ** 3)

    def test_lmi_constraints(self):

        # Overwrite initial constraint
        R = 3
        self.problem.list_of_constraints = [(self.x0 - self.xs) ** 2 <= R**2]

        # Define new expression
        expr = Expression()

        # Enforce this expression to be at most ||x0 - xs||
        matrix = np.array([[(self.x0 - self.xs) ** 2, expr], [expr, 1]])
        self.problem.add_psd_matrix(matrix=matrix)

        # Overwrite performance metric to evaluate the maximal value expr can take
        self.problem.list_of_performance_metrics = [expr]
        pepit_tau = self.problem.solve(verbose=0)

        # This must be R
        self.assertAlmostEqual(pepit_tau, R, delta=R * 10 ** -3)

        # The value stored in expr must be equal to the optimal value of this SDP.
        self.assertAlmostEqual(expr.eval(), pepit_tau, delta=pepit_tau * 10 ** -3)

    def test_lmi_constraints_in_real_problem(self):

        # Define new expression from points
        # (which is supposed to be transparent from the theoretical problem point of vue)
        point = Point()
        expr = point ** 2

        # Enforce this expression to be at most ||x1 - xs||
        matrix = np.array([[(self.x1 - self.xs) ** 2, expr], [expr, 1]])
        self.problem.add_psd_matrix(matrix=matrix)

        # Overwrite performance metric to evaluate the maximal value expr can take
        self.problem.list_of_performance_metrics = [expr]
        pepit_tau = self.problem.solve(verbose=0)

        # This must be the square root of theoretical tau
        theoretical_tau = np.sqrt(self.theoretical_tau)
        self.assertAlmostEqual(pepit_tau, theoretical_tau, delta=theoretical_tau * 10 ** -3)

        # The value stored in expr must be the square of the one stored in point.
        self.assertAlmostEqual(np.sum(point.eval()**2), expr.eval(), delta=expr.eval() * 10 ** -3)

        # The value stored in expr must be equal to the optimal value of this SDP.
        self.assertAlmostEqual(expr.eval(), pepit_tau, delta=pepit_tau * 10 ** -3)

    def test_trace_trick(self):

        # Compute pepit_tau very basically
        pepit_tau = self.problem.solve(verbose=0)

        # Return the full problem and verify the problem value is still pepit_tau
        prob = self.problem.solve(verbose=0, return_full_cvxpy_problem=True, dimension_reduction_heuristic=None)
        self.assertAlmostEqual(prob.value, pepit_tau, delta=10 ** -2)

        # Return the full dimension reduction problem
        # and verify that its value is not pepit_tau anymore but the heuristic value
        prob2 = self.problem.solve(verbose=0, return_full_cvxpy_problem=True, dimension_reduction_heuristic="trace")
        self.assertAlmostEqual(prob2.value, 1 / 2, delta=10 ** -2)

        # Verify that, even with dimension reduction (using trace heuristic),
        # the solve method returns the worst-case performance, not the chosen heuristic value.
        pepit_tau2 = self.problem.solve(verbose=0, dimension_reduction_heuristic="trace")
        self.assertAlmostEqual(pepit_tau, pepit_tau2, delta=10 ** -2)

        # Verify that, even with dimension reduction (using 2 steps of local regularization of the log det heuristic),
        # the solve method returns the worst-case performance, not the chosen heuristic value.
        pepit_tau3 = self.problem.solve(verbose=0, dimension_reduction_heuristic="logdet2")
        self.assertAlmostEqual(pepit_tau, pepit_tau3, delta=10 ** -2)

    def tearDown(self):

        Point.counter = 0
        Expression.counter = 0
        Function.counter = 0
