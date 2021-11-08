import unittest

from PEPit.tools.dict_operations import prune_dict

from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.function import Function
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.constraint import Constraint
from PEPit.pep import PEP


class TestPEP(unittest.TestCase):

    def setUp(self):
        # smooth-strongly convex gradient descent set up
        self.L = 1.
        self.mu = 0.1
        self.gamma = 1 / self.L

        # Instantiate PEP
        self.problem = PEP()

        # Declare a strongly convex smooth function
        self.func = self.problem.declare_function(SmoothStronglyConvexFunction, param={'mu': self.mu, 'L': self.L})

        # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
        self.xs = self.func.stationary_point()

        # Then define the starting point x0 of the algorithm
        self.x0 = self.problem.set_initial_point()

        # Set the initial constraint that is the distance between x0 and x^*
        self.problem.set_initial_condition((self.x0 - self.xs) ** 2 <= 1)

        # Run n steps of the GD method
        self.x1 = self.x0 - self.gamma * self.func.gradient(self.x0)

        # Set the performance metric to the function values accuracy
        self.problem.set_performance_metric((self.x1 - self.xs)**2)


    def test_is_instance(self):

        self.assertIsInstance(self.problem, PEP)
        self.assertEqual(len(self.problem.list_of_functions), 1)
        self.assertEqual(len(self.problem.list_of_points), 1)
        self.assertEqual(len(self.problem.list_of_conditions), 1)
        self.assertEqual(len(self.problem.list_of_performance_metrics), 1)
        self.assertEqual(len(self.func.list_of_constraints), 0)

        pepit_tau = self.problem.solve(verbose=False)
        self.assertEqual(len(self.func.list_of_constraints), 2)
        self.assertEqual(Point.counter, 3)
        self.assertEqual(Expression.counter, 2)
        self.assertEqual(Function.counter, 1)

    def test_eval_points_and_function_values(self):

        self.problem.solve(verbose=False)

        for triplet in self.func.list_of_points:

            point, gradient, function_value = triplet

            if point.get_is_leaf():
                self.assertIsNot(point.value, None)
            if gradient.get_is_leaf():
                self.assertIsNot(gradient.value, None)
            if function_value.get_is_function_value():
                self.assertIsNot(function_value.value, None)

    def test_eval_constraint_dual_values(self):

        pepit_tau = self.problem.solve(verbose=False)
        theoretical_tau = max((1 - self.mu*self.gamma)**2, (1 - self.L * self.gamma)**2)
        self.assertAlmostEqual(pepit_tau, theoretical_tau, delta=theoretical_tau * 10 ** -3)

        for condition in self.problem.list_of_conditions:

            self.assertIsInstance(condition.dual_variable_value, float)
            self.assertAlmostEqual(condition.dual_variable_value, pepit_tau, delta=pepit_tau * 10 ** -3)

        for constraint in self.func.list_of_constraints:

            self.assertIsInstance(constraint.dual_variable_value, float)
            self.assertAlmostEqual(constraint.dual_variable_value, 2 * self.gamma * max(abs(1 - self.mu*self.gamma), abs(1 - self.L*self.gamma)), delta=2 * self.gamma * 10 ** 3)

    def tearDown(self):

        Point.counter = 0
        Expression.counter = 0
        Function.counter = 0


