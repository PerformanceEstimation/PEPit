import unittest

from PEPit.pep import PEP
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.constraint import Constraint
from PEPit.function import Function

from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction


class TestConstraints(unittest.TestCase):

    def setUp(self):

        # Smooth-strongly convex gradient descent set up
        self.L = 1.
        self.mu = 0.1
        self.gamma = 1 / self.L

        # Instantiate PEP
        self.problem = PEP()

        # Declare a strongly convex smooth function
        self.func = self.problem.declare_function(SmoothStronglyConvexFunction, L=self.L, mu=self.mu)

        # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
        self.xs = self.func.stationary_point()

        # Then define the starting point x0 of the algorithm
        self.x0 = self.problem.set_initial_point()

        # Set the initial constraint that is the distance between x0 and x^*
        self.initial_condition = (self.x0 - self.xs) ** 2 <= 1
        self.problem.set_initial_condition(self.initial_condition)

        # Run n steps of the GD method
        self.x1 = self.x0 - self.gamma * self.func.gradient(self.x0)

        # Set the performance metric to the function values accuracy
        self.performance_metric = (self.x1 - self.xs) ** 2
        self.problem.set_performance_metric(self.performance_metric)

        self.solution = self.problem.solve(verbose=0, dimension_reduction_heuristic="logdet10")

    def test_is_instance(self):

        self.assertIsInstance(self.func, Function)
        self.assertIsInstance(self.func, SmoothStronglyConvexFunction)
        self.assertIsInstance(self.xs, Point)
        self.assertIsInstance(self.x0, Point)
        self.assertIsInstance(self.x1, Point)
        for i in range(len(self.problem.list_of_constraints)):
            self.assertIsInstance(self.problem.list_of_constraints[i], Constraint)
            self.assertIsInstance(self.problem.list_of_constraints[i].expression, Expression)
        for i in range(len(self.func.list_of_class_constraints)):
            self.assertIsInstance(self.func.list_of_class_constraints[i], Constraint)
            self.assertIsInstance(self.func.list_of_class_constraints[i].expression, Expression)

    def test_counter(self):

        self.assertIs(self.func.counter, 0)
        self.assertIs(self.xs.counter, 0)
        self.assertIs(self.x0.counter, 1)
        self.assertIs(self.x1.counter, None)

        # conditions are first added as Constraint in PEP
        for i in range(len(self.problem.list_of_constraints)):
            self.assertIs(self.problem.list_of_constraints[i].counter, i)

        # class constraints are added after initial conditions in PEP
        for i in range(len(self.func.list_of_class_constraints)):
            self.assertIs(self.func.list_of_class_constraints[i].counter, i + len(self.problem.list_of_constraints))

    def test_name(self):

        self.assertIsNone(self.initial_condition.get_name())
        self.assertIsNone(self.performance_metric.get_name())

        self.initial_condition.set_name("init")
        self.performance_metric.set_name("perf")

        self.assertEqual(self.initial_condition.get_name(), "init")
        self.assertEqual(self.performance_metric.get_name(), "perf")

    def test_equality_inequality(self):

        for i in range(len(self.func.list_of_class_constraints)):
            self.assertIsInstance(self.func.list_of_class_constraints[i].equality_or_inequality, str)
            self.assertIn(self.func.list_of_class_constraints[i].equality_or_inequality, {'equality', 'inequality'})

        for i in range(len(self.problem.list_of_constraints)):
            self.assertIsInstance(self.problem.list_of_constraints[i].equality_or_inequality, str)
            self.assertIn(self.problem.list_of_constraints[i].equality_or_inequality, {'equality', 'inequality'})

    def test_eval(self):

        for i in range(len(self.func.list_of_class_constraints)):
            self.assertIsInstance(self.func.list_of_class_constraints[i].eval(), float)

        for i in range(len(self.problem.list_of_constraints)):
            self.assertIsInstance(self.problem.list_of_constraints[i].eval(), float)

    def test_eval_dual(self):

        for i in range(len(self.func.list_of_class_constraints)):
            self.assertIsInstance(self.func.list_of_class_constraints[i].eval_dual(), float)

        for i in range(len(self.problem.list_of_constraints)):
            self.assertIsInstance(self.problem.list_of_constraints[i].eval_dual(), float)

        self.assertEqual(len([constraint.eval_dual() for constraint in self.func.list_of_class_constraints]), 2)
        for constraint in self.func.list_of_class_constraints:
            self.assertAlmostEqual(constraint.eval_dual(), 1.8, places=4)
