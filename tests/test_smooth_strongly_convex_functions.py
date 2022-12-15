import unittest

from PEPit import PEP
from PEPit.point import Point
from PEPit.constraint import Constraint
from PEPit.function import Function
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction


class TestConstraintsStronglyConvex(unittest.TestCase):

    def setUp(self):

        self.L1 = 1.
        self.mu1 = 0.1
        self.L2 = 10.
        self.mu2 = 0.001

        self.pep = PEP()

        self.func1 = SmoothStronglyConvexFunction(L=self.L1, mu=self.mu1)
        self.func2 = SmoothStronglyConvexFunction(L=self.L2, mu=self.mu2)

        self.point1 = Point(is_leaf=True, decomposition_dict=None)
        self.point2 = Point(is_leaf=True, decomposition_dict=None)

    def test_is_instance(self):

        self.assertIsInstance(self.func1, Function)
        self.assertIsInstance(self.func2, Function)
        self.assertIsInstance(self.func2, SmoothStronglyConvexFunction)
        self.assertIsInstance(self.func1, SmoothStronglyConvexFunction)

    def compute_linear_combination(self):

        new_function = self.func1 + self.func2

        return new_function

    def test_linear_combination(self):

        new_function = self.compute_linear_combination()

        self.assertIsInstance(new_function, Function)
        self.assertEqual(new_function.decomposition_dict, {self.func1: 1, self.func2: 1})

    def test_counter(self):

        function = SmoothStronglyConvexFunction(L=self.L2, mu=self.mu1)

        self.assertIs(self.func1.counter, 0)
        self.assertIs(self.func2.counter, 1)
        self.assertIs(function.counter, 2)
        self.assertIs(self.point1.counter, 0)
        self.assertIs(self.point2.counter, 1)

        self.assertIs(len(function.list_of_constraints), 0)

        # Add points and constraints
        function.oracle(self.point1)
        function.oracle(self.point2)
        function.add_class_constraints()

        self.assertIs(len(function.list_of_points), 2)
        self.assertIs(len(function.list_of_constraints), 2)
        self.assertIs(function.list_of_constraints[-1].counter, 1)

    def test_add_constraints(self):
        new_function = self.compute_linear_combination()
        point3 = Point(is_leaf=True, decomposition_dict=None)

        # Add three points
        new_function.oracle(self.point1)
        new_function.oracle(self.point2)
        new_function.oracle(point3)
        self.func1.add_class_constraints()
        self.func2.add_class_constraints()

        # Count constraints
        self.assertEqual(len(new_function.list_of_constraints), 0)
        self.assertEqual(len(self.func1.list_of_constraints), 6)
        self.assertEqual(len(self.func2.list_of_constraints), 6)

        # Test constraints type
        for i in range(len(self.func1.list_of_constraints)):
            self.assertIsInstance(self.func1.list_of_constraints[i], Constraint)
            self.assertIsInstance(self.func2.list_of_constraints[i], Constraint)

    def test_sum_smooth_strongly_convex_functions(self):

        new_function = self.compute_linear_combination()
        new_function.oracle(self.point1)
        new_function.oracle(self.point2)
        self.func1.add_class_constraints()
        self.func2.add_class_constraints()

        # Test sum of smooth strongly convex functions
        L = self.L1 + self.L2
        mu = self.mu1 + self.mu2

        for i, point_i in enumerate(new_function.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(new_function.list_of_points):

                xj, gj, fj = point_j

                if i != j:
                    # Interpolation conditions of smooth strongly convex functions class
                    self.assertLessEqual(- fi + fj +
                                         gj * (xi - xj)
                                         + 1 / (2 * L) * (gi - gj) ** 2
                                         + mu / (2 * (1 - mu / L)) * (xi - xj - 1 / L * (gi - gj)) ** 2, 0)
