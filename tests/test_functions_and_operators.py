import unittest
import numpy as np

from PEPit import PEP
from PEPit.point import Point
from PEPit.function import Function

from PEPit.functions import ConvexFunction
from PEPit.functions import ConvexIndicatorFunction
from PEPit.functions import ConvexLipschitzFunction
from PEPit.functions import ConvexQGFunction
from PEPit.functions import ConvexSupportFunction
from PEPit.functions import RsiEbFunction
from PEPit.functions import SmoothConvexFunction
from PEPit.functions import SmoothConvexLipschitzFunction
from PEPit.functions import SmoothFunction
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.functions import StronglyConvexFunction

from PEPit.operators import CocoerciveOperator
from PEPit.operators import CocoerciveStronglyMonotoneOperator
from PEPit.operators import LipschitzOperator
from PEPit.operators import LipschitzStronglyMonotoneOperator
from PEPit.operators import MonotoneOperator
from PEPit.operators import NegativelyComonotoneOperator
from PEPit.operators import StronglyMonotoneOperator


class TestFunctionsAndOperators(unittest.TestCase):

    def setUp(self):

        self.pep = PEP()

        self.func1 = ConvexFunction()
        self.func2 = ConvexIndicatorFunction(D=np.inf)
        self.func3 = ConvexLipschitzFunction(M=1)
        self.func4 = ConvexQGFunction(L=1)
        self.func5 = ConvexSupportFunction(M=np.inf)
        self.func6 = RsiEbFunction(mu=.1, L=1)
        self.func7 = SmoothConvexFunction(L=1)
        self.func8 = SmoothConvexLipschitzFunction(L=1, M=1)
        self.func9 = SmoothFunction(L=1)
        self.func10 = SmoothStronglyConvexFunction(mu=.1, L=1)
        self.func11 = StronglyConvexFunction(mu=.1)

        self.operator1 = CocoerciveOperator(beta=1.)
        self.operator2 = CocoerciveStronglyMonotoneOperator(mu=.1, beta=1.)
        self.operator3 = LipschitzOperator(L=1.)
        self.operator4 = LipschitzStronglyMonotoneOperator(mu=.1, L=1.)
        self.operator5 = MonotoneOperator()
        self.operator6 = NegativelyComonotoneOperator(rho=1)
        self.operator7 = StronglyMonotoneOperator(mu=.1)

        self.point1 = Point(is_leaf=True, decomposition_dict=None)
        self.point2 = Point(is_leaf=True, decomposition_dict=None)

        self.all_functions_and_operators = [
            self.func1,
            self.func2,
            self.func3,
            self.func4,
            self.func5,
            self.func6,
            self.func7,
            self.func8,
            self.func9,
            self.func10,
            self.func11,
            self.operator1,
            self.operator2,
            self.operator3,
            self.operator4,
            self.operator5,
            self.operator6,
            self.operator7,
        ]

        self.all_points = [
            self.point1
            , self.point2,
        ]

        self.operators_weights = np.random.randn(len(self.all_functions_and_operators))
        self.new_operator = np.dot(self.operators_weights, self.all_functions_and_operators)

        self.points_weights = np.random.randn(len(self.all_points))
        self.new_point = np.dot(self.points_weights, self.all_points)

    def test_is_instance(self):

        for function in self.all_functions_and_operators:
            self.assertIsInstance(function, Function)

        self.assertIsInstance(self.func1, ConvexFunction)
        self.assertIsInstance(self.func2, ConvexIndicatorFunction)
        self.assertIsInstance(self.func3, ConvexLipschitzFunction)
        self.assertIsInstance(self.func4, ConvexQGFunction)
        self.assertIsInstance(self.func5, ConvexSupportFunction)
        self.assertIsInstance(self.func6, RsiEbFunction)
        self.assertIsInstance(self.func7, SmoothConvexFunction)
        self.assertIsInstance(self.func8, SmoothConvexLipschitzFunction)
        self.assertIsInstance(self.func9, SmoothFunction)
        self.assertIsInstance(self.func10, SmoothStronglyConvexFunction)
        self.assertIsInstance(self.func11, StronglyConvexFunction)
        self.assertIsInstance(self.operator1, CocoerciveOperator)
        self.assertIsInstance(self.operator2, CocoerciveStronglyMonotoneOperator)
        self.assertIsInstance(self.operator3, LipschitzOperator)
        self.assertIsInstance(self.operator4, LipschitzStronglyMonotoneOperator)
        self.assertIsInstance(self.operator5, MonotoneOperator)
        self.assertIsInstance(self.operator6, NegativelyComonotoneOperator)
        self.assertIsInstance(self.operator7, StronglyMonotoneOperator)

        self.assertIsInstance(self.point1, Point)
        self.assertIsInstance(self.point2, Point)

        self.assertIsInstance(self.new_operator, Function)
        self.assertIsInstance(self.new_point, Point)

    def test_linear_combination(self):

        self.assertEqual(self.new_operator.decomposition_dict,
                         {function: weight for (function, weight) in zip(self.all_functions_and_operators,
                                                                         self.operators_weights)})

        self.assertEqual(self.new_point.decomposition_dict,
                         {point: weight for (point, weight) in zip(self.all_points,
                                                                   self.points_weights)})

    def test_counter(self):

        # Test functions counters
        for order, function in enumerate(self.all_functions_and_operators):
            self.assertEqual(function.counter, order)
        self.assertIs(self.new_operator.counter, None)

        # Test points counters
        for order, point in enumerate(self.all_points):
            self.assertEqual(point.counter, order)
        self.assertIs(self.new_point.counter, None)

    def test_add_constraints(self):

        # Verify no function implemented class constraints yet
        for function in self.all_functions_and_operators:
            self.assertEqual(len(function.list_of_class_constraints), 0)
        self.assertEqual(len(self.new_operator.list_of_class_constraints), 0)

        # Call oracles on combination of functions
        for point in self.all_points:
            self.new_operator.oracle(point)
        self.new_operator.oracle(self.new_point)
        self.new_operator.stationary_point()
        num_points_eval = len(self.all_points) + 2

        # Add function class constraints
        for function in self.all_functions_and_operators:
            function.add_class_constraints()

        # Verify the number of points stored in list_of_points attributes
        for function in self.all_functions_and_operators:
            self.assertEqual(len(function.list_of_points), num_points_eval)
            self.assertEqual(len(function.list_of_stationary_points), 0)
        self.assertEqual(len(self.new_operator.list_of_points), num_points_eval)
        self.assertEqual(len(self.new_operator.list_of_stationary_points), 1)

        # Verify the number of class constraints is coherent with the function classes
        self.assertEqual(len(self.func1.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.func2.list_of_class_constraints), num_points_eval ** 2)
        self.assertEqual(len(self.func3.list_of_class_constraints), num_points_eval ** 2)
        self.assertEqual(len(self.func4.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.func5.list_of_class_constraints), num_points_eval ** 2)
        self.assertEqual(len(self.func6.list_of_class_constraints), 0)
        self.assertEqual(len(self.func7.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.func8.list_of_class_constraints), num_points_eval ** 2)
        self.assertEqual(len(self.func9.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.func10.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.func11.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.operator1.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.operator2.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.operator3.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.operator4.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.operator5.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.operator6.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.operator7.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.new_operator.list_of_class_constraints), 0)
