import unittest
import numpy as np
import pandas

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
from PEPit.functions import SmoothStronglyConvexQuadraticFunction

from PEPit.operators import CocoerciveOperator
from PEPit.operators import CocoerciveStronglyMonotoneOperator
from PEPit.operators import LipschitzOperator
from PEPit.operators import LipschitzStronglyMonotoneOperator
from PEPit.operators import MonotoneOperator
from PEPit.operators import NegativelyComonotoneOperator
from PEPit.operators import StronglyMonotoneOperator
from PEPit.operators import NonexpansiveOperator
from PEPit.operators import LinearOperator
from PEPit.operators import SymmetricLinearOperator
from PEPit.operators import SkewSymmetricLinearOperator


class TestFunctionsAndOperators(unittest.TestCase):

    def setUp(self):

        self.pep = PEP()

        self.func1 = ConvexFunction(name="f1")
        self.func2 = ConvexIndicatorFunction(D=np.inf, name="f2")
        self.func3 = ConvexLipschitzFunction(M=1, name="f3")
        self.func4 = ConvexQGFunction(L=1, name="f4")
        self.func5 = ConvexSupportFunction(M=np.inf, name="f5")
        self.func6 = RsiEbFunction(mu=.1, L=1, name="f6")
        self.func7 = SmoothConvexFunction(L=1, name="f7")
        self.func8 = SmoothConvexLipschitzFunction(L=1, M=1, name="f8")
        self.func9 = SmoothFunction(L=1, name="f9")
        self.func10 = SmoothStronglyConvexFunction(mu=.1, L=1, name="f10")
        self.func11 = StronglyConvexFunction(mu=.1, name="f11")
        self.func12 = SmoothStronglyConvexQuadraticFunction(L=1, mu=.1, name="f12")

        self.operator1 = CocoerciveOperator(beta=1., name="op1")
        self.operator2 = CocoerciveStronglyMonotoneOperator(mu=.1, beta=1., name="op2")
        self.operator3 = LipschitzOperator(L=1., name="op3")
        self.operator4 = LipschitzStronglyMonotoneOperator(mu=.1, L=1., name="op4")
        self.operator5 = MonotoneOperator(name="op5")
        self.operator6 = NegativelyComonotoneOperator(rho=1, name="op6")
        self.operator7 = StronglyMonotoneOperator(mu=.1, name="op7")
        self.operator8 = NonexpansiveOperator(name="op8")
        self.operator9 = LinearOperator(L=1, name="op9")
        self.operator10 = SymmetricLinearOperator(mu=.1, L=1, name="op10")
        self.operator11 = SkewSymmetricLinearOperator(L=1, name="op11")

        self.point1 = Point(is_leaf=True, decomposition_dict=None, name="pt1")
        self.point2 = Point(is_leaf=True, decomposition_dict=None, name="pt2")

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
            self.func12,
            self.operator1,
            self.operator2,
            self.operator3,
            self.operator4,
            self.operator5,
            self.operator6,
            self.operator7,
            self.operator8,
            self.operator9,
            self.operator10,
            self.operator11,
        ]

        self.all_points = [
            self.point1,
            self.point2,
        ]

        self.operators_weights = np.random.randn(len(self.all_functions_and_operators))
        self.new_operator: Function = np.dot(self.operators_weights, self.all_functions_and_operators)
        self.new_operator.set_name("combined op")

        self.points_weights = np.random.randn(len(self.all_points))
        self.new_point: Point = np.dot(self.points_weights, self.all_points)
        self.new_point.set_name("combined point")

        self.verbose = 0

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
        self.assertIsInstance(self.func12, SmoothStronglyConvexQuadraticFunction)
        self.assertIsInstance(self.operator1, CocoerciveOperator)
        self.assertIsInstance(self.operator2, CocoerciveStronglyMonotoneOperator)
        self.assertIsInstance(self.operator3, LipschitzOperator)
        self.assertIsInstance(self.operator4, LipschitzStronglyMonotoneOperator)
        self.assertIsInstance(self.operator5, MonotoneOperator)
        self.assertIsInstance(self.operator6, NegativelyComonotoneOperator)
        self.assertIsInstance(self.operator7, StronglyMonotoneOperator)
        self.assertIsInstance(self.operator8, NonexpansiveOperator)
        self.assertIsInstance(self.operator9, LinearOperator)
        self.assertIsInstance(self.operator10, SymmetricLinearOperator)
        self.assertIsInstance(self.operator11, SkewSymmetricLinearOperator)

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

        # Call adjunct of linear operator
        self.operator9.T.gradient(self.new_point)

        # Call stationary point on all functions
        for function in self.all_functions_and_operators:
            function.stationary_point()

        # Set the number of points each function has been evaluated on.
        # Note self.new_operator has been evaluated on 1 point less than the other functions.
        num_points_eval = len(self.all_points) + 3

        # Add function class constraints
        for function in self.all_functions_and_operators:
            function.set_class_constraints()

        # Verify the number of points stored in list_of_points attributes
        for function in self.all_functions_and_operators:
            self.assertEqual(len(function.list_of_stationary_points), 1)
            self.assertEqual(len(function.list_of_points), num_points_eval)
        self.assertEqual(len(self.new_operator.list_of_points), num_points_eval - 1)
        self.assertEqual(len(self.new_operator.list_of_stationary_points), 1)

        # Verify the number of class constraints is coherent with the function classes
        self.assertEqual(len(self.func1.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.func2.list_of_class_constraints), num_points_eval ** 2)
        self.assertEqual(len(self.func3.list_of_class_constraints), num_points_eval ** 2)
        self.assertEqual(len(self.func4.list_of_class_constraints), num_points_eval * (num_points_eval - 1)
                         + (num_points_eval - 1))
        self.assertEqual(len(self.func5.list_of_class_constraints), num_points_eval ** 2)
        self.assertEqual(len(self.func6.list_of_class_constraints), 2 * (num_points_eval - 1))
        self.assertEqual(len(self.func7.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.func8.list_of_class_constraints), num_points_eval ** 2)
        self.assertEqual(len(self.func9.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.func10.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.func11.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.func12.list_of_class_constraints), num_points_eval * (num_points_eval + 1) / 2)
        self.assertEqual(len(self.operator1.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.operator2.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.operator3.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.operator4.list_of_class_constraints), num_points_eval * (num_points_eval - 1))
        self.assertEqual(len(self.operator5.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.operator6.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.operator7.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.operator8.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.operator9.list_of_class_constraints), num_points_eval)
        self.assertEqual(len(self.operator10.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.operator11.list_of_class_constraints), num_points_eval * (num_points_eval - 1) / 2)
        self.assertEqual(len(self.new_operator.list_of_class_constraints), 0)

    def test_name(self):

        all_names = ["f{}".format(i) for i in range(1, 13)] + ["op{}".format(i) for i in range(1, 12)]
        for function, name in zip(self.all_functions_and_operators, all_names):
            self.assertEqual(function.get_name(), name)

    def test_tables_of_constraints(self):

        # Artificially evaluate at least one point on all the functions but self.func7
        new_operator = self.new_operator + self.operator9.T
        del new_operator.decomposition_dict[self.func7]
        new_operator.gradient(self.new_point)

        # Define and solve a PEP that depends only on self.func7
        self.pep.set_initial_condition((self.point1 - self.point2)**2 <= 1)
        self.pep.set_performance_metric((self.func7.gradient(self.point1) - self.func7.gradient(self.point2))**2)
        self.pep.solve(verbose=self.verbose)

        # Verify that all the functions have still been taken into account
        # and that their tables of constraints are as expected
        for function in self.all_functions_and_operators:
            tables_of_constraints = function.tables_of_constraints
            for table in tables_of_constraints.values():
                self.assertIsInstance(table, pandas.DataFrame)
                self.assertNotEqual(table.shape, (0,))

        # Make further test on self.func7 table of class constraints
        tables_of_duals = self.func7.get_class_constraints_duals()
        self.assertEqual(len(tables_of_duals), 1)
        table_of_duals = tables_of_duals["smoothness_convexity"]
        self.assertEqual(table_of_duals.columns.name, "IC_f7")
        self.assertEqual(table_of_duals.columns[0], self.point1.get_name())
        self.assertEqual(table_of_duals.columns[1], self.point2.get_name())
        self.assertLessEqual(np.sum(np.array(table_of_duals) - np.array([[0, 2], [2, 0]]))**2, 10**-4)
