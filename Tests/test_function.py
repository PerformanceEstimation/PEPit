
import unittest

from PEPit.Tools.dict_operations import prune_dict

from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.function import Function
from PEPit.Function_classes.convex_function import ConvexFunction


class TestFunction(unittest.TestCase):

    def setUp(self):

        self.func1 = Function(is_leaf=True, decomposition_dict=None)
        self.func2 = ConvexFunction(dict(), is_leaf=True, decomposition_dict=None)

        self.point = Point(is_leaf=True, decomposition_dict=None)

    def test_is_instance(self):

        self.assertIsInstance(self.func1, Function)
        self.assertIsInstance(self.func2, Function)

    def test_counter(self):

        composite_function = self.func1 + self.func2
        self.assertIs(self.func1.counter, 0)
        self.assertIs(self.func2.counter, 1)
        self.assertIs(composite_function.counter, None)
        self.assertIs(Function.counter, 2)

        new_function = Function(is_leaf=True, decomposition_dict=None)
        self.assertIs(new_function.counter, 2)
        self.assertIs(Function.counter, 3)

    def compute_linear_combination(self):

        new_function = - self.func1 + 2 * self.func2 - self.func2 / 5

        return new_function

    def test_linear_combination(self):

        new_function = self.compute_linear_combination()

        self.assertIsInstance(new_function, Function)
        self.assertEqual(new_function.decomposition_dict, {self.func1: -1, self.func2: 9 / 5})

    def test_oracle(self):

        new_function = self.compute_linear_combination()
        new_function.oracle(point=self.point)

        # On new_function
        self.assertEqual(len(new_function.list_of_points), 1)

        point, grad, val = new_function.list_of_points[0]
        self.assertIsInstance(point, Point)
        self.assertIsInstance(grad, Point)
        self.assertIsInstance(val, Expression)

        self.assertIs(point, self.point)
        self.assertTrue(grad._is_leaf)
        self.assertTrue(val._is_function_value)

        # On func1
        self.assertEqual(len(self.func1.list_of_points), 1)

        point1, grad1, val1 = self.func1.list_of_points[0]
        self.assertIsInstance(point1, Point)
        self.assertIsInstance(grad1, Point)
        self.assertIsInstance(val1, Expression)

        self.assertTrue(grad1._is_leaf)
        self.assertTrue(val1._is_function_value)

        # On func2
        self.assertEqual(len(self.func2.list_of_points), 1)

        point2, grad2, val2 = self.func2.list_of_points[0]
        self.assertIsInstance(point2, Point)
        self.assertIsInstance(grad2, Point)
        self.assertIsInstance(val2, Expression)

        self.assertFalse(grad2._is_leaf)
        self.assertFalse(val2._is_function_value)

        # Combination
        self.assertIs(point1, self.point)
        self.assertIs(point2, self.point)

        self.assertEqual((-grad1 + 9 * grad2 / 5).decomposition_dict, grad.decomposition_dict)
        self.assertEqual(prune_dict((-val1 + 9 * val2 / 5).decomposition_dict), val.decomposition_dict)

    def test_oracle_with_predetermined_values(self):

        new_function = self.compute_linear_combination()
        new_function.oracle(point=self.point)

        grad1, val1 = self.func1.oracle(point=self.point)
        grad2, val2 = self.func2.oracle(point=self.point)

        grad, val = new_function.oracle(point=self.point)

        self.assertEqual(prune_dict(val.decomposition_dict), prune_dict((-val1 + 9/5*val2).decomposition_dict))
        print(self.func1.list_of_points)
        self.assertNotEqual(prune_dict(grad.decomposition_dict), prune_dict((-grad1 + 9/5*grad2).decomposition_dict))
        # self.assertEqual(len(self.func1.list_of_points), 2)
        # self.assertEqual(len(self.func2.list_of_points), 2)
        other_grad1, other_val1 = self.func1.list_of_points[2][1:]
        other_grad2, other_val2 = self.func2.list_of_points[2][1:]
        self.assertEqual(val1.decomposition_dict, other_val1.decomposition_dict)
        self.assertEqual(val2.decomposition_dict, other_val2.decomposition_dict)
        self.assertEqual(prune_dict(grad.decomposition_dict), prune_dict((-other_grad1 + 9/5*other_grad2).decomposition_dict))

    def test_optimal_point(self):

        new_function = self.compute_linear_combination()
        new_function.optimal_point()

        # On new_function
        self.assertEqual(len(new_function.list_of_points), 1)

        point, grad, val = new_function.list_of_points[0]
        self.assertIsInstance(point, Point)
        self.assertIsInstance(grad, Point)
        self.assertIsInstance(val, Expression)

        self.assertTrue(point._is_leaf)
        self.assertFalse(grad._is_leaf)
        self.assertTrue(val._is_function_value)

        self.assertEqual(grad.decomposition_dict, dict())
        self.assertEqual((grad ** 2).decomposition_dict, dict())

        # On func1
        self.assertEqual(len(self.func1.list_of_points), 1)

        point1, grad1, val1 = self.func1.list_of_points[0]
        self.assertIsInstance(point1, Point)
        self.assertIsInstance(grad1, Point)
        self.assertIsInstance(val1, Expression)

        self.assertTrue(grad1._is_leaf)
        self.assertTrue(val1._is_function_value)

        # On func2
        self.assertEqual(len(self.func2.list_of_points), 1)

        point2, grad2, val2 = self.func2.list_of_points[0]
        self.assertIsInstance(point2, Point)
        self.assertIsInstance(grad2, Point)
        self.assertIsInstance(val2, Expression)

        self.assertFalse(grad2._is_leaf)
        self.assertFalse(val2._is_function_value)

        # Combination
        self.assertIs(point1, point)
        self.assertIs(point2, point)

        self.assertEqual((-grad1 + 9 * grad2 / 5).decomposition_dict, grad.decomposition_dict)
        self.assertEqual(prune_dict((-val1 + 9 * val2 / 5).decomposition_dict), val.decomposition_dict)

    def tearDown(self):

        Point.counter = 0
        Expression.counter = 0
        Function.counter = 0
