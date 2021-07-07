
import unittest

from PEPit.point import Point
from PEPit.expression import Expression


class TestPoint(unittest.TestCase):

    def setUp(self):

        self.point1 = Point(is_leaf=True, decomposition_dict=None)
        self.point2 = Point(is_leaf=True, decomposition_dict=None)

        self.inner_product = self.point1 * self.point2
        self.function_value = Expression(is_function_value=True, decomposition_dict=None)

    def test_is_instance(self):

        self.assertIsInstance(self.inner_product, Expression)
        self.assertIsInstance(self.function_value, Expression)

    def test_counter(self):

        composite_expression = self.inner_product + self.function_value
        self.assertIs(self.inner_product.counter, None)
        self.assertIs(self.function_value.counter, 0)
        self.assertIs(composite_expression.counter, None)
        self.assertIs(Expression.counter, 1)

        new_expression = Expression(is_function_value=True, decomposition_dict=None)
        self.assertIs(new_expression.counter, 1)
        self.assertIs(Expression.counter, 2)

    def test_linear_combination(self):

        new_expression = - self.inner_product + 2 * self.function_value - self.function_value / 5

        self.assertIsInstance(new_expression, Expression)
        self.assertEqual(new_expression.decomposition_dict, {(self.point1, self.point2): -1,
                                                             self.function_value: 9 / 5})

    def test_constraint(self):

        constraint = self.inner_product <= self.function_value

        self.assertIsInstance(constraint, Expression)
        self.assertEqual(constraint.decomposition_dict, (self.inner_product - self.function_value).decomposition_dict)

    def tearDown(self):

        Point.counter = 0
        Expression.counter = 0
