import unittest

from PEPit import PEP
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.constraint import Constraint


class TestExpression(unittest.TestCase):

    def setUp(self):
        self.pep = PEP()

        self.point1 = Point(is_leaf=True, decomposition_dict=None)
        self.point2 = Point(is_leaf=True, decomposition_dict=None)

        self.inner_product = self.point1 * self.point2
        self.function_value = Expression(is_leaf=True, decomposition_dict=None, name="fx")

    def test_is_instance(self):

        self.assertIsInstance(self.inner_product, Expression)
        self.assertIsInstance(self.function_value, Expression)

    def test_name(self):

        self.assertIsNone(self.inner_product.get_name())

        self.inner_product.set_name("x1*x2")

        self.assertEqual(self.inner_product.get_name(), "x1*x2")
        self.assertEqual(self.function_value.get_name(), "fx")

    def test_counter(self):

        composite_expression = self.inner_product + self.function_value
        self.assertIs(self.inner_product.counter, None)
        self.assertIs(self.function_value.counter, 0)
        self.assertIs(composite_expression.counter, None)
        self.assertIs(Expression.counter, 1)

        new_expression = Expression(is_leaf=True, decomposition_dict=None)
        self.assertIs(new_expression.counter, 1)
        self.assertIs(Expression.counter, 2)

    def test_linear_combination(self):

        new_expression = 1 + 2 * (4 - (- self.inner_product * 3) - 5
                                  + 2 * self.function_value - self.function_value / 5 + 2)

        self.assertIsInstance(new_expression, Expression)
        self.assertEqual(new_expression.decomposition_dict, {1: 3,
                                                             (self.point1, self.point2): 6,
                                                             self.function_value: 18 / 5})

    def test_constraint(self):

        constraint = self.inner_product <= self.function_value

        self.assertIsInstance(constraint, Constraint)
        self.assertEqual(constraint.expression.decomposition_dict, (self.inner_product - self.function_value).decomposition_dict)
