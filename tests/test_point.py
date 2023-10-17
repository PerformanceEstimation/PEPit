import unittest

from PEPit import PEP
from PEPit.point import Point
from PEPit.expression import Expression


class TestPoint(unittest.TestCase):

    def setUp(self):
        self.pep = PEP()

        self.A = Point(is_leaf=True, decomposition_dict=None, name="pointA")
        self.B = Point(is_leaf=True, decomposition_dict=None)

    def test_is_instance(self):

        self.assertIsInstance(self.A, Point)
        self.assertIsInstance(self.B, Point)

    def test_name(self):

        self.assertEqual(self.A.get_name(), "pointA")

        self.assertIsNone(self.B.get_name())
        self.B.set_name("pointB")
        self.assertEqual(self.B.get_name(), "pointB")

        C = self.A + self.B
        self.assertIsNone(C.get_name())
        C.set_name("pointC")
        self.assertEqual(C.get_name(), "pointC")

    def test_counter(self):

        C = self.A + self.B
        self.assertIs(self.A.counter, 0)
        self.assertIs(self.B.counter, 1)
        self.assertIs(C.counter, None)
        self.assertIs(Point.counter, 2)

        D = Point(is_leaf=True, decomposition_dict=None)
        self.assertIs(D.counter, 2)
        self.assertIs(Point.counter, 3)

    def test_linear_combination(self):

        new_point = - self.A * 1. + 2 * self.B - self.B / 5

        self.assertIsInstance(new_point, Point)
        self.assertEqual(new_point.decomposition_dict, {self.A: -1, self.B: 9 / 5})

    def test_rmul_between_two_points(self):

        inner_product = self.A * self.B

        self.assertIsInstance(inner_product, Expression)
        self.assertFalse(inner_product._is_leaf)
        self.assertEqual(inner_product.decomposition_dict, {(self.A, self.B): 1})

    def test_pow(self):

        norm_square = (self.A - self.B) ** 2

        self.assertIsInstance(norm_square, Expression)
        self.assertFalse(norm_square._is_leaf)
        self.assertEqual(norm_square.decomposition_dict, {(self.A, self.A): 1,
                                                          (self.A, self.B): -1,
                                                          (self.B, self.A): -1,
                                                          (self.B, self.B): 1
                                                          })
