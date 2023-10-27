import unittest

from PEPit import PEP
from PEPit.point import Point
from PEPit.function import Function
from PEPit.functions.block_smooth_convex_function import BlockSmoothConvexFunction
from PEPit.functions.smooth_convex_function import SmoothConvexFunction


class TestConstraintsBlockSmoothConvex(unittest.TestCase):

    def setUp(self):
        self.pep = PEP()

        self.L1 = 1.
        self.L2 = [1.]
        self.L3 = [1., 2., 10.]

        self.partition2 = self.pep.declare_block_partition(d=1)
        self.partition3 = self.pep.declare_block_partition(d=3)

        self.func1 = SmoothConvexFunction(L=self.L1)
        self.func2 = BlockSmoothConvexFunction(L=self.L2, partition=self.partition2)
        self.func3 = BlockSmoothConvexFunction(L=self.L3, partition=self.partition3)

        self.point1 = Point(is_leaf=True, decomposition_dict=None)
        self.point2 = Point(is_leaf=True, decomposition_dict=None)
        self.point3 = Point(is_leaf=True, decomposition_dict=None)

        self.func1.gradient(self.point1)
        self.func1.gradient(self.point2)
        self.func1.gradient(self.point3)
        self.func2.gradient(self.point1)
        self.func2.gradient(self.point2)
        self.func2.gradient(self.point3)
        self.func3.gradient(self.point1)
        self.func3.gradient(self.point2)
        self.func3.gradient(self.point3)

    def test_is_instance(self):
        self.assertIsInstance(self.func1, Function)
        self.assertIsInstance(self.func2, Function)
        self.assertIsInstance(self.func3, Function)
        self.assertIsInstance(self.func3, BlockSmoothConvexFunction)
        self.assertIsInstance(self.func2, BlockSmoothConvexFunction)
        self.assertIsInstance(self.func1, SmoothConvexFunction)

    def test_sizes(self):
        self.assertEqual(len(self.func2.L), 1)
        self.assertEqual(len(self.func3.L), 3)

    def test_interpolation_numbers(self):
        self.func1.set_class_constraints()
        self.func2.set_class_constraints()
        self.func3.set_class_constraints()

        self.assertEqual(len(self.func1.list_of_class_constraints), 6)
        self.assertEqual(len(self.func2.list_of_class_constraints), 6)
        self.assertEqual(len(self.func3.list_of_class_constraints), 18)
