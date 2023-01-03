import unittest

from PEPit.pep import PEP
from PEPit.block_partition import BlockPartition
from PEPit.point import Point


class TestExpression(unittest.TestCase):

    def setUp(self):
        self.problem = PEP()

        self.point1 = Point()
        self.point2 = Point()
        self.point3 = Point()

        self.partition0 = BlockPartition(d=2)
        self.partition1 = BlockPartition(d=2)
        self.partition2 = self.problem.declare_block_partition(d=5)
        self.partition3 = self.problem.declare_block_partition(d=1)

        self.partition0.get_block(self.point1, 0)

        self.partition1.get_block(self.point1, 0)
        self.partition1.get_block(self.point2, 1)
        self.partition1.get_block(self.point3, 1)

        self.partition2.get_block(self.point1, 0)
        self.partition2.get_block(self.point1, 2)
        self.partition2.get_block(self.point2, 4)
        self.partition2.get_block(self.point2, 2)

        self.partition0.add_partition_constraints()
        self.partition1.add_partition_constraints()
        self.partition2.add_partition_constraints()

    def test_instances(self):
        self.assertIsInstance(self.partition0, BlockPartition)

    def test_counter(self):
        self.assertIs(BlockPartition.counter, 4)

    def test_list_size(self):
        self.assertIs(len(BlockPartition.list_of_partitions), 4)

    def test_list_elements(self):
        self.assertIs(BlockPartition.list_of_partitions[0], self.partition0)
        self.assertIs(BlockPartition.list_of_partitions[1], self.partition1)
        self.assertIs(BlockPartition.list_of_partitions[2], self.partition2)

    def test_same_blocks(self):
        pt1 = self.partition1.get_block(self.point1, 0)
        pt2 = self.partition1.get_block(self.point1, 0)
        self.assertIsInstance(pt1, Point)
        self.assertIsInstance(pt2, Point)
        self.assertEqual(pt1.decomposition_dict, pt2.decomposition_dict)
        self.assertEqual(self.partition1.get_block(self.point1, 1), self.partition1.get_block(self.point1, 1))
        self.assertEqual(self.partition1.get_block(self.point2, 0), self.partition1.get_block(self.point2, 0))
        self.assertEqual(self.partition1.get_block(self.point2, 1), self.partition1.get_block(self.point2, 1))

    def test_no_partition(self):
        self.assertEqual(self.partition3.get_block(self.point1, 0).decomposition_dict, self.point1.decomposition_dict)
        self.assertEqual(self.partition3.get_block(self.point2, 0).decomposition_dict, self.point2.decomposition_dict)
        self.assertEqual(self.partition3.get_block(self.point3, 0).decomposition_dict, self.point3.decomposition_dict)

    def test_sizes(self):
        self.assertIs(self.partition1.get_nb_blocks(), 2)
        self.assertIs(self.partition2.get_nb_blocks(), 5)

    def test_number_of_points(self):
        self.assertIs(len(self.partition1.blocks_dict), 3)
        self.assertIs(len(self.partition2.blocks_dict), 2)

    def test_constraints(self):
        self.assertIs(len(self.partition0.list_of_constraints), 1)
        self.assertIs(len(self.partition1.list_of_constraints), 9)

    def test_decompose(self):
        pt1 = self.partition1.get_block(self.point1, 0) + self.partition1.get_block(self.point1, 1)
        pt2 = self.point1
        self.assertIs(pt1.decomposition_dict[self.point1], pt2.decomposition_dict[self.point1])
