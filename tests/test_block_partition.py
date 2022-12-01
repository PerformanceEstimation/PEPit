import unittest

from PEPit.block_partition import Block_partition
from PEPit.point import Point
from PEPit.constraint import Constraint


class TestExpression(unittest.TestCase):

    def setUp(self):

        self.point1 = Point()
        self.point2 = Point()
        self.point3 = Point()

        self.partition0 = Block_partition(d=2)
        self.partition1 = Block_partition(d=2)
        self.partition2 = Block_partition(d=5)
        
        self.partition0.get_block(self.point1,0)
        
        self.partition1.get_block(self.point1,0)
        self.partition1.get_block(self.point2,1)
        self.partition1.get_block(self.point3,1)
        
        self.partition2.get_block(self.point1,0)
        self.partition2.get_block(self.point1,2)
        self.partition2.get_block(self.point2,4)
        self.partition2.get_block(self.point2,2)
        
        self.partition0.add_partition_constraints()
        self.partition1.add_partition_constraints()
        self.partition2.add_partition_constraints()
        

    def test_sizes(self):
    
        self.assertIs(self.partition1.get_nb_blocks(), 2)
        self.assertIs(self.partition2.get_nb_blocks(), 5)

    def test_number_of_points(self):
    
        self.assertIs(len(self.partition1.list_of_points), 3) 
        self.assertIs(len(self.partition2.list_of_points), 2)
        
    def test_constraints(self):
    
        self.assertIs(len(self.partition0.list_of_constraints), 1) 
        self.assertIs(len(self.partition1.list_of_constraints), 9)  
    
    def test_decompose(self):
        pt1 = self.partition1.get_block(self.point1,0)+self.partition1.get_block(self.point1,1)
        pt2 = self.point1
        self.assertIs(pt1.decomposition_dict[self.point1],pt2.decomposition_dict[self.point1])
        
    def tearDown(self):
        Point.counter = 0
        Constraint.counter = 0
