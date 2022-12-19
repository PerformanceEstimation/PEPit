from PEPit.point import Point
from PEPit.point import null_point
from PEPit.constraint import Constraint

class BlockPartition(object):
    """
    A :class:`BlockPartition` encodes an abstract block partitioning (of :math:`d` blocks of variables) of the ambient space.
    
    Attributes:
        blocks_dict (dict): dictionary of lists of :class:`Point` objects.
                            Keys are :class:`Point` objects. The lists corresponds to the decompositions in blocks.
        
        list_of_constraints (list): The list of :class:`Constraint` objects associated with this :class:`Function`.
        
        d (int): encodes the number of blocks (d>1).
        	
    Example:
    >>> 
    """
    # Class counter.
    # It counts the number of partitions defined from scratch.
    counter = 0
    list_of_partitions = list()
    

    def __init__(self, d):
        """
        :class:`block_decomposition`
        
        Raises:
            AssertionError: if d <= 1
        """
        assert d > 1
        self.d = d #controls that d>1 (otherwise useless and simple to create bugs)
        self.list_of_constraints = list()
        self.blocks_dict = {}
        BlockPartition.counter += 1
        BlockPartition.list_of_partitions.append(self)
        

    def get_nb_blocks(self):
        """
            Return the number of blocks of this partition.
        """
        return self.d

    def get_block(self, point, block_number):
        """
            Decomposes a :class:`Point` as a sum of into :math:`d` orthogonal :class:`Point` according
            the partitioning.
            
            Args:
                point (Point): any other :class:`Point` object.
                block_number (int): an integer between 0 and the number of blocks (corresponding to the block
                                    index, between 0 and d-1)
                
            Returns:
                point (Point): a :class:`Point` corresponding to the block `block_number` of the partition.

            Raises:
                AssertionError: if provided `point` is not a :class:`Point` or `block_number` is not a valid integer
                                (which should be strictly smaller than the number of blocks)
        """
        assert isinstance(point, Point) or isinstance(block_number, int)
        assert block_number <= self.d-1
        
        # case 1: point is already in the list: do nothing (just return)
        # case 2: point is not partitioned yet: partition (and return)
        if point not in self.blocks_dict:
            partitioned_point = list()
            accumulation = null_point
            # fill the partition with d-1 new :class:`Point`. The last element is set so that the sum is equal to point.
            for i in range(self.d-1):
                newpoint = Point()
                accumulation = accumulation + newpoint
                partitioned_point.append(newpoint)
                
            partitioned_point.append(point-accumulation)
            self.blocks_dict[point] = partitioned_point
        return self.blocks_dict[point][block_number]
              
        
    def add_constraint(self, constraint):
        """
            Store a new :class:`Constraint` to the list of constraints of this :class:`BlockPartition`.
            
            Args:
                constraint (Constraint): typically resulting from a comparison of 2 :class:`Expression` objects.
            
            Raises:
                AssertionError: if provided `constraint` is not a :class:`Constraint` object.
        """
        # Verify constraint is an actual Constraint object
        assert isinstance(constraint, Constraint)
        
        # Add constraint to the list of self's constraints
        self.list_of_constraints.append(constraint)

    def add_partition_constraints(self):
        """
            Formulates the list of orthogonality constraints for self.
        """
        for xi in self.blocks_dict:
            for xj in self.blocks_dict:
                    for k in range(self.d):
                        for l in range(self.d):
                            if k > l:
                                self.add_constraint( self.blocks_dict[xi][k] * self.blocks_dict[xj][l] == 0 )
