from PEPit.point import Point, null_point
from PEPit.constraint import Constraint


class BlockPartition(object):
    """
    A :class:`BlockPartition` encodes an abstract block partitioning
    (of :math:`d` blocks of variables) of the ambient space.
    
    Attributes:
        blocks_dict (dict): dictionary describing the decomposition of :class:`Point` objects.
                            Keys are :class:`Point` objects.
                            Values are lists of :class:`Point` objects corresponding to the decompositions of the key
                            in blocks.
        list_of_constraints (list): The list of :class:`Constraint` objects associated with this
                                    :class:`BlockPartition`.
        d (int): encodes the number of blocks (:math:`d \\geq 1`).
        counter (int): counts the number of :class:`BlockPartition` objects.

    Example:
        >>> from PEPit import PEP
        >>> pep = PEP()
        >>> block_partition = pep.declare_block_partition(d=5)

    """
    # Class counter.
    # It counts the number of partitions defined from scratch.
    counter = 0
    list_of_partitions = list()

    def __init__(self, d):
        """
        :class:`BlockPartition` objects can also be instantiated via the following arguments

        Args:
            d (int): encodes the number of blocks (:math:`d \\geq 1`).

        Instantiating the :class:`BlockPartition` object of the first example can be done by

        Raises:
            AssertionError: if provided :math:`d` is not a positive integer.

        Example:
            >>> block_partition = BlockPartition(d=5)

        """
        # Verify that d is not 0
        assert isinstance(d, int) and d >= 1

        # Store attributes
        self.d = d
        self.list_of_constraints = list()
        self.blocks_dict = dict()
        self.counter = BlockPartition.counter

        # Update class counter and list_of_partitions
        BlockPartition.counter += 1
        BlockPartition.list_of_partitions.append(self)

    def get_nb_blocks(self):
        """
        Return the number of blocks of this partition.

        """
        return self.d

    def get_block(self, point, block_number):
        """
        Decompose a :class:`Point` as a sum of into :math:`d` orthogonal :class:`Point` according to the partitioning.

        Args:
            point (Point): any :class:`Point` object.
            block_number (int): an integer between 0 and the number of blocks
                                (corresponding to the block index, between 0 and d-1)

        Returns:
            point (Point): a :class:`Point` corresponding to the projection of `point`
                           onto the block `block_number` of the partitioning.

        Raises:
            AssertionError: if provided `point` is not a :class:`Point` or `block_number` is not a valid integer
                            (which should be nonnegative and strictly smaller than the number of blocks)

        """
        assert isinstance(point, Point) and isinstance(block_number, int)
        assert 0 <= block_number <= self.d - 1

        # Case 1: point is already in the list: do nothing (just return)
        # Case 2: point is not partitioned yet: partition (and return)
        if point not in self.blocks_dict.keys():
            point_partition = list()
            accumulation = null_point
            # Fill the partition with d-1 new :class:`Point`.
            for i in range(self.d-1):
                new_point = Point()
                accumulation += new_point
                point_partition.append(new_point)
            # The last element is set so that the sum is equal to point.
            point_partition.append(point - accumulation)
            self.blocks_dict[point] = point_partition
        # Return the desired block projection of point
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
        Formulate the list of orthogonality constraints induced by the partitioning.

        """
        for xi_decomposed in self.blocks_dict.values():
            for xj_decomposed in self.blocks_dict.values():
                for k in range(self.d):
                    for l in range(k):
                        self.add_constraint(xi_decomposed[k] * xj_decomposed[l] == 0)
