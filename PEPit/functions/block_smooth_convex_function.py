import numpy as np

from PEPit.function import Function
from PEPit.block_partition import BlockPartition


class BlockSmoothConvexFunction(Function):
    """
    The :class:`BlockSmoothConvexFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    by implementing necessary constraints for interpolation of the class of smooth convex functions by blocks.

    Attributes:
        partition (BlockPartition): partitioning of the variables (in blocks).
        L (list): smoothness parameters (one per block).
        
    Smooth convex functions by blocks are characterized by a list of parameters :math:`L_i` (one per block),
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import BlockSmoothConvexFunction
        >>> problem = PEP()
        >>> partition = problem.declare_block_partition(d=3)
        >>> L = [1, 4, 10]
        >>> func = problem.declare_function(function_class=BlockSmoothConvexFunction, partition=partition, L=L)

    References:

    `[1] Z. Shi, R. Liu (2016).
    Better worst-case complexity analysis of the block coordinate descent method for large scale machine learning.
    In 2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA).
    <https://arxiv.org/pdf/1608.04826.pdf>`_

    """

    def __init__(self,
                 partition,
                 L,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
            partition (BlockPartition): a :class:`BlockPartition`.
            L (list): smoothness parameters (list of floats).
                      The size of the list must be equal to the number of blocks of the partition.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Smooth convex functions by blocks are necessarily differentiable, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store partition and L
        assert isinstance(partition, BlockPartition)
        if partition.get_nb_blocks() > 1:
            assert isinstance(L, list)
            assert len(L) == partition.get_nb_blocks()
            for Li in L:
                assert Li < np.inf

        self.partition = partition
        self.L = L

    def add_class_constraints(self):
        """
        Formulates the list of necessary constraints for interpolation for self (block smooth convex function);
        see [1, Lemma 1.1].

        """
        # Set function ID
        function_id = self.get_name()
        if function_id is None:
            function_id = "Function_{}".format(self.counter)

        # Set tables_of_constraints attributes
        for k in range(self.partition.get_nb_blocks()):
            self.tables_of_constraints["smoothness_convexity_block_{}".format(k)] = [[]]*len(self.list_of_points)

        # Browse list of points and create interpolation constraints
        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i
            xi_id = xi.get_name()
            if xi_id is None:
                xi_id = "Point_{}".format(i)

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j
                xj_id = xj.get_name()
                if xj_id is None:
                    xj_id = "Point_{}".format(j)

                if point_i == point_j:
                    for k in range(self.partition.get_nb_blocks()):
                        self.tables_of_constraints["smoothness_convexity_block_{}".format(k)][i].append(0)

                else:

                    for k in range(self.partition.get_nb_blocks()):

                        # partial gradients for block k
                        gik = self.partition.get_block(gi, k)
                        gjk = self.partition.get_block(gj, k)
                        
                        # Necessary conditions for interpolation
                        constraint = (fi - fj >= gj * (xi - xj) + 1 / (2 * self.L[k]) * (gik - gjk) ** 2)
                        constraint.set_name("IC_{}_smoothness_convexity_block_{}({}, {})".format(function_id, k,
                                                                                                 xi_id, xj_id))
                        self.tables_of_constraints["smoothness_convexity_block_{}".format(k)][i].append(constraint)
                        self.list_of_class_constraints.append(constraint)
