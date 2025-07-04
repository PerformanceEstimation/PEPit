from PEPit.function import Function
from PEPit import PSDMatrix
from PEPit.block_partition import BlockPartition
from PEPit.expression import Expression
import numpy as np

class Refined_BlockSmoothConvexFunction(Function):
    def __init__(self,
                 partition,
                 L,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
    """
    The :class:`Refined_BlockSmoothConvexFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    by implementing necessary constraints for interpolation of the class of smooth convex functions by blocks.
    The implemented constraint is that of [2, Section 3.1].

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
    
    `[2] A. Rubbens, J.M. Hendrickx, A. Taylor (2025).
    A constructive approach to strengthen algebraic descriptions of function and operator classes.
    <https://arxiv.org/pdf/2504.14377.pdf>`_

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

        
    def last_call_before_problem_formulation(self):
        """
        Last call before modeling and solving the full PEP. Add necessarily intermediate variable before solving.
        
        """
        nb_pts = len(self.list_of_points)
        preallocate = nb_pts * (nb_pts**2  - 1) * (self.partition.get_nb_blocks()**2)
        self.s = np.ndarray((preallocate,),dtype=Expression)
        for i in range(preallocate):
            self.s[i] = Expression()
    
    def add_class_constraints(self):
        """
        Formulates the list of necessary constraints for interpolation for self (block smooth convex function);
        see [2, Proposition 3.9].

        """
        # Set function ID
        function_id = self.get_name()
        if function_id is None:
            function_id = "Function_{}".format(self.counter)

        # Set tables_of_constraints attributes
        for m in range(self.partition.get_nb_blocks()):
            for l in range(self.partition.get_nb_blocks()):
                self.tables_of_constraints["smoothness_convexity_block_{}{}".format(m, l)] = [[[]]]*len(self.list_of_points)

        # Browse list of points and create interpolation constraints
        counter = 0
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
                
                for k, point_k in enumerate(self.list_of_points):
                    xk, gk, fk = point_k
                    xk_id = xk.get_name()
                    if xk_id is None:
                        xk_id = "Point_{}".format(k)
                        
                    if not (point_i == point_j and point_i == point_k):
                        for m in range(self.partition.get_nb_blocks()):
                            for l in range(self.partition.get_nb_blocks()):
                                # partial gradients for block m
                                gim = self.partition.get_block(gi, m)
                                gjm = self.partition.get_block(gj, m)
                                gkm = self.partition.get_block(gk, m)
                                
                                # partial gradients for block l
                                gjl = self.partition.get_block(gj, l)
                                gkl = self.partition.get_block(gk, l)
                                
                                # Necessary conditions for interpolation
                                constraint = ( self.s[counter] >= 0)
                                A = -(-fi + fk + gk * (xi - xk) + 1 / (2 * self.L[m]) * (gim - gkm) ** 2)
                                B = -(-fi + fj + gj * (xi - xj) + 1 / (2 * self.L[m]) * (gim - gjm) ** 2)
                                C = -( 1 / (2 * self.L[l]) * (gjl - gkl) ** 2 - 1 / (2 * self.L[m]) * (gjm - gkm) ** 2)
                                
                                T = np.array([[A, (A+B+C)/2-self.s[counter]], [(A+B+C)/2-self.s[counter], B]], dtype=Expression)
                                psd_matrix = PSDMatrix(matrix_of_expressions=T)
                                self.list_of_class_psd.append(psd_matrix)
                                
                                constraint.set_name("IC_{}_smoothness_convexity_block_{}{}({}, {}, {})".format(function_id, m, l, xi_id, xj_id, xk_id))
                                self.tables_of_constraints["smoothness_convexity_block_{}{}".format(m, l)][i].append(constraint)
                                self.list_of_class_constraints.append(constraint)
                                counter += 1 
