import numpy as np

from PEPit import Function
from PEPit import Expression
from PEPit import PSDMatrix


class SkewSymmetricLinearOperator(Function):
    """
    The :class:`SkewSymmetricLinearOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints for the class of skew-symmetric linear operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        L (float): singular values upper bound

    Skew-Symmetric Linear operators are characterized by parameters :math:`L`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import SkewSymmetricLinearOperator
        >>> problem = PEP()
        >>> M = problem.declare_function(function_class=SkewSymmetricLinearOperator, L=1.)

    References:

    `[1] N. Bousselmi, J. Hendrickx, F. Glineur (2023).
    Interpolation Conditions for Linear Operators and applications to Performance Estimation Problems.
    arXiv preprint
    <https://arxiv.org/pdf/2302.08781.pdf>`_

    """

    def __init__(self,
                 L,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
            L (float): The singular values upper bound.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Skew-Symmetric Linear operators are necessarily continuous,
            hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store L
        self.L = L

    @staticmethod
    def set_antisymmetric_linear_constraint_i_j(xi, gi, fi,
                                                xj, gj, fj,
                                                ):
        """
        Formulates the list of interpolation constraints for self (Skew-symmetric linear operator).
        """
        # Interpolation conditions of symmetric linear operators class
        constraint = (xi * gj == - xj * gi)

        return constraint

    @staticmethod
    def set_diagonal_linear_constraint_i(xi, gi, fi,
                                         ):
        """
        Formulates the list of interpolation constraints for self (Skew-symmetric linear operator).
        """
        # Interpolation conditions of symmetric linear operators class
        constraint = (xi * gi == 0)

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of necessary and sufficient conditions for interpolation of self
        (Skew-Symmetric Linear operator), see [1, Corollary 3.2].
        """

        # Add the class constraint for antisymmetric linear operators
        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="antisymmetric_linearity",
                                                      set_class_constraint_i_j=
                                                      self.set_antisymmetric_linear_constraint_i_j,
                                                      symmetry=True,
                                                      )
        self.add_constraints_from_one_list_of_points(list_of_points=self.list_of_points,
                                                     constraint_name="diagonal_linearity",
                                                     set_class_constraint_i=
                                                     self.set_diagonal_linear_constraint_i,
                                                     )

        # Create a PSD matrix to enforce the singular values to be smaller than L
        N = len(self.list_of_points)
        T = np.empty([N, N], dtype=Expression)

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                T[i, j] = - gi * gj + (self.L ** 2) * xi * xj

        psd_matrix = PSDMatrix(matrix_of_expressions=T)
        self.list_of_class_psd.append(psd_matrix)
