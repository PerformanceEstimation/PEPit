import numpy as np

from PEPit import Function
from PEPit import Expression
from PEPit import PSDMatrix


class SymmetricLinearOperator(Function):
    """
    The :class:`SymmetricLinearOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints for the class of symmetric linear operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        mu (float): eigenvalues lower bound
        L (float): eigenvalues upper bound

    Symmetric Linear operators are characterized by parameters :math:`\\mu` and `L`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import SymmetricLinearOperator
        >>> problem = PEP()
        >>> M = problem.declare_function(function_class=SymmetricLinearOperator, mu=.1, L=1.)

    References:

    `[1] N. Bousselmi, J. Hendrickx, F. Glineur (2023).
    Interpolation Conditions for Linear Operators and applications to Performance Estimation Problems.
    arXiv preprint
    <https://arxiv.org/pdf/2302.08781.pdf>`_

    """

    def __init__(self,
                 mu,
                 L,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
            mu (float): The eigenvalues lower bound.
            L (float): The eigenvalues upper bound.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Symmetric Linear operators are necessarily continuous,
            hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store L and mu
        self.mu = mu
        self.L = L

    @staticmethod
    def set_symmetric_linear_constraint_i_j(xi, gi, fi,
                                            xj, gj, fj,
                                            ):
        """
        Formulates the list of interpolation constraints for self (Symmetric linear operator).
        """
        # Interpolation conditions of symmetric linear operators class
        constraint = (xi * gj == xj * gi)

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of necessary and sufficient conditions for interpolation of self
        (Symmetric Linear operator), see [1, Theorem 3.3].

        """
        # Add the class constraint for symmetric linear operators
        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="symmetric_linearity",
                                                      set_class_constraint_i_j=self.set_symmetric_linear_constraint_i_j,
                                                      symmetry=True,
                                                      )

        # Create a PSD matrix to enforce the eigenvalues to lie into the interval [mu, L]
        N = len(self.list_of_points)
        T = np.empty([N, N], dtype=Expression)

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                T[i, j] = self.L * gi * xj - gi * gj - self.mu * self.L * xi * xj + self.mu * xi * gj

        psd_matrix = PSDMatrix(matrix_of_expressions=T)
        self.list_of_class_psd.append(psd_matrix)
