import numpy as np

from PEPit import Function
from PEPit import Expression
from PEPit import PSDMatrix


class LinearOperator(Function):
    """
    The :class:`LinearOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of linear operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        L (float): singular values upper bound
        T (Function): the adjunct linear operator

    Linear operators are characterized by the parameter :math:`L`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import LinearOperator
        >>> problem = PEP()
        >>> M = problem.declare_function(function_class=LinearOperator, L=1.)

    References:

    `[1] N. Bousselmi, J. Hendrickx, F. Glineur  (2023).
    Interpolation Conditions for Linear Operators and applications to Performance Estimation Problems.
    arXiv preprint.
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
            Linear operators are necessarily continuous, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store L
        self.L = L

        # Define an adjunct operator with no class constraint
        # Its list of points is what is important
        self.T = Function(is_leaf=True)
        self.T.counter = None
        Function.counter -= 1

        def no_class_constraint_for_transpose():
            pass

        self.T.add_class_constraints = no_class_constraint_for_transpose

    def add_class_constraints(self):
        """
        Formulates the list of necessary and sufficient conditions for interpolation of self
        (Linear operator), see [1, Theorem 3.1].
        """

        # Add interpolation constraints for linear operator
        for point_xy in self.list_of_points:

            xi, yi, fi = point_xy

            for point_uv in self.T.list_of_points:
                uj, vj, hj = point_uv

                # Constraint X^T V = Y^T U
                self.list_of_class_constraints.append(xi * vj == yi * uj)

        # Add constraint of singular value upper bound of self
        N1 = len(self.list_of_points)
        T1 = np.empty([N1, N1], dtype=Expression)

        for i, point_i in enumerate(self.list_of_points):

            xi, yi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):
                xj, yj, fj = point_j

                # Constraint Y^T Y <= L^2 X^T X
                T1[i, j] = (self.L ** 2) * xi * xj - yi * yj

        psd_matrix1 = PSDMatrix(matrix_of_expressions=T1)
        self.list_of_class_psd.append(psd_matrix1)

        # Add constraint of singular value upper bound of self.T
        N2 = len(self.T.list_of_points)
        T2 = np.empty([N2, N2], dtype=Expression)

        for i, point_i in enumerate(self.T.list_of_points):

            ui, vi, hi = point_i

            for j, point_j in enumerate(self.T.list_of_points):
                uj, vj, hj = point_j

                # Constraint V^T V <= L^2 U^T U
                T2[i, j] = (self.L ** 2) * ui * uj - vi * vj

        psd_matrix2 = PSDMatrix(matrix_of_expressions=T2)
        self.list_of_class_psd.append(psd_matrix2)
