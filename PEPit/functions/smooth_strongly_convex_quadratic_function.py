import numpy as np
from PEPit.function import Function
from PEPit import Expression
from PEPit import Point
from PEPit import PSDMatrix


class SmoothStronglyConvexQuadraticFunction(Function):
    """
    The :class:`SmoothStronglyConvexQuadraticFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    by implementing interpolation constraints of the class of smooth strongly convex quadratic functions.

    Attributes:
        mu (float): strong convexity parameter
        L (float): smoothness parameter

    Smooth strongly convex quadratic functions are characterized by parameters :math:`\\mu` and `L`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import SmoothStronglyConvexQuadraticFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=SmoothStronglyConvexQuadraticFunction, mu=.1, L=1.)

    References:

    `[1] N. Bousselmi, J. Hendrickx, F. Glineur  (2023).
    Interpolation Conditions for Linear Operators and applications to Performance Estimation Problems.
    arXiv preprint.
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
            mu (float): The strong convexity parameter.
            L (float): The smoothness parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Smooth strongly convex quadratic functions are necessarily differentiable,
            hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store mu and L
        self.mu = mu
        self.L = L
        self.b = Point()
        self.c = Expression()

    def set_value_constraint_i(self,
                               xi, gi, fi):
        """
        Set the value of the function.

        """
        # Value constraint
        constraint = (fi == 0.5 * xi * (gi-self.b) + self.b * xi + self.c )

        return constraint

    def set_symmetry_constraint_i_j(self,
                                    xi, gi, fi,
                                    xj, gj, fj,
                                    ):
        """
        Ensure the Hessian is symmetric.
        """

        # Symmetry constraint
        constraint = ( xi * (gj-self.b) == xj * (gi-self.b) )

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (smooth strongly convex quadratic function);
        see [1, Theorem 3.9].
        """

        # Add the quadratic interpolation constraint
        self.add_constraints_from_one_list_of_points(list_of_points=self.list_of_points,
                                                     constraint_name="value",
                                                     set_class_constraint_i=self.set_value_constraint_i,
                                                     )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="symmetry",
                                                      set_class_constraint_i_j=self.set_symmetry_constraint_i_j,
                                                      symmetry=True,
                                                      )


        # Create a PSD matrix to enforce the smoothness and strong convexity
        N = len(self.list_of_points)
        T = np.empty((N, N), dtype=Expression)

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):
                xj, gj, fj = point_j

                T[i, j] = (self.L + self.mu) * 1/2 * ( (gi-self.b) * xj + (gj-self.b) * xi ) - (gi-self.b) * (gj-self.b) - self.mu * self.L * xi * xj

        psd_matrix = PSDMatrix(matrix_of_expressions=T)
        self.list_of_class_psd.append(psd_matrix)
