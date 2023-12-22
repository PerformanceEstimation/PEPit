import numpy as np
from PEPit.function import Function


class SmoothStronglyConvexFunction(Function):
    """
    The :class:`SmoothStronglyConvexFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    by implementing interpolation constraints of the class of smooth strongly convex functions.

    Attributes:
        mu (float): strong convexity parameter
        L (float): smoothness parameter

    Smooth strongly convex functions are characterized by parameters :math:`\\mu` and :math:`L`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import SmoothStronglyConvexFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=SmoothStronglyConvexFunction, mu=.1, L=1.)

    References:

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
    Smooth strongly convex interpolation and exact worst-case performance of first-order methods.
    Mathematical Programming, 161(1-2), 307-345.
    <https://arxiv.org/pdf/1502.05666.pdf>`_

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
            Smooth strongly convex functions are necessarily differentiable, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store mu and L
        self.mu = mu
        self.L = L

        if self.L == np.inf:
            print("\033[96m(PEPit) Smooth strongly convex functions are necessarily differentiable. To instantiate\n"
                  "a strongly convex function, please avoid using the class SmoothStronglyConvexFunction with\n"
                  "L == np.inf. Instead, please use the class StronglyConvexFunction (which accounts for the fact\n"
                  "that there might be several sub-gradients at the same point).\033[0m")

    def set_smoothness_strong_convexity_constraint_i_j(self,
                                                       xi, gi, fi,
                                                       xj, gj, fj,
                                                       ):
        """
        Formulates the list of interpolation constraints for self (smooth strongly convex function).
        """
        # Interpolation conditions of smooth strongly convex functions class
        constraint = (fi - fj >=
                      gj * (xi - xj)
                      + 1 / (2 * self.L) * (gi - gj) ** 2
                      + self.mu / (2 * (1 - self.mu / self.L)) * (
                              xi - xj - 1 / self.L * (gi - gj)) ** 2)

        return constraint

    def add_class_constraints(self):
        """
        Add class constraints.
        """
        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="smoothness_strong_convexity",
                                                      set_class_constraint_i_j=
                                                      self.set_smoothness_strong_convexity_constraint_i_j,
                                                      )
