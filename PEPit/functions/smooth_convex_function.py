import numpy as np
from PEPit.function import Function


class SmoothConvexFunction(Function):
    """
    The :class:`SmoothConvexFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    by implementing interpolation constraints of the class of smooth convex functions.

    Attributes:
        L (float): smoothness parameter

    Smooth convex functions are characterized by the smoothness parameter `L`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import SmoothConvexFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=SmoothConvexFunction, L=1.)

    References:

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
    Smooth strongly convex interpolation and exact worst-case performance of first-order methods.
    Mathematical Programming, 161(1-2), 307-345.
    <https://arxiv.org/pdf/1502.05666.pdf>`_

    """

    def __init__(self,
                 L,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
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
            Smooth convex functions are necessarily differentiable, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store L
        self.L = L

        if self.L == np.inf:
            print("\033[96m(PEPit) The class of smooth convex functions is necessarily differentiable.\n"
                  "To instantiate a convex function, please avoid using the class SmoothConvexFunction with \n"
                  "L == np.inf. Instead, please use the class ConvexFunction (which accounts for the fact \n"
                  "that there might be several subgradients at the same point).\033[0m")

    def set_smoothness_convexity_constraint_i_j(self,
                                                xi, gi, fi,
                                                xj, gj, fj,
                                                ):
        """
        Formulates the list of interpolation constraints for self (smooth convex function).
        """
        # Interpolation conditions of smooth convex functions class
        constraint = (fi - fj >= gj * (xi - xj) + 1 / (2 * self.L) * (gi - gj) ** 2)

        return constraint

    def add_class_constraints(self):
        """
        Add class constraints.
        """
        # Add Smoothness convexity interpolation constraints
        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="smoothness_convexity",
                                                      set_class_constraint_i_j=
                                                      self.set_smoothness_convexity_constraint_i_j,
                                                      )
