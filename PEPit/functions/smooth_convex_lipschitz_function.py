import numpy as np
from PEPit.function import Function


class SmoothConvexLipschitzFunction(Function):
    """
    The :class:`SmoothConvexLipschitzFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    by implementing interpolation constraints of the class of smooth convex Lipschitz continuous functions.

    Attributes:
        L (float): smoothness parameter
        M (float): Lipschitz continuity parameter

    Smooth convex Lipschitz continuous functions are characterized by the smoothness parameters `L`
    and Lipschitz continuity parameter `M`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import SmoothConvexLipschitzFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=SmoothConvexLipschitzFunction, L=1., M=1.)

    References:

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
    Exact worst-case performance of first-order methods for composite convex optimization.
    SIAM Journal on Optimization, 27(3):1283–1313.
    <https://arxiv.org/pdf/1512.07516.pdf>`_

    """

    def __init__(self,
                 L,
                 M,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
            L (float): The smoothness parameter.
            M (float): The Lipschitz continuity parameter of self.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Smooth convex Lipschitz continuous functions are necessarily differentiable,
            hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store L and M
        self.L = L
        self.M = M

        if self.L == np.inf:
            print("\033[96m(PEPit) Smooth convex Lipschitz continuous functions are necessarily differentiable.\n"
                  "To instantiate a convex Lipschitz continuous function, please avoid using the class\n"
                  "SmoothConvexLipschitzFunction with L == np.inf.\n"
                  "Instead, please use the class ConvexLipschitzFunction (which accounts for the fact \n"
                  "that there might be several subgradients at the same point).\033[0m")

    def set_smoothness_convexity_constraint_i_j(self,
                                                xi, gi, fi,
                                                xj, gj, fj,
                                                ):
        """
        Formulates the list of interpolation constraints for smooth convex functions.
        """
        # Interpolation conditions of smooth convex functions class
        constraint = (fi - fj >= gj * (xi - xj) + 1 / (2 * self.L) * (gi - gj) ** 2)

        return constraint

    def set_lipschitz_continuity_constraint_i(self,
                                              xi, gi, fi):
        """
        Formulates the Lipschitz continuity constraint by bounding the gradients.

        """
        # Lipschitz condition on the function (bounded gradient)
        constraint = (gi ** 2 <= self.M ** 2)

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for smooth convex functions; see [1, Theorem 4],
        and add the Lipschitz continuity interpolation constraints.
        """
        # Add Smoothness convexity interpolation constraints
        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="smoothness_convexity",
                                                      set_class_constraint_i_j=
                                                      self.set_smoothness_convexity_constraint_i_j,
                                                      )

        # Add Lipschitz continuity interpolation constraints
        self.add_constraints_from_one_list_of_points(list_of_points=self.list_of_points,
                                                     constraint_name="lipschitz_continuity",
                                                     set_class_constraint_i=self.set_lipschitz_continuity_constraint_i,
                                                     )
