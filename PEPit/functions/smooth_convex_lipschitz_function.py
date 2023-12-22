import numpy as np
from PEPit.functions.smooth_convex_function import SmoothConvexFunction


class SmoothConvexLipschitzFunction(SmoothConvexFunction):
    """
    The :class:`SmoothConvexLipschitzFunction` class implements smooth convex Lipschitz continuous functions
    as particular cases of :class:`SmoothConvexFunction`.

    Attributes:
        L (float): smoothness parameter
        M (float): Lipschitz parameter

    Smooth convex Lipschitz continuous functions are characterized by the smoothness parameters `L` and `M`,
    hence can be instantiated as

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
                 L=1.,
                 M=1.,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
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

        Note:
            Smooth convex Lipschitz continuous functions are necessarily differentiable,
            hence `reuse_gradient` is set to True.

        """
        # Inherit from SmoothConvexFunction as a special case of it.
        super().__init__(L=L,
                         is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         )

        # Add M attributes that SmoothConvexFunction does not have.
        self.M = M

        if self.L == np.inf and self.M == np.inf:
            print("\033[96m(PEPit) Smooth convex Lipschitz continuous functions are necessarily differentiable.\n"
                  "To instantiate a convex function, please avoid using the class SmoothConvexLipschitzFunction with "
                  "L == np.inf and M == np.inf.\n"
                  "Instead, please use the class ConvexFunction (which accounts for the fact \n"
                  "that there might be several subgradients at the same point).\033[0m")

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (smooth convex function); see [1, Theorem 4],
        and add the Lipschitz continuity interpolation constraints.
        """
        # Add smooth convex interpolation constraints.
        super().add_class_constraints()

        # Add Lipschitz continuity interpolation constraints.
        for point_i in self.list_of_points:

            xi, gi, fi = point_i

            # Lipschitz condition on the function (bounded gradient)
            self.list_of_class_constraints.append(gi**2 <= self.M**2)
