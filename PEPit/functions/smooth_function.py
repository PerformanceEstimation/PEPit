import numpy as np
from PEPit.function import Function


class SmoothFunction(Function):
    """
    The :class:`SmoothFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of smooth (not necessarily convex) functions.

    Attributes:
        L (float): smoothness parameter

    Smooth functions are characterized by the smoothness parameter `L`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import SmoothFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=SmoothFunction, L=1.)

    References:
        `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
        Exact worst-case performance of first-order methods for composite convex optimization.
        SIAM Journal on Optimization, 27(3):1283–1313.
        <https://arxiv.org/pdf/1512.07516.pdf>`_

    """

    def __init__(self,
                 L=1.,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
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

        Note:
            Smooth functions are necessarily differentiable, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True)

        # Store L
        self.L = L

        if self.L == np.inf:
            print("\033[96m(PEPit) The class of L-smooth functions with L == np.inf implies no constraint: \n"
                  "it contains all differentiable functions. This might imply issues in your code.\033[0m")

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (smooth (not necessarily convex) function),
        see [1, Theorem 3.10].
        """

        for point_i in self.list_of_points:

            xi, gi, fi = point_i

            for point_j in self.list_of_points:

                xj, gj, fj = point_j

                if point_i != point_j:

                    # Interpolation conditions of smooth functions class
                    self.list_of_class_constraints.append(fi - fj
                                        - self.L/4 * (xi - xj)**2
                                        - 1/2 * (gi + gj) * (xi - xj)
                                        + 1/(4 * self.L) * (gi - gj)**2
                                        <= 0)
