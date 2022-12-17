import numpy as np

from PEPit.function import Function


class ConvexIndicatorFunction(Function):
    """
    The :class:`ConvexIndicatorFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing interpolation constraints for the class of closed convex indicator functions.

    Attributes:
        D (float): upper bound on the diameter of the feasible set, possibly set to np.inf

    Convex indicator functions are characterized by a parameter `D`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import ConvexIndicatorFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=ConvexIndicatorFunction, D=1)

    References:
        `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
        Exact worst-case performance of first-order methods for composite convex optimization.
        SIAM Journal on Optimization, 27(3):1283–1313.
        <https://arxiv.org/pdf/1512.07516.pdf>`_

    """

    def __init__(self,
                 D=np.inf,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """

        Args:
            D (float): Diameter of the support of self.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

        # Store the diameter D in an attribute
        self.D = D

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (closed convex indicator function),
        see [1, Theorem 3.6].
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if point_i == point_j:
                    self.list_of_class_constraints.append(fi == 0)

                else:
                    self.list_of_class_constraints.append(gi * (xj - xi) <= 0)
                    if self.D != np.inf:
                        self.list_of_class_constraints.append((xi - xj) ** 2 <= self.D ** 2)
