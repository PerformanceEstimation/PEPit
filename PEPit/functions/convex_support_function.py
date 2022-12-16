import numpy as np

from PEPit.function import Function


class ConvexSupportFunction(Function):
    """
    The :class:`ConvexSupportFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing interpolation constraints for the class of closed convex support functions.

    Attributes:
        M (float): upper bound on the Lipschitz constant

    Convex support functions are characterized by a parameter `M`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import ConvexSupportFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=ConvexSupportFunction, M=1)

    References:
        `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
        Exact worst-case performance of first-order methods for composite convex optimization.
        SIAM Journal on Optimization, 27(3):1283–1313.
        <https://arxiv.org/pdf/1512.07516.pdf>`_

    """

    def __init__(self,
                 M=np.inf,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """

        Args:
            M (float): Lipschitz constant of self.
            is_leaf (bool): True if self is defined from scratch.
                            False is self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

        # Store the Lipschitz constant in an attribute
        self.M = M

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (closed convex support function),
        see [1, Corollary 3.7].
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if point_i == point_j:
                    self.list_of_class_constraints.append(gi * xi - fi == 0)
                    if self.M != np.inf:
                        self.list_of_class_constraints.append(gi ** 2 <= self.M ** 2)

                else:
                    self.list_of_class_constraints.append(xj * (gi - gj) <= 0)
