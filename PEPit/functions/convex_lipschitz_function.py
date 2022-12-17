from PEPit.function import Function


class ConvexLipschitzFunction(Function):
    """
    The :class:`ConvexLipschitzFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of convex closed proper (CCP) Lipschitz continuous functions.

    Attributes:
        M (float): Lipschitz parameter

    CCP Lipschitz continuous functions are characterized by a parameter `M`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import ConvexLipschitzFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=ConvexLipschitzFunction, M=1.)

    References:
        `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
        Exact worst-case performance of first-order methods for composite convex optimization.
        SIAM Journal on Optimization, 27(3):1283–1313.
        <https://arxiv.org/pdf/1512.07516.pdf>`_

    """

    def __init__(self,
                 M=1.,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """

        Args:
            M (float): The Lipschitz continuity parameter of self.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        """
        # Inherit directly from Function.
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

        # param M
        self.M = M

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (CCP Lipschitz continuous function),
        see [1, Theorem 3.5].
        """

        for point_i in self.list_of_points:

            xi, gi, fi = point_i

            # Lipschitz condition on the function (bounded gradient)
            self.list_of_class_constraints.append(gi**2 <= self.M**2)

            for point_j in self.list_of_points:

                xj, gj, fj = point_j

                if point_i != point_j:

                    # Interpolation conditions of convex functions class
                    self.list_of_class_constraints.append(fi - fj >= gj * (xi - xj))
