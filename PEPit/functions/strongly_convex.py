from PEPit.function import Function


class StronglyConvexFunction(Function):
    """
    The :class:`StronglyConvexFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of strongly convex closed proper functions (strongly convex
    functions whose epigraphs are non-empty closed sets).

    Attributes:
        mu (float): strong convexity parameter

    Strongly convex functions are characterized by the strong convexity parameter :math:`\\mu`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=StronglyConvexFunction, param={'mu': .1})

    References:
        `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
        Smooth strongly convex interpolation and exact worst-case performance of first-order methods.
        Mathematical Programming, 161(1-2), 307-345.
        <https://arxiv.org/pdf/1502.05666.pdf>`_

    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """

        Args:
            param (dict): contains the values of mu
            is_leaf (bool): True if self is defined from scratch.
                            False is self is defined as linear combination of leaf .
            decomposition_dict (dict): decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

        # Store mu and L
        self.mu = param['mu']

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (strongly convex closed proper function),
        see [1, Corollary 2].
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if i != j:

                    # Interpolation conditions of smooth strongly convex functions class
                    self.add_constraint(fi - fj >=
                                        gj * (xi - xj)
                                        + self.mu / 2 * (xi - xj) ** 2)
