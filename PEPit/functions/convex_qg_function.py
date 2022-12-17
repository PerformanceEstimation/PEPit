from PEPit.function import Function


class ConvexQGFunction(Function):
    """
    The :class:`ConvexQGFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of quadratically upper bounded (:math:`\\text{QG}^+` [1]),
    i.e. :math:`\\forall x, f(x) - f_\\star \\leqslant \\frac{L}{2} \\|x-x_\\star\\|^2`, and convex functions.

    Attributes:
        L (float): The quadratic upper bound parameter

    General quadratically upper bounded (:math:`\\text{QG}^+`) convex functions are characterized
    by the quadratic growth parameter `L`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import ConvexQGFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=ConvexQGFunction, L=1)

    References:

    `[1] B. Goujaud, A. Taylor, A. Dieuleveut (2022).
    Optimal first-order methods for convex functions with a quadratic upper bound.
    <https://arxiv.org/pdf/2205.15033.pdf>`_

    """

    def __init__(self,
                 L=1,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """

        Args:
            L (float): The quadratic upper bound parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

        # Store L
        self.L = L

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (quadratically maximally growing convex function);
        see [1, Theorem 2.6].
        """

        for point_i in self.list_of_stationary_points:

            xi, gi, fi = point_i

            for point_j in self.list_of_points:

                xj, gj, fj = point_j

                if point_i != point_j:
                    # Interpolation conditions of convex functions class
                    self.list_of_class_constraints.append(fi - fj >= gj * (xi - xj) + 1 / (2 * self.L) * gj ** 2)

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if i != j:
                    # Interpolation conditions of convex functions class
                    self.list_of_class_constraints.append(fi - fj >= gj * (xi - xj))
