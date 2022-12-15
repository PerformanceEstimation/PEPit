from PEPit.function import Function


class StronglyMonotoneOperator(Function):
    """
    The :class:`StronglyMonotoneOperator` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing interpolation constraints of the class of strongly monotone
    (maximally monotone) operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        mu (float): strong monotonicity parameter

    Strongly monotone (and maximally monotone) operators are characterized by the parameter :math:`\\mu`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import StronglyMonotoneOperator
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=StronglyMonotoneOperator, mu=.1)

    References:
        Discussions and appropriate pointers for the problem of
        interpolation of maximally monotone operators can be found in:
        `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
        Operator splitting performance estimation: Tight contraction factors and optimal parameter selection.
        SIAM Journal on Optimization, 30(3), 2251-2271.
        <https://arxiv.org/pdf/1812.00146.pdf>`_

    """

    def __init__(self,
                 mu,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """

        Args:
            mu (float): Strong monotonicity parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)
        # Store mu
        self.mu = mu

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (strongly monotone maximally monotone operator),
        see, e.g., [1, Proposition 1].
        """

        for point_i in self.list_of_points:

            xi, gi, fi = point_i

            for point_j in self.list_of_points:

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of strongly monotone operator class
                    self.list_of_class_constraints.append((gi - gj) * (xi - xj) - self.mu * (xi - xj) ** 2 >= 0)
