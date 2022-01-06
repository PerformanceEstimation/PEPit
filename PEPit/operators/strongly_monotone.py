from PEPit.function import Function


class StronglyMonotoneOperator(Function):
    """
    The :class:`StronglyMonotoneOperator` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing interpolation constraints of the class of strongly monotone
    (maximally monotone) operators.

    Note:
        Operators'values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        mu (float): strong monotonicity parameter

    Strongly monotone (and maximally monotone) operators are characterized by the parameter :math:`\\mu`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=StronglyMonotoneOperator, param={'mu': .1})

    References:
        Discussions and appropriate pointers for the problem of interpolation of maximally monotone operators can be found in:
        `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
        Operator splitting performance estimation: Tight contraction factors and optimal parameter selection.
        SIAM Journal on Optimization, 30(3), 2251-2271.
        <https://arxiv.org/pdf/1812.00146.pdf>`_

    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """

        Args:
            param (dict): contains the values of mu.
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
        # Store mu
        self.mu = param['mu']

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (strongly monotone maximally monotone operator),
        see, e.g., [1, Proposition 1].
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of strongly monotone operator class
                    self.add_constraint((gi - gj) * (xi - xj) - self.mu * (xi - xj) ** 2 >= 0)
