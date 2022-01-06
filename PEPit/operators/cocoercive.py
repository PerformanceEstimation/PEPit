from PEPit.function import Function


class CocoerciveOperator(Function):
    """
    The :class:`CocoerciveOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of cocoercive (and maximally monotone) operators.

    Note:
        Operators'values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        beta (float): cocoercivity parameter

    Cocoercive operators are characterized by the parameter :math:`\\beta`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=CocoerciveOperator, param={'beta': 1})

    References:
        `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
        Operator splitting performance estimation: Tight contraction factors and optimal parameter selection.
        SIAM Journal on Optimization, 30(3), 2251-2271.
        <https://arxiv.org/pdf/1812.00146.pdf>`_

    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
        """

        Args:
            param (dict): contains the values of beta.
            is_leaf (bool): True if self is defined from scratch.
                            False is self is defined as linear combination of leaf .
            decomposition_dict (dict): decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        Note:
            Cocoercive operators are necessarily continuous, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True)

        # Store the beta parameter
        self.beta = param['beta']

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (cocoercive maximally monotone operator),
        see, e.g., [1, Proposition 2].
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of cocoercive operator class
                    self.add_constraint((gi - gj) * (xi - xj) - self.beta * (gi - gj) ** 2 >= 0)
