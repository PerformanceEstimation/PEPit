from PEPit.function import Function


class LipschitzOperator(Function):
    """
    The :class:`LipschitzOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of Lipschitz continuous operators.

    Note:
        Operators'values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        L (float) Lipschitz constant

    Cocoercive operators are characterized by the parameter :math:`L`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=LipschitzOperator, param={'L': 1})

    Notes:
        By setting L=1, we define a non expansive operator.

        By setting L<1, we define a contracting operator.

    References:
        For details about interpolation conditions, we refer to the following:
        `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2018).
        Operator Splitting Performance Estimation: Tight contraction factors and optimal parameter selection.
        <https://arxiv.org/pdf/1812.00146.pdf>`_

    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
        """

        Args:
            param (dict): contains the values of L.
            is_leaf (bool): True if self is defined from scratch.
                            False is self is defined as linear combination of leaf .
            decomposition_dict (dict): decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        Note:
            Lipschitz continuous operators are necessarily continuous, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True)
        # Store L
        self.L = param['L']

    def add_class_constraints(self):
        """
        Add all the interpolation conditions of the strongly monotone operators provided in [1].
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of Lipschitz operator class
                    self.add_constraint((gi - gj) ** 2 - self.L ** 2 * (xi - xj) ** 2 <= 0)
