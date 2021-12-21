from PEPit.function import Function


class MonotoneOperator(Function):
    """
    The :class:`MonotoneOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of monotone operators.

    Note:
        Operators'values can be requested through `gradient` and `function values` should not be used.

    Monotone operators are not characterized by any parameter, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=MonotoneOperator, param=dict())

    References:
        For details about interpolation conditions, we refer to the following:
        `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2018).
        Operator Splitting Performance Estimation: Tight contraction factors and optimal parameter selection.
        <https://arxiv.org/pdf/1812.00146.pdf>`_

    """

    def __init__(self,
                 _,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """
        Args:
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

    def add_class_constraints(self):
        """
        Add all the interpolation conditions of the monotone operators provided in [1].
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of monotone operator class
                    self.add_constraint((gi - gj) * (xi - xj) >= 0)
