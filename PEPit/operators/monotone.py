from PEPit.function import Function


class MonotoneOperator(Function):
    """
    MonotoneOperator class

    Example:
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=MonotoneOperator, param=dict())

    References:
        For details about interpolation conditions, we refer to the fllowing :
        [1] E. K. Ryu, A. B. Taylor, C. Bergeling, and P. Giselsson,
        "Operator Splitting Performance Estimation: Tight contraction factors
        and optimal parameter selection," arXiv:1812.00146, 2018.

    """

    def __init__(self,
                 _,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """
        Monotone operators are are not characterized by any parameter.

        Args:
            is_leaf (bool): If True, it is a basis function. Otherwise it is a linear combination of such functions.
            decomposition_dict (dict): Decomposition in the basis of functions.
            reuse_gradient (bool): If true, the function can have only one subgradient per point.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

    def add_class_constraints(self):
        """
        Add all the interpolation condition of the monotone operator
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of monotone operator class
                    self.add_constraint((gi - gj) * (xi - xj) >= 0)
