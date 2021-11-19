from PEPit.function import Function


class CocoerciveOperator(Function):
    """
    CocoerciveOperator class

    Attributes:
        beta (float): cocoercivity constant

    Example:
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=CocoerciveOperator, param={'beta': 1})

    References:
        [1] E. K. Ryu, A. B. Taylor, C. Bergeling, and P. Giselsson,
        "Operator Splitting Performance Estimation: Tight contraction factors
        and optimal parameter selection," arXiv:1812.00146, 2018.

    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 is_differentiable=False):
        """
        Cocoercive operators are not characterized by any parameter.

        Args:
            is_leaf (bool): If True, it is a basis function. Otherwise it is a linear combination of such functions.
            decomposition_dict (dict): Decomposition in the basis of functions.
            is_differentiable (bool): If true, the function can have only one subgradient per point.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         is_differentiable=is_differentiable)

        # Store the beta parameter
        self.beta = param['beta']

    def add_class_constraints(self):
        """
        Add all the interpolation condition of the cocoercive operator
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of cocoercive operator class
                    self.add_constraint((gi - gj) * (xi - xj) - self.beta * (gi - gj) ** 2 >= 0)
