from PEPit.function import Function


class LipschitzStronglyMonotoneOperator(Function):
    """
    LipschitzStronglyMonotoneOperator class

    Attributes:
        mu (float): strongly monotone constant
        L (float): Lipschitz constant

    Example:
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=LipschitzStronglyMonotoneOperator, param={'mu': .1, 'L': 1})

    References:
        For details about interpolation conditions, we refer to the fllowing :
        [1] E. K. Ryu, A. B. Taylor, C. Bergeling, and P. Giselsson,
        "Operator Splitting Performance Estimation: Tight contraction factors
        and optimal parameter selection," arXiv:1812.00146, 2018.

    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 is_differentiable=True):
        """
        Lipschitz strongly monotone operators are characterized by
        their Lipschitz constant L
        and their strong monotony constant mu.

        Args:
            is_leaf (bool): If True, it is a basis function. Otherwise it is a linear combination of such functions.
            decomposition_dict (dict): Decomposition in the basis of functions.
            is_differentiable (bool): If true, the function can have only one subgradient per point.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         is_differentiable=is_differentiable)
        # Store L and mu
        self.mu = param['mu']
        self.L = param['L']

    def add_class_constraints(self):
        """
        Add all the interpolation condition of the Lipschitz strongly-monotone operator
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of strongly monotone operator class
                    self.add_constraint((gi - gj) * (xi - xj) - self.mu * (xi - xj)**2 >= 0)
                    # Interpolation conditions of Lipschitz operator class
                    self.add_constraint((gi - gj)**2 - self.L**2 * (xi - xj)**2 <= 0)
