from PEPit.function import Function


class MonotoneOperator(Function):
    """
    Tis routine implements the interpolation conditions for monotone operators.

    To generate a monotone operator 'h' from an instance of PEP called P :
    >> problem = pep()
    >> h = problem.DeclareFunction(MonotoneOperator, {})

    NOTE : PEPit was initially tough for evaluating performances of optimization algorithms.
    Operators are represented in the same way as functions, but function values are not accessible.

    For details about interpolation conditions, we refer to the fllowing :
    [1] E. K. Ryu, A. B. Taylor, C. Bergeling, and P. Giselsson,
      "Operator Splitting Performance Estimation: Tight contraction factors
      and optimal parameter selection," arXiv:1812.00146, 2018.

    """

    def __init__(self,
                 _,
                 is_leaf=True,
                 decomposition_dict=None,
                 is_differentiable=False):
        """
        Class of monotone operators.
        It does not need any additional parameter.

        :param is_leaf: (bool) If True, it is a basis function. Otherwise it is a linear combination of such functions.
        :param decomposition_dict: (dict) Decomposition in the basis of functions.
        :param is_differentiable: (bool) If true, the function can have only one subgradient per point.
        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         is_differentiable=is_differentiable)

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