from PEPit.function import Function


class ConvexFunction(Function):
    """
    ConvexFunction class

    Example:
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=ConvexFunction, param=dict())

    References:


    """

    def __init__(self,
                 _,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """
        Convex functions are not characterized by any parameter.

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
        Add all the interpolation conditions of convex functions
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if i != j:

                    # Interpolation conditions of convex functions class
                    self.add_constraint(fi - fj >= gj * (xi - xj))
