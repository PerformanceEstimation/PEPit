from PEPit.function import Function


class ConvexLipschitzFunction(Function):
    """
    ConvexLipschitzFunction class

    Attributes:
        L (float): Lipschitz constant

    Example:
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=ConvexLipschitzFunction, param={'L': 1})

    References:


    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """
        Convex Lipschitz functions are characterized by their lipschitz parameter L.

        Args:
            param (dict): contains the value of L
            is_leaf (bool): If True, it is a basis function. Otherwise it is a linear combination of such functions.
            decomposition_dict (dict): Decomposition in the basis of functions.

        """
        # Inherit directly from Function.
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

        # param M
        self.M = param['M']

    def add_class_constraints(self):
        """
        Add all the interpolation condition of the convex functions
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            # Lipschitz condition on the function (bounded gradient)
            self.add_constraint(gi**2 <= self.M**2)

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if i != j:

                    # Interpolation conditions of convex functions class
                    self.add_constraint(fi - fj >= gj * (xi - xj))
