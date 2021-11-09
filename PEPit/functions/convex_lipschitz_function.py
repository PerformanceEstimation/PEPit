from PEPit.function import Function


class ConvexLipschitzFunction(Function):
    """
    Convex Smooth Function
    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 is_differentiable=False):
        """
        Class of convex smooth functions.
        The differentiability is necessarily verified.

        :param param: (dict) contains the value of L
        :param is_leaf: (bool) If True, it is a basis function. Otherwise it is a linear combination of such functions.
        :param decomposition_dict: (dict) Decomposition in the basis of functions.
        """
        # Inherit directly from Function.
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         is_differentiable=is_differentiable)

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