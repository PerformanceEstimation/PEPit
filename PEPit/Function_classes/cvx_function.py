from PEPit.function import Function


class CvxFunction(Function):
    """
    Convex Function
    """

    def __init__(self,
                 _,
                 is_leaf=True,
                 decomposition_dict=None,
                 is_differentiable=True):
        """
        Class of convex functions.
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
        Add all the interpolation condition of the convex functions
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if i != j:

                    # Interpolation conditions of convex functions class
                    self.add_constraint(fi - fj >= gj * (xi - xj))
