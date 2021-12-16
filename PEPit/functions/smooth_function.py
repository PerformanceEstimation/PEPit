from PEPit.function import Function


class SmoothFunction(Function):
    """
    SmoothFunction class

    Attributes:
        L (float): smoothness constant

    Example:
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=SmoothFunction, param={'L': 1})

    References:


    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
        """
        Smooth functions are characterized by their smoothness constant L.

        Args:
            param (dict): contains the values L
            is_leaf (bool): If True, it is a basis function. Otherwise it is a linear combination of such functions.
            decomposition_dict (dict): Decomposition in the basis of functions.
        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

        # Store L
        self.L = param['L']

    def add_class_constraints(self):
        """
        Add all the interpolation condition of the strongly convex smooth functions
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj) | (fi != fj):

                    # Interpolation conditions of smooth functions class
                    self.add_constraint(fi - fj
                                        - self.L/4 * (xi - xj)**2
                                        - 1/2 * (gi + gj) * (xi - xj)
                                        + 1/(4 * self.L) * (gi - gj)**2
                                        <= 0)
