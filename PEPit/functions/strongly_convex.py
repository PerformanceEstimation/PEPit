from PEPit.function import Function


class StronglyConvexFunction(Function):
    """
    StronglyConvexFunction class

    Attributes:
        mu (float): strong convexity constant

    Example:
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=StronglyConvexFunction, param={'mu': .1})

    References:


    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """
        Strongly convex functions are characterized by their strong convexity constant mu.

        Args:
            param (dict): contains the values of mu and L
            is_leaf (bool): If True, it is a basis function. Otherwise it is a linear combination of such functions.
            decomposition_dict (dict): Decomposition in the basis of functions.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

        # Store mu and L
        self.mu = param['mu']

    def add_class_constraints(self):
        """
        Add all the interpolation condition of the strongly convex smooth functions
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if i != j:

                    # Interpolation conditions of smooth strongly convex functions class
                    self.add_constraint(fi - fj >=
                                        gj * (xi - xj)
                                        + self.mu / 2 * (xi - xj) ** 2)
