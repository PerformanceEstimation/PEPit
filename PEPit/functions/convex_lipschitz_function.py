from PEPit.function import Function


class ConvexLipschitzFunction(Function):
    """
    The :class:`ConvexLipschitzFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of convex Lipschitz continuous functions.

    Attributes:
        M (float): Lipschitz constant

    Convex Lipschitz continuous functions are characterized by the parameter `M`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=ConvexLipschitzFunction, param={'M': 1})

    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """

        Args:
            param (dict): contains the value of M
            is_leaf (bool): True if self is defined from scratch.
                            False is self is defined as linear combination of leaf .
            decomposition_dict (dict): decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        """
        # Inherit directly from Function.
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

        # param M
        self.M = param['M']

    def add_class_constraints(self):
        """
        Add all the interpolation conditions of the convex Lipschitz continuous functions.
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
