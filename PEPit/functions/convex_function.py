from PEPit.function import Function


class ConvexFunction(Function):
    """
    The :class:`ConvexFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of convex, closed and proper (CCP) functions (i.e., convex
    functions whose epigraphs are non-empty closed sets).

    General CCP functions are not characterized by any parameter, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=ConvexFunction, param=dict())

    """

    def __init__(self,
                 _,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """

        Args:
            is_leaf (bool): True if self is defined from scratch.
                            False is self is defined as linear combination of leaf .
            decomposition_dict (dict): decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (CCP function).
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if i != j:

                    # Interpolation conditions of convex functions class
                    self.add_constraint(fi - fj >= gj * (xi - xj))
