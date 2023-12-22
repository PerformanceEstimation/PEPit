from PEPit.function import Function


class ConvexLipschitzFunction(Function):
    """
    The :class:`ConvexLipschitzFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of convex closed proper (CCP)
    Lipschitz continuous functions.

    Attributes:
        M (float): Lipschitz parameter

    CCP Lipschitz continuous functions are characterized by a parameter `M`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import ConvexLipschitzFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=ConvexLipschitzFunction, M=1.)

    References:

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
    Exact worst-case performance of first-order methods for composite convex optimization.
    SIAM Journal on Optimization, 27(3):1283–1313.
    <https://arxiv.org/pdf/1512.07516.pdf>`_

    """

    def __init__(self,
                 M,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False,
                 name=None):
        """

        Args:
            M (float): The Lipschitz continuity parameter of self.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        """
        # Inherit directly from Function.
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient,
                         name=name,
                         )

        # param M
        self.M = M

    def set_lipschitz_continuity_constraint_i(self,
                                              xi, gi, fi):
        """
        Formulates the Lipschitz continuity constraint by bounding the gradients.

        """
        # Lipschitz condition on the function (bounded gradient)
        constraint = (gi ** 2 <= self.M ** 2)

        return constraint

    @staticmethod
    def set_convexity_constraint_i_j(xi, gi, fi,
                                     xj, gj, fj,
                                     ):
        """
        Formulates the list of interpolation constraints for self (CCP function).
        """
        # Interpolation conditions of convex functions class
        constraint = (fi - fj >= gj * (xi - xj))

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (CCP Lipschitz continuous function),
        see [1, Theorem 3.5].
        """

        self.add_constraints_from_one_list_of_points(list_of_points=self.list_of_points,
                                                     constraint_name="lipschitz_continuity",
                                                     set_class_constraint_i=self.set_lipschitz_continuity_constraint_i,
                                                     )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="convexity",
                                                      set_class_constraint_i_j=self.set_convexity_constraint_i_j,
                                                      )
