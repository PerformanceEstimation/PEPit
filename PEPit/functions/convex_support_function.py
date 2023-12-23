import numpy as np

from PEPit.function import Function


class ConvexSupportFunction(Function):
    """
    The :class:`ConvexSupportFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing interpolation constraints for the class of closed convex support functions.

    Attributes:
        M (float): upper bound on the Lipschitz constant

    Convex support functions are characterized by a parameter `M`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import ConvexSupportFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=ConvexSupportFunction, M=1)

    References:

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
    Exact worst-case performance of first-order methods for composite convex optimization.
    SIAM Journal on Optimization, 27(3):1283–1313.
    <https://arxiv.org/pdf/1512.07516.pdf>`_

    """

    def __init__(self,
                 M=np.inf,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False,
                 name=None):
        """

        Args:
            M (float): Lipschitz constant of self. Default value set to infinity.
            is_leaf (bool): True if self is defined from scratch.
                            False is self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient,
                         name=name,
                         )

        # Store the Lipschitz constant in an attribute
        self.M = M

    @staticmethod
    def set_fenchel_value_constraint_i(xi, gi, fi):
        """
        Set the value of the Fenchel transform to 0.

        """
        # Set constraint
        constraint = (gi * xi - fi == 0)

        return constraint

    def set_lipschitz_continuity_constraint_i(self, xi, gi, fi):
        """
        Set Lipschitz continuity constraint so that its Fenchel transform has a bounded support.

        """
        # Set constraint
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
        constraint = (xj * (gi - gj) <= 0)

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (closed convex support function),
        see [1, Corollary 3.7].
        """

        self.add_constraints_from_one_list_of_points(list_of_points=self.list_of_points,
                                                     constraint_name="fenchel_value",
                                                     set_class_constraint_i=self.set_fenchel_value_constraint_i,
                                                     )

        if self.M != np.inf:
            self.add_constraints_from_one_list_of_points(list_of_points=self.list_of_points,
                                                         constraint_name="lipschitz_continuity",
                                                         set_class_constraint_i=
                                                         self.set_lipschitz_continuity_constraint_i,
                                                         )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="convexity",
                                                      set_class_constraint_i_j=self.set_convexity_constraint_i_j,
                                                      )
