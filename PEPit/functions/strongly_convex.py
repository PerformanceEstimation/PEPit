from PEPit.function import Function


class StronglyConvexFunction(Function):
    """
    The :class:`StronglyConvexFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of strongly convex closed proper functions (strongly convex
    functions whose epigraphs are non-empty closed sets).

    Attributes:
        mu (float): strong convexity parameter

    Strongly convex functions are characterized by the strong convexity parameter :math:`\\mu`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import StronglyConvexFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=StronglyConvexFunction, mu=.1)

    References:

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
    Smooth strongly convex interpolation and exact worst-case performance of first-order methods.
    Mathematical Programming, 161(1-2), 307-345.
    <https://arxiv.org/pdf/1502.05666.pdf>`_

    """

    def __init__(self,
                 mu,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False,
                 name=None):
        """

        Args:
            mu (float): The strong convexity parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
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

        # Store mu
        self.mu = mu

    def set_strong_convexity_constraint_i_j(self,
                                            xi, gi, fi,
                                            xj, gj, fj,
                                            ):
        """
        Set strong convexity interpolation constraints.

        """
        # Set constraints
        constraint = (fi - fj >=
                      gj * (xi - xj)
                      + self.mu / 2 * (xi - xj) ** 2)

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (strongly convex closed proper function),
        see [1, Corollary 2].
        """

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="strong_convexity",
                                                      set_class_constraint_i_j=self.set_strong_convexity_constraint_i_j,
                                                      )
