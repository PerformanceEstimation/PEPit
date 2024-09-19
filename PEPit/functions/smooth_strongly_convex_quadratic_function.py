from PEPit.function import Function


class SmoothStronglyConvexQuadraticFunction(Function):
    """
    The :class:`SmoothStronglyConvexQuadraticFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    by implementing interpolation constraints of the class of smooth strongly convex quadratic functions.

    Attributes:
        mu (float): strong convexity parameter
        L (float): smoothness parameter

    Smooth strongly convex quadratic functions are characterized by parameters :math:`\\mu` and `L`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import SmoothStronglyConvexQuadraticFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=SmoothStronglyConvexQuadraticFunction, mu=.1, L=1.)

    References:

    `[1] N. Bousselmi, J. Hendrickx, F. Glineur  (2023).
    Interpolation Conditions for Linear Operators and applications to Performance Estimation Problems.
    arXiv preprint.
    <https://arxiv.org/pdf/2302.08781.pdf>`_

    """

    def __init__(self,
                 mu,
                 L,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
            mu (float): The strong convexity parameter.
            L (float): The smoothness parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Smooth strongly convex quadratic functions are necessarily differentiable,
            hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store mu and L
        self.mu = mu
        self.L = L

        # Create a stationary point
        self.stationary_point()

    def set_value_constraint_i(self,
                               xi, gi, fi):
        """
        Set the value of the function.

        """
        # Select one stationary point
        xs, _, fs = self.list_of_stationary_points[0]

        # Value constraint
        constraint = (fi - fs == 0.5 * (xi - xs) * gi)

        return constraint

    def set_symmetry_constraint_i_j(self,
                                    xi, gi, fi,
                                    xj, gj, fj,
                                    ):
        """
        Ensure the Hessian is symmetric.
        """
        # Select one stationary point
        xs = self.list_of_stationary_points[0][0]

        # Symmetry constraint
        constraint = ((xi - xs) * gj == (xj - xs) * gi)

        return constraint

    def set_smoothness_strong_convexity_constraint_i_j(self,
                                                       xi, gi, fi,
                                                       xj, gj, fj,
                                                       ):
        """
        Formulates the list of interpolation constraints for self (smooth strongly convex function).
        """
        # Interpolation conditions of smooth strongly convex functions class
        constraint = (fi - fj >=
                      gj * (xi - xj)
                      + 1 / (2 * self.L) * (gi - gj) ** 2
                      + self.mu / (2 * (1 - self.mu / self.L)) * (
                              xi - xj - 1 / self.L * (gi - gj)) ** 2)

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (smooth strongly convex quadratic function);
        see [1, Theorem 3.9].
        """
        # Add the quadratic interpolation constraint
        self.add_constraints_from_one_list_of_points(list_of_points=self.list_of_points,
                                                     constraint_name="value",
                                                     set_class_constraint_i=self.set_value_constraint_i,
                                                     )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="symmetry",
                                                      set_class_constraint_i_j=self.set_symmetry_constraint_i_j,
                                                      symmetry=True,
                                                      )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="smoothness_strong_convexity",
                                                      set_class_constraint_i_j=
                                                      self.set_smoothness_strong_convexity_constraint_i_j,
                                                      )
