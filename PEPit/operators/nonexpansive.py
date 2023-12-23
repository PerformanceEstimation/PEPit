from PEPit.function import Function


class NonexpansiveOperator(Function):
    """
    The :class:`NonexpansiveOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of (possibly inconsistent) nonexpansive operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        v (Point): infimal displacement vector.

    Nonexpansive operators are not characterized by any parameter, hence can be initiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import NonexpansiveOperator
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=NonexpansiveOperator)

    Notes:
        Any nonexpansive operator has a unique vector called `infimal displacement vector`, which we denote by v.
        
        If a nonexpansive operator is consistent, i.e., has a fixed point, then v=0.

        If v is nonzero, a nonexpansive operator is inconsistent, i.e., does not have a fixed point.

    References:

        Discussions and appropriate pointers for the interpolation problem can be found in:

        `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
        Operator splitting performance estimation: Tight contraction factors and optimal parameter selection.
        SIAM Journal on Optimization, 30(3), 2251-2271.
        <https://arxiv.org/pdf/1812.00146.pdf>`_

        `[2] J. Park, E. Ryu (2023).
        Accelerated Infeasibility Detection of Constrained Optimization and Fixed-Point Iterations.
        arXiv preprint:2303.15876.
        <https://arxiv.org/pdf/2303.15876.pdf>`_

    """

    def __init__(self,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Nonexpansive continuous operators are necessarily continuous, hence `reuse_gradient` is set to True.

            Setting self.v = None corresponds to case when a nonexpansive operator is consistent.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store the infimal displacement vector v to None by default.
        self.v = None

    @staticmethod
    def set_nonexpansiveness_constraint_i_j(xi, gi, fi,
                                            xj, gj, fj,
                                            ):
        """
        Set Lipschitz continuity constraint for operators.

        """
        # Set constraint
        constraint = ((gi - gj) ** 2 - (xi - xj) ** 2 <= 0)

        return constraint

    def set_infimal_displacement_vector_constraint_i(self,
                                                     xi, gi, fi,
                                                     ):
        """
        Set infimal displacement vector constraint of nonexpansive operators.

        """
        # Infimal displacement vector constraint
        constraint = (self.v ** 2 - (xi - gi) * self.v <= 0)

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (Nonexpansive operator),
        see [2, Theorem 10].
        """

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="nonexpansiveness",
                                                      set_class_constraint_i_j=
                                                      self.set_nonexpansiveness_constraint_i_j,
                                                      symmetry=True,
                                                      )
        if self.v is not None:
            self.add_constraints_from_one_list_of_points(list_of_points=self.list_of_points,
                                                         constraint_name="infimal_displacement_vector",
                                                         set_class_constraint_i=
                                                         self.set_infimal_displacement_vector_constraint_i,
                                                         )
