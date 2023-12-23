from PEPit.function import Function


class StronglyMonotoneOperator(Function):
    """
    The :class:`StronglyMonotoneOperator` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing interpolation constraints of the class of strongly monotone
    (maximally monotone) operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        mu (float): strong monotonicity parameter

    Strongly monotone (and maximally monotone) operators are characterized by the parameter :math:`\\mu`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import StronglyMonotoneOperator
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=StronglyMonotoneOperator, mu=.1)

    References:
        Discussions and appropriate pointers for the problem of
        interpolation of maximally monotone operators can be found in:
        `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
        Operator splitting performance estimation: Tight contraction factors and optimal parameter selection.
        SIAM Journal on Optimization, 30(3), 2251-2271.
        <https://arxiv.org/pdf/1812.00146.pdf>`_

    """

    def __init__(self,
                 mu,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False,
                 name=None):
        """

        Args:
            mu (float): Strong monotonicity parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
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

    def set_strong_monotonicity_constraint_i_j(self,
                                               xi, gi, fi,
                                               xj, gj, fj,
                                               ):
        """
        Set strong monotonicity constraint for operators.

        """
        # Set constraint
        constraint = ((gi - gj) * (xi - xj) - self.mu * (xi - xj) ** 2 >= 0)

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of necessary conditions for interpolation of self (Lipschitz strongly monotone and
        maximally monotone operator), see, e.g., discussions in [1, Section 2].
        """

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="strong_monotonicity",
                                                      set_class_constraint_i_j=
                                                      self.set_strong_monotonicity_constraint_i_j,
                                                      symmetry=True,
                                                      )
