from PEPit.function import Function


class CocoerciveStronglyMonotoneOperator(Function):
    """
    The :class:`CocoerciveStronglyMonotoneOperator` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing some necessary constraints verified by the class of cocoercive
    and strongly monotone (maximally) operators.

    Warnings:
        Those constraints might not be sufficient, thus the caracterized class might contain more operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        mu (float): strong monotonicity parameter
        beta (float): cocoercivity parameter

    Cocoercive operators are characterized by the parameters :math:`\\mu` and :math:`\\beta`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import CocoerciveStronglyMonotoneOperator
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=CocoerciveStronglyMonotoneOperator, mu=.1, beta=1.)

    References:

    `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
    Operator splitting performance estimation: Tight contraction factors and optimal parameter selection.
    SIAM Journal on Optimization, 30(3), 2251-2271.
    <https://arxiv.org/pdf/1812.00146.pdf>`_

    """

    def __init__(self,
                 mu,
                 beta,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
            mu (float): The strong monotonicity parameter.
            beta (float): The cocoercivity parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Cocoercive operators are necessarily continuous, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store the mu and beta parameters
        self.mu = mu
        self.beta = beta

        if self.mu == 0:
            print("\033[96m(PEPit) The class of cocoercive and strongly monotone operators is necessarily continuous."
                  " \n"
                  "To instantiate a cocoercive (non strongly) monotone operator,"
                  " please avoid using the class CocoerciveStronglyMonotoneOperator\n"
                  "with mu == 0. Instead, please use the class CocoerciveOperator.\033[0m")

        if self.beta == 0:
            print("\033[96m(PEPit) The class of cocoercive and strongly monotone operators is necessarily continuous."
                  " \n"
                  "To instantiate a non cocoercive strongly monotone operator,"
                  " please avoid using the class CocoerciveStronglyMonotoneOperator\n"
                  "with beta == 0. Instead, please use the class StronglyMonotoneOperator.\033[0m")

    def set_cocoercivity_constraint_i_j(self,
                                        xi, gi, fi,
                                        xj, gj, fj,
                                        ):
        """
        Formulates the list of interpolation constraints for self (cocoercive strongly monotone operators).
        """
        # Interpolation conditions of cocoercive operators class
        constraint = ((gi - gj) * (xi - xj) - self.beta * (gi - gj) ** 2 >= 0)

        return constraint

    def set_strong_monotonicity_constraint_i_j(self,
                                               xi, gi, fi,
                                               xj, gj, fj,
                                               ):
        """
        Formulates the list of interpolation constraints for self (cocoercive strongly monotone operators).
        """
        # Interpolation conditions of strongly monotone operators class
        constraint = ((gi - gj) * (xi - xj) - self.mu * (xi - xj) ** 2 >= 0)

        return constraint

    def add_class_constraints(self):
        """
        Add interpolation constraints for self (cocoercive strongly monotone operator).
        """
        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="cocoercivity",
                                                      set_class_constraint_i_j=self.set_cocoercivity_constraint_i_j,
                                                      symmetry=True,
                                                      )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="strong_monotonicity",
                                                      set_class_constraint_i_j=
                                                      self.set_strong_monotonicity_constraint_i_j,
                                                      symmetry=True,
                                                      )
