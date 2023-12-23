import numpy as np
from PEPit.function import Function


class LipschitzStronglyMonotoneOperator(Function):
    """
    The :class:`LipschitzStronglyMonotoneOperator` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing some constraints (which are not necessary and sufficient for interpolation)
    for the class of Lipschitz continuous strongly monotone (and maximally monotone) operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Warning:
        Lipschitz strongly monotone operators do not enjoy known interpolation conditions. The conditions implemented
        in this class are necessary but a priori not sufficient for interpolation. Hence, the numerical results
        obtained when using this class might be non-tight upper bounds (see Discussions in [1, Section 2]).

    Attributes:
        mu (float): strong monotonicity parameter
        L (float): Lipschitz parameter

    Lipschitz continuous strongly monotone operators are characterized by parameters :math:`\\mu` and `L`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import LipschitzStronglyMonotoneOperator
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=LipschitzStronglyMonotoneOperator, mu=.1, L=1.)

    References:
        `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
        Operator splitting performance estimation: Tight contraction factors and optimal parameter selection.
        SIAM Journal on Optimization, 30(3), 2251-2271.
        <https://arxiv.org/pdf/1812.00146.pdf>`_

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
            mu (float): The strong monotonicity parameter.
            L (float): The Lipschitz continuity parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Lipschitz continuous strongly monotone operators are necessarily continuous,
            hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )
        # Store L and mu
        self.mu = mu
        self.L = L

        if self.L == np.inf:
            print("\033[96m(PEPit) The class of Lipschitz strongly monotone operators is necessarily continuous.\n"
                  "To instantiate an operator, please avoid using the class LipschitzStronglyMonotoneOperator with\n"
                  " L == np.inf. Instead, please use the class StronglyMonotoneOperator (which accounts for the fact\n"
                  "that the image of the operator at certain points might not be a singleton).\033[0m")

    def set_strong_monotonicity_constraint_i_j(self,
                                               xi, gi, fi,
                                               xj, gj, fj,
                                               ):
        """
        Set strong monotonicity constraint for operators.

        """
        # Set constraint
        constraint = ((gi - gj) * (xi - xj) - self.mu * (xi - xj)**2 >= 0)

        return constraint

    def set_lipschitz_continuity_constraint_i_j(self,
                                                xi, gi, fi,
                                                xj, gj, fj,
                                                ):
        """
        Set Lipschitz continuity constraint for operators.

        """
        # Set constraint
        constraint = ((gi - gj) ** 2 - self.L ** 2 * (xi - xj) ** 2 <= 0)

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

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="lipschitz_continuity",
                                                      set_class_constraint_i_j=
                                                      self.set_lipschitz_continuity_constraint_i_j,
                                                      symmetry=True,
                                                      )
