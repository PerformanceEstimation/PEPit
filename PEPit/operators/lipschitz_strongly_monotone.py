from PEPit.function import Function


class LipschitzStronglyMonotoneOperator(Function):
    """
    The :class:`LipschitzStronglyMonotoneOperator` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing some constraints (which are not necessary and sufficient for interpolation)
    for the class of Lipschitz continuous strongly monotone (and maximally monotone) operators.

    Note:
        Operators'values can be requested through `gradient` and `function values` should not be used.

    Warning:
        Lipschitz strongly monotone operators do not enjoy known interpolation conditions. The conditions implemented
        in this class are necessary but a priori not sufficient for interpolation. Hence the numerical results
        obtained when using this class might be non-tight upper bounds (see Discussions in [1, Section 2]).

    Attributes:
        mu (float): strong monotonicity parameter
        L (float): Lipschitz parameter

    Lipschitz continuous strongly monotone operators are characterized by parameters :math:`\\mu` and `L`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=LipschitzStronglyMonotoneOperator, param={'mu': .1, 'L': 1})

    References:
        `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
        Operator splitting performance estimation: Tight contraction factors and optimal parameter selection.
        SIAM Journal on Optimization, 30(3), 2251-2271.
        <https://arxiv.org/pdf/1812.00146.pdf>`_

    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
        """

        Args:
            param (dict): contains the values of mu and L.
            is_leaf (bool): True if self is defined from scratch.
                            False is self is defined as linear combination of leaf .
            decomposition_dict (dict): decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        Note:
            Lipschitz continuous strongly monotone operators are necessarily continuous,
            hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True)
        # Store L and mu
        self.mu = param['mu']
        self.L = param['L']

    def add_class_constraints(self):
        """
        Formulates the list of necessary conditions for interpolation of self (Lipschitz strongly monotone and
        maximally monotone operator), see, e.g., discussions in [1, Section 2].
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of strongly monotone operator class
                    self.add_constraint((gi - gj) * (xi - xj) - self.mu * (xi - xj)**2 >= 0)
                    # Interpolation conditions of Lipschitz operator class
                    self.add_constraint((gi - gj)**2 - self.L**2 * (xi - xj)**2 <= 0)
