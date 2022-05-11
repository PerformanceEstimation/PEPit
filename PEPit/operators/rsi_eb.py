from PEPit.function import Function


class RsiEbOperator(Function):
    """
    The :class:`RsiEbOperator` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing some constraints (which are not necessary and sufficient for interpolation)
    for the class of RSI and EB operators.

    Note:
        Operators'values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        mu (float): Restricted sequent inequality parameter
        L (float): Error bound parameter

    RSI EB operators are characterized by parameters :math:`\\mu` and `L`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import RsiEbOperator
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=RsiEbOperator, mu=.1, L=1)

    References:

        `[1] C. Guille-Escuret, B. Goujaud, A. Ibrahim, I. Mitliagkas (2022).
        Gradient Descent Is Optimal Under Lower Restricted Secant Inequality And Upper Error Bound.
        arXiv 2203.00342.
        <https://arxiv.org/pdf/2203.00342.pdf>`_

    """

    def __init__(self,
                 mu,
                 L=1,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """

        Args:
            mu (float): The restricted secant inequality parameter.
            L (float): The upper error bound parameter.
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
        # Store mu and L
        self.mu = mu
        self.L = L

    def add_class_constraints(self):
        """
        Formulates the list of necessary conditions for interpolation of self (Lipschitz strongly monotone and
        maximally monotone operator), see, e.g., discussions in [1, Section 2].
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_stationary_points):

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of strongly monotone operator class
                    self.add_constraint((gi - gj) * (xi - xj) - self.mu * (xi - xj)**2 >= 0)
                    # Interpolation conditions of Lipschitz operator class
                    self.add_constraint((gi - gj)**2 - self.L**2 * (xi - xj)**2 <= 0)
