from PEPit.function import Function


class RsiEbFunction(Function):
    """
    The :class:`RsiEbFunction` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing the interpolation constraints of the class of functions verifying
    the "lower" restricted secant inequality (:math:`\\text{RSI}^-`) and the "upper" error bound (:math:`\\text{EB}^+`).

    Attributes:
        mu (float): Restricted sequent inequality parameter
        L (float): Error bound parameter

    :math:`\\text{RSI}^-` and :math:`\\text{EB}^+` functions are characterized by parameters :math:`\\mu` and `L`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import RsiEbFunction
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=RsiEbFunction, mu=.1, L=1)

    References:

        A definition of the class of :math:`\\text{RSI}^-` and :math:`\\text{EB}^+` functions can be found in [1].

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
                            False if self is defined as linear combination of leaf .
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
        Formulates the list of necessary conditions for interpolation of self, see [1, Theorem 1].
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_stationary_points):

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of RSI function class
                    self.list_of_class_constraints.append((gi - gj) * (xi - xj) - self.mu * (xi - xj)**2 >= 0)
                    # Interpolation conditions of EB function class
                    self.list_of_class_constraints.append((gi - gj)**2 - self.L**2 * (xi - xj)**2 <= 0)
