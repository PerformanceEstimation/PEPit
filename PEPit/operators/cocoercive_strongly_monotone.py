import numpy as np
from PEPit.function import Function


class CocoerciveStronglyMonotoneOperator(Function):
    """
    The :class:`CocoerciveStronglyMonotoneOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing some necessary constraints verified by the class of cocoercive and strongly monotone (maximally) operators.

    Warnings:
        Those constraints might not be sufficient, thus the caracterized class might contain more operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        mu (float): strong monotonicity parameter
        beta (float): cocoercivity parameter

    Cocoercive operators are characterized by the parameters :math:`\\mu` and :math:`\\beta`, hence can be instantiated as

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
                 beta=1.,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
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

        Note:
            Cocoercive operators are necessarily continuous, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True)

        # Store the mu and beta parameters
        self.mu = mu
        self.beta = beta

        if self.mu == 0:
            print("\033[96m(PEPit) The class of cocoercive and strongly monotone operators is necessarily continuous."
                  " \n"
                  "To instantiate a cocoercive (non strongly) monotone opetator,"
                  " please avoid using the class CocoerciveStronglyMonotoneOperator\n"
                  "with mu == 0. Instead, please use the class CocoerciveOperator.\033[0m")

        if self.beta == 0:
            print("\033[96m(PEPit) The class of cocoercive and strongly monotone operators is necessarily continuous."
                  " \n"
                  "To instantiate a non cocoercive strongly monotone opetator,"
                  " please avoid using the class CocoerciveStronglyMonotoneOperator\n"
                  "with beta == 0. Instead, please use the class StronglyMonotoneOperator.\033[0m")

    def add_class_constraints(self):
        """
        Formulates a list of necessary constraints for self (cocoercive strongly monotone operator).
        """

        for point_i in self.list_of_points:

            xi, gi, fi = point_i

            for point_j in self.list_of_points:

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Necessary conditions of cocoercive strongly monotone operator class
                    self.list_of_class_constraints.append((gi - gj) * (xi - xj) - self.mu * (xi - xj) ** 2 >= 0)
                    self.list_of_class_constraints.append((gi - gj) * (xi - xj) - self.beta * (gi - gj) ** 2 >= 0)
