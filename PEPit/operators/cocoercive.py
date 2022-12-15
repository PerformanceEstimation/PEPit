import numpy as np
from PEPit.function import Function


class CocoerciveOperator(Function):
    """
    The :class:`CocoerciveOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of cocoercive (and maximally monotone) operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        beta (float): cocoercivity parameter

    Cocoercive operators are characterized by the parameter :math:`\\beta`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import CocoerciveOperator
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=CocoerciveOperator, beta=1.)

    References:
        `[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
        Operator splitting performance estimation: Tight contraction factors and optimal parameter selection.
        SIAM Journal on Optimization, 30(3), 2251-2271.
        <https://arxiv.org/pdf/1812.00146.pdf>`_

    """

    def __init__(self,
                 beta=1.,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
        """

        Args:
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

        # Store the beta parameter
        self.beta = beta

        if self.beta == 0:
            print("\033[96m(PEPit) The class of cocoercive operators is necessarily continuous. \n"
                  "To instantiate a monotone opetator, please avoid using the class CocoerciveOperator\n"
                  "with beta == 0. Instead, please use the class Monotone (which accounts for the fact \n"
                  "that the image of the operator at certain points might not be a singleton).\033[0m")

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (cocoercive maximally monotone operator),
        see, e.g., [1, Proposition 2].
        """

        for point_i in self.list_of_points:

            xi, gi, fi = point_i

            for point_j in self.list_of_points:

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of cocoercive operator class
                    self.list_of_class_constraints.append((gi - gj) * (xi - xj) - self.beta * (gi - gj) ** 2 >= 0)
