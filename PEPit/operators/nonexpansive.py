import numpy as np
from PEPit.function import Function


class NonexpansiveOperator(Function):
    """
    The :class:`NonexpansiveOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of (possibly inconsistent) nonexpansive operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

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

        [2] J. Park, E. Ryu (2023).
        Accelerated Infeasibility Detection of Constrained Optimization and Fixed-Point Iterations.
        arXiv preprint:2303.15876.

    """

    def __init__(self,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
        """

        Args:
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        Note:
            Nonexpansive continuous operators are necessarily continuous, hence `reuse_gradient` is set to True.

            Setting self.v = None corresponds to case when a nonexpansive operator is consistent.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True)
        # Store the infimal displacement vector v
        self.v = None


    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (Nonexpansive operator),
        see [1, 2].
        """

        for point_i in self.list_of_points:

            xi, gi, fi = point_i

            for point_j in self.list_of_points:

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of nonexpansive operator class
                    self.list_of_class_constraints.append((gi - gj) ** 2 - (xi - xj) ** 2 <= 0)
        
        if self.v != None:

            for point_i in self.list_of_points:

                xi, gi, fi = point_i
                # Interpolation conditions of infimal displacement vector of nonexpansive operator class
                self.list_of_class_constraints.append(self.v ** 2 - (xi - gi) * self.v <= 0)
