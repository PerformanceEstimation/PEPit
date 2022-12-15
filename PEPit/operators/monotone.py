from PEPit.function import Function


class MonotoneOperator(Function):
    """
    The :class:`MonotoneOperator` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing interpolation constraints for the class of maximally monotone operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    General maximally monotone operators are not characterized by any parameter, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import MonotoneOperator
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=MonotoneOperator)

    References:
        [1] H. H. Bauschke and P. L. Combettes (2017).
        Convex Analysis and Monotone Operator Theory in Hilbert Spaces.
        Springer New York, 2nd ed.

    """

    def __init__(self,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """
        Args:
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (maximally monotone operator),
        see, e.g., [1, Theorem 20.21].
        """

        for point_i in self.list_of_points:

            xi, gi, fi = point_i

            for point_j in self.list_of_points:

                xj, gj, fj = point_j

                if (xi != xj) | (gi != gj):
                    # Interpolation conditions of monotone operator class
                    self.list_of_class_constraints.append((gi - gj) * (xi - xj) >= 0)
