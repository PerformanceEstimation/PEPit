from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction


class SmoothConvexFunction(SmoothStronglyConvexFunction):
    """
    The :class:`SmoothConvexFunction` implements smooth convex functions as particular cases
    of :class:`SmoothStronglyConvexFunction`.

    Attributes:
        L (float): smoothness parameter

    Smooth convex functions are characterized by the smoothness parameter `L`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=SmoothConvexFunction, param={'L': 1})

    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
        """

        Args:
            param (dict): contains the value of L
            is_leaf (bool): True if self is defined from scratch.
                            False is self is defined as linear combination of leaf .
            decomposition_dict (dict): decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        Note:
            Smooth convex functions are necessarily differentiable, hence `reuse_gradient` is set to True.

        """
        # Inherit from SmoothStronglyConvexFunction as a special case of it with mu=0.
        super().__init__(param={'mu': 0, 'L': param['L']},
                         is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True)
