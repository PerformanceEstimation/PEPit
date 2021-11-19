from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction


class SmoothConvexFunction(SmoothStronglyConvexFunction):
    """
    SmoothConvexFunction class

    Attributes:
        L (float): smoothness constant

    Example:
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=SmoothConvexFunction, param={'L': 1})

    References:


    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 is_differentiable=True):
        """
        Convex smooth functions are characterized by their smoothness constant L.
        They are necessarily differentiable.

        Args:
            param (dict): contains the value of L
            is_leaf (bool): If True, it is a basis function. Otherwise it is a linear combination of such functions.
            decomposition_dict (dict): Decomposition in the basis of functions.

        """
        # Inherit from SmoothStronglyConvexFunction as a special case of it with mu=0.
        super().__init__(param={'mu': 0, 'L': param['L']},
                         is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         is_differentiable=is_differentiable)
