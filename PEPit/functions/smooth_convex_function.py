from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction


class SmoothConvexFunction(SmoothStronglyConvexFunction):
    """
    Convex Smooth Function
    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 is_differentiable=True):
        """
        Class of convex smooth functions.
        The differentiability is necessarily verified.

        :param param: (dict) contains the value of L
        :param is_leaf: (bool) If True, it is a basis function. Otherwise it is a linear combination of such functions.
        :param decomposition_dict: (dict) Decomposition in the basis of functions.
        """
        # Inherit directly from SmoothStronglyConvexFunction as a special case of it with mu=0.
        super().__init__(param={'mu': 0, 'L': param['L']},
                         is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         is_differentiable=is_differentiable)
