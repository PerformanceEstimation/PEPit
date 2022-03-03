import numpy as np
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
        >>> from PEPit.functions import SmoothConvexFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=SmoothConvexFunction, L=1.)

    """

    def __init__(self,
                 L=1.,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
        """

        Args:
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            L (float): The smoothness parameter.

        Note:
            Smooth convex functions are necessarily differentiable, hence `reuse_gradient` is set to True.

        """
        # Inherit from SmoothStronglyConvexFunction as a special case of it with mu=0.
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         mu=0,
                         L=L)

        if self.L == np.inf:
            print("\033[96m(PEPit) The class of smooth convex functions is necessarily differentiable.\n"
                  "When setting L to infinity, please use the class of convex functions (ConvexFunction),\n "
                  "that allows to compute several subgradients at the same point each time one is required.\033[0m")
