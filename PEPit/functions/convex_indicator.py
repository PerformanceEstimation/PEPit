import numpy as np

from PEPit.function import Function


class ConvexIndicatorFunction(Function):
    """
    The :class:`ConvexIndicatorFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    implementing the interpolation constraints of the class of convex indicator functions.

    Attributes:
        D (float): diameter of the feasible set

    Convex indicator functions are characterized by the parameter `D`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=ConvexIndicatorFunction, param={'D': 1})

    References:
        `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
        Smooth strongly convex interpolation and exact worst-case performance of first-order method.
        Mathematical Programming.<https://arxiv.org/pdf/1502.05666.pdf>`_

        `[2] A. Taylor, J. Hendrickx, F. Glineur (2017).
        Exact Worst-case Performance of First-order Methods for Composite Convex Optimization.
        SIAM Journal on Optimization.<https://arxiv.org/pdf/1512.07516.pdf>`_

    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """

        Args:
            param (dict): contains the values of D
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

        # Store the diameter D in an attribute
        self.D = param['D']

    def add_class_constraints(self):
        """
        Add constraints of convex indicator functions.
        """

        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j

                if xi == xj:
                    self.add_constraint(fi == 0)

                else:
                    self.add_constraint(gi * (xj - xi) <= 0)
                    if self.D != np.inf:
                        self.add_constraint((xi - xj) ** 2 <= self.D ** 2)
