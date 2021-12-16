import numpy as np

from PEPit.function import Function


class ConvexIndicatorFunction(Function):
    """
    ConvexIndicatorFunction class

    Attributes:
        D (float): diameter of the feasible set

    Example:
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=ConvexIndicatorFunction, param={'D': 1})

    Note:
        The default value of D is infinity, which is equivalent to no constraint.

    References:
        [1] Taylor, Adrien B., Julien M. Hendrickx, and François Glineur.
        "Smooth strongly convex interpolation and exact worst-case
        performance of first-order methods."
        Mathematical Programming 161.1-2 (2017): 307-345.

        [2] Taylor, Adrien B., Julien M. Hendrickx, and François Glineur.
        "Exact Worst-case Performance of First-order Methods for Composite
        Convex Optimization."to appear in SIAM Journal on Optimization (2017)

    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False):
        """
        Convex indicator functions are characterized by the diameter of the feasible set.

        Args:
            param (dict): contains the values of mu and L
            is_leaf (bool): If True, it is a basis function. Otherwise it is a linear combination of such functions.
            decomposition_dict (dict): Decomposition in the basis of functions.
        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient)

        # Store the diameter D in an attribute
        self.D = param['D']

    def add_class_constraints(self):
        """
        Add constraints of convex indicator functions
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
