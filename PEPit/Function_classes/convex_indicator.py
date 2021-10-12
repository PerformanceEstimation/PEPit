import numpy as np

from PEPit.function import Function


class ConvexIndicatorFunction(Function):
    """
    This routine implements the interpolation conditions for convex
    indicator functions with bounded domains. Two parameters may be provided:
           - a diameter D (D is nonnegative, and possibly infinite).

    Note: not defining the value of D automatically corresponds to
          set D to infinite.

    To generate a convex indicator function 'h' with diameter D=infinite and
    a radius R=1 from an instance of PEP called P:
    >> problem = pep()
    >> D= np.inf
    >> h = problem.declare_function('ConvexIndicator',{'D' :D});

    For details about interpolation conditions, we refer to the following
    references:

    [1] Taylor, Adrien B., Julien M. Hendrickx, and François Glineur.
    "Smooth strongly convex interpolation and exact worst-case
    performance of first-order methods."
    Mathematical Programming 161.1-2 (2017): 307-345.

    [2] Taylor, Adrien B., Julien M. Hendrickx, and François Glineur.
    "Exact Worst-case Performance of First-order Methods for Composite
    Convex Optimization."to appear in SIAM Journal on Optimization (2017)

    :param D:
    :param R:
    :return:
    """

    def __init__(self,
                 param,
                 is_leaf=True,
                 decomposition_dict=None,
                 is_differentiable=False):
        """
        Class of convex indicator functions.
        The differentiability is not necessarily verified.

        :param param: (dict) contains the values of mu and L
        :param is_leaf: (bool) If True, it is a basis function. Otherwise it is a linear combination of such functions.
        :param decomposition_dict: (dict) Decomposition in the basis of functions.
        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         is_differentiable=is_differentiable)

        # Store D ad R
        self.D = param['D']  # diameter

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
