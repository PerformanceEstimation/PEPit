import numpy as np
import pandas as pd
from PEPit.function import Function


class SmoothStronglyConvexFunction(Function):
    """
    The :class:`SmoothStronglyConvexFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    by implementing interpolation constraints of the class of smooth strongly convex functions.

    Attributes:
        mu (float): strong convexity parameter
        L (float): smoothness parameter

    Smooth strongly convex functions are characterized by parameters :math:`\\mu` and :math:`L`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import SmoothStronglyConvexFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=SmoothStronglyConvexFunction, mu=.1, L=1.)

    References:
        `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
        Smooth strongly convex interpolation and exact worst-case performance of first-order methods.
        Mathematical Programming, 161(1-2), 307-345.
        <https://arxiv.org/pdf/1502.05666.pdf>`_

    """

    def __init__(self,
                 mu,
                 L=1.,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
            mu (float): The strong convexity parameter.
            L (float): The smoothness parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Smooth strongly convex functions are necessarily differentiable, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store mu and L
        self.mu = mu
        self.L = L

        if self.L == np.inf:
            print("\033[96m(PEPit) The class of smooth strongly convex functions is necessarily differentiable.\n"
                  "To instantiate a strongly convex function, please avoid using the class SmoothStronglyConvexFunction\n"
                  "with L == np.inf. Instead, please use the class StronglyConvexFunction (which accounts for the fact\n"
                  "that there might be several subgradients at the same point).\033[0m")

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (smooth strongly convex function); see [1, Theorem 4].
        """

        for point_i in self.list_of_points:

            xi, gi, fi = point_i

            for point_j in self.list_of_points:

                xj, gj, fj = point_j

                if point_i != point_j:
                    # Interpolation conditions of smooth strongly convex functions class
                    constraint = (fi - fj >=
                                  gj * (xi - xj)
                                  + 1 / (2 * self.L) * (gi - gj) ** 2
                                  + self.mu / (2 * (1 - self.mu / self.L)) * (
                                          xi - xj - 1 / self.L * (gi - gj)) ** 2)
                    constraint.set_name("IC_{}({}, {})".format(self.name, xi.name, xj.name))
                    self.list_of_class_constraints.append(constraint)

    def get_class_constraint_duals(self):

        n = len(self.list_of_points)
        list_of_duals = [constraint.eval_dual() for constraint in self.list_of_class_constraints]
        assert len(list_of_duals) == n*(n-1)
        complete_list_of_duals = [0]
        for i in range(n-1):
            complete_list_of_duals += list_of_duals[i*n: (i+1)*n]
            complete_list_of_duals += [0]
        tab_of_duals = np.array(complete_list_of_duals).reshape(n, n)
        point_names = [point[0].name for point in self.list_of_points]
        df = pd.DataFrame(tab_of_duals, columns=point_names, index=point_names)
        if self.name:
            df.columns.name = "IC_{}".format(self.name)
        print(df)
