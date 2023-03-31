import numpy as np
from PEPit.function import Function
from PEPit import Expression
from PEPit import PSDMatrix

class SmoothStronglyConvexQuadraticFunction(Function):
    """
    The :class:`SmoothStronglyConvexQuadraticFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    by implementing interpolation constraints of the class of smooth strongly convex quadratic functions.

    Attributes:
        mu (float): strong convexity parameter
        L (float): smoothness parameter

    Smooth strongly convex quadratic functions are characterized by parameters :math:`\\mu` and `L`, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import SmoothStronglyConvexQuadraticFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=SmoothStronglyConvexQuadraticFunction, mu=.1, L=1.)

    References:
        `[1] N. Bousselmi, J. Hendrickx, F. Glineur  (2023).
        Interpolation Conditions for Linear Operators and applications to Performance Estimation Problems.
        arXiv preprint
        <https://arxiv.org/pdf/2302.08781.pdf>`_

    """

    def __init__(self,
                 mu,
                 L=1.,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
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

        Note:
            Smooth strongly convex quadratic functions are necessarily differentiable, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True)

        # Store mu and L
        self.mu = mu
        self.L = L

        if self.L == np.inf:
            print("\033[96m(PEPit) The class of smooth strongly convex quadratic functions is necessarily differentiable.\n"
                  "To instantiate a strongly convex quadratic function, please avoid using the class SmoothStronglyConvexQuadraticFunction\n"
                  "with L == np.inf. Instead, please use the class StronglyConvexQuadraticFunction (which accounts for the fact\n"
                  "that there might be several subgradients at the same point).\033[0m")

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (smooth strongly convex quadratic function); see [1, Theorem 3.9].
        """
        
        N = len(self.list_of_points)
        T = np.empty([N, N], dtype = Expression)

        i = 0
        for point_i in self.list_of_points:

            xi, gi, fi = point_i

            j = 0
            for point_j in self.list_of_points:

                xj, gj, fj = point_j

                if point_i != point_j:
                    
                    self.list_of_class_constraints.append(xi*gj == xj*gi)
                    
                else:
                    self.list_of_class_constraints.append(fi == 0.5*xi*gi)

                T[i,j] = self.L*gi*xj - gi*gj - self.mu*self.L*xi*xj + self.mu*xi*gj
                j = j + 1
            i = i + 1
    
        psd_matrix = PSDMatrix(matrix_of_expressions=T)
        self.list_of_class_psd.append(psd_matrix)