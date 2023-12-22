import numpy as np
from PEPit.function import Function
from PEPit import Expression
from PEPit import PSDMatrix

class SymmetricLinearOperator(Function):
    """
    The :class:`SymmetricLinearOperator` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing interpolation constraints for the class of symmetric linear operators.

    Note:
        Operator values can be requested through `gradient` and `function values` should not be used.

    Attributes:
        mu (float): eigenvalues lower bound
        L (float): eigenvalues upper bound

    Symmetric Linear operators are characterized by parameters :math:`\\mu` and `L`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import SymmetricLinearOperator
        >>> problem = PEP()
        >>> M = problem.declare_function(function_class=SymmetricLinearOperator, mu=.1, L=1.)

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
            mu (float): The eigenvalues lower bound.
            L (float): The eigenvalues upper bound.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        Note:
            Symmetric Linear operators are necessarily continuous,
            hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True)
        # Store L and mu
        self.mu = mu
        self.L = L

        if self.L == np.inf:
            print("\033[96m(PEPit) The class of Symmetric Linear operators is necessarily continuous.\n"
                  "To instantiate an operator, please avoid using the class SymmetricLinearOperator with\n"
                  " L == np.inf. \033[0m")

    def add_class_constraints(self):
        """
        Formulates the list of necessary and sufficient conditions for interpolation of self
        (Symmetric Linear operator), see [1, Theorem 3.3].
        """

        N = len(self.list_of_points)
        T = np.empty([N, N], dtype = Expression)

        i = 0
        for point_i in self.list_of_points:

            xi, gi, fi = point_i
            
            j = 0
            for point_j in self.list_of_points:

                xj, gj, fj = point_j
                
                if (i != j):
                
                    self.list_of_class_constraints.append(xi*gj == xj*gi)

                T[i,j] = self.L*gi*xj - gi*gj - self.mu*self.L*xi*xj + self.mu*xi*gj
                j = j + 1
            i = i + 1
            
        psd_matrix = PSDMatrix(matrix_of_expressions=T)
        self.list_of_class_psd.append(psd_matrix)