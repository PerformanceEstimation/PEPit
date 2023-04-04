import numpy as np
from PEPit.function import Function
from PEPit import Expression
from PEPit import PSDMatrix
from PEPit.point import Point

class LinearOperator(Function):
    """
    The :class:`LinearOperator` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing interpolation constraints for the class of linear operators.

    Note:
        Operator values can be requested through `gradient` or todo, and `function values` should not be used.

    Attributes:
        L (float): singular values upper bound

    Linear operators are characterized by parameters :math:`L`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import LinearOperator
        >>> problem = PEP()
        >>> M = problem.declare_function(function_class=LinearOperator, L=1.)

    References:
        `[1] N. Bousselmi, J. Hendrickx, F. Glineur  (2023).
        Interpolation Conditions for Linear Operators and applications to Performance Estimation Problems.
        arXiv preprint
        <https://arxiv.org/pdf/2302.08781.pdf>`_

    """

    def __init__(self,
                 second_list_of_points=[],
                 L=1.,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True):
        """

        Args:
            L (float): The singular values upper bound.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.

        Note:
            Linear operators are necessarily continuous,
            hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True)
        # Store L and mu
        self.L = L

        if self.L == np.inf:
            print("\033[96m(PEPit) The class of Linear operators is necessarily continuous.\n"
                  "To instantiate an operator, please avoid using the class LinearOperator with\n"
                  " L == np.inf. \033[0m")

        self.second_list_of_points = second_list_of_points

    def add_class_constraints(self):
        """
        Formulates the list of necessary and sufficient conditions for interpolation of self
        (Linear operator), see [1, Theorem 3.1].
        """

        for point_xy in self.list_of_points:
            
            xi, yi, fi = point_xy
            
            for point_uv in self.second_list_of_points:
                
                uj, vj, hj = point_uv
                
                # Constraint X^T V = Y^T U
                self.list_of_class_constraints.append(xi*vj == yi*uj)

        N1 = len(self.list_of_points)
        T1 = np.empty([N1, N1], dtype = Expression)

        i = 0
        for point_i in self.list_of_points:

            xi, yi, fi = point_i
            
            j = 0
            for point_j in self.list_of_points:

                xj, yj, fj = point_j

                # Constraint Y^T Y \preceq L^2 X^T X
                T1[i,j] = (self.L**2)*xi*xj - yi*yj
                j = j + 1
            i = i + 1
            
        psd_matrix1 = PSDMatrix(matrix_of_expressions=T1)
        self.list_of_class_psd.append(psd_matrix1)
        
        N2 = len(self.second_list_of_points)
        T2 = np.empty([N2, N2], dtype = Expression)

        i = 0
        for point_i in self.second_list_of_points:

            ui, vi, hi = point_i
            
            j = 0
            for point_j in self.second_list_of_points:

                uj, vj, hj = point_j

                # Constraint V^T V \preceq L^2 U^T U
                T2[i,j] = (self.L**2)*ui*uj - vi*vj
                j = j + 1
            i = i + 1
            
        psd_matrix2 = PSDMatrix(matrix_of_expressions=T2)
        self.list_of_class_psd.append(psd_matrix2)
        
        
    def gradient_transpose(self, point):
        """
        Return the transpose of the operator evaluated at `point`, i.e. M^T(point)

        Args:
            point (Point): any point.

        Returns:
            Point: a gradient transpose (M^T) (:class:`Point`) of this :class:`Function` on point (:class:`Point`).

        """

        # Verify point is a Point
        assert isinstance(point, Point)

        # Call oracle but only return the gradient
        g = Point(is_leaf=True, decomposition_dict=None)
        f = Expression(is_leaf=True, decomposition_dict=None)
        # Store it
        self.second_list_of_points.append((point, g, f))
        
        return g