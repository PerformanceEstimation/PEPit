import numpy as np

from PEPit.expression import Expression
from PEPit.function import Function
from PEPit import PSDMatrix


class LipschitzStronglyMonotoneOperatorExpensive(Function):
    """
    The :class:`LipschitzStronglyMonotoneOperatorExpensive` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing some constraints (which are not necessary and sufficient for interpolation)
    for the class of Lipschitz continuous strongly monotone (and maximally monotone) operators. 
    Those conditions are presented in [1, Proposition 3.15] (details in [1, Appendix E]) and are stronger than
    those used in [2].

    Warning:
        Lipschitz strongly monotone operators do not enjoy known interpolation conditions. The conditions implemented
        in this class are necessary but a priori not sufficient for interpolation. Hence, the numerical results
        obtained when using this class might be non-tight upper bounds (see Discussions in [1, Section 2]).

    Note:
        Operator values can be requested through `gradient`, and `function values` should not be used.

    Attributes:
        mu (float): strong monotonicity parameter
        L (float): Lipschitz parameter

    Lipschitz continuous strongly monotone operators are characterized by parameters :math:`\\mu` and `L`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import LipschitzStronglyMonotoneOperatorExpensive
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=LipschitzStronglyMonotoneOperatorExpensive, mu=.1, L=1.)

    References:

    `[1] A. Rubbens, J.M. Hendrickx, A. Taylor (2025).
    A constructive approach to strengthen algebraic descriptions of function and operator classes.
    <https://arxiv.org/pdf/2504.14377.pdf>`_

    `[2] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
    Operator splitting performance estimation: Tight contraction factors and optimal parameter selection.
    SIAM Journal on Optimization, 30(3), 2251-2271.
    <https://arxiv.org/pdf/1812.00146.pdf>`_

    """

    def __init__(self,
                 mu,
                 L,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
            mu (float): The strong monotonicity parameter.
            L (float): The Lipschitz continuity parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Lipschitz continuous strongly monotone operators are necessarily continuous,
            hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )
        # Store L and mu
        self.mu = mu
        self.L = L

        if self.L == np.inf:
            print("\033[96m(PEPit) The class of Lipschitz strongly monotone operators is necessarily continuous.\n"
                  "To instantiate an operator, please avoid using the class LipschitzStronglyMonotoneOperator with\n"
                  " L == np.inf. Instead, please use the class StronglyMonotoneOperator (which accounts for the fact\n"
                  "that the image of the operator at certain points might not be a singleton).\033[0m")

    def last_call_before_problem_formulation(self):
        """
        Adds necessary variables to the PEP to be able to formulate the necessary interpolation conditions.
        """
        nb_pts = len(self.list_of_points)
        preallocate = nb_pts * (nb_pts ** 2 - 1)
        self.Mab = np.ndarray((9, preallocate), dtype=Expression)
        self.Mba = np.ndarray((9, preallocate), dtype=Expression)
        for i in range(preallocate):
            self.Mab[:, i] = [Expression() for _ in range(9)]
            self.Mba[:, i] = [Expression() for _ in range(9)]

    def get_psd_constraint_i_j_k(self,
                                 xi, ti,
                                 xj, tj,
                                 xk, tk,
                                 M, opt,
                                 ):
        """
        Formulates necessary interpolation constraints for self (Lipschitz strongly monotone operators).
        
        """

        if opt == 1:
            Aij = (ti - tj) ** 2 - self.L ** 2 * (xi - xj) ** 2
            Aik = (ti - tk) ** 2 - self.L ** 2 * (xi - xk) ** 2
            Ajk = (tk - tj) ** 2 - self.L ** 2 * (xk - xj) ** 2
            Bij = 2 * self.L * (- (ti - tj) * (xi - xj) + self.mu * (xi - xj) ** 2)
            Bik = 2 * self.L * (- (ti - tk) * (xi - xk) + self.mu * (xi - xk) ** 2)
            Bjk = 2 * self.L * (- (tk - tj) * (xk - xj) + self.mu * (xk - xj) ** 2)
        else:
            Bij = (ti - tj) ** 2 - self.L ** 2 * (xi - xj) ** 2
            Bik = (ti - tk) ** 2 - self.L ** 2 * (xi - xk) ** 2
            Bjk = (tk - tj) ** 2 - self.L ** 2 * (xk - xj) ** 2
            Aij = 2 * self.L * (- (ti - tj) * (xi - xj) + self.mu * (xi - xj) ** 2)
            Aik = 2 * self.L * (- (ti - tk) * (xi - xk) + self.mu * (xi - xk) ** 2)
            Ajk = 2 * self.L * (- (tk - tj) * (xk - xj) + self.mu * (xk - xj) ** 2)

        M14 = M[0]
        M15 = M[1]
        M16 = M[2]
        M17 = M[3]
        M26 = M[4]
        M27 = M[5]
        M34 = M[6]
        M37 = M[7]
        M46 = M[8]

        M25 = -M14
        M23 = -M15
        M35 = -M16
        M45 = -M27
        M56 = -M37
        M57 = -M46
        M55 = Aij + 2 * self.mu * Bij - Ajk - Aik - 2 * M17 - 2 * M26 - 2 * M34
        T = np.array([[-Bij, 0, 0, M14, M15, M16, M17],
                      [0, -Ajk, M23, 0, M25, M26, M27],
                      [0, M23, -Bij, M34, M35, 0, M37],
                      [M14, 0, M34, -Bjk, M45, M46, 0],
                      [M15, M25, M35, M45, M55, M56, M57],
                      [M16, M26, 0, M46, M56, -Aik, 0],
                      [M17, M27, M37, 0, M57, 0, -Bik]], dtype=Expression)

        return T

    def add_class_constraints(self):
        """
        Formulates the list of necessary conditions for interpolation of self (Lipschitz strongly monotone and
        maximally monotone operator), see, e.g., discussions in [2, Appendix E].
        
        """

        # Set function ID
        function_id = self.get_name()
        if function_id is None:
            function_id = "Function_{}".format(self.counter)

        # Browse list of points and create necessary constraints for interpolation [1, Proposition 3.15]
        counter = 0
        for i, point_i in enumerate(self.list_of_points):

            xi, ti, _ = point_i
            xi_id = xi.get_name()
            if xi_id is None:
                xi_id = "Point_{}".format(i)

            for j, point_j in enumerate(self.list_of_points):

                xj, tj, _ = point_j
                xj_id = xj.get_name()
                if xj_id is None:
                    xj_id = "Point_{}".format(j)

                for k, point_k in enumerate(self.list_of_points):
                    xk, tk, _ = point_k
                    xk_id = xk.get_name()
                    if xk_id is None:
                        xk_id = "Point_{}".format(k)

                    if not (point_i == point_j and point_i == point_k):
                        T = self.get_psd_constraint_i_j_k(xi, ti,
                                                          xj, tj,
                                                          xk, tk,
                                                          self.Mab[:, counter], 1)
                        psd_matrix = PSDMatrix(matrix_of_expressions=T,
                                               name="IC_{}_{}({}, {}, {})".format(function_id,
                                                                                  "lipschitz_strongly_monotone_lmi_1",
                                                                                  xi_id, xj_id, xk_id))
                        self.list_of_class_psd.append(psd_matrix)
                        T = self.get_psd_constraint_i_j_k(xi, ti,
                                                          xj, tj,
                                                          xk, tk,
                                                          self.Mba[:, counter], 0)
                        psd_matrix = PSDMatrix(matrix_of_expressions=T,
                                               name="IC_{}_{}({}, {}, {})".format(function_id,
                                                                                  "lipschitz_strongly_monotone_lmi_0",
                                                                                  xi_id, xj_id, xk_id))
                        self.list_of_class_psd.append(psd_matrix)
                        counter += 1
