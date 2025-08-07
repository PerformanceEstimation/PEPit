import numpy as np

from PEPit.function import Function
from PEPit.expression import Expression
from PEPit import PSDMatrix


class SmoothQuadraticLojasiewiczFunctionExpensive(Function):
    """
    The :class:`SmoothQuadraticLojasiewiczFunctionExpensive` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing some constraints (which are not necessary and sufficient for interpolation)
    for the class of smooth (not necessarily convex) functions that also satisfy a quadratic Lojasiewicz inequality
    (sometimes also referred to as a Polyak-Lojasiewicz inequality). Extensive descriptions of such classes of
    functions can be found in [1, 2].
    
    The conditions implemented here are presented in [3, Proposition 3.4].

    Warning:
        Smooth functions satisfying a Lojasiewicz property do not enjoy known interpolation conditions.
        The conditions implemented in this class are necessary but a priori not sufficient for interpolation.
        Hence, the numerical results obtained when using this class might be non-tight upper bounds.

    Attributes:
        L (float): smoothness parameter
        mu (float): quadratic Lojasiewicz parameter

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import SmoothQuadraticLojasiewiczFunctionExpensive
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=SmoothQuadraticLojasiewiczFunctionExpensive, mu=.1, L=1.)

    References:

    `[1] S. Lojasiewicz (1963).
    Une propriété topologique des sous-ensembles analytiques réels.
    Les équations aux dérivées partielles, 117 (1963), 87–89.
    <https://aif.centre-mersenne.org/item/10.5802/aif.1384.pdf>`_

    `[2] J. Bolte, A. Daniilidis, and A. Lewis (2007).
    The Łojasiewicz inequality for nonsmooth subanalytic functions with applications to subgradient dynamical systems.
    SIAM Journal on Optimization 17, 1205–1223.
    <https://bolte.perso.math.cnrs.fr/Loja.pdf>`_

    `[3] A. Rubbens, J.M. Hendrickx, A. Taylor (2025).
    A constructive approach to strengthen algebraic descriptions of function and operator classes.
    <https://arxiv.org/pdf/2504.14377.pdf>`_

    """

    def __init__(self,
                 L,
                 mu,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """
        Args:
            L (float): The smoothness parameter.
            mu (float): The quadratic Lojasiewicz parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Smooth functions are necessarily differentiable, hence `reuse_gradient` is set to True.

        """

        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        assert 0 <= mu <= L

        # Store L and mu
        self.mu = mu
        self.L = L

    def last_call_before_problem_formulation(self):
        """
        Adds necessary variables to the PEP to be able to formulate the necessary interpolation conditions.
        Further, if the list of stationary points is empty, it adds a stationary point to it.
        
        """
        if self.list_of_stationary_points == list():
            self.stationary_point()

        nb_pts = len(self.list_of_points)
        preallocate = nb_pts ** 2
        self.M13 = np.ndarray((preallocate,), dtype=Expression)
        self.M14 = np.ndarray((preallocate,), dtype=Expression)
        self.M24 = np.ndarray((preallocate,), dtype=Expression)
        for i in range(preallocate):
            self.M13[i] = Expression()
            self.M14[i] = Expression()
            self.M24[i] = Expression()

    def set_quadratic_lojasiewicz_constraint_i_j(self,
                                                 xi, gi, fi,
                                                 xj, gj, fj,
                                                 ):
        """
        Formulates necessary interpolation constraints for quadratic Lojasiewicz functions, see [3, Proposition 3.4].
        
        """

        constraint = (fi - fj <= 1 / (2 * self.mu) * gi ** 2)

        return constraint

    def set_upper_quadratic_lojasiewicz_constraint_i_j(self,
                                                       xi, gi, fi,
                                                       xj, gj, fj,
                                                       ):
        """
        Formulates necessary interpolation constraints for smooth functions, see [3, Proposition 3.4].
        The name refers to the fact this inequality is very similar to the classical quadratic Lojasiewicz inequality
        in the reverse sense.
        
        """

        constraint = (fi - fj >= 1 / (2 * self.L) * gi ** 2)

        return constraint

    def add_class_constraints(self):
        """
        Formulates a list of necessary conditions for interpolation of self (smooth Lojasiewicz functions),
        see, e.g., discussions around [3, Proposition 3.4].

        """
        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_stationary_points,
                                                      constraint_name="quadratic_lojasiewicz",
                                                      set_class_constraint_i_j=self.set_quadratic_lojasiewicz_constraint_i_j,
                                                      )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_stationary_points,
                                                      constraint_name="upper_quadratic_lojasiewicz",
                                                      set_class_constraint_i_j=self.set_upper_quadratic_lojasiewicz_constraint_i_j,
                                                      )

        # Set function ID
        function_id = self.get_name()
        if function_id is None:
            function_id = "Function_{}".format(self.counter)

        # Browse list of points and create necessary constraints for interpolation [3, Lemma 3.4]
        counter = 0
        _, _, fs = self.list_of_stationary_points[0]
        for i, point_i in enumerate(self.list_of_points):

            xi, gi, fi = point_i
            xi_id = xi.get_name()
            if xi_id is None:
                xi_id = "Point_{}".format(i)

            for j, point_j in enumerate(self.list_of_points):

                xj, gj, fj = point_j
                xj_id = xj.get_name()
                if xj_id is None:
                    xj_id = "Point_{}".format(j)

                if point_i != point_j:
                    A = - fi + fj + 1 / 2 * (gi + gj) * (xi - xj) + 1 / (4 * self.L) * (gi - gj) ** 2 - self.L / 4 * (
                            xi - xj) ** 2
                    B = (self.L + self.mu) * (fi - fs - 1 / (2 * self.L) * gi ** 2)
                    C = (self.L - self.mu) * (fj - fs + 1 / (2 * self.L) * gj ** 2)

                    D = 2 * self.mu * (B - C - (self.L + 3 * self.mu) * A) / (2 * self.L + self.mu)
                    E = 4 * self.mu ** 2 * ((self.L + self.mu) * A - 2 * B) / (2 * self.L + self.mu) ** 2
                    F = - 2 * self.mu * A - D - E - 8 * self.mu ** 3 * B / (2 * self.L + self.mu) ** 3

                    M22 = - 6 * self.mu * A - D - 2 * self.M13[counter]
                    M33 = - 6 * self.mu * A - 2 * D - E - 2 * self.M24[counter]

                    T = np.array([[-2 * self.mu * A, 0, self.M13[counter], self.M14[counter]],
                                  [0, M22, -self.M14[counter], self.M24[counter]],
                                  [self.M13[counter], -self.M14[counter], M33, 0],
                                  [self.M14[counter], self.M24[counter], 0, F]], dtype=Expression)

                    psd_matrix = PSDMatrix(matrix_of_expressions=T,
                                           name="IC_{}_{}({}, {})".format(function_id,
                                                                          "smooth_quadratic_lojasiewicz_lmi",
                                                                          xi_id, xj_id))
                    self.list_of_class_psd.append(psd_matrix)
                    counter += 1
