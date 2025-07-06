from PEPit.function import Function


class LojasiewiczSmoothFunction(Function):
    """
    The :class:`LojasiewiczSmoothFunction` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing some constraints (which are not necessary and sufficient for interpolation)
    for the class of smooth functions (not necessarily convex) that also satisfy a quadratic Lojasiewicz inequality
    (sometimes also referred to as a Polyak-Lojasiewicz inequality). Extensive descriptions of such classes of
    functions can be found in [1, 2].
    
    The conditions implemented here are presented in [4, Proposition 3.4] with smoothness conditions from [3].

    Warning:
        Smooth functions satisfying a Lojasiewicz property do not enjoy known interpolation conditions.
        The conditions implemented in this class are necessary but a priori not sufficient for interpolation.
        Hence, the numerical results obtained when using this class might be non-tight upper bounds.

    Attributes:
        mu (float): Lojasiewicz parameter
        L (float): Lipschitz parameter
        
    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import LojasiewiczSmoothFunction
        >>> problem = PEP()
        >>> h = problem.declare_function(function_class=LojasiewiczSmoothFunction, mu=.5, L=1.)

    References:
    	`[1] S. Lojasiewicz (1963).
    	Une propriété topologique des sous-ensembles analytiques réels.
    	Les équations aux dérivées partielles, 117 (1963), 87–89.
    	<https://aif.centre-mersenne.org/item/10.5802/aif.1384.pdf>`_
    	
    	`[2] J. Bolte, A. Daniilidis, and A. Lewis (2007).
    	The Łojasiewicz inequality for nonsmooth subanalytic functions with applications to subgradient dynamical systems.
    	SIAM Journal on Optimization 17, 1205–1223.
    	<https://bolte.perso.math.cnrs.fr/Loja.pdf>`_
    	
    	`[3] A. Taylor, J. Hendrickx, F. Glineur (2017).
    	Exact worst-case performance of first-order methods for composite convex optimization.
    	SIAM Journal on Optimization, 27(3):1283–1313.
    	<https://arxiv.org/pdf/1512.07516.pdf>`_
    	
    	`[4] A. Rubbens, J.M. Hendrickx, A. Taylor (2025).
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
            mu (float): The Lojasiewicz parameter.
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
        assert L >= 0
        assert mu >= 0
        assert L >= mu
        
        self.mu = mu
        self.L = L
        
    def last_call_before_problem_formulation(self):
        """
        Last call before modeling and solving the full PEP. Check that there is a stationnary point encoded; if not, create one.
        
        """
        if self.list_of_stationary_points == list():
            self.stationary_point()
        
    def set_LojaSimple(self,
                       xi, gi, fi,
                       xj, gj, fj,
                      ):
        """
        Formulates necessary interpolation constraints for self (smooth Lojasiewicz functions), see [4, Proposition 3.4].
        
        """
        
        
        constraint = (fi - fj <= gi**2 / 2 / self.mu)

        return constraint
    
    def set_LowerSimple(self,
                        xi, gi, fi,
                        xj, gj, fj,
                       ):
        """
        Formulates necessary interpolation constraints for self (smooth Lojasiewicz functions), see [4, Proposition 3.4].
        
        """
        
        constraint = (fi - fj >= gi**2 / 2 / self.L)

        return constraint
    
    def set_SmoothnessSimple(self,
                             xi, gi, fi,
                             xj, gj, fj,
                            ):
        """
        Formulates the list of interpolation constraints for self (smooth (not necessarily convex) function),
        see [3, Theorem 3.10].
        """
        
        constraint = (fi - fj >= 1/2 * (gi + gj) * (xi - xj) + 1 / (4 * self.L) * (gj - gi) ** 2 - self.L/4 * (xj - xi)**2 )

        return constraint
    
    def add_class_constraints(self):
        """
        Formulates necessary interpolation constraints for self (smooth Lojasiewicz functions).
        
        """
        
        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_stationary_points,
                                                      constraint_name="basic_Lojasiewicz",
                                                      set_class_constraint_i_j=self.set_LojaSimple,
                                                      )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_stationary_points,
                                                      constraint_name="lower_bound",
                                                      set_class_constraint_i_j=self.set_LowerSimple,
                                                      )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="smoothness",
                                                      set_class_constraint_i_j=self.set_SmoothnessSimple,
                                                      )
