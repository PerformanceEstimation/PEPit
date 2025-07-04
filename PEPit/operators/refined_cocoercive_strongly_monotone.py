from PEPit.function import Function
from PEPit.expression import Expression
from PEPit import PSDMatrix
import numpy as np


class Refined_CocoerciveStronglyMonotoneOperator(Function):
    """
    The :class:`CocoerciveStronglyMonotoneOperator` class overwrites the `add_class_constraints` method
    of :class:`Function`, implementing some necessary constraints verified by the class of cocoercive
    and strongly monotone (maximally) operators. Those conditions are presented in [1, Appendix F] and are
    stronger than those used in [2].

    Warnings:
        Those constraints might not be sufficient, thus the caracterized class might contain more operators.

    Note:
        Operator values can be requested through `gradient`, and `function values` should not be used.

    Attributes:
        mu (float): strong monotonicity parameter
        beta (float): cocoercivity parameter

    Cocoercive operators are characterized by the parameters :math:`\\mu` and :math:`\\beta`,
    hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.operators import CocoerciveStronglyMonotoneOperator
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=CocoerciveStronglyMonotoneOperator, mu=.1, beta=1.)

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
                 beta,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """

        Args:
            mu (float): The strong monotonicity parameter.
            beta (float): The cocoercivity parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf .
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Cocoercive operators are necessarily continuous, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=True,
                         name=name,
                         )

        # Store the mu and beta parameters
        self.mu = mu
        self.beta = beta

        if self.mu == 0:
            print("\033[96m(PEPit) The class of cocoercive and strongly monotone operators is necessarily continuous."
                  " \n"
                  "To instantiate a cocoercive (non strongly) monotone operator,"
                  " please avoid using the class CocoerciveStronglyMonotoneOperator\n"
                  "with mu == 0. Instead, please use the class CocoerciveOperator.\033[0m")

        if self.beta == 0:
            print("\033[96m(PEPit) The class of cocoercive and strongly monotone operators is necessarily continuous."
                  " \n"
                  "To instantiate a non cocoercive strongly monotone operator,"
                  " please avoid using the class CocoerciveStronglyMonotoneOperator\n"
                  "with beta == 0. Instead, please use the class StronglyMonotoneOperator.\033[0m")

        
    def last_call_before_problem_formulation(self):
        """
        Adds necessary variables to the PEP to be able to formulate the necessary interpolation conditions.
        """
        nb_pts = len(self.list_of_points)
        preallocate = nb_pts * (nb_pts**2-1)
        self.Mab = np.ndarray((9,preallocate),dtype=Expression)
        self.Mba = np.ndarray((9,preallocate),dtype=Expression)
        for i in range(preallocate):
        	 self.Mab[:,i] = [Expression() for _ in range(9)]
        	 self.Mba[:,i] = [Expression() for _ in range(9)]
        
    def get_psd_constraint_i_j_k(self,
                                                xi, ti,
                                                xj, tj,
                                                xk, tk,
                                                M, opt,
                                                ):
        """
        Formulates the necessary interpolation constraints for self (cocoercive strongly monotone operators).
        """
        
        if opt == 1:
        	Aij = - (ti-tj)*(xi-xj) + self.mu * (xi-xj)**2
        	Aik = - (ti-tk)*(xi-xk) + self.mu * (xi-xk)**2
        	Ajk = - (tk-tj)*(xk-xj) + self.mu * (xk-xj)**2
        	Bij = - (ti-tj)*(xi-xj) + self.beta * (ti-tj)**2
        	Bik = - (ti-tk)*(xi-xk) + self.beta * (ti-tk)**2
        	Bjk = - (tk-tj)*(xk-xj) + self.beta * (tk-tj)**2
        else:
        	Bij = - (ti-tj)*(xi-xj) + self.mu * (xi-xj)**2
        	Bik = - (ti-tk)*(xi-xk) + self.mu * (xi-xk)**2
        	Bjk = - (tk-tj)*(xk-xj) + self.mu * (xk-xj)**2
        	Aij = - (ti-tj)*(xi-xj) + self.beta * (ti-tj)**2
        	Aik = - (ti-tk)*(xi-xk) + self.beta * (ti-tk)**2
        	Ajk = - (tk-tj)*(xk-xj) + self.beta * (tk-tj)**2
        
        
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
        M55 = Aij - Ajk - Aik - 2*(1-2*self.beta*self.mu)*Bij - 2*M17 - 2*M26 - 2*M34
        T = np.array([[-Bij,0,0,M14,M15,M16,M17],
        		[0,-Ajk,M23,0,M25,M26,M27],
        		[0,M23,-Bij,M34,M35,0,M37],
        		[M14,0,M34,-Bjk,M45,M46,0],
        		[M15,M25,M35,M45,M55,M56,M57],
        		[M16,M26,0,M46,M56,-Aik,0],
        		[M17,M27,M37,0,M57,0,-Bik]], dtype=Expression)
                      	
        return T
        	 
    def add_class_constraints(self):
        """
        Formulates the list of necessary conditions for interpolation of self (cocoercive strongly monotone and
        maximally monotone operator), see, e.g., discussions in [2, Appendix F].
        """
        
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
                        				  self.Mab[:,counter], 1)
                        psd_matrix = PSDMatrix(matrix_of_expressions=T)
                        self.list_of_class_psd.append(psd_matrix)
                        T = self.get_psd_constraint_i_j_k(xi, ti,
                        				  xj, tj,
                        				  xk, tk,
                        				  self.Mba[:,counter],0)
                        psd_matrix = PSDMatrix(matrix_of_expressions=T)
                        self.list_of_class_psd.append(psd_matrix)
                        self.list_of_class_psd.append(psd_matrix)
                        counter += 1 
