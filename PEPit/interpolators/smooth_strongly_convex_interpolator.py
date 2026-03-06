import cvxpy as cp
import numpy as np
from PEPit.interpolator import Interpolator

class SmoothStronglyConvexInterpolator(Interpolator):
    """
    The class :class:`Interpolator` is designed to help identifying worst-case examples.

    Attributes:
        func (Function): list of triplets
        options (str): 'lowest' or 'highest'
        d (int): dimension of the interpolation
        d_eff (int): effective dimension of the interpolation
        
    """
    def __init__(self, func, options='lowest'):
        
        super().__init__(func=func)
        self.L = func.L
        self.mu = func.mu
        self.options = options
        self.d = 2 # EXTRACT
        # MUST CHECK THAT THE PROBLEM WAS EVALUATED
        

    def __set_constraint__(self,
                           xi, gi, fi,
                           xj, gj, fj,
                           ):
        if self.L is not np.inf:
            constraint = (0 >= fj - fi + gj @ (xi - xj)
                          + 1 / 2 / (self.L-self.mu) * cp.norm(gi - gj,2) ** 2
                          + self.mu * self.L / 2 / (self.L-self.mu) *  cp.norm(xi - xj,2) ** 2
                          - self.mu / (self.L - self.mu) * (gi-gj)@(xi-xj) )
        else:
            constraint = (0 >= fj - fi + gj @ (xi - xj)
                          + self.mu / 2 *  cp.norm(xi - xj,2) ** 2 )
        
        return constraint


    def evaluate(self, x):
    	
    	x_trimmed = x...
        fx = cp.Variable(1)
        gx = cp.Variable((self.d,))
        cons = []
        for i, point_i in enumerate(func.list_of_points):
            xi, gi, fi = point_i
            cons.append(self.__set_constraint__(xi.eval(),gi.eval(),fi.eval(),x,gx,fx))
            cons.append(self.__set_constraint__(x,gx,fx,xi.eval(),gi.eval(),fi.eval()))
            
        #self.x_list, self.g_list, self.f_list =  list_of_triplets
        #self.nb_eval = len(self.x_list)
        #for i in range(self.nb_eval):
        #    fi = self.f_list[i]
        #    gi = self.g_list[i]
        #    xi = self.x_list[i]
            
        
        if self.options == 'highest':
            prob = cp.Problem(cp.Maximize(fx), cons)
        if self.options == 'lowest':
            prob = cp.Problem(cp.Minimize(fx), cons)
        prob.solve(verbose=False)
        return fx.value
    	    
        
        
