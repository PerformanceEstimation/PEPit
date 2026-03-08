import cvxpy as cp
import numpy as np
from PEPit.interpolator import Interpolator

class SmoothStronglyConvexInterpolator(Interpolator):
    """
    The class :class:`Interpolator` is designed to help identifying worst-case examples.
    Given a new coordinate vector (


    Attributes:
        func (Function): list of triplets
        options (str): 'lowest' or 'highest'
        d (int): dimension of the interpolation
        d_eff (int): effective dimension of the interpolation
        
    """
    def __init__(self, func, L=np.inf, mu=0, options='lowest'):
        
        super().__init__(func=func)
        self.L = L
        self.mu = mu
        self.options = options

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
    	
    	k = x.shape[0]
    	if k > self.d:
    	    raise ValueError("Error: specified input to evaluate has dimension larger than target dimension")
    	x_padded = np.pad(x, (0, self.d - k))
    	
    	fx = cp.Variable(1)
    	gx = cp.Variable((self.d,))
    	cons = []
    	for i, point_i in enumerate(self.func.list_of_points):
    	    xi, gi, fi = point_i
    	    cons.append(self.__set_constraint__(xi.eval(),gi.eval(),fi.eval(),x_padded,gx,fx))
    	    cons.append(self.__set_constraint__(x_padded,gx,fx,xi.eval(),gi.eval(),fi.eval()))
        
    	if self.options == 'highest':
    	    prob = cp.Problem(cp.Maximize(fx), cons, solver=self.solver)
    	if self.options == 'lowest':
    	    prob = cp.Problem(cp.Minimize(fx), cons, solver=self.solver )
    	prob.solve(verbose=False)
    	return fx.value.squeeze()
    	    
        
        
