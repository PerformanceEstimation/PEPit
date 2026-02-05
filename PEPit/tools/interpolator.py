import cvxpy as cp
import numpy as np

class Interpolator(object):
    """
    The class :class:`Interpolator` is designed to help identifying worst-case examples.

    Attributes:
        list_of_triplets (list): list of triplets
        
        L (float): curvature of the quadratic upper bound on the function (possibly np.inf)
        m (float): curvature of the quadratic lower bound on the function (possibly np.inf)
        d (int): dimension
        
    """
    def __init__(self, list_of_triplets, mu, L, d, options='lowest'):
        self.x_list, self.g_list, self.f_list =  list_of_triplets
        self.nb_eval = len(self.x_list)
        self.L = L
        self.mu = mu
        self.d = d
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
        fx = cp.Variable(1)
        gx = cp.Variable((self.d,))
        cons = []
        for i in range(self.nb_eval):
            fi = self.f_list[i]
            gi = self.g_list[i]
            xi = self.x_list[i]
            cons.append(self.__set_constraint__(xi,gi,fi,x,gx,fx))
            cons.append(self.__set_constraint__(x,gx,fx,xi,gi,fi))
        
        if self.options == 'highest':
            prob = cp.Problem(cp.Maximize(fx), cons)
        if self.options == 'lowest':
            prob = cp.Problem(cp.Minimize(fx), cons)
        prob.solve(verbose=False)
        return fx.value
    	    
        
        
