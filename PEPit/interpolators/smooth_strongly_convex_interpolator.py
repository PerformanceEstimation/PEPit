import numpy as np
import cvxpy as cp

from PEPit.interpolator import Interpolator


class SmoothStronglyConvexInterpolator(Interpolator):
    """
    The class :class:`Interpolator` is designed to help identifying worst-case examples.
    
    This class implements the construction [1, Theorem 3.14] that allows to interpolate smooth strongly convex functions.
    That is, given a set of triplets :math:`(x_i,g_i,f_i)` for which there exists such an interpolating function
    (see [2, Theorem 4]).
    
    Interpolators are interfaced to PEPit via the :class:`Function`. More precisely, functions associated to
    interpolation conditions and constructive interpolating constructions are featured with the `get_interpolator()`
    method, that allows to receive an interpolator associated with the triplets of the corresponding solved PEP.
    
    A complete demo is available
    `here <https://github.com/PerformanceEstimation/PEPit/blob/master/ressources/demo/PEPit_demo_extract_worst_case_examples.ipynb>`_.


    Attributes:
        func (Function): PEPit function that contains the list of triplets.
        L (float): smoothness parameter for the interpolant.
        mu (float): strong convexity parameter for the interpolant.
        options (str): Either "lowest" or "highest"; determines whether to use the minimum or maximum possible interpolant.

    References:

    `[1] A. Taylor (2017).
    Convex interpolation and performance estimation of first-order methods for convex optimization.
    PhD thesis, UCLouvain.
    <https://dial.uclouvain.be/pr/boreal/object/boreal%3A182881/>`_

    `[2] A. Taylor, J. Hendrickx, F. Glineur (2017).
    Smooth strongly convex interpolation and exact worst-case performance of first-order methods.
    Mathematical Programming, 161(1-2), 307-345.
    <https://arxiv.org/pdf/1502.05666.pdf>`_
        
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
        """
        Implements the constraints necessary for the construction [1, Theorem 3.14] in the evaluate() method.

        """
        if self.L is not np.inf:
            constraint = (0 >= fj - fi + gj @ (xi - xj)
                          + 1 / 2 / (self.L-self.mu) * cp.norm(gi - gj, 2) ** 2
                          + self.mu * self.L / 2 / (self.L-self.mu) * cp.norm(xi - xj, 2) ** 2
                          - self.mu / (self.L - self.mu) * (gi-gj)@(xi-xj))
        else:
            constraint = (0 >= fj - fi + gj @ (xi - xj)
                          + self.mu / 2 * cp.norm(xi - xj, 2) ** 2)
        
        return constraint
                
    def evaluate(self, x):
        """
        Computes the specific (lowest/highest) smooth strongly convex interpolant.
        The interpolation is formulated and solved as an optimization problem [1, Theorem 3.14].
        
        """
        k = x.shape[0]
        if k > self.d:
            raise ValueError("Error: specified input to evaluate has dimension larger than target dimension")
        x_padded = np.pad(x, (0, self.d - k))
        
        fx = cp.Variable(1)
        gx = cp.Variable((self.d,))
        cons = []
        for i, point_i in enumerate(self.func.list_of_points):
            xi, gi, fi = point_i
            cons.append(self.__set_constraint__(xi.eval(), gi.eval(), fi.eval(), x_padded, gx, fx))
            cons.append(self.__set_constraint__(x_padded, gx, fx, xi.eval(), gi.eval(), fi.eval()))
        
        if self.options == 'highest':
            prob = cp.Problem(cp.Maximize(fx), cons)
        if self.options == 'lowest':
            prob = cp.Problem(cp.Minimize(fx), cons)
        prob.solve(verbose=False, solver=self.solver)

        return fx.value.squeeze()
