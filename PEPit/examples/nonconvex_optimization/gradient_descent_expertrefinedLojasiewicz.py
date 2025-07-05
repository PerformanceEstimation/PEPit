from PEPit import PEP
from PEPit.functions import ExpertRefined_LojasiewiczSmoothFunction
import numpy as np

def wc_gradient_descent_expertLojaciewicz(L, mu, gamma, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and satisfies a quadratic Lojasiewicz inequality:
    
    .. math:: f(x)-f_\\star \\leqslant \\frac{1}{2\\mu}\|\\nabla f(x) \|^2,
    
    details can be found in [1,2,3]. The example here relies on the :class:`ExpertRefined_LojasiewiczSmoothFunction`
    description of smooth Lojasiewicz functions (based on [4, Proposition 3.4]).

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\gamma)` such that the guarantee

    .. math:: f(x_n)-f_\\star \\leqslant \\tau(n, L, \\mu, \\gamma) (f(x_0) - f(x_\\star))

    is valid, where :math:`x_n` is the n-th iterates obtained with the gradient method with fixed step-size.

    **Algorithm**:
    Gradient descent is described as follows, for :math:`t \in \\{ 0, \\dots, n-1\\}`,

    .. math:: x_{t+1} = x_t - \\gamma \\nabla f(x_t),

    where :math:`\\gamma` is a step-size and.

    **Theoretical guarantee**: We compare with the guarantees from [4, Theorem 3].

    **References**:
    	`[1] S. Lojasiewicz (1963).
    	Une propriété topologique des sous-ensembles analytiques réels.
    	Les équations aux dérivées partielles, 117 (1963), 87–89.
    	<https://aif.centre-mersenne.org/item/10.5802/aif.1384.pdf>`_
    	
    	`[2] B. Polyak (1963).
    	Gradient methods for the minimisation of functionals
    	USSR Computational Mathematics and Mathematical Physics 3(4), 864–878.
    	<https://www.sciencedirect.com/science/article/abs/pii/0041555363903823>`_
    	
    	`[3] J. Bolte, A. Daniilidis, and A. Lewis (2007).
    	The Łojasiewicz inequality for nonsmooth subanalytic functions with applications to subgradient dynamical systems.
    	SIAM Journal on Optimization 17, 1205–1223.
    	<https://bolte.perso.math.cnrs.fr/Loja.pdf>`_
    	
    	`[4] H. Abbaszadehpeivasti, E. de Klerk, M. Zamani (2023).
    	Conditions for linear convergence of the gradient method for non-convex optimization.
    	Optimization Letters.
    	<https://arxiv.org/pdf/2204.00647>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): Lojasiewicz parameter.
        gamma (float): step-size.
        n (int): number of iterations.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Example:
        >>> L = 1
        >>> mu = .2
        >>> gamma = 1 / L
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_expertLojaciewicz(L=L, gamma=gamma, n=1, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 6x6
	(PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
	(PEPit) Setting up the problem: Adding initial conditions and general constraints ...
	(PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
	(PEPit) Setting up the problem: interpolation conditions for 1 function(s)
				Function 1 : Adding 8 scalar constraint(s) ...
				Function 1 : 8 scalar constraint(s) added
				Function 1 : Adding 25 lmi constraint(s) ...
		 	Size of PSD matrix 1: 4x4
		 	Size of PSD matrix 2: 4x4
		 	Size of PSD matrix 3: 4x4
		 	Size of PSD matrix 4: 4x4
		 	Size of PSD matrix 5: 4x4
		 	Size of PSD matrix 6: 4x4
		 	Size of PSD matrix 7: 4x4
		 	Size of PSD matrix 8: 4x4
		 	Size of PSD matrix 9: 4x4
		 	Size of PSD matrix 10: 4x4
		 	Size of PSD matrix 11: 4x4
		 	Size of PSD matrix 12: 4x4
		 	Size of PSD matrix 13: 4x4
		 	Size of PSD matrix 14: 4x4
		 	Size of PSD matrix 15: 4x4
		 	Size of PSD matrix 16: 4x4
		 	Size of PSD matrix 17: 4x4
		 	Size of PSD matrix 18: 4x4
		 	Size of PSD matrix 19: 4x4
		 	Size of PSD matrix 20: 4x4
		 	Size of PSD matrix 21: 4x4
		 	Size of PSD matrix 22: 4x4
		 	Size of PSD matrix 23: 4x4
		 	Size of PSD matrix 24: 4x4
		 	Size of PSD matrix 25: 4x4
				Function 1 : 25 lmi constraint(s) added
	(PEPit) Setting up the problem: additional constraints for 0 function(s)
	(PEPit) Compiling SDP
	(PEPit) Calling SDP solver
	(PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.3846731925021345
	(PEPit) Primal feasibility check:
			The solver found a Gram matrix that is positive semi-definite
			All required PSD matrices are indeed positive semi-definite up to an error of 5.789964695245766e-09
			All the primal scalar constraints are verified
	(PEPit) Dual feasibility check:
			The solver found a residual matrix that is positive semi-definite
			All the dual matrices to lmi are positive semi-definite
			All the dual scalar values associated with inequality constraints are nonnegative up to an error of 4.449496465102487e-09
	(PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 7.311915166714283e-07
	(PEPit) Final upper bound (dual): 0.38467319656798615 and lower bound (primal example): 0.3846731925021345 
	(PEPit) Duality gap: absolute: 4.065851655354891e-09 and relative: 1.0569625683838963e-08
	*** Example file: worst-case performance of gradient descent with fixed step-size ***
	*** 	 (smooth problem satisfying a Lojasiewicz inequality; expert version) ***
		PEPit guarantee:	 f(x_1) - f(x_*) <= 0.384673 (f(x_0)-f_*)
		Theoretical guarantee:	 f(x_1) - f(x_*) <= 0.384673 (f(x_0)-f_*)

    """
    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(ExpertRefined_LojasiewiczSmoothFunction, L=L, mu=mu)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition( func(x0) - fs <= 1)
    x = x0
    for i in range(n):
        g = func.gradient(x)
        x = x - gamma * g
    
    problem.set_performance_metric( func(x) - fs )
    
    # Solve
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 4 / 3 * L / n
    
    # for convenience (see bounds in [4, Theorem 3])
    m, mp = -L, mu
    if ( gamma >= 0 and gamma <= 1/L):
        theoretical_tau = (mp * ( 1 - L * gamma) + np.sqrt( (L-m) * (m-mp) * (2-L*gamma) * mp * gamma + (L-m)**2 )**2 / (L-m+mp)**2)
        theoretical_tau = theoretical_tau**n
    elif (gamma >= 1/L and gamma <= 3/( m + L + np.sqrt( m**2 - L *m + L**2 ) ) ) :
        theoretical_tau = ( ( L * gamma - 2 ) * ( m * gamma - 2 ) * mp * gamma ) / ( (L+m-mp) * gamma - 2) + 1
        theoretical_tau = theoretical_tau**n 
    elif ( gamma >= 3 / ( m + L + np.sqrt( m**2 - m * L + L**2) ) and gamma <= 2/L ):
        theoretical_tau = ( L * gamma - 1 )**2 / ( ( L * gamma - 1 )**2 + mp * gamma * ( 2 - L * gamma) )
        theoretical_tau = theoretical_tau**n 
    else:
        theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of gradient descent with fixed step-size ***')
        print('*** \t (smooth problem satisfying a Lojasiewicz inequality; expert version) ***')
        print('\tPEPit guarantee:\t f(x_1) - f(x_*) <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_1) - f(x_*) <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau
    
    
if __name__ == "__main__":
    L, mu, gamma, n = 1, .2, 1, 1
    verbose = 0
    pepit_tau, theoretical_tau = wc_gradient_descent_expertLojaciewicz(L=L, mu=mu, gamma=gamma, n=n, wrapper="cvxpy", solver=None, verbose=1)
