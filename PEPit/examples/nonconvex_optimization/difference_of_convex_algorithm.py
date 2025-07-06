from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import shifted_optimization_step


def wc_difference_of_convex_algorithm(mu1, mu2, L1, L2, n, alpha = 0, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the minimization problem

    .. math:: F_\\star \\triangleq \\min_x f_1(x)-f_2(x),

    where :math:`f_1` and :math:`f_2` are convex functions, respectively :math:`L_1`-smooth and
    :math:`\\mu_1`-strongly convex and  :math:`L_2`-smooth and :math:`\\mu_2`-strongly convex.

    This code computes a worst-case guarantee for **DCA** (difference-of-convex algorithm, also known as the
    convex-concave procedure). That is, it computes the smallest possible :math:`\\tau(n, \\mu_1, L_1,\\mu_2, L_2)`
    such that the guarantee

    .. math:: \\min_{t\\leqslant n} \\|\\nabla f_1(x_t)-\\nabla f_2(x_t)\\|^2 \\leqslant \\tau(n, \\mu_1, L_1,\\mu_2, L_2) (f(x_0) - f(x_n))

    is valid, where :math:`x_n` is the n-th iterates obtained with DCA.

    **Algorithm**:
    DCA is described as follows, for :math:`t \in \\{ 0, \\dots, n-1\\}`,

    .. math:: x_{t+1} \\in \\argmin_x f_1(x) - \\langle \\nabla f_2(x_t), x\\rangle,
    

    **Theoretical guarantee**: The results are compared with [1, Theorem 3]; a more complete picture can be obtained from [2], also by
    possibly allowing for non-convex functions :math:`f_1` and :math:`f_2` (i.e., possibly negative values for :math:`\\mu_1`, :math:`\\mu_2`.

    **References**:

    `[1] H. Abbaszadehpeivasti, E. de Klerk, M. Zamani (2021).
    On the rate of convergence of the difference-of-convex algorithm (DCA).
    Journal of Optimization Theory and Applications, 202(1), 475-496.
    <https://arxiv.org/pdf/2109.13566>`_
    
    `[2] T. Rotaru, P. Patrinos, F. Glineur (2025).
    Tight Analysis of Difference-of-Convex Algorithm (DCA) Improves Convergence Rates for Proximal Gradient Descent.
    Journal of Optimization Theory and Applications, 202(1), 475-496.
    <https://arxiv.org/pdf/2503.04486>`_


    Args:
        mu1 (float): strong convexity parameter for f1.
        mu2 (float): strong convexity parameter for f2.
        L1 (float): smoothness parameter for f1.
        L2 (float): smoothness parameter for f2.
        alpha (float): boosting parameter (defaulted to 0).
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
        theoretical_tau (float): reference theoretical value [1, Theorem 3].

    Example:
        >>> L1, L2, mu1, mu2 = 2., 5., .15, .1
        >>> pepit_tau, theory = wc_difference_of_convex_algorithm(mu1=mu1, mu2=mu2, L1=L1, L2=L2, n=5, alpha = 0, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 15x15
        (PEPit) Setting up the problem: performance measure is the minimum of 6 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (7 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
				Function 1 : Adding 42 scalar constraint(s) ...
				Function 1 : 42 scalar constraint(s) added
				Function 2 : Adding 42 scalar constraint(s) ...
				Function 2 : 42 scalar constraint(s) added
	(PEPit) Setting up the problem: additional constraints for 0 function(s)
	(PEPit) Compiling SDP
	(PEPit) Calling SDP solver
	(PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.4911306712600766
	(PEPit) Primal feasibility check:
			The solver found a Gram matrix that is positive semi-definite
			All the primal scalar constraints are verified
	(PEPit) Dual feasibility check:
			The solver found a residual matrix that is positive semi-definite
			All the dual scalar values associated with inequality constraints are nonnegative up to an error of 8.62220939235375e-09
	(PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 1.719678585376814e-07
	(PEPit) Final upper bound (dual): 0.4911306767128882 and lower bound (primal example): 0.4911306712600766 
	(PEPit) Duality gap: absolute: 5.452811591144524e-09 and relative: 1.1102567830175294e-08
	*** Example file: worst-case performance of DCA ***
		PEPit guarantee:	 min_i ||f'(x_i)||^2 <= 0.491131 (f(x_0)-f_*)
        	Theoretical guarantee:	 min_i ||f'(x_i)||^2 <= 0.491131 (f(x_0)-f_*)
    
    """
    
    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    f1 = problem.declare_function(SmoothStronglyConvexFunction, L=L1, mu=mu1)
    f2 = problem.declare_function(SmoothStronglyConvexFunction, L=L2, mu=mu2)
    F = f1 - f2

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = F.stationary_point()
    Fs = F(xs)
	
    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()
    
    x = x0
    g1x = f1.gradient(x0)
    g2x = f2.gradient(x0)
    
    problem.set_initial_condition( f1(x) - f2(x) - Fs <= 1 )

    
    for i in range(n):
    	problem.set_performance_metric( (g1x-g2x)**2 )
    	problem.add_constraint( Fs <= f1.value(x) - f2.value(x) - 1/2/(L1-mu2) * (g1x-g2x)**2 )
    	y, _, _ = shifted_optimization_step(g2x, f1)
    	x = ( 1 + alpha ) * y - alpha * x
    	g1x, f1x = f1.oracle(x)
    	g2x, f1x = f2.oracle(x)
    	
    problem.set_performance_metric( (g1x-g2x)**2 )
    problem.add_constraint( Fs <= f1.value(x) - f2.value(x) - 1/2/(L1-mu2) * (g1x-g2x)**2 )
    
    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)
    
    # Compute theoretical guarantee (for comparison)
    if alpha == 0:
    	Delta = 1;
    	A = 2 * (L1 * L2 - mu1 * L2 * (L1 >= L2) - mu2 * L1 * (L2 > L1))
    	B = L1 + L2 + mu1 * (L1 / L2 - 3) * (L1 >= L2) + mu2 * (L2 / L1 - 3) * (L2 > L1)
    	C = (L1 * L2 - mu1 * L2 * (L1 >= L2) - mu2 * L1 * (L2 > L1)) / (L1 - mu2)
    	theoretical_tau = A * Delta / (B * n + C)
    else:
    	theoretical_tau = None
	
    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of DCA ***')
        print('\tPEPit guarantee:\t min_i ||f\'(x_i)||^2 <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t min_i ||f\'(x_i)||^2 <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":    
    L1, L2, mu1, mu2 = 2., 5., .15, .1
    pepit_tau, theoretical_tau = wc_difference_of_convex_algorithm(mu1=mu1, mu2=mu2, L1=L1, L2=L2, n=5, alpha = 0, wrapper="mosek", solver=None, verbose=1)
