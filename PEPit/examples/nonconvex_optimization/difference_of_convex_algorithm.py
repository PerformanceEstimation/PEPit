from PEPit import PEP
from PEPit.functions import SmoothFunction


def wc_difference_of_convex_algorithm(mu1, mu2, L1, L2, n, alpha = 0, wrapper="cvxpy", solver=None, verbose=1):(L, gamma, n, wrapper="cvxpy", solver=None, verbose=1):
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

    .. math:: x_{t+1} \\in \\argmin_x f_1(x) - \\langle \\nabla f_2(x_t), x\\rangke,
    

    **Theoretical guarantee**: The results are compared with

    **References**:

    `[1] Taylor, A. B. (2017).
    Convex interpolation and performance estimation of first-order methods for convex optimization.
    PhD Thesis, UCLouvain.
    <https://dial.uclouvain.be/downloader/downloader.php?pid=boreal:182881&datastream=PDF_01>`_

    `[2] H. Abbaszadehpeivasti, E. de Klerk, M. Zamani (2021).
    The exact worst-case convergence rate of the gradient method with fixed step lengths for L-smooth functions.
    Optimization Letters, 16(6), 1649-1661.
    <https://arxiv.org/pdf/2104.05468v3.pdf>`_

    Args:
        L (float): the smoothness parameter.
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
        >>> gamma = 1 / L
        >>> pepit_tau, theoretical_tau = wc_gradient_descent(L=L, gamma=gamma, n=5, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 7x7
        (PEPit) Setting up the problem: performance measure is the minimum of 6 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 30 scalar constraint(s) ...
        			Function 1 : 30 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.26666666551166657
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 4.5045561757111027e-10
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 4.491578307483797e-09
        (PEPit) Final upper bound (dual): 0.2666666657156721 and lower bound (primal example): 0.26666666551166657 
        (PEPit) Duality gap: absolute: 2.0400553468746807e-10 and relative: 7.650207583915017e-10
        *** Example file: worst-case performance of gradient descent with fixed step-size ***
        	PEPit guarantee:	 min_i ||f'(x_i)||^2 <= 0.266667 (f(x_0)-f_*)
        	Theoretical guarantee:	 min_i ||f'(x_i)||^2 <= 0.266667 (f(x_0)-f_*)
    
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
    
    Delta = 1;
    A = 2 * (L1 * L2 - mu1 * L2 * (L1 >= L2) - mu2 * L1 * (L2 > L1));
    B = L1 + L2 + mu1 * (L1 / L2 - 3) * (L1 >= L2) + mu2 * (L2 / L1 - 3) * (L2 > L1);
    C = (L1 * L2 - mu1 * L2 * (L1 >= L2) - mu2 * L1 * (L2 > L1)) / (L1 - mu2);
    theory = A * Delta / (B * n + C);
	
    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theory


if __name__ == "__main__":
    L = 1
    gamma = 1 / L
    pepit_tau, theoretical_tau = wc_gradient_descent(L=L, gamma=gamma, n=5, wrapper="cvxpy", solver=None, verbose=1)
