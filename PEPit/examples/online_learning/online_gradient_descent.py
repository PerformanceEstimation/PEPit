import numpy as np
from PEPit import PEP
from PEPit.functions import ConvexLipschitzFunction
from PEPit.functions import ConvexIndicatorFunction
from PEPit.primitive_steps import proximal_step

def wc_online_gradient_descent(M, D, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the online convex minimization problem, whose goal is to sequentially minimize the regret

    .. math:: R_n \\triangleq \\min_{x\\in Q} \sum_{i=1}^n f_i(x_i)-f_i(x),

    where the functions :math:`f_i` are :math:`M`-Lipschitz and convex, and where :math:`Q` is a
    bounded closed convex set with diameter upper bounded by :math:`D`. We also denote by :math:`x_\\star\\in Q`
    the solution to the minimization problem defining :math:`R_n` (i.e., :math:`x_\\star` is a reference point).
    Classical references on the topic include [1, 2]; such algorithms were studied using the performance
    estimation technique in [3] and using the related IQCs in [4].

    This code computes a worst-case guarantee for **online gradient descent** (OGD) with a step-size :math:`\\gamma=D/M/\\sqrt{n}`.
    That is, it computes the smallest possible :math:`\\tau(n, M, D)` such that the guarantee

    .. math:: R_n \\leqslant \\tau(n, M, D)

    is valid for any such sequence of queries of OGD; that is, :math:`x_t` are the query points of OGD.

    In short, for given values of :math:`n`, :math:`M`, :math:`D`: 
    :math:`\\tau(n, M, D)` is computed as the worst-case value of :math:`R_n`.

    **Algorithm**:
    Online gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma \\nabla f_t(x_t),

    where :math:`\\gamma=D/M/\\sqrt{n}` is a step-size.

    **Theoretical guarantee**:
    We compare the numerical results with those of [2, Section 2.1.2]:

    .. math:: R_n \\leqslant MD\\sqrt{n}.


    **References**:

    `[1] E. Hazan (2016).
    Introduction to online convex optimization.
    Foundations and Trends in Optimization, 2(3-4), 157-325.
    <https://arxiv.org/pdf/1912.13213>`_

    `[2] F. Orabona (2025).
    A Modern Introduction to Online Learning.
    <https://arxiv.org/pdf/1912.13213>`_
    
    `[3] J. Weibel, P. Gaillard, W.M. Koolen, A. Taylor (2025).
    Optimized projection-free algorithms for online learning: construction and worst-case analysis
    <https://arxiv.org/pdf/2506.05855>`_
    
    `[4] F. Jakob, A. Iannelli (2025).
    Online Convex Optimization and Integral Quadratic Constraints: A new approach to regret analysis
    <https://arxiv.org/pdf/2503.23600?>`_

    Args:
        M (float): the Lipschitz parameter.
        D (float): the diameter of the set.
        n (int): time horizon.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> M,D,n = 1,.5,2
        >>> pepit_tau, theoretical_tau = wc_online_gradient_descent(M=M, D=D, n=n, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 10x10
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (0 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 3 function(s)
        			Function 1 : Adding 4 scalar constraint(s) ...
        			Function 1 : 4 scalar constraint(s) added
        			Function 2 : Adding 4 scalar constraint(s) ...
        			Function 2 : 4 scalar constraint(s) added
        			Function 3 : Adding 28 scalar constraint(s) ...
        			Function 3 : 28 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.7071068079799386
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 1.1705181160638522e-08
        		All the primal scalar constraints are verified up to an error of 4.569347711314009e-08
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 4.8776329641953e-09
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 2.614006627199603e-07
        (PEPit) Final upper bound (dual): 0.7071068126252953 and lower bound (primal example): 0.7071068079799386 
        (PEPit) Duality gap: absolute: 4.6453566548976255e-09 and relative: 6.569526134486629e-09
        *** Example file: worst-case regret of online gradient descent (fixed step-sizes) ***
        	PEPit guarantee:	 R_n <= 0.707107
        	Theoretical guarantee:	 R_n <= 0.707107
    
    """
    # Instantiate PEP
    problem = PEP()
    
    M_list = [ M  for i in range(n)]
    gamma = D/M/np.sqrt(n)
    
    # Declare a sequence of M-Lipschitz convex functions fi and an indicator function with Diameter D
    fis = [problem.declare_function(ConvexLipschitzFunction, M=M_list[i])  for i in range(n)]
    h = problem.declare_function(function_class=ConvexIndicatorFunction, D=D)
    
    F = np.sum(fis)
    # Defining a reference point
    xRef = problem.set_initial_point()
    xRef,_,_ = proximal_step(xRef, h, 1) # project the reference point
    gRef, FRef = F.oracle(xRef)
    
    # Then define the starting point x0 of the algorithm
    x1 = problem.set_initial_point()
    x1,_,_ = proximal_step(x1, h, 1) # project the initial point
    
    # Run n steps of gradient descent with step-size gamma
    x = x1
    g_saved = [gRef for _ in range(n)]
    f_saved = [FRef for _ in range(n)]
    for i in range(n):
        g_saved[i], f_saved[i] = fis[i].oracle(x)
        x,_,_ = proximal_step(x - gamma * g_saved[i], h, gamma)
        
    # Set the performance metric to the regret
    problem.set_performance_metric(np.sum(f_saved) - FRef)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee
    theoretical_tau = M*D*np.sqrt(n)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case regret of online gradient descent (fixed step-sizes) ***')
        print('\tPEPit guarantee:\t R_n <= {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t R_n <= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau

if __name__ == "__main__":
    M,D,n = 1,.5,2
    pepit_tau, theoretical_tau = wc_online_gradient_descent(M=M, D=D, n=n, wrapper="cvxpy", solver=None, verbose=1) 
