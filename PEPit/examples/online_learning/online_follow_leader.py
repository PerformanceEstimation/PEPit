import numpy as np
from PEPit import PEP
from PEPit.functions import ConvexLipschitzFunction
from PEPit.functions import ConvexIndicatorFunction
from PEPit.primitive_steps import proximal_step

def wc_online_follow_leader(M, D, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the online convex minimization problem, whose goal is to sequentially minimize the regret

    .. math:: R_n \\triangleq \\min_{x\\in Q} \sum_{i=1}^n f_i(x_i)-f_i(x_\\star),

    where the functions :math:`f_i` are :math:`M`-Lipschitz and convex, and where :math:`Q` is a
    bounded closed convex set with diameter upper bounded by :math:`D`, and where :math:`x_\\star\\in Q`
    is a reference point. Classical references on the topic include [1, 2].

    This code computes a worst-case guarantee for **follow the leader** (FTL).
    That is, it computes the smallest possible :math:`\\tau(n, M, D)` such that the guarantee

    .. math:: R_n \\leqslant \\tau(n, M, D)

    is valid for any such sequence of queries of FTL; that is, :math:`x_t` are the query points of OGD.

    In short, for given values of :math:`n`, :math:`M`, :math:`D`: 
    :math:`\\tau(n, M, D)` is computed as the worst-case value of :math:`R_n`.

    **Algorithm**:
    Follow the leader is described by

    .. math:: x_{t+1} \\in \\text{argmin}_{x\\in Q} \\left( \sum_{i=1}^t f_i(x) \\right).

    **Theoretical guarantee**: The follow the leader strategy is known to have a linear regret
    (see, e.g., [1, Chapter 5]); we do not compare to any guarantee here.


    **References**:

    `[1] E. Hazan (2016).
    Introduction to online convex optimization.
    Foundations and Trends in Optimization, 2(3-4), 157-325.
    <https://arxiv.org/pdf/1912.13213>`_

    `[2] F. Orabona (2025).
    A Modern Introduction to Online Learning.
    <https://arxiv.org/pdf/1912.13213>`_

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
        >>> M, D, n = 1,.5,2
        >>> pepit_tau, theoretical_tau = wc_online_follow_leader(M, D, n, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 10x10
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (0 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 3 function(s)
				Function 1 : Adding 9 scalar constraint(s) ...
				Function 1 : 9 scalar constraint(s) added
				Function 2 : Adding 4 scalar constraint(s) ...
				Function 2 : 4 scalar constraint(s) added
				Function 3 : Adding 15 scalar constraint(s) ...
				Function 3 : 15 scalar constraint(s) added
	(PEPit) Setting up the problem: additional constraints for 0 function(s)
	(PEPit) Compiling SDP
	(PEPit) Calling SDP solver
	(PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.9330127036100446
	(PEPit) Primal feasibility check:
			The solver found a Gram matrix that is positive semi-definite
			All the primal scalar constraints are verified up to an error of 5.581592632530885e-09
	(PEPit) Dual feasibility check:
			The solver found a residual matrix that is positive semi-definite
			All the dual scalar values associated with inequality constraints are nonnegative up to an error of 2.1864238886638586e-10
	(PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 2.363987770546799e-08
	(PEPit) Final upper bound (dual): 0.9330127041164931 and lower bound (primal example): 0.9330127036100446 
	(PEPit) Duality gap: absolute: 5.064484387418133e-10 and relative: 5.428097996760877e-10
	*** Example file: worst-case regret of online follow the leader ***
		PEPit guarantee:	 R_n <= 0.933013

    """
    # Instantiate PEP
    problem = PEP()
    
    M_list = [ M  for i in range(n)]
    gamma = D/M/np.sqrt(n)
    
    # Declare a sequence of M-Lipschitz convex functions fi and an indicator function with Diameter D
    fis = [problem.declare_function(ConvexLipschitzFunction, M=M_list[i])  for i in range(n)]
    h = problem.declare_function(function_class=ConvexIndicatorFunction, D=D)
    
    F = np.sum(fis)
    Ftot = F + h

    # Defining a reference point
    xRef = problem.set_initial_point()
    xRef,_,_ = proximal_step(xRef, h, 1) # project the reference point
    gRef, FRef = F.oracle(xRef)

    x1 = problem.set_initial_point()
    x1,_,_ = proximal_step(x1, h, 1) # project the reference point
    
    # Run n steps of gradient descent with step-size gamma
    x = x1
    x_saved = list()
    g_saved = list()
    f_saved = list()
    f_occ = h
    for i in range(n):
        x_saved.append(x-xRef)
        g, f = fis[i].oracle(x)
        f_saved.append(f-fis[i].value(xRef))
        g_saved.append(g)
        f_occ = f_occ + fis[i]
        if i < n-1:
            x = f_occ.stationary_point()
    
    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(np.sum(f_saved))

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee
    theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case regret of online follow the leader ***')
        print('\tPEPit guarantee:\t R_n <= {:.6}'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau

if __name__ == "__main__":
    M, D, n = 1,.5,2
    pepit_tau, theoretical_tau = wc_online_follow_leader(M, D, n, wrapper="cvxpy", solver=None, verbose=1) 
