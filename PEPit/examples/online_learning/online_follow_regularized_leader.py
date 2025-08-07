import numpy as np
from PEPit import PEP
from PEPit.functions import ConvexLipschitzFunction
from PEPit.functions import ConvexIndicatorFunction
from PEPit.primitive_steps import proximal_step


def wc_online_follow_regularized_leader(M, D, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the online convex minimization problem, whose goal is to sequentially minimize the regret

    .. math:: R_n \\triangleq \\min_{x\\in Q} \sum_{i=1}^n f_i(x_i)-f_i(x),

    where the functions :math:`f_i` are :math:`M`-Lipschitz and convex, and where :math:`Q` is a
    bounded closed convex set with diameter upper bounded by :math:`D`. We also denote by :math:`x_\\star\\in Q`
    the solution to the minimization problem defining :math:`R_n` (i.e., :math:`x_\\star` is a reference point).
    Classical references on the topic include [1, 2]; such algorithms were studied using the performance
    estimation technique in [3].

    This code computes a worst-case guarantee for **follow the regularized leader** (FTRL).
    That is, it computes the smallest possible :math:`\\tau(n, M, D)` such that the guarantee

    .. math:: R_n \\leqslant \\tau(n, M, D)

    is valid for any such sequence of queries of FTRL; that is, :math:`x_t` are the query points of OGD.

    In short, for given values of :math:`n`, :math:`M`, :math:`D`: 
    :math:`\\tau(n, M, D)` is computed as the worst-case value of :math:`R_n`.

    **Algorithm**:
    Follow the regularized leader is described by

    .. math:: x_{t+1} \\in \\text{argmin}_{x\\in Q} \\left( \sum_{i=1}^t f_i(x) + \\tfrac{\eta}{2}\\|x-x_1\\|^2 \\right).

    **Theoretical guarantee**: The follow the regularized leader strategy is known to enjoy sublinear regret
    (see, e.g., [1, Theorem 5.2]); we compare with the bound:

    .. math:: R_n \\leqslant MD\\sqrt{n}
    
    with a regularization strength :math:`\\eta=D/M/\\sqrt{n}`. 


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
        >>> M, D, n = 1, .5, 2
        >>> pepit_tau, theoretical_tau = wc_online_follow_regularized_leader(M=M, D=D, n=n, wrapper="cvxpy", solver=None, verbose=1)
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
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.7071068096527651
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 7.3442036099876085e-09
        		All the primal scalar constraints are verified up to an error of 2.9812280422092385e-08
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 1.6889222586942237e-08
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 1.2145136205173416e-07
        (PEPit) Final upper bound (dual): 0.707106811831799 and lower bound (primal example): 0.7071068096527651 
        (PEPit) Duality gap: absolute: 2.1790339532756775e-09 and relative: 3.08161924553622e-09
        *** Example file: worst-case regret of online follow the regularized leader ***
        	PEPit guarantee:	 R_n <= 0.707107
        	Theoretical guarantee:	 R_n <= 0.707107
    
    """
    # Instantiate PEP
    problem = PEP()

    M_list = [M for i in range(n)]
    eta = D / (M * np.sqrt(n))

    # Declare a sequence of M-Lipschitz convex functions fi and an indicator function with Diameter D
    fis = [problem.declare_function(ConvexLipschitzFunction, M=M_list[i]) for i in range(n)]
    h = problem.declare_function(function_class=ConvexIndicatorFunction, D=D)

    F = np.sum(fis)
    # Defining a reference point
    xRef = problem.set_initial_point()
    xRef, _, _ = proximal_step(xRef, h, 1)  # project the reference point
    gRef, FRef = F.oracle(xRef)

    # Then define the starting point x0 of the algorithm
    x1 = problem.set_initial_point()
    x1, _, _ = proximal_step(x1, h, 1)  # project the initial point

    # Run n steps of FTRL (regularization eta)
    f_occ = h
    x = x1
    x_saved = list()
    g_saved = list()
    f_saved = list()
    for i in range(n):
        x_saved.append(x - xRef)
        g, f = fis[i].oracle(x)
        f_saved.append(f)
        g_saved.append(g)
        f_occ = f_occ + fis[i]
        if i < n - 1:
            x, _, _ = proximal_step(x1, f_occ, eta)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(np.sum(f_saved) - FRef)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee
    theoretical_tau = M * D * np.sqrt(n)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case regret of online follow the regularized leader ***')
        print('\tPEPit guarantee:\t R_n <= {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t R_n <= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    M, D, n = 1, .5, 2
    pepit_tau, theoretical_tau = wc_online_follow_regularized_leader(M=M, D=D, n=n,
                                                                     wrapper="cvxpy", solver=None,
                                                                     verbose=1)
