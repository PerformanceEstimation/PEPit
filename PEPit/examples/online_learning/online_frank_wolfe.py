import numpy as np
from PEPit import PEP
from PEPit.functions import ConvexLipschitzFunction
from PEPit.functions import ConvexIndicatorFunction
from PEPit.primitive_steps import proximal_step
from PEPit.primitive_steps import linear_optimization_step

def wc_online_frank_wolfe(M, D, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the online convex minimization problem, whose goal is to sequentially minimize the regret

    .. math:: R_n \\triangleq \\min_{x\\in Q} \sum_{i=1}^n f_i(x_i)-f_i(x),

    where the functions :math:`f_i` are :math:`M`-Lipschitz and convex, and where :math:`Q` is a
    bounded closed convex set with diameter upper bounded by :math:`D`. We also denote by :math:`x_\\star\\in Q`
    the solution to the minimization problem defining :math:`R_n` (i.e., :math:`x_\\star` is a reference point).
    Classical references on the topic include [1, 2].

    This code computes a worst-case guarantee for **online Frank-Wolfe** (OFW), see [1, Algorithm 27];
    the code uses the choice [3, Section 2] here. That is, it computes the smallest possible
    :math:`\\tau(n, M, D)` such that the guarantee

    .. math:: R_n \\leqslant \\tau(n, M, D)

    is valid for any such sequence of queries of OFW; that is, :math:`x_t` are the query points of OFW.

    In short, for given values of :math:`n`, :math:`M`, :math:`D`: 
    :math:`\\tau(n, M, D)` is computed as the worst-case value of :math:`R_n`.

    **Algorithm**:
    Online Frank-Wolfe is described by
    
        .. math::
            \\begin{eqnarray}
                \\text{dir}_t & = & x_t-x_1 + \\eta \\sum_{s=1}^t g_s \\\\
                v_{t} & = & \\text{argmin}_{v\\in Q} \\langle \\text{dir}_t;v\\rangle\\\\
                x_{t+1} & = & (1-\\sigma) x_t + \\sigma v_t
            \\end{eqnarray}

    where :math:`\\eta=\\tfrac{D}{2M}\\left(\\frac{3}{n} \\right)^{3/4}` and :math:`\\sigma=\\min\\{1,\\sqrt{3/n}\\}`.

    **Theoretical guarantee**:
    We compare the numerical results with those of [3, Theorem 2.1]:

    .. math:: R_n \\leqslant \\frac{4}{3^{3/4}} MDn^{3/4}


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
        >>> M, D, n = 1,.5,2
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
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.9330127185046186
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 3.831994018595078e-09
        		All the primal scalar constraints are verified up to an error of 1.2383195857612606e-08
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 7.157025748731026e-08
        (PEPit) Final upper bound (dual): 0.9330127200996287 and lower bound (primal example): 0.9330127185046186 
        (PEPit) Duality gap: absolute: 1.595010012955811e-09 and relative: 1.709526549125938e-09
        *** Example file: worst-case regret of online Frank-Wolfe ***
        	PEPit guarantee:	 R_n <= 0.933013
        	Theoretical guarantee:	 R_n <= 1.47558
    
    """
    # Instantiate PEP
    problem = PEP()
    
    M_list = [ M  for i in range(n)]
    eta = D/2/M * (3/n)**(3/4)
    sigma = min([1,np.sqrt(3/n)])
    
    # Declare a sequence of M-Lipschitz convex functions fi and an indicator function with Diameter D
    fis = [problem.declare_function(ConvexLipschitzFunction, M=M_list[i])  for i in range(n)]
    h = problem.declare_function(function_class=ConvexIndicatorFunction, D=D)
    
    F = np.sum(fis)
    # Defining a reference point
    xRef = problem.set_initial_point()
    xRef,_,_ = proximal_step(xRef, h, 1) # project the reference point
    
    # Then define the starting point x0 of the algorithm
    x1 = problem.set_initial_point()
    x1,_,_ = proximal_step(x1, h, 1) # project the initial point
    
    # Run n steps of the online Conditional Gradient / Frank-Wolfe method starting from x1
    x = x1
    acc_g = 0 * xRef
    regret = 0 * xRef**2
    
    for i in range(n):
        g, f = fis[i].oracle(x)
        regret = regret + f - fis[i](xRef)
        acc_g = acc_g + g
        dir_t = (x-x1) + eta * acc_g
        v, _, _ = linear_optimization_step(dir_t, h)
        x = (1-sigma) * x + sigma * v
        
    # Set the performance metric to the function values accuracy
    problem.set_performance_metric(regret)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee
    theoretical_tau = 4/3**(3/4) * M * D * n**(3/4)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case regret of online Frank-Wolfe ***')
        print('\tPEPit guarantee:\t R_n <= {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t R_n <= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau

if __name__ == "__main__":
    M, D, n = 1,.5,2
    pepit_tau, theoretical_tau = wc_online_frank_wolfe(M=M, D=D, n=n, wrapper="cvxpy", solver=None, verbose=1) 
