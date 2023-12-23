from math import sqrt

from PEPit import PEP
from PEPit.functions import ConvexLipschitzFunction


def wc_subgradient_method(M, n, gamma, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is convex and :math:`M`-Lipschitz. This problem is a (possibly non-smooth) minimization problem.

    This code computes a worst-case guarantee for the **subgradient method**. That is, it computes
    the smallest possible :math:`\\tau(n, M, \\gamma)` such that the guarantee

    .. math:: \\min_{0 \leqslant t \leqslant n} f(x_t) - f_\\star \\leqslant \\tau(n, M, \\gamma) 

    is valid, where :math:`x_t` are the iterates of the **subgradient method** after :math:`t\\leqslant n` steps,
    where :math:`x_\\star` is a minimizer of :math:`f`, and when :math:`\\|x_0-x_\\star\\|\\leqslant 1`.

    In short, for given values of :math:`M`, the step-size :math:`\\gamma` and the number of iterations :math:`n`,
    :math:`\\tau(n, M, \\gamma)` is computed as the worst-case value of
    :math:`\\min_{0 \leqslant t \leqslant n} f(x_t) - f_\\star` when  :math:`\\|x_0-x_\\star\\| \\leqslant 1`.

    **Algorithm**:
    For :math:`t\\in \\{0, \\dots, n-1 \\}`

        .. math::
            :nowrap:

            \\begin{eqnarray}
                g_{t} & \\in & \\partial f(x_t) \\\\
                x_{t+1} & = & x_t - \\gamma g_t
            \\end{eqnarray}

    **Theoretical guarantee**: The **tight** bound is obtained in [1, Section 3.2.3] and [2, Eq (2)]

        .. math:: \\min_{0 \\leqslant t \\leqslant n} f(x_t)- f(x_\\star) \\leqslant \\frac{M}{\\sqrt{n+1}}\|x_0-x_\\star\|,

    and tightness follows from the lower complexity bound for this class of problems, e.g., [3, Appendix A].

    **References**: Classical references on this topic include [1, 2].

    `[1] Y. Nesterov (2003).
    Introductory lectures on convex optimization: A basic course.
    Springer Science & Business Media.
    <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.693.855&rep=rep1&type=pdf>`_

    `[2] S. Boyd, L. Xiao, A. Mutapcic (2003).
    Subgradient Methods (lecture notes).
    <https://web.stanford.edu/class/ee392o/subgrad_method.pdf>`_

    `[3] Y. Drori, M. Teboulle (2016).
    An optimal variant of Kelley's cutting-plane method.
    Mathematical Programming, 160(1), 321-351.
    <https://arxiv.org/pdf/1409.2636.pdf>`_

    Args:
        M (float): the Lipschitz parameter.
        n (int): the number of iterations.
        gamma (float): step-size.
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
        >>> M = 2
        >>> n = 6
        >>> gamma = 1 / (M * sqrt(n + 1))
        >>> pepit_tau, theoretical_tau = wc_subgradient_method(M=M, n=n, gamma=gamma, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 9x9
        (PEPit) Setting up the problem: performance measure is the minimum of 7 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 64 scalar constraint(s) ...
        			Function 1 : 64 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.7559287513714278
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite
        		All the primal scalar constraints are verified
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative up to an error of 1.0475429120359347e-09
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 8.251581484359109e-08
        (PEPit) Final upper bound (dual): 0.7559287543574007 and lower bound (primal example): 0.7559287513714278 
        (PEPit) Duality gap: absolute: 2.9859729133718815e-09 and relative: 3.950071892297578e-09
        *** Example file: worst-case performance of subgradient method ***
        	PEPit guarantee:	 min_(0 \leq t \leq n) f(x_i) - f_* <= 0.755929 ||x_0 - x_*||
        	Theoretical guarantee:	 min_(0 \leq t \leq n) f(x_i) - f_* <= 0.755929 ||x_0 - x_*||
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex lipschitz function
    func = problem.declare_function(ConvexLipschitzFunction, M=M)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the subgradient method
    x = x0
    gx, fx = func.oracle(x)

    for _ in range(n):
        problem.set_performance_metric(fx - fs)
        x = x - gamma * gx
        gx, fx = func.oracle(x)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = M / sqrt(n + 1)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of subgradient method ***')
        print('\tPEPit guarantee:\t min_(0 \leq t \leq n) f(x_i) - f_* <= {:.6} ||x_0 - x_*||'.format(pepit_tau))
        print('\tTheoretical guarantee:\t min_(0 \leq t \leq n) f(x_i) - f_* <= {:.6} ||x_0 - x_*||'.format(
            theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    M = 2
    n = 6
    gamma = 1 / (M * sqrt(n + 1))
    pepit_tau, theoretical_tau = wc_subgradient_method(M=M, n=n, gamma=gamma, wrapper="cvxpy", solver=None, verbose=1)
