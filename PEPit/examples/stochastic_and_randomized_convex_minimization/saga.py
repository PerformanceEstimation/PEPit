import numpy as np

from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_saga(L, mu, n, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the finite sum convex minimization problem

    .. math:: F_\\star \\triangleq \\min_x \\left\\{F(x) \\equiv h(x) + \\frac{1}{n} \\sum_{i=1}^{n} f_i(x)\\right\\},

    where the functions :math:`f_i` are assumed to be :math:`L`-smooth :math:`\\mu`-strongly convex, and :math:`h` is
    closed, proper, and convex with a proximal operator readily available. In the sequel, we use the notation 
    :math:`\\mathbb{E}` for denoting the expectation over the uniform distribution of the index
    :math:`i \\sim \\mathcal{U}\\left([|1, n|]\\right)`,
    e.g., :math:`F(x)\\equiv h(x)+\\mathbb{E}[f_i(x)]`.

    This code computes the exact rate for a Lyapunov (or energy) function for **SAGA** [1].
    That is, it computes the smallest possible :math:`\\tau(n,L,\\mu)` such this Lyapunov function decreases geometrically

    .. math:: \\mathbb{E}[V^{(1)}] \\leqslant \\tau(n, L, \\mu) V^{(0)},
    
    where the value of the Lyapunov function at iteration :math:`t` is denoted by :math:`V^{(t)}` and is defined as

    .. math:: V^{(t)} \\triangleq \\frac{1}{n} \sum_{i=1}^n \\left(f_i(\\phi_i^{(t)}) - f_i(x^\\star) - \\langle \\nabla f_i(x^\\star); \\phi_i^{(t)} - x^\\star\\rangle\\right) + \\frac{1}{2 n \\gamma (1-\\mu \\gamma)} \\|x^{(t)} - x^\\star\\|^2,

    with :math:`\\gamma = \\frac{1}{2(\\mu n+L)}` (this Lyapunov function was proposed in [1, Theorem 1]).
    We consider the case :math:`t=0` in the code below, without loss of generality.

    In short, for given values of :math:`n`, :math:`L`, and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of :math:`\\mathbb{E}[V^{(1)}]`
    when :math:`V(x^{(0)}) \\leqslant 1`.

    **Algorithm**: One iteration of SAGA [1] is described as follows: at iteration :math:`t`, pick
    :math:`j\\in\\{1,\ldots,n\\}` uniformely at random and set:

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\phi_j^{(t+1)} & = & x^{(t)} \\\\
                w^{(t+1)} & = & x^{(t)} - \\gamma \\left[ \\nabla f_j (\\phi_j^{(t+1)}) - \\nabla f_j(\\phi_j^{(t)}) + \\frac{1}{n} \\sum_{i=1}^n(\\nabla f_i(\\phi^{(t)}))\\right] \\\\
                x^{(t+1)} & = & \\mathrm{prox}_{\\gamma h} (w^{(t+1)})\\triangleq \\arg\\min_x \\left\\{ \\gamma h(x)+\\frac{1}{2}\\|x-w^{(t+1)}\\|^2\\right\\}
            \\end{eqnarray}

    **Theoretical guarantee**: The following **upper** bound (empirically tight) can be found in [1, Theorem 1]:

    .. math:: \\mathbb{E}[V^{(t+1)}] \\leqslant \\left(1-\\gamma\\mu \\right)V^{(t)}

    **References**:

    `[1] A. Defazio, F. Bach, S. Lacoste-Julien (2014). SAGA: A fast incremental gradient method with support for
    non-strongly convex composite objectives. In Advances in Neural Information Processing Systems (NIPS).
    <http://papers.nips.cc/paper/2014/file/ede7e2b6d13a41ddf9f4bdef84fdc737-Paper.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        n (int): number of functions.
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
        >>> pepit_tau, theoretical_tau = wc_saga(L=1, mu=.1, n=5, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 27x27
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 6 function(s)
        			Function 1 : Adding 30 scalar constraint(s) ...
        			Function 1 : 30 scalar constraint(s) added
        			Function 2 : Adding 6 scalar constraint(s) ...
        			Function 2 : 6 scalar constraint(s) added
        			Function 3 : Adding 6 scalar constraint(s) ...
        			Function 3 : 6 scalar constraint(s) added
        			Function 4 : Adding 6 scalar constraint(s) ...
        			Function 4 : 6 scalar constraint(s) added
        			Function 5 : Adding 6 scalar constraint(s) ...
        			Function 5 : 6 scalar constraint(s) added
        			Function 6 : Adding 6 scalar constraint(s) ...
        			Function 6 : 6 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.9666666468916689
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 9.722904118628244e-09
        		All the primal scalar constraints are verified up to an error of 1.6644436180148978e-07
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 3.3298968511181845e-07
        (PEPit) Final upper bound (dual): 0.9666666567891312 and lower bound (primal example): 0.9666666468916689 
        (PEPit) Duality gap: absolute: 9.897462294183867e-09 and relative: 1.0238754306884701e-08
        *** Example file: worst-case performance of SAGA for Lyapunov function V_t ***
        	PEPit guarantee:	 V^(1) <= 0.966667 V^(0)
        	Theoretical guarantee:	 V^(1) <= 0.966667 V^(0)
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function and n smooth strongly convex ones
    h = problem.declare_function(ConvexFunction)
    fn = [problem.declare_function(SmoothStronglyConvexFunction,
                                   L=L, mu=mu, reuse_gradient=True) for _ in range(n)]

    # Define the objective as a linear combination of the former
    func = h + np.mean(fn)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()

    # Then define the initial points
    phi = [problem.set_initial_point() for _ in range(n)]
    x0 = problem.set_initial_point()

    # Compute the initial value of the Lyapunov function, for a given parameter
    gamma = 1 / 2 / (mu * n + L)
    c = 1 / 2 / gamma / (1 - mu * gamma) / n
    g, f = [x0 for _ in range(n)], [x0 for _ in range(n)]
    g0, f0 = [x0 for _ in range(n)], [x0 for _ in range(n)]
    gs, fs = [x0 for _ in range(n)], [x0 for _ in range(n)]
    init_lyapunov = c * (xs - x0) ** 2

    for i in range(n):
        g[i], f[i] = fn[i].oracle(phi[i])
        gs[i], fs[i] = fn[i].oracle(xs)
        init_lyapunov = init_lyapunov + 1 / n * (f[i] - fs[i] - gs[i] * (phi[i] - xs))

    # Set the initial constraint as the Lyapunov bounded by 1
    problem.set_initial_condition(init_lyapunov <= 1)

    # Compute the expected value of the Lyapunov function after one iteration
    # (so: expectation over n possible scenarios: one for each element fi in the function).
    final_lyapunov_avg = (xs - xs) ** 2
    for i in range(n):
        g0[i], f0[i] = fn[i].oracle(x0)
        w = x0 - gamma * (g0[i] - g[i])
        for j in range(n):
            w = w - gamma / n * g[j]
        x1, _, _ = proximal_step(w, h, gamma)
        final_lyapunov = c * (x1 - xs) ** 2
        for j in range(n):
            if i != j:
                gi, fi = g[j], f[j]
                final_lyapunov = final_lyapunov + 1 / n * (fi - fs[j] - gs[j] * (phi[j] - xs))
            else:
                gi, fi = g0[i], f0[i]
                final_lyapunov = final_lyapunov + 1 / n * (fi - fs[j] - gs[j] * (x0 - xs))
        final_lyapunov_avg = final_lyapunov_avg + final_lyapunov / n

    # Set the performance metric to the distance average to optimal point
    problem.set_performance_metric(final_lyapunov_avg)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison) : the bound is given in Theorem 1 of [1]
    theoretical_tau = (1 - gamma * mu)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of SAGA for Lyapunov function V_t ***')
        print('\tPEPit guarantee:\t V^(1) <= {:.6} V^(0)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t V^(1) <= {:.6} V^(0)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_saga(L=1, mu=.1, n=5, wrapper="cvxpy", solver=None, verbose=1)
