import numpy as np

from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_saga(L, mu, n, verbose=True):
    """
    Consider the finite sum convex minimization problem

    .. math:: F^\star = \\min_x \\left\\{F(x) \\equiv h(x) + \\frac{1}{n} \\sum_{i=1}^{n} f_i(x)\\right\\},

    where the functions :math:`f_i` are assumed to be :math:`L`-smooth :math:`\\mu`-strongly convex, and :math:`h` is
    closed, proper, and convex with a proximal operator readily available.

    This code computes the exact rate for the Lyapunov function from the original SAGA work [1, Theorem 1].
    That is, it computes the smallest possible :math:`\\tau(n,L,\\mu)` such a Lyapunov function decreases geometrically

    .. math:: V^{(k+1)} \\leqslant \\tau(n,L,\\mu) V^{(k)},

    where the value of the Lyapunov function at iteration :math:`k`is denoted by :math:`V_k` is

    .. math:: V^{(k)} = \\frac{1}{n} \sum_{i=1}^n \\left(f_i(\\phi_i^{(k)}) - f_i(x^\\star) - \\langle \\nabla f_i(x^\\star); \\phi_i^{(k)} - x^\\star\\rangle\\right) + \\frac{1}{2 n \\gamma (1-\\mu \\gamma)} ||x^{(k)} - x^\\star||^2,

    with :math:`\\gamma = \\frac{1}{2(\\mu n+L)}`.

    **Algorithm**:
    One iteration of SAGA [1] is described as follows: at iteration :math:`k`, pick :math:`j\\in\\{1,\ldots,n\\}` uniformely at random and set:

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\phi_j^{(k+1)} &&= x^{(k)}\\\\
                w^{(k+1)} &&= x^{(k)} - \\gamma \\left[ \\nabla f_j (\\phi_j^{(k+1)}) - \\nabla f_j(\\phi_j^{(k)}) + \\frac{1}{n} \\sum_{i=1}^n(\\nabla f_i(\\phi^{(k)}))\\right]\\\\
                x^{(k+1)} &&= \mathrm{prox}_{\\gamma h} (w^{(k+1)})
            \\end{eqnarray}

    **Theoretical guarantee**: The following upper bound can be found in [1, Theorem 1]:

    .. math:: V^{(k+1)}\\leqslant \\left(1-\\gamma\\mu \\right)V^{(k)}

    **References**:
    [1] A. Defazio, F. Bach, S. Lacoste-Julien (2014). SAGA: A fast incremental gradient method with support for
    non-strongly convex composite objectives. Advances in neural information processing systems (NIPS).

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param n: (int) number of functions.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value

    Example:
        >>> wc, theory = wc_saga(L=1, mu=.1, n=5, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 27x27
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 6 function(s)
                 function 1 : 30 constraint(s) added
                 function 2 : 6 constraint(s) added
                 function 3 : 6 constraint(s) added
                 function 4 : 6 constraint(s) added
                 function 5 : 6 constraint(s) added
                 function 6 : 6 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: MOSEK); optimal value: 0.9666666451997894
        *** Example file: worst-case performance of SAGA for Lyapunov function V_k ***
	        PEP-it guarantee:       V^(k+1) <= 0.966667 V^k
	        Theoretical guarantee:  V^(k+1) <= 0.966667 V^k
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function and n smooth strongly convex ones
    h = problem.declare_function(ConvexFunction,
                                 param={}, is_differentiable=True)
    fn = [problem.declare_function(SmoothStronglyConvexFunction,
                                   param={'L': L, 'mu': mu}, is_differentiable=True) for _ in range(n)]

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
    init_lyapunov = c * (xs - x0) ** 2
    for i in range(n):
        gi, fi = fn[i].oracle(phi[i])
        gis, fis = fn[i].oracle(xs)
        init_lyapunov = init_lyapunov + 1 / n * (fi - fis - gis * (phi[i] - xs))

    # Set the initial constraint as the Lyapunov bounded by 1
    problem.set_initial_condition(init_lyapunov <= 1)

    # Compute the expected value of the Lyapunov function after one iteration
    # (so: expectation over n possible scenarios: one for each element fi in the function).
    final_lyapunov_avg = (xs - xs) ** 2
    for i in range(n):
        w = x0 - gamma * (fn[i].gradient(x0) - fn[i].gradient(phi[i]))
        for j in range(n):
            w = w - gamma/n * fn[j].gradient(phi[j])
        x1, _, _ = proximal_step(w, h, gamma)
        final_lyapunov = c * (x1 - xs) ** 2
        for j in range(n):
            gis, fis = fn[j].oracle(xs)
            if i != j:
                gi, fi = fn[j].oracle(phi[j])
                final_lyapunov = final_lyapunov + 1 / n * (fi - fis - gis * (phi[j] - xs))
            else:
                gi, fi = fn[j].oracle(x0)
                final_lyapunov = final_lyapunov + 1 / n * (fi - fis - gis * (x0 - xs))
        final_lyapunov_avg = final_lyapunov_avg + final_lyapunov / n

    # Set the performance metric to the distance average to optimal point
    problem.set_performance_metric(final_lyapunov_avg)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison) : the bound is given in Theorem 1 of [1]
    kappa = 1 / gamma / mu
    theoretical_tau = (1 - 1 / kappa)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of SAGA for Lyapunov function V_k ***')
        print('\tPEP-it guarantee:\t V^(k+1) <= {:.6} V^k'.format(pepit_tau))
        print('\tTheoretical guarantee:\t V^(k+1) <= {:.6} V^k'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 5
    L = 1
    mu = 0.1

    pepit_tau, theoretical_tau = wc_saga(L=L,
                                         mu=mu,
                                         n=n)
