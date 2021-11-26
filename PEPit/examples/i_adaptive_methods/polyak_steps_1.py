from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_ps_1(L, mu, gamma, verbose=True):
    """
    Consider the minimization problem

    .. math:: f_\star = \\min_x f(x),

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex, and :math:`x_\\star=\\arg\\min_x f(x)`.

    This code computes a worst-case guarantee for a variant of a **gradient method** relying on **Polyak step sizes** (PS).
    That is, it computes the smallest possible :math:`\\tau(L, \\mu, \\gamma)` such that the guarantee

    .. math:: ||x_{k+1} - x_\\star||^2 \\leqslant \\tau(L, \\mu, \\gamma) ||x_{k} - x_\\star||^2

    is valid, where :math:`x_k` is the output of the gradient method with PS and :math:`\\gamma` is the effective
    value of the step size of the gradient method with PS.

    In short, for given values of :math:`L`, :math:`\\mu`, and :math:`\\gamma`, :math:`\\tau(L, \\mu, \\gamma)` is computed as the worst-case
    value of :math:`||x_{k+1} - x_\\star||^2` when :math:`||x_{k} - x_\\star||^2 \\leqslant 1`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{k+1} = x_k - \\gamma \\nabla f(x_k),

    where :math:`\\gamma` is a step size. The Polyak step size rule under consideration here corresponds to choosing
    of :math:`\\gamma` satisfying:

    .. math:: \\gamma || \\nabla f(x_k) ||^2 = 2  (f(x_k) - f_\star).

    **Theoretical guarantee**: The gradient method with the variant of Polyak step sizes under consideration enjoys the
    (tight) theoretical guarantee [1, Proposition 1]:

    .. math:: ||x_{k+1} - x_\\star||^2 \\leqslant  \\tau(L,\\mu,\\gamma) ||x_{k} - x_\\star||^2,

    where :math:`\\gamma` is the effective step size used at iteration :math:`k` and

    .. math::
            :nowrap:

            \\begin{eqnarray}
                \\tau(L,\\mu,\\gamma) &&= \\left\\{\\begin{array}{ll} \\frac{(\\gamma L-1)(1-\\gamma \\mu)}{\\gamma(L+\\mu)-1}  & \\text{if } \\gamma\in[\\tfrac{1}{L},\\tfrac{1}{\\mu}],\\\\
                0& \\text{otherwise.} \\end{array}\\right.
            \\end{eqnarray}

    **References**:
    [1] M. Barré, A. Taylor, A. d’Aspremont (2020). Complexity guarantees for Polyak steps with momentum.
    In Conference on Learning Theory (pp. 452-478).


    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param gamma: (float) the step size.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value

    Example:
        >>> L, mu = 1, 0.1
        >>> gamma = 1.5 / L
        >>> pepit_tau, theoretical_tau = wc_ps_1(L=L, mu=mu, gamma=gamma, verbose=True):
        (PEP-it) Setting up the problem: size of the main PSD matrix: 4x4
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (2 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
        		 function 1 : 6 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: MOSEK); optimal value: 0.6538461538466018
        *** Example file: worst-case performance of Polyak steps ***
	        PEP-it guarantee:       ||x_1 - x_*||^2  <= 0.653846 ||x_0 - x_*||^2
	        Theoretical guarantee:  ||x_1 - x_*||^2  <= 0.653846 ||x_0 - x_*||^2
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, param={'L': L, 'mu': mu})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value gn and fn
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Set the initial condition to the distance between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the Polayk steps at iteration 1
    x1 = x0 - gamma * g0
    _, _ = func.oracle(x1)

    # Set the initial condition to the Polyak step size
    problem.set_initial_condition(gamma * g0 ** 2 == 2 * (f0 - fs))

    # Set the performance metric to the distance between x_1 and x_* = xs
    problem.set_performance_metric((x1 - xs) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    if gamma >= 1/L and gamma <= 1/mu:
        theoretical_tau = (gamma * L - 1) * (1 - gamma * mu) / (gamma * (L + mu) - 1)
    else:
        theoretical_tau = 0.


    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of Polyak steps ***')
        print('\tPEP-it guarantee:\t\t||x_1 - x_*||^2  <= {:.6} ||x_0 - x_*||^2 '.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x_1 - x_*||^2  <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    mu = 0.1
    gamma = 2 / (L + mu)

    pepit_tau, theoretical_tau = wc_ps_1(L=L,
                                         mu=mu,
                                         gamma=gamma)
