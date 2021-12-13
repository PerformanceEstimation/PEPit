import numpy as np

from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_point_saga(L, mu, n, verbose=True):
    """
    Consider the finite sum minimization problem

    .. math:: F_\\star \\triangleq \\min_x {F(x) \\equiv \\frac{1}{n} \\left(f_1(x) + ... + f_n(x)\\right)},

    where :math:`f_1, \\dots, f_n` are assumed :math:`L`-smooth and :math:`\\mu`-strongly convex,
    and with proximal operator available.

    This code computes the exact rate for the Lyapunov function from the original **Point SAGA** paper,
    given in [1, Theorem 5].

    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

    .. math:: V(x_1) \\leqslant \\tau(n, L, \\mu) V(x_0)

    with

    .. math:: V(x) = \\frac{1}{L \\mu}\\frac{1}{n} \\sum_{i \\leq n} \\|\\nabla f_i(x) - \\nabla f_i(x_\\star)\\|^2 + \\|x - x_\\star\\|^2,

    where :math:`x_\\star` denotes the minimizer of :math:`F`.

    In short, for given values of :math:`n`, :math:`L`, and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of :math:`V(x_1)` when
    :math:`V(x_0) \\leqslant 1`.

    **Algorithm**:
    Point SAGA is described by

    .. math::
        \\begin{eqnarray}
            \\gamma & = & \\frac{\\sqrt{(n - 1)^2 + 4n\\frac{L}{\\mu}}}{2Ln} - \\frac{\\left(1 - \\frac{1}{n}\\right)}{2L} \\\\
            j & \\sim & \\mathcal{U}\\left([|1, n|]\\right) \\\\
            z_t & = & x_t + \\gamma \\left(g_j^t - \\frac{1}{n} \\sum_{i\\leq n}g_i^t \\right) \\\\
            x_{t+1} & = & \\mathrm{prox}(z_t, f_j, \\gamma) \\\\
            g_j^{t+1} & = & \\frac{1}{\\gamma}(z_t - x_{t+1})
        \\end{eqnarray}

    **Theoretical guarantee**:
    A theoretical **upper** bound is given in [1, Theorem 5].

    .. math:: V(x_1) \\leqslant \\frac{1}{1 + \\mu\\gamma} V(x_0)

    **References**:
    [1] Aaron Defazio. "A Simple Practical Accelerated Method for Finite Sums." (2014).

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        n (int): number of functions.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_point_saga(L=1, mu=.01, n=10, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 31x31
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 10 function(s)
                 function 1 : 2 constraint(s) added
                 function 2 : 2 constraint(s) added
                 function 3 : 2 constraint(s) added
                 function 4 : 2 constraint(s) added
                 function 5 : 2 constraint(s) added
                 function 6 : 2 constraint(s) added
                 function 7 : 2 constraint(s) added
                 function 8 : 2 constraint(s) added
                 function 9 : 2 constraint(s) added
                 function 10 : 2 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.9714053941143999
        *** Example file: worst-case performance of Point SAGA for a given Lyapunov function ***
            PEP-it guarantee:		 V1(x_0, x_*) <= 0.971405 VO(x_0, x_*)
            Theoretical guarantee:	 V1(x_0, x_*) <= 0.973292 VO(x_0, x_*)

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a sum of strongly convex functions
    fn = [problem.declare_function(SmoothStronglyConvexFunction, param={'L': L, 'mu': mu}) for _ in range(n)]
    func = np.mean(fn)

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the initial values
    phi = [problem.set_initial_point() for _ in range(n)]
    x0 = problem.set_initial_point()

    # Parameters of the scheme and of the Lyapunov function
    gamma = np.sqrt((n - 1) ** 2 + 4 * n * L / mu) / 2 / L / n - (1 - 1 / n) / 2 / L
    c = 1 / (mu * L)

    # Compute the initial value of the Lyapunov function
    init_lyapunov = (xs - x0) ** 2
    gs = [fn[i].gradient(xs) for i in range(n)]
    for i in range(n):
        init_lyapunov = init_lyapunov + c / n * (gs[i] - phi[i]) ** 2

    # Set the initial constraint as the Lyapunov bounded by 1
    problem.set_initial_condition(init_lyapunov <= 1.)

    # Compute the expected value of the Lyapunov function after one iteration
    # (so: expectation over n possible scenarios:  one for each element fi in the function).
    final_lyapunov_avg = (xs - xs) ** 2
    for i in range(n):
        w = x0 + gamma * phi[i]
        for j in range(n):
            w = w - gamma / n * phi[j]
        x1, gx1, _ = proximal_step(w, fn[i], gamma)
        final_lyapunov = (xs - x1) ** 2
        for j in range(n):
            if i != j:
                final_lyapunov = final_lyapunov + c / n * (phi[j] - gs[j]) ** 2
            else:
                final_lyapunov = final_lyapunov + c / n * (gs[j] - gx1) ** 2
        final_lyapunov_avg = final_lyapunov_avg + final_lyapunov / n

    # Set the performance metric to the distance average to optimal point
    problem.set_performance_metric(final_lyapunov_avg)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison) : the bound is given in theorem 5 of [1]
    kappa = mu * gamma / (1 + mu * gamma)
    theoretical_tau = (1 - kappa)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of Point SAGA for a given Lyapunov function ***')
        print('\tPEP-it guarantee:\t\t V1(x_0, x_*) <= {:.6} VO(x_0, x_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t V1(x_0, x_*) <= {:.6} VO(x_0, x_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_point_saga(L=1, mu=.01, n=10, verbose=True)
