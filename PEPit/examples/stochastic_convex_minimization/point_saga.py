import numpy as np

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_point_saga(L, mu, n, verbose=True):
    """
    Consider the finite sum minimization problem

    .. math:: F^\\star \\triangleq \\min_x \\left\\{F(x) \\equiv \\frac{1}{n} \\sum_{i=1}^n f_i(x)\\right\\},

    where :math:`f_1, \\dots, f_n` are :math:`L`-smooth and :math:`\\mu`-strongly convex, and with proximal operator
    readily available.

    This code computes a tight (one-step) worst-case guarantee using a Lyapunov function for **Point SAGA** [1].
    The Lyapunov (or energy) function at a point :math:`x` is given in [1, Theorem 5]:

    .. math:: V(x) = \\frac{1}{L \\mu}\\frac{1}{n} \\sum_{i \\leq n} \\|\\nabla f_i(x) - \\nabla f_i(x_\\star)\\|^2 + \\|x - x^\\star\\|^2,

    where :math:`x^\\star` denotes the minimizer of :math:`F`. The code computes the smallest possible
    :math:`\\tau(n, L, \\mu)` such that the guarantee (in expectation):

    .. math:: \\mathbb{E}\\left[V\\left(x^{(1)}\\right)\\right] \\leqslant \\tau(n, L, \\mu) V\\left(x^{(0)}\\right),

    is valid (note that we use the notation :math:`x^{(0)},x^{(1)}` to denote two consecutive iterates for convenience; as the
    bound is valid for all :math:`x^{(0)}`, it is also valid for any pair of consecutive iterates of the algorithm).

    In short, for given values of :math:`n`, :math:`L`, and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of :math:`\\mathbb{E}\\left[V\\left(x^{(1)}\\right)\\right]` when :math:`V\\left(x^{(0)}\\right) \\leqslant 1`.

    **Algorithm**:
    Point SAGA is described by

    .. math::
        \\begin{eqnarray}
            \\text{Set }\\gamma & = & \\frac{\\sqrt{(n - 1)^2 + 4n\\frac{L}{\\mu}}}{2Ln} - \\frac{\\left(1 - \\frac{1}{n}\\right)}{2L} \\\\
            \\text{Pick random }j & \\sim & \\mathcal{U}\\left([|1, n|]\\right) \\\\
            z^{(t)} & = & x_t + \\gamma \\left(g_j^{(t)} - \\frac{1}{n} \\sum_{i\\leq n}g_i^{(t)} \\right), \\\\
            x^{(t+1)} & = & \\mathrm{prox}_{\\gamma f_j}(z^{(t)})\\triangleq \\arg\\min_x\\left\\{ \\gamma f_j(x)+\\frac{1}{2} \\|x-z^{(t)}\\|^2 \\right\\}, \\\\
            g_j^{(t+1)} & = & \\frac{1}{\\gamma}(z^{(t)} - x^{(t+1)}).
        \\end{eqnarray}

    **Theoretical guarantee**: A theoretical **upper** bound is given in [1, Theorem 5].

    .. math:: \\mathbb{E}\\left[V\\left(x^{(t+1)}\\right)\\right] \\leqslant \\frac{1}{1 + \\mu\\gamma} V\\left(x^{(t)}\\right)

    **References**:

    `[1] A. Defazio (2016). A simple practical accelerated method for finite sums.
    Advances in Neural Information Processing Systems (NIPS), 29, 676-684.
    <https://proceedings.neurips.cc/paper/2016/file/4f6ffe13a5d75b2d6a3923922b3922e5-Paper.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        n (int): number of functions.
        verbose (bool): if True, print conclusion

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_point_saga(L=1, mu=.01, n=10, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 31x31
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 10 function(s)
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
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); optimal value: 0.9714053958034508
        *** Example file: worst-case performance of Point SAGA for a given Lyapunov function ***
	        PEPit guarantee:        E[V(x^(1))] <= 0.971405 V(x^(0))
	        Theoretical guarantee:  E[V(x^(1))] <= 0.973292 V(x^(0))

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
        print('\tPEPit guarantee:\t E[V(x^(1))] <= {:.6} V(x^(0))'.format(pepit_tau))
        print('\tTheoretical guarantee:\t E[V(x^(1))] <= {:.6} V(x^(0))'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_point_saga(L=1, mu=.01, n=10, verbose=True)
