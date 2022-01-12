import numpy as np

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def wc_sgd_overparametrized(L, mu, gamma, n, verbose=True):
    """
    Consider the finite sum minimization problem

    .. math:: F_\\star \\triangleq \\min_x \\left\\{F(x) \\equiv \\frac{1}{n} \\sum_{i=1}^n f_i(x)\\right\\},

    where :math:`f_1, ..., f_n` are :math:`L`-smooth and :math:`\\mu`-strongly convex. In addition, we assume a zero
    variance at the optimal point (which is denoted by :math:`x_\\star`):

    .. math:: \\mathbb{E}\\left[\\|\\nabla f_i(x_\\star)\\|^2\\right] = \\frac{1}{n} \\sum_{i=1}^n \\|\\nabla f_i(x_\\star)\\|^2 = 0,

    which happens for example in machine learning in the interpolation regime,
    that is if there exists a model :math:`x_\\star`
    such that the loss :math:`\\mathcal{L}` on any observation :math:`(z_i)_{i \\in [|1, n|]}`,
    :math:`\\mathcal{L}(x_\\star, z_i) = f_i(x_\\star)` is zero.

    This code computes a worst-case guarantee for one step of the **stochastic gradient descent** (SGD) in expectation,
    for the distance to optimal point. That is, it computes the smallest possible :math:`\\tau(L, \\mu, \\gamma, n)` such that

    .. math:: \\mathbb{E}\\left[\\|x_1 - x_\\star\\|^2\\right] \\leqslant \\tau(L, \\mu, \\gamma, n) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_1` is the output of one step of SGD.

    **Algorithm**: One iteration of SGD is described by:

    .. math::
        \\begin{eqnarray}
            \\text{Pick random }i & \\sim & \\mathcal{U}\\left([|1, n|]\\right), \\\\
            x_{t+1} & = & x_t - \\gamma \\nabla f_{i}(x_t),
        \\end{eqnarray}

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**: An empirically tight one-iteration guarantee is provided in the code of PESTO [1]:

        .. math:: \\mathbb{E}\\left[\\|x_1 - x_\\star\\|^2\\right] \\leqslant \\frac{1}{2}\\left(1-\\frac{\\mu}{L}\\right)^2 \\|x_0-x_\\star\\|^2,

    when :math:`\\gamma=\\frac{1}{L}`. Note that we observe the guarantee does not depend on the number `math:`n` of
    functions for this particular setting, thereby implying that the guarantees are also valid for expectation
    minimization settings (i.e., when :math:`n` goes to infinity).

    **References**: Empirically tight guarantee provided in code of [1]. Using SDPs for analyzing SGD-type method was
    proposed in [2, 3].

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017). Performance Estimation Toolbox (PESTO): automated worst-case
    analysis of first-order optimization methods. In 56th IEEE Conference on Decision and Control (CDC).
    <https://github.com/AdrienTaylor/Performance-Estimation-Toolbox>`_

    `[2] B. Hu, P. Seiler, L. Lessard (2020). Analysis of biased stochastic gradient descent using sequential
    semidefinite programs. Mathematical programming (to appear).
    <https://arxiv.org/pdf/1711.00987.pdf>`_

    `[3] A. Taylor, F. Bach (2019). Stochastic first-order methods: non-asymptotic and computer-aided analyses
    via potential functions. Conference on Learning Theory (COLT).
    <https://arxiv.org/pdf/1902.00947.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        gamma (float): the step-size.
        n (int): number of functions.
        verbose (bool): if True, print conclusion.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> mu = 0.1
        >>> L = 1
        >>> gamma = 1 / L
        >>> pepit_tau, theoretical_tau = wc_sgd_overparametrized(L=L, mu=mu, gamma=gamma, n=5, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 11x11
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (2 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 5 function(s)
                 function 1 : 2 constraint(s) added
                 function 2 : 2 constraint(s) added
                 function 3 : 2 constraint(s) added
                 function 4 : 2 constraint(s) added
                 function 5 : 2 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); optimal value: 0.8099999998856641
        *** Example file: worst-case performance of stochastic gradient descent with fixed step-size and with zero variance at the optimal point ***
            PEPit guarantee:		 E[||x_1 - x_*||^2] <= 0.81 ||x0 - x_*||^2
            Theoretical guarantee:	 E[||x_1 - x_*||^2] <= 0.81 ||x0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    fn = [problem.declare_function(SmoothStronglyConvexFunction, param={'L': L, 'mu': mu}) for _ in range(n)]
    func = np.mean(fn)

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the bounded variance and the distance between initial point and optimal one
    var = np.mean([f.gradient(xs) ** 2 for f in fn])

    problem.set_initial_condition(var <= 0.)
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute the *expected* distance to optimality after running one step of the stochastic gradient descent
    distavg = np.mean([(x0 - gamma * f.gradient(x0) - xs) ** 2 for f in fn])

    # Set the performance metric to the distance average to optimal point
    problem.set_performance_metric(distavg)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    kappa = L / mu
    theoretical_tau = (1 - 1 / kappa) ** 2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of stochastic gradient descent'
              ' with fixed step-size and with zero variance at the optimal point ***')
        print('\tPEPit guarantee:\t E[||x_1 - x_*||^2] <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t E[||x_1 - x_*||^2] <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    mu = 0.1
    L = 1
    gamma = 1 / L
    pepit_tau, theoretical_tau = wc_sgd_overparametrized(L=L, mu=mu, gamma=gamma, n=5, verbose=True)
