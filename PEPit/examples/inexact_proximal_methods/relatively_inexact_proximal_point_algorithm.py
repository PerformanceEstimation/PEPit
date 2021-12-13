import numpy as np

from PEPit.pep import PEP
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.inexact_proximal_step import inexact_proximal_step


def wc_relatively_inexact_proximal_point_algorithm(n, gamma, sigma, verbose=True):
    """
    Consider the non-smooth convex minimization problem,

    .. math:: f_\\star \\triangleq \\min_x f(x)

    where :math:`f` is closed, convex, and proper. We denote by :math:`x_\\star` some optimal point of :math:`f` (hence
    :math:`0\\in\\partial f(x_\\star)`). An approximate proximal operator is assumed to be available.

    This code computes a worst-case guarantee for an **inexact proximal point method**. That is, it computes the
    smallest possible :math:`\\tau(n, \\gamma, \\sigma)` such that the guarantee

    .. math:: f(x_n) - f(x_\\star) \\leqslant \\tau(n, \\gamma, \\sigma) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the method, :math:`\\gamma` is some step-size, and :math:`\\sigma` is the
    level of accuracy of the approximate proximal point oracle.

    **Algorithm**: The approximate proximal point method under consideration is described by

    .. math:: x_{t+1} \\approx_{\\sigma} \\arg\\min_x \\left\\{ \\gamma f(x)+\\frac{1}{2} \\|x-x_t\\|^2 \\right\\},

    where the notation ":math:`\\approx_{\\sigma}`" corresponds to require the existence of some vector
    :math:`s_{t+1}\\in\\partial f(x_{t+1})` and :math:`e_{t+1}` such that

        .. math:: x_{t+1}  =  x_t - \\gamma (s_{t+1} - e_{t+1}) \\quad \\quad \\text{with }\\|e_{t+1}\\|^2  \\leqslant  \\frac{\\sigma^2}{\\gamma^2}\\|x_{t+1} - x_t\\|^2.

    We note that the case :math:`\\sigma=0` implies :math:`e_{t+1}=0` and this operation reduces to a standard proximal
    step with step-size :math:`\\gamma`.

    **Theoretical guarantee**: The following empirical **upper** bound is provided in [1, Section 3.5.1],

        .. math:: f(x_n) - f(x_\\star) \\leqslant \\frac{1 + \\sigma}{4 \\gamma n^{\\sqrt{1 - \\sigma^2}}}\\|x_0 - x_\\star\\|^2.

    **References**: The precise formulation is presented in [1, Section 3.5.1].

    `[1] M. Barre, A. Taylor, F. Bach (2020). Principled analyses and design of first-order methods with inexact
    proximal operators. <https://arxiv.org/pdf/2006.06041.pdf>`_

    Args:
        n (int): number of iterations.
        gamma (float): the step-size.
        sigma (float): noise parameter.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_relatively_inexact_proximal_point_algorithm(n=8, gamma=10, sigma=.65, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 18x18
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 88 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal_inaccurate (solver: SCS); optimal value: 0.00810915174704416
        *** Example file: worst-case performance of an inexact proximal point method in distance in function values ***
            PEP-it guarantee:		 f(x_n) - f(x_*) <= 0.00810915 ||x_0 - x_*||^2
            Theoretical guarantee:	 f(x_n) - f(x_*) <= 0.00849444 ||x_0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function.
    f = problem.declare_function(ConvexFunction, param={})

    # Start by defining its unique optimal point xs = x_*
    xs = f.stationary_point()

    # Then define the starting point z0, that is the previous step of the algorithm.
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the Inexact Proximal Point Method starting from x0
    x = [x0 for _ in range(n + 1)]
    opt = 'PD_gapII'
    for i in range(n):
        x[i + 1], _, fx, _, _, _, epsVar = inexact_proximal_step(x[i], f, gamma, opt)
        f.add_constraint(epsVar <= ((sigma / gamma) * (x[i + 1] - x[i])) ** 2)

    # Set the performance metric to the final distance in function values
    problem.set_performance_metric(f.value(x[n]) - f.value(xs))

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (1 + sigma) / (4 * gamma * n ** np.sqrt(1 - sigma ** 2))

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of an inexact proximal point method in distance in function values ***')
        print('\tPEP-it guarantee:\t\t f(x_n) - f(x_*) <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n) - f(x_*) <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_relatively_inexact_proximal_point_algorithm(n=8, gamma=10, sigma=.65, verbose=True)
