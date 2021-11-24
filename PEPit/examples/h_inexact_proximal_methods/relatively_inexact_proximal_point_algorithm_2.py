import numpy as np

from PEPit.pep import PEP
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.inexact_proximal_step import inexact_proximal_step


def wc_rippm2(n, gamma, sigma, verbose=True):
    """
    Consider the non-smooth convex minimization problem,
        min_x { f(x) }
    where f(x) is closed convex and proper. Proximal operator is assumed to be available.

    This code computes a worst-case guarantee for an Inexact Proximal Point Method,
    where x_\star = argmin_x (f(x)), that is :
        x_{n+1} = x_n - gamma * (f'(x_{n+1} - e)
    with gamma * (f(x_{n+1}) + f^*(v) - (v, x_{n+1}) <= sigma**2/2*||x_{n+1} - x_n||^2
    and with x = (x_{n - x_{n+1})/gamma

    That is, it computes the smallest possible tau(n,sigma,gamma) such that the guarantee
        f(x_n) - f(x_\star) <= tau(n,sigma,gamma) * ||x_0 - x_\star||^2.
    is valid, where z_n is the output os the operator, an z_\star a fixed point of this operator.

    The precise formulation is presented in [1, Section 3].
    [1] M. Barre, A. Taylor, F. Bach. Principled analyses and design of
    first-order methods with inexact proximal operators (2020).

    :param n: (int) number of iterations.
    :param gamma: (float) the step size.
    :param sigma: (float) noise parameter.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
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
    x = [x0 for i in range(n + 1)]
    opt = 'PD_gapIII'
    for i in range(n):
        x[i + 1], _, fx, _, _, _, epsVar = inexact_proximal_step(x[i], f, gamma, opt)
        f.add_constraint(epsVar <= (sigma ** 2) / 2 * (x[i + 1] - x[i]) ** 2)

    # Set the performance metric to the final distance in function values
    problem.set_performance_metric(f.value(x[n]) - f.value(xs))

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (1 + sigma) / (4 * gamma * n ** np.sqrt(1 - sigma ** 2))

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Inexact Proximal Point Method in distance ***')
        print('\tPEP-it guarantee:\t  f(x_n) - f(x_*) <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n) - f(x_*) <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    # Choose random scheme parameters
    gamma = 10
    sigma = np.sqrt(0.4)
    # Number of iterations
    n = 8

    pepit_tau, theoretical_tau = wc_rippm2(n=n,
                                           gamma=gamma,
                                           sigma=sigma)
