import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.inexactproximal_step import inexact_proximal_step


def wc_orippm(n, gamma, sigma, verbose=True):
    """
    Consider the composite non-smooth convex minimization problem,
        min_x { f(x) }
    where f(x) is convex. Proximal operator is assumed to be available.

    This code computes a worst-case guarantee for an Optimized Inexact Proximal Point Method,
    where x_* = argmin_x (f(x)).

    That is, it computes the smallest possible tau(n,sigma,gamma) such that the guarantee
        f(x_n) - f(x_*) <= tau(n,sigma,gamma) * ||x_0 - x_*||^2.
    is valid, where z_n is the output os the operator, an z_* a fixed point of this operator.

    The precise formulation is presented in [1].
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
    f = problem.declare_function(ConvexFunction, {})

    # Start by defining its unique optimal point xs = x_*
    xs = f.optimal_point()

    # Then define the starting point x0, that is the previous step of the algorithm.
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute n steps of the Proximal Inexact Proximal Point Method starting from x0
    x, z, = x0, x0
    theta = 0

    opt = 'Orip-style'
    for i in range(n):
        theta = (1 + np.sqrt(4*theta**2 + 1))/2
        y = (1 - 1/theta) * x + 1/theta * z
        x, _, fx, _, v, _, epsVar = inexact_proximal_step(y, f, gamma, opt)
        z = z - 2 * gamma / (1 + sigma) * theta * v
        f.add_constraint(epsVar <= sigma/(1 + sigma) * v**2)

    # Set the performance metric to the final distance in function values
    problem.set_performance_metric(f.value(x) - f.value(xs))

    # Solve the PEP
    pepit_tau = problem.solve()

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (1 + sigma)/4/gamma/(theta**2)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Optimized Inexact Proximal Point Method in distance ***')
        print('\tPEP-it guarantee:\t  f(x_n) - f(x_*) <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n) - f(x_*) <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    # Choose random scheme parameters
    gamma = 2
    sigma = 3
    # Number of iterations
    n = 10

    pepit_tau, theoretical_tau = wc_orippm(n=n,
                                          gamma=gamma,
                                          sigma=sigma)