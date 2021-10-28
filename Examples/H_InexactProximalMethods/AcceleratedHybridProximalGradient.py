import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Primitive_steps.inexactproximal_step import inexact_proximal_step


def wc_ahpg(mu, gamma, sigma, A0, verbose=True):
    """
    Consider the composite non-smooth strongly convex minimization problem,
        min_x { f(x) }
    where f(x) is convex, for which the proximal operator is available.

    This code computes a worst-case guarantee for an Accelerated Hybrid Proximal Gradient,
    where x_* = argmin_x (f(x)).

    That is, it computes the smallest possible tau(mu,n,sigma,gamma) such that the guarantee
        Phi_n <= tau(mu,n,sigma,gamma) * Phi_{n+1}
    is valid, where phi_{n+1} = A_{n+1}(F(x_{n+1} - F_*) + (1 + mu * A_{n+1})/2 * ||z_{n+1} - x_*||^2.
    We are going to verify that :
        max(Phi_{n+1} - Phi_{n}) <= 0

    The moethod originates from [1} and was adapted to deal with strong convexity in [2].
    [1] R. D. Monteiro and B. F. Svaiter. An accelerated hybrid proximal
     extragradient method for convex optimization and its implications
     to second-order methods, SIAM Journal on Optimization (2013).

    [2] M. Barre, A. Taylor, F. Bach. Principled analyses and design of
    first-order methods with inexact proximal operators (2020).

    :param mu: (float) strong convexity parameter.
    :param gamma: (float) the step size.
    :param sigma: (float) noise parameter.
    :param A0: (float) Lyapunov parameter.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex and a smooth strongly convex function.
    f = problem.declare_function(SmoothStronglyConvexFunction, {'mu':mu, 'L':np.inf})

    # Start by defining its unique optimal point xs = x_*, and its associated function value xs = x_*.
    xs = f.optimal_point()
    fs = f.value(xs)

    # Then define the starting point z0 and x0, that is the previous step of the algorithm.
    x0 = problem.set_initial_point()
    z0 = problem.set_initial_point()
    f0 = f.value(x0)

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute one step of the Accelerated Hybrid Proximal Gradient starting from x0
    a0 = (gamma + 2 * A0 * gamma * mu + np.sqrt(4 * gamma * A0 * (A0 * mu + 1) * (gamma * mu + 1) + gamma**2))/2
    A1 = A0 + a0
    opt = 'PD_gapI'

    y = x0 + (A1 - A0) * (A0 * mu + 1) / (A0 * mu * (2 * A1 - A0) + A1) * (z0 - x0)
    x1, _, f1, w, v, _, epsVar = inexact_proximal_step(y, f, gamma, opt)
    f.add_constraint(epsVar <= sigma**2/2 * (y - x1)**2)
    z1 = z0 + (A1 - A0)/(A1 * mu + 1) * (mu * (w - z0) - v)

    phi0 = A0 * (f0 - fs) + (1 + mu * A0) / 2 * (z0 - xs)**2
    phi1 = A1 * (f1 - fs) + (1 + mu * A1) / 2 * (z1 - xs) ** 2

    # Set the performance metric to the final distance between zn and zs
    problem.set_performance_metric(phi1 - phi0)

    # Solve the PEP
    pepit_tau = problem.solve()

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 0.

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Accelerated Hybrid Proximal gradient in distance ***')
        print('\tPEP-it guarantee:\t  phi(n+1) - phi(n) <= {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t phi(n+1) - phi(n) <= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    # Choose the function parameter
    mu = 1
    # Choose scheme parameters
    gamma = 1
    sigma = 1
    # Choose the Lyapunov parameter
    A0 = 10

    pepit_tau, theoretical_tau = wc_ahpg(mu=mu,
                                        gamma=gamma,
                                        sigma=sigma,
                                        A0=A0)