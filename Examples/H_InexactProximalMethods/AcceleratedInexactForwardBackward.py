import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Primitive_steps.inexactproximal_step import inexact_proximal_step


def wc_aifb(mu, L, gamma, sigma, xi, zeta, A0, verbose=True):
    """
    Consider the composite non-smooth strongly convex minimization problem,
        min_x { f(x) + g(x)}
    where f(x) is smothh convex, and g is non-smooth strongly convex.
    Both proximal operators are assumed to be available.

    This code computes a worst-case guarantee for an Accelerated Hybrid Proximal Gradient,
    where x_* = argmin_x (f(x)).

    That is, it computes the smallest possible tau(mu,n,sigma,gamma) such that the guarantee
        Phi_n <= tau(mu,n,sigma,gamma) * Phi_{n+1}
    is valid, where phi_{n+1} = A_{n+1}(F(x_{n+1} - F_*) + (1 + mu * A_{n+1})/2 * ||z_{n+1} - x_*||^2.
    We are going to verify that :
        max(Phi_{n+1} - Phi_{n}) <= 0

    The moethod originates from [1, Section 4.3].

    [1] M. Barre, A. Taylor, F. Bach. Principled analyses and design of
    first-order methods with inexact proximal operators (2020).

    :param mu: (float) strong convexity parameter.
    :param L: (float) smoothness parameter.
    :param gamma: (float) the step size.
    :param sigma: (float) noise parameter.
    :param xi: (float) Lyapunov and scheme parameter.
    :param zeta: (float) Lyapunov and scheme parameter.
    :param A0: (float) Lyapunov parameter.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a non-smooth strongly convex function, and a smooth convex function.
    g = problem.declare_function(SmoothStronglyConvexFunction, {'mu': mu, 'L': np.inf})
    f = problem.declare_function(SmoothStronglyConvexFunction, {'mu':0, 'L':L})
    F = f + g

    # Start by defining its unique optimal point xs = x_*, and its associated function value xs = x_*.
    xs = F.optimal_point()
    fs = F.value(xs)

    # Then define the starting point z0 and x0, that is the previous step of the algorithm.
    x0 = problem.set_initial_point()
    z0 = problem.set_initial_point()
    f0 = f.value(x0)
    g0 = g.value(x0)

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Set the scheme parameters
    eta = (1 - zeta**2) * gamma
    opt = 'PD_gapI'
    a0 = (eta + 2 * A0 * eta * mu + np.sqrt(4 * eta * A0 * (A0 * mu + 1) * (eta * mu + 1) + eta ** 2)) / 2
    A1 = A0 + a0

    # Compute one step of the Accelerated Hybrid Proximal Gradient starting from x0
    y = x0 + (A1 - A0) * (A0 * mu + 1) / (A0 * mu * (2 * A1 - A0) + A1) * (z0 - x0)
    dfy, fy = f.oracle(y)
    x1, _, g1, w, v, _, epsVar = inexact_proximal_step(y - gamma * dfy, g, gamma, opt)
    f.add_constraint(epsVar <= sigma**2/2 * (y - x1)**2 + gamma**2 * zeta**2 / 2 * (v + dfy) ** 2 + xi/2)
    f1 = f.value(x1)
    z1 = z0 + (A1 - A0) / (A1 * mu + 1) * (mu * (w - z0) - (v + dfy))

    phi0 = A0 * (f0 + g0 - fs) + (1 + mu * A0) / 2 * (z0 - xs)**2
    phi1 = A1 * (f1 + g1 - fs) + (1 + mu * A1) / 2 * (z1 - xs) ** 2

    # Set the performance metric to the final distance between zn and zs
    problem.set_performance_metric(phi1 - phi0 - A1 / 2 / gamma * xi)

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
    L = 2
    # Choose scheme parameters
    sigma = 0.2
    gamma = (1 - sigma ** 2) / L # the step size should be in [0, (1 - sigma**2)/L
    # Choose the scheme (and Lyapunov) parameter
    zeta = 0.9
    xi = 3
    A0 = 1

    pepit_tau, theoretical_tau = wc_aifb(mu=mu,
                                        L=2,
                                        gamma=gamma,
                                        sigma=sigma,
                                        xi=xi,
                                        zeta=zeta,
                                        A0=A0)