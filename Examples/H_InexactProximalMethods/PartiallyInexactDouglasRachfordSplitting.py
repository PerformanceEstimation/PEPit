import random as rd

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.inexactproximal_step import inexact_proximal_step
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_pidrs(mu, L, n, gamma, sigma, verbose=True):
    """
    Consider the composite non-smooth strongly convex minimization problem,
        min_x { F(x) = f(x) + g(x) }
    where f(x) is L-smooth and mu-strongly convex, and g is closed convex and proper.
    Both proximal operators are assumed to be available.

    This code computes a worst-case guarantee for a partially inexact Douglas Rachford
    Splitting (DRS) method, where x_* = argmin_x (F(x) = f(x) + g(x)).

    That is, it computes the smallest possible tau(n,L,mu,sigma,gamma) such that the guarantee
        ||z_{n+1} - z_*||^2 <= tau(n,L,mu,sigma,gamma) * ||z_{n} - z_*||^2.
    is valid, where z_n is the output os the operator, an z_* a fixed point of this operator.

    The exact method is from [1], its PEP formulation and solution from [2].
    The precise formulation we used is described in (2, Section 4.4].

    [1] J. Eckstein and W. Yao, Relative-error approximate versions of
    Douglas–Rachford splitting and special cases of the ADMM.
    Mathematical Programming (2018).

    [2] M. Barre, A. Taylor, F. Bach. Principled analyses and design of
    first-order methods with inexact proximal operators (2020).

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong convexity parameter.
    :param n: (int) number of iterations.
    :param gamma: (float) the step size.
    :param sigma: (float) noise parameter.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex and a smooth strongly convex function.
    f = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})
    g = problem.declare_function(ConvexFunction, param={})

    # Define the function to optimize as the sum of func1 and func2
    F = f + g

    # Start by defining its unique optimal point xs = x_*, its function value fs = F(x_*)
    # and zs te fixed point of the operator.
    xs = F.stationary_point()
    zs = xs + gamma * f.gradient(xs)

    # Then define the starting point z0, that is the previous step of the algorithm.
    z0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between z0 and zs = z_*
    problem.set_initial_condition((z0 - zs) ** 2 <= 1)

    # Compute n steps of the partially inexact Douglas Rachford Splitting starting from z0
    z = z0
    opt = 'PD_gapII'
    for _ in range(n):
        x, dfx, _, _, _, _, epsVar = inexact_proximal_step(z, f, gamma, opt)
        y, _, _ = proximal_step(x - gamma * dfx, g, gamma)
        f.add_constraint(epsVar <= ((sigma / gamma) * (y - z + gamma * dfx)) ** 2)
        z = z + (y - x)

    # Set the performance metric to the final distance between zn and zs
    problem.set_performance_metric((z - zs) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = max(((1 - sigma + gamma * mu * sigma) / (1 - sigma + gamma * mu)) ** 2,
                          ((sigma + (1 - sigma) * gamma * L) / (1 + (1 - sigma) * gamma * L)) ** 2) ** n

    # Print conclusion if required
    if verbose:
        print('*** Example file:'
              ' worst-case performance of the Partially Inexact Douglas Rachford Splitting in distance ***')
        print('\tPEP-it guarantee:\t ||z_n - z_*||^2 <= {:.6} ||z_0 - z_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t||z_n - z_*||^2 <= {:.6} ||z_0 - z_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    mu = 1.
    L = 5.
    # Choose random scheme parameters
    gamma = 1.4
    sigma = 0.2
    # Number of iterations
    n = 5

    pepit_tau, theoretical_tau = wc_pidrs(mu=mu,
                                          L=L,
                                          n=n,
                                          gamma=gamma,
                                          sigma=sigma)
