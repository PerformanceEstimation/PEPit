import numpy as np

from PEPit.pep import PEP
from PEPit.operators.lipschitz_strongly_monotone import LipschitzStronglyMonotoneOperator
from PEPit.operators.strongly_monotone import StronglyMonotoneOperator
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_drs(L, mu, alpha, theta, verbose=True):
    """
    Consider the monotone inclusion problem
        Find x, 0 \in Ax + Bx,
    where A is L-Lipschitz and monotone and B is (maximally) mu-strongly monotone.
    We denote JA and JB the respective resolvents of A and B.

    This code computes a worst-case guarantee for the Douglas Rachford Splitting (DRS). One iteration of the algorithm
    starting from a point w is as follows:
        x = JB(w)
        y = JA( 2* x - w)
        z = w - theta * (x - y)

    Given two initial points w_0 and w_1, this code computes the smallest possible tau(n,L) such that the guarantee
        || z_0 - z_1||^2 <= tau(n,L) * || w_0 - w_1||^2,
    is valid, where z_0 and z_1 are obtained after one iteration of DRS from respectively w_0 and w_1.

    Theoretical rates can be found in the following paper (section 4, Theorem 4.1)
    [1] Walaa M. Moursi, and Lieven Vandenberghe. "Douglas–Rachford
         Splitting for the Sum of a Lipschitz Continuous and a Strongly
         Monotone Operator." (2019)

    The methodology using PEPs is presented in
    [2] Ernest K. Ryu, Adrien B. Taylor, C. Bergeling, and P. Giselsson.
        "Operator Splitting Performance Estimation: Tight contraction
        factors and optimal parameter selection." (2018).
    since the results of [2] tightened that of [1], we compare with [2, Theorem 3] below.

    :param L: (float) the Lipschitz parameter.
    :param mu: (float) the strongly monotone parameter.
    :param alpha: (float) the step size in the resolvent.
    :param theta: (float) algorithm parameter.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a monotone operator
    A = problem.declare_function(LipschitzStronglyMonotoneOperator, param={'L': L, 'mu': 0})
    B = problem.declare_function(StronglyMonotoneOperator, param={'mu': mu})

    # Then define starting points w0 and w1
    w0 = problem.set_initial_point()
    w1 = problem.set_initial_point()

    # Set the initial constraint that is the distance between w0 and w1
    problem.set_initial_condition((w0 - w1) ** 2 <= 1)

    # Compute one step of the Douglas Rachford Splitting starting from w0
    x0, _, _ = proximal_step(w0, B, alpha)
    y0, _, _ = proximal_step(2 * x0 - w0, A, alpha)
    z0 = w0 - theta * (x0 - y0)

    # Compute one step of the Douglas Rachford Splitting starting from w1
    x1, _, _ = proximal_step(w1, B, alpha)
    y1, _, _ = proximal_step(2 * x1 - w1, A, alpha)
    z1 = w1 - theta * (x1 - y1)

    # Set the performance metric to the distance between z0 and z1
    problem.set_performance_metric((z0 - z1) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    mu = alpha * mu
    L = alpha * L
    c = np.sqrt(((2 * (theta - 1) * mu + theta - 2) ** 2 + L ** 2 * (theta - 2 * (mu + 1)) ** 2) / (L ** 2 + 1))
    if theta * (theta + c) / (mu + 1) ** 2 / c * (
            c + mu * ((2 * (theta - 1) * mu + theta - 2) - L ** 2 * (theta - 2 * (mu + 1))) / (L ** 2 + 1)) >= 0:
        theoretical_tau = ((theta + c) / 2 / (mu + 1)) ** 2
    elif (L <= 1) & (mu >= (L ** 2 + 1) / (L - 1) ** 2) & (theta <= - (2 * (mu + 1) * (L + 1) *
                                                                       (mu + (mu - 1) * L ** 2 - 2 * mu * L - 1)) / (
                                                                   mu + L * (L ** 2 + L + 1) + 2 * mu ** 2 * (
                                                                   L - 1) + mu * L * (1 - (L - 3) * L) + 1)):
        theoretical_tau = (1 - theta * (L + mu) / (L + 1) / (mu + 1)) ** 2
    else:
        theoretical_tau = (2 - theta) / 4 / mu / (L ** 2 + 1) * (
                    theta * (1 - 2 * mu + L ** 2) - 2 * mu * (L ** 2 - 1)) * \
                          (theta * (1 + 2 * mu + L ** 2) - 2 * (mu + 1) * (L ** 2 + 1)) / (
                                      theta * (1 + 2 * mu - L ** 2) - 2 * (mu + 1) * (1 - L ** 2))

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Douglas Rachford Splitting***')
        print('\tPEP-it guarantee:\t ||z_1 - z_0||^2 <= {:.6} ||w_1 - w_0||^2'.format(pepit_tau))
        print('\tTheoretical guarantee :\t ||z_1 - z_0||^2 <= {:.6} ||w_1 - w_0||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    mu = 0.1
    alpha = 1.3
    theta = .9

    pepit_tau, theoretical_tau = wc_drs(L=L,
                                        mu=mu,
                                        alpha=alpha,
                                        theta=theta)
