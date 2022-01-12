from math import sqrt

from PEPit import PEP
from PEPit.operators import LipschitzStronglyMonotoneOperator
from PEPit.operators import StronglyMonotoneOperator
from PEPit.primitive_steps import proximal_step


def wc_douglas_rachford_splitting(L, mu, alpha, theta, verbose=True):
    """
    Consider the monotone inclusion problem

    .. math:: \\mathrm{Find}\\, x:\\, 0\\in Ax + Bx,

    where :math:`A` is :math:`L`-Lipschitz and maximally monotone and :math:`B` is (maximally) :math:`\\mu`-strongly
    monotone. We denote by :math:`J_{\\alpha A}` and :math:`J_{\\alpha B}` the resolvents of respectively A and B, with
    step-sizes :math:`\\alpha`.

    This code computes a worst-case guarantee for the **Douglas-Rachford splitting** (DRS). That is, given two initial points
    :math:`w^{(0)}_t` and :math:`w^{(1)}_t`, this code computes the smallest possible :math:`\\tau(L, \\mu, \\alpha, \\theta)`
    (a.k.a. "contraction factor") such that the guarantee

    .. math:: \\|w^{(0)}_{t+1} - w^{(1)}_{t+1}\\|^2 \\leqslant \\tau(L, \\mu, \\alpha, \\theta) \\|w^{(0)}_{t} - w^{(1)}_{t}\\|^2,

    is valid, where :math:`w^{(0)}_{t+1}` and :math:`w^{(1)}_{t+1}` are obtained after one iteration of DRS from
    respectively :math:`w^{(0)}_{t}` and :math:`w^{(1)}_{t}`.

    In short, for given values of :math:`L`, :math:`\\mu`, :math:`\\alpha` and :math:`\\theta`, the contraction
    factor :math:`\\tau(L, \\mu, \\alpha, \\theta)` is computed as the worst-case value of
    :math:`\\|w^{(0)}_{t+1} - w^{(1)}_{t+1}\\|^2` when :math:`\\|w^{(0)}_{t} - w^{(1)}_{t}\\|^2 \\leqslant 1`.

    **Algorithm**: One iteration of the Douglas-Rachford splitting is described as follows, for :math:`t \in \\{ 0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{t+1} & = & J_{\\alpha B} (w_t),\\\\
                y_{t+1} & = & J_{\\alpha A} (2x_{t+1}-w_t),\\\\
                w_{t+1} & = & w_t - \\theta (x_{t+1}-y_{t+1}).
            \\end{eqnarray}

    **Theoretical guarantee**: Theoretical worst-case guarantees can be found in [1, section 4, Theorem 4.3].
    Since the results of [2] tighten that of [1], we compare with [2, Theorem 4.3] below. The theoretical results
    are complicated and we do not copy them here.

    **References**: The detailed PEP methodology for studying operator splitting is provided in [2].

    `[1] W. Moursi, L. Vandenberghe (2019). Douglas–Rachford Splitting for the Sum of a Lipschitz Continuous and
    a Strongly Monotone Operator. Journal of Optimization Theory and Applications 183, 179–198.
    <https://arxiv.org/pdf/1805.09396.pdf>`_

    `[2] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020). Operator splitting performance estimation:
    Tight contraction factors and optimal parameter selection. SIAM Journal on Optimization, 30(3), 2251-2271.
    <https://arxiv.org/pdf/1812.00146.pdf>`_

    Args:
        L (float): the Lipschitz parameter.
        mu (float): the strongly monotone parameter.
        alpha (float): the step-size in the resolvent.
        theta (float): algorithm parameter.
        verbose (bool): if True, print conclusion.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Example:
        >>> pepit_tau, theoretical_tau  = wc_douglas_rachford_splitting(L=1, mu=.1, alpha=1.3, theta=.9, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 6x6
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
                 function 1 : 4 constraint(s) added
                 function 2 : 2 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.928770693164459
        *** Example file: worst-case performance of the Douglas Rachford Splitting***
            PEPit guarantee:		 ||w_(t+1)^0 - w_(t+1)^1||^2 <= 0.928771 ||w_(t)^0 - w_(t)^1||^2
            Theoretical guarantee:	 ||w_(t+1)^0 - w_(t+1)^1||^2 <= 0.928771 ||w_(t)^0 - w_(t)^1||^2

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

    # Compute theoretical guarantee (for comparison), see [2, Theorem 3]
    mu = alpha * mu
    L = alpha * L
    c = sqrt(((2 * (theta - 1) * mu + theta - 2) ** 2 + L ** 2 * (theta - 2 * (mu + 1)) ** 2) / (L ** 2 + 1))
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
        print('\tPEPit guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= {:.6} ||w_(t)^0 - w_(t)^1||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= {:.6} ||w_(t)^0 - w_(t)^1||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_douglas_rachford_splitting(L=1, mu=.1, alpha=1.3, theta=.9, verbose=True)
