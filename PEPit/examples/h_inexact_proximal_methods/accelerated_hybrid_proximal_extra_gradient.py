import numpy as np

from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.primitive_steps.inexact_proximal_step import inexact_proximal_step


def wc_ahpe(mu, eta, sigma, Ak, verbose=True):
    """
    Consider the convex minimization problem,

    .. math:: \\min_x { f(x) }

    where :math:`f(x)` is closed, proper and convex (possibly :math:`mu`-strongly convex),
    and for which an approximate proximal operator is available.

    This code verifies a potential function
    for the **Accelerated Hybrid Proximal Extragradient (A-HPE)** introduced in [1].

    That is, it verifies that the potential function

    .. math:: \\Phi_k = A_k (f(x_k) - f_\\star) + \\frac{1 + \\mu A_k}{2} \\|z_k - x_\\star\\|^2

    is decreasing along the iterations, that is that

    .. math:: \\Phi_{k+1} - \\Phi_k \\leq 0

    is valid for all :math:`f` satisfying the previous assumptions, and for any initialization of the **A-HPE**.

    **Algorithm**:

    The algorithm is presented in [2, section 3.1]

        .. math::
            \\begin{eqnarray}
                A_{k+1} & = & A_k + \\frac{\\eta_k + 2 A_k \\mu \\eta_k + \\sqrt{\\eta_k^2 + 4 \\eta_k A_k (1 + \\eta_k \\mu)(1 + A_k \\mu)}}{2} \\\\
                y_k & = & x_k + \\frac{(A_{k+1} - A_k)(A_k \\mu + 1)}{A_{k+1} + A_k \\mu (2 A_{k+1} - A_k)} (z_k - x_k) \\\\
                TODO \\\\
                x1, _, f1, w, v, _, epsVar = inexact_proximal_step(y, f, gamma, 'PD_gapI') \\\\
                f.add_constraint(epsVar <= sigma ** 2 / 2 * (y - x1) ** 2) \\\\
                z_{k+1} = z_k + {A_{k+1} - A_k}{1 + \\mu A_{k+1}} (\\mu (w - z_k) - v)
            \\end{eqnarray}

    **Theoretical guarantee**:

    The theoretical guarantee is obtained in [2, Theorem 3.2],
        TODO the bound is not 0 in this TH??
        .. math:: \\Phi_{k+1} - \\Phi_k \\leq 0

    **References**:

    The method originates from [1].
    It was adapted to deal with strong convexity in [2].
    The PEP methodology for analyzing such methods was proposed in [3].

    `[1] R. D. Monteiro and B. F. Svaiter. An accelerated hybrid proximal extragradient method for
    convex optimization and its implications to second-order methods, SIAM Journal on Optimization (2013).
    <http://www.optimization-online.org/DB_FILE/2011/05/3030.pdf>`_

    `[2] M. Barre, A. Taylor, F. Bach. A note on approximate accelerated forward-backward
    methods with absolute and relative errors, and possibly strongly convex objectives (2021).
    <https://arxiv.org/pdf/2106.15536.pdf>`_

    `[3] M. Barre, A. Taylor, F. Bach. Principled analyses and design of first-order methods
    with inexact proximal operators (2020).
    <https://arxiv.org/pdf/2006.06041.pdf>`_

    Args:
        mu (float): strong convexity parameter.
        eta (float): step size.
        sigma (float): noise parameter.
        Ak (float): Lyapunov parameter.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_ahpe(mu=1, eta=1, sigma=1, Ak=10, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 8x8
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (0 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 14 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: -1.4370803363076676e-13
        *** Example file: worst-case performance of the Accelerated Hybrid Proximal gradient in distance ***
            PEP-it guarantee:		 phi(k+1) - phi(k) <= -1.43708e-13
            Theoretical guarantee:	 phi(k+1) - phi(k) <= 0.0

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex and a smooth strongly convex function.
    f = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': np.inf})

    # Start by defining its unique optimal point xs = x_*, and its associated function value xs = x_*.
    xs = f.stationary_point()
    fs = f.value(xs)

    # Then define the starting point z0 and x0, that is the previous step of the algorithm.
    xk = problem.set_initial_point()
    zk = problem.set_initial_point()
    fk = f.value(xk)

    # Compute one step of the Accelerated Hybrid Proximal Extragradient starting from x0
    Akp1 = Ak + (eta + 2 * Ak * eta * mu + np.sqrt(4 * eta * Ak * (Ak * mu + 1) * (eta * mu + 1) + eta ** 2)) / 2
    yk = xk + (Akp1 - Ak) * (Ak * mu + 1) / (Ak * mu * (2 * Akp1 - Ak) + Akp1) * (zk - xk)
    xkp1, _, fkp1, w, v, _, epsVar = inexact_proximal_step(yk, f, eta, 'PD_gapI')
    f.add_constraint(epsVar <= sigma ** 2 / 2 * (yk - xkp1) ** 2)
    zkp1 = zk + (Akp1 - Ak) / (Akp1 * mu + 1) * (mu * (w - zk) - v)

    # Set the performance metric to the difference between the potential after one iteration minus its original value
    # (the potential is verified if the maximum of the difference is less than zero).
    phi_k = Ak * (fk - fs) + (1 + mu * Ak) / 2 * (zk - xs) ** 2
    phi_kp1 = Akp1 * (fkp1 - fs) + (1 + mu * Akp1) / 2 * (zkp1 - xs) ** 2
    problem.set_performance_metric(phi_kp1 - phi_k)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 0.

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Accelerated Hybrid Proximal gradient in distance ***')
        print('\tPEP-it guarantee:\t\t phi(k+1) - phi(k) <= {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t phi(k+1) - phi(k) <= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_ahpe(mu=1, eta=1, sigma=1, Ak=10, verbose=True)
