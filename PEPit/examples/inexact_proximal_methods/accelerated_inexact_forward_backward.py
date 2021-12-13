import numpy as np

from PEPit.pep import PEP
from PEPit.primitive_steps.inexact_proximal_step import inexact_proximal_step
from PEPit.functions.strongly_convex import StronglyConvexFunction
from PEPit.functions.smooth_convex_function import SmoothConvexFunction


def wc_accelerated_inexact_forward_backward(mu, L, gamma, sigma, xi, zeta, A0, verbose=True):
    """
    Consider the composite convex minimization problem,

    .. math:: F_\\star \\triangleq \\min_x \\left\\{F(x) \\equiv f(x) + g(x) \\right\\},

    where :math:`f` is :math:`L`-smooth convex, and :math:`g` is :math:`\\mu`-strongly convex (possibly non-smooth, and
    possibly with :math:`\\mu=0`). We further assume that one can readily evaluate the gradient of :math:`f` and
    that one has access to an inexact version of the proximal operator of :math:`g`.

    This code verifies a potential (or Lyapunov/energy) function for an **inexact accelerated forward-backward method**
    presented in [1, Algorithm 3.1]. That is, it verifies that

    .. math:: \\Phi_{t+1} \\leq \\Phi_t

    is valid, where :math:`\\Phi_t \\triangleq A_t (F(x_t) - F_\\star) + \\frac{1 + \\mu A_t}{2} \\|z_t - x_\\star\\|^2` is a potential function.
    For doing that, we verify that the maximum value of :math:`\\Phi_{t+1} - \\Phi_t` is less than zero (maximum over all
    problem instances and initializations).

    **Algorithm**:

    The method is presented in [1, Algorithm 3.1]. For simplicity, we instantiate [1, Algorithm 3.1] using simple
    values for its parameters (:math:`\\xi_t=0`, :math:`\\sigma_t=0`, :math:`\\lambda_t =\\tfrac{1}{L}` in the notation
    of [1]), and without backtracking, arriving to:

        .. math::
            :nowrap:

            \\begin{eqnarray}
                 \\eta_t && = (1-\\zeta_t^2) \\lambda \\\\
                 A_{t+1} && = A_t + \\frac{\\eta_t+2A_t \\mu\\eta_t+\\sqrt{\\eta_t^2+4\\eta_t A_t(1+\\eta_t\\mu)(1+A_t\\mu)}}{2},\\\\
                 y_{t} && = x_t + \\frac{(A_{t+1}-A_t)(1+\\mu A_t)}{A_{t+1}+A_t(2A_{t+1}-A_t)\\mu} (z_t-x_t),\\\\
                 (x_{t+1},v_{t+1}) && \\approx_{\\varepsilon_t,\\mu} \\left(\\mathrm{prox}_{\lambda g}\\left(y_t-\\lambda \\nabla f(y_t)\\right),\,
                 \\mathrm{prox}_{ g^*/\\lambda}\\left(\\frac{y_t-\\lambda \\nabla f(y_t)}{\\lambda}\\right)\\right),\\\\
                 && \\text{with } \\varepsilon_t = \\frac{\\zeta_t^2\\lambda^2}{2(1+\\lambda\\mu)^2}\|v_{t+1}+\\nabla f(y_t) \|^2,\\\\
                 z_{t+1} && = z_t+\\frac{A_{t+1}-A_t}{1+\\mu A_{t+1}}\\left(\\mu (x_{t+1}-z_t)-(v_{t+1}+\\nabla f(y_t))\\right),\\\\
            \\end{eqnarray}

    where :math:`\\{\\varepsilon_t\\}_{t\\geqslant 0}` is some sequence of accuracy parameters, and :math:`\\{\\eta_t\\}_{t\\geqslant 0}`
    and :math:`\\{A_t\\}_{t\\geqslant 0}` are some scalar sequences of parameters for the method.

    The line with ":math:`\\approx_{\\varepsilon,\\mu}`" can be described as the pair :math:`(x_{t+1},v_{t+1})` satisfying
    an accuracy requirement provided by [1, Definition 2.3]. More precisely (but without providing any intuition), it requires
    the existence of some :math:`w_{t+1}` such that :math:`v_{t+1}-\\mu x_{t+1} + \\mu w_{t+1} \\in \\partial g(w_{t+1})` for
    which the following condition is satisfied:

     .. math::
            :nowrap:

            \\begin{eqnarray}
                &\\lambda (1+\\lambda \\mu)\\Big(g(x_{t+1})-g(w_{t+1})+\\tfrac{\\mu}{2}\\|x_{t+1}-w_{t+1}\\|^2- \\langle x_{t+1}-w_{t+1},v_{t+1} \\rangle \\Big)\\\\
                &\\quad+\\tfrac{1}{2} \\|x_{t+1} - y_t +\\lambda (v_{t+1}+\\nabla f(y_t))\\|^2\\\\
                &\\quad \\leqslant \\tfrac{\\zeta_t^2\\lambda^2}{2}\|v_{t+1}+\\nabla f(y_t)\|^2.
            \\end{eqnarray}

    **Theoretical guarantee**:

    A theoretical guarantee is obtained in [1, Theorem 3.2]:

        .. math:: \\Phi_{t+1} - \\Phi_t \\leq 0.

    **References**:

    The method and theoretical result can be found in [1, Section 3].

    `[1] M. Barre, A. Taylor, F. Bach (2021). A note on approximate accelerated forward-backward methods with
    absolute and relative errors, and possibly strongly convex objectives. arXiv:2106.15536v2. <https://arxiv.org/pdf/2106.15536v2.pdf>`_

    Args: TODOUPDATE: virer les inutiles (aussi dans signature et tests)
        mu (float): strong convexity parameter.
        L (float): smoothness parameter.
        gamma (float): the step-size.
        sigma (float): noise parameter.
        xi (float): Lyapunov and scheme parameter.
        zeta (float): Lyapunov and scheme parameter.
        A0 (float): Lyapunov parameter.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> TODOTODO

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a non-smooth strongly convex function, and a smooth convex function.
    f = problem.declare_function(SmoothConvexFunction, param={'L': L})
    g = problem.declare_function(StronglyConvexFunction, param={'mu': mu})
    F = f + g

    # Start by defining its unique optimal point xs = x_*, and its associated function value xs = x_*.
    xs = F.stationary_point()
    fs = F.value(xs)

    # Then define the starting point z0 and x0, that is the previous step of the algorithm.
    x0 = problem.set_initial_point()
    z0 = problem.set_initial_point()
    f0 = f.value(x0)
    g0 = g.value(x0)

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Set the scheme parameters
    eta = (1 - zeta ** 2) * gamma
    opt = 'PD_gapI'
    a0 = (eta + 2 * A0 * eta * mu + np.sqrt(4 * eta * A0 * (A0 * mu + 1) * (eta * mu + 1) + eta ** 2)) / 2
    A1 = A0 + a0

    # Compute one step of the Accelerated Hybrid Proximal Gradient starting from x0
    y = x0 + (A1 - A0) * (A0 * mu + 1) / (A0 * mu * (2 * A1 - A0) + A1) * (z0 - x0)
    dfy, fy = f.oracle(y)
    x1, _, g1, w, v, _, epsVar = inexact_proximal_step(y - gamma * dfy, g, gamma, opt)
    f.add_constraint(epsVar <= sigma ** 2 / 2 * (y - x1) ** 2 + gamma ** 2 * zeta ** 2 / 2 * (v + dfy) ** 2 + xi / 2)
    f1 = f.value(x1)
    z1 = z0 + (A1 - A0) / (A1 * mu + 1) * (mu * (w - z0) - (v + dfy))

    phi0 = A0 * (f0 + g0 - fs) + (1 + mu * A0) / 2 * (z0 - xs) ** 2
    phi1 = A1 * (f1 + g1 - fs) + (1 + mu * A1) / 2 * (z1 - xs) ** 2

    # Set the performance metric to the final distance between zn and zs
    problem.set_performance_metric(phi1 - phi0 - A1 / 2 / gamma * xi)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 0.

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Accelerated Hybrid Proximal gradient in distance ***')
        print('\tPEP-it guarantee:\t\t phi(n+1) - phi(n) <= {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t phi(n+1) - phi(n) <= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    L = 2
    sigma = .2
    pepit_tau, theoretical_tau = wc_accelerated_inexact_forward_backward(mu=1, L=L, gamma=(1 - sigma ** 2) / L, sigma=sigma, xi=3, zeta=0.9, A0=1, verbose=True)
