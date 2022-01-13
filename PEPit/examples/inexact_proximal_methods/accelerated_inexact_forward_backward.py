from math import sqrt

from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.functions import SmoothConvexFunction
from PEPit.primitive_steps import inexact_proximal_step


def wc_accelerated_inexact_forward_backward(L, zeta, n, verbose=True):
    """
    Consider the composite convex minimization problem,

    .. math:: F_\\star \\triangleq \\min_x \\left\\{F(x) \\equiv f(x) + g(x) \\right\\},

    where :math:`f` is :math:`L`-smooth convex, and :math:`g` is closed, proper, and convex.
    We further assume that one can readily evaluate the gradient of :math:`f` and that one has access to an inexact
    version of the proximal operator of :math:`g` (whose level of accuracy is controlled by some
    parameter :math:`\\zeta\\in (0,1)`).

    This code computes a worst-case guarantee for an **accelerated inexact forward backward** (AIFB) method (a.k.a.,
    inexact accelerated proximal gradient method). That is, it computes the smallest possible
    :math:`\\tau(n, L, \\zeta)` such that the guarantee

    .. math :: F(x_n) - F(x_\\star) \\leqslant \\tau(n, L, \\zeta) \\|x_0 - x_\\star\\|^2,

    is valid, where :math:`x_n` is the output of the IAFB, and where :math:`x_\\star` is a minimizer of :math:`F`.

    In short, for given values of :math:`n`, :math:`L` and :math:`\\zeta`, :math:`\\tau(n, L, \\zeta)` is computed as
    the worst-case value of :math:`F(x_n) - F(x_\\star)` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**: Let :math:`t\\in\\{0,1,\\ldots,n\\}`. The method is presented in, e.g., [1, Algorithm 3.1].
    For simplicity, we instantiate [1, Algorithm 3.1] using simple values for its parameters and for the problem
    setting (in the notation of [1]: :math:`A_0\\triangleq 0`, :math:`\\mu=0`, :math:`\\xi_t \\triangleq0`,
    :math:`\\sigma_t\\triangleq 0`, :math:`\\lambda_t \\triangleq\\gamma\\triangleq\\tfrac{1}{L}`,
    :math:`\\zeta_t\\triangleq\\zeta`, :math:`\\eta \\triangleq (1-\\zeta^2) \\gamma`), and without backtracking,
    arriving to:

        .. math::
            :nowrap:

            \\begin{eqnarray}
                 A_{t+1} && = A_t + \\frac{\\eta+\\sqrt{\\eta^2+4\\eta A_t}}{2},\\\\
                 y_{t} && = x_t + \\frac{A_{t+1}-A_t}{A_{t+1}} (z_t-x_t),\\\\
                 (x_{t+1},v_{t+1}) && \\approx_{\\varepsilon_t} \\left(\\mathrm{prox}_{\\gamma g}\\left(y_t-\\gamma \\nabla f(y_t)\\right),\,
                 \\mathrm{prox}_{ g^*/\\gamma}\\left(\\frac{y_t-\\gamma \\nabla f(y_t)}{\\gamma}\\right)\\right),\\\\
                 && \\text{with } \\varepsilon_t = \\frac{\\zeta^2\\gamma^2}{2}\|v_{t+1}+\\nabla f(y_t) \|^2,\\\\
                 z_{t+1} && = z_t-(A_{t+1}-A_t)\\left(v_{t+1}+\\nabla f(y_t)\\right),\\\\
            \\end{eqnarray}

    where :math:`\\{\\varepsilon_t\\}_{t\\geqslant 0}` is some sequence of accuracy parameters (whose values are fixed
    within the algorithm as it runs), and :math:`\\{A_t\\}_{t\\geqslant 0}` is some scalar sequence of parameters
    for the method (typical of accelerated methods).

    The line with ":math:`\\approx_{\\varepsilon}`" can be described as the pair :math:`(x_{t+1},v_{t+1})` satisfying
    an accuracy requirement provided by [1, Definition 2.3]. More precisely (but without providing any intuition), it requires
    the existence of some :math:`w_{t+1}` such that :math:`v_{t+1} \\in \\partial g(w_{t+1})` and for which the accuracy
    requirement

     .. math:: \\gamma^2 || x_{t+1} - y_t + \\gamma v_{t+1} ||^2 + \\gamma  (g(x_{t+1}) - g(w_{t+1}) - v_{t+1}(x_{t+1} - w_{t+1})) \\leqslant \\varepsilon_t,

    is valid.

    **Theoretical guarantee**: A theoretical upper bound is obtained in [1, Corollary 3.5]:

        .. math:: F(x_n)-F_\\star\\leqslant \\frac{2L \\|x_0-x_\\star\\|^2}{(1-\\zeta^2)n^2}.

    **References**: The method and theoretical result can be found in [1, Section 3].

    `[1] M. Barre, A. Taylor, F. Bach (2021). A note on approximate accelerated forward-backward methods with
    absolute and relative errors, and possibly strongly convex objectives. arXiv:2106.15536v2.
    <https://arxiv.org/pdf/2106.15536v2.pdf>`_

    Args:
        L (float): smoothness parameter.
        zeta (float): relative approximation parameter in (0,1).
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Example:
        >>> pepit_tau, theoretical_tau = wc_accelerated_inexact_forward_backward(L=1.3, zeta=.45, n=11, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 59x59
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
                 function 1 : 156 constraint(s) added
                 function 2 : 528 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.018869997698251897
        *** Example file: worst-case performance of an inexact accelerated forward backward method ***
        PEPit guarantee:	     F(x_n)-F_* <= 0.01887 ||x_0 - x_*||^2
        Theoretical guarantee:	 F(x_n)-F_* <= 0.0269437 ||x_0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex and a convex function
    f = problem.declare_function(SmoothConvexFunction, param={'L': L})
    h = problem.declare_function(ConvexFunction, param={})
    F = f + h

    # Start by defining its unique optimal point xs = x_* and its function value Fs = F(x_*)
    xs = F.stationary_point()
    Fs = F.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Some algorithmic parameters (for convenience)
    gamma = 1 / L
    eta = (1 - zeta ** 2) * gamma
    A = [0]

    # Compute n steps of the IAFB method starting from x0
    x = x0
    z = x0
    for i in range(n):
        A.append(A[i] + (eta + sqrt(eta ** 2 + 4 * eta * A[i])) / 2)
        y = x + (1 - A[i] / A[i + 1]) * (z - x)
        gy = f.gradient(y)
        x, sx, hx, _, vx, _, epsVar = inexact_proximal_step(y - gamma * gy, h, gamma, opt='PD_gapI')
        h.add_constraint(epsVar <= (zeta * gamma) ** 2 / 2 * (vx + gy) ** 2)  # this is the accuracy requirement
        z = z - (A[i + 1] - A[i]) * (vx + gy)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric((f.value(x) + hx) - Fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 2 * L / (1 - zeta ** 2) / n ** 2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of an inexact accelerated forward backward method ***')
        print('\tPEPit guarantee:\t F(x_n)-F_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t F(x_n)-F_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_accelerated_inexact_forward_backward(L=1.3, zeta=.45, n=11, verbose=True)
