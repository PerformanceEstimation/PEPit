from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import proximal_step
from PEPit.primitive_steps import inexact_proximal_step


def wc_partially_inexact_douglas_rachford_splitting(mu, L, n, gamma, sigma, verbose=True):
    """
    Consider the composite strongly convex minimization problem,

        .. math:: F_\\star \\triangleq \min_x \\left\\{ F(x) \\equiv f(x) + g(x) \\right\\}

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex, and :math:`g` is closed convex and proper. We
    denote by :math:`x_\\star = \\arg\\min_x F(x)` the minimizer of :math:`F`. The (exact) proximal operator of :math:`g`,
    and an approximate version of the proximal operator of :math:`f` are assumed to be available.

    This code computes a worst-case guarantee for a **partially inexact Douglas-Rachford Splitting** (DRS). That is, it
    computes the smallest possible :math:`\\tau(n,L,\\mu,\\sigma,\\gamma)` such that the guarantee

        .. math:: \\|z_{n} - z_\\star\\|^2 \\leqslant \\tau(n,L,\\mu,\\sigma,\\gamma)  \\|z_0 - z_\\star\\|^2

    is valid, where :math:`z_n` is the output of the DRS (initiated at :math:`x_0`), :math:`z_\\star` is its fixed point,
    :math:`\\gamma` is a step-size, and :math:`\\sigma` is the level of inaccuracy.

    **Algorithm**: The partially inexact Douglas-Rachford splitting under consideration is described by

        .. math::
            :nowrap:

            \\begin{eqnarray}
                 x_{t} && \\approx_{\\sigma} \\arg\\min_x \\left\\{ \\gamma f(x)+\\frac{1}{2} \\|x-z_t\\|^2 \\right\\},\\\\
                 y_{t} && = \\arg\\min_y \\left\\{ \\gamma g(y)+\\frac{1}{2} \\|y-(x_t-\\gamma \\nabla f(x_t))\\|^2 \\right\\},\\\\
                 z_{t+1} && = z_t + y_t - x_t.
            \\end{eqnarray}

    More precisely, the notation ":math:`\\approx_{\\sigma}`" correspond to require the existence of some
    :math:`e_{t}` such that

        .. math::
            :nowrap:

            \\begin{eqnarray}
                 x_{t} && = z_t - \\gamma (\\nabla f(x_t) - e_t),\\\\
                 y_{t} && =  \\arg\\min_y \\left\\{ \\gamma g(y)+\\frac{1}{2} \\|y-(x_t-\\gamma \\nabla f(x_t))\\|^2 \\right\\},\\\\
                  && \\text{with } \|e_t\|^2 \\leqslant \\frac{\\sigma^2}{\\gamma^2}\|y_{t} - z_t + \\gamma \\nabla f(x_t) \|^2,\\\\
                 z_{t+1} && = z_t + y_t - x_t.
            \\end{eqnarray}

    **Theoretical guarantee**: The following **tight** theoretical bound is due to [2, Theorem 5.1]:

        .. math:: \|z_{n} - z_\\star\|^2  \\leqslant \max\\left(\\frac{1 - \\sigma + \\gamma \\mu \\sigma}{1 - \\sigma + \\gamma \\mu},
                             \\frac{\\sigma + (1 - \\sigma) \\gamma L}{1 + (1 - \\sigma) \\gamma L)}\\right)^{2n} \|z_0 - z_\\star\|^2.

    **References**: The method is from [1], its PEP formulation and the worst-case analysis from [2],
    see [2, Section 4.4] for more details.

    `[1] J. Eckstein and W. Yao (2018). Relative-error approximate versions of Douglas–Rachford splitting and
    special cases of the ADMM. Mathematical Programming, 170(2), 417-444.
    <https://link.springer.com/article/10.1007/s10107-017-1160-5>`_

    `[2] M. Barre, A. Taylor, F. Bach (2020). Principled analyses and design of
    first-order methods with inexact proximal operators, arXiv 2006.06041v2.
    <https://arxiv.org/pdf/2006.06041v2.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        n (int): number of iterations.
        gamma (float): the step-size.
        sigma (float): noise parameter.
        verbose (bool, optional): if True, print conclusion

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_partially_inexact_douglas_rachford_splitting(mu=.1, L=5, n=5, gamma=1.4, sigma=.2, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 18x18
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
                 function 1 : 40 constraint(s) added
                 function 2 : 30 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.2812035034317067
        *** Example file: worst-case performance of the partially inexact Douglas Rachford splitting ***
            PEPit guarantee:		 ||z_n - z_*||^2 <= 0.281204 ||z_0 - z_*||^2
            Theoretical guarantee:	 ||z_n - z_*||^2 <= 0.281206 ||z_0 - z_*||^2

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
    for _ in range(n):
        x, dfx, _, _, _, _, epsVar = inexact_proximal_step(z, f, gamma, opt='PD_gapII')
        y, _, _ = proximal_step(x - gamma * dfx, g, gamma)
        f.add_constraint(epsVar <= 1/2 * (sigma  * (y - z + gamma * dfx)) ** 2)
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
        print('*** Example file: worst-case performance of the partially inexact Douglas Rachford splitting ***')
        print('\tPEPit guarantee:\t ||z_n - z_*||^2 <= {:.6} ||z_0 - z_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||z_n - z_*||^2 <= {:.6} ||z_0 - z_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_partially_inexact_douglas_rachford_splitting(mu=.1, L=5, n=5, gamma=1.4, sigma=.2, verbose=True)
