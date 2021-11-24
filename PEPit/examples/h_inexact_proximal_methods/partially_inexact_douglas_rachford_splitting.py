from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.functions.convex_function import ConvexFunction
from PEPit.primitive_steps.inexact_proximal_step import inexact_proximal_step
from PEPit.primitive_steps.proximal_step import proximal_step


def wc_pidrs(mu, L, n, gamma, sigma, verbose=True):
    """
    Consider the composite non-smooth strongly convex minimization problem,

        .. math:: \min_x { F(x) = f(x) + g(x) }

    where :math:`f` is :math:`L`-smooth and :math:`\\mu`-strongly convex, and :math:`g` is closed convex and proper.
    Both proximal operators are assumed to be available.

    This code computes a worst-case guarantee for a partially inexact **Douglas Rachford
    Splitting (DRS) method**, where :math:`x_\star = \\arg\\min_x (F(x) = f(x) + g(x))`.

    That is, it computes the smallest possible :math:`\\tau(n,L,\\mu,\\sigma,\\gamma)` such that the guarantee

        .. math:: ||z_{n+1} - z_\star||^2 \\leqslant \\tau(n,L,\\mu,\\sigma,\\gamma)  ||z_{n} - z_\star||^2

    is valid, where :math:`z_n` is the output of the operator, and :math:`z_\star` a fixed point of this operator.

    **Algorithm**:

        .. math:: x_{k+1} = z_k - \\gamma (v_{k+1} - e), \ v_{k+1} \\in \\partial f(x_{k+1})
        .. math:: y_{k+1} = prox_{\\gamma g}(x_{k+1} - \\gamma v_{k+1})
        .. math:: ||e||^2 \\leqslant \\frac{\\sigma^2}{\\gamma^2}(y_{k+1} - z_k + \\gamma v_{k+1})^2
        .. math:: z_{k+1} = z_{k} + y_{k+1} - x_{k+1}

    **Theoretical guarantee**:

        The theoretical **tight** bound is obtained in [2, Theorem 5.1],

        .. math:: \max\\left(\\frac{1 - \\sigma + \\gamma \\mu \\sigma}{1 - \\sigma + \\gamma \\mu},
                             \\frac{\\sigma + (1 - \\sigma) \\gamma L}{1 + (1 - \\sigma) \\gamma L)}\\right)^{2n}

    **References**:

        The exact method is from [1], its PEP formulation and solution from [2].
        The precise formulation we used is described in (2, Section 4.4).

        [1] J. Eckstein and W. Yao, Relative-error approximate versions of
        Douglas–Rachford splitting and special cases of the ADMM.
        Mathematical Programming (2018).

        [2] M. Barre, A. Taylor, F. Bach. Principled analyses and design of
        first-order methods with inexact proximal operators (2020).

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        n (int): number of iterations.
        gamma (float): the step size.
        sigma (float): noise parameter.
        verbose (bool, optional): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_pidrs(0.1, 5, 5, 1.4, 0.2)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 18x18
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 2 function(s)
                 function 1 : 40 constraint(s) added
                 function 2 : 30 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.27473239411965494
        *** Example file: worst-case performance of the Partially Inexact Douglas Rachford Splitting in distance ***
            PEP-it guarantee:	 ||z_n - z_*||^2 <= 0.274732 ||z_0 - z_*||^2
            Theoretical guarantee:	||z_n - z_*||^2 <= 0.274689 ||z_0 - z_*||^2

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
