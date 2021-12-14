import numpy as np

from PEPit.pep import PEP
from PEPit.functions.convex_function import ConvexFunction
from PEPit.functions.convex_indicator import ConvexIndicatorFunction
from PEPit.primitive_steps.bregman_gradient_step import bregman_gradient_step


def wc_no_lips_in_bregman_divergence(L, gamma, n, verbose=True):
    """
    Consider the constrainted composite convex minimization problem

    .. math:: F_\\star \\triangleq \\min_x {F(x) \\equiv f_1(x) + f_2(x)},

    where :math:`f_1` is convex and :math:`L`-smooth relatively to :math:`h`,
    :math:`h` being closed proper and convex,
    and where :math:`f_2` is a closed convex indicator function.

    This code computes a worst-case guarantee for the **NoLips** method.
    That is, it computes the smallest possible :math:`\\tau(n, L)` such that the guarantee

    .. math:: \\min_{t\\leqslant n} D_h(x_{t-1}, x_t) \\leqslant \\tau(n, L) D_h(x_\\star, x_0),

    is valid, where :math:`x_n` is the output of the **NoLips** method,
    where :math:`x_\\star` is a minimizer of :math:`F`,
    and where :math:`D_h` is the Bregman divergence generated by :math:`h`.
    In short, for given values of :math:`n` and :math:`L`,
    :math:`\\tau(n, L)` is computed as the worst-case value of
    :math:`\\min_{t\\leqslant n} D_h(x_{t-1}, x_t)` when :math:`D_h(x_\\star, x_0) \\leqslant 1`.

    **Algorithm**:
    This method is presented in [2, Algorithm 1]

        .. math:: x_{t+1} = \\arg\\min_{u \\in \\mathrm{Dom}(f_2)} \\langle \\nabla f_1(x_t) \\mid u - x_t \\rangle + \\frac{1}{\\gamma} D_h(u, x_t)

    **Theoretical guarantee**:
    The **upper** guarantee obtained in [2, Proposition 4] is

        .. math:: \\tau(n, L, \\mu) = \\frac{2}{n (n - 1)}

        for any :math:`\\gamma \\leq \\frac{1}{L}`.

    References:

        The detailed approach is availaible in [1]. The formulation as a PEP, and the tightness are proven in [2].

        `[1] Heinz H. Bauschke, Jérôme Bolte, and Marc Teboulle. "A Descent Lemma
        Beyond Lipschitz Gradient Continuity: First-Order Methods Revisited and Applications." (2017)
        <https://cmps-people.ok.ubc.ca/bauschke/Research/103.pdf>`_

        `[2] Radu-Alexandru Dragomir, Adrien B. Taylor, Alexandre d’Aspremont, and
        Jérôme Bolte. "Optimal Complexity and Certification of Bregman First-Order Methods". (2019)
        <https://arxiv.org/pdf/1911.08510.pdf>`_

    Notes:
        Disclaimer: This example requires some experience with PESTO and PEPs ([2], section 4).

    Args:
        L (float): relative-smoothness parameter
        gamma (float): step-size.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst-case value, theoretical value

    Example:
        >>> L = 1
        >>> gamma = 1 / L
        >>> pepit_tau, theoretical_tau = wc_no_lips_in_bregman_divergence(L=L, gamma=gamma, n=10, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 36x36
        (PEP-it) Setting up the problem: performance measure is minimum of 10 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 3 function(s)
                 function 1 : 132 constraint(s) added
                 function 2 : 462 constraint(s) added
                 function 3 : 121 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.022158793967249273
        *** Example file: worst-case performance of the NoLips_2 in Bregman distance ***
            PEP-it guarantee:		 min_t Dh(x_(t-1), x_t) <= 0.0221588 Dh(x_*, x_0)
            Theoretical guarantee:	 min_t Dh(x_(t-1), x_t) <= 0.0222222 Dh(x_*, x_0)

    """

    # Instantiate PEP
    problem = PEP()

    # Declare two convex functions and a convex indicator function
    d = problem.declare_function(ConvexFunction, param={}, is_differentiable=True)
    func1 = problem.declare_function(ConvexFunction, param={}, is_differentiable=True)
    h = (d + func1) / L
    func2 = problem.declare_function(ConvexIndicatorFunction, param={'D': np.inf})

    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    ghs, hs = h.oracle(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()
    gh0, h0 = h.oracle(x0)
    gf0, f0 = func1.oracle(x0)

    # Set the initial constraint that is the Bregman distance between x0 and x^*
    problem.set_initial_condition(hs - h0 - gh0 * (xs - x0) <= 1)

    # Compute n steps of the NoLips starting from x0
    x1, x2 = x0, x0
    gfx = gf0
    ghx = gh0
    hx1, hx2 = h0, h0
    for i in range(n):
        x2, _, _ = bregman_gradient_step(gfx, ghx, func2 + h, gamma)
        gfx, _ = func1.oracle(x2)
        ghx, hx2 = h.oracle(x2)
        Dhx = hx1 - hx2 - ghx * (x1 - x2)
        # update the iterates
        x1 = x2
        hx1 = hx2
        # Set the performance metric to the Bregman distance to the optimum
        problem.set_performance_metric(Dhx)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 2 / (n * (n - 1))

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the NoLips_2 in Bregman distance ***')
        print('\tPEP-it guarantee:\t\t min_t Dh(x_(t-1), x_t) <= {:.6} Dh(x_*, x_0)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t min_t Dh(x_(t-1), x_t) <= {:.6} Dh(x_*, x_0) '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    L = 1
    gamma = 1 / L
    pepit_tau, theoretical_tau = wc_no_lips_in_bregman_divergence(L=L, gamma=gamma, n=10, verbose=True)