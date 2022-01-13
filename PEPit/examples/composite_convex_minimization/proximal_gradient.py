from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_proximal_gradient(L, mu, gamma, n, verbose=True):
    """
    Consider the composite convex minimization problem

    .. math:: F_\\star \\triangleq \\min_x \\{F(x) \\equiv f_1(x) + f_2(x)\\},

    where :math:`f_1` is :math:`L`-smooth and :math:`\\mu`-strongly convex,
    and where :math:`f_2` is closed convex and proper.

    This code computes a worst-case guarantee for the **proximal gradient** method (PGM).
    That is, it computes the smallest possible :math:`\\tau(n, L, \\mu)` such that the guarantee

    .. math :: \\|x_n - x_\\star\\|^2 \\leqslant \\tau(n, L, \\mu) \\|x_0 - x_\\star\\|^2,

    is valid, where :math:`x_n` is the output of the **proximal gradient**,
    and where :math:`x_\\star` is a minimizer of :math:`F`.
    In short, for given values of :math:`n`, :math:`L` and :math:`\\mu`,
    :math:`\\tau(n, L, \\mu)` is computed as the worst-case value of
    :math:`\\|x_n - x_\\star\\|^2` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**: Proximal gradient is described by

        .. math::
            \\begin{eqnarray}
                y_t & = & x_t - \\gamma \\nabla f_1(x_t), \\\\
                x_{t+1} & = & \\arg\\min_x \\left\\{f_2(x)+\\frac{1}{2\gamma}\|x-y_t\|^2 \\right\\},
            \\end{eqnarray}

    for :math:`t \in \\{ 0, \\dots, n-1\\}` and where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**: It is well known that a **tight** guarantee for PGM is provided by

    .. math :: \\|x_n - x_\\star\\|^2 \\leqslant \\max\\{(1-L\\gamma)^2,(1-\\mu\\gamma)^2\\}^n \\|x_0 - x_\\star\\|^2,

    which can be found in, e.g., [1, Theorem 3.1]. It is a folk knowledge and the result can be found in many references
    for gradient descent; see, e.g.,[2, Section 1.4: Theorem 3], [3, Section 5.1] and [4, Section 4.4].

    **References**:

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2018). Exact worst-case convergence rates of the proximal gradient
    method for composite convex minimization. Journal of Optimization Theory and Applications, 178(2), 455-476.
    <https://arxiv.org/pdf/1705.04398.pdf>`_

    [2] B. Polyak (1987). Introduction to Optimization. Optimization Software New York.

    `[1] E. Ryu, S. Boyd (2016). A primer on monotone operator methods.
    Applied and Computational Mathematics 15(1), 3-43.
    <https://web.stanford.edu/~boyd/papers/pdf/monotone_primer.pdf>`_

    `[4] L. Lessard, B. Recht, A. Packard (2016). Analysis and design of optimization algorithms via
    integral quadratic constraints. SIAM Journal on Optimization 26(1), 57–95.
    <https://arxiv.org/pdf/1408.3595.pdf>`_

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        gamma (float): proximal step-size.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (float): theoretical value.

    Example:
        >>> pepit_tau, theoretical_tau = wc_proximal_gradient(L=1, mu=.1, gamma=1, n=2, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 7x7
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
                 function 1 : 6 constraint(s) added
                 function 2 : 6 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.6560999999942829
        *** Example file: worst-case performance of the Proximal Gradient Method in function values***
            PEPit guarantee:		 ||x_n - x_*||^2 <= 0.6561 ||x0 - xs||^2
            Theoretical guarantee:	 ||x_n - x_*||^2 <= 0.6561 ||x0 - xs||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function and a closed convex proper function
    f1 = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': L})
    f2 = problem.declare_function(ConvexFunction, param={})
    func = f1 + f2

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the proximal gradient method starting from x0
    x = x0
    for _ in range(n):
        y = x - gamma * f1.gradient(x)
        x, _, _ = proximal_step(y, f2, gamma)

    # Set the performance metric to the distance between x and xs
    problem.set_performance_metric((x - xs) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = max((1 - mu*gamma)**2, (1 - L*gamma)**2)**n

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Proximal Gradient Method in function values***')
        print('\tPEPit guarantee:\t ||x_n - x_*||^2 <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ||x_n - x_*||^2 <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_proximal_gradient(L=1, mu=.1, gamma=1, n=2, verbose=True)
