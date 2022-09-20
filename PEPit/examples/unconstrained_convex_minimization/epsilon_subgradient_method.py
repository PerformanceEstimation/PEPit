from math import sqrt

from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import epsilon_subgradient_step


def wc_epsilon_subgradient_method(M, n, gamma, eps, R, verbose=1):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is closed, convex, and proper. This problem is a (possibly non-smooth) minimization problem.

    This code computes a worst-case guarantee for the :math:`\\varepsilon` **-subgradient method**. That is, it computes
    the smallest possible :math:`\\tau(n, M, \\gamma, \\varepsilon, R)` such that the guarantee

    .. math:: \\min_{0 \leqslant t \leqslant n} f(x_t) - f_\\star \\leqslant \\tau(n, M, \\gamma, \\varepsilon, R) 

    is valid, where :math:`x_t` are the iterates of the :math:`\\varepsilon` **-subgradient method** after :math:`t\\leqslant n` steps,
    where :math:`x_\\star` is a minimizer of :math:`f`, where :math:`M` is an upper bound on the norm of all
    :math:`\\varepsilon`-subgradients encountered, and when :math:`\\|x_0-x_\\star\\|\\leqslant R`.

    In short, for given values of :math:`M`, of the accuracy :math:`\\varepsilon`, of the step-size :math:`\\gamma`, of the initial
    distance :math:`R`, and of the number of iterations :math:`n`, :math:`\\tau(n, M, \\gamma, \\varepsilon, R)` is computed as the worst-case value of
    :math:`\\min_{0 \leqslant t \leqslant n} f(x_t) - f_\\star`.

    **Algorithm**:
    For :math:`t\\in \\{0, \\dots, n-1 \\}`

        .. math::
            :nowrap:

            \\begin{eqnarray}
                g_{t} & \\in & \\partial_{\\varepsilon} f(x_t) \\\\
                x_{t+1} & = & x_t - \\gamma g_t
            \\end{eqnarray}

    **Theoretical guarantee**: An upper bound is obtained in [1, Lemma 2]:

        .. math:: \\min_{0 \\leqslant t \\leqslant n} f(x_t)- f(x_\\star) \\leqslant \\frac{R^2+2(n+1)\\gamma\\varepsilon+(n+1) \\gamma^2 M^2}{2(n+1) \\gamma}.

    **References**: 

    `[1] R.D. Millán, M.P. Machado (2019). Inexact proximal epsilon-subgradient methods for composite convex optimization problems.
    Journal of Global Optimization 75.4 (2019): 1029-1060.
    <https://arxiv.org/pdf/1805.10120.pdf>`_
   

    Args:
        M (float): the bound on norms of epsilon-subgradients.
        n (int): the number of iterations.
        gamma (float): step-size.
        eps (float): the bound on the value of epsilon (inaccuracy).
        R (float): the bound on initial distance to an optimal solution.
        verbose (int): Level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> M, n, eps, R = 2, 6, .1, 1
        >>> gamma = 1 / sqrt(n + 1)
        >>> pepit_tau, theoretical_tau = wc_epsilon_subgradient_method(M=M, n=n, gamma=gamma, eps=eps, R=R, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 21x21
        (PEPit) Setting up the problem: performance measure is minimum of 7 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (14 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                         function 1 : Adding 188 scalar constraint(s) ...
                         function 1 : 188 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 1.0191201198697333
        *** Example file: worst-case performance of the epsilon-subgradient method ***
                PEPit guarantee:         min_(0 <= t <= n) f(x_i) - f_* <= 1.01912
                Theoretical guarantee:   min_(0 <= t <= n) f(x_i) - f_* <= 1.04491

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function
    func = problem.declare_function(ConvexFunction)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)

    # Run n steps of the epsilon subgradient method
    x = x0

    for _ in range(n):
        x, gx, fx, epsilon = epsilon_subgradient_step(x, func, gamma)
        problem.set_performance_metric(fx - fs)
        problem.add_constraint(epsilon <= eps)
        problem.add_constraint(gx ** 2 <= M ** 2)

    # Set the performance metric to the function value accuracy
    gx, fx = func.oracle(x)
    problem.add_constraint(gx ** 2 <= M ** 2)
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (R ** 2 + 2 * (n + 1) * gamma * eps + (n + 1) * gamma ** 2 * M ** 2) / (2 * (n + 1) * gamma)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the epsilon-subgradient method ***')
        print('\tPEPit guarantee:\t min_(0 <= t <= n) f(x_i) - f_* <= {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t min_(0 <= t <= n) f(x_i) - f_* <= {:.6}'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    M, n, eps, R = 2, 6, .1, 1
    gamma = 1 / sqrt(n + 1)
    pepit_tau, theoretical_tau = wc_epsilon_subgradient_method(M=M, n=n, gamma=gamma, eps=eps, R=R, verbose=1)
