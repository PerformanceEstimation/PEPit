from math import sqrt
import numpy as np

from PEPit import PEP
from PEPit.functions import ConvexQGFunction


def wc_gradient_descent_qg_convex(L, n, verbose=True):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is :math:`L-\\text{QG}^+` and convex.

    This code computes a worst-case guarantee for **gradient descent** with fixed step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(n, L, \\gamma)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, L, \\gamma) \\| x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of gradient descent with fixed step-size :math:`\\gamma`, and
    where :math:`x_\\star` is a minimizer of :math:`f`.

    In short, for given values of :math:`n`, :math:`L`, and :math:`\\gamma`, :math:`\\tau(n, L, \\gamma)` is computed as the worst-case
    value of :math:`f(x_n)-f_\\star` when :math:`||x_0 - x_\\star||^2 \\leqslant 1`.

    **Algorithm**:
    Gradient descent is described by

    .. math:: x_{t+1} = x_t - \\gamma \\nabla f(x_t),

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:
    When :math:`\\gamma < \\frac{1}{L}`, the **tight** theoretical guarantee can be found in [1, ?]:  #TODO add theorem

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{L}{2}\\max{\\frac{1}{2n L \\gamma + 1}, L \\gamma} \\|x_0-x_\\star\\|^2.

    **References**:

    The detailed approach is available in [1, Theorem ?].  #TODO add theorem

    #TODO add ref

    Args:
        L (float): the quadratic growth parameter.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_gradient_descent_qg_convex(L=L, gamma=.2 / L, n=4, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 7x7
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 35 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.19230811671886025
        *** Example file: worst-case performance of gradient descent with fixed step-sizes ***
            PEP-it guarantee:		 f(x_n)-f_* <= 0.192308 ||x_0 - x_*||^2
            Theoretical guarantee:	 f(x_n)-f_* <= 0.192308 ||x_0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(ConvexQGFunction, param={'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x = problem.set_initial_point()
    g, f = func.oracle(x)

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x - xs) ** 2 <= 1)

    # GD loop
    u = 1
    for i in range(n):
        # Run 1 step of the GD method and update u accordingly.
        u = u / 2 + sqrt((u / 2) ** 2 + 2)
        gamma = 1 / (L * u)
        x = x - gamma * g
        g, f = func.oracle(x)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * u)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric((f - fs))

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose, tracetrick=False)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of gradient descent with fixed step-sizes ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    L = 1
    n_list = np.arange(1, 20)
    pepit_taus = list()
    th_taus = list()
    for n in n_list:
        pepit_tau, theoretical_tau = wc_gradient_descent_qg_convex(L=L, n=n, verbose=False)
        pepit_taus.append(pepit_tau)
        th_taus.append(theoretical_tau)

    import matplotlib.pyplot as plt
    plt.plot(n_list, pepit_taus, '-x')
    plt.plot(n_list, th_taus, '--')
    plt.plot(n_list, L / (4 * np.sqrt(n_list)), '--')
    plt.xlabel("Number of iterations")
    plt.ylabel("Worst-case bounds")
    plt.legend(["Worst-case guarantee", "Conjecture", "Approximation"])
    plt.title("Comparison between conjectured and true worst-case guarantee")
    plt.show()
