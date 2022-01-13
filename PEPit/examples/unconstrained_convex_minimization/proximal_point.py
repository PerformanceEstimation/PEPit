from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_proximal_point(gamma, n, verbose=True):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is closed, proper, and convex (and potentially non-smooth).

    This code computes a worst-case guarantee for the **proximal point method** with step-size :math:`\\gamma`.
    That is, it computes the smallest possible :math:`\\tau(n,\\gamma)` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, \\gamma)  \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the proximal point method, and where :math:`x_\\star` is a
    minimizer of :math:`f`.

    In short, for given values of :math:`n` and :math:`\\gamma`,
    :math:`\\tau(n,\\gamma)` is computed as the worst-case value of :math:`f(x_n)-f_\\star`
    when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.

    **Algorithm**:

    The proximal point method is described by

        .. math:: x_{t+1} = \\arg\\min_x \\left\\{f(x)+\\frac{1}{2\gamma}\\|x-x_t\\|^2 \\right\\},

    where :math:`\\gamma` is a step-size.

    **Theoretical guarantee**:

    The **tight** theoretical guarantee can be found in [1, Theorem 4.1]:

        .. math:: f(x_n)-f_\\star \\leqslant \\frac{\\|x_0-x_\\star\\|^2}{4\\gamma n},

    where tightness is obtained on, e.g., one-dimensional linear problems on the positive orthant.

    **References**:

    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017). Exact worst-case performance of first-order methods for composite
    convex optimization. SIAM Journal on Optimization, 27(3):1283–1313. <https://arxiv.org/pdf/1512.07516.pdf>`_

    Args:
        gamma (float): step-size.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_proximal_point(gamma=3, n=4, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 6x6
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 20 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); optimal value: 0.020833335685727362
        *** Example file: worst-case performance of proximal point method ***
            PEPit guarantee:           f(x_n)-f_* <= 0.0208333 ||x_0 - x_*||^2
            Theoretical guarantee:      f(x_n)-f_* <= 0.0208333 ||x_0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function
    func = problem.declare_function(ConvexFunction, param={})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the proximal point method
    x = x0
    for _ in range(n):
        x, _, fx = proximal_step(x, func, gamma)

    # Set the performance metric to the final distance to optimum in function values
    problem.set_performance_metric(fx - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 1 / (4 * gamma * n)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of proximal point method ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_proximal_point(gamma=3, n=4, verbose=True)
