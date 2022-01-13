from math import sqrt

from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_accelerated_proximal_point(A0, gammas, n, verbose=True):
    """
    Consider the minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is  convex and possibly non-smooth.

    This code computes a worst-case guarantee an **accelerated proximal point** method, aka **fast proximal point** method (FPP).
    That is, it computes the smallest possible :math:`\\tau(n, A_0,\\vec{\\gamma})` such that the guarantee

    .. math:: f(x_n) - f_\\star \\leqslant \\tau(n, A_0, \\vec{\\gamma}) \\left(f(x_0) - f_\\star + \\frac{A_0}{2}  \\|x_0 - x_\\star\\|^2\\right)

    is valid, where :math:`x_n` is the output of FPP (with step-size :math:`\\gamma_t` at step :math:`t\\in \\{0, \\dots, n-1\\}`) and where :math:`x_\\star` is a minimizer of :math:`f` and :math:`A_0` is a positive number.

    In short, for given values of :math:`n`,  :math:`A_0` and :math:`\\vec{\\gamma}`, :math:`\\tau(n)` is computed as the worst-case value
    of :math:`f(x_n)-f_\\star` when :math:`f(x_0) - f_\\star + \\frac{A_0}{2} \\|x_0 - x_\\star\\|^2 \\leqslant 1`, for the following method.

    **Algorithm**:
    For :math:`t\\in \\{0, \\dots, n-1\\}`:

       .. math::
           :nowrap:

           \\begin{eqnarray}
               y_{t+1} & = & (1-\\alpha_{t} ) x_{t} + \\alpha_{t} v_t \\\\
               x_{t+1} & = & \\arg\\min_x \\left\\{f(x)+\\frac{1}{2\\gamma_t}\\|x-y_{t+1}\\|^2 \\right\\}, \\\\
               v_{t+1} & = & v_t + \\frac{1}{\\alpha_{t}} (x_{t+1}-y_{t+1})
           \\end{eqnarray}

    with

       .. math::
           :nowrap:

           \\begin{eqnarray}
               \\alpha_{t} & = & \\frac{\\sqrt{(A_t \\gamma_t)^2 + 4 A_t \\gamma_t} - A_t \\gamma_t}{2} \\\\
               A_{t+1} & = & (1 - \\alpha_{t}) A_t
           \\end{eqnarray}

    and :math:`v_0=x_0`.



    **Theoretical guarantee**:
    A theoretical **upper** bound can be found in [1, Theorem 2.3.]:

    .. math:: f(x_n)-f_\\star \\leqslant \\frac{4}{A_0 (\\sum_{t=0}^{n-1} \\sqrt{\\gamma_t})^2}\\left(f(x_0) - f_\\star + \\frac{A_0}{2}  \\|x_0 - x_\\star\\|^2 \\right).

    **References**:
    The accelerated proximal point was first obtained and analyzed in [1].

    `[1] O. Güler (1992).
    New proximal point algorithms for convex minimization.
    SIAM Journal on Optimization, 2(4):649–664.
    <https://epubs.siam.org/doi/abs/10.1137/0802032?mobileUi=0>`_


    Args:
       A0 (float): initial value for parameter A_0.
       gammas (list): sequence of step-sizes.
       n (int): number of iterations.
       verbose (bool): if True, print conclusion

    Returns:
       pepit_tau (float): worst-case value
       theoretical_tau (float): theoretical value


    Example:
       >>> pepit_tau, theoretical_tau = wc_accelerated_proximal_point(A0=5, gammas=[(i + 1) / 1.1 for i in range(3)], n=3, verbose=True)
       (PEPit) Setting up the problem: size of the main PSD matrix: 6x6
       (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
       (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
       (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 20 constraint(s) added
       (PEPit) Compiling SDP
       (PEPit) Calling SDP solver
       (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.01593113594082973
       *** Example file: worst-case performance of optimized gradient method ***
           PEPit guarantee:       f(x_n)-f_* <= 0.0159311  (f(x_0) - f_* + A0/2* ||x_0 - x_*||^2)
           Theoretical guarantee:  f(x_n)-f_* <= 0.0511881  (f(x_0) - f_* + A0/2* ||x_0 - x_*||^2)

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

    # Set the initial constraint that is a well-chosen distance between x0 and x^*
    problem.set_initial_condition(func.value(x0) - fs + A0 / 2 * (x0 - xs) ** 2 <= 1)

    # Run the fast proximal point method
    x, v = x0, x0
    A = A0
    for i in range(n):
        alpha = (sqrt((A * gammas[i]) ** 2 + 4 * A * gammas[i]) - A * gammas[i]) / 2
        y = (1 - alpha) * x + alpha * v
        x, _, _ = proximal_step(y, func, gammas[i])
        v = v + 1 / alpha * (x - y)
        A = (1 - alpha) * A

    # Set the performance metric to the final distance to optimum in function values
    problem.set_performance_metric(func.value(x) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    accumulation = 0
    for i in range(n):
        accumulation += sqrt(gammas[i])
    theoretical_tau = 4 / A0 / accumulation ** 2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of fast proximal point method ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0) - f_* + A/2* ||x_0 - x_*||^2)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0) - f_* + A/2* ||x_0 - x_*||^2)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_accelerated_proximal_point(A0=5, gammas=[(i + 1) / 1.1 for i in range(3)], n=3, verbose=True)
