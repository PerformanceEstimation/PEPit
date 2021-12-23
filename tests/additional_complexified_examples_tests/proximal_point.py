from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.primitive_steps import proximal_step


def wc_proximal_point_complexified(gamma, n, verbose=True):
    """
    See description in `PEPit/examples/unconstrained_convex_minimization/proximal_point.py`.
    This example is for testing purposes; the worst-case result is supposed to be the same as that of the other routine,
    but the parameterization is different (convex function to be minimized is explicitly formed as a sum of two convex
    functions). That is, the minimization problem is

    .. math:: f_\\star = \\min_x \\{f(x) \\equiv f_1(x) + f_2(x)\\},

    where :math:`f_1` and :math:`f_2` are closed, proper, and convex (and potentially non-smooth) functions.

    Args:
        gamma (float): the step size parameter.
        n (int): number of iterations.
        verbose (bool): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_proximal_point_complexified(gamma=1, n=2)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 7x7
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 2 function(s)
                 function 1 : 6 constraint(s) added
                 function 2 : 6 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.12500022120131604
        *** Example file: worst-case performance of the Proximal Point Method in function values***
            PEP-it guarantee:		 f(x_n)-f_* <= 0.125 ||x0 - xs||^2
            Theoretical guarantee :	 f(x_n)-f_* <= 0.125 ||x0 - xs||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function
    convex_1 = problem.declare_function(ConvexFunction, param={})
    convex_2 = problem.declare_function(ConvexFunction, param={})
    func = convex_1 + convex_2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the proximal point method
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
        print('*** Example file: worst-case performance of the Proximal Point Method in function values***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee :\t f(x_n)-f_* <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_proximal_point_complexified(gamma=1, n=2)
