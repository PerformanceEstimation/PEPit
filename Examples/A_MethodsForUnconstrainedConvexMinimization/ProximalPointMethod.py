from PEPit.pep import PEP
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_ppa(gamma, n, verbose=True):
    """
    Consider the minimization problem
        f_* = min_x f(x),
    where f is convex.

    This code computes a worst-case guarantee for the proximal point method. That is, it computes
    the smallest possible tau(n) such that the guarantee
        f(x_n) - f_* <= tau(n) * ||x_0 - x_*||^2
    is valid, where x_n is the output of the proximal point method, and where x_* is the minimizer of f.

    In short, for given values of n, tau(n) is computed as the worst-case value of f(x_n)-f_* when
   ||x_0 - x_*||^2 <= 1.

    :param gamma: (float) step size.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function
    func = problem.declare_function(ConvexFunction, {})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
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
    pepit_tau = problem.solve()

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 1/4/gamma/n

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of proximal point method ***')
        print('\tPEP-it guarantee:\t\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 2
    gamma = 1

    rate = wc_ppa(gamma=gamma,
                  n=n)
