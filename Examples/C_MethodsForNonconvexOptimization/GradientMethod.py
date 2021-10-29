from PEPit.pep import PEP
from PEPit.Function_classes.smooth_function import SmoothFunction


def wc_gd(L, gamma, n, verbose=True):
    """
    Consider the minimization problem
        f_* = min_x f(x),
    where f is L-smooth.

    This code computes a worst-case guarantee for the gradient method fixed step size. That is, it computes
    the smallest possible tau(n, L, mu) such that the guarantee
        min_n ||f'(x_n)||^2 <= tau(n, L) * (f(x_0) - f(x_n))
    is valid, where x_n is the output of the gradient descent with fixed step sizes,
    and where x_n is the n-th iterates obtained with the gradient method.

    :param L: (float) the smoothness parameter.
    :param gamma: (float) the step size.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothFunction, param={'L': L})

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value g0 and f0
    x0 = problem.set_initial_point()
    g0, f0 = func.oracle(x0)

    # Run n steps of GD method with step size gamma
    x = x0
    gx, fx = g0, f0
    for i in range(n):
        x = x - gamma * gx
        # Set the performance metric to the minimum of the gradient norm over the iterations
        problem.set_performance_metric(gx ** 2)
        gx, fx = func.oracle(x)

    # Set the initial constraint that is the difference between fN and f0
    problem.set_initial_condition(f0 - fx <= 1)

    # Set the performance metric to the minimum of the gradient norm over the iterations
    problem.set_performance_metric(gx ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 2 * L / n

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of gradient descent with fixed step size ***')
        print('\tPEP-it guarantee:\t\t min_i (f\'(x_i)) ** 2 <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t min_i (f\'(x_i)) <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 5
    L = 1
    gamma = 1 / L

    pepit_tau, theoretical_tau = wc_gd(L=L,
                                       gamma=gamma,
                                       n=n)
