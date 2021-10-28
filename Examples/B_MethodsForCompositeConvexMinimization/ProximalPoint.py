from PEPit.pep import PEP
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_ppa(gamma, n, verbose=True):
    """
    Consider the non-smooth convex minimization problem
        f_* = min_x f1(x) + f2(x),
    where f is convex, and where f2 is a closed convex and proper function.

    This code computes a worst-case guarantee for the proximal point method.
    That is, the code computes the smallest possible tau(n) such that the guarantee
        f(x_n) - f_* <= tau(n) * ||x0 - x_*||^2,
    is valid, where x_n is the output of the proximal gradient, and where x_* is a minimizer of f.


    :param gamma: (float) the step size parameter.
    :param n: (int) number of iterations.
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a convex function
    f1 = problem.declare_function(ConvexFunction, {})
    f2 = problem.declare_function(ConvexFunction,{})
    func = f1 + f2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.optimal_point()
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
    pepit_tau = problem.solve()

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 1/4/gamma/n

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Proximal Point Method in function values***')
        print('\tPEP-it guarantee:\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2'.format(pepit_tau))
        print('\tTheoretical guarantee :\t f(y_n)-f_* <= {:.6} ||x0 - xs||^2 '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 2
    gamma = 1

    pepit_tau, theoretical_tau = wc_ppa(gamma=gamma,
                                        n=n)
