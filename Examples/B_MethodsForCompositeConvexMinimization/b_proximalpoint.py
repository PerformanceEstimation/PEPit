from PEPit.pep import PEP
from PEPit.Function_classes.cvx_function import CvxFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_ppa(gamma, n):
    """
    DEF PROBLEM (minimisation fct convexe)

    DEF ALGO

    DEF PIRE CAS

    REF avec solution analytique?

    This code computes the worst-case convergence bound for the proximal point method towards a minimizer of a convex function f.
    That is, we compute the smallest value of "tau" such that the inequality
    f(x_n)-f_* <= tau(N,gamma) * || x_0 - x_* ||^2
    is valid for all x_0 and x_*, and all L-smooth m-strongly convex function f with x_* being a minimizer of f and
    x_{n} being computed as a gradient step; x_{k+1}=x_k - gamma * grad f (x_k) (k=0,...,n-1)

    :param gamma: (float) step size.
    :param n: (int) number of iterations.
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    f1 = problem.declare_function(CvxFunction,{})
    #f2 = problem.declare_function(CvxFunction,{})
    func = 2*f1

    # Start by defining its unique optimal point
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the GD method
    x = x0
    for _ in range(n):
        x,_,fx = proximal_step(x,func,gamma)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric(fx-fs)

    # Solve the PEP
    wc = problem.solve()

    # Return the rate of the evaluated method
    # This should match 1/4/gamma/n
    return wc


if __name__ == "__main__":

    n = 2
    gamma = 1

    rate = wc_ppa(gamma, n)

    print('{}'.format(rate))
