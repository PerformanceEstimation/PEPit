from PEPit.pep import PEP
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_pgd(L, mu, gamma, n):
    """
    DEF PROBLEM (minimisation fct convexe)

    DEF ALGO

    DEF PIRE CAS

    REF avec solution analytique?

    :param L:
    :param mu:
    :param gamma:
    :param n:
    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    f1 = problem.declare_function(SmoothStronglyConvexFunction, {'mu': mu, 'L': L})
    f2 = problem.declare_function(ConvexFunction, {})
    func = f1 + f2

    # Start by defining its unique optimal point
    xs = func.optimal_point()

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run the GD method
    x = x0
    for _ in range(n):
        y = x - gamma * f1.gradient(x)
        x, _, _ = proximal_step(y, f2, gamma)

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric((x - xs) ** 2)

    # Solve the PEP
    wc = problem.solve()

    # Return the worst-case rate of the method, which should be max((1-gamma*mu)^2,(1-gamma*L)^2)^n
    return wc


if __name__ == "__main__":
    n = 2
    L = 1
    mu = .1
    gamma = 1

    rate = wc_pgd(L, mu, gamma, n)

    print('{}'.format(rate))
