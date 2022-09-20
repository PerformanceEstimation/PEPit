import numpy as np

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction
from PEPit.point import Point


def wc_randomized_coordinate_descent_smooth_convex(L, gamma, d, n, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is convex and :math:`L`-smooth.

    This code computes a worst-case guarantee for **randomized block-coordinate descent** with fixed step-size :math:`\\gamma`.
    That is, it verifies that the inequality holds (the expectation is over the index of the block of coordinates that is randomly selected)

    .. math:: \\mathbb{E}_i[\\phi(x_{t+1}^{(i)})] \\leqslant \\phi(x_{t}),

    where :math:`x_{t+1}^{(i)}` denotes the value of the iterate :math:`x_{t+1}` in the scenario
    where the :math:`i` th block of coordinates is selected for the update  with fixed step-size
    :math:`\\gamma`, and :math:`d` is the number of blocks of coordinates.

    In short, for given values of :math:`L`, :math:`d`, and :math:`\\gamma`, it computes the worst-case value
    of :math:`\\mathbb{E}_i[\\phi(x_{t+1}^{(i)})]` such that :math:`\\phi(x_{t}) \\leqslant 1`.

    **Algorithm**:
    Randomized block-coordinate descent is described by

    .. math::
        \\begin{eqnarray}
            \\text{Pick random }i & \\sim & \\mathcal{U}\\left([|1, d|]\\right), \\\\
            x_{t+1}^{(i)} & = & x_t - \\gamma \\nabla_i f(x_t),
        \\end{eqnarray}

    where :math:`\\gamma` is a step-size and :math:`\\nabla_i f(x_t)` is the partial derivative corresponding to the block :math:`i`.

    **Theoretical guarantee**:
    When :math:`\\gamma \\leqslant \\frac{1}{L}`, the **tight** theoretical guarantee can be found in [1, Appendix I, Theorem 16]:

    .. math:: \\mathbb{E}_i[\\phi(x^{(i)}_{t+1})] \\leqslant \\phi(x_{t}),

    where :math:`\\phi(x_t) = d_t (f(x_t) - f_\\star) + \\frac{L}{2} \|x_t - x_\\star\|^2`, :math:`d_{t+1} = d_t + \\frac{\\gamma L}{d}`,
    and :math:`d_t \\geqslant 1`.

    **References**:

    `[1] A. Taylor, F. Bach (2021). Stochastic first-order methods: non-asymptotic and computer-aided
    analyses via potential functions. In Conference on Learning Theory (COLT).
    <https://arxiv.org/pdf/1902.00947.pdf>`_

    Args:
        L (float): the smoothness parameter.
        gamma (float): the step-size.
        d (int): the dimension.
        n (int): number of iterations.
        verbose (int): Level of information details to print.

                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> L = 1
        >>> pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_convex(L=L, gamma=1 / L, d=2, n=4, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 12x12
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (9 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                         function 1 : Adding 42 scalar constraint(s) ...
                         function 1 : 42 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.9999978377393944
        *** Example file: worst-case performance of randomized  coordinate gradient descent ***
                PEPit guarantee:         E[phi_(n+1)(x_(n+1))] <= 0.999998 phi_n(x_n)
                Theoretical guarantee:   E[phi_(n+1)(x_(n+1))] <= 1.0 phi_n(x_n)

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Run n-1 steps of the algorithm (and keep the last one)
    x = x0
    for _ in range(n - 1):
        # Compute the gradients
        g = func.gradient(x)
        gradients = []
        for i in range(d - 1):
            gradients.append(Point())
        gd = g - np.sum(gradients)  # Define the last point as a function of the whole gradient and past iterates
        gradients.append(gd)
        # Add orthogonality constraints
        for i in range(d):
            for j in range(d):
                if i != j:
                    problem.add_constraint(gradients[i] * gradients[j] == 0)
        # Select randomly a value for i_k in [1, d]
        i_k = np.random.randint(d)
        # Compute a randomized coordinate descent step
        x = x - gamma * gradients[i_k]

    # Compute the Lyapunov at iteration n-1
    if n >= 2:
        dn = (n - 1) * gamma * L / d + 1
    else:
        dn = 1
    phi1 = dn * (func(x) - fs) + L / 2 * (x - xs) ** 2
    problem.set_initial_condition(phi1 == 1)

    # Run the last step of the algorithm and keep gradients in memory
    g = func.gradient(x)
    gradients = []
    for i in range(d - 1):
        gradients.append(Point())
    gd = g - np.sum(gradients)  # Define the last point as a function of the whole gradient and past iterates
    gradients.append(gd)
    # Add orthogonality constraints
    for i in range(d):
        for j in range(d):
            if i != j:
                problem.add_constraint(gradients[i] * gradients[j] == 0)

    # Compute the d possible value for x1 using coordinate descent
    x1 = []
    for grad in gradients:
        x1.append(x - gamma * grad)

    var = np.mean([(dn + gamma * L / d) * (func(xn) - fs) + L / 2 * (xn - xs) ** 2 for xn in x1])

    # Set the performance metric to the expected Lyapunov at iteration n
    problem.set_performance_metric(var)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 1.

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of randomized  coordinate gradient descent ***')
        print('\tPEPit guarantee:\t E[phi_(n+1)(x_(n+1))] <= {:.6} phi_n(x_n)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t E[phi_(n+1)(x_(n+1))] <= {:.6} phi_n(x_n)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_convex(L=L, gamma=1 / L, d=2, n=4, verbose=1)
