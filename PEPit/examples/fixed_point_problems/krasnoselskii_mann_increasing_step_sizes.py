from PEPit import PEP
from PEPit.operators import LipschitzOperator


def wc_krasnoselskii_mann_increasing_step_sizes(n, verbose=True):
    """
    Consider the fixed point problem

    .. math:: \\mathrm{Find}\\, x:\\, x = Ax,

    where :math:`A` is a non-expansive operator, that is a :math:`L`-Lipschitz operator with :math:`L=1`.

    This code computes a worst-case guarantee for the **Krasnolselskii-Mann** method. That is, it computes
    the smallest possible :math:`\\tau(n)` such that the guarantee

        .. math:: \\frac{1}{4}\\|x_n - Ax_n\\|^2 \\leqslant \\tau(n) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the KM method, and :math:`x_\\star` is some fixed point of :math:`A`
    (i.e., :math:`x_\\star=Ax_\\star`).

    **Algorithm**: The KM method is described by

        .. math:: x_{t+1} = \\frac{1}{t + 2} x_{t} + \\left(1 - \\frac{1}{t + 2}\\right) Ax_{t}.

    **Reference**: This scheme was first studied using PEPs in [1].

    `[1] F. Lieder (2018). Projection Based Methods for Conic Linear Programming
    Optimal First Order Complexities and Norm Constrained Quasi Newton Methods.  PhD thesis, HHU Düsseldorf.
    <https://docserv.uni-duesseldorf.de/servlets/DerivateServlet/Derivate-49971/Dissertation.pdf>`_

    Args:
        n (int): number of iterations.
        verbose (bool, optional): if True, print conclusion

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (None): no theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_krasnoselskii_mann_increasing_step_sizes(n=3, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 6x6
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 20 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: SCS); optimal value: 0.11963406474148795
        *** Example file: worst-case performance of Kranoselskii-Mann iterations ***
            PEPit guarantee:		 1/4||xN - AxN||^2 <= 0.119634 ||x0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a non expansive operator
    A = problem.declare_function(LipschitzOperator, param={'L': 1.})

    # Start by defining its unique optimal point xs = x_*
    xs, _, _ = A.fixed_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the difference between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    x = x0
    for i in range(n):
        x = 1 / (i + 2) * x + (1 - 1 / (i + 2)) * A.gradient(x)

    # Set the performance metric to distance between xN and AxN
    problem.set_performance_metric((1 / 2 * (x - A.gradient(x))) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = None

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of Kranoselskii-Mann iterations ***')
        print('\tPEPit guarantee:\t 1/4||xN - AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_krasnoselskii_mann_increasing_step_sizes(n=3, verbose=True)
