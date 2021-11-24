import numpy as np

from PEPit.pep import PEP
from PEPit.operators.lipschitz import LipschitzOperator
from PEPit.primitive_steps.fixed_point import fixed_point


def wc_km(n, verbose=True):
    """
    Consider the fixed point problem

        .. math:: \\text{Find} \ x \ \\text{such that} \ x = Ax,

    where :math:'A' is a non-expansive operator, that is a :math:`L`-Lipschitz operator with :math:`L=1`.

    This code computes a worst-case guarantee for the **Krasnolselskii-Mann**. That is, it computes
    the smallest possible :math:`\tau(n)` such that the guarantee

        .. math:: \\frac{1}{4}|| x_n - Ax_n||^2 \\leqslant \\tau(n) ||x_0 - x_\star||^2

    is valid, where :math:`x_n` is the output of the Krasnolseskii-Mann iterations, and :math:`x_\star` the fixed point of :math:`A`.

    **Algorithm**:

        .. math:: x_{i+1} = 1 / (i + 2) * x_{i} + (1 - 1 / (i + 2)) Ax_{i}

    **Theoretical guarantee**:

    The theoretical **upper bound** is given in [1, Theorem 4.9]

        .. math:: t_n = 1 - \\frac{1}{n+1}

        If :math:`\\frac{1}{2} \\leqslant t_n \\leqslant \\frac{1}{2}(1+\\sqrt{\\frac{n}{n+1}})`,

            .. math:: \\tau(n) = \\frac{1}{n+1}\\frac{n}{n+1}^n \\frac{1}{4 t_n (1 - t_n)}
        Else :

            .. math:: \\tau(n) = (2t_n - 1)^{2n}

    **Reference**:

    This scheme was first studied using PEPs in [1, Theorem 4.9]:

    [1] Felix Lieder. "Projection Based Methods for Conic Linear Programming
    Optimal First Order Complexities and Norm Constrained Quasi Newton
    Methods."  PhD thesis (2018)

    Args:
        L (float): the Lipschitz parameter.
        n (int): number of iterations.
        verbose (bool, optional): if True, print conclusion

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_km(n=n)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 6x6
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 20 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.11963406475118304
        *** Example file: worst-case performance of Kranoselskii-Mann iterations ***
            PEP-it guarantee:		 1/4|| xN - AxN ||^2 <= 0.119634 ||x0 - x_*||^2
            Theoretical guarantee:	 1/4|| xN - AxN ||^2 <= 0.140625 ||x0 - x_*||^2
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a non expansive operator
    A = problem.declare_function(LipschitzOperator, param={'L': 1.})

    # Start by defining its unique optimal point xs = x_*
    xs, _, _ = fixed_point(A)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the difference between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    x = x0
    for i in range(n):
        x = 1 / (i + 2) * x + (1 - 1 / (i + 2)) * A.gradient(x)
    Ax = A.gradient(x)

    # Set the performance metric to distance between xN and AxN
    problem.set_performance_metric((1 / 2 * (x - Ax)) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    tn = 1 - 1 / (n + 1)
    if (tn >= 1 / 2) & (tn <= 1 / 2 * (1 + np.sqrt(n / (n + 1)))):
        theoretical_tau = 1 / (n + 1) * (n / (n + 1)) ** n / (4 * tn * (1 - tn))
    if (tn <= 1) & (tn > 1 / 2 * (1 + np.sqrt(n / (n + 1)))):
        theoretical_tau = (2 * tn - 1) ** (2 * n)

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of Kranoselskii-Mann iterations ***')
        print('\tPEP-it guarantee:\t\t 1/4|| xN - AxN ||^2 <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t 1/4|| xN - AxN ||^2 <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    n = 3
    pepit_tau, theoretical_tau = wc_km(n=n)
