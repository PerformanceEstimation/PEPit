import numpy as np

from PEPit.pep import PEP
from PEPit.operators.lipschitz import LipschitzOperator
from PEPit.primitive_steps.fixed_point import fixed_point


def wc_krasnoselskii_mann_constant_step_sizes(n, gamma, verbose=True):
    """
    Consider the fixed point problem

    .. math:: \\mathrm{Find}\\, x:\\, x = Ax,

    where :math:'A' is a non-expansive operator, that is a :math:`L`-Lipschitz operator with :math:`L=1`.

    This code computes a worst-case guarantee for the **Krasnolselskii-Mann** method. That is, it computes
    the smallest possible :math:`\tau(n)` such that the guarantee

        .. math:: \\frac{1}{4}\\|x_n - Ax_n\\|^2 \\leqslant \\tau(n) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the Krasnolseskii-Mann iterations, and :math:`x_\\star` the fixed point of :math:`A.`

    **Algorithm**:

        .. math:: x_{t+1} = \\left(1 - \\gamma\\right) x_{t} + \\gamma Ax_{t}

    **Theoretical guarantee**:

        The theoretical **upper** bound is given in [1, Theorem 4.9]

            .. math:: \\tau(n) = \\left\{
                      \\begin{eqnarray}
                          \\frac{1}{n+1}\\left(\\frac{n}{n+1}\\right)^n \\frac{1}{4 \\gamma (1 - \\gamma)}, & ~~~ \\text{if} & ~~~~~~~~~~~~~~~~ \\frac{1}{2} & \\leqslant & \\gamma & \\leqslant & \\frac{1}{2}\\left(1+\\sqrt{\\frac{n}{n+1}}\\right) \\\\
                          (\\gamma - 1)^{2n} ~~~~~~~~~~~~~~~~, & ~~~ \\text{if} & \\frac{1}{2}\\left(1+\\sqrt{\\frac{n}{n+1}}\\right) & < & \\gamma & \\leqslant & 1. ~~~~~~~~~~~~~~~~
                      \\end{eqnarray}
                      \\right.

    **Reference**:

        This scheme was first studied using PEPs in [1, Theorem 4.9]:

        `[1] Felix Lieder. "Projection Based Methods for Conic Linear Programming
        Optimal First Order Complexities and Norm Constrained Quasi Newton
        Methods."  PhD thesis (2018)
        <https://docserv.uni-duesseldorf.de/servlets/DerivateServlet/Derivate-49971/Dissertation.pdf>`_

    Args:
        n (int): number of iterations.
        gamma (float): step-size between 1/2 and 1
        verbose (bool, optional): if True, print conclusion

    Returns:
        tuple: worst-case value, theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_krasnoselskii_mann_constant_step_sizes(n=3, gamma=3/4, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 6x6
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 1 function(s)
                 function 1 : 20 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.14062586461718285
        *** Example file: worst-case performance of Kranoselskii-Mann iterations ***
            PEP-it guarantee:		 1/4||xN - AxN||^2 <= 0.140626 ||x0 - x_*||^2
            Theoretical guarantee:	 1/4||xN - AxN||^2 <= 0.140625 ||x0 - x_*||^2

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
        x = (1-gamma) * x + gamma * A.gradient(x)

    # Set the performance metric to distance between xN and AxN
    problem.set_performance_metric((1 / 2 * (x - A.gradient(x))) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    if 1/2 <= gamma <= 1 / 2 * (1 + np.sqrt(n / (n + 1))):
        theoretical_tau = 1 / (n + 1) * (n / (n + 1)) ** n / (4 * gamma * (1 - gamma))
    elif 1 / 2 * (1 + np.sqrt(n / (n + 1))) < gamma <= 1:
        theoretical_tau = (2 * gamma - 1) ** (2 * n)
    else:
        raise ValueError("{} is not a valid value for the step-size \'gamma\'."
                         " \'gamma\' must be a number between 1/2 and 1".format(gamma))

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of Kranoselskii-Mann iterations ***')
        print('\tPEP-it guarantee:\t\t 1/4||xN - AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t 1/4||xN - AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_krasnoselskii_mann_constant_step_sizes(n=3, gamma=3 / 4, verbose=True)