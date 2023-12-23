from math import sqrt

from PEPit import PEP
from PEPit.operators import LipschitzOperator


def wc_krasnoselskii_mann_constant_step_sizes(n, gamma, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the fixed point problem

    .. math:: \\mathrm{Find}\\, x:\\, x = Ax,

    where :math:`A` is a non-expansive operator, that is a :math:`L`-Lipschitz operator with :math:`L=1`.

    This code computes a worst-case guarantee for the **Krasnolselskii-Mann** (KM) method with constant step-size.
    That is, it computes the smallest possible :math:`\\tau(n)` such that the guarantee

        .. math:: \\frac{1}{4}\\|x_n - Ax_n\\|^2 \\leqslant \\tau(n) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the KM method, and :math:`x_\\star` is some fixed point of :math:`A`
    (i.e., :math:`x_\\star=Ax_\\star`).

    **Algorithm**: The constant step-size KM method is described by

        .. math:: x_{t+1} = \\left(1 - \\gamma\\right) x_{t} + \\gamma Ax_{t}.

    **Theoretical guarantee**: A theoretical **upper** bound is provided by [1, Theorem 4.9]

            .. math:: \\tau(n) = \\left\{
                      \\begin{eqnarray}
                          \\frac{1}{n+1}\\left(\\frac{n}{n+1}\\right)^n \\frac{1}{4 \\gamma (1 - \\gamma)}\quad & \\text{if } \\frac{1}{2}\\leqslant \\gamma  \\leqslant \\frac{1}{2}\\left(1+\\sqrt{\\frac{n}{n+1}}\\right) \\\\
                          (\\gamma - 1)^{2n} \quad & \\text{if } \\frac{1}{2}\\left(1+\\sqrt{\\frac{n}{n+1}}\\right) <  \\gamma \\leqslant  1.
                      \\end{eqnarray}
                      \\right.

    **Reference**:

    `[1] F. Lieder (2018).
    Projection Based Methods for Conic Linear Programming
    Optimal First Order Complexities and Norm Constrained Quasi Newton Methods.
    PhD thesis, HHU Düsseldorf.
    <https://docserv.uni-duesseldorf.de/servlets/DerivateServlet/Derivate-49971/Dissertation.pdf>`_

    Args:
        n (int): number of iterations.
        gamma (float): step-size between 1/2 and 1
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_krasnoselskii_mann_constant_step_sizes(n=3, gamma=3 / 4, wrapper="cvxpy", solver=None, verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 6x6
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 1 function(s)
        			Function 1 : Adding 10 scalar constraint(s) ...
        			Function 1 : 10 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: MOSEK); optimal value: 0.1406249823498115
        (PEPit) Primal feasibility check:
        		The solver found a Gram matrix that is positive semi-definite up to an error of 2.6923151650319117e-09
        		All the primal scalar constraints are verified up to an error of 1.7378567473969042e-09
        (PEPit) Dual feasibility check:
        		The solver found a residual matrix that is positive semi-definite
        		All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 4.412159703651858e-08
        (PEPit) Final upper bound (dual): 0.14062498615478927 and lower bound (primal example): 0.1406249823498115 
        (PEPit) Duality gap: absolute: 3.804977777299712e-09 and relative: 2.7057623145755453e-08
        *** Example file: worst-case performance of Kranoselskii-Mann iterations ***
        	PEPit guarantee:	 1/4||xN - AxN||^2 <= 0.140625 ||x0 - x_*||^2
        	Theoretical guarantee:	 1/4||xN - AxN||^2 <= 0.140625 ||x0 - x_*||^2
    
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a non expansive operator
    A = problem.declare_function(LipschitzOperator, L=1.)

    # Start by defining its unique optimal point xs = x_*
    xs, _, _ = A.fixed_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the difference between x0 and xs
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    x = x0
    for i in range(n):
        x = (1 - gamma) * x + gamma * A.gradient(x)

    # Set the performance metric to distance between xN and AxN
    problem.set_performance_metric((1 / 2 * (x - A.gradient(x))) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    if 1 / 2 <= gamma <= 1 / 2 * (1 + sqrt(n / (n + 1))):
        theoretical_tau = 1 / (n + 1) * (n / (n + 1)) ** n / (4 * gamma * (1 - gamma))
    elif 1 / 2 * (1 + sqrt(n / (n + 1))) < gamma <= 1:
        theoretical_tau = (2 * gamma - 1) ** (2 * n)
    else:
        raise ValueError("{} is not a valid value for the step-size \'gamma\'."
                         " \'gamma\' must be a number between 1/2 and 1".format(gamma))

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of Kranoselskii-Mann iterations ***')
        print('\tPEPit guarantee:\t 1/4||xN - AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t 1/4||xN - AxN||^2 <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_krasnoselskii_mann_constant_step_sizes(n=3, gamma=3 / 4,
                                                                           wrapper="cvxpy", solver=None,
                                                                           verbose=1)
