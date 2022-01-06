from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.operators import MonotoneOperator
from PEPit.operators import CocoerciveOperator
from PEPit.primitive_steps import proximal_step


def wc_three_operator_splitting(L, mu, beta, alpha, theta, verbose=True):
    """
    Consider the monotone inclusion problem

    .. math:: \\mathrm{Find}\\, x:\\, 0\\in Ax + Bx + Cx,

    where :math:`A` is maximally monotone, :math:`B` is :math:`\\beta`-cocoercive and C is the gradient of some
    :math:`L`-smooth :math:`\\mu`-strongly convex function. We denote by :math:`J_{\\alpha A}` and :math:`J_{\\alpha B}`
    the resolvent of respectively :math:`A` and :math:`B`, with step-size :math:`\\alpha`.

    This code computes a worst-case guarantee for the **three operator splitting** (TOS).
    That is, given two initial points :math:`w^{(0)}_t` and :math:`w^{(1)}_t`,
    this code computes the smallest possible :math:`\\tau(L, \\mu, \\beta, \\alpha, \\theta)`
    (a.k.a. "contraction factor") such that the guarantee

    .. math:: \\|w^{(0)}_{t+1} - w^{(1)}_{t+1}\\|^2 \\leqslant \\tau(L, \\mu, \\beta, \\alpha, \\theta) \\|w^{(0)}_{t} - w^{(1)}_{t}\\|^2,

    is valid, where :math:`w^{(0)}_{t+1}` and :math:`w^{(1)}_{t+1}` are obtained after one iteration of TOS from
    respectively :math:`w^{(0)}_{t}` and :math:`w^{(1)}_{t}`.

    In short, for given values of :math:`L`, :math:`\\mu`, :math:`\\beta`, :math:`\\alpha` and :math:`\\theta`,
    the contraction factor :math:`\\tau(L, \\mu, \\beta, \\alpha, \\theta)` is computed as the worst-case value of
    :math:`\\|w^{(0)}_{t+1} - w^{(1)}_{t+1}\\|^2` when :math:`\\|w^{(0)}_{t} - w^{(1)}_{t}\\|^2 \\leqslant 1`.

    **Algorithm**:
    One iteration of the algorithm is described in [1]. For :math:`t \in \\{ 0, \\dots, n-1\\}`,

        .. math::
            :nowrap:

            \\begin{eqnarray}
                x_{t+1} & = & J_{\\alpha B} (w_t),\\\\
                y_{t+1} & = & J_{\\alpha A} (2x_{t+1} - w_t - C x_{t+1}),\\\\
                w_{t+1} & = & w_t - \\theta (x_{t+1} - y_{t+1}).
            \\end{eqnarray}

    **References**: The TOS was proposed in [1], the analysis of such operator splitting methods using PEPs was proposed in [2].

    `[1] D. Davis, W. Yin (2017). A three-operator splitting scheme and its optimization applications.
    Set-valued and variational analysis, 25(4), 829-858.
    <https://arxiv.org/pdf/1504.01032.pdf>`_

    `[2] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020). Operator splitting performance estimation:
    Tight contraction factors and optimal parameter selection. SIAM Journal on Optimization, 30(3), 2251-2271.
    <https://arxiv.org/pdf/1812.00146.pdf>`_

    Args:
        L (float): smoothness constant of C.
        mu (float): strong convexity of C.
        beta (float): cocoercivity of B.
        alpha (float): step-size (in the resolvants).
        theta (float): overrelaxation parameter.
        verbose (bool): if True, print conclusion.

    Returns:
        pepit_tau (float): worst-case value.
        theoretical_tau (None): no theoretical value.

    Example:
        >>> pepit_tau, theoretical_tau = wc_three_operator_splitting(L=1, mu=.1, beta=1, alpha=.9, theta=1.3, verbose=True)
        (PEPit) Setting up the problem: size of the main PSD matrix: 8x8
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: initial conditions (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 3 function(s)
                 function 1 : 2 constraint(s) added
                 function 2 : 2 constraint(s) added
                 function 3 : 2 constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (solver: MOSEK); optimal value: 0.7796890317399627
        *** Example file: worst-case contraction factor of the Three Operator Splitting ***
                PEPit guarantee:	 ||w_(t+1)^0 - w_(t+1)^1||^2 <= 0.779689 ||w_(t)^0 - w_(t)^1||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a monotone operator
    A = problem.declare_function(MonotoneOperator, param={})
    B = problem.declare_function(CocoerciveOperator, param={'beta': beta})
    C = problem.declare_function(SmoothStronglyConvexFunction, param={'L': L, 'mu': mu})

    # Then define the starting points w0 and w1
    w0 = problem.set_initial_point()
    w1 = problem.set_initial_point()

    # Set the initial constraint that is the distance between w0 and w1
    problem.set_initial_condition((w0 - w1) ** 2 <= 1)

    # Compute one step of the Three Operator Splitting starting from w0
    x0, _, _ = proximal_step(w0, B, alpha)
    y0, _, _ = proximal_step(2 * x0 - w0 - alpha * C.gradient(x0), A, alpha)
    z0 = w0 - theta * (x0 - y0)

    # Compute one step of the Three Operator Splitting starting from w1
    x1, _, _ = proximal_step(w1, B, alpha)
    y1, _, _ = proximal_step(2 * x1 - w1 - alpha * C.gradient(x1), A, alpha)
    z1 = w1 - theta * (x1 - y1)

    # Set the performance metric to the distance between z0 and z1
    problem.set_performance_metric((z0 - z1) ** 2)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = None

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case contraction factor of the Three Operator Splitting ***')
        print('\tPEPit guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= {:.6} ||w_(t)^0 - w_(t)^1||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_three_operator_splitting(L=1, mu=.1, beta=1, alpha=.9, theta=1.3, verbose=True)
