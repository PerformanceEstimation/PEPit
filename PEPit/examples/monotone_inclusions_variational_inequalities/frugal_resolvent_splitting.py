from PEPit import PEP, null_point
from PEPit.primitive_steps import proximal_step
from PEPit.functions import SmoothStronglyConvexFunction
from numpy import array

def wc_frugal_resolvent_splitting(L, W, l_values, mu_values, alpha=1, gamma=0.5, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the the problem

    .. math:: \\mathrm{Find}\\, x:\\, 0 = \\sum_{i=1}^{n} A_i(x),

    where :math:`A_i` is the subgradient of an :math:`l_i`-smooth and :math:`\\mu_i`-strongly convex function for all :math:`i \\leq n`. 
    We denote by :math:`J_{\\alpha A_i}` the resolvent of :math:`\\alpha A_i`. 
    We denote the lifted vector operator :math:`\\mathbf{A}` as :math:`\\mathbf{A} = [A_1, \\dots, A_n]`, 
    and use lifted :math:`\\mathbf{x}^T = [x_1, \\dots, x_n]` and :math:`\\mathbf{v}^T = [v_1, \\dots, v_n]`. 
    We denote by :math:`L, W \\in \\mathbb{R}^{n \\times n}` the algorithm design matrices, and by :math:`l` and :math:`\\mu` the vectors of smoothness and strong convexity constants of the functions associated with the lifted operator :math:`\\mathbf{A}`. 

    This code computes a worst-case guarantee for any frugal resolvent splitting with design matrices :math:`L, W`. 
    As shown in [1] and [2], this can include the Malitsky-Tam [3], Ryu Three Operator Splitting [4], Douglas-Rachford [5], or block splitting algorithms [1].
    That is, given two lifted initial points :math:`\\mathbf{v}^{(0)}_t` and :math:`\\mathbf{v}^{(1)}_t` (each of which sums to 0),
    this code computes the smallest possible :math:`\\tau(L, W, l, \\mu, \\alpha, \\gamma)`
    (a.k.a. "contraction factor") such that the guarantee

    .. math:: \\|\\mathbf{v}^{(0)}_{t+1} - \\mathbf{v}^{(1)}_{t+1}\\|^2 \\leqslant \\tau(L, W, l, \\mu, \\alpha, \\gamma) \\|\\mathbf{v}^{(0)}_{t} - \\mathbf{v}^{(1)}_{t}\\|^2,

    is valid, where :math:`\\mathbf{v}^{(0)}_{t+1}` and :math:`\\mathbf{v}^{(1)}_{t+1}` are obtained after one iteration of the frugal resolvent splitting from respectively :math:`\\mathbf{v}^{(0)}_{t}` and :math:`\\mathbf{v}^{(1)}_{t}`.

    In short, for given values of :math:`L`, :math:`W`, :math:`l`, :math:`\\mu`, :math:`\\alpha` and :math:`\\gamma`, the contraction factor :math:`\\tau(L, W, l, \\mu, \\alpha, \\gamma)` is computed as the worst-case value of
    :math:`\\|\\mathbf{v}^{(0)}_{t+1} - \\mathbf{v}^{(1)}_{t+1}\\|^2` when :math:`\\|\\mathbf{v}^{(0)}_{t} - \\mathbf{v}^{(1)}_{t}\\|^2 \\leqslant 1`.

    **Algorithm**: One iteration of the parameterized frugal resolvent splitting is described as follows:

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\mathbf{x}_{t+1} & = & J_{\\alpha \\mathbf{A}} (\\mathbf{L} \\mathbf{x}_{t+1} + \\mathbf{v}_t),\\\\
                \\mathbf{v}_{t+1} & = & \\mathbf{v}_t - \\gamma \\mathbf{W} \\mathbf{x}_{t+1}.
            \\end{eqnarray}

    :math:`L` is assumed to be strictly lower triangular to make each resolvent :math:`J_{A_i}` of the algorithm rely only on :math:`\\mathbf{v}_i` and the results of the previous resolvents in that iteration (and not subsequent resolvents).

    :math:`W` is assumed to be positive semi-definite with a nullspace equal to the span of the ones vector, so that :math:`W 1 = 0`, ensuring that the output vector sums to 0 like the input.
    
    **References**:

    `[1] R. Bassett, P. Barkley (2024). 
    Optimal Design of Resolvent Splitting Algorithms. arxiv:2407.16159.
    <https://arxiv.org/pdf/2407.16159.pdf>`_

    `[2] M. Tam (2023). Frugal and decentralised resolvent splittings defined by nonexpansive operators. Optimization Letters pp 1–19. <https://arxiv.org/pdf/2211.04594.pdf>`_

    `[3] Y. Malitsky, M. Tam (2023). Resolvent splitting for sums of monotone operators
    with minimal lifting. Mathematical Programming 201(1-2):231–262. <https://arxiv.org/pdf/2108.02897.pdf>`_

    `[4] E. Ryu (2020). Uniqueness of drs as the 2 operator resolvent-splitting and
    impossibility of 3 operator resolvent-splitting. Mathematical Programming 182(1-
    2):233–273. <https://arxiv.org/pdf/1802.07534>`_

    `[5] J. Eckstein, D. Bertsekas (1992). On the Douglas—Rachford splitting method and the proximal point algorithm for maximal monotone operators. Mathematical Programming 55:293–318. <https://link.springer.com/content/pdf/10.1007/BF01581204.pdf>`_


    Args:
        L (ndarray): n x n numpy array of resolvent multipliers for step 1.
        W (ndarray): n x n numpy array of resolvent multipliers for step 2.
        l_values (array): n smoothness parameters for the functions.
        mu_values (array): n strong convexity parameters for the functions.
        alpha (float): resolvent scaling parameter.
        gamma (float): step size parameter.
        wrapper (str): the name of the wrapper to be used.
        solver (str): the name of the solver the wrapper should use.
        verbose (int): level of information details to print.

                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + solver details.

    Returns:
        pepit_tau (float): worst-case value

    Example:
        >>> pepit_tau = wc_frugal_resolvent_splitting(
                            L=array([[0,0],[2,0]]), 
                            W=array([[1,-1],[-1,1]]),
                            l_values=[2, 1000],
                            mu_values=[1, 0],
                            alpha=1,
                            gamma=0.5,
                            wrapper="cvxpy", 
                            verbose=1)
        (PEPit) Setting up the problem: size of the Gram matrix: 6x6
        (PEPit) Setting up the problem: performance measure is the minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
                                Function 1 : Adding 2 scalar constraint(s) ...
                                Function 1 : 2 scalar constraint(s) added
                                Function 2 : Adding 2 scalar constraint(s) ...
                                Function 2 : 2 scalar constraint(s) added
        (PEPit) Setting up the problem: additional constraints for 0 function(s)
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: SCS); optimal value: 0.6941671049622371
        (PEPit) Postprocessing: solver's output is not entirely feasible (smallest eigenvalue of the Gram matrix is: -4.99e-09 < 0).
        Small deviation from 0 may simply be due to numerical error. Big ones should be deeply investigated.
        In any case, from now the provided values of parameters are based on the projection of the Gram matrix onto the cone of symmetric semi-definite matrix.
        (PEPit) Primal feasibility check:
                        The solver found a Gram matrix that is positive semi-definite up to an error of 4.987058462382981e-09
                        All the primal scalar constraints are verified up to an error of 4.26979977485864e-08
        (PEPit) Dual feasibility check:
                        The solver found a residual matrix that is positive semi-definite up to an error of 7.726891430196974e-15
                        All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 4.912845922439013e-08
        (PEPit) Final upper bound (dual): 0.6941669727111097 and lower bound (primal example): 0.6941671049622371 
        (PEPit) Duality gap: absolute: -1.3225112738268763e-07 and relative: -1.9051771027075984e-07
        *** Example file: worst-case performance of parameterized frugal resolvent splitting ***
                PEPit guarantee:         ||v_(t+1)^0 - v_(t+1)^1||^2 <= 0.694167 ||v_(t)^0 - v_(t)^1||^2
        >>> comparison()
        ---------------------------------------------------------------------------
        Contraction factors of different designs with fixed step size, optimal step size, and optimal W matrix
        when n=4 and each function has l=2, mu=1
        ---------------------------------------------------------------------------
                                Contraction Factors
        Design   constant step size 0.5  optimized step size     optimized W matrix
        ---------------------------------------------------------------------------
        MT       0.858                   0.758                   0.750
        Full     0.423                   0.101                   0.077
        Block    0.444                   0.088                   0.059
        ---------------------------------------------------------------------------
        Optimized step sizes and W matrix found using the dual of the PEP as in [1]. 
        MT is the Malitsky-Tam algorithm from [3].
        Full is the fully connected algorithm in which L has no zero entries in the lower triangle.
        Block is the 2-Block design from [1].

    """
    # Store the number of operators
    n = W.shape[0]

    # Validate input sizes are consistent
    assert L.shape == W.shape == (n,n)
    assert len(l_values) == len(mu_values) == n

    # Instantiate PEP
    problem = PEP()

    # Declare operators (works for LipschitzStronglyMonotoneOperator as well)
    operators = [problem.declare_function(SmoothStronglyConvexFunction, L=l, mu=mu) for l, mu in zip(l_values, mu_values)]

    # Define the starting points v0 and v1 for the n-1 independent values
    v0 = [problem.set_initial_point() for _ in range(n-1)]
    v1 = [problem.set_initial_point() for _ in range(n-1)]

    # Add the n-th value to v so that it is implicitly constrained to equal 0
    v0.append(-sum(v0, start=null_point))
    v1.append(-sum(v1, start=null_point))
    
    # Set the initial constraint that is the distance between v0 and v1
    problem.set_initial_condition(sum((v0[i] - v1[i]) ** 2 for i in range(n)) <= 1)

    # Constraint on the lifted starting points so each sums to 0
    # v0constraint = Constraint(expression=sum((v0[i] for i in range(n)), start=null_point)**2, equality_or_inequality="equality")
    # v1constraint = Constraint(expression=sum((v1[i] for i in range(n)), start=null_point)**2, equality_or_inequality="equality")
    # problem.set_initial_condition(v0constraint)
    # problem.set_initial_condition(v1constraint)

    # Define the step for each element of the lifted vector    
    def resolvent(i, x, v, L, alpha):
        Lx = sum((L[i, j]*x[j] for j in range(i)), start=null_point)
        x, _, _ = proximal_step(v[i] + Lx, operators[i], alpha)
        return x

    x0 = []
    x1 = []
    for i in range(n):
        x0.append(resolvent(i, x0, v0, L, alpha))
        x1.append(resolvent(i, x1, v1, L, alpha))

    # z is the updated version of v
    z0 = []
    z1 = []
    for i in range(n):
        z0.append(v0[i] - gamma*W[i,:]@x0)
        z1.append(v1[i] - gamma*W[i,:]@x1)
    
    # Set the performance metric to the distance between z0 and z1
    problem.set_performance_metric(sum((z0[i] - z1[i]) ** 2 for i in range(n)))
    
    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of parameterized frugal resolvent splitting ***')
        print('\tPEPit guarantee:\t ||v_(t+1)^0 - v_(t+1)^1||^2 <= {:.6} ||v_(t)^0 - v_(t)^1||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau

def comparison():
    # Comparison for 4 operators for Malitsky-Tam, Fully Connected, and 2-Block designs
    # with and without optimized step sizes and W matrices
    l_values = [2, 2, 2, 2]
    mu_values = [1, 1, 1, 1]

    # Malitsky-Tam [3]
    L_MT = array([[0,0,0,0],
                  [1,0,0,0],
                  [0,1,0,0],
                  [1,0,1,0]])
    W_MT = array([[1,-1,0,0],
                  [-1,2,-1,0],
                  [0,-1,2,-1],
                  [0,0,-1,1]])
    W_MT_opt = array([
        [ 1.071, -1.071, -0.   , -0.   ],
        [-1.071,  2.071, -1.   , -0.   ],
        [-0.   , -1.   ,  2.5  , -1.5  ],
        [-0.   , -0.   , -1.5  ,  1.5  ]])

    # Fully Connected
    L_full = array([[0,0,0,0],
                    [2/3,0,0,0],
                    [2/3,2/3,0,0],
                    [2/3,2/3,2/3,0]])
    W_full = array([[2, -2/3, -2/3, -2/3],
                    [-2/3, 2, -2/3, -2/3],
                    [-2/3, -2/3, 2, -2/3],
                    [-2/3, -2/3, -2/3, 2]])
    W_full_opt = array([
        [ 2.226, -0.577, -0.714, -0.936],
        [-0.577,  1.94 , -0.604, -0.759],
        [-0.714, -0.604,  1.976, -0.658],
        [-0.936, -0.759, -0.658,  2.353]])

    # 2-Block [1]
    L_block = array([[0,0,0,0],
                     [0,0,0,0],
                     [1,1,0,0],
                     [1,1,0,0]])
    W_block = array([[ 2.,  0., -1., -1.],
                     [ 0.,  2., -1., -1.],
                     [-1., -1.,  2.,  0.],
                     [-1., -1.,  0.,  2.]])
    W_block_opt = array([
        [ 2.2, -0.2, -1. , -1. ],
        [-0.2,  2.2, -1. , -1. ],
        [-1. , -1. ,  2.2, -0.2],
        [-1. , -1. , -0.2,  2.2]])
    
    print('---------------------------------------------------------------------------')
    print('Contraction factors of different designs with fixed step size, optimal step size, and optimal W matrix\n when n=4 and each function has l=2, mu=1')
    print('---------------------------------------------------------------------------')
    print('\t\t\tContraction Factors\t\t\t')
    print('Design\t', 'constant step size 0.5\t', 'optimized step size\t', 'optimized W matrix\t')
    print('---------------------------------------------------------------------------')

    # Malitsky-Tam [3]
    tau_MT = wc_frugal_resolvent_splitting(L_MT, W_MT, l_values, mu_values, gamma=0.5, verbose=-1)
    tau_MT_opt_step = wc_frugal_resolvent_splitting(L_MT, W_MT, l_values, mu_values, gamma=1.09, verbose=-1)
    tau_MT_opt_W = wc_frugal_resolvent_splitting(L_MT, W_MT_opt, l_values, mu_values, gamma=1, verbose=-1)
    # string format for the output of the function rounding to 3 decimal places with tab separation
    print('MT \t {:.3f} \t\t\t {:.3f} \t\t\t {:.3f}'.format(tau_MT, tau_MT_opt_step, tau_MT_opt_W))

    # Fully Connected
    tau_f = wc_frugal_resolvent_splitting(L_full, W_full, l_values, mu_values, gamma=0.5, verbose=-1)
    tau_f_opt_step = wc_frugal_resolvent_splitting(L_full, W_full, l_values, mu_values, gamma=1.09, verbose=-1)
    tau_f_opt_W = wc_frugal_resolvent_splitting(L_full, W_full_opt, l_values, mu_values, gamma=1, verbose=-1)
    print('Full \t {:.3f} \t\t\t {:.3f} \t\t\t {:.3f}'.format(tau_f, tau_f_opt_step, tau_f_opt_W))

    # 2-Block [1]
    tau_b = wc_frugal_resolvent_splitting(L_block, W_block, l_values, mu_values, gamma=0.5, verbose=-1)
    tau_b_opt_step = wc_frugal_resolvent_splitting(L_block, W_block, l_values, mu_values, gamma=1.09, verbose=-1)
    tau_b_opt_W = wc_frugal_resolvent_splitting(L_block, W_block_opt, l_values, mu_values, gamma=1, verbose=-1)
    print('Block \t {:.3f} \t\t\t {:.3f} \t\t\t {:.3f}'.format(tau_b, tau_b_opt_step, tau_b_opt_W))
    print('---------------------------------------------------------------------------')
    print('''    Optimized step sizes and W matrix found using the dual of the PEP as in [1]. 
    MT is the Malitsky-Tam algorithm from [3].
    Full is the fully connected algorithm in which L has no zero entries in the lower triangle.
    Block is the 2-Block design from [1].''')
    return 0

if __name__ == "__main__":
    # Douglas-Rachford [5]
    print("\n1. Basic test using Douglas-Rachford matrices\n")
    pepit_tau = wc_frugal_resolvent_splitting(
                            L=array([[0,0],[2,0]]), 
                            W=array([[1,-1],[-1,1]]),
                            l_values=[2, 1000],
                            mu_values=[1, 0],
                            alpha=1,
                            gamma=0.5,
                            wrapper="cvxpy", 
                            verbose=1)

    # Comparison for 4 operators for Malitsky-Tam, Fully Connected, and 2-Block designs
    # with and without optimized step sizes and W matrices from [1]
    print('\n2. Comparison of various algorithm designs')
    comparison()

    
