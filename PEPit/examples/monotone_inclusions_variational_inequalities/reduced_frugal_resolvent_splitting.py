from PEPit import PEP, null_point
from PEPit.primitive_steps import proximal_step
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.operators import MonotoneOperator
from numpy import array

def wc_reduced_frugal_resolvent_splitting(L, M, problem, alpha=1, gamma=0.5, wrapper="cvxpy", solver=None, verbose=1):
    """
    Consider the the problem

    .. math:: \\mathrm{Find}\\, x:\\, 0 \\in \\sum_{i=1}^{n} A_i(x),

    where :math:`A_i` is a maximal monotone operator for all :math:`i \\leq n`. 
    We denote by :math:`J_{\\alpha A_i}` the resolvent of :math:`\\alpha A_i`. 
    We denote the lifted vector operator :math:`\\mathbf{A}` as :math:`\\mathbf{A} = [A_1, \\dots, A_n]`, 
    and use lifted :math:`\\mathbf{x}^T = [x_1, \\dots, x_n]` and :math:`\\mathbf{w}^T = [w_1, \\dots, w_d]`. 
    We denote by :math:`L \\in \\mathbb{R}^{n \\times n}` and :math:`M \\in \\mathbb{R}^{n-1 \\times n}` the reduced algorithm design matrices. 

    This code computes a worst-case guarantee for any reduced frugal resolvent splitting with design matrices :math:`L, M`. 
    As shown in [1] and [2], this can include the Malitsky-Tam [3], Ryu Three Operator Splitting [4], Douglas-Rachford [5], and the reduced version of the block splitting algorithms in [1].
    That is, given two lifted initial points :math:`\\mathbf{w}^{(0)}_t` and :math:`\\mathbf{w}^{(1)}_t`
    this code computes the smallest possible :math:`\\tau(L, M, \\alpha, \\gamma)`
    (a.k.a. "contraction factor") such that the guarantee

    .. math:: \\|\\mathbf{w}^{(0)}_{t+1} - \\mathbf{w}^{(1)}_{t+1}\\|^2 \\leqslant \\tau(L, M, \\alpha, \\gamma) \\|\\mathbf{w}^{(0)}_{t} - \\mathbf{w}^{(1)}_{t}\\|^2,

    is valid, where :math:`\\mathbf{w}^{(0)}_{t+1}` and :math:`\\mathbf{w}^{(1)}_{t+1}` are obtained after one iteration of the reduced frugal resolvent splitting from respectively :math:`\\mathbf{w}^{(0)}_{t}` and :math:`\\mathbf{w}^{(1)}_{t}`.

    In short, for given values of :math:`L`, :math:`M`, :math:`\\alpha` and :math:`\\gamma`, the contraction factor :math:`\\tau(L, M, \\alpha, \\gamma)` is computed as the worst-case value of
    :math:`\\|\\mathbf{w}^{(0)}_{t+1} - \\mathbf{w}^{(1)}_{t+1}\\|^2` when :math:`\\|\\mathbf{w}^{(0)}_{t} - \\mathbf{w}^{(1)}_{t}\\|^2 \\leqslant 1`.

    **Algorithm**: One iteration of the reduced parameterized frugal resolvent splitting is described as follows:

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\mathbf{x}_{t+1} & = & J_{\\alpha \\mathbf{A}} (\\mathbf{L} \\mathbf{x}_{t+1} - \\mathbf{M}^T \\mathbf{w}_t),\\\\
                \\mathbf{w}_{t+1} & = & \\mathbf{w}_t + \\gamma \\mathbf{M} \\mathbf{x}_{t+1}.
            \\end{eqnarray}

    :math:`L` is assumed to be strictly lower triangular to make each resolvent :math:`i` of the algorithm rely only on :math:`\\mathbf{w}_t` and the results of the previous resolvents in that iteration (and not subsequent resolvents). 
    
    :math:`M` is assumed to have nullspace equal to the span of the ones vector, so that :math:`M 1 = 0`.

    **References**:

    `[1] R. Bassett, P. Barkley (2024). 
    Optimal Design of Resolvent Splitting Algorithms. arxiv:2407.16159.
    <https://arxiv.org/pdf/2407.16159.pdf>`_

    `[2] M. Tam (2023). Frugal and decentralised resolvent splittings defined by nonexpansive operators. Optimization Letters pp 1–19. <https://arxiv.org/pdf/2211.04594.pdf>`_

    `[3] Y. Malitsky, M. Tam (2023). Resolvent splitting for sums of monotone operators
    with minimal lifting. Mathematical Programming 201(1-2):231–262. <https://arxiv.org/pdf/2108.02897.pdf>`_

    `[4] E. Ryu (2020). Uniqueness of DRS as the 2 operator resolvent-splitting and impossibility of 3 operator resolvent-splitting. Mathematical Programming 182(1-
    2):233–273. <https://arxiv.org/pdf/1802.07534>`_

    `[5] J. Eckstein, D. Bertsekas (1992). On the Douglas—Rachford splitting method and the proximal point algorithm for maximal monotone operators. Mathematical Programming 55:293–318. <https://link.springer.com/content/pdf/10.1007/BF01581204.pdf>`_


    Args:
        L (ndarray): n x n numpy array of resolvent multipliers for step 1.
        M (ndarray): (n-1) x n numpy array of multipliers for steps 1 and 2.
        problem (PEP): PEP problem with exactly n maximal monotone operators.
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
        >>> problem = PEP()
        >>> problem.declare_function(SmoothStronglyConvexFunction, L=2, mu=1)
        >>> problem.declare_function(MonotoneOperator)
        >>> pepit_tau = wc_reduced_frugal_resolvent_splitting(
                            L=array([[0,0],[2,0]]), 
                            M=array([[1,-1]]),
                            problem=problem)
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
        (PEPit) Solver status: optimal (wrapper:cvxpy, solver: SCS); optimal value: 0.694445372649345
        (PEPit) Postprocessing: solver's output is not entirely feasible (smallest eigenvalue of the Gram matrix is: -6.83e-08 < 0).
        Small deviation from 0 may simply be due to numerical error. Big ones should be deeply investigated.
        In any case, from now the provided values of parameters are based on the projection of the Gram matrix onto the cone of symmetric semi-definite matrix.
        (PEPit) Primal feasibility check:
                The solver found a Gram matrix that is positive semi-definite up to an error of 6.830277482116174e-08
                All the primal scalar constraints are verified up to an error of 5.489757276544438e-07
        (PEPit) Dual feasibility check:
                The solver found a residual matrix that is positive semi-definite up to an error of 8.242470267419779e-16
                All the dual scalar values associated with inequality constraints are nonnegative
        (PEPit) The worst-case guarantee proof is perfectly reconstituted up to an error of 1.710626393869319e-06
        (PEPit) Final upper bound (dual): 0.6944445325135953 and lower bound (primal example): 0.694445372649345 
        (PEPit) Duality gap: absolute: -8.401357497467288e-07 and relative: -1.209793862606597e-06
        *** Example file: worst-case performance of reduced parameterized frugal resolvent splitting ***
            PEPit guarantee:	 ||w_(t+1)^0 - w_(t+1)^1||^2 <= 0.694445 ||w_(t)^0 - w_(t)^1||^2
        >>> comparison()
        ----------------------------------------------------------------
        Contraction factors of different designs with constant step size 0.5 and optimized step size
        with 4 smooth strongly convex functions having l=2, mu=1.
        ----------------------------------------------------------------
                 Contraction factor with         Contraction factor with
        Design   constant step size 0.5          optimized step size
        ----------------------------------------------------------------
        MT       0.837                           0.603
        Full     0.423                           0.101
        Block    0.445                           0.067
        ----------------------------------------------------------------
        Optimized step sizes found using the dual of the PEP as in [1].
        MT is the Malitsky-Tam algorithm from [3].
        Full is the fully connected algorithm in which L has no zero entries in the lower triangle.
        Block is the 2-Block design from [1].   

    """
    # Store the number of operators
    n = L.shape[0]
    d = M.shape[0]

    # Get problem operators
    operators = problem.list_of_functions

    # Validate input sizes are consistent
    assert L.shape == (n,n)
    assert M.shape[1] == len(operators) == n

    # Then define the starting points v0 and v1
    n = L.shape[0]
    d = M.shape[0]
    w0 = [problem.set_initial_point() for _ in range(d)]
    w1 = [problem.set_initial_point() for _ in range(d)]
    
    # Set the initial constraint that is the distance between v0 and v1
    problem.set_initial_condition(sum((w0[i] - w1[i]) ** 2 for i in range(d)) <= 1)

    # Define the step for each element of the lifted vector    
    def resolvent(i, x, w, L, M, alpha):
        Lx = sum((L[i, j]*x[j] for j in range(i)), start=null_point)
        x, _, _ = proximal_step(-M.T[i,:]@w + Lx, operators[i], alpha)
        return x

    x0 = []
    x1 = []
    for i in range(n):
        x0.append(resolvent(i, x0, w0, L, M, alpha))
        x1.append(resolvent(i, x1, w1, L, M, alpha))

    # z is the updated version of w
    z0 = []
    z1 = []
    for i in range(d):
        z0.append(w0[i] + gamma*M[i,:]@x0)
        z1.append(w1[i] + gamma*M[i,:]@x1)
    
    # Set the performance metric to the distance between z0 and z1
    problem.set_performance_metric(sum((z0[i] - z1[i]) ** 2 for i in range(d)))
    
    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of reduced parameterized frugal resolvent splitting ***')
        print('\tPEPit guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= {:.6} ||w_(t)^0 - w_(t)^1||^2'.format(pepit_tau))

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
    M_MT = array([
        [-1.,  1.,  0.,  0.],
        [ 0., -1.,  1.,  0.],
        [ 0.,  0., -1.,  1.]])

    # Fully Connected
    L_full = array([[0,  0,  0,  0],
                    [2/3,0,  0,  0],
                    [2/3,2/3,0,  0],
                    [2/3,2/3,2/3,0]])
    M_full = array([
        [-1.155,  1.155,  0.   ,  0.   ],
        [-0.667, -0.667,  1.333,  0.   ],
        [-0.471, -0.471, -0.471,  1.414]])

    # 2-Block [1]
    L_block = array([[0,0,0,0],
                     [0,0,0,0],
                     [1,1,0,0],
                     [1,1,0,0]])
    M_block = array([
        [-1.   ,  1.   ,  0.   ,  0.   ],
        [-0.707, -0.707,  1.414,  0.   ],
        [-0.707, -0.707,  0.   ,  1.414]])

    print('----------------------------------------------------------------')
    print('Contraction factors of different designs with constant step size 0.5 and optimized step size\n with 4 smooth strongly convex functions having l=2, mu=1.')
    print('----------------------------------------------------------------')
    print('\t', 'Contraction factor with\t', 'Contraction factor with')
    print('Design\t', 'constant step size 0.5\t\t', 'optimized step size')
    print('----------------------------------------------------------------')

    # Malitsky-Tam [3]
    taus = []
    for gamma in [0.5, 1.405]:
        problem = PEP()
        for l, mu in zip(l_values, mu_values):
            problem.declare_function(SmoothStronglyConvexFunction, L=l, mu=mu)
        taus.append(wc_reduced_frugal_resolvent_splitting(L_MT, M_MT, problem, gamma=gamma, verbose=-1))

    print('MT \t {:.3f} \t\t\t\t {:.3f}'.format(*taus))

    # Fully Connected
    taus = []
    for gamma in [0.5, 1.09]:
        problem = PEP()
        for l, mu in zip(l_values, mu_values):
            problem.declare_function(SmoothStronglyConvexFunction, L=l, mu=mu)
        taus.append(wc_reduced_frugal_resolvent_splitting(L_full, M_full, problem, gamma=gamma, verbose=-1))
        
    print('Full \t {:.3f} \t\t\t\t {:.3f}'.format(*taus))

    # 2-Block [1]
    taus = []
    for gamma in [0.5, 1.12]:
        problem = PEP()
        for l, mu in zip(l_values, mu_values):
            problem.declare_function(SmoothStronglyConvexFunction, L=l, mu=mu)
        taus.append(wc_reduced_frugal_resolvent_splitting(L_block, M_block, problem, gamma=gamma, verbose=-1))
        
    print('Block \t {:.3f} \t\t\t\t {:.3f}'.format(*taus))
    print('----------------------------------------------------------------')
    print('''        Optimized step sizes found using the dual of the PEP as in [1].
        MT is the Malitsky-Tam algorithm from [3].
        Full is the fully connected algorithm in which L has no zero entries in the lower triangle.
        Block is the 2-Block design from [1].''')
    return 0

if __name__ == "__main__":
    # Douglas-Rachford [5]
    print("\n1. Basic test using Douglas-Rachford matrices\n")
    
    # Instantiate PEP
    problem = PEP()

    # Declare operators
    problem.declare_function(SmoothStronglyConvexFunction, L=2, mu=1)
    problem.declare_function(MonotoneOperator)
    pepit_tau = wc_reduced_frugal_resolvent_splitting(
                            L=array([[0,0],[2,0]]), 
                            M=array([[1,-1]]),
                            problem=problem)

    # Comparison for 4 operators for Malitsky-Tam, Fully Connected, and 2-Block designs
    # with and without optimized step sizes from [1]
    print('\n2. Comparison of various algorithm designs')
    comparison()

    
