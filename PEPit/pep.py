import numpy as np
import cvxpy as cp

from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.constraint import Constraint
from PEPit.function import Function


class PEP(object):
    """
    The class :class:`PEP` is the main class of this framework.
    A :class:`PEP` object encodes a complete performance estimation problem.
    It stores the following information.

    Attributes:
        list_of_functions (list): list of leaf :class:`Function` objects that are defined through the pipeline.
        list_of_points (list): list of :class:`Point` objects that are defined out of the scope of a :class:`Function`.
                               Typically the initial :class:`Point`.
        list_of_constraints (list): list of :class:`Constraint` objects that are defined
                                   out of the scope of a :class:`Function`.
                                   Typically the initial :class:`Constraint`.
        list_of_performance_metrics (list): list of :class:`Expression` objects.
                                            The pep maximizes the minimum of all performance metrics.
        counter (int): counts the number of :class:`PEP` objects.
                       Ideally, only one is defined at a time.

    """
    # Class counter.
    # It counts the number of PEP defined instantiated.
    counter = 0

    def __init__(self):
        """
        A :class:`PEP` object can be instantiated without any argument

        Example:
            >>> pep = PEP()

        """

        # Set all counters to 0 to recreate points, expressions and functions from scratch at the beginning of each PEP.
        Point.counter = 0
        Expression.counter = 0
        Function.counter = 0

        # Update the class counter
        self.counter = PEP.counter
        PEP.counter += 1

        # Initialize list of functions,
        # points and constraints that are independent of the functions,
        # as well as the list of matrices that must be constraint to be positive symmetric definite
        # and the list of performance metrics.
        # The PEP will maximize the minimum of the latest.
        self.list_of_functions = list()
        self.list_of_points = list()
        self.list_of_constraints = list()
        self.list_of_performance_metrics = list()
        self.list_of_psd = list()

    def declare_function(self, function_class, **kwargs):
        """
        Instantiate a leaf :class:`Function` and store it in the attribute `list_of_functions`.

        Args:
            function_class (class): a subclass of :class:`Function` that overwrite the `add_class_constraints` method.
            kwargs (dict): dictionary of parameters that characterize the function class.
                           Can also contains the boolean `reuse_gradient`,
                           that enforces using only one subgradient per point.

        Returns:
            f (Function): the newly created function.

        """

        # Create the function
        f = function_class(is_leaf=True, decomposition_dict=None, **kwargs)

        # Store it in list_of_functions
        self.list_of_functions.append(f)

        # Return it
        return f

    def set_initial_point(self):
        """
        Create a new leaf :class:`Point` and store it in the attribute `list_of_points`.

        Returns:
            x (Point): the newly created :class:`Point`.

        """

        # Create a new point from scratch
        x = Point(is_leaf=True, decomposition_dict=None)

        # Store it in list_of_points
        self.list_of_points.append(x)

        # Return it
        return x

    def set_initial_condition(self, condition):
        """
        Store a new :class:`Constraint` to the list of constraints of this :class:`PEP`.
        Typically an condition of the form :math:`\\|x_0 - x_\\star\\|^2 \\leq 1`.

        Args:
            condition (Constraint): typically resulting from a comparison of 2 :class:`Expression` objects.

        Raises:
            AssertionError: if provided `constraint` is not a :class:`Constraint` object.

        """

        # Call add_constraint method
        self.add_constraint(constraint=condition)

    def add_constraint(self, constraint):
        """
        Store a new :class:`Constraint` to the list of constraints of this :class:`PEP`.

        Args:
            constraint (Constraint): typically resulting from a comparison of 2 :class:`Expression` objects.

        Raises:
            AssertionError: if provided `constraint` is not a :class:`Constraint` object.

        """

        # Verify constraint is an actual Constraint object
        assert isinstance(constraint, Constraint)

        # Add constraint to the list of self's constraints
        self.list_of_constraints.append(constraint)

    def add_psd_matrix(self, matrix):
        """
        Store a new matrix of :class:`Expression`s that we enforce to be positive semidefinite.

        Args:
            matrix (Iterable of Iterable of Expression): a square matrix of :class:`Expression`.

        Raises:
            AssertionError: if provided matrix is not a square matrix.
            TypeError: if provided matrix does not contain only Expressions.

        """

        # Change iterable into ndarray
        matrix = np.array(matrix)

        # Verify constraint is a square matrix
        size = matrix.shape[0]
        assert matrix.shape == (size, size)

        # Transform scalars into Expressions
        for i in range(size):
            for j in range(size):
                # The entry must be an Expression ...
                if isinstance(matrix[i, j], Expression):
                    pass
                # ... or a python scalar. If so, store it as an Expression
                elif isinstance(matrix[i, j], int) or isinstance(matrix[i, j], float):
                    matrix[i, j] = Expression(is_leaf=False, decomposition_dict={1: matrix[i, j]})
                # Raise an Exception in any other scenario
                else:
                    raise TypeError("PSD matrices contains only Expressions and / or scalar values!"
                                    "Got {}".format(type(matrix[i, j])))

        # Add constraint to the list of self's constraints
        self.list_of_psd.append(matrix)

    def set_performance_metric(self, expression):
        """
        Store a performance metric in the attribute `list_of_performance_metrics`.
        The objective of the PEP (which is maximized) is the minimum of the elements of `list_of_performance_metrics`.

        Args:
            expression (Expression): a new performance metric.

        """

        # Store performance metric in the appropriate list
        self.list_of_performance_metrics.append(expression)

    @staticmethod
    def _expression_to_cvxpy(expression, F, G):
        """
        Create a cvxpy compatible expression from an :class:`Expression`.

        Args:
            expression (Expression): any expression.
            F (cvxpy Variable): a vector representing the function values.
            G (cvxpy Variable): a matrix representing the Gram matrix of all leaf :class:`Point` objects.

        Returns:
            cvxpy_variable (cvxpy Variable): The expression in terms of F and G.

        """
        cvxpy_variable = 0
        Fweights = np.zeros((Expression.counter,))
        Gweights = np.zeros((Point.counter, Point.counter))

        # If simple function value, then simply return the right coordinate in F
        if expression.get_is_leaf():
            Fweights[expression.counter] += 1
        # If composite, combine all the cvxpy expression found from leaf expressions
        else:
            for key, weight in expression.decomposition_dict.items():
                # Function values are stored in F
                if type(key) == Expression:
                    assert key.get_is_leaf()
                    Fweights[key.counter] += weight
                # Inner products are stored in G
                elif type(key) == tuple:
                    point1, point2 = key
                    assert point1.get_is_leaf()
                    assert point2.get_is_leaf()
                    Gweights[point1.counter, point2.counter] += weight
                # Constants are simply constants
                elif key == 1:
                    cvxpy_variable += weight
                # Others don't exist and raise an Exception
                else:
                    raise TypeError("Expressions are made of function values, inner products and constants only!")

        cvxpy_variable += F @ Fweights + cp.sum(cp.multiply(G, Gweights))

        # Return the input expression in a cvxpy variable
        return cvxpy_variable

    def solve(self, verbose=1, return_full_cvxpy_problem=False,
              dimension_reduction_heuristic=None, eig_regularization=1e-3, tol_dimension_reduction=1e-5,
              **kwargs):
        """
        Transform the :class:`PEP` under the SDP form, and solve it.

        Args:
            verbose (int): Level of information details to print (Override the CVXPY solver verbose parameter).
                           0: No verbose at all
                           1: PEPit information is printed but not CVXPY's
                           2: Both PEPit and CVXPY details are printed
            return_full_cvxpy_problem (bool): If True, return the cvxpy Problem object.
                                              If False, return the worst case value only.
                                              Set to False by default.
            dimension_reduction_heuristic (str, optional): An heuristic to reduce the dimension of the solution
                                                           (rank of the Gram matrix).
                                                           Available heuristics are:
                                                            - "trace": minimize :math:`Tr(G)`
                                                            - "logdet{an integer n}": minimize
                                                              :math:`\\log\\left(\\mathrm{Det}(G)\\right)`
                                                              using n iterations of local approximation problems.
                                                           Set to None to deactivate it (default value).
            eig_regularization (float, optional): The regularization we use to make
                                                  :math:`G + \\mathrm{eig_regularization}I_d \succ 0`.
                                                  (only used when "dimension_reduction_heuristic" is not None)
                                                  The default value is 1e-5.
            tol_dimension_reduction (float, optional): The error tolerance in the heuristic minimization problem.
                                                       Precisely, the second problem minimizes "optimal_value - tol"
                                                       (only used when "dimension_reduction_heuristic" is not None)
                                                       The default value is 1e-5.
            kwargs (keywords, optional): Additional CVXPY solver specific arguments.

        Returns:
            float or cp.Problem: Value of the performance metric of cp.Problem object corresponding to the SDP.
                                 The value only is returned by default.

        """
        # Set CVXPY verbose to True if verbose mode is at least 2
        kwargs["verbose"] = verbose >= 2

        # Create all class constraints
        for function in self.list_of_functions:
            function.add_class_constraints()

        # Define the cvxpy variables
        objective = cp.Variable((1,))
        F = cp.Variable((Expression.counter,))
        G = cp.Variable((Point.counter, Point.counter), PSD=True)
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' size of the main PSD matrix: {}x{}'.format(Point.counter, Point.counter))

        # Express the constraints from F, G and objective
        constraints_list = list()

        # Defining performance metrics
        # Note maximizing the minimum of all the performance metrics
        # is equivalent to maximize objective which is constraint to be smaller than all the performance metrics.
        for performance_metric in self.list_of_performance_metrics:
            assert isinstance(performance_metric, Expression)
            constraints_list.append(objective <= self._expression_to_cvxpy(performance_metric, F, G))
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' performance measure is minimum of {} element(s)'.format(len(self.list_of_performance_metrics)))

        # Defining initial conditions and general constraints
        for condition in self.list_of_constraints:
            assert isinstance(condition, Constraint)
            if condition.equality_or_inequality == 'inequality':
                constraints_list.append(self._expression_to_cvxpy(condition.expression, F, G) <= 0)
            elif condition.equality_or_inequality == 'equality':
                constraints_list.append(self._expression_to_cvxpy(condition.expression, F, G) == 0)
            else:
                raise ValueError('The attribute \'equality_or_inequality\' of a constraint object'
                                 ' must either be \'equality\' or \'inequality\'.'
                                 'Got {}'.format(condition.equality_or_inequality))
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' initial conditions and general constraints ({} constraint(s) added)'.format(len(self.list_of_constraints)))

        # Defining class constraints
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' interpolation conditions for {} function(s)'.format(len(self.list_of_functions)))
        function_counter = 0
        for function in self.list_of_functions:
            function_counter += 1
            for constraint in function.list_of_constraints:
                assert isinstance(constraint, Constraint)
                if constraint.equality_or_inequality == 'inequality':
                    constraints_list.append(self._expression_to_cvxpy(constraint.expression, F, G) <= 0)
                elif constraint.equality_or_inequality == 'equality':
                    constraints_list.append(self._expression_to_cvxpy(constraint.expression, F, G) == 0)
                else:
                    raise ValueError('The attribute \'equality_or_inequality\' of a constraint object'
                                     ' must either be \'equality\' or \'inequality\'.'
                                     'Got {}'.format(constraint.equality_or_inequality))
            if verbose:
                print('\t\t function', function_counter, ':', len(function.list_of_constraints), 'constraint(s) added')

        # Defining lmi constraints
        if verbose:
            print('(PEPit) Setting up the problem: {} lmi constraint(s) added'.format(len(self.list_of_psd)))
        for psd_counter, psd_matrix in enumerate(self.list_of_psd):
            M = cp.Variable(psd_matrix.shape, PSD=True)
            for i in range(psd_matrix.shape[0]):
                for j in range(psd_matrix.shape[1]):
                    constraints_list.append(M[i, j] == self._expression_to_cvxpy(psd_matrix[i, j], F, G))
            if verbose:
                print('\t\t Size of PSD matrix {}: {}x{}'.format(psd_counter+1, *psd_matrix.shape))

        # Create the cvxpy problem
        if verbose:
            print('(PEPit) Compiling SDP')
        prob = cp.Problem(objective=cp.Maximize(objective), constraints=constraints_list)

        # Solve it
        if verbose:
            print('(PEPit) Calling SDP solver')
        prob.solve(**kwargs)
        if verbose:
            print('(PEPit) Solver status: {} (solver: {}); optimal value: {}'.format(prob.status,
                                                                                     prob.solver_stats.solver_name,
                                                                                     prob.value))

        # Store the obtained value
        wc_value = prob.value

        # Raise explicit error when wc_value in infinite
        if wc_value == np.inf:
            raise UserWarning("PEPit didn't find any nontrivial worst-case guarantee. "
                              "It seems that the optimal value of your problem is unbounded.")

        # Perform a dimension reduction if required
        if dimension_reduction_heuristic:

            # Print the estimated dimension before dimension reduction
            nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(G.value)
            if verbose:
                print('(PEPit) Postprocessing: {} eigenvalue(s) > {} before dimension reduction'.format(nb_eigenvalues,
                                                                                                        eig_threshold))
                print('(PEPit) Calling SDP solver')

            # Add the constraint that the objective stay close to its actual value
            constraints_list.append(objective >= wc_value - tol_dimension_reduction)

            # Translate the heuristic into cvxpy objective and solve the associated problem
            if dimension_reduction_heuristic == "trace":
                heuristic = cp.trace(G)
                prob = cp.Problem(objective=cp.Minimize(heuristic), constraints=constraints_list)
                prob.solve(**kwargs)
                nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(G.value)
            elif dimension_reduction_heuristic.startswith("logdet"):
                niter = int(dimension_reduction_heuristic[6:])
                for i in range(1, 1+niter):
                    W = np.linalg.inv(corrected_G_value + eig_regularization * np.eye(Point.counter))
                    heuristic = cp.sum(cp.multiply(G, W))
                    prob = cp.Problem(objective=cp.Minimize(heuristic), constraints=constraints_list)
                    prob.solve(**kwargs)

                    # Print the estimated dimension after i dimension reduction steps
                    nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(G.value)
                    if verbose:
                        print('(PEPit) Solver status: {} (solver: {});'
                              ' objective value: {}'.format(prob.status,
                                                            prob.solver_stats.solver_name,
                                                            wc_value))
                        print('(PEPit) Postprocessing: {} eigenvalue(s) > {} after {} dimension reduction step(s)'.format(
                            nb_eigenvalues, eig_threshold, i))

            else:
                raise ValueError("The argument \'dimension_reduction_heuristic\' must be \'trace\'"
                                 "or \`logdet\` followed by an interger."
                                 "Got {}".format(dimension_reduction_heuristic))

            # Print the estimated dimension after dimension reduction
            if verbose:
                print('(PEPit) Solver status: {} (solver: {});'
                      ' objective value: {}'.format(prob.status,
                                                    prob.solver_stats.solver_name,
                                                    wc_value))
                print('(PEPit) Postprocessing: {} eigenvalue(s) > {} after dimension reduction'.format(nb_eigenvalues,
                                                                                                       eig_threshold))
                                                                                                       
            # Store the actualized obtained value
            wc_value = objective.value[0]	

        # Store all the values of points and function values
        self._eval_points_and_function_values(F.value, G.value, verbose=verbose)

        # Store all the dual values in constraints
        self._eval_constraint_dual_values(prob.constraints)

        # Return the value of the minimal performance metric or the full cvxpy Problem object
        if return_full_cvxpy_problem:
            # Return the cvxpy Problem object
            return prob
        else:
            # Return the value of the minimal performance metric
            return wc_value

    @staticmethod
    def get_nb_eigenvalues_and_corrected_matrix(M):
        """
        Compute the number of True non zero eigenvalues of M, and recompute M with corrected eigenvalues.

        Args:
            M (nd.array): a 2 dimensional array, supposedly symmetric.

        Returns:
            nb_eigenvalues (int): The number of eigenvalues of M estimated to be strictly positive zero.
            eig_threshold (float): The threshold used to determine whether an eigenvalue is 0 or not.
            corrected_S (nd.array): Updated M with zero eigenvalues instead of small ones.

        """

        # Symmetrize M to get rid of small computation errors.
        S = (M + M.T) / 2

        # Get eig_val and eig_vec.
        eig_val, eig_vec = np.linalg.eigh(S)

        # Get the right threshold to use.
        eig_threshold = max(np.max(eig_val)/1e3, 2 * np.max(-eig_val))

        # Correct eig_val accordingly.
        non_zero_eig_vals = eig_val >= eig_threshold
        nb_eigenvalues = int(np.sum(non_zero_eig_vals))
        corrected_eig_val = non_zero_eig_vals * eig_val

        # Recompute M (or S) accordingly.
        corrected_S = eig_vec @ np.diag(corrected_eig_val) @ eig_vec.T

        # Get the highest eigenvalue that have been set to 0.
        eig_threshold = max(np.max(eig_val[non_zero_eig_vals == 0]), 0)

        return nb_eigenvalues, eig_threshold, corrected_S

    def _eval_points_and_function_values(self, F_value, G_value, verbose):
        """
        Store values of :class:`Point` and :class:`Expression objects at optimum after the PEP has been solved.

        Args:
            F_value (nd.array): value of the cvxpy variable F
            G_value (nd.array): value of the cvxpy variable G
            verbose (bool): if True, details of computation are printed

        """

        # Write the gram matrix G as M.T M to extract points' values
        eig_val, eig_vec = np.linalg.eig(G_value)

        # Verify negative eigenvalues are only precision mistakes and get rid of negative eigenvalues
        if np.min(eig_val) < 0:
            if verbose:
                print("\033[96m(PEPit) Postprocessing: solver\'s output is not entirely feasible"
                      " (smallest eigenvalue of the Gram matrix is: {:.3} < 0).\n"
                      " Small deviation from 0 may simply be due to numerical error."
                      " Big ones should be deeply investigated.\n"
                      " In any case, from now the provided values of parameters are based on the projection of the Gram"
                      " matrix onto the cone of symmetric semi-definite matrix.\033[0m".format(np.min(eig_val)))
            eig_val = np.maximum(eig_val, 0)

        # Extracts points values
        points_values = np.linalg.qr((np.sqrt(eig_val) * eig_vec).T, mode='r')

        # Iterate over point and function value
        # Set the attribute value of all leaf variables to the right value
        # Note the other ones are not stored until user asks to eval them
        for point in self.list_of_points:
            if point.get_is_leaf():
                point._value = points_values[:, point.counter]
        for function in self.list_of_functions:
            if function.get_is_leaf():
                for triplet in function.list_of_points:
                    point, gradient, function_value = triplet
                    if point.get_is_leaf():
                        point._value = points_values[:, point.counter]
                    if gradient.get_is_leaf():
                        gradient._value = points_values[:, gradient.counter]
                    if function_value.get_is_leaf():
                        function_value._value = F_value[function_value.counter]
        for matrix in self.list_of_psd:
            size = matrix.shape[0]
            for i in range(size):
                for j in range(size):
                    expression = matrix[i, j]
                    if expression.get_is_leaf():
                        expression._value = F_value[expression.counter]
                    else:
                        for sub_expression in expression.decomposition_dict:
                            # Distinguish 3 cases: function values, inner products, and constant values
                            if type(sub_expression) == Expression:
                                assert sub_expression.get_is_leaf()
                                sub_expression._value = F_value[sub_expression.counter]
                            elif type(sub_expression) == tuple:
                                point1, point2 = sub_expression
                                assert point1.get_is_leaf()
                                assert point2.get_is_leaf()
                                point1._value = points_values[:, point1.counter]
                                point2._value = points_values[:, point2.counter]
                            elif sub_expression == 1:
                                pass
                            # Raise Exception out of those 3 cases
                            else:
                                raise TypeError(
                                    "Expressions are made of function values, inner products and constants only!"
                                    "Got {}".format(type(sub_expression)))

    def _eval_constraint_dual_values(self, cvx_constraints):
        """
        Store all dual values in associated :class:`Constraint` objects.

        Args:
            cvx_constraints (list): a list of cvxpy formatted constraints.

        Returns:
             position_of_minimal_objective (np.float): the position, in the list of performance metric,
                                                       of the one that is actually reached.

        """

        # Set counter
        counter = len(self.list_of_performance_metrics)

        # The dual variables associated to performance metric all have nonnegative values of sum 1.
        # Generally, only 1 performance metric is used.
        # Then its associated dual values is 1 while the others'associated dual values are 0.
        performance_metric_dual_values = np.array([constraint.dual_value for constraint in cvx_constraints[:counter]])
        performance_metric_dual_values = performance_metric_dual_values.reshape(-1)
        position_of_minimal_objective = np.argmax(performance_metric_dual_values)

        # Store all dual values of initial conditions (Generally the rate)
        for condition in self.list_of_constraints:
            condition._dual_variable_value = cvx_constraints[counter].dual_value
            counter += 1

        # Store all the class constraints dual values, providing the proof of the desired rate.
        for function in self.list_of_functions:
            for constraint in function.list_of_constraints:
                constraint._dual_variable_value = cvx_constraints[counter].dual_value
                counter += 1

        # Return the position of the reached performance metric
        return position_of_minimal_objective
