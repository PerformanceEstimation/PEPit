import numpy as np
import cvxpy as cp

from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.constraint import Constraint
from PEPit.function import Function
from PEPit.psd_matrix import PSDMatrix


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
        list_of_psd (list): list of :class:`PSDMatrix` objects.
                            The PEP consider the associated LMI constraints psd_matrix >> 0.
        _list_of_constraints_sent_to_cvxpy (list): a list of all the :class:`Constraint` objects that are sent to CVXPY
                                                   for solving the SDP. It should not be updated manually.
                                                   Only the `solve` method takes care of it.
        _list_of_cvxpy_constraints (list): a list of all the CVXPY Constraints objects that have been sent by PEPit
                                           for solving the SDP. It should not be updated manually.
                                           Only the `solve` method takes care of it.
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
        self._reset_classes()

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

        # Initialize lists of constraints that are used to solve the SDP.
        # Those lists should not be updated by hand, only the solve method does update them.
        self._list_of_constraints_sent_to_cvxpy = list()
        self._list_of_cvxpy_constraints = list()

    @staticmethod
    def _reset_classes():
        """
        Reset all classes attributes to initial values when instantiating a new :class:`PEP` object.

        """

        Constraint.counter = 0
        Expression.counter = 0
        Expression.list_of_leaf_expressions = list()
        Function.counter = 0
        Function.list_of_functions = list()
        PEP.counter = 0
        Point.counter = 0
        Point.list_of_leaf_points = list()
        PSDMatrix.counter = 0

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

    def add_psd_matrix(self, matrix_of_expressions):
        """
        Store a new matrix of :class:`Expression`\s that we enforce to be positive semidefinite.

        Args:
            matrix_of_expressions (Iterable of Iterable of Expression): a square matrix of :class:`Expression`.

        Raises:
            AssertionError: if provided matrix is not a square matrix.
            TypeError: if provided matrix does not contain only Expressions.

        """
        matrix = PSDMatrix(matrix_of_expressions=matrix_of_expressions)

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

    def send_constraint_to_cvxpy(self, constraint, F, G):
        """
        Transform a PEPit :class:`Constraint` into a CVXPY one
        and add the 2 formats of the constraints into the tracking lists.

        Args:
            constraint (Constraint): a :class:`Constraint` object to be sent to CVXPY.
            F (CVXPY Variable): a CVXPY Variable referring to function values.
            G (CVXPY Variable): a CVXPY Variable referring to points and gradients.

        Raises:
            ValueError if the attribute `equality_or_inequality` of the :class:`Constraint`
            is neither `equality`, nor `inequality`.

        """

        # Sanity check
        assert isinstance(constraint, Constraint)

        # Add constraint to the attribute _list_of_constraints_sent_to_cvxpy to keep track of
        # all the constraints that have been sent to CVXPY as well as the order.
        self._list_of_constraints_sent_to_cvxpy.append(constraint)

        # Distinguish equality and inequality
        if constraint.equality_or_inequality == 'inequality':
            cvxpy_constraint = self._expression_to_cvxpy(constraint.expression, F, G) <= 0
        elif constraint.equality_or_inequality == 'equality':
            cvxpy_constraint = self._expression_to_cvxpy(constraint.expression, F, G) == 0
        else:
            # Raise an exception otherwise
            raise ValueError('The attribute \'equality_or_inequality\' of a constraint object'
                             ' must either be \'equality\' or \'inequality\'.'
                             'Got {}'.format(constraint.equality_or_inequality))

        # Add the corresponding CVXPY constraint to the list of constraints to be sent to CVXPY
        self._list_of_cvxpy_constraints.append(cvxpy_constraint)

    def send_lmi_constraint_to_cvxpy(self, psd_counter, psd_matrix, F, G, verbose):
        """
        Transform a PEPit :class:`PSDMatrix` into a CVXPY symmetric PSD matrix
        and add the 2 formats of the constraints into the tracking lists.

        Args:
            psd_counter (int): a counter useful for the verbose mode.
            psd_matrix (PSDMatrix): a matrix of expressions that is constrained to be PSD.
            F (CVXPY Variable): a CVXPY Variable referring to function values.
            G (CVXPY Variable): a CVXPY Variable referring to points and gradients.
            verbose (int): Level of information details to print (Override the CVXPY solver verbose parameter).

                            - 0: No verbose at all
                            - 1: PEPit information is printed but not CVXPY's
                            - 2: Both PEPit and CVXPY details are printed

        """

        # Sanity check
        assert isinstance(psd_matrix, PSDMatrix)

        # Add psd_matrix to the attribute _list_of_constraints_sent_to_cvxpy to keep track of
        # all the constraints that have been sent to CVXPY as well as the order.
        self._list_of_constraints_sent_to_cvxpy.append(psd_matrix)

        # Create a symmetric matrix in CVXPY
        M = cp.Variable(psd_matrix.shape, symmetric=True)

        # Store the lmi constraint
        cvxpy_constraints_list = [M >> 0]

        # Store one correspondence constraint per entry of the matrix
        for i in range(psd_matrix.shape[0]):
            for j in range(psd_matrix.shape[1]):
                cvxpy_constraints_list.append(M[i, j] == self._expression_to_cvxpy(psd_matrix[i, j], F, G))

        # Print a message if verbose mode activated
        if verbose:
            print('\t\t Size of PSD matrix {}: {}x{}'.format(psd_counter + 1, *psd_matrix.shape))

        # Add the corresponding CVXPY constraints to the list of constraints to be sent to CVXPY
        self._list_of_cvxpy_constraints += cvxpy_constraints_list

    def solve(self, verbose=1, return_full_cvxpy_problem=False,
              dimension_reduction_heuristic=None, eig_regularization=1e-3, tol_dimension_reduction=1e-5,
              **kwargs):
        """
        Transform the :class:`PEP` under the SDP form, and solve it.

        Args:
            verbose (int): Level of information details to print (Override the CVXPY solver verbose parameter).

                            - 0: No verbose at all
                            - 1: PEPit information is printed but not CVXPY's
                            - 2: Both PEPit and CVXPY details are printed
            return_full_cvxpy_problem (bool): If True, return the cvxpy Problem object.
                                              If False, return the worst case value only.
                                              Set to False by default.
            dimension_reduction_heuristic (str, optional): An heuristic to reduce the dimension of the solution
                                                           (rank of the Gram matrix). Set to None to deactivate
                                                           it (default value). Available heuristics are:
                                                           
                                                            - "trace": minimize :math:`Tr(G)`
                                                            - "logdet{an integer n}": minimize
                                                              :math:`\\log\\left(\\mathrm{Det}(G)\\right)`
                                                              using n iterations of local approximation problems.

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

        # Initialize lists of constraints that are used to solve the SDP.
        # Those lists should not be updated by hand, only the solve method does update them.
        # If solve is called again, they should be reinitialized.
        self._list_of_constraints_sent_to_cvxpy = list()
        self._list_of_cvxpy_constraints = list()

        # Store functions that have class constraints as well as functions that have personal constraints
        list_of_leaf_functions = [function for function in Function.list_of_functions
                                  if function.get_is_leaf()]
        list_of_functions_with_constraints = [function for function in Function.list_of_functions
                                              if len(function.list_of_constraints) > 0 or len(function.list_of_psd) > 0]

        # Create all class constraints
        for function in list_of_leaf_functions:
            function.add_class_constraints()

        # Define the cvxpy variables
        objective = cp.Variable()
        F = cp.Variable((Expression.counter,))
        G = cp.Variable((Point.counter, Point.counter), symmetric=True)
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' size of the main PSD matrix: {}x{}'.format(Point.counter, Point.counter))

        # Express the constraints from F, G and objective
        # Start with the main LMI condition
        self._list_of_cvxpy_constraints = [G >> 0]

        # Defining performance metrics
        # Note maximizing the minimum of all the performance metrics
        # is equivalent to maximize objective which is constraint to be smaller than all the performance metrics.
        for performance_metric in self.list_of_performance_metrics:
            assert isinstance(performance_metric, Expression)
            self._list_of_cvxpy_constraints.append(objective <= self._expression_to_cvxpy(performance_metric, F, G))
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' performance measure is minimum of {} element(s)'.format(len(self.list_of_performance_metrics)))

        # Defining initial conditions and general constraints
        if verbose:
            print('(PEPit) Setting up the problem: Adding initial conditions and general constraints ...')
        for condition in self.list_of_constraints:
            self.send_constraint_to_cvxpy(condition, F, G)
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' initial conditions and general constraints ({} constraint(s) added)'.format(len(self.list_of_constraints)))

        # Defining general lmi constraints
        if len(self.list_of_psd) > 0:
            if verbose:
                print('(PEPit) Setting up the problem: {} lmi constraint(s) added'.format(len(self.list_of_psd)))
            for psd_counter, psd_matrix in enumerate(self.list_of_psd):
                self.send_lmi_constraint_to_cvxpy(psd_counter, psd_matrix, F, G, verbose)

        # Defining class constraints
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' interpolation conditions for {} function(s)'.format(len(list_of_leaf_functions)))
        function_counter = 0
        for function in list_of_leaf_functions:
            function_counter += 1

            if verbose:
                print('\t\t function', function_counter, ':', 'Adding', len(function.list_of_class_constraints), 'scalar constraint(s) ...')

            for constraint in function.list_of_class_constraints:
                self.send_constraint_to_cvxpy(constraint, F, G)

            if verbose:
                print('\t\t function', function_counter, ':', len(function.list_of_class_constraints), 'scalar constraint(s) added')

            if len(function.list_of_class_psd) > 0:
                if verbose:
                    print('\t\t function', function_counter, ':', 'Adding', len(function.list_of_class_psd), 'lmi constraint(s) ...')

                for psd_counter, psd_matrix in enumerate(function.list_of_class_psd):
                    self.send_lmi_constraint_to_cvxpy(psd_counter, psd_matrix, F, G, verbose)

                if verbose:
                    print('\t\t function', function_counter, ':', len(function.list_of_class_psd), 'lmi constraint(s) added')

        # Other function constraints
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' constraints for {} function(s)'.format(len(list_of_functions_with_constraints)))
        function_counter = 0
        for function in list_of_functions_with_constraints:
            function_counter += 1

            if len(function.list_of_constraints) > 0:
                if verbose:
                    print('\t\t function', function_counter, ':', 'Adding', len(function.list_of_constraints),
                          'scalar constraint(s) ...')

                for constraint in function.list_of_constraints:
                    self.send_constraint_to_cvxpy(constraint, F, G)

                if verbose:
                    print('\t\t function', function_counter, ':', len(function.list_of_constraints),
                          'scalar constraint(s) added')

            if len(function.list_of_psd) > 0:
                if verbose:
                    print('\t\t function', function_counter, ':', 'Adding', len(function.list_of_psd),
                          'lmi constraint(s) ...')

                for psd_counter, psd_matrix in enumerate(function.list_of_psd):
                    self.send_lmi_constraint_to_cvxpy(psd_counter, psd_matrix, F, G, verbose)

                if verbose:
                    print('\t\t function', function_counter, ':', len(function.list_of_psd),
                          'lmi constraint(s) added')

        # Create the cvxpy problem
        if verbose:
            print('(PEPit) Compiling SDP')
        prob = cp.Problem(objective=cp.Maximize(objective), constraints=self._list_of_cvxpy_constraints)

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

        # Keep dual values before dimension reduction in memory
        # Dimension aims at finding low dimension lower bound functions,
        # but solves a different problem with an extra condition and different objective,
        # leading to different dual values. The ones we store here provide the proof of the obtained guarantee.
        assert self._list_of_cvxpy_constraints == prob.constraints
        dual_values = [constraint.dual_value for constraint in prob.constraints]

        # Perform a dimension reduction if required
        if dimension_reduction_heuristic:

            # Print the estimated dimension before dimension reduction
            nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(G.value)
            if verbose:
                print('(PEPit) Postprocessing: {} eigenvalue(s) > {} before dimension reduction'.format(nb_eigenvalues,
                                                                                                        eig_threshold))
                print('(PEPit) Calling SDP solver')

            # Add the constraint that the objective stay close to its actual value
            self._list_of_cvxpy_constraints.append(objective >= wc_value - tol_dimension_reduction)

            # Translate the heuristic into cvxpy objective and solve the associated problem
            if dimension_reduction_heuristic == "trace":
                heuristic = cp.trace(G)
                prob = cp.Problem(objective=cp.Minimize(heuristic), constraints=self._list_of_cvxpy_constraints)
                prob.solve(**kwargs)

                # Store the actualized obtained value
                wc_value = objective.value

                # Compute minimal number of dimensions
                nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(G.value)

            elif dimension_reduction_heuristic.startswith("logdet"):
                niter = int(dimension_reduction_heuristic[6:])
                for i in range(1, 1+niter):
                    W = np.linalg.inv(corrected_G_value + eig_regularization * np.eye(Point.counter))
                    heuristic = cp.sum(cp.multiply(G, W))
                    prob = cp.Problem(objective=cp.Minimize(heuristic), constraints=self._list_of_cvxpy_constraints)
                    prob.solve(**kwargs)

                    # Store the actualized obtained value
                    wc_value = objective.value

                    # Compute minimal number of dimensions
                    nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(G.value)

                    # Print the estimated dimension after i dimension reduction steps
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

        # Store all the values of points and function values
        self._eval_points_and_function_values(F.value, G.value, verbose=verbose)

        # Store all the dual values in constraints
        self._eval_constraint_dual_values(dual_values)

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
        nb_zeros = M.shape[0]-nb_eigenvalues
        corrected_eig_val = non_zero_eig_vals * eig_val

        # Recompute M (or S) accordingly.
        corrected_S = eig_vec @ np.diag(corrected_eig_val) @ eig_vec.T

        # Get the highest eigenvalue that has been set to 0, if any.
        eig_threshold = 0
        
        if nb_zeros > 0:
            eig_threshold = max(np.max(eig_val[non_zero_eig_vals == 0]), 0)

        return nb_eigenvalues, eig_threshold, corrected_S

    def _eval_points_and_function_values(self, F_value, G_value, verbose):
        """
        Store values of :class:`Point` and :class:`Expression objects at optimum after the PEP has been solved.

        Args:
            F_value (nd.array): value of the cvxpy variable F
            G_value (nd.array): value of the cvxpy variable G
            verbose (bool): if True, details of computation are printed

        Raises:
            TypeError if some matrix in `self.list_of_psd` contains some entry that :class:`Expression` objects
            composed of other things than leaf :class:`Expression`s or tuple of :class:`Points`.

        """

        # Write the gram matrix G as M.T M to extract points' values
        eig_val, eig_vec = np.linalg.eigh(G_value)

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
        for point in Point.list_of_leaf_points:
            point._value = points_values[:, point.counter]
        for expression in Expression.list_of_leaf_expressions:
            expression._value = F_value[expression.counter]

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

    def _eval_constraint_dual_values(self, dual_values):
        """
        Store all dual values in associated :class:`Constraint` and :class:`PSDMatrix` objects.

        Args:
            dual_values (list): the list of dual values of the problem constraints.

        Returns:
             position_of_minimal_objective (np.float): the position, in the list of performance metric,
                                                       of the one that is actually reached.

        Raises:
            TypeError if the attribute `_list_of_constraints_sent_to_cvxpy` of this object
            is neither a :class:`Constraint` object, nor a :class:`PSDMatrix` one.

        """
        # Store residual, dual value of the main lmi
        self.residual = dual_values[0]
        assert self.residual.shape == (Point.counter, Point.counter)

        # Set counter
        counter = len(self.list_of_performance_metrics)+1

        # The dual variables associated to performance metric all have nonnegative values of sum 1.
        # Generally, only 1 performance metric is used.
        # Then its associated dual values is 1 while the others'associated dual values are 0.
        performance_metric_dual_values = np.array(dual_values[1:counter])
        position_of_minimal_objective = np.argmax(performance_metric_dual_values)

        for constraint_or_psd in self._list_of_constraints_sent_to_cvxpy:
            if isinstance(constraint_or_psd, Constraint):
                constraint_or_psd._dual_variable_value = dual_values[counter]
                counter += 1
            elif isinstance(constraint_or_psd, PSDMatrix):
                assert dual_values[counter].shape == constraint_or_psd.shape
                constraint_or_psd._dual_variable_value = dual_values[counter]
                counter += 1
                size = constraint_or_psd.shape[0] * constraint_or_psd.shape[1]
                constraint_or_psd.entries_dual_variable_value = np.array(dual_values[counter:counter + size]
                                                                         ).reshape(constraint_or_psd.shape)
                counter += size
            else:
                raise TypeError("The list of constraints that are sent to CVXPY should contain only"
                                "\'Constraint\' objects of \'PSDMatrix\' objects."
                                "Got {}".format(type(constraint_or_psd)))

        # Verify nothing is left
        assert len(dual_values) == counter

        # Return the position of the reached performance metric
        return position_of_minimal_objective
