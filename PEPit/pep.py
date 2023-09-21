import numpy as np
import sys
import importlib.util

from PEPit.wrappers.cvxpy_wrapper import Cvxpy_wrapper
from PEPit.wrappers.mosek_wrapper import Mosek_wrapper
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.constraint import Constraint
from PEPit.function import Function
from PEPit.psd_matrix import PSDMatrix
from PEPit.block_partition import BlockPartition
                  
## Add references to your wrappers here.
MOSEK = {'libname': 'mosek', 'wrapper': Mosek_wrapper}
CVXPY = {'libname': 'cvxpy', 'wrapper': Cvxpy_wrapper}

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
        _list_of_constraints_sent_to_mosek (list): a list of all the :class:`Constraint` objects that are sent to MOSEK
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
        # Set all counters to 0 to recreate
        # points, expressions, functions and block partitions from scratch at the beginning of each PEP.
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
        self._list_of_constraints_sent_to_mosek = list()

    @staticmethod
    def _reset_classes():
        """
        Reset all classes attributes to initial values when instantiating a new :class:`PEP` object.

        """

        BlockPartition.counter = 0
        BlockPartition.list_of_partitions = list()
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

    @staticmethod
    def declare_block_partition(d):
        """
        Instantiate a :class:`BlockPartition` and store it in the attribute `list_of_partitions`.

        Args:
            d (int): number of blocks in the :class:`BlockPartition`.

        Returns:
            block_partition (BlockPartition): the newly created partition.

        """

        # Create the partition
        block_partition = BlockPartition(d)

        # Return it
        return block_partition

    def set_performance_metric(self, expression):
        """
        Store a performance metric in the attribute `list_of_performance_metrics`.
        The objective of the PEP (which is maximized) is the minimum of the elements of `list_of_performance_metrics`.

        Args:
            expression (Expression): a new performance metric.

        """

        # Store performance metric in the appropriate list
        self.list_of_performance_metrics.append(expression)
        
    def solve(self, verbose=1, return_full_problem=False,
              dimension_reduction_heuristic=None, eig_regularization=1e-3, tol_dimension_reduction=1e-5, solver=CVXPY,
              **kwargs):
        """
        Transform the :class:`PEP` under the SDP form, and solve it. Parse the options for solving the SDPs,
        instantiate the concerning wrappers and call the main internal solve option for solving the PEP.

        Args:
            verbose (int): Level of information details to print (Override the CVXPY solver verbose parameter).

                            - 0: No verbose at all
                            - 1: PEPit information is printed but not CVXPY's
                            - 2: Both PEPit and CVXPY details are printed
            return_full_problem (bool): If True, return the problem object (whose type depends on the solver).
                                        If False, only return the worst-case value.
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
            solver (dict, optional): Reference to a solver, interfaced by a :class:`PEPit.Wrapper`. 
                                     Default is CVXPY, other native option include MOSEK.
            kwargs (keywords, optional): Additional solver-specific arguments.

        Returns:
            float or Problem: Value of the performance metric of the PEP, or a reference to a solver-specific representation
                              of the PEP. The value is returned by default.

        """
        # Check that the solver is installed, if it is not, switch to CVXPY.
        find_solver = importlib.util.find_spec(solver['libname'])
        solver_found = find_solver is not None
        if not solver_found:
            solver = CVXPY
            print('(PEPit) {} not found, switching to cvxpy'.format(solver['libname']))
        
        # Initiate a wrapper to interface with the solver 
        wrap = solver['wrapper']()
        
        # Check that a valid license to the solver is found. Otherwise, switch to CVXPY.
        if not wrap.check_license():
            print('(PEPit) No valid {} license found, switching to cvxpy'.format(solver['libname']))
            solver = CVXPY
            wrap = solver['wrapper']()
        
        # Call the internal solve methods, which formulates and solves the PEP via the SDP solver.
        out = self._generic_solve(wrap, verbose, return_full_problem, dimension_reduction_heuristic, eig_regularization, tol_dimension_reduction, **kwargs)
        return out

    def _generic_solve(self, wrapper, verbose=1, return_full_problem=False,
              dimension_reduction_heuristic=None, eig_regularization=1e-3, tol_dimension_reduction=1e-5,
              **kwargs):      
        """
        Internal solve method. Transate the :class:`PEP` to an SDP, and solve it via the wrapper.

        Args:
            wrapper (Wrapper): Interface to the solver.
            verbose (int): Level of information details to print (Override the CVXPY solver verbose parameter).

                            - 0: No verbose at all
                            - 1: PEPit information is printed but not CVXPY's
                            - 2: Both PEPit and CVXPY details are printed
            return_full_problem (bool): If True, return the problem object (whose type depends on the solver).
                                        If False, only return the worst-case value.
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
            kwargs (keywords, optional): Additional solver-specific arguments.

        Returns:
            float or Problem: Value of the performance metric of the PEP, or a reference to a solver-specific representation
                              of the PEP. The value is returned by default.

        """
        
        # Create an expression that serve for the objective (min of the performance measures)   
        tau = Expression(is_leaf=True)
        objective = tau
        
        # Store functions that have class constraints as well as functions that have personal constraints
        list_of_leaf_functions = [function for function in Function.list_of_functions
                                  if function.get_is_leaf()]
        list_of_functions_with_constraints = [function for function in Function.list_of_functions
                                              if len(function.list_of_constraints) > 0 or len(function.list_of_psd) > 0]

        # Create all class constraints
        for function in list_of_leaf_functions:
            function.add_class_constraints()
            
        # Create all partition constraints
        for partition in BlockPartition.list_of_partitions:
            partition.add_partition_constraints()

        # Define the variables (G,F)
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' size of the main PSD matrix: {}x{}'.format(Point.counter, Point.counter))

        # Defining performance metrics
        # Note maximizing the minimum of all the performance metrics
        # is equivalent to maximize objective which is constraint to be smaller than all the performance metrics.
        for performance_metric in self.list_of_performance_metrics:
            assert isinstance(performance_metric, Expression)
            wrapper.send_constraint_to_solver(tau <= performance_metric)
        
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' performance measure is minimum of {} element(s)'.format(len(self.list_of_performance_metrics)))

        # Defining initial conditions and general constraints
        if verbose:
            print('(PEPit) Setting up the problem: Adding initial conditions and general constraints ...')
        for condition in self.list_of_constraints:
            wrapper.send_constraint_to_solver(condition)
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' initial conditions and general constraints ({} constraint(s) added)'.format(len(self.list_of_constraints)))

        # Defining general lmi constraints
        if len(self.list_of_psd) > 0:
            if verbose:
                print('(PEPit) Setting up the problem: {} lmi constraint(s) added'.format(len(self.list_of_psd)))
            for psd_counter, psd_matrix in enumerate(self.list_of_psd):
                wrapper.send_lmi_constraint_to_solver(psd_counter, psd_matrix)

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
                wrapper.send_constraint_to_solver(constraint)

            if verbose:
                print('\t\t function', function_counter, ':', len(function.list_of_class_constraints), 'scalar constraint(s) added')

            if len(function.list_of_class_psd) > 0:
                if verbose:
                    print('\t\t function', function_counter, ':', 'Adding', len(function.list_of_class_psd), 'lmi constraint(s) ...')

                for psd_counter, psd_matrix in enumerate(function.list_of_class_psd):
                    wrapper.send_lmi_constraint_to_solver(psd_counter, psd_matrix)

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
                    wrapper.send_constraint_to_solver(constraint)

                if verbose:
                    print('\t\t function', function_counter, ':', len(function.list_of_constraints),
                          'scalar constraint(s) added')

            if len(function.list_of_psd) > 0:
                if verbose:
                    print('\t\t function', function_counter, ':', 'Adding', len(function.list_of_psd),
                          'lmi constraint(s) ...')

                for psd_counter, psd_matrix in enumerate(function.list_of_psd):
                    wrapper.send_lmi_constraint_to_solver(psd_counter, psd_matrix)

                if verbose:
                    print('\t\t function', function_counter, ':', len(function.list_of_psd),
                          'lmi constraint(s) added')

        # Defining block partition constraints
        if verbose and len(BlockPartition.list_of_partitions) > 0:
            print('(PEPit) Setting up the problem: {} partition(s) added'.format(len(BlockPartition.list_of_partitions)))

        partition_counter = 0
        for partition in BlockPartition.list_of_partitions:
            partition_counter += 1
            if verbose:
                print('\t\t partition', partition_counter, 'with', partition.get_nb_blocks(),
                      'blocks: Adding', len(partition.list_of_constraints), 'scalar constraint(s)...')
            for constraint in partition.list_of_constraints:
                wrapper.send_constraint_to_solver(constraint)
            if verbose:
                print('\t\t partition', partition_counter, 'with', partition.get_nb_blocks(),
                      'blocks:', len(partition.list_of_constraints), 'scalar constraint(s) added')

        # Instantiate the problem
        if verbose:
            print('(PEPit) Compiling SDP')
        wrapper.generate_problem(objective)

        # Solve it
        if verbose:
            print('(PEPit) Calling SDP solver')
        solver_status, solver_name, wc_value, prob = wrapper.solve(**kwargs)
        if verbose:
            print('(PEPit) Solver status: {} (solver: {}); optimal value: {}'.format(solver_status,
                                                                                     solver_name,
                                                                                     wc_value))

        # Raise explicit error when wc_value in infinite
        if wc_value == np.inf:
            raise UserWarning("PEPit didn't find any nontrivial worst-case guarantee. "
                              "It seems that the optimal value of your problem is unbounded.")

        # Keep dual values before dimension reduction in memory
        # Dimension aims at finding low dimension lower bound functions,
        # but solves a different problem with an extra condition and different objective,
        # leading to different dual values. The ones we store here provide the proof of the obtained guarantee.
        dual_values, self.residual, dual_objective = wrapper.eval_constraint_dual_values()
        G_value, F_value = wrapper.get_primal_variables()

        # Perform a dimension reduction if required
        if dimension_reduction_heuristic:

            # Print the estimated dimension before dimension reduction
            nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(G_value)
            wrapper.prepare_heuristic(wc_value, tol_dimension_reduction)
            if verbose:
                print('(PEPit) Postprocessing: {} eigenvalue(s) > {} before dimension reduction'.format(nb_eigenvalues,
                                                                                                        eig_threshold))
                print('(PEPit) Calling SDP solver')

            # Translate the heuristic into the objective and solve the associated problem
            if dimension_reduction_heuristic == "trace":
                wrapper.heuristic(np.identity(Point.counter))
                solver_status, solver_name, wc_value, prob = wrapper.solve(**kwargs)

                # Compute minimal number of dimensions
                G_value, F_value = wrapper.get_primal_variables()
                nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(G_value)

            elif dimension_reduction_heuristic.startswith("logdet"):
                niter = int(dimension_reduction_heuristic[6:])
                for i in range(1, 1+niter):
                    W = np.linalg.inv(corrected_G_value + eig_regularization * np.eye(Point.counter))
                    wrapper.heuristic(W)
                    solver_status, solver_name, wc_value, prob = wrapper.solve(**kwargs)

                    # Compute minimal number of dimensions
                    G_value, F_value = wrapper.get_primal_variables()
                    nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(G_value)

                    # Print the estimated dimension after i dimension reduction steps
                    if verbose:
                        print('(PEPit) Solver status: {} (solver: {});'
                              ' objective value: {}'.format(solver_status,
                                                            solver_name,
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
                      ' objective value: {}'.format(solver_status,
                                                    solver_name,
                                                    wc_value))
                print('(PEPit) Postprocessing: {} eigenvalue(s) > {} after dimension reduction'.format(nb_eigenvalues,
                                                                                                       eig_threshold))

        # Store all the values of points and function values
        self._eval_points_and_function_values(F_value, G_value, verbose=verbose)

        # Store all the dual values in constraints
        #_, dual_objective = self._eval_constraint_dual_values(dual_values, wrapper)
        if verbose:
            print('(PEPit) Final upper bound (dual): {} and lower bound (primal example): {} '.format(dual_objective, wc_value))
            print('(PEPit) Duality gap: absolute: {} and relative: {}'.format(dual_objective-wc_value, (dual_objective-wc_value)/wc_value))

        # Return the value of the minimal performance metric or the full cvxpy Problem object
        if return_full_problem:
            # Return the problem object (specific to solver) 
            return prob
        else:
            # Return the value of the minimal performance metric
            return wc_value    


    def _verify_accuracy():
        """XXXX TODOTODOs

        """
        # CHECK FEASIBILITY (primal & dual; all constraints (LMIs and linear constraints))!
    	# CHECK PD GAP
        primal_lin_accuracy = 0.
        primal_PSD_accuracy = 0.
        dual_lin_accuracy = 0.
        dual_PSD_accuracy = 0.
        PD_gap = 0.
        return primal_lin_accuracy, primal_PSD_accuracy, dual_lin_accuracy, dual_PSD_accuracy, PD_gap
 
        
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
