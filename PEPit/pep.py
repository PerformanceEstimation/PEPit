import importlib.util

import numpy as np

from PEPit.tools.dict_operations import prune_dict, symmetrize_dict

from PEPit.wrappers import WRAPPERS
from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.constraint import Constraint
from PEPit.function import Function
from PEPit.psd_matrix import PSDMatrix
from PEPit.block_partition import BlockPartition


class PEP(object):
    """
    The class :class:`PEP` is the main class of this framework.
    A :class:`PEP` object encodes a complete performance estimation problem.
    It stores the following information.

    Attributes:
        counter (int): counts the number of :class:`PEP` objects.
                       Ideally, only one is defined at a time.

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

        wrapper_name (str): name of the used wrapper.
        wrapper (Wrapper): :class:`Wrapper` object that interfaces between the :class:`PEP` and the solver.

        _list_of_constraints_sent_to_wrapper (list): list of :class:`Constraint` objects sent to the wrapper.
        _list_of_psd_sent_to_wrapper (list): list of :class:`PSDMatrix` objects sent to the wrapper.

        objective (Expression): the expression to be maximized by the solver.
                                It is set by the method `solve`. And should not be updated otherwise.

        G_value (ndarray): the value of the Gram matrix G that the solver found.
        F_value (ndarray): the value of the vector of :class:`Expression`s F that the solver found.

        residual (ndarray): the dual value found by the solver to the lmi constraints G >> 0.

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
        # The PEP will maximize the minimum of the latter.
        self.list_of_functions = list()
        self.list_of_points = list()
        self.list_of_constraints = list()
        self.list_of_performance_metrics = list()
        self.list_of_psd = list()

        # Initialize wrapper information
        # The wrapper will be determined in the method "solve".
        self.wrapper_name = None
        self.wrapper = None

        # Initialize lists of constraints that will be sent to the wrapper to solve the SDP.
        # Those lists should not be updated by hand, only the solve method does update them.
        self._list_of_constraints_sent_to_wrapper = list()
        self._list_of_psd_sent_to_wrapper = list()

        # The attribute objective will contain a leaf Expression when set in the method "solve".
        self.objective = None
        # The Gram matrix G and the vector of Expressions F will obtain value from the solver,
        # stored in the 2 following attributes.
        # From those values, all Points and Expressions receive a primal value.
        self.G_value = None
        self.F_value = None
        # All PEP Constraints receive also a dual value.
        # The constraint G >= 0 is the only constraint that is not defined from the class Constraint.
        # Its dual value, called residual, is then stored in the following attribute.
        self.residual = None

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
                           Can also contain the boolean `reuse_gradient`,
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

    def set_initial_point(self, name=None):
        """
        Create a new leaf :class:`Point` and store it in the attribute `list_of_points`.

        Args:
            name (str, optional): name of the object. Not overwriting is None. None by default.

        Returns:
            x (Point): the newly created :class:`Point`.

        """

        # Create a new point from scratch
        x = Point(is_leaf=True, decomposition_dict=None)

        # Set name
        if name is not None:
            x.set_name(name=name)

        # Store it in list_of_points
        self.list_of_points.append(x)

        # Return it
        return x

    def set_initial_condition(self, condition, name=None):
        """
        Store a new :class:`Constraint` to the list of constraints of this :class:`PEP`.
        Typically, a condition of the form :math:`\\|x_0 - x_\\star\\|^2 \\leq 1`.

        Args:
            condition (Constraint): typically resulting from a comparison of 2 :class:`Expression` objects.
            name (str, optional): name of the object. Not overwriting is None. None by default.

        Raises:
            AssertionError: if provided `constraint` is not a :class:`Constraint` object.

        """
        # Set name
        if name is not None:
            condition.set_name(name=name)

        # Call add_constraint method
        self.add_constraint(constraint=condition)

    def add_constraint(self, constraint, name=None):
        """
        Store a new :class:`Constraint` to the list of constraints of this :class:`PEP`.

        Args:
            constraint (Constraint): typically resulting from a comparison of 2 :class:`Expression` objects.
            name (str, optional): name of the object. Not overwriting is None. None by default.

        Raises:
            AssertionError: if provided `constraint` is not a :class:`Constraint` object.

        """

        # Verify constraint is an actual Constraint object
        assert isinstance(constraint, Constraint)

        # Set name
        if name is not None:
            constraint.set_name(name=name)

        # Add constraint to the list of self's constraints
        self.list_of_constraints.append(constraint)

    def add_psd_matrix(self, matrix_of_expressions, name=None):
        """
        Store a new matrix of :class:`Expression`\s that we enforce to be positive semi-definite.

        Args:
            matrix_of_expressions (Iterable of Iterable of Expression): a square matrix of :class:`Expression`.
            name (str, optional): name of the object. Not overwriting is None. None by default.

        Returns:
            (PSDMatrix) the :class:`PSDMatrix` to be added to the :class:`PEP`.

        Raises:
            AssertionError: if provided matrix is not a square matrix.
            TypeError: if provided matrix does not contain only Expressions.

        """
        if isinstance(matrix_of_expressions, PSDMatrix):
            matrix = matrix_of_expressions
        else:
            matrix = PSDMatrix(matrix_of_expressions=matrix_of_expressions)

        # Set name
        if name is not None:
            matrix.set_name(name=name)

        # Add constraint to the list of self's constraints
        self.list_of_psd.append(matrix)

        return matrix

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

    def set_performance_metric(self, expression, name=None):
        """
        Store a performance metric in the attribute `list_of_performance_metrics`.
        The objective of the PEP (which is maximized) is the minimum of the elements of `list_of_performance_metrics`.

        Args:
            expression (Expression): a new performance metric.
            name (str, optional): name of the object. Not overwriting is None. None by default.

        """
        assert isinstance(expression, Expression)

        # Set name
        if name is not None:
            expression.set_name(name=name)

        # Store performance metric in the appropriate list
        self.list_of_performance_metrics.append(expression)

    def solve(self, wrapper="cvxpy", return_primal_or_dual="dual", verbose=1,
              dimension_reduction_heuristic=None, eig_regularization=1e-3, tol_dimension_reduction=1e-4, **kwargs):
        """
        Transform the :class:`PEP` under the SDP form, and solve it. Parse the options for solving the SDPs,
        instantiate the concerning wrappers and call the main internal solve option for solving the PEP.

        Args:
            wrapper (str, optional): Reference to a solver, interfaced by a :class:`PEPit.Wrapper`.
                                     Default is "cvxpy", other native option include "mosek".
            return_primal_or_dual (str, optional): If "dual", it returns a worst-case upper bound of the PEP
                                                   (dual value of the objective).
                                                   If "primal", it returns a worst-case lower bound of the PEP
                                                   (primal value of the objective).
                                                   Default is "dual".
                                                   Note both value should be almost the same by strong duality.

            verbose (int, optional): Level of information details to print
                                     (Override the solver verbose parameter).

                                     - 0: No verbose at all
                                     - 1: PEPit information is printed but not solver's
                                     - 2: Both PEPit and solver details are printed
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
            float: Worst case guarantee of the PEP.

        """
        wrapper_name = wrapper.lower()

        # Check that the solver is installed, if it is not, switch to CVXPY.
        found_python_package = importlib.util.find_spec(wrapper_name)
        if found_python_package is None:
            if verbose:
                print('\033[96m(PEPit) {} not found in system environment,'
                      ' switching to cvxpy\033[0m'.format(wrapper_name))
            wrapper_name = "cvxpy"

        # Initiate a wrapper to interface with the solver
        wrapper = WRAPPERS[wrapper_name](verbose=verbose)

        # Check that a valid license to the solver is found. Otherwise, switch to CVXPY.
        if not wrapper.check_license():
            if verbose:
                print('\033[96m(PEPit) No valid {} license found, switching to cvxpy\033[96m'.format(wrapper_name))
            wrapper_name = "cvxpy"
            wrapper = WRAPPERS[wrapper_name](verbose=verbose)

        # Store wrapper information in self
        self.wrapper_name = wrapper_name
        self.wrapper = wrapper

        # Call the internal solve methods, which formulates and solves the PEP via the SDP solver.
        out = self._solve_with_wrapper(wrapper, verbose, return_primal_or_dual,
                                       dimension_reduction_heuristic,
                                       eig_regularization, tol_dimension_reduction, **kwargs)
        return out

    def _solve_with_wrapper(self, wrapper, verbose=1, return_primal_or_dual="dual",
                            dimension_reduction_heuristic=None, eig_regularization=1e-3, tol_dimension_reduction=1e-4,
                            **kwargs):
        """
        Internal solve method. Translate the :class:`PEP` to an SDP, and solve it via the wrapper.

        Args:
            wrapper (Wrapper): Interface to the solver.
            verbose (int, optional): Level of information details to print
                                     (Override the CVXPY solver verbose parameter).

                            - 0: No verbose at all
                            - 1: PEPit information is printed but not CVXPY's
                            - 2: Both PEPit and solver details are printed
            return_primal_or_dual (str, optional): If "dual", it returns a worst-case upper bound of the PEP
                                                   (dual value of the objective).
                                                   If "primal", it returns a worst-case lower bound of the PEP
                                                   (primal value of the objective).
                                                   Default is "dual".
                                                   Note both value should be almost the same by strong duality.
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
            float: Worst-case guarantee of the PEP.

        """

        # Create an expression that serve for the objective (min of the performance measures)
        self.objective = Expression(is_leaf=True)

        # Store functions that have class constraints as well as functions that have personal constraints
        list_of_leaf_functions = [function for function in Function.list_of_functions
                                  if function.get_is_leaf()]
        list_of_functions_with_constraints = [function for function in Function.list_of_functions
                                              if len(function.list_of_constraints) > 0 or len(function.list_of_psd) > 0]

        # Create all class constraints
        for function in list_of_leaf_functions:
            function.set_class_constraints()

        # Create all partition constraints
        for partition in BlockPartition.list_of_partitions:
            partition.add_partition_constraints()

        # Report the creation of variables (G,F)
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' size of the Gram matrix: {}x{}'.format(Point.counter, Point.counter))
        wrapper.set_main_variables()

        # Initialize the lists of constraints sent to wrapper
        self._list_of_constraints_sent_to_wrapper = list()
        self._list_of_psd_sent_to_wrapper = list()

        # Defining performance metrics
        # Note maximizing the minimum of all the performance metrics
        # is equivalent to maximize objective which is constraint to be smaller than all the performance metrics.

        for performance_metric in self.list_of_performance_metrics:
            assert isinstance(performance_metric, Expression)
            performance_metric_constraint = (self.objective <= performance_metric)
            wrapper.send_constraint_to_solver(performance_metric_constraint)
            self._list_of_constraints_sent_to_wrapper.append(performance_metric_constraint)

        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' performance measure is the minimum of {} element(s)'.format(len(self.list_of_performance_metrics)))

        # Defining initial conditions and general constraints
        if verbose:
            print('(PEPit) Setting up the problem: Adding initial conditions and general constraints ...')
        for condition in self.list_of_constraints:
            wrapper.send_constraint_to_solver(condition)
            self._list_of_constraints_sent_to_wrapper.append(condition)
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' initial conditions and general constraints ({} constraint(s) added)'.format(
                    len(self.list_of_constraints)))

        # Defining general lmi constraints
        if len(self.list_of_psd) > 0:
            if verbose:
                print('(PEPit) Setting up the problem: {} lmi constraint(s) added'.format(len(self.list_of_psd)))
            for psd_counter, psd_matrix in enumerate(self.list_of_psd):
                wrapper.send_lmi_constraint_to_solver(psd_counter, psd_matrix)
                self._list_of_psd_sent_to_wrapper.append(psd_matrix)

        # Defining class constraints
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' interpolation conditions for {} function(s)'.format(len(list_of_leaf_functions)))
        function_counter = 0
        for function in list_of_leaf_functions:
            function_counter += 1

            if verbose:
                print('\t\t\tFunction', function_counter, ':', 'Adding', len(function.list_of_class_constraints),
                      'scalar constraint(s) ...')

            for constraint in function.list_of_class_constraints:
                wrapper.send_constraint_to_solver(constraint)
                self._list_of_constraints_sent_to_wrapper.append(constraint)

            if verbose:
                print('\t\t\tFunction', function_counter, ':', len(function.list_of_class_constraints),
                      'scalar constraint(s) added')

            if len(function.list_of_class_psd) > 0:
                if verbose:
                    print('\t\t\tFunction', function_counter, ':', 'Adding', len(function.list_of_class_psd),
                          'lmi constraint(s) ...')

                for psd_counter, psd_matrix in enumerate(function.list_of_class_psd):
                    wrapper.send_lmi_constraint_to_solver(psd_counter, psd_matrix)
                    self._list_of_psd_sent_to_wrapper.append(psd_matrix)

                if verbose:
                    print('\t\t\tFunction', function_counter, ':', len(function.list_of_class_psd),
                          'lmi constraint(s) added')

        # Other function constraints
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' additional constraints for {} function(s)'.format(len(list_of_functions_with_constraints)))
        function_counter = 0
        for function in list_of_functions_with_constraints:
            function_counter += 1

            if len(function.list_of_constraints) > 0:
                if verbose:
                    print('\t\t\tFunction', function_counter, ':', 'Adding', len(function.list_of_constraints),
                          'scalar constraint(s) ...')

                for constraint in function.list_of_constraints:
                    wrapper.send_constraint_to_solver(constraint)
                    self._list_of_constraints_sent_to_wrapper.append(constraint)

                if verbose:
                    print('\t\t\tFunction', function_counter, ':', len(function.list_of_constraints),
                          'scalar constraint(s) added')

            if len(function.list_of_psd) > 0:
                if verbose:
                    print('\t\t function', function_counter, ':', 'Adding', len(function.list_of_psd),
                          'lmi constraint(s) ...')

                for psd_counter, psd_matrix in enumerate(function.list_of_psd):
                    wrapper.send_lmi_constraint_to_solver(psd_counter, psd_matrix)
                    self._list_of_psd_sent_to_wrapper.append(psd_matrix)

                if verbose:
                    print('\t\t\tFunction', function_counter, ':', len(function.list_of_psd),
                          'lmi constraint(s) added')

        # Defining block partition constraints
        if verbose and len(BlockPartition.list_of_partitions) > 0:
            print(
                '(PEPit) Setting up the problem: {} partition(s) added'.format(len(BlockPartition.list_of_partitions)))

        partition_counter = 0
        for partition in BlockPartition.list_of_partitions:
            partition_counter += 1
            if verbose:
                print('\t\t\tPartition', partition_counter, 'with', partition.get_nb_blocks(),
                      'blocks: Adding', len(partition.list_of_constraints), 'scalar constraint(s)...')
            for constraint in partition.list_of_constraints:
                wrapper.send_constraint_to_solver(constraint)
                self._list_of_constraints_sent_to_wrapper.append(constraint)
            if verbose:
                print('\t\t\tPartition', partition_counter, 'with', partition.get_nb_blocks(),
                      'blocks:', len(partition.list_of_constraints), 'scalar constraint(s) added')

        # Instantiate the problem
        if verbose:
            print('(PEPit) Compiling SDP')
        wrapper.generate_problem(self.objective)

        # Solve it
        if verbose:
            print('(PEPit) Calling SDP solver')
        solver_status, solver_name, wc_value = wrapper.solve(**kwargs)
        if verbose:
            print('(PEPit) Solver status: {} (wrapper:{}, solver: {}); optimal value: {}'.format(solver_status,
                                                                                                 self.wrapper_name,
                                                                                                 solver_name,
                                                                                                 wc_value))

        # Raise explicit error when wc_value in infinite
        if wc_value is None:
            if verbose:
                print("\033[96m(PEPit) Problem issue: PEPit didn't find any nontrivial worst-case guarantee. "
                      "It seems that the optimal value of your problem is unbounded.\033[0m")

            # Skip the following as no variable has a value
            return wc_value

        # Keep dual values before dimension reduction in memory
        # Dimension aims at finding low dimension lower bound functions,
        # but solves a different problem with an extra condition and different objective,
        # leading to different dual values. The ones we store here provide the proof of the obtained guarantee.
        self.residual = wrapper.assign_dual_values()
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
                solver_status, solver_name, wc_value = wrapper.solve(**kwargs)

                # Compute minimal number of dimensions
                G_value, F_value = wrapper.get_primal_variables()
                nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(G_value)

            elif dimension_reduction_heuristic.startswith("logdet"):
                niter = int(dimension_reduction_heuristic[6:])
                for i in range(1, 1 + niter):
                    W = np.linalg.inv(corrected_G_value + eig_regularization * np.eye(Point.counter))
                    wrapper.heuristic(W)
                    solver_status, solver_name, wc_value = wrapper.solve(**kwargs)

                    # Compute minimal number of dimensions
                    G_value, F_value = wrapper.get_primal_variables()
                    nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(
                        G_value)

                    # Print the estimated dimension after niter dimension reduction steps
                    if verbose:
                        print('(PEPit) Solver status: {} (solver: {});'
                              ' objective value: {}'.format(solver_status,
                                                            solver_name,
                                                            wc_value))
                        print(
                            '(PEPit) Postprocessing: {} eigenvalue(s) > {} after {} dimension reduction step(s)'.format(
                                nb_eigenvalues, eig_threshold, i))

            else:
                raise ValueError("The argument \'dimension_reduction_heuristic\' must be \'trace\'"
                                 "or \`logdet\` followed by an integer."
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
        self.G_value = G_value
        self.F_value = F_value
        self._eval_points_and_function_values(F_value, G_value, verbose=verbose)
        dual_objective = self.check_feasibility(wc_value, verbose=verbose)

        # Return the value of the minimal performance metric
        if return_primal_or_dual == "dual":
            return dual_objective
        elif return_primal_or_dual == "primal":
            return wc_value
        else:
            raise ValueError("The argument \'return_primal_or_dual\' must be \'dual\' or \`primal\`."
                             "Got {}".format(return_primal_or_dual))

    def check_feasibility(self, wc_value, verbose=1):
        """
        Check primal feasibility and display precision.
        Check dual feasibility and display precision.
        Compute and display primal-dual gap.

        Args:
            wc_value (float): the primal value of the PEP objective returned by the solver.
            verbose (int, optional): If larger or equal than 1, print intermediate information.

        Returns:
            (float): the dual value of the PEP objective.

        Notes:
            The dual feasibility consists in

                - verifying that the dual values associated to inequality constraints are nonnegative,
                - and verifying that the residual corresponds to the right linear combination of the constraints.

            The second point essentially means that verifying the dual feasibility consists in reconstructing the proof.

        """

        ################################################################################################################
        #################################################### Primal ####################################################
        ################################################################################################################
        if verbose:
            print("(PEPit) Primal feasibility check:")

        # Verify that the given wc_value corresponds to the objective value
        assert wc_value == self.objective.eval()

        # Grab the smallest eigenvalue of G
        G_min_eig_val = np.min(np.linalg.eigh(self.G_value)[0])
        if verbose:
            message = "\t\tThe solver found a Gram matrix that is positive semi-definite"
            if G_min_eig_val < 0:
                message += " up to an error of {}".format(-G_min_eig_val)
            print(message)

        # Grab the smallest eigenvalue of all the PSD matrices
        if self._list_of_psd_sent_to_wrapper:
            psd_min_eig_val = np.min([np.min(np.linalg.eigh(psd_matrix.eval())[0])
                                      for psd_matrix in self._list_of_psd_sent_to_wrapper])
            if verbose:
                message = "\t\tAll required PSD matrices are indeed positive semi-definite"
                if psd_min_eig_val < 0:
                    message += " up to an error of {}".format(-psd_min_eig_val)
                print(message)

        # Get the max value of all transgression of the constraints
        if self._list_of_constraints_sent_to_wrapper:
            max_constraint_error = np.max(
                [constraint.eval()
                 for constraint in self._list_of_constraints_sent_to_wrapper
                 if constraint.equality_or_inequality == "inequality"]
                + [np.abs(constraint.eval())
                   for constraint in self._list_of_constraints_sent_to_wrapper
                   if constraint.equality_or_inequality == "equality"]
            )
            if verbose:
                message = "\t\tAll the primal scalar constraints are verified"
                if max_constraint_error > 0:
                    message += " up to an error of {}".format(max_constraint_error)
                print(message)

        ################################################################################################################
        ##################################################### Dual #####################################################
        ################################################################################################################
        if verbose:
            print("(PEPit) Dual feasibility check:")

        # Verify that all dual variables are nonnegative.
        # Moreover, linear combination of the constraints with the right coefficients should lead to objective <= tau

        # Residual >= 0
        residual_min_eig_val = np.min(np.linalg.eigh(self.residual)[0])
        if verbose:
            message = "\t\tThe solver found a residual matrix that is positive semi-definite"
            if residual_min_eig_val < 0:
                message += " up to an error of {}".format(-residual_min_eig_val)
            print(message)
        # - <Gram, residual> <= 0
        constraints_combination = -np.dot(Point.list_of_leaf_points, np.dot(self.residual, Point.list_of_leaf_points))

        # LMI constraints
        # Dual >= 0
        if self._list_of_psd_sent_to_wrapper:
            lmi_dual_min_eig_val = np.min([np.min(np.linalg.eigh(psd_matrix.eval_dual())[0])
                                           for psd_matrix in self._list_of_psd_sent_to_wrapper])
            if verbose:
                message = "\t\tAll the dual matrices to lmi are positive semi-definite"
                if lmi_dual_min_eig_val < 0:
                    message += " up to an error of {}".format(-lmi_dual_min_eig_val)
                print(message)
            # - <psd_matrix, lmi_dual> <= 0
            for psd_matrix in self._list_of_psd_sent_to_wrapper:
                constraints_combination -= np.sum(psd_matrix.eval_dual() * psd_matrix.matrix_of_expressions)

        # Scalar constraints
        # Dual of inequality constraints >= 0
        inequality_constraint_dual_values = [constraint.eval_dual()
                                             for constraint in self._list_of_constraints_sent_to_wrapper
                                             if constraint.equality_or_inequality == "inequality"]
        if inequality_constraint_dual_values:
            inequality_constraint_dual_min_value = np.min(inequality_constraint_dual_values)
            if verbose:
                message = "\t\tAll the dual scalar values associated with inequality constraints are nonnegative"
                if inequality_constraint_dual_min_value < 0:
                    message += " up to an error of {}".format(-inequality_constraint_dual_min_value)
                print(message)
        # + <expression, dual> <= 0
        for constraint in self._list_of_constraints_sent_to_wrapper:
            constraints_combination += constraint.eval_dual() * constraint.expression

        # Proof reconstruction
        # At this stage, constraints_combination must be equal to "objective - tau"
        # which constitutes the proof as it has to be non-positive.
        # Compute an expression that should be exactly equal to the constant tau.
        dual_objective_expression = self.objective - constraints_combination
        # Operation over the decomposition dict of dual_objective_expression
        dual_objective_expression_decomposition_dict = prune_dict(
            symmetrize_dict(
                dual_objective_expression.decomposition_dict
            )
        )
        # Get the actual dual_objective from its dict
        if 1 in dual_objective_expression_decomposition_dict.keys():
            dual_objective = dual_objective_expression_decomposition_dict[1]
        else:
            dual_objective = 0.
        # Compute the remaining terms, that should be small and only due to numerical stability errors
        remaining_terms = np.sum(np.abs([value for key, value in dual_objective_expression_decomposition_dict.items()
                                         if key != 1]))
        if verbose:
            message = "(PEPit) The worst-case guarantee proof is perfectly reconstituted"
            if remaining_terms > 0:
                message += " up to an error of {}".format(remaining_terms)
            print(message)

        ################################################################################################################
        ################################################## Duality Gap #################################################
        ################################################################################################################
        absolute_duality_gap = dual_objective - wc_value
        if verbose:
            print('(PEPit) Final upper bound (dual): {} and lower bound (primal example): {} '.format(dual_objective,
                                                                                                      wc_value))

        if wc_value != 0:
            relative_duality_gap = (dual_objective - wc_value) / wc_value
            if verbose:
                print('(PEPit) Duality gap: absolute: {} and relative: {}'.format(absolute_duality_gap,
                                                                                  relative_duality_gap))
        else:
            relative_duality_gap = 0

        if abs(absolute_duality_gap) > 10**-3 and abs(relative_duality_gap) > 10**-6:
            message = "\033[96m(PEPit) Warning: the duality gap seems surprisingly large"
            if absolute_duality_gap < 0:
                message += " and negative"
            message += ".\n\t\tThe solver might not have converged properly.\n"\
                       "\t\tWe recommend to use another solver for confirmation.\033[0m"
            print(message)

        return dual_objective

    @staticmethod
    def get_nb_eigenvalues_and_corrected_matrix(M):
        """
        Compute the number of True non-zero eigenvalues of M, and recompute M with corrected eigenvalues.

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
        eig_threshold = max(np.max(eig_val) / 1e3, 2 * np.max(-eig_val))

        # Correct eig_val accordingly.
        non_zero_eig_vals = eig_val >= eig_threshold
        nb_eigenvalues = int(np.sum(non_zero_eig_vals))
        nb_zeros = M.shape[0] - nb_eigenvalues
        corrected_eig_val = non_zero_eig_vals * eig_val

        # Recompute M (or S) accordingly.
        corrected_S = eig_vec @ np.diag(corrected_eig_val) @ eig_vec.T

        # Get the highest eigenvalue that has been set to 0, if any.
        eig_threshold = 0

        if nb_zeros > 0:
            eig_threshold = max(np.max(eig_val[non_zero_eig_vals == 0]), 0)

        return nb_eigenvalues, eig_threshold, corrected_S

    def _eval_points_and_function_values(self, F_value, G_value, verbose=1):
        """
        Store values of :class:`Point` and :class:`Expression objects at optimum after the PEP has been solved.

        Args:
            F_value (nd.array): value of the cvxpy variable F
            G_value (nd.array): value of the cvxpy variable G
            verbose (int, optional): If larger or equal than 1, print intermediate information.

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
