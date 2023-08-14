import numpy as np
import mosek as mosek
import sys
import PEPit.cvxpy_wrapper as cpw
import PEPit.mosek_wrapper as mkw

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
        
    #
    # GENERAL SOLVE
    #
    #
    def solve(self, verbose=1, return_full_cvxpy_problem=False,
              dimension_reduction_heuristic=None, eig_regularization=1e-3, tol_dimension_reduction=1e-5,
              **kwargs):
        #out = self._mosek_solve(verbose, return_full_cvxpy_problem, dimension_reduction_heuristic, eig_regularization, tol_dimension_reduction, **kwargs)
        #out = self._solve_cvxpy(verbose, return_full_cvxpy_problem, dimension_reduction_heuristic, eig_regularization, tol_dimension_reduction, **kwargs)
        #wrap = cpw.Cvxpy_wrapper()
        wrap = mkw.Mosek_wrapper()
        out = self._generic_solve(wrap, verbose, return_full_cvxpy_problem, dimension_reduction_heuristic, eig_regularization, tol_dimension_reduction, **kwargs)
        return out

    def _generic_solve(self, wrapper, verbose=1, return_full_problem=False,
              dimension_reduction_heuristic=None, eig_regularization=1e-3, tol_dimension_reduction=1e-5,
              **kwargs):
        
        ## TODO: turn the generic 'verbose' of the solver ON ?
        
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

        # Define the cvxpy variables
        self.talkative_description(verbose)

        # Defining performance metrics
        # Note maximizing the minimum of all the performance metrics
        # is equivalent to maximize objective which is constraint to be smaller than all the performance metrics.
        for performance_metric in self.list_of_performance_metrics:
            assert isinstance(performance_metric, Expression)
            wrapper.send_constraint_to_solver(tau <= performance_metric)
            #cp_wrap._list_of_cvxpy_constraints.append(objective <= cp_wrap._expression_to_cvxpy(performance_metric))
        self.talkative_performance_measure(verbose)

        # Defining initial conditions and general constraints
        self.talkative_handling_constraints(verbose)
        for condition in self.list_of_constraints:
            wrapper.send_constraint_to_solver(condition)
        self.talkative_constraints(verbose)

        # Defining general lmi constraints
        if len(self.list_of_psd) > 0:
            self.talkative_LMIs(verbose, len(self.list_of_psd))
            for psd_counter, psd_matrix in enumerate(self.list_of_psd):
                wrapper.send_lmi_constraint_to_solver(psd_counter, psd_matrix, verbose)

        # Defining class constraints
        self.talkative_functions(verbose, len(list_of_leaf_functions))
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
                    wrapper.send_lmi_constraint_to_solver(psd_counter, psd_matrix, verbose)

                if verbose:
                    print('\t\t function', function_counter, ':', len(function.list_of_class_psd), 'lmi constraint(s) added')

        # Other function constraints
        self.talkative_function_constraints(verbose,len(list_of_functions_with_constraints))
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
                    wrapper.send_lmi_constraint_to_solver(psd_counter, psd_matrix, verbose)

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

        # Create the cvxpy problem
        if verbose:
            print('(PEPit) Compiling SDP')
        prob = wrapper.generate_problem(objective)

        # Solve it
        if verbose:
            print('(PEPit) Calling SDP solver')
        solver_status, solver_name, wc_value = wrapper.solve(**kwargs)
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

           

            # Translate the heuristic into cvxpy objective and solve the associated problem
            if dimension_reduction_heuristic == "trace":
                wrapper.heuristic()
                solver_status, solver_name, wc_value = wrapper.solve(**kwargs)

                # Compute minimal number of dimensions
                G_value, F_value = wrapper.get_primal_variables()
                nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(G_value)

            elif dimension_reduction_heuristic.startswith("logdet"):
                niter = int(dimension_reduction_heuristic[6:])
                for i in range(1, 1+niter):
                    W = np.linalg.inv(corrected_G_value + eig_regularization * np.eye(Point.counter))
                    wrapper.heuristic(W)
                    solver_status, solver_name, wc_value = wrapper.solve(**kwargs)

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
        
        print('dual: {}, primal: {}'.format(dual_objective, wc_value))

        # Return the value of the minimal performance metric or the full cvxpy Problem object
        if return_full_problem:
            # Return the problem object (specific to solver) 
            return prob
        else:
            # Return the value of the minimal performance metric
            return wc_value    

    #
    # MOSEK PARTS
    #
    #

            

    def _mosek_solve(self, verbose=1, return_full_mosek_problem=False,
              dimension_reduction_heuristic=None, eig_regularization=1e-3, tol_dimension_reduction=1e-5,
              **kwargs):
        """XXXX

        """
        inf = 1.0 # for symbolic purposes
        
        # we start by creating an expression for handling the objective function (minimum among all performance metrics)
        ## TODO: modifier (soit cette fonction soit approche CVXPY): soit utiliser tau comme une variable PEP des deux côtés, soit d'aucun des deux (pour l'instant juste coté mosek)
        tau = Expression(is_leaf=True)
        
        with mosek.Task() as task: #initiate MOSEK's task
            if verbose >= 2:
                task.set_Stream(mosek.streamtype.log, streamprinter) # printer
            

            # Initialize lists of constraints that are used to solve the SDP.
            # Those lists should not be updated by hand, only the solve method does update them.
            # If solve is called again, they should be reinitialized.
            self._list_of_constraints_sent_to_mosek = list()
            
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

            # Start the formulation: initiaze the MOSEK variable
            self.talkative_description(verbose)
            
            task.appendbarvars([Point.counter]) # init the Gram matrix
            task.appendvars(Expression.counter) # init function value variables (the additional variable is "tau" for handling the objective)
            
            for i in range(Expression.counter):
                task.putvarbound(i, mosek.boundkey.fr, -inf, +inf) # no bounds on function values
                
            # Defining performance metrics
            # Note maximizing the minimum of all the performance metrics
            # is equivalent to maximize objective which is constraint to be smaller than all the performance metrics.
            for performance_metric in self.list_of_performance_metrics:
                assert isinstance(performance_metric, Expression)
                self.send_constraint_to_mosek(tau <= performance_metric, task)
            self.talkative_performance_measure(verbose)
            
            # Defining initial conditions and general constraints
            self.talkative_handling_constraints(verbose)
            for condition in self.list_of_constraints:
                self.send_constraint_to_mosek(condition, task)
            self.talkative_constraints(verbose)

            # Defining general lmi constraints
            if len(self.list_of_psd) > 0:
                self.talkative_LMIs(verbose, len(self.list_of_psd))
                for psd_counter, psd_matrix in enumerate(self.list_of_psd):
                    self.send_lmi_constraint_to_mosek(psd_counter, psd_matrix, task, verbose)
                    
            # Defining class constraints
            self.talkative_functions(verbose, len(list_of_leaf_functions))
            function_counter = 0
            for function in list_of_leaf_functions:
                function_counter += 1

                if verbose:
                    print('\t\t function', function_counter, ':', 'Adding', len(function.list_of_class_constraints), 'scalar constraint(s) ...')

                for constraint in function.list_of_class_constraints:
                    self.send_constraint_to_mosek(constraint, task)
                
                if verbose:
                    print('\t\t function', function_counter, ':', len(function.list_of_class_constraints), 'scalar constraint(s) added')

                if len(function.list_of_class_psd) > 0:
                    if verbose:
                        print('\t\t function', function_counter, ':', 'Adding', len(function.list_of_class_psd), 'lmi constraint(s) ...')

                    for psd_counter, psd_matrix in enumerate(function.list_of_class_psd):
                        self.send_lmi_constraint_to_mosek(psd_counter, psd_matrix, task, verbose)

                    if verbose:
                        print('\t\t function', function_counter, ':', len(function.list_of_class_psd), 'lmi constraint(s) added')

            # Other function constraints
            self.talkative_function_constraints(verbose,len(list_of_functions_with_constraints))
            function_counter = 0
            for function in list_of_functions_with_constraints:
                function_counter += 1

                if len(function.list_of_constraints) > 0:
                    if verbose:
                        print('\t\t function', function_counter, ':', 'Adding', len(function.list_of_constraints),
                              'scalar constraint(s) ...')

                    for constraint in function.list_of_constraints:
                        self.send_constraint_to_mosek(constraint, task)

                    if verbose:
                        print('\t\t function', function_counter, ':', len(function.list_of_constraints),
                              'scalar constraint(s) added')

                if len(function.list_of_psd) > 0:
                    if verbose:
                        print('\t\t function', function_counter, ':', 'Adding', len(function.list_of_psd),
                              'lmi constraint(s) ...')

                    for psd_counter, psd_matrix in enumerate(function.list_of_psd):
                        self.send_lmi_constraint_to_mosek(psd_counter, psd_matrix, task, verbose)

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
                    self.send_constraint_to_mosek(constraint, task)
                if verbose:
                    print('\t\t partition', partition_counter, 'with', partition.get_nb_blocks(),
                          'blocks:', len(partition.list_of_constraints), 'scalar constraint(s) added')
                          
            # Solve the problem
            if verbose:
                print('(PEPit) Calling SDP solver')
            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.maximize)
            # Solve the problem and print summary
            task.optimize()
            if verbose >= 2:
                task.solutionsummary(mosek.streamtype.msg)
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)
            if verbose:
                print('(PEPit) Solver status: ', solsta, ' (solver: MOSEK);'
                      'objective value:{}'.format(task.getprimalobj(mosek.soltype.itr)))


            # Raise explicit error when wc_value in infinite
            if prosta == mosek.prosta.dual_infeas:
                raise UserWarning("PEPit didn't find any nontrivial worst-case guarantee. "
                                  "It seems that the optimal value of your problem is unbounded.")
            # Store the obtained value
            wc_value = task.getprimalobj(mosek.soltype.itr)
            Gram_value = self._get_Gram_from_mosek(task.getbarxj(mosek.soltype.itr, 0), Point.counter)
            xx = task.getxx(mosek.soltype.itr)
            F_value = xx
            tau_value = xx[-1]
            ##TODO: store explicit upper bound (dual obj)
            ##TODO: primal-dual associations (return dual vars)
            
            # Perform a dimension reduction if required
            if dimension_reduction_heuristic:

                # Print the estimated dimension before dimension reduction
                nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(Gram_value)
                if verbose:
                    print('(PEPit) Postprocessing: {} eigenvalue(s) > {} before dimension reduction'.format(nb_eigenvalues,
                                                                                                            eig_threshold))
                    print('(PEPit) Calling SDP solver')

                # Add the constraint that the objective stay close to its actual value
                self.send_constraint_to_mosek(tau >= wc_value - tol_dimension_reduction, task)

                # Translate the heuristic into cvxpy objective and solve the associated problem
                if dimension_reduction_heuristic == "trace":
                    task.putclist([tau.counter], [0.0])
                
                    A_i = np.arange(0,Point.counter)
                    A_j = A_i
                    A_val = np.ones((Point.counter,))
                    
                    sym_A = task.appendsparsesymmat(Point.counter,A_i,A_j,A_val) 
                    task.putbarcj(0,[sym_A],[-1.0]) #-1 here (we minimize)
                    
                    task.optimize()

                    # Store the actualized obtained value
                    xx = task.getxx(mosek.soltype.itr)
                    wc_value = xx[-1]

                    # Compute minimal number of dimensions
                    Gram_value = self._get_Gram_from_mosek(task.getbarxj(mosek.soltype.itr, 0), Point.counter)
                    nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(Gram_value)

                elif dimension_reduction_heuristic.startswith("logdet"):
                    niter = int(dimension_reduction_heuristic[6:])
                    task.putclist([tau.counter], [0.0])
                    for i in range(1, 1+niter):
                        W = np.linalg.inv(corrected_G_value + eig_regularization * np.eye(Point.counter))
                        No_zero_ele =np.argwhere(np.tril(W))
                        W_i = No_zero_ele[:,0]
                        W_j = No_zero_ele[:,1]
                        W_val = W[W_i, W_j]
                        sym_W = task.appendsparsesymmat(Point.counter,W_i,W_j,W_val)
                        task.putbarcj(0,[sym_W],[-1.0]) #-1 here (we minimize)
                        task.optimize()

                        # Store the actualized obtained value
                        xx = task.getxx(mosek.soltype.itr)
                        wc_value = xx[-1]

                        # Compute minimal number of dimensions
                        Gram_value = self._get_Gram_from_mosek(task.getbarxj(mosek.soltype.itr, 0), Point.counter)
                        nb_eigenvalues, eig_threshold, corrected_G_value = self.get_nb_eigenvalues_and_corrected_matrix(Gram_value)

                        # Print the estimated dimension after i dimension reduction steps
                        solsta = task.getsolsta(mosek.soltype.itr)
                        if verbose:
                            print('(PEPit) Solver status: ', solsta, ' (solver: MOSEK);'
                                  'objective value:{}'.format(task.getprimalobj(mosek.soltype.itr)))
                            print('(PEPit) Postprocessing: {} eigenvalue(s) > {} after {} dimension reduction step(s)'.format(
                                  nb_eigenvalues, eig_threshold, i))

                else:
                    raise ValueError("The argument \'dimension_reduction_heuristic\' must be \'trace\'"
                                     "or \`logdet\` followed by an interger."
                                     "Got {}".format(dimension_reduction_heuristic))

                # Print the estimated dimension after dimension reduction
                solsta = task.getsolsta(mosek.soltype.itr)
                if verbose:
                    print('(PEPit) Solver status: ', solsta, ' (solver: MOSEK);'
                          ' objective value: {}'.format(task.getprimalobj(mosek.soltype.itr)))
                    print('(PEPit) Postprocessing: {} eigenvalue(s) > {} after dimension reduction'.format(nb_eigenvalues,
                                                                                                       eig_threshold))

            # Store all the values of points and function values
            Gram_value = self._get_Gram_from_mosek(task.getbarxj(mosek.soltype.itr, 0), Point.counter)
            xx = task.getxx(mosek.soltype.itr)
            F_value = xx
            tau_value = xx[-1]
            wc_value = tau_value
            self._eval_points_and_function_values(F_value, Gram_value, verbose=verbose)

            # Store all the dual values in constraints
            #### TODOTODO
            #### self._eval_constraint_dual_values(dual_values)

            # Return the value of the minimal performance metric or the full cvxpy Problem object
            if return_full_mosek_problem:
                # Return the cvxpy Problem object
                return task
            else:
                # Return the value of the minimal performance metric
                return wc_value
            
            
                        

        
    #
    # EVALUATION PARTS
    #
    #
    
    @staticmethod
    def _verify_accuracy():
        """XXXX

        """
        # CHECK FEASIBILITY (primal & dual)
    	# CHECK PD GAP
    @staticmethod
    def _get_Gram_from_mosek(tril, size):
        # MOSEK returns:
        # the primal solution for a semidefinite variable. Only the lower triangular part of
        # is returned because the matrix by construction is symmetric. The format is that the columns are stored sequentially in the natural order.
        G = np.zeros((size,size))
        counter = 0
        for j in range(size):
            for i in range(size-j):
                G[j+i,j] = tril[counter]
                G[j,j+i] = tril[counter]
                counter += 1
        return G
 
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


        
    def talkative_performance_measure(self,verbose):
        """
        Todo
        
        """
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' performance measure is minimum of {} element(s)'.format(len(self.list_of_performance_metrics)))
        
    def talkative_description(self,verbose):
        """
        Todo
        
        """
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' size of the main PSD matrix: {}x{}'.format(Point.counter, Point.counter))
        
    def talkative_handling_constraints(self,verbose):
        """
        Todo
        
        """
        if verbose:
            print('(PEPit) Setting up the problem: Adding initial conditions and general constraints ...')
        
    def talkative_constraints(self,verbose):
        """
        Todo
        
        """
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' initial conditions and general constraints ({} constraint(s) added)'.format(len(self.list_of_constraints)))
        
    def talkative_LMIs(self,verbose,size):
        """
        Todo
        
        """
        if verbose:
                print('(PEPit) Setting up the problem: {} lmi constraint(s) added'.format(size))
        
    def talkative_functions(self,verbose,size):
        """
        Todo
        
        """
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' interpolation conditions for {} function(s)'.format(size))
        
    def talkative_function_constraints(self,verbose,size):
        """
        Todo
        
        """
        if verbose:
            print('(PEPit) Setting up the problem:'
                  ' constraints for {} function(s)'.format(size))
                  
MOSEK = {}
CVXPY = {}
