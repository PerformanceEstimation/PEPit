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
        list_of_conditions (list): list of :class:`Constraint` objects that are defined out of the scope of a :class:`Function`.
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
        # points and conditions that are independent of the functions,
        # as well as the list of performance metric.
        # The PEP will maximize the minimum of the latest.
        self.list_of_functions = list()
        self.list_of_points = list()
        self.list_of_conditions = list()
        self.list_of_performance_metrics = list()

    def declare_function(self, function_class, param, reuse_gradient=None):
        """
        Instantiate a leaf :class:`Function` and store it in the attribute `list_of_functions`.

        Args:
            function_class (class): a subclass of :class:`Function` that overwrite the `add_class_constraints` method.
            param (dict): dictionary of parameters that characterize the function class.
            reuse_gradient (bool): whether the function only admit one gradient on a given :class:`Point`.
                                   Typically True when the function is assumed differentiable.

        Returns:
            f (Function): the newly created function.

        """

        # Create the function
        if reuse_gradient is None:
            f = function_class(param, is_leaf=True, decomposition_dict=None)
        else:
            f = function_class(param, is_leaf=True, decomposition_dict=None, reuse_gradient=reuse_gradient)

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
        Store a :class:`Constraint` in the attribute `list_of_conditions`.
        Typically an initial condition.

        Args:
            condition (Constraint): the given constraint.

        """

        # Store condition in the appropriate list
        self.list_of_conditions.append(condition)

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

    def solve(self, solver=None, verbose=1, tracetrick=False, return_full_cvxpy_problem=False):
        """
        Transform the :class:`PEP` under the SDP form, and solve it.

        Args:
            solver (str or None): The name of the underlying solver.
            verbose (int): Level of information details to print (0 or 1)
            tracetrick (bool): Apply trace heuristic as a proxy for minimizing
                               the dimension of the solution (rank of the Gram matrix).
            return_full_cvxpy_problem (bool): If True, return the cvxpy Problem object.
                                              If False, return the worst case value only.
                                              Set to False by default.

        Returns:
            float or cp.Problem: Value of the performance metric of cp.Problem object corresponding to the SDP.
                                 The value only is returned by default.

        """

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

        # Defining initial conditions
        for condition in self.list_of_conditions:
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
                  ' initial conditions ({} constraint(s) added)'.format(len(self.list_of_conditions)))

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

        # Create the cvxpy problem
        if verbose:
            print('(PEPit) Compiling SDP')
        prob = cp.Problem(objective=cp.Maximize(objective), constraints=constraints_list)

        # Solve it
        if verbose:
            print('(PEPit) Calling SDP solver')
        prob.solve(solver=solver)
        if verbose:
            print('(PEPit) Solver status: {} (solver: {}); optimal value: {}'.format(prob.status,
                                                                                      prob.solver_stats.solver_name,
                                                                                      prob.value))

        wc_value = prob.value
        if tracetrick:
            eig_threshold = 1e-5
            if verbose:
                eig_val, _ = np.linalg.eig(G.value)
                nb_eigen = len([element for element in eig_val if element > eig_threshold])
                print('(PEPit) Postprocessing: applying trace heuristic.'
                      ' Currently {} eigenvalue(s) > {} before resolve.'.format(nb_eigen, eig_threshold))
                print('(PEPit) Calling SDP solver')
            tol_tracetrick = 1e-5
            constraints_list.append(objective >= wc_value - tol_tracetrick)
            prob = cp.Problem(objective=cp.Minimize(cp.trace(G)), constraints=constraints_list)
            prob.solve(solver=solver)
            wc_value = objective.value[0]
            if verbose:
                print('(PEPit) Solver status: {} (solver: {});'
                      ' objective value: {}'.format(prob.status,
                                                    prob.solver_stats.solver_name,
                                                    wc_value))
                eig_val, _ = np.linalg.eig(G.value)
                nb_eigen = len([element for element in eig_val if element > eig_threshold])
                print('(PEPit) Postprocessing: {} eigenvalue(s) > {} after trace heuristic'.format(nb_eigen,
                                                                                                    eig_threshold))

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
                print("\033[93m(PEPit) Postprocessing: solver\'s output is not entirely feasible"
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
        for condition in self.list_of_conditions:
            condition._dual_variable_value = cvx_constraints[counter].dual_value
            counter += 1

        # Store all the class constraints dual values, providing the proof of the desired rate.
        for function in self.list_of_functions:
            for constraint in function.list_of_constraints:
                constraint._dual_variable_value = cvx_constraints[counter].dual_value
                counter += 1

        # Return the position of the reached performance metric
        return position_of_minimal_objective
