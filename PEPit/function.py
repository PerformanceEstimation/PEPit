import numpy as np
import pandas as pd

from PEPit.point import Point
from PEPit.expression import Expression
from PEPit.constraint import Constraint
from PEPit.psd_matrix import PSDMatrix

from PEPit.tools.dict_operations import merge_dict, prune_dict


class Function(object):
    """
    A :class:`Function` object encodes a function or an operator.

    Warnings:
        This class must be overwritten by a child class that encodes some conditions on the :class:`Function`.
        In particular, the method `add_class_constraints` must be overwritten.
        See the :class:`PEPit.functions` and :class:`PEPit.operators` modules.

    Some :class:`Function` objects are defined from scratch as **leaf** :class:`Function` objects,
    and some are linear combinations of pre-existing ones.

    Attributes:
        name (str): A name set through the set_name method. None is no name is given.
        _is_leaf (bool): True if self is defined from scratch.
                         False if self is defined as linear combination of leaves.
        decomposition_dict (dict): decomposition of self as linear combination of leaf :class:`Function` objects.
                                   Keys are :class:`Function` objects and values are their associated coefficients.
        reuse_gradient (bool): If True, the same subgradient is returned
                               when one requires it several times on the same :class:`Point`.
                               If False, a new subgradient is computed each time one is required.
        list_of_points (list): A list of triplets storing the points where this :class:`Function` has been evaluated,
                               as well as the associated subgradients and function values.
        list_of_stationary_points (list): The sublist of `self.list_of_points` of
                                          stationary points (characterized by some subgradient=0).
        list_of_constraints (list): The list of :class:`Constraint` objects associated with this :class:`Function`.
        list_of_psd (list): The list of :class:`PSDMatrix` objects associated with this :class:`Function`.
        list_of_class_constraints (list): The list of class interpolation :class:`Constraint` objects.
        list_of_class_psd (list): The list of :class:`PSDMatrix` objects associated with a class
                                  interpolation constraints.
        tables_of_constraints (dict): A dictionary containing all the constraints, sorted by table.
        counter (int): counts the number of **leaf** :class:`Function` objects.

    Note:
        PEPit was initially tough for evaluating performances of optimization algorithms.
        Operators are represented in the same way as functions, but function values must not be used (they don't have
        any sense in this framework). Use gradient to access an operator value.

    :class:`Function` objects can be added or subtracted together.
    They can also be multiplied and divided by a scalar value.

    Example:
        >>> func1 = Function()
        >>> func2 = Function()
        >>> new_func = (- func1 + func2) / 5

    """
    # Class counter.
    # It counts the number of functions defined from scratch.
    # The others are linear combination of those functions.
    counter = 0
    list_of_functions = list()

    def __init__(self,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False,
                 name=None,
                 ):
        """
        :class:`Function` objects can also be instantiated via the following arguments.

        Args:
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as a linear combination of leaves.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str, optional): name of the object. None by default.
                                  Can be updated later through the method `set_name`.

        Note:
            If `is_leaf` is True, then `decomposition_dict` must be provided as None.
            Then `self.decomposition_dict` will be set to `{self: 1}`.

        Note:
            `reuse_gradient` is typically set to True when this :class:`Function` is differentiable,
            that is there exists only one subgradient per :class:`Point`.

        Instantiating the :class:`Function` object of the first example can be done by

        Example:
            >>> func1 = Function()
            >>> func2 = Function()
            >>> new_func = Function(is_leaf=False, decomposition_dict = {func1: -1/5, func2: 1/5})

        """
        # Initialize name of the function
        self.name = name

        # Store inputs
        self._is_leaf = is_leaf
        self.reuse_gradient = reuse_gradient
        Function.list_of_functions.append(self)

        # If leaf function, the decomposition is updated,
        # the object counter is set
        # and the class counter updated.
        # Otherwise, the decomposition_dict is stored in an attribute and the object counter is set to None
        if is_leaf:
            assert decomposition_dict is None
            self.decomposition_dict = {self: 1}
            self.counter = Function.counter
            Function.counter += 1
        else:
            assert type(decomposition_dict) == dict
            self.decomposition_dict = decomposition_dict
            self.counter = None

        # Initialize list of points and constraints.
        # An optimal point will be stored in the 2 lists "list_of_stationary_points" and "list_of_points".
        self.list_of_stationary_points = list()
        self.list_of_points = list()
        self.list_of_constraints = list()
        self.list_of_psd = list()
        self.list_of_class_constraints = list()
        self.list_of_class_psd = list()

        # Initialize dictionary that will contain the tables of constraints
        self.tables_of_constraints = dict()

    def set_name(self, name):
        """
        Assign a name to self for easier identification purpose.

        Args:
            name (str): a name to be given to self.

        """
        self.name = name

    def get_name(self):
        """
        Returns (str): the attribute name.
        """
        return self.name

    def get_is_leaf(self):
        """

        Returns:
            self._is_leaf (bool): allows to access the protected attribute `_is_leaf`.

        """
        return self._is_leaf

    def __add__(self, other):
        """
        Add 2 :class:`Function` objects together, leading to a new :class:`Function` object.

        Args:
            other (Function): any other :class:`Function` object.

        Returns:
            self + other (Function): The sum of the 2 :class:`Function` objects.

        Raises:
            AssertionError: if provided `other` is not a :class:`Function`.

        """

        # Verify other is a function
        assert isinstance(other, Function)

        # Merge decomposition dicts of self and other
        merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)

        # Create and return the newly created function
        return Function(is_leaf=False,
                        decomposition_dict=merged_decomposition_dict,
                        reuse_gradient=self.reuse_gradient and other.reuse_gradient)

    def __sub__(self, other):
        """
        Subtract 2 :class:`Function` objects together, leading to a new :class:`Function` object.

        Args:
            other (Function): any other :class:`Function` object.

        Returns:
            self - other (Function): The difference between the 2 :class:`Function` objects.

        Raises:
            AssertionError: if provided `other` is not a :class:`Function`.

        """

        # A-B = A+(-B)
        return self.__add__(-other)

    def __neg__(self):
        """
        Compute the opposite of a :class:`Function`.

        Returns:
            - self (Function): the opposite of self

        """

        # -A = (-1)*A
        return self.__rmul__(other=-1)

    def __rmul__(self, other):
        """
        Multiply this :class:`Function` with a scalar value, leading to a new :class:`Function` object.

        Args:
            other (int or float): any scalar value.

        Returns:
            other * self (Function): The product of this function and `other`.

        Raises:
            AssertionError: if provided `other` is not a python scalar value.

        """

        # Verify other is a scalar constant
        assert isinstance(other, float) or isinstance(other, int)

        # Multiply uniformly self's decomposition_dict by other
        new_decomposition_dict = dict()
        for key, value in self.decomposition_dict.items():
            new_decomposition_dict[key] = value * other

        # Create and return the newly created Function
        return Function(is_leaf=False,
                        decomposition_dict=new_decomposition_dict,
                        reuse_gradient=self.reuse_gradient)

    def __mul__(self, other):
        """
        Multiply this :class:`Function` with a scalar value, leading to a new :class:`Function` object.

        Args:
            other (int or float): any scalar value.

        Returns:
            self * other (Function): The product of this function and `other`.

        Raises:
            AssertionError: if provided `other` is not a python scalar value.

        """
        return self.__rmul__(other=other)

    def __truediv__(self, denominator):
        """
        Divide this :class:`Function` with a scalar value, leading to a new :class:`Function` object.

        Args:
            denominator (int or float): any scalar value.

        Returns:
            self / denominator (Function): The ratio between of function and `other`.

        Raises:
            AssertionError: if provided `other` is not a python scalar value.

        """

        # P / v = P * (1/v)
        return self.__rmul__(other=1 / denominator)

    def add_constraint(self, constraint, name=None):
        """
        Store a new :class:`Constraint` to the list of constraints of this :class:`Function`.

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

        Raises:
            AssertionError: if provided matrix is not a square matrix.
            TypeError: if provided matrix does not contain only Expressions.

        """
        matrix = PSDMatrix(matrix_of_expressions=matrix_of_expressions)

        # Set name
        if name is not None:
            matrix.set_name(name=name)

        # Add constraint to the list of self's constraints
        self.list_of_psd.append(matrix)

    def add_constraints_from_one_list_of_points(self, list_of_points,
                                                constraint_name, set_class_constraint_i):
        """
        Add a class constraint of one input on all the points of `list_of_points`.
        Creates a table corresponding to this set of constraints.

        Args:
            list_of_points (list): the list of points the constraint apply on.
            constraint_name (str): generic name for this constraint.
            set_class_constraint_i (Callable): a function that takes a point in input and returns a :class:`Constraint`.

        """

        # Set function ID
        function_id = self.get_name()
        if function_id is None:
            function_id = "Function_{}".format(self.counter)

        # Initialize table of constraints
        table_of_constraints = list()

        # Browse list of points and create interpolation constraints
        for i, point_i in enumerate(list_of_points):

            xi, gi, fi = point_i
            xi_id = xi.get_name()
            if xi_id is None:
                xi_id = "Point_{}".format(i)

            # Create interpolation constraint
            constraint = set_class_constraint_i(xi, gi, fi)

            # Set name to newly created constraint
            constraint.set_name("IC_{}_{}({})".format(function_id, constraint_name, xi_id))

            # Add constraint to the table of constraints
            table_of_constraints.append(constraint)

            # Add constraint to the list of class constraints
            self.list_of_class_constraints.append(constraint)

        # Complete table of constraints
        table_of_constraints = np.array(table_of_constraints).reshape(1, -1)
        point_names = [point[0].name or "Point_{}".format(point_index) for point_index, point in
                       enumerate(list_of_points)]
        if table_of_constraints.shape != (0,):
            df = pd.DataFrame(table_of_constraints, columns=point_names)
            df.columns.name = "IC_{}".format(function_id)

            # Add the table of constraints to the attribute tables_of_constraints
            self.tables_of_constraints[constraint_name] = df

    def add_constraints_from_two_lists_of_points(self, list_of_points_1, list_of_points_2,
                                                 constraint_name, set_class_constraint_i_j,
                                                 symmetry=False):
        """
        Add a class constraint of two inputs on all the couple of points of
        `list_of_points_1` :math:`\\times` `list_of_points_2`.
        Creates a table corresponding to this set of constraints.

        Args:
            list_of_points_1 (list): the first list of points the grid is generated from.
            list_of_points_2 (list): the second list of points the grid is generated from.
            constraint_name (str): generic name for this constraint.
            set_class_constraint_i_j (Callable): a function that takes two points in input
                                                 and returns a :class:`Constraint`.
            symmetry (bool, optional): A boolean specifying if the constraint function is symmetric or not.
                                       If so, the number of constraints is divided by 2.
                                       Set to False by default.

        """

        # Set function ID
        function_id = self.get_name()
        if function_id is None:
            function_id = "Function_{}".format(self.counter)

        # Initialize table of constraints
        table_of_constraints = list()

        # Browse list of points and create interpolation constraints
        for i, point_i in enumerate(list_of_points_1):

            xi, gi, fi = point_i
            xi_id = xi.get_name()
            if xi_id is None:
                xi_id = "Point_{}".format(i)

            # Initialize row of constraints
            row_of_constraints = list()

            for j, point_j in enumerate(list_of_points_2):

                xj, gj, fj = point_j
                xj_id = xj.get_name()
                if xj_id is None:
                    xj_id = "Point_{}".format(j)

                if i == j or (i > j and symmetry):
                    row_of_constraints.append(0)

                else:
                    # Create interpolation constraint
                    constraint = set_class_constraint_i_j(xi, gi, fi,
                                                          xj, gj, fj,
                                                          )

                    # Set name to newly created constraint
                    constraint.set_name("IC_{}_{}({}, {})".format(function_id, constraint_name, xi_id, xj_id))

                    # Add constraint to the row of constraints
                    row_of_constraints.append(constraint)

                    # Add constraint to the list of class constraints
                    self.list_of_class_constraints.append(constraint)

            # Add row of constraints to the table of constraints
            table_of_constraints.append(row_of_constraints)

        # Complete table of constraints
        table_of_constraints = np.array(table_of_constraints)
        point_names_1 = [point[0].name or "Point_{}".format(point_index) for point_index, point in
                         enumerate(list_of_points_1)]
        point_names_2 = [point[0].name or "Point_{}".format(point_index) for point_index, point in
                         enumerate(list_of_points_2)]
        if table_of_constraints.shape != (0,):
            df = pd.DataFrame(table_of_constraints, columns=point_names_2, index=point_names_1)
            df.columns.name = "IC_{}".format(function_id)

            # Add the table of constraints to the attribute tables_of_constraints
            self.tables_of_constraints[constraint_name] = df

    def get_class_constraints_duals(self):
        """
        This method browses all the tables of :class:`Constraint`s generated by the `add_class_constraints` method,
        grab the dual variables values associated with the constraints and return a dictionary with all the dual tables.

        Returns:
             dict: a dictionary which keys are the names of the generic constraints provided to the methods
                   `add_constraints_from_one_list_of_points` and `add_constraints_from_two_lists_of_points`,
                   and the keys are pandas.DataFrames containing the dual values associated with the constraints.

        """

        # Initialize the dictionary.
        tables_of_duals = dict()

        # Browse the tables.
        for key, table in self.tables_of_constraints.items():

            # Create a table of dual values from the current table of constraints.
            new_table = []

            # Browse the table of constraints.
            for index, row in table.iterrows():
                new_row = []
                for element in row:
                    # If there is an actual constraint, return its dual value.
                    if isinstance(element, Constraint):
                        new_row.append(element.eval_dual())
                    # If there is a scalar number, simply return it.
                    elif isinstance(element, float) or isinstance(element, int):
                        new_row.append(element)
                    else:
                        raise TypeError("The elements of table of constraints must either be Constraints objects"
                                        " or scalar values. Got {}.".format(type(element)))
                new_table.append(new_row)

            # Transform the table into a pandas.DataFrame.
            new_table = np.array(new_table)
            df = pd.DataFrame(new_table, columns=table.columns, index=table.index)

            # Add a name to the pandas.DataFrame.
            df.columns.name = table.columns.name

            # Store into the dict to be returned.
            tables_of_duals[key] = df

        # Return the dictionary of tables of dual values.
        return tables_of_duals

    def set_class_constraints(self):
        """
        This method is run by the :class:`PEP` just before solving the problem.
        It reinitializes the list_of_class_constraints attributes before filling it.

        """
        self.list_of_class_constraints = list()
        self.add_class_constraints()

    def add_class_constraints(self):
        """
        Warnings:
            Needs to be overwritten with interpolation conditions (or necessary conditions for interpolation for
            obtaining possibly non-tight upper bounds on the worst-case performance).

        This method is run by the :class:`PEP` just before solving the problem.
        It evaluates interpolation conditions for the 2 lists of points that is stored in this :class:`Function`.

        Raises:
            NotImplementedError: This method must be overwritten in children classes

        """

        raise NotImplementedError("This method must be overwritten in children classes")

    def _is_already_evaluated_on_point(self, point):
        """
        Check whether this :class:`Function` is already evaluated on the :class:`Point` "point" or not.

        Args:
            point (Point): the point we want to check whether the function is evaluated on or not.

        Returns:
            tuple or None: if "self" is evaluated on "point",
                           return the tuple "gradient, function value" associated to "point".
                           Otherwise, return None.

        """

        # Browse the list of point "self" has been evaluated on
        for triplet in self.list_of_points:
            if triplet[0].decomposition_dict == point.decomposition_dict:
                # If "self" has been evaluated on "point", then break the loop and return its corresponding data
                return triplet[1:]

        # If "self" has not been evaluated on "point" yet, then return None.
        return None

    def _separate_leaf_functions_regarding_their_need_on_point(self, point):
        """
        Separate leaf functions in 3 categories depending whether they need new evaluation or not of
        gradient and function value on the :class:`Point` "point".

        Args:
            point (Point): the point we look at

        Returns:
            tuple of lists: 3 lists or functions arranged with respect to their need.
                            Note functions are returned with their corresponding weight in the decomposition of self.

        """

        # Initialize the 3 lists
        list_of_functions_which_need_nothing = list()
        list_of_functions_which_need_gradient_only = list()
        list_of_functions_which_need_gradient_and_function_value = list()

        # Separate all leaf function in 3 categories based on their need
        for function, weight in self.decomposition_dict.items():
            # If function has already been evaluated on point, then one should reuse some evaluation.
            # Note the method "is_already_evaluated_on_point" returns a non-empty tuple if the function has already
            # been evaluated, and None otherwise.
            # Those outputs are respectively evaluated as True and False by the following test.
            if function._is_already_evaluated_on_point(point=point):

                # If function is differentiable, one should keep both previous gradient and previous function value.
                if function.reuse_gradient:
                    list_of_functions_which_need_nothing.append((function, weight))

                # If function is not differentiable, one should keep only previous function value.
                else:
                    list_of_functions_which_need_gradient_only.append((function, weight))

            # If function has not been evaluated on point, then it need new gradient and function value
            else:
                list_of_functions_which_need_gradient_and_function_value.append((function, weight))

        # Return the 3 lists in a specific order: from the smallest need to the biggest one.
        return list_of_functions_which_need_nothing, list_of_functions_which_need_gradient_only,\
            list_of_functions_which_need_gradient_and_function_value

    def add_point(self, triplet):
        """
        Add a triplet (point, gradient, function_value) to the list of points of this function.

        Args:
            triplet (tuple): A tuple containing 3 elements:
                             point (:class:`Point`),
                             gradient (:class:`Point`),
                             and function value (:class:`Expression`).

        """

        # Unpack triplet
        point, g, f = triplet

        # Verify the type of each element
        assert isinstance(point, Point)
        assert isinstance(g, Point)
        assert isinstance(f, Expression)

        # Prune the decomposition dict of each element to verify if the point is optimal or not by testing gradient=0.
        for element in triplet:
            element.decomposition_dict = prune_dict(element.decomposition_dict)

        # Store the point in list_of_points
        self.list_of_points.append(triplet)

        # If gradient==0, then store the point in list_of_optimal_points too
        if g.decomposition_dict == dict():
            self.list_of_stationary_points.append(triplet)

        # If self is not a leaf function, create gradient and function value for each of the latest
        # and combine under the constraint that the full gradient and function value is fixed.
        if not self._is_leaf:

            # Remove the leaf functions that are not really involved in the decomposition of self
            # to avoid division by zero at the end.
            self.decomposition_dict = prune_dict(self.decomposition_dict)

            # Separate all leaf function in 3 categories based on their need
            tuple_of_lists_of_functions = self._separate_leaf_functions_regarding_their_need_on_point(point=point)

            # Reduce into 2 lists according to the fact a function needs something of not
            list_of_functions_which_need_nothing = tuple_of_lists_of_functions[0]
            list_of_functions_which_need_something = tuple_of_lists_of_functions[1] + tuple_of_lists_of_functions[2]

            # If no function needs something, we are done!
            # If any function needs something, we build it
            if list_of_functions_which_need_something != list():

                # Store the number of such leaf function and the gradient and function value of self on point
                total_number_of_involved_leaf_functions = len(self.decomposition_dict.keys())
                gradient_of_last_leaf_function = g
                value_of_last_leaf_function = f
                number_of_currently_computed_gradients_and_values = 0

                for function, weight in list_of_functions_which_need_nothing + list_of_functions_which_need_something:

                    # Keep track of the gradient and function value of self minus those from visited leaf functions
                    if number_of_currently_computed_gradients_and_values < total_number_of_involved_leaf_functions - 1:
                        grad, val = function.oracle(point)
                        gradient_of_last_leaf_function = gradient_of_last_leaf_function - weight * grad
                        value_of_last_leaf_function = value_of_last_leaf_function - weight * val

                        number_of_currently_computed_gradients_and_values += 1

                    # The latest function must receive fully conditioned gradient and function value
                    else:
                        gradient_of_last_leaf_function = gradient_of_last_leaf_function / weight
                        value_of_last_leaf_function = value_of_last_leaf_function / weight

                        function.add_point((point, gradient_of_last_leaf_function, value_of_last_leaf_function))

    def oracle(self, point):
        """
        Return a gradient (or a subgradient) and the function value of self evaluated at `point`.

        Args:
            point (Point): any point.

        Returns:
            tuple: a (sub)gradient (:class:`Point`) and a function value (:class:`Expression`).

        """

        # Verify point is a Point
        assert isinstance(point, Point)

        # If those values already exist, simply return them.
        # If not, instantiate them before returning.
        # Note if the non-differentiable case, the gradient is recomputed anyway.
        associated_grad_and_function_val = self._is_already_evaluated_on_point(point=point)
        # "associated_grad_and_function_val" is a tuple (True) or None (False)

        # If "self" has already been evaluated on "point" and is differentiable,
        # then break the loop and return the previously computed values.
        if associated_grad_and_function_val and self.reuse_gradient:
            return associated_grad_and_function_val

        # If "self" has already been evaluated on "point" but is not differentiable,
        # then the function value is fixed by the previously computed one,
        # but a new sub-gradient remains to be defined
        if associated_grad_and_function_val and not self.reuse_gradient:
            f = associated_grad_and_function_val[-1]

        # Here we separate the list of leaf functions according to their needs
        list_of_functions_which_need_nothing, list_of_functions_which_need_gradient_only,\
            list_of_functions_which_need_gradient_and_function_value =\
            self._separate_leaf_functions_regarding_their_need_on_point(point=point)

        # If "self" has not been evaluated on "point" yet, then we need to compute new gradient and function value
        # Here we deal with the function value computation
        if associated_grad_and_function_val is None:

            # If no leaf function need a new function value, it means that they all have one,
            # and then "self" 's one is determined by linear combination.
            if list_of_functions_which_need_gradient_and_function_value == list():
                f = Expression(is_leaf=False, decomposition_dict=dict())
                for function, weight in self.decomposition_dict.items():
                    f += weight * function.value(point=point)

            # Otherwise, we define a new one.
            else:
                f = Expression(is_leaf=True, decomposition_dict=None)

        # Here we deal with the gradient computation.
        # We come to this point if "self" need a new gradient,
        # either because it is not differentiable or because it had never been computed so far.

        # If "self" gradient is determined by its leaf functions, then we compute it.
        if list_of_functions_which_need_gradient_and_function_value == list()\
                and list_of_functions_which_need_gradient_only == list():
            g = Point(is_leaf=False, decomposition_dict=dict())
            for function, weight in self.decomposition_dict.items():
                g += weight * function.gradient(point=point)

        # Otherwise, we create a new one.
        else:
            g = Point(is_leaf=True, decomposition_dict=None)

        # Store it
        self.add_point(triplet=(point, g, f))

        # Return gradient and function value
        return g, f

    def gradient(self, point, name=None):
        """
        Return the gradient (or a subgradient) of this :class:`Function` evaluated at `point`.

        Args:
            point (Point): any point.
            name (str, optional): name of the object. Not overwriting is None. None by default.

        Returns:
            Point: a gradient (:class:`Point`) of this :class:`Function` on point (:class:`Point`).

        Note:
            the method subgradient does the exact same thing.

        """

        return self.subgradient(point, name=name)

    def subgradient(self, point, name=None):
        """
        Return a subgradient of this :class:`Function` evaluated at `point`.

        Args:
            point (Point): any point.
            name (str, optional): name of the object. Not overwriting is None. None by default.

        Returns:
            Point: a subgradient (:class:`Point`) of this :class:`Function` on point (:class:`Point`).

        Note:
            the method gradient does the exact same thing.

        """

        # Verify point is a Point
        assert isinstance(point, Point)

        # Call oracle but only return the gradient
        g, _ = self.oracle(point)

        # Set name
        if name is not None:
            g.set_name(name=name)

        return g

    def value(self, point, name=None):
        """
        Return the function value of this :class:`Function` on point.

        Args:
            point (Point): any point.
            name (str, optional): name of the object. Not overwriting is None. None by default.

        Returns:
            Point: the function value (:class:`Expression`) of this :class:`Function` on point (:class:`Point`).

        """

        # Verify point is a Point
        assert isinstance(point, Point)

        # Check whether "self" has already been evaluated on "point"
        associated_grad_and_function_val = self._is_already_evaluated_on_point(point=point)

        # "associated_grad_and_function_val" is a tuple (True) or None (False)
        if associated_grad_and_function_val:
            # If the value already exist, simply return it
            f = associated_grad_and_function_val[-1]
        else:
            # Otherwise, call oracle but only return the function value
            _, f = self.oracle(point)

        # Set name
        if name is not None:
            f.set_name(name=name)

        # Return the function value
        return f

    def __call__(self, point):
        """
        Call the method `value` on point.

        Args:
            point (Point): any point.

        Returns:
            Point: the function value (:class:`Expression`) of this :class:`Function` on point (:class:`Point`).

        """

        # Call the method value on point
        return self.value(point=point)

    def stationary_point(self, return_gradient_and_function_value=False, name=None):
        """
        Create a new stationary point, as well as its zero gradient and its function value.

        Args:
            return_gradient_and_function_value (bool): if True, return the triplet point (:class:`Point`),
                                                       gradient (:class:`Point`), function value (:class:`Expression`).
                                                       Otherwise, return only the point (:class:`Point`).
            name (str, optional): name of the object. Not overwriting is None. None by default.

        Returns:
            Point or tuple: an optimal point

        """

        # Create a new point, null gradient and new function value
        point = Point(is_leaf=True, decomposition_dict=None)
        g = Point(is_leaf=False, decomposition_dict=dict())
        f = Expression(is_leaf=True, decomposition_dict=None)

        # Add the triplet to the list of points of the function as well as to its list of stationary points
        self.add_point((point, g, f))

        # Set name
        if name is not None:
            point.set_name(name=name)

        # Return the required information
        if return_gradient_and_function_value:
            return point, g, f
        else:
            return point

    def fixed_point(self, name=None):

        """
        This routine outputs a fixed point of this function, that is :math:`x` such that :math:`x\\in\\partial f(x)`.
        If self is an operator :math:`A`, the fixed point is such that :math:`Ax = x`.

        Args:
            name (str, optional): name of the object. Not overwriting is None. None by default.

        Returns:
            x (Point): a fixed point of the differential of self.
            x (Point): \\nabla f(x) = x.
            fx (Expression): a function value (useful only if self is a function).

        """

        # Define a point and function value
        x = Point()
        fx = Expression()

        # Add triplet to self's list of points (by definition gx = x)
        self.add_point((x, x, fx))

        # Set name
        if name is not None:
            x.set_name(name=name)

        # Return the aforementioned triplet
        return x, x, fx
