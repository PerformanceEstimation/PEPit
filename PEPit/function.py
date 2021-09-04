from PEPit.Tools.dict_operations import merge_dict, prune_dict
from PEPit.point import Point
from PEPit.expression import Expression


class Function(object):
    """
    Function or Operator
    """
    # Class counter.
    # It counts the number of functions defined from scratch.
    # The others are linear combination of those functions.
    counter = 0

    def __init__(self,
                 is_leaf=True,
                 decomposition_dict=None,
                 is_differentiable=False,
                 ):
        """
        A function is a linear combination of basis functions.

        :param is_leaf: (bool) If True, it is a basis function. Otherwise it is a linear combination of such functions.
        :param decomposition_dict: (dict) Decomposition in the basis of functions.
        :param is_differentiable: (bool) If true, the function can have only one subgradient per point.
        """

        # Store inputs
        self._is_leaf = is_leaf
        self.is_differentiable = is_differentiable

        # If basis function, the decomposition is updated,
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
        # An optimal point will be stored in the 2 lists "list_of_optimal_points" and "list_of_points".
        self.list_of_optimal_points = list()
        self.list_of_points = list()
        self.list_of_constraints = list()

    def __add__(self, other):
        """
        Add 2 functions together, leading to a new function.

        :param other: (Function) Any other function
        :return: (Function) The sum of the 2 functions
        """

        # Verify other is a function
        assert isinstance(other, Function)

        # Merge decomposition dicts of self and other
        merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)

        # Create and return the newly created function
        return Function(is_leaf=False,
                        decomposition_dict=merged_decomposition_dict,
                        is_differentiable=self.is_differentiable and other.is_differentiable)

    def __sub__(self, other):
        """
        Subtract 2 functions together, leading to a new function.

        :param other: (Function) Any other function
        :return: (Function) The difference between the 2 functions
        """

        # A-B = A+(-B)
        return self.__add__(-other)

    def __neg__(self):
        """
        Compute the opposite of a function.

        :return: (Function) - function
        """

        # -A = (-1)*A
        return self.__rmul__(other=-1)

    def __rmul__(self, other):
        """
        Multiply a function by a scalar value

        :param other: (int or float) Any scalar constant
        :return: (Function) other * self
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
                        is_differentiable=self.is_differentiable)

    def __mul__(self, other):
        """
        Multiply a function by a scalar value

        :param other: (int or float) Any scalar constant
        :return: (Function) self * other
        """
        return self.__rmul__(other=other)

    def __truediv__(self, denominator):
        """
        Divide a function by a scalar value

        :param denominator: (int or float) the value to divide by.
        :return: (Function) The resulting function
        """

        # P / v = P * (1/v)
        return self.__rmul__(other=1 / denominator)

    def add_constraint(self, constraint):
        """
        Add a constraint to the list of constraints of the function

        :param constraint: (Constraint) typically resulting from an inequality between 2 expressions.
        """

        self.list_of_constraints.append(constraint)

    def add_class_constraints(self):
        """
        Needs to be overwritten with interpolation conditions.
        This methods is run by the PEP just before solving,
        applying the interpolation condition from the 2 lists of points.
        """

        raise NotImplementedError

    def is_already_evaluated_on_point(self, point):
        """
        Check whether the "self" function is already evaluated on the point "point" or not.

        :param point: (Point) the point we want to check whether the function is evaluated on or not.
        :return: (tuple or None) if "self" is evaluated on "point",
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

    def separate_basis_functions_regarding_their_need_on_point(self, point):
        """
        Separate basis functions in 3 categories depending whether they need new evaluation or not of
        gradient and function value on the point "point".

        :param point: (Point) the point we look at
        :return: (tuple of lists) 3 lists or functions arranged with respect to their need.
                                  Note functions are returned with their corresponding weight in the decomposition of self.
        """

        # Initialize the 3 lists
        list_of_functions_which_need_nothing = list()
        list_of_functions_which_need_gradient_only = list()
        list_of_functions_which_need_gradient_and_function_value = list()

        # Separate all basis function in 3 categories based on their need
        for function, weight in self.decomposition_dict.items():
            # If function has already been evaluated on point, then one should reuse some evaluation.
            # Note the method "is_already_evaluated_on_point" returns a non empty tuple if the function has already
            # been evaluated, and None otherwise.
            # Those outputs are respectively evaluated as True and False by the following test.
            if function.is_already_evaluated_on_point(point=point):

                # If function is differentiable, one should keep both previous gradient and previous function value.
                if function.is_differentiable:
                    list_of_functions_which_need_nothing.append((function, weight))

                # If function is not differentiable, one should keep only previous function value.
                else:
                    list_of_functions_which_need_gradient_only.append((function, weight))

            # If function has not been evaluated on point, then it need new gradient and function value
            else:
                list_of_functions_which_need_gradient_and_function_value.append((function, weight))

        # Return the 3 lists in a specific order: from the smallest need to the biggest one.
        return list_of_functions_which_need_nothing, list_of_functions_which_need_gradient_only, list_of_functions_which_need_gradient_and_function_value

    def add_point(self, triplet):
        """
        Add a triplet (point, gradient, function_value) to the list of points of this function.

        :param triplet: (tuple) A tuple containing 3 elements: point, gradient, and function value
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
            self.list_of_optimal_points.append(triplet)

        # If self is not a basis function, create gradient and function value for each of the latest
        # and combine under the constraint that the full gradient and function value is fixed.
        if not self._is_leaf:

            # Remove the basis functions that are not really involved in the decomposition of self
            # to avoid division by zero at the end.
            self.decomposition_dict = prune_dict(self.decomposition_dict)

            # Separate all basis function in 3 categories based on their need
            tuple_of_lists_of_functions = self.separate_basis_functions_regarding_their_need_on_point(point=point)

            # Store the number of such basis function and the gradient and function value of self on point
            total_number_of_involved_basis_functions = len(self.decomposition_dict.keys())
            gradient_of_last_basis_function = g
            value_of_last_basis_function = f
            number_of_currently_computed_gradients_and_values = 0

            for list_of_functions in tuple_of_lists_of_functions:
                for function, weight in list_of_functions:

                    # Keep track of the gradient and function value of self minus those from visited basis functions
                    if number_of_currently_computed_gradients_and_values < total_number_of_involved_basis_functions - 1:
                        grad, val = function.oracle(point)
                        gradient_of_last_basis_function = gradient_of_last_basis_function - weight * grad
                        value_of_last_basis_function = value_of_last_basis_function - weight * val

                        number_of_currently_computed_gradients_and_values += 1

                    # The latest function must receive fully conditioned gradient and function value
                    else:
                        gradient_of_last_basis_function = gradient_of_last_basis_function / weight
                        value_of_last_basis_function = value_of_last_basis_function / weight

                        function.add_point((point, gradient_of_last_basis_function, value_of_last_basis_function))

    def oracle(self, point):
        """
        Return the gradient and the function value of self in point

        :param point: (Point) Any point
        :return: (tuple) A gradient and a function value
        """

        # Verify point is a Point
        assert isinstance(point, Point)

        # If those values already exist, simply return them.
        # If not, instantiate them before returning.
        # Note if the non differentiable case, the gradient is recomputed anyway.
        associated_grad_and_function_val = self.is_already_evaluated_on_point(point=point)
        # "associated_grad_and_function_val" is a tuple (True) or None (False)

        # If "self" has already been evaluated on "point" and is differentiable,
        # then break the loop and return the previously computed values.
        if associated_grad_and_function_val and self.is_differentiable:
            return associated_grad_and_function_val

        # If "self" has already been evaluated on "point" but is not differentiable,
        # then the function value is fixed by the previously computed one,
        # but a new sub-gradient remains to be defined
        if associated_grad_and_function_val and not self.is_differentiable:
            f = associated_grad_and_function_val[-1]

        # Here we separate the list of basis functions according to their needs
        list_of_functions_which_need_nothing, list_of_functions_which_need_gradient_only, list_of_functions_which_need_gradient_and_function_value = self.separate_basis_functions_regarding_their_need_on_point(point=point)

        # If "self" has not been evaluated on "point" yet, then we need to compute new gradient and function value
        # Here we deal with the function value computation
        if associated_grad_and_function_val is None:

            # If no basis function need a new function value, it means that they all have one,
            # and then "self"'s one is determined by linear combination.
            if list_of_functions_which_need_gradient_and_function_value == list():
                f = Expression(is_function_value=False, decomposition_dict=dict())
                for function, weight in self.decomposition_dict.items():
                    f += weight * function.value(point=point)

            # Otherwise, we define a new one.
            else:
                f = Expression(is_function_value=True, decomposition_dict=None)

        # Here we deal with the gradient computation.
        # We come to this point if "self" need a new gradient,
        # either because it is not differentiable or because it had never been computed so far.

        # If "self" function value is determined by its basis functions, then we compute it.
        if list_of_functions_which_need_gradient_and_function_value == list() and list_of_functions_which_need_gradient_only == list():
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

    def gradient(self, point):
        """
        Return the gradient of self in point.

        :param point: (Point) Any point
        :return: (Point) The gradient of self in point
        """

        return self.subgradient(point)

    def subgradient(self, point):
        """
        Return the gradient of self in point.

        :param point: (Point) Any point
        :return: (Point) The gradient of self in point
        """

        # Call oracle but only return the gradient
        g, _ = self.oracle(point)

        return g

    def value(self, point):
        """
        Return the function value of self in point.

        :param point: (Point) Any point
        :return: (Point) The function value of self in point
        """

        # Call oracle but only return the function value
        _, f = self.oracle(point)

        return f

    def optimal_point(self, return_gradient_and_function_value=False):
        """
        Create a new optimal point, as well as its null gradient and its function value

        :param return_gradient_and_function_value: (bool) If True, return the triplet point, gradient, function value.
                                                          Otherwise, return only the point.
        :return: (Point or tuple) The optimal point
        """

        # Create a new point, null gradient and new function value
        point = Point(is_leaf=True, decomposition_dict=None)
        g = Point(is_leaf=False, decomposition_dict=dict())
        f = Expression(is_function_value=True, decomposition_dict=None)

        # Add the triplet to the list of points of the function as well as to its list of optimal points
        self.add_point((point, g, f))

        # Return the required information
        if return_gradient_and_function_value:
            return point, g, f
        else:
            return point
