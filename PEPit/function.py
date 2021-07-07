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
                 is_differentiable=True,
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
            self.counter = self.__class__.counter
            self.__class__.counter += 1
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
        assert other.__class__ == Function

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
        assert type(other) in {int, float}

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

    def add_point(self, triplet):
        """
        Add a triplet (point, gradient, function_value) to the list of points of this function.

        :param triplet: (tuple) A tuple containing 3 elements: point, gradient, and function value
        """

        # Verify the type of each element
        assert triplet[0].__class__ == Point
        assert triplet[1].__class__ == Point
        assert triplet[2].__class__ == Expression

        # Prune the decomposition dict of each element to verify if the point is optimal or not by testing gradient=0.
        for element in triplet:
            element.decomposition_dict = prune_dict(element.decomposition_dict)

        # Store the point in list_of_points
        self.list_of_points.append(triplet)

        # If gradient==0, then store the point in list_of_optimal_points too
        if triplet[1].decomposition_dict == dict():
            self.list_of_optimal_points.append(triplet)

    def add_constraint(self, constraint):
        """
        Add a constraint to the list of constraints of the function

        :param constraint: (Expression) typically resulting from an inequality between 2 expressions.
        """

        self.list_of_constraints.append(constraint)

    def add_class_constraints(self):
        """
        Needs to be overwritten with interpolation conditions.
        This methods is run by the PEP just before solving,
        applying the interpolation condition from the 2 lists of points.
        """

        raise NotImplementedError

    def oracle(self, point):
        """
        Return the gradient and the function value of self in point

        :param point: (Point) Any point
        :return: (tuple) A gradient and a function value
        """

        # Verify point is a Point
        assert point.__class__ == Point

        # If those values already exist, simply return them.
        # If not, instantiate them before returning.
        # Note if the non differentiable case, the gradient is recomputed anyway.
        for associated_point in self.list_of_points:
            if associated_point[0].decomposition_dict == point.decomposition_dict:
                if self.is_differentiable:
                    return associated_point[1], associated_point[2]
                else:
                    g = self.define_new_gradient_only(point)
                    self.add_point((point, g, associated_point[2]))
                    return g, associated_point[2]

        g, f = self.define_new_gradient_and_value(point)

        # Store the point
        self.add_point((point, g, f))

        # Return gradient and function value
        return g, f

    def define_new_gradient_only(self, point):
        """
        Define a new gradient

        :param point: (Point) Anny point
        :return: (Point) newly computed gradient
        """

        # If the function a basis one, then compute a simple gradient.
        # Otherwise, iterate over all the basis functions involved in the decomposition of self and combine.
        if self._is_leaf:
            g = Point(is_leaf=True, decomposition_dict=None)
        else:
            g = Point(is_leaf=False, decomposition_dict=dict())

            for function, weight in self.decomposition_dict.items():
                grad = function.gradient(point)
                g = g + weight * grad

        # Return the newly created gradient
        return g

    def define_new_gradient_and_value(self, point):
        """
        Define a new gradient and a new function value

        :param point: (Point) Anny point
        :return: (tuple) newly computed gradient and function value
        """

        # If the function a basis one, then compute a simple gradient and function value.
        # Otherwise, iterate over all the basis functions involved in the decomposition of self and combine.
        if self._is_leaf:
            g = Point(is_leaf=True, decomposition_dict=None)
            f = Expression(is_function_value=True, decomposition_dict=None)
        else:
            g = Point(is_leaf=False, decomposition_dict=dict())
            f = Expression(is_function_value=False, decomposition_dict=dict())

            for function, weight in self.decomposition_dict.items():
                grad, val = function.oracle(point)
                g = g + weight * grad
                f = f + weight * val

        # Return the newly created gradient and function value
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

        # Create a new point
        point = Point(is_leaf=True, decomposition_dict=None)

        # Create a null gradient
        g = Point(is_leaf=False, decomposition_dict=dict())

        # Create a new function value
        f = Expression(is_function_value=True, decomposition_dict=None)

        # Add the triplet to the list of points of the function as well as to its list of optimal points
        self.add_point((point, g, f))

        # If self is not a basis function, create gradient and function value
        # for each of the latest and combine under the constraint that the full gradient is null.
        if not self._is_leaf:

            # Remove the basis functions that are not really involved in the decomposition of self
            # to avoid division by zero at the end.
            self.decomposition_dict = prune_dict(self.decomposition_dict)

            # Store the number of such basis function and the gradient and function value of self on point
            total_number_of_involved_basis_functions = len(self.decomposition_dict.keys())
            gradient_of_last_basis_function = g
            value_of_last_basis_function = f
            number_of_currently_computed_gradients_and_values = 0

            for function, weight in self.decomposition_dict.items():

                # Keep track of the gradient and function value of self minus those from visited basis functions
                if number_of_currently_computed_gradients_and_values < total_number_of_involved_basis_functions - 1:
                    grad, val = function.oracle(point)
                    gradient_of_last_basis_function = gradient_of_last_basis_function - weight * grad
                    value_of_last_basis_function = value_of_last_basis_function - weight * val

                    number_of_currently_computed_gradients_and_values += 1

                # The latest function must receive a fully conditioned gradient and a fully conditioned function value
                else:
                    gradient_of_last_basis_function = gradient_of_last_basis_function / weight
                    value_of_last_basis_function = value_of_last_basis_function / weight

                    function.add_point((point, gradient_of_last_basis_function, value_of_last_basis_function))

        # Return the required information
        if return_gradient_and_function_value:
            return point, g, f
        else:
            return point
