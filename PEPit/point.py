from PEPit.Tools.dict_operations import merge_dict, prune_dict, multiply_dicts
from PEPit.expression import Expression


class Point(object):
    """
    Point or Gradient
    """

    # Class counter.
    # It counts the dimension of the system of points,
    # namely the number of points needed to linearly generate the others.
    counter = 0

    def __init__(self,
                 is_leaf=True,
                 decomposition_dict=None,
                 ):
        """
        A point can either be a basis point, or a linear combination of basis points.
        The decomposition dict encodes the decomposition of the point in the basis.
        A point contains also a value that is computed only when the pep is solved.
        Finally, a basis point contains a counter to keep track of the order in which they where defined.

        :param is_leaf: (bool) if True, the point defines a new dimension, hence linearly independent from the others.
        :param decomposition_dict: (dict) the decomposition in the basis of points.
                                          None if the point defines a new direction.
        """

        # Store is_leaf in a protected attribute
        self._is_leaf = is_leaf

        # Initialize the value attribute to None until the PEP is solved
        self.value = None

        # If leaf, the decomposition is updated w.r.t the new direction,
        # the object counter is set
        # and the class counter updated.
        # Otherwise, the decomposition_dict is stored in an attribute and the object counter is set to None
        if is_leaf:
            assert decomposition_dict is None
            self.decomposition_dict = {self: 1}
            self.counter = Point.counter
            Point.counter += 1
        else:
            assert type(decomposition_dict) == dict
            self.decomposition_dict = decomposition_dict
            self.counter = None

    def __add__(self, other):
        """
        Add 2 points together, leading to a new point.

        :param other: (Point) Any other point
        :return: (Point) The sum of the 2 points
        """

        # Verify that other is a Point
        assert isinstance(other, Point)

        # Update the linear decomposition of the sum of 2 points from their respective basis decomposition
        merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)
        merged_decomposition_dict = prune_dict(merged_decomposition_dict)

        # Create and return the newly created Point that cannot be a leaf, by definition
        return Point(is_leaf=False, decomposition_dict=merged_decomposition_dict)

    def __sub__(self, other):
        """
        Subtract 2 points together, leading to a new point.

        :param other: (Point) Any other point
        :return: (Point) The difference between the 2 points
        """

        # A-B = A+(-B)
        return self.__add__(-other)

    def __neg__(self):
        """
        Compute the opposite of a point.

        :return: (Point) - point
        """

        # -A = (-1)*A
        return self.__rmul__(other=-1)

    def __rmul__(self, other):
        """
        Multiply 1 point to the left by a constant scalar value or another point.

        :param other: (int or float or Point) Any scalar value or any Point.
        :return: (Expression) other * self
        """

        # Multiplying by a scalar value is applying an homothety
        if isinstance(other, int) or isinstance(other, float):
            # Build the decomposition of the new point
            new_decomposition_dict = dict()
            for key, value in self.decomposition_dict.items():
                new_decomposition_dict[key] = value * other
            # Create and return the newly created point
            return Point(is_leaf=False, decomposition_dict=new_decomposition_dict)
        # Multiplying by another point leads to an expression encoding the inner product of the 2 points.
        elif isinstance(other, Point):
            # Compute the decomposition dict of the new expression
            decomposition_dict = multiply_dicts(self.decomposition_dict, other.decomposition_dict)
            # Create and return the new expression
            return Expression(is_function_value=False, decomposition_dict=decomposition_dict)
        else:
            # Raise an error if the user tries to multiply a point by anything else
            raise TypeError("Points can be multiplied by scalar constants and other points only!")

    def __mul__(self, other):
        """
        Multiply 1 point to the right by a constant scalar value or another point.

        :param other: (int or float or Point) Any scalar value or any Point.
        :return: (Expression) self * other
        """

        return self.__rmul__(other=other)

    def __truediv__(self, denominator):
        """
        Divide a point by a scalar value

        :param denominator: (int or float) the value to divide by.
        :return: (Point) The resulting point
        """
        # Verify the type of denominator
        assert isinstance(denominator, float) or isinstance(denominator, int)
        # P / v = P * (1/v)
        return self.__rmul__(1 / denominator)

    def __pow__(self, power):
        """
        Compute the square norm of a point.

        :param power: (int) must be 2.
        :return: (Expression) Inner product of point by itself.
        """
        # Works only for power=2
        assert power == 2

        # Return the inner product of a point by itself
        return self.__rmul__(self)

    def eval(self):
        """
        Compute, store and return the value of a point.
        Raise Exception if the PEP did not run yet.

        :return: (np.array) The value of the point.
        """

        # If the attribute value is not None, then simply return it.
        # Otherwise, compute it and return it.
        if self.value is None:
            # If leaf, the PEP would have filled the attribute at the end of the solve.
            if self._is_leaf:
                raise ValueError("The PEP must be solved to evaluate Points!")
            # If linear combination, combine the values of the basis, and store the result before returning it.
            else:
                value = 0
                for point, weight in self.decomposition_dict.items():
                    value += weight * point.eval()
                self.value = value

        return self.value
