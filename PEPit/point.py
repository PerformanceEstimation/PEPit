import numpy as np

from PEPit.expression import Expression

from PEPit.tools.dict_operations import merge_dict, prune_dict, multiply_dicts


class Point(object):
    """
    A :class:`Point` encodes an element of a pre-Hilbert space, either a point or a gradient.

    Attributes:
        _is_leaf (bool): True if self is defined from scratch
                         (not as linear combination of other :class:`Point` objects).
                         False if self is defined as linear combination of other points.
        _value (nd.array): numerical value of self obtained after solving the PEP via SDP solver.
                           Set to None before the call to the method `PEP.solve` from the :class:`PEP`.
        decomposition_dict (dict): decomposition of self as a linear combination of leaf :class:`Point` objects.
                                   Keys are :class:`Point` objects.
                                   And values are their associated coefficients.
        counter (int): counts the number of leaf :class:`Point` objects.

    :class:`Point` objects can be added or subtracted together.
    They can also be multiplied and divided by a scalar value.

    Example:
        >>> point1 = Point()
        >>> point2 = Point()
        >>> new_point = (- point1 + point2) / 5

    As in any pre-Hilbert space, there exists a scalar product.
    Therefore, :class:`Point` objects can be multiplied together.

    Example:
        >>> point1 = Point()
        >>> point2 = Point()
        >>> new_expr = point1 * point2

    The output is a scalar of type :class:`Expression`.

    The corresponding squared norm can also be computed.

    Example:
        >>> point = Point()
        >>> new_expr = point ** 2

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
        :class:`Point` objects can also be instantiated via the following arguments

        Args:
            is_leaf (bool): True if self is a :class:`Point` defined from scratch
                            (not as linear combination of other :class:`Point` objects).
                            False if self is a linear combination of existing :class:`Point` objects.
            decomposition_dict (dict): decomposition of self as a linear combination of **leaf** :class:`Point` objects.
                                       Keys are :class:`Point` objects.
                                       And values are their associated coefficients.

        Note:
            If `is_leaf` is True, then `decomposition_dict` must be provided as None.
            Then `self.decomposition_dict` will be set to `{self: 1}`.

        Instantiating the :class:`Point` object of the first example can be done by

        Example:
            >>> point1 = Point()
            >>> point2 = Point()
            >>> new_point = Point(is_leaf=False, decomposition_dict = {point1: -1/5, point2: 1/5})

        """

        # Store is_leaf in a protected attribute
        self._is_leaf = is_leaf

        # Initialize the value attribute to None until the PEP is solved
        self._value = None

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

    def get_is_leaf(self):
        """

        Returns:
            self._is_leaf (bool): allows to access the protected attribute `_is_leaf`.

        """
        return self._is_leaf

    def __add__(self, other):
        """
        Add 2 :class:`Point` objects together, leading to a new :class:`Point`.

        Args:
            other (Point): any other :class:`Point` object.

        Returns:
            self + other (Expression): The sum of the 2 :class:`Point` objects.

        Raises:
            TypeError: if provided `other` is not a :class:`Point`.

        """

        # Verify that other is a Point
        assert isinstance(other, Point)

        # Update the linear decomposition of the sum of 2 points from their respective leaf decomposition
        merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)
        merged_decomposition_dict = prune_dict(merged_decomposition_dict)

        # Create and return the newly created Point that cannot be a leaf, by definition
        return Point(is_leaf=False, decomposition_dict=merged_decomposition_dict)

    def __sub__(self, other):
        """
        Subtract 2 :class:`Point` objects together, leading to a new :class:`Point`.

        Args:
            other (Point): any other :class:`Point` object.

        Returns:
            self - other (Expression): The difference between the 2 :class:`Point` objects.

        Raises:
            TypeError: if provided `other` is not a :class:`Point`.

        """

        # A-B = A+(-B)
        return self.__add__(-other)

    def __neg__(self):
        """
        Compute the opposite of a :class:`Point`.

        Returns:
            - self (Point): the opposite of self.

        """

        # -A = (-1)*A
        return self.__rmul__(other=-1)

    def __rmul__(self, other):
        """
        Multiply a :class:`Point` by a scalar value or another :class:`Point`.

        Args:
            other (int or float or Point): any scalar constant or :class:`Point` object.

        Returns:
            other * self (Point or Expression): resulting product.

        Raises:
            TypeError: if provided `other` is neither a scalar value nor a :class:`Point`.

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
            return Expression(is_leaf=False, decomposition_dict=decomposition_dict)
        else:
            # Raise an error if the user tries to multiply a point by anything else
            raise TypeError("Points can be multiplied by scalar constants and other points only!"
                            "Got {}".format(type(other)))

    def __mul__(self, other):
        """
        Multiply a :class:`Point` by a scalar value or another :class:`Point`.

        Args:
            other (int or float or Point): any scalar constant or :class:`Point` object.

        Returns:
            self * other (Point or Expression): resulting product.

        Raises:
            TypeError: if provided `other` is neither a scalar value nor a :class:`Point`.

        """

        return self.__rmul__(other=other)

    def __truediv__(self, denominator):
        """
        Divide a :class:`Point` by a scalar value.

        Args:
            denominator (int or float): any scalar constant.

        Returns:
            self / other (Point): The ratio between this :class:`Point` and the scalar value `other`.

        Raises:
            TypeError: if provided `other` is not a scalar value.

        """
        # Verify the type of denominator
        assert isinstance(denominator, float) or isinstance(denominator, int)

        # P / v = P * (1/v)
        return self.__rmul__(1 / denominator)

    def __pow__(self, power):
        """
        Compute the squared norm of this :class:`Point`.

        Returns:
            self ** 2 (Expression): square norm of self.

        Raises:
            AssertionError: if provided `power` is not 2.

        """

        # Works only for power=2
        assert power == 2

        # Return the inner product of a point by itself
        return self.__rmul__(self)

    def eval(self):
        """
        Compute, store and return the value of this :class:`Point`.

        Returns:
            self._value (np.array): The value of this :class:`Point` after the corresponding PEP was solved numerically.

        Raises:
            ValueError("The PEP must be solved to evaluate Points!") if the PEP has not been solved yet.

        """

        # If the attribute value is not None, then simply return it.
        # Otherwise, compute it and return it.
        if self._value is None:
            # If leaf, the PEP would have filled the attribute at the end of the solve.
            if self._is_leaf:
                raise ValueError("The PEP must be solved to evaluate Points!")
            # If linear combination, combine the values of the leaf, and store the result before returning it.
            else:
                value = np.zeros(Point.counter)
                for point, weight in self.decomposition_dict.items():
                    value += weight * point.eval()
                self._value = value

        return self._value
