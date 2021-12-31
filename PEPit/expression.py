import warnings
import numpy as np

from PEPit.constraint import Constraint

from PEPit.tools.dict_operations import merge_dict


class Expression(object):
    """
    An :class:`Expression` is a linear combination of
    functions values,
    inner products of points and / or gradients (product of 2 :class:`Point` objects),
    and constant scalar values.

    Attributes:
        _is_leaf (bool): True if self is a function value defined from scratch
                                   (not as linear combination of other function values).
                                   False if self is a linear combination of existing :class:`Expression` objects.
        _value (float): numerical value of self obtained after solving the PEP via SDP solver.
                        Set to None before the call to the method `PEP.solve` from the :class:`PEP`.
        decomposition_dict (dict): decomposition of self as a linear combination of **leaf** :class:`Expression` objects.
                                   Keys are :class:`Expression` objects or tuple of 2 :class:`Point` objects.
                                   And values are their associated coefficients.
        counter (int): counts the number of **leaf** :class:`Expression` objects.

    :class:`Expression` objects can be added or subtracted together.
    They can also be added, subtracted, multiplied and divided by a scalar value.

    Example:
        >>> expr1 = Expression()
        >>> expr2 = Expression()
        >>> new_expr = (- expr1 + expr2 - 1) / 5

    :class:`Expression` objects can also be compared together

    Example:
        >>> expr1 = Expression()
        >>> expr2 = Expression()
        >>> inequality1 = expr1 <= expr2
        >>> inequality2 = expr1 >= expr2
        >>> equality = expr1 == expr2

    The three outputs `inequality1`, `inequality2` and `equality` are then :class:`Constraint` objects.

    """
    # Class counter.
    # It counts the number of function values needed to linearly generate the expressions.
    counter = 0

    def __init__(self,
                 is_leaf=True,
                 decomposition_dict=None,
                 ):
        """
        :class:`Expression` objects can also be instantiated via the following arguments

        Args:
            is_leaf (bool): True if self is a function value defined from scratch
                                      (not as linear combination of other function values).
                                      False if self is a linear combination of existing :class:`Expression` objects.
            decomposition_dict (dict): decomposition of self as a linear combination of **leaf** :class:`Expression` objects.
                                       Keys are :class:`Expression` objects or tuple of 2 :class:`Point` objects.
                                       And values are their associated coefficients.

        Note:
            If `is_leaf` is True, then `decomposition_dict` must be provided as None.
            Then `self.decomposition_dict` will be set to `{self: 1}`.

        Instantiating the :class:`Expression` object of the first example can be done by

        Example:
            >>> expr1 = Expression()
            >>> expr2 = Expression()
            >>> new_expr = Expression(is_leaf=False, decomposition_dict = {expr1: -1/5, expr2: 1/5, 1: -1/5})

        """
        # Store is_leaf in a protected attribute
        self._is_leaf = is_leaf

        # Initialize the value attribute to None until the PEP is solved
        self._value = None

        # If leaf function value, the decomposition is updated,
        # the object counter is set
        # and the class counter updated.
        # Otherwise, the decomposition_dict is stored in an attribute and the object counter is set to None
        if is_leaf:
            assert decomposition_dict is None
            self.decomposition_dict = {self: 1}
            self.counter = Expression.counter
            Expression.counter += 1
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
        Add 2 :class:`Expression` objects together, leading to a new :class:`Expression` object.
        Note an :class:`Expression` can also be added to a python `float` or `int`.

        Args:
            other (Expression or int or float): any other :class:`Expression` object or scalar constant.

        Returns:
            self + other (Expression): The sum of the 2 :class:`Expression` objects.

        Raises:
            TypeError: if provided `other` is neither an :class:`Expression` nor a scalar value.

        """

        # If other is an Expression, merge the decomposition_dicts
        if isinstance(other, Expression):
            merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)
        # It other is a scalar constant, add it to the decomposition_dict of self
        elif isinstance(other, int) or isinstance(other, float):
            merged_decomposition_dict = merge_dict(self.decomposition_dict, {1: other})
        # Raise an Exception in any other scenario
        else:
            raise TypeError("Expression can be added only to other expression or scalar values")

        # Create and return the newly created Expression
        return Expression(is_leaf=False, decomposition_dict=merged_decomposition_dict)

    def __sub__(self, other):
        """
        Subtract 2 :class:`Expression` objects together, leading to a new :class:`Expression` object.
        Note a python `float` or `int` can also be subtracted from an :class:`Expression`.

        Args:
            other (Expression or int or float): any other :class:`Expression` object or scalar constant.

        Returns:
            self - other (Expression): the difference between the 2 :class:`Expression` objects.

        Raises:
            TypeError: if provided `other` is neither an :class:`Expression` nor a scalar value.

        """
        # A-B = A+(-B)
        return self.__add__(-other)

    def __neg__(self):
        """
        Compute the opposite of an :class:`Expression`.

        Returns:
            - self (Expression): the opposite of self.

        """

        # -A = (-1)*A
        return self.__rmul__(other=-1)

    def __rmul__(self, other):
        """
        Multiply an :class:`Expression` by a scalar value.

        Args:
            other (int or float): any scalar constant

        Returns:
            other * self (Expression): the product of the 2 :class:`Expression` objects.

        Raises:
            AssertionError: if provided `other` is not a scalar value.

        """

        # Verify other is a scalar constant
        assert isinstance(other, int) or isinstance(other, float)

        # Multiply uniformly self's decomposition_dict by other
        new_decomposition_dict = dict()
        for key, value in self.decomposition_dict.items():
            new_decomposition_dict[key] = value * other

        # Create and return the newly created Expression
        return Expression(is_leaf=False, decomposition_dict=new_decomposition_dict)

    def __mul__(self, other):
        """
        Multiply an :class:`Expression` by a scalar value.

        Args:
            other (int or float): any scalar constant

        Returns:
            self * other (Expression): the product of the 2 :class:`Expression` objects.

        Raises:
            AssertionError: if provided `other` is not a scalar value.

        """

        return self.__rmul__(other=other)

    def __truediv__(self, denominator):
        """
        Divide an :class:`Expression` by a scalar value.

        Args:
            denominator (int or float): the scalar value to divide by.

        Returns:
            self / other (Expression): the ratio between the 2 :class:`Expression` objects.

        Raises:
            AssertionError: if provided `other` is not a scalar value.

        """

        # P / v = P * (1/v)
        return self.__rmul__(other=1 / denominator)

    def __le__(self, other):
        """
        Create an inequality :class:`Constraint` object from an inequality between two :class:`Expression` objects.

        Args:
            other (Expression of int or float): any :class:`Expression` of python scalar object.

        Returns:
            self :math:`\\leq` other (Expression): :class:`Constraint` object encoding the corresponding inequality.

        """

        return Constraint(self - other, equality_or_inequality='inequality')

    def __lt__(self, other):
        """
        Create an inequality :class:`Constraint` object from an inequality between two :class:`Expression` objects.

        Args:
            other (Expression of int or float): any :class:`Expression` of python scalar object.

        Returns:
            self < other (Expression): :class:`Constraint` object encoding the corresponding inequality.

        Note:
            The input inequality is strict,
            but optimizing over the interior set is equivalent to considering the large one,
            so we refer to the latest.

        Raises:
            Warnings("Strict constraints will lead to the same solution as under soft constraints")

        """

        warnings.warn("Strict constraints will lead to the same solution as under soft constraints")
        return self.__le__(other=other)

    def __ge__(self, other):
        """
        Create an inequality :class:`Constraint` object from an inequality between two :class:`Expression` objects.

        Args:
            other (Expression of int or float): any :class:`Expression` of python scalar object.

        Returns:
            other :math:`\\leq` self (Expression): :class:`Constraint` object encoding the corresponding inequality.

        """
        return -self <= -other

    def __gt__(self, other):
        """
        Create an inequality :class:`Constraint` object from an inequality between two :class:`Expression` objects.

        Args:
            other (Expression of int or float): any :class:`Expression` of python scalar object.

        Returns:
            other < self (Expression): :class:`Constraint` object encoding the corresponding inequality.

        Note:
            The input inequality is strict,
            but optimizing over the interior set is equivalent to considering the large one,
            so we refer to the latest.

        Raises:
            Warnings("Strict constraints will lead to the same solution as under soft constraints")

        """

        warnings.warn("Strict constraints will lead to the same solution as under soft constraints")
        return self.__ge__(other=other)

    def __eq__(self, other):
        """
        Create an equality :class:`Constraint` object from an equality between two :class:`Expression` objects.

        Args:
            other (Expression of int or float): any :class:`Expression` of python scalar object.

        Returns:
            self = other (Expression): :class:`Constraint` object encoding the corresponding equality.

        """

        return Constraint(self - other, equality_or_inequality='equality')

    def __hash__(self):
        return super().__hash__()

    def eval(self):
        """
        Compute, store and return the value of this :class:`Expression`.

        Returns:
            self._value (np.array): Value of this :class:`Expression` after the corresponding PEP was solved numerically.

        Raises:
            ValueError("The PEP must be solved to evaluate Expressions!") if the PEP has not been solved yet.

            TypeError("Expressions are made of function values, inner products and constants only!")

        """

        # If the attribute value is not None, then simply return it.
        # Otherwise, compute it and return it.
        if self._value is None:
            # If leaf function value, the PEP would have filled the attribute at the end of the solve.
            if self._is_leaf:
                raise ValueError("The PEP must be solved to evaluate Expressions!")
            # If linear combination,
            # combine the values of the leaf expressions,
            # and store the result before returning it.
            else:
                value = 0
                for key, weight in self.decomposition_dict.items():
                    # Distinguish 3 cases: function values, inner products, and constant values
                    if type(key) == Expression:
                        assert key.get_is_leaf()
                        value += weight * key.eval()
                    elif type(key) == tuple:
                        point1, point2 = key
                        assert point1.get_is_leaf()
                        assert point2.get_is_leaf()
                        value += weight * np.dot(point1.eval(), point2.eval())
                    elif key == 1:
                        value += weight
                    # Raise Exception out of those 3 cases
                    else:
                        raise TypeError("Expressions are made of function values, inner products and constants only!"
                                        "Got {}".format(type(key)))
                # Store the value
                self._value = value

        # Return the value
        return self._value
