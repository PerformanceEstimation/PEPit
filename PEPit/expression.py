import warnings
import numpy as np

from PEPit.Tools.dict_operations import merge_dict

from PEPit.constraint import Constraint


class Expression(object):
    """
    Function value, inner product, or constant
    """
    # Class counter.
    # It counts the number of function values needed to linearly generate the expressions.
    counter = 0

    def __init__(self,
                 is_function_value=True,
                 decomposition_dict=None,
                 ):
        """
        An expression is a linear combination of functions values, inner products and constant scalar values.

        :param is_function_value: (bool) If true, the expression is a basis function value.
        :param decomposition_dict: (dict) Decomposition in the basis of function values, inner products and constants.
        """
        # Store is_function_value in a protected attribute
        self._is_function_value = is_function_value

        # Initialize the value attribute to None until the PEP is solved
        self.value = None

        # If basis function value, the decomposition is updated,
        # the object counter is set
        # and the class counter updated.
        # Otherwise, the decomposition_dict is stored in an attribute and the object counter is set to None
        if is_function_value:
            assert decomposition_dict is None
            self.decomposition_dict = {self: 1}
            self.counter = Expression.counter
            Expression.counter += 1
        else:
            assert type(decomposition_dict) == dict
            self.decomposition_dict = decomposition_dict
            self.counter = None

    def __add__(self, other):
        """
        Add 2 expressions together, leading to a new expression. Note an expression can also be added to a constant.

        :param other: (Expression or int or float) Any other expression or scalar constant
        :return: (Expression) The sum of the 2 expressions
        """

        # If other is an Expression, merge the decomposition_dicts
        if isinstance(other, Expression):
            merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)
        # It other is a scalar constant, add it to the decomposition_dict of self
        elif type(other) in {int, float}:
            merged_decomposition_dict = merge_dict(self.decomposition_dict, {1: other})
        # Raise an Exception in any other scenario
        else:
            raise TypeError("Expression can be added only to other expression or scalar values")

        # Create and return the newly created Expression
        return Expression(is_function_value=False, decomposition_dict=merged_decomposition_dict)

    def __sub__(self, other):
        """
        Subtract 2 Expressions together, leading to a new expression.

        :param other: (Expression or int or float) Any other expression or scalar constant
        :return: (Expression) The difference between the 2 expressions
        """

        # A-B = A+(-B)
        return self.__add__(-other)

    def __neg__(self):
        """
        Compute the opposite of an expression.

        :return: (Expression) - expression
        """

        # -A = (-1)*A
        return self.__rmul__(other=-1)

    def __rmul__(self, other):
        """
        Multiply an expression by a scalar value

        :param other: (int or float) Any scalar constant
        :return: (Expression) other * self
        """

        # Verify other is a scalar constant
        assert type(other) in {int, float}

        # Multiply uniformly self's decomposition_dict by other
        new_decomposition_dict = dict()
        for key, value in self.decomposition_dict.items():
            new_decomposition_dict[key] = value * other

        # Create and return the newly created Expression
        return Expression(is_function_value=False, decomposition_dict=new_decomposition_dict)

    def __mul__(self, other):
        """
        Multiply an expression by a scalar value

        :param other: (int or float) Any scalar constant
        :return: (Expression) self * other
        """

        return self.__rmul__(other=other)

    def __truediv__(self, denominator):
        """
        Divide an expression by a scalar value

        :param denominator: (int or float) the value to divide by.
        :return: (Expression) The resulting expression
        """

        # P / v = P * (1/v)
        return self.__rmul__(other=1 / denominator)

    def __le__(self, other):
        """
        Create a non-positive expression from an inequality

        :param other: (Expression of int or float)
        :return: (Expression) Expression <= 0 must be equivalent to the input inequality
        """

        return Constraint(self - other, equality_or_inequality='inequality')

    def __lt__(self, other):
        """
        Create a non-positive expression from an inequality

        :param other: (Expression of int or float)
        :return: (Expression) Expression <= 0 must be equivalent to the input inequality

        Note: The input inequality is strict, but optimizing over the interior set is equivalent,
        so we refer to the large inequality.
        """

        warnings.warn("Strict constraints will lead to the same solution as under soft constraints")
        return self.__le__(other=other)

    def __ge__(self, other):
        """
        Create a non-positive expression from an inequality

        :param other: (Expression of int or float)
        :return: (Expression) Expression <= 0 must be equivalent to the input inequality
        """

        return other <= self

    def __gt__(self, other):
        """
        Create a non-positive expression from an inequality

        :param other: (Expression of int or float)
        :return: (Expression) Expression <= 0 must be equivalent to the input inequality

        Note: The input inequality is strict, but optimizing over the interior set is equivalent,
        so we refer to the large inequality.
        """

        warnings.warn("Strict constraints will lead to the same solution as under soft constraints")
        return self.__ge__(other=other)

    # TODO define __eq__. Currently it raises an error.
    # def __eq__(self, other):
    #     """
    #     Create a null expression from an equality
    #
    #     :param other: (Expression of int or float)
    #     :return: (Expression) Expression <= 0 must be equivalent to the input inequality
    #     """
    #
    #     return Constraint(self - other, equality_or_inequality='equality')

    def eval(self):
        """
        Compute, store and return the value of an expression.
        Raise Exception if the PEP did not run yet.

        :return: (np.array) The value of the expression.
        """

        # If the attribute value is not None, then simply return it.
        # Otherwise, compute it and return it.
        if self.value is None:
            # If basis function value, the PEP would have filled the attribute at the end of the solve.
            if self._is_function_value:
                raise ValueError("The PEP must be solved to evaluate Points!")
            # If linear combination, combine the values of the basis, and store the result before returning it.
            else:
                value = 0
                for key, weight in self.decomposition_dict.items():
                    # Distinguish 3 cases: function values, inner products, and constant values
                    if type(key) == Expression:
                        assert key._is_function_value
                        value += weight * key.eval()
                    elif type(key) == tuple:
                        point1, point2 = key
                        assert point1._is_leaf
                        assert point2._is_leaf
                        value += weight * np.dot(point1.eval(), point2.eval())
                    elif key == 1:
                        value += weight
                    # Raise Exception out of those 3 cases
                    else:
                        raise TypeError("Expressions are made of function values, inner products and constants only!")
                # Store the value
                self.value = value

        # Return the value
        return self.value
