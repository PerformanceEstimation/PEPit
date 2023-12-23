import numpy as np

from PEPit.expression import Expression


class PSDMatrix(object):
    """
    A :class:`PSDMatrix` encodes a square matrix of :class:`Expression` objects that is constrained to be symmetric PSD.

    Attributes:
        name (str): A name set through the set_name method. None is no name is given.
        matrix_of_expressions (Iterable of Iterable of Expression): a square matrix of :class:`Expression` objects.
        shape (tuple of ints): the shape of the underlying matrix of :class:`Expression` objects.
        _value (2D ndarray of floats): numerical values of :class:`Expression` objects
                                       obtained after solving the PEP via SDP solver.
                                       Set to None before the call to the method `PEP.solve` from the :class:`PEP`.
        _dual_variable_value (2D ndarray of floats): the associated dual matrix
                                                     from the numerical solution to the corresponding PEP.
                                                     Set to None before the call to `PEP.solve` from the :class:`PEP`.
        entries_dual_variable_value (2D ndarray of floats): the dual of each correspondence between entries of
                                                            the matrix and the underlying :class:`Expression` objects.
        counter (int): counts the number of :class:`PSDMatrix` objects.

    Example:
        >>> # Defining t <= sqrt(expr) for a given expression expr.
        >>> from PEPit import Expression
        >>> from PEPit import PSDMatrix
        >>> expr = Expression()
        >>> t = Expression()
        >>> psd_matrix = PSDMatrix(matrix_of_expressions=[[expr, t], [t, 1]])
        >>> # The last line means that the matrix [[expr, t], [t, 1]] is constrained to be PSD.
        >>> # This is equivalent to det([[expr, t], [t, 1]]) >= 0, i.e. expr - t^2 >= 0.

    """
    # Class counter.
    # It counts the number of generated constraints
    counter = 0

    def __init__(self,
                 matrix_of_expressions,
                 name=None,
                 ):
        """
        :class:`PSDMatrix` objects are instantiated via the following argument.

        Args:
            matrix_of_expressions (Iterable of Iterable of Expression): a square matrix of :class:`Expression`.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Instantiating the :class:`PSDMatrix` objects of the first example can be done by

        Example:
            >>> import numpy as np
            >>> from PEPit import Expression
            >>> from PEPit import PSDMatrix
            >>> matrix_of_expressions = np.array([Expression() for i in range(4)]).reshape(2, 2)
            >>> psd_matrix = PSDMatrix(matrix_of_expressions=matrix_of_expressions)

        Raises:
            AssertionError: if provided matrix is not a square matrix.
            TypeError: if provided matrix does not contain only Expressions and / or scalar values.

        """
        # Initialize name of the psd matrix
        self.name = name

        # Update the counter
        self.counter = PSDMatrix.counter
        PSDMatrix.counter += 1

        # Store the underlying matrix of expressions
        self.matrix_of_expressions = self._store(matrix_of_expressions)
        self.shape = self.matrix_of_expressions.shape

        # The value of the underlying expression must be stored in self._value.
        self._value = None

        # Moreover, the associated dual variable value must be stored in self._dual_variable_value.
        self._dual_variable_value = None
        self.entries_dual_variable_value = None

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

    @staticmethod
    def _store(matrix_of_expressions):
        """
        Store a new matrix of :class:`Expression`\s that we enforce to be positive semi-definite.

        Args:
            matrix_of_expressions (Iterable of Iterables of Expressions): a square matrix of :class:`Expression`.

        Raises:
            AssertionError: if provided matrix is not a square matrix.
            TypeError: if provided matrix does not contain only Expressions and / or scalar values.

        """

        # Change iterable into ndarray
        matrix = np.array(matrix_of_expressions)

        # Verify constraint is a square matrix
        size = matrix.shape[0]
        assert matrix.shape == (size, size)

        # Transform scalars into Expressions
        for i in range(size):
            for j in range(size):
                # The entry must be an Expression ...
                if isinstance(matrix[i, j], Expression):
                    pass
                # ... or a python scalar. If so, store it as an Expression
                elif isinstance(matrix[i, j], int) or isinstance(matrix[i, j], float):
                    matrix[i, j] = Expression(is_leaf=False, decomposition_dict={1: matrix[i, j]})
                # Raise an Exception in any other scenario
                else:
                    raise TypeError("PSD matrices contains only Expressions and / or scalar values!"
                                    "Got {}".format(type(matrix[i, j])))

        # Return processed matrix
        return matrix

    def __getitem__(self, item):
        """
        Access to an element of the underlying matrix of expressions
        using subscript of self.

        Args:
            item (2 ints): line and column coordinates access to element of the matrix of expressions

        Returns:
            self.matrix_of_expression[item] (Expression): the expression placed at `item` in the matrix of expressions.

        """
        return self.matrix_of_expressions[item]

    def eval(self):
        """
        Compute, store and return the value of the underlying matrix of :class:`Expression` objects.

        Returns:
            self._value (np.array): The value of the underlying matrix of :class:`Expression` objects
                                    after the corresponding PEP was solved numerically.

        Raises:
            ValueError("The PEP must be solved to evaluate PSDMatrix!") if the PEP has not been solved yet.

        """

        # If the attribute value is not None, then simply return it.
        # Otherwise, compute it and return it.
        if self._value is None:
            try:
                self._value = np.array([[expression.eval() for expression in line]
                                        for line in self.matrix_of_expressions])
            except ValueError:
                raise ValueError("The PEP must be solved to evaluate PSDMatrix!")

        # Return the value
        return self._value

    def eval_dual(self):
        """
        Compute, store and return the value of the dual variable of this :class:`PSDMatrix`.

        Returns:
            self._dual_variable_value (ndarray of floats): The value of the dual variable of this :class:`PSDMatrix`
                                                           after the corresponding PEP was solved numerically.

        Raises:
            ValueError("The PEP must be solved to evaluate PSDMatrix dual variables!")
            if the PEP has not been solved yet.

        """

        # If the attribute _dual_variable_value is not None, then simply return it.
        # Otherwise, raise a ValueError.
        if self._dual_variable_value is None:
            # The PEP would have filled the attribute after solving the problem.
            raise ValueError("The PEP must be solved to evaluate PSDMatrix dual variables!")

        return self._dual_variable_value
