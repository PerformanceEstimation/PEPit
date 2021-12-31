class Constraint(object):
    """
    A :class:`Constraint` encodes either an equality or an inequality between two :class:`Expression` objects.

    A :class:`Constraint` must be understood either as
    `self.expression` = 0 or `self.expression` :math:`\\leqslant` 0
    depending on the value of `self.equality_or_inequality`.

    Attributes:
        expression (Expression): The :class:`Expression` that is compared to 0.
        equality_or_inequality (str): "equality" or "inequality". Encodes the type of constraint.
        _value (float): numerical value of self.expression obtained after solving the PEP via SDP solver.
                        Set to None before the call to the method `PEP.solve` from the :class:`PEP`.
        _dual_variable_value (float): the associated dual variable from the numerical solution to the corresponding PEP.
                                      Set to None before the call to `PEP.solve` from the :class:`PEP`
        counter (int): counts the number of :class:`Constraint` objects.

    A :class:`Constraint` results from a comparison between two :class:`Expression` objects.

    Example:
        >>> from PEPit import Expression
        >>> expr1 = Expression()
        >>> expr2 = Expression()
        >>> inequality1 = expr1 <= expr2
        >>> inequality2 = expr1 >= expr2
        >>> equality = expr1 == expr2

    """
    # Class counter.
    # It counts the number of generated constraints
    counter = 0

    def __init__(self, expression, equality_or_inequality):
        """
        :class:`Constraint` objects can also be instantiated via the following arguments.

        Args:
            expression (Expression): an object of class Expression
            equality_or_inequality (str): either 'equality' or 'inequality'.

        Instantiating the :class:`Constraint` objects of the first example can be done by

        Example:
            >>> from PEPit import Expression
            >>> expr1 = Expression()
            >>> expr2 = Expression()
            >>> inequality1 = Constraint(expression=expr1-expr2, equality_or_inequality="inequality")
            >>> inequality2 = Constraint(expression=expr2-expr1, equality_or_inequality="inequality")
            >>> equality = Constraint(expression=expr1-expr2, equality_or_inequality="equality")

        Raises:
            AssertionError: if provided `equality_or_inequality` argument is neither "equality" nor "inequality".

        """

        # Update the counter
        self.counter = Constraint.counter
        Constraint.counter += 1

        # Store the underlying expression
        self.expression = expression

        # Verify that 'equality_or_inequality' is well defined and store its value
        assert equality_or_inequality in {'equality', 'inequality'}
        self.equality_or_inequality = equality_or_inequality

        # The value of the underlying expression must be stored in self._value.
        self._value = None

        # Moreover, the associated dual variable value must be stored in self._dual_variable_value.
        self._dual_variable_value = None

    def eval(self):
        """
        Compute, store and return the value of the underlying :class:`Expression` of this :class:`Constraint`.

        Returns:
            self._value (np.array): The value of the underlying :class:`Expression` of this :class:`Constraint`
                                    after the corresponding PEP was solved numerically.

        Raises:
            ValueError("The PEP must be solved to evaluate Constraints!") if the PEP has not been solved yet.

        """

        # If the attribute value is not None, then simply return it.
        # Otherwise, compute it and return it.
        if self._value is None:

            try:
                self._value = self.expression.eval()
            except ValueError("The PEP must be solved to evaluate Expressions!"):
                raise ValueError("The PEP must be solved to evaluate Constraints!")

        return self._value

    def eval_dual(self):
        """
        Compute, store and return the value of the dual variable of this :class:`Constraint`.

        Returns:
            self._dual_variable_value (float): The value of the dual variable of this :class:`Constraint`
                                               after the corresponding PEP was solved numerically.

        Raises:
            ValueError("The PEP must be solved to evaluate Constraints dual variables!")
            if the PEP has not been solved yet.

        """

        # If the attribute _dual_variable_value is not None, then simply return it.
        # Otherwise, raise a ValueError.
        if self._dual_variable_value is None:
            # The PEP would have filled the attribute at the end of the solve.
            raise ValueError("The PEP must be solved to evaluate Constraints dual variables!")

        return self._dual_variable_value
