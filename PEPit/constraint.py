class Constraint(object):
    """
    Equality or inequality between two expressions

    Attributes:
        counter (int)
        expression (Expression)
        equality_or_inequality (str): self is Expression = 0 or Expression :math:`\\leqslant` 0
        dual_variable_value (float)

    """
    # Class counter.
    # It counts the number of generated constraints
    counter = 0

    def __init__(self, expression, equality_or_inequality):
        """
        A constraint is defined by an expression that is either equal to or non-greater than 0.
        The constraint must be understood as Expression == 0 or Expression <=0.

        Args:
            expression (Expression): an object of class Expression
            equality_or_inequality (str): either 'equality' or 'inequality'.

        """

        # Update the counter
        self.counter = Constraint.counter
        Constraint.counter += 1

        # Store the underlying expression
        self.expression = expression

        # Verify that 'equality_or_inequality' is well defined and store its value
        assert equality_or_inequality in {'equality', 'inequality'}
        self.equality_or_inequality = equality_or_inequality

        # After solving the PEP, one can find the value of the underlying expression in self.expression.value.
        # Moreover, the associated dual variable value must be stored in self.dual_variable_value.
        self.dual_variable_value = None
