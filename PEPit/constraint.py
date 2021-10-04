class Constraint(object):
    """
    Equality or inequality between two constaints
    """
    # Class counter.
    # It counts the number of generated constraints
    counter = 0

    def __init__(self, expression, equality_or_inequality):

        self.counter = Constraint.counter
        Constraint.counter += 1

        self.expression = expression
        assert equality_or_inequality in {'equality', 'inequality'}
        self.equality_or_inequality = equality_or_inequality

        self.dual_variable_value = None
