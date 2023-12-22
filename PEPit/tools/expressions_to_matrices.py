import numpy as np

from PEPit.point import Point
from PEPit.expression import Expression


def expression_to_matrices(expression):
    """
    Translate an expression from an :class:`Expression` to a matrix, a vector, and a constant such that

        ..math:: \\mathrm{Tr}(\\text{Gweights}\\,G) + \\text{Fweights}^T F + \\text{cons}

    corresponds to the expression.

    Args:
        expression (Expression): any expression.

    Returns:
        Gweights (numpy array): weights of the entries of G in the :class:`Expression`.
        Fweights (numpy array): weights of the entries of F in the :class:`Expression`
        cons (float): constant term in the :class:`Expression`

    """
    cons = 0
    Fweights = np.zeros((Expression.counter,))
    Gweights = np.zeros((Point.counter, Point.counter))

    # If simple function value, then simply return the right coordinate in F
    if expression.get_is_leaf():
        Fweights[expression.counter] += 1
    # If composite, combine all the cvxpy expression found from leaf expressions
    else:
        for key, weight in expression.decomposition_dict.items():
            # Function values are stored in F
            if type(key) == Expression:
                assert key.get_is_leaf()
                Fweights[key.counter] = weight
            # Inner products are stored in G
            elif type(key) == tuple:
                point1, point2 = key
                assert point1.get_is_leaf()
                assert point2.get_is_leaf()
                Gweights[point1.counter, point2.counter] = weight
            # Constants are simply constants
            elif key == 1:
                cons = weight
            # Others don't exist and raise an Exception
            else:
                raise TypeError("Expressions are made of function values, inner products and constants only!")

    Gweights = (Gweights + Gweights.T) / 2

    return Gweights, Fweights, cons


def expression_to_sparse_matrices(expression):
    """
    Translate an expression from an :class:`Expression` to a matrix, a vector, and a constant such that

        ..math:: \\mathrm{Tr}(\\text{Gweights}\\,G) + \\text{Fweights}^T F + \\text{cons}

    where :math:`\\text{Gweights}` and :math:`\\text{Fweights}` are expressed in sparse formats.

    Args:
        expression (Expression): any expression.

    Returns:
        Gweights_indi (numpy array): Set of line indices for the sparse representation of the constraint matrix (multiplying G).
        Gweights_indj (numpy array): Set of column indices for the sparse representation of the constraint matrix (multiplying G).
        Gweights_val (numpy array): Set of values for the sparse representation of the constraint matrix (multiplying G).
        Fweights_ind (numpy array): Set of indices for the sparse representation of the constraint vector (multiplying F).
        Fweights_val (numpy array): Set of values of the sparse representation of the constraint vector (multiplying F).
        cons_val (float): Constant part of the constraint.

    """
    cons_val = 0
    Fweights_ind = list()
    Fweights_val = list()
    Gweights_indi = list()
    Gweights_indj = list()
    Gweights_val = list()

    # If simple function value, then simply return the right coordinate in F
    if expression.get_is_leaf():
        Fweights_ind.append(expression.counter)
        Fweights_val.append(1)
    # If composite, combine all the cvxpy expression found from leaf expressions
    else:
        for key, weight in expression.decomposition_dict.items():
            # Function values are stored in F
            if type(key) == Expression:
                assert key.get_is_leaf()
                Fweights_ind.append(key.counter)
                Fweights_val.append(weight)
            # Inner products are stored in G
            elif type(key) == tuple:
                point1, point2 = key
                assert point1.get_is_leaf()
                assert point2.get_is_leaf()

                weight_sym = 0  # weight of the symmetrical entry
                if (point2, point1) in expression.decomposition_dict:
                    if point1.counter >= point2.counter:  # if both entry and symmetrical entry: only append in one case
                        weight_sym = expression.decomposition_dict[(point2, point1)]
                        Gweights_val.append((weight + weight_sym) / 2)
                        Gweights_indi.append(point1.counter)
                        Gweights_indj.append(point2.counter)
                else:
                    Gweights_val.append((weight + weight_sym) / 2)
                    Gweights_indi.append(max(point1.counter, point2.counter))
                    Gweights_indj.append(min(point1.counter, point2.counter))
            # Constants are simply constants
            elif key == 1:
                cons_val = weight
            # Others don't exist and raise an Exception
            else:
                raise TypeError("Expressions are made of function values, inner products and constants only!")

    Fweights_ind = np.array(Fweights_ind)
    Fweights_val = np.array(Fweights_val)
    Gweights_indi = np.array(Gweights_indi)
    Gweights_indj = np.array(Gweights_indj)
    Gweights_val = np.array(Gweights_val)

    return Gweights_indi, Gweights_indj, Gweights_val, Fweights_ind, Fweights_val, cons_val
