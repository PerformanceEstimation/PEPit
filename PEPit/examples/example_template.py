from PEPit import PEP
# TODO import what you need from the pipeline
# from PEPit.functions import ``THE FUNCTION CLASSES YOU NEED``
# from PEPit.operators import ``THE OPERATOR CLASSES YOU NEED``
# from primitive_steps import ``THE PRIMITIVE STEPS YOU NEED``


def wc_example_template(arg1, arg2, arg3, verbose=True):
    """
    Consider the ``CHARACTERISTIC (eg., convex)`` minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x),

    where :math:`f` is ``CLASS (eg., smooth convex)``.

    This code computes a worst-case guarantee for the ** ``NAME OF THE METHOD`` **.
    That is, it computes the smallest possible :math:`\\tau(arg_1, arg_2, arg_3)` such that the guarantee

    .. math:: \\text{PERFORMANCE METRIC} \\leqslant \\tau(arg_1, arg_2, arg_3) \\text{ INITIALIZATION}

    is valid, where ``NOTATION OF THE OUTPUT`` is the output of the ** ``NAME OF THE METHOD`` **,
    and where :math:`x_\\star` is the minimizer of :math:`f`.
    In short, for given values of ``ARGUMENTS``,
    :math:`\\tau(arg_1, arg_2, arg_3)` is computed as the worst-case value of
    :math:`\\text{PERFORMANCE METRIC}` when :math:`\\text{INITIALIZATION} \\leqslant 1`.

    **Algorithm**:
    The ``NAME OF THE METHOD`` of this example is provided in ``REFERENCE WITH SPECIFIED ALGORITHM`` by

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\text{MAIN STEP}
            \\end{eqnarray}

    **Theoretical guarantee**:
    A ``TIGHT, UPPER OR LOWER`` guarantee can be found in ``REFERENCE WITH SPECIFIED THEOREM``:

    .. math:: \\text{PERFORMANCE METRIC} \\leqslant \\text{THEORETICAL BOUND} \\text{ INITIALIZATION}

    **References**:

    `[1] F. Name, F. Name, F. Name (YEAR).
    Title.
    Conference or journal (Acronym of conference or journal).
    <https://arxiv.org/pdf/KEY.pdf OR OTHER URL>`_

    `[2] F. Name, F. Name, F. Name (YEAR).
    Title.
    Conference or journal (Acronym of journal or conference).
    <https://arxiv.org/pdf/KEY.pdf OR OTHER URL>`_

    `[3] F. Name, F. Name, F. Name (YEAR).
    Title.
    Conference or journal (Acronym of journal or conference).
    <https://arxiv.org/pdf/KEY.pdf OR OTHER URL>`_

    Args:
        arg1 (type1): description of arg1.
        arg2 (type2): description of arg2.
        arg3 (type3): description of arg3.
        verbose (bool): if True, print conclusion

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (float): theoretical value

    Example:
        >>> pepit_tau, theoretical_tau = wc_example_template(arg1=value1, arg2=value2, arg3=value3, verbose=True)
        ``OUTPUT MESSAGE``

    """

    # Instantiate PEP
    problem = PEP()

    # # Declare functions
    # func = problem.declare_function(function_class=function_class,  # TODO specify
    #                                 param=param,  # TODO specify
    #                                 reuse_gradient=reuse_gradient,  # TODO specify
    #                                 )
    #
    # # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    # xs = func.stationary_point()
    # fs = func.value(xs)
    #
    # # Then define the starting point x0 of the algorithm
    # x0 = problem.set_initial_point()
    #
    # # Set the initial constraint that is the distance between x0 and x^*
    # problem.set_initial_condition(initialization <= 1)  # TODO specify
    #
    # # Run n steps of the fast gradient method
    # x = x0
    # for i in range(n):
    #     # TODO specify
    #     #################
    #     ### Main STEP ###
    #     #################
    #
    # # Set the performance metric to the function value accuracy
    # problem.set_performance_metric(performance_metric)  # TODO specify
    #
    # # Solve the PEP
    # pepit_tau = problem.solve(verbose=verbose)
    #
    # # Theoretical guarantee (for comparison)
    # theoretical_tau = theoretical_tau  # TODO specify

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of ``NAME OF THE METHOD`` ***')
        print('\tPEPit guarantee:\t ``PERFORMANCE METRIC`` <= {:.6} ``INITIALIZATION``'.format(pepit_tau))
        print('\tTheoretical guarantee:\t ``PERFORMANCE METRIC`` <= {:.6} ``INITIALIZATION``'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    pepit_tau, theoretical_tau = wc_example_template(arg1=value1, arg2=value2, arg3=value3, verbose=True)
