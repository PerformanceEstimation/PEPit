from PEPit.Primitive_steps.inexactgradient import inexactgradient


def inexactgradient_step(x0, f, gamma, epsilon, notion='absolute'):
    """
    TODO: CANEVAS DESCRIPTION DE CE QU'on FAIT

    :param x0:
    :param f:
    :param gamma:
    :param epsilon:
    :param notion:
    :return:
    """
    dx0, fx0 = inexactgradient(x0, f, epsilon, notion=notion)
    x = x0 - gamma * dx0
    return x, dx0, fx0
