from PEPit.point import Point
from PEPit.expression import Expression


def inexact_proximal_step(x0, f, gamma, opt='PD_gapII'):
    """
    This routine encodes an inexact proximal operation with step size :math:`\\gamma`. That is, it outputs a tuple
    :math:`(x, g\\in \\partial f(x), f(x), w, v\\in\\partial f(w), f(w), \\varepsilon)` which are described as follows.

    First, :math:`x` is an approximation to the proximal point of :math:`x_0` on function :math:`f`:

        .. math:: x \\approx \\mathrm{prox}_{\\gamma f}(x_0)\\triangleq\\arg\\min_x \\left\\{ \\gamma f(x) + \\frac{1}{2}\\|x-x_0\\|^2\\right\\},

    where the meaning of :math:`\\approx` depends on the option "opt" and is explained below.
    The notions of inaccuracy implemented within this routine are specified using primal and dual proximal problems, denoted by

        .. math::
            :nowrap:

            \\begin{eqnarray}
            &\\Phi^{(p)}_{\\gamma f}(x; x_0) \\triangleq \\gamma f(x) + \\frac{1}{2}\\|x-x_0\\|^2,\\\\
            &\\Phi^{(d)}_{\\gamma f}(v; x_0) \\triangleq -\\gamma f^*(v)-\\frac{1}{2}\\|x_0-\\gamma v\\|^2 + \\frac{1}{2}\\|x_0\\|^2,\\\\
            \\end{eqnarray}

    where :math:`\\Phi^{(p)}_{\\gamma f}(x;x_0)` and :math:`\\Phi^{(d)}_{\\gamma f}(v;x_0)` respectively denote the primal
    and the dual proximal problems, and where :math:`f^*` is the Fenchel conjugate of :math:`f`. The options below
    encode different meanings of ":math:`\\approx`" by specifying accuracy requirements on primal-dual pairs:

        .. math:: (x,v) \\approx_{\\varepsilon} \\left(\\mathrm{prox}_{\\gamma f}(x_0),\\,\mathrm{prox}_{f^*/\\gamma}(x_0/\\gamma)\\right),

    where :math:`\\approx_{\\varepsilon}` corresponds to require the primal-dual pair :math:`(x,v)` to satisfy some
    primal-dual accuracy requirement:

        .. math:: \\Phi^{(p)}_{\\gamma f}(x;x_0)-\\Phi^{(d)}_{\\gamma f}(v;x_0) \\leqslant \\varepsilon,

    where :math:`\\varepsilon\\geqslant 0` is the error magnitude, which is returned to the user so that one can constrain
    it to be bounded by some other values.

    **Relation to the exact proximal operation:**  In the exact case (no error in the computation, :math:`\\varepsilon=0`),
    :math:`v` corresponds to the solution of the dual proximal problem and one can write

        .. math:: x = x_0-\\gamma g,

    with :math:`g=v=\mathrm{prox}_{f^*/\\gamma}(x_0/\\gamma)\\in\\partial f(x)`, and :math:`x=w`.

    **Reformulation of the primal-dual gap:** In regard with the exact proximal computation; the inexact case under
    consideration here can be described as performing

        .. math:: x = x_0-\\gamma v + e,

    where :math:`v` is an :math:`\\epsilon`-subgradient of :math:`f` at :math:`x` (notation :math:`v\\in\\partial_{\\epsilon} f(x)`)
    and :math:`e` is some additional computation error. Those elements allow for a common convenient reformulation of
    the primal-dual gap, written in terms of the magnitudes of :math:`\\epsilon` and of :math:`e`:

        .. math:: \\Phi^{(p)}_{\\gamma f}(x;x_0)-\\Phi^{(d)}_{\\gamma f}(v;x_0) = \\frac{1}{2} \|e\|^2 + \\gamma \\epsilon.

    **Options:** The following options are available (a list of such choices is presented in [4]; we provide a reference
    for each of those choices below).

        - 'PD_gapI' : the constraint imposed on the output is the vanilla (see, e.g., [2])

            .. math:: \\Phi^{(p)}_{\\gamma f}(x;x_0)-\\Phi^{(d)}_{\\gamma f}(v;x_0) \\leqslant \\varepsilon.

        This approximation requirement is used in one PEPit example: an accelerated inexact forward backward.

        - 'PD_gapII' : the constraint is stronger than the vanilla primal-dual gap, as more structure is imposed (see, e.g., [1,5]) :

            .. math:: \\Phi^{(p)}_{\\gamma f}(x;x_0)-\\Phi^{(d)}_{\\gamma f}(g;x_0) \\leqslant \\varepsilon,

        where we imposed that :math:`v\\triangleq g\\in\\partial f(x)` and :math:`w\\triangleq x`. This approximation
        requirement is used in two PEPit examples: in a relatively inexact proximal point algorithm and in a partially
        inexact Douglas-Rachford splitting.

        - 'PD_gapIII' : the constraint is stronger than the vanilla primal-dual gap, as more structure is imposed (see, e.g., [3]):

            .. math:: \\Phi^{(p)}_{\\gamma f}(x;x_0)-\\Phi^{(d)}_{\\gamma f}(\\tfrac{x_0 - x}{\\gamma};x_0) \\leqslant \\varepsilon,

        where we imposed that :math:`v \\triangleq \\frac{x_0 - x}{\\gamma}`.

    References:

        `[1] R.T. Rockafellar (1976). Monotone operators and the proximal point algorithm. SIAM journal on control
        and optimization, 14(5), 877-898.
        <https://epubs.siam.org/doi/pdf/10.1137/0314056>`_

        `[2] R.D. Monteiro, B.F. Svaiter (2013). An accelerated hybrid proximal extragradient method for convex
        optimization and its implications to second-order methods. SIAM Journal on Optimization, 23(2), 1092-1125.
        <https://epubs.siam.org/doi/abs/10.1137/110833786>`_

        `[3] S. Salzo, S. Villa (2012). Inexact and accelerated proximal point algorithms.
        Journal of Convex analysis, 19(4), 1167-1192.
        <http://www.optimization-online.org/DB_FILE/2011/08/3128.pdf>`_

        `[4] M. Barre, A. Taylor, F. Bach (2020). Principled analyses and design of
        first-order methods with inexact proximal operators.
        <https://arxiv.org/pdf/2006.06041v3.pdf>`_

        `[5] A. d’Aspremont, D. Scieur, A. Taylor (2021). Acceleration Methods. Foundations and Trends
        in Optimization: Vol. 5, No. 1-2.
        <https://arxiv.org/pdf/2101.09545.pdf>`_

    Args:
        x0 (Point): point for which we aim to compute an approximate proximal step.
        f (Function): function whose proximal operator is approximated.
        gamma (float): step size of the proximal step.
        opt (string): option (type of error requirement) among 'PD_gapI', 'PD_gapII', and  'PD_gapIII'.

    Returns:
        x (Point): the approximated proximal point.
        gx (Point): a (sub)gradient of f at x (subgradient used in evaluating the accuracy criterion).
        fx (Expression): f evaluated at x.
        w (Point): a point w such that v (see next output) is a subgradient of f at w.
        v (Point): the approximated proximal point of the dual problem, (sub)gradient of f evaluated at w.
        fw (Expression): f evaluated at w.
        eps_var (Expression): value of the primal-dual gap (which can be further bounded by the user).

    """

    if opt == 'PD_gapI':
        # This option constrain x and v to satisfy the following primal-dual gap requirement
        # PD_gap(x,v) <= epsVar.
        v = Point()
        w = Point()
        fw = Expression()
        f.add_point((w, v, fw))

        x = Point()
        gx = Point()
        fx = Expression()
        f.add_point((x, gx, fx))

        eps_var = Expression()
        e = x - x0 + gamma * v
        eps_sub = fx - fw - v * (x - w)
        f.add_constraint(e ** 2 / 2 + gamma * eps_sub <= eps_var)

    elif opt == 'PD_gapII':
        # This option constrain x to satisfy the following requirement: ||e||^2 / 2 <= epsVar,
        # with x = x_0 - gamma * g + e.
        e = Point()
        gx = Point()
        x = x0 - gamma * gx + e
        fx = Expression()
        f.add_point((x, gx, fx))
        eps_var = Expression()
        f.add_constraint(e ** 2 / 2 <= eps_var)
        w, v, fw = x, gx, fx

    elif opt == 'PD_gapIII':
        # This option constrain x, v, and w to satisfy the following requirement:
        # gamma * (fx - fw - v*(x - w)) <= epsVar
        x, gx, w = Point(), Point(), Point()
        v = (x0 - x) / gamma
        fw, fx = Expression(), Expression()
        f.add_point((x, gx, fx))
        f.add_point((w, v, fw))
        eps_var = Expression()
        eps_sub = fx - fw - v * (x - w)
        f.add_constraint(gamma * eps_sub <= eps_var)

    else:
        raise ValueError("inexact_proximal_step supports only opt in ['PD_gapI', 'PD_gapII', 'PD_gapIII'],"
                         " got {}".format(opt))

    return x, gx, fx, w, v, fw, eps_var
