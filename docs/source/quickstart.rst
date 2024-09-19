Quick start guide
=================

The toolbox implements the **performance estimation approach**, pioneered by Drori and Teboulle [2].
A gentle introduction to performance estimation problems is provided in this
`blog post
<https://francisbach.com/computer-aided-analyses/>`_.

The PEPit implementation is in line with the framework as exposed in [3,4]
and follow-up works (for which proper references are provided in the `example files
<https://pepit.readthedocs.io/en/latest/examples.html#>`_).
A gentle introduction to the toolbox is provided in [1].

When to use PEPit?
-------------------

The general purpose of the toolbox is to help the researchers producing worst-case guarantees
for their favorite first-order methods.

This toolbox is presented under the form of a Python package.
For people who are more comfortable with Matlab, we report to
`PESTO
<https://github.com/AdrienTaylor/Performance-Estimation-Toolbox>`_.

How tu use PEPit?
------------------

Installation
^^^^^^^^^^^^

PEPit is available on PyPI, hence can be installed very simply by running the command line:

``pip install pepit``

Now you are all set!
You should be able to run

.. code-block::

    import PEPit

in an Python interpreter.


Basic usage: getting worst-case guarantees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main object is called a **PEP**.
It stores the problem you will describe to **PEPit**.

First create a `**PEP**
<https://pepit.readthedocs.io/en/latest/api/main_modules.html#PEPit.PEP>`_ object.

.. code-block::

    from PEPit import PEP


.. code-block::

    problem = PEP()


From now, you can declare `functions
<https://pepit.readthedocs.io/en/latest/api/functions_and_operators.html>`_ thanks to the `declare_function
<https://pepit.readthedocs.io/en/latest/api/main_modules.html#PEPit.PEP.declare_function>`_ method.

.. code-block::

    from PEPit.functions import SmoothConvexFunction

.. code-block::

    func = problem.declare_function(SmoothConvexFunction, L=1)

.. warning::
    To enforce the same subgradient to be returned each time one is required,
    we introduced the attribute `reuse_gradient` in the `Function
    <https://pepit.readthedocs.io/en/0.3.2/api/main_modules.html#function>`_ class.
    Some classes of functions contain only differentiable functions (e.g. smooth convex functions).
    In those, the `reuse_gradient` attribute is set to True by default.

    When the same subgradient is used several times in the same code and when it is difficult to
    to keep track of it (through proximal calls for instance), it may be useful to set this parameter
    to True even if the function is not differentiable. This helps reducing the number of constraints,
    and improve the accuracy of the underlying semidefinite program. See for instance the code for
    `improved interior method 
    <https://pepit.readthedocs.io/en/latest/examples/b.html#improved-interior-method>`_ or
    `no Lips in Bregman divergence
    <https://pepit.readthedocs.io/en/latest/examples/b.html#no-lips-in-bregman-divergence>`_.

You can also define a new `point
<https://pepit.readthedocs.io/en/0.3.2/api/main_modules.html#point>`_ with

.. code-block::

    x0 = problem.set_initial_point()


and store the value of `func` on `x0`

.. code-block::

    f0 = func(x0)

or

.. code-block::

    f0 = func.value(x0)


as well as the (sub)gradient of `func` on `x0`

.. code-block::

    g0 = func.gradient(x0)

or

.. code-block::

    g0 = func.subgradient(x0)


There is a more compact way to do it using the `oracle
<https://pepit.readthedocs.io/en/0.3.2/api/main_modules.html#PEPit.Function.oracle>`_ method.

.. code-block::

    g0, f0 = func.oracle(x0)

You can declare a stationary point of `func`, defined as a point which gradient on `func` is zero, as follow:

.. code-block::

    xs = func.stationary_point()

Then you can define the associated function value using:

.. code-block::

    fs = func(xs)

Alternatively, you can use an option of the `stationary_point
<https://pepit.readthedocs.io/en/0.3.2/api/main_modules.html#PEPit.Function.stationary_point>`_ method to get the stationary point and properties of func on the latter.

.. code-block::

    xs, gs, fs = func.stationary_point(return_gradient_and_function_value=True)


You can combine points and gradients naturally

.. code-block::

    x = x0
    for _ in range(n):
        x = x - gamma * func.gradient(x)

You must declare some initial conditions like

.. code-block::

    problem.set_initial_condition((x0 - xs) ** 2 <= 1)


as well as performance metrics like

.. code-block::

    problem.set_performance_metric(func(x) - fs)


Finally, you can ask PEPit to solve the system for you and return the worst-case guarantee of your method.

.. code-block::

    pepit_tau = problem.solve()

.. warning::
    Performance estimation problems consist in reformulating the problem of finding a worst-case scenario as a semidefinite
    program (SDP). The dimension of the corresponding SDP is directly related to the number of function and gradient evaluations
    in a given code.
    
    We encourage the users to perform as few function and subgradient evaluations as possible, as the size of the
    corresponding SDP grows with the number of subgradient/function evaluations at different points.


Derive proofs and adversarial objectives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When one calls the `solve
<https://pepit.readthedocs.io/en/0.3.2/api/main_modules.html#PEPit.PEP.solve>`_ method,
**PEPit** does much more that just finding the worst-case value.

In particular, it stores possible values of each points, gradients and function values that achieve this worst-case guarantee,
as well as the dual variable values associated with each constraint.

Values and dual variables values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's consider the above example.
After solving the **PEP**, you can ask **PEPit**

.. code-block::

    print(x.eval())

which returns one possible value of the output of the described algorithm at optimum.

You can also ask for gradients and function values

.. code-block::

    print(func.gradient(x).eval())
    print(func(x).eval())

Recovering the values of all the points,
gradients and function values at optimum allows you
to reconstruct the function that achieves the worst-case complexity of your method.

You can also get the dual variables values of constraints at optimum,
which essentially allows you to write the proof of the worst-case guarantee you just obtained.

Let's consider again the previous example, but this time,
let's store a constraint before using it.

.. code-block::

    constraint = (x0 - xs) ** 2 <= 1
    problem.set_initial_condition(constraint)

Then, after solving the system, you can require its associated dual variable value with

.. code-block::

    constraint.eval_dual()

Naming PEPit objects
~~~~~~~~~~~~~~~~~~~~

In order to ease the proof reconstruction, PEPit now allows to associate names to the created objects.
This is particularly useful on `constraints
<https://pepit.readthedocs.io/en/0.3.2/api/main_modules.html#constraint>`_ in order to associate the found dual values to some recognisable constraints.

As an example, if a user creates several constraints in a row as

.. code-block::

    for _ in range(n):
        constraint = ...
        constraint.set_name(name)
        problem.add_constraint(constraint)

the latter can easily list their names in front of their dual values with

.. code-block::

    for constraint in problem.list_of_constraints:
        print("the constraint {} comes with the dual values {}.".format(constraint.get_name(), constraint.eval_dual()))

Functions generally contain several "interpolation constraints".
If a user sets a name to a function as well as to all the points the oracle has been called on,
then, its interpolation constraints will be attributed a name accordingly.
Then, using the method `get_class_constraints_duals
<https://pepit.readthedocs.io/en/0.3.2/api/main_modules.html#PEPit.Function.get_class_constraints_duals>`_,
the user has access to the tables of dual values related to its interpolation constraints.

Output pdf
~~~~~~~~~~

In a later release, we will provide an option to output a pdf file summarizing all those pieces of information.

Simpler worst-case scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes, there are several solutions to the PEP problem.
For obtaining simpler worst-case scenarios, one would prefer a low dimension solutions to the SDP.
To this end, we provide **heuristics** based on the trace norm or log det minimization for reducing
the dimension of the numerical solution to the SDP.

You can use the trace heuristic by specifying

.. code-block::

    problem.solve(dimension_reduction_heuristic="trace")
    
You can use n iterations of the log det heuristic by specifying "logdet{n}". For example, for
using 5 iterations of the logdet heuristic:

.. code-block::

    problem.solve(dimension_reduction_heuristic="logdet5")


Finding Lyapunov
^^^^^^^^^^^^^^^^

In a later release, we will provide tools to help finding good Lyapunov functions to study a given method.

This tool will be based on the method described in [7].

References
----------

[1] B. Goujaud, C. Moucer, F. Glineur, J. Hendrickx, A. Taylor, A. Dieuleveut.
`PEPit: computer-assisted worst-case analyses of first-order optimization methods in Python.
<https://arxiv.org/pdf/2201.04040.pdf>`_

[2] Drori, Yoel, and Marc Teboulle.
`Performance of first-order methods for smooth convex minimization: a novel approach.
<https://arxiv.org/pdf/1206.3209.pdf>`_
Mathematical Programming 145.1-2 (2014): 451-482

[3] Taylor, Adrien B., Julien M. Hendrickx, and François Glineur.
`Smooth strongly convex interpolation and exact worst-case performance of first-order methods.
<https://arxiv.org/pdf/1502.05666.pdf>`_
Mathematical Programming 161.1-2 (2017): 307-345.

[4] Taylor, Adrien B., Julien M. Hendrickx, and François Glineur.
`Exact worst-case performance of first-order methods for composite convex optimization.
<https://arxiv.org/pdf/1512.07516.pdf>`_
SIAM Journal on Optimization 27.3 (2017): 1283-1313.

[5] Steven Diamond and Stephen Boyd.
`CVXPY: A Python-embedded modeling language for convex optimization.
<https://arxiv.org/pdf/1603.00943.pdf>`_
Journal of Machine Learning Research (JMLR) 17.83.1--5 (2016).

[6] Agrawal, Akshay and Verschueren, Robin and Diamond, Steven and Boyd, Stephen.
`A rewriting system for convex optimization problems.
<https://arxiv.org/pdf/1709.04494.pdf>`_
Journal of Control and Decision (JCD) 5.1.42--60 (2018).

[7] Adrien Taylor, Bryan Van Scoy, Laurent Lessard.
`Lyapunov Functions for First-Order Methods: Tight Automated Convergence Guarantees.
<https://arxiv.org/pdf/1803.06073.pdf>`_
International Conference on Machine Learning (ICML).
