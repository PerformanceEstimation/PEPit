Quick start guide
=================

The toolbox implements the **performance estimation approach**, pioneered by Drori and Teboulle [2].
A gentle introduction to performance estimation problems is provided in this
`blog post
<https://francisbach.com/computer-aided-analyses/>`_.

The PEPit implementation is in line with the framework as exposed in [3,4]
and follow-up works (for which proper references are provided in the example files).
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

PEPit is available on pypi, hence can be installed very simply by running

``pip install pepit``

To solve the SDPs, PEPit relies on CVXPY [5, 6]. Please also run

``pip install cvxpy``

Now you are all set!
You should be able to run

.. code-block::

    import PEPit

in an Python interpreter.


Basic usage: getting worst-case guarantees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main object is called a **PEP**.
It stores the problem you will describe to **PEPit**.

First create a **PEP** object.

.. code-block::

    from PEPit import PEP


.. code-block::

    problem = PEP()


From now, you can declare functions thanks to the `declare_function` method.

.. code-block::

    func = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': 0, 'L': L})


You can also define a new point with

.. code-block::

    x0 = problem.set_initial_point()


and give a name to the value of `func` on `x0`

.. code-block::

    f0 = func.value(x0)


as well as the (sub)gradient of `func` on `x0`

.. code-block::

    g0 = func.gradient(x0)


or

.. code-block::

    g0 = func.subgradient(x0)


There is a more compact way to do it using the `oracle` method.

.. code-block::

    g0, f0 = func.oracle(x0)

You can declare a stationary point of `func`, defined as a point which gradient on `func` is zero, as follow:

.. code-block::

    xs = func.stationary_point()


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

    problem.set_performance_metric(func.value(x) - fs)


Finally, you can ask PEPit to solve the system for you and return the worst-case guarantee of your method.

.. code-block::

    pepit_tau = problem.solve()


Derive proofs and adversarial objectives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When one can the `solve` method,
**PEPit** does much more that just finding the worst-case value.

In particular, it stores possible values of each points, gradients and function values that achieve this worst-case guarantee,
as well as the dual variable values associated with each constraint.

Values and dual variables values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's consider the above example.
After solving the **PEP**, you can ask **PEPit**

.. code-block::

    print(x.value)

which returns one possible value of the output of the described algorithm at optimum.

You can also ask for gradients and function values

.. code-block::

    print(func.gradient(x).value)
    print(func.value(x).value)

Recovering the values of all the points,
gradients and function values at optimum allows you
to reconstruct the function that achieves the worst-case complexity of your method.

You can also get the dual variables values of constraints at optimum,
which essentially allows you to write the proof of the worst-case guarantee you just obtained.

Let's consider again the previous example, but this time,
let's give a name to a constraint before using it.

.. code-block::

    constraint = (x0 - xs) ** 2 <= 1
    problem.set_initial_condition(constraint)

Then, after solving the system, you can require its associated dual variable value with

.. code-block::

    constraint.dual_variable_value

Output pdf
~~~~~~~~~~

In a latter release, we will provide an option to output a pdf file summarizing all those pieces of information.

Simplify proofs
^^^^^^^^^^^^^^^

Sometimes, there are several solutions to the PEP problem.
In order to simplify the proof, one would prefer a low dimension solution.
To this end, we provide an **heuristic** based on the trace to reduce the dimension of the provided solution.

You can use it  by specifying

.. code-block::

    problem.solve(tracetrick=True)

Finding Lyapunov
^^^^^^^^^^^^^^^^

In a latter release, we will provide tools to help finding good Lyapunov functions to study a given method.

This tool will be based on the very recent work [7].

References
----------

[1] B. Goujaud, C. Moucer, F. Glineur, J. Hendrickx, A. Taylor, A. Dieuleveut. "PEPit: computer-assisted worst-case analyses of first-order optimization methods in Python."

[2] Drori, Yoel, and Marc Teboulle. "Performance of first-order methods for smooth convex minimization: a novel approach." Mathematical Programming 145.1-2 (2014): 451-482

[3] Taylor, Adrien B., Julien M. Hendrickx, and François Glineur. "Smooth strongly convex interpolation and exact worst-case performance of first-order methods." Mathematical Programming 161.1-2 (2017): 307-345.

[4] Taylor, Adrien B., Julien M. Hendrickx, and François Glineur. "Exact worst-case performance of first-order methods for composite convex optimization." SIAM Journal on Optimization 27.3 (2017): 1283-1313.

[5] Steven Diamond and Stephen Boyd. "CVXPY: A Python-embedded modeling language for convex optimization." Journal of Machine Learning Research (JMLR) 17.83.1--5 (2016).

[6] Agrawal, Akshay and Verschueren, Robin and Diamond, Steven and Boyd, Stephen. "A rewriting system for convex optimization problems." Journal of Control and Decision (JCD) 5.1.42--60 (2018).

[7] Adrien Taylor, Bryan Van Scoy, Laurent Lessard. "Lyapunov Functions for First-Order Methods: Tight Automated Convergence Guarantees." International Conference on Machine Learning (ICML).
