Contributing
============

PEPit is designed for allowing users to easily contribute to add new features to the package.
Classes of functions (or operators) as well as black-box oracles can be implemented
by following the canvas from respectively
`PEPit/functions/
<https://pepit.readthedocs.io/en/latest/api/functions.html>`_
(or `PEPit/operators/
<https://pepit.readthedocs.io/en/latest/api/operators.html>`_
and `PEPit/primitive_steps/
<https://pepit.readthedocs.io/en/latest/api/steps.html>`_).

We encourage authors of research papers presenting novel optimization methods
and/or a novel convergence results to submit the corresponding PEPit files in the directory
`PEPit/examples/
<https://pepit.readthedocs.io/en/latest/examples.html>`_.

.. contents::
   :depth: 1
   :local:

General guidelines
------------------

We kindly ask you follow common guidelines, namely that the provided code:

- sticks as much as possible to the PEP8 convention.

- is commented with Google style docstring.

- is well covered by tests.

- is aligned with the documentation.

- is also mentioned in the ``whatsnew'' section of the documentation.

Adding a new function or operator class
---------------------------------------

To add a new function / operator class,
please follow the format used for the other function / operator classes.

In particular:

- your class must inherit from the class ``Function`` and overwrite its ``add_class_constraints`` method.

- the docstring must be complete.
  In particular, it must contains the list of attributes and arguments
  as well as an example of usage via the ``declare_function`` method of the class ``PEP``.
  It must also contain a clickable reference to the paper introducing it.

Adding a step / an oracle
-------------------------

To add a new oracle / step,
please add a new file containing the oracle function in ``PEPit/primitive_steps``.

Remark that transforming the mathematical formulation of an oracle into its PEP equivalent
may require additional tricks,
see e.g. `PEPit/primitive_steps/proximal_step.py
<https://pepit.readthedocs.io/en/latest/api/steps.html#proximal-step>`_,
or `PEPit/primitive_steps/linear_optimization_step.py
<https://pepit.readthedocs.io/en/latest/api/steps.html#linear-optimization-step>`_.

Please make sure that your docstring contains the mathematical derivation of the latest from the previous.

Adding a new method as an example
---------------------------------

We don't require a specific code format for a new example.
However, we ask the associated docstring to be precisely organized as follow:

- Define Problem solved (introducing function notations and assumptions).

- Name method in boldface formatting.

- Introduce performance metric, initial condition and parameters (``performance_metric < tau(parameters) initialization``).

- Describe method main step and cite reference with specified algorithm.

- Provide theoretical result (``Upper/Lower/Tight`` in boldface formatting + ``performance_metric < theoretical_bound initialization``).

- Reference block containing relevant clickable references (preferably to arxiv with specified version of the paper) in the format:
  (``First name initial letter``, ``last name`` (``YEAR``). ``Title``. ``Journal or conference`` (``Acronym of journal or conference``).

- Args block containing parameters with their type and short description.

- Returns block containing ``pepit_tau`` and ``theoretical_tau``.

- Example block containing a minimal work example of the coded function.

We provide, in ``PEPit/examples/example_template.py``, a template that can be filled very quickly
to help the contributor to share their method easily.

New example template
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: PEPit.examples.example_template.wc_example_template

New example test template
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def test_[NAME_METHOD](self):
        PARAMS = PARAMS

        wc, theory = wc_[NAME_METHOD](PARAMS=PARAMS, verbose=self.verbose)

        # If theoretical upper bound is tight
        self.assertAlmostEqual(theory, wc, delta=self.relative_precision * theory)

        # If theoretical upper bound is not tight
        self.assertLessEqual(wc, theory * (1 + self.relative_precision))

        # If theoretical lower bound is not tight
        self.assertLessEqual(theory, wc * (1 + self.relative_precision))
