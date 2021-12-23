Contributing
============

PEPit allows to study a lot of optimization methods.
Some of them can be found in ``PEPit.examples``.
If you publish or have published a result on a new method or a new guarantee of an existing method,
you are very welcome to contribute to PEPit by adding a file in ``PEPit/examples``.

You are also welcome to add features you might need to the pipeline.

General guidelines
------------------

We only ask you follow common guidelines, namely that the provided code:

- sticks as much as possible to the PEP8 convention.

- is commented with Google style docstring.

- is well tested.

For new example, the guidelines follow.

Creating new example
--------------------

We don't require a specific code format for a new example.
However, we ask the associated docstring to be organized as follow:

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

New example template
^^^^^^^^^^^^^^^^^^^^

We provide an example template in ``PEPit/examples/example_template.py``.

.. autofunction:: PEPit.examples.example_template.wc_example_template
