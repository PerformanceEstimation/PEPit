# PEPit

## Install the code

This code run under Python 3.6+.

- Please run ``pip install -r Infrastructure.requirements.txt``
- Then run ``pip install -e .``

You are all set!

## Convention of the code

- ``PEPit`` directory contains the main code while ``Tests`` directory contains all the associated tests.
- We use PEP8 convention rules.

## Convention of the VCS

- The ``master`` branch is exclusively used for deployed versions of the code.
- The ``develop`` branch must be the main one and must not be broken at any time.
- The other branches are named either ``feature/...`` or ``fix/..`` or eventually ``hotfix/..`` to highlight the importance of the PR.
- All branches must be approved before merge. We use PRs and the ``git rebase`` command to sync any branch on ``develop``.

## Documentation

# TODO Remove all the above and replace by quick starter (+ add our names somewhere).
