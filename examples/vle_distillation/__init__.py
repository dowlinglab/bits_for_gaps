"""The paper's H2O-PrOH VLE / distillation case study, built on the public
``bits_for_gaps`` API.

Not shipped in the ``bits_for_gaps`` pip wheel -- repo-only, importable in dev/CI via
``tests/conftest.py``'s ``sys.path`` insert of the repo's ``examples/`` directory (see
that file). Julia/Clapeyron (``activity_model.py``) are an optional, lazily-imported
dependency: everything else here imports without Julia installed.
"""
