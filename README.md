# BITS for GAPS

[![CI](https://github.com/dowlinglab/bits_for_gaps/actions/workflows/ci.yml/badge.svg)](https://github.com/dowlinglab/bits_for_gaps/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/bits_for_gaps.svg)](https://pypi.org/project/bits_for_gaps/)
[![Docs](https://readthedocs.org/projects/bits-for-gaps/badge/?version=latest)](https://bits-for-gaps.readthedocs.io/en/latest/?badge=latest)

**B**ayesian **I**nformation-**T**heoretic **S**ampling for hierarchical **GA**ussian **P**rocess **S**urrogates.

A framework for information-theoretic sequential experimental design with Bayesian
hierarchical Gaussian-process surrogates. Prior physical knowledge is encoded through
priors on the GP hyperparameters; sampling is guided by maximizing the predictive
differential entropy, so hyperparameter uncertainty (not just predictive variance)
drives data acquisition.

Reference: K. D. Jones and A. W. Dowling, "BITS for GAPS: Bayesian Information-Theoretic
Sampling for hierarchical GAussian Process Surrogates," *Computers & Chemical
Engineering* **211** (2026) 109650. https://doi.org/10.1016/j.compchemeng.2026.109650

## Install

```bash
pip install bits_for_gaps
```

The **core library is pure Python** (GPflow / TensorFlow / NumPy / SciPy) with no Julia
dependency. Julia + Clapeyron are only needed for the `vle_distillation` example, which
isn't part of the PyPI package -- see "From source" below.

**macOS note:** set `export PYTHON_JULIACALL_HANDLE_SIGNALS=yes` before importing
`juliacall`, or Julia crashes with a bus error (SIGBUS).

### From source (for `examples/`, `paper/`, and development)

```bash
git clone https://github.com/dowlinglab/bits_for_gaps
cd bits_for_gaps
conda env create -f environment.yml
conda activate bits_for_gaps
pip install -e ".[dev]"          # core + test tools
# pip install -e ".[dev,vle]"    # add the Julia/Clapeyron VLE example backend
```

## Layout

```
src/bits_for_gaps/   the library (algorithm)
examples/            worked examples (incl. the paper's VLE/distillation case study)
paper/               scripts + reference metrics to reproduce the published figures
tests/               unit / integration / regression tests
docs/                Sphinx documentation (ReadTheDocs)
```

## Quick test

```bash
pytest -q
```

## Docs

Full docs (installation, a pure-Python quickstart, theory notes, the VLE example,
reproducing the paper's figures, and the API reference):
https://bits-for-gaps.readthedocs.io

To build and browse locally instead:

```bash
pip install -e ".[docs]"
sphinx-build -W docs docs/_build/html
open docs/_build/html/index.html   # or your platform's equivalent
```
