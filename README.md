# BITS for GAPS

[![CI](https://github.com/dowlinglab/bits_for_gaps/actions/workflows/ci.yml/badge.svg)](https://github.com/dowlinglab/bits_for_gaps/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/bits_for_gaps.svg)](https://pypi.org/project/bits_for_gaps/)
[![Docs](https://readthedocs.org/projects/bits-for-gaps/badge/?version=latest)](https://bits-for-gaps.readthedocs.io/en/latest/?badge=latest)

**B**ayesian **I**nformation-**T**heoretic **S**ampling for hierarchical **GA**ussian **P**rocess **S**urrogates.

![BITS for GAPS framework overview: a hierarchical Gaussian process surrogate (GAPS) places priors on the kernel hyperparameters; Bayesian information-theoretic sampling (BITS) repeatedly selects the next input by maximizing the predictive differential entropy and collects data there; the result is an information-optimal surrogate whose predictions average over the hyperparameter posterior.](https://raw.githubusercontent.com/dowlinglab/bits_for_gaps/main/docs/graphical_abstract.jpg)

*Graphical abstract from the paper (© 2026 The Authors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).*

A framework for information-theoretic sequential experimental design with Bayesian
hierarchical Gaussian-process surrogates. Prior physical knowledge is encoded through
priors on the GP hyperparameters; sampling is guided by maximizing the predictive
differential entropy, so hyperparameter uncertainty (not just predictive variance)
drives data acquisition.

Reference: K. D. Jones and A. W. Dowling, "BITS for GAPS: Bayesian Information-Theoretic
Sampling for hierarchical GAussian Process Surrogates," *Computers & Chemical
Engineering* **211** (2026) 109650. https://doi.org/10.1016/j.compchemeng.2026.109650

The paper is **bundled in this repository** so you can read the method alongside the code:
[`paper/bits_for_gaps_paper.pdf`](paper/bits_for_gaps_paper.pdf). It is redistributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (© 2026 The Authors, published by
Elsevier Ltd); the DOI above is the canonical citation.

## Install

```bash
pip install bits_for_gaps
```

The **core library is pure Python** (GPflow / TensorFlow / NumPy / SciPy) with no Julia
dependency. Julia + Clapeyron are only needed for the `vle_distillation` example, which
isn't part of the PyPI package -- see "From source" below.

Supports **Python 3.9-3.12**. Python 3.13+ isn't available: this package depends on
GPflow, and GPflow requires `numpy<2` in every release -- no NumPy 1.x publishes a
Python 3.13 wheel. That's an upstream constraint, not something this package can work
around; see [`docs/installation.md`](docs/installation.md) for the full explanation.

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

## Provenance

The research code behind the paper was originally developed in a private repository over the
course of the study. It was then migrated here and reorganized into an installable, tested
package: the algorithm was separated from the vapor–liquid-equilibrium case study, generalized
to arbitrary input dimension, and covered by a test suite. That private repository holds only
the development history — **nothing you need to use this package or to reproduce the paper's
figures is missing from this repository.** The data the figure scripts read is committed here
under `paper/data/` (see [`paper/REPRODUCTION.md`](paper/REPRODUCTION.md)).

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
