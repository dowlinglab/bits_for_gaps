# Installation

`bits_for_gaps`'s **core is pure Python** (NumPy, SciPy, GPflow, TensorFlow,
TensorFlow Probability) -- it has no Julia dependency. Julia is only needed for the
optional VLE/distillation example, and even then only when you actually call into it.

## Core library

```bash
pip install bits_for_gaps
```

```{admonition} Frozen dependency stack, Python 3.9-3.12
:class: note
The core pins an exact, verified-working dependency stack rather than floating version
ranges: NumPy 1.26, SciPy 1.13, GPflow 2.9.2, TensorFlow 2.16.2, TensorFlow Probability
0.24.0. This is a deliberate reproducibility choice -- GPflow's TensorFlow dependency
makes casual version bumps risky, so modernizing the stack is left as a separate, later
effort. The package itself supports **Python 3.9 through 3.12**, verified with a real
test-suite run (including the exact-value regression pins) on every one of those four
versions.
```

```{admonition} Why not Python 3.13+?
:class: important
This is an **upstream constraint**, not a project choice: `bits_for_gaps` depends on
GPflow, and GPflow requires `numpy<2` in every release it has ever published, including
the latest (2.11.1). No 1.x release of NumPy publishes a Python 3.13 wheel -- the first
NumPy version that does is 2.1.0. So GPflow (and therefore this package) cannot run on
Python 3.13 at all, regardless of what this package's own pins say. TensorFlow's newest
release similarly has no Python 3.14 wheels yet. Python 3.12 is the ceiling until GPflow
adds NumPy 2 support upstream -- no timeline is promised or implied.
```

Verify it imports (no Julia touched):

```bash
python -c "import bits_for_gaps; print(bits_for_gaps.__version__)"
```

## From source (for `examples/`, `paper/`, or development)

```bash
git clone https://github.com/dowlinglab/bits_for_gaps
cd bits_for_gaps
conda env create -f environment.yml   # Python 3.12 + the pinned stack (see the file's
                                       # header comment for using 3.9 instead)
conda activate bits_for_gaps
pip install -e .
```

## Development install

```bash
pip install -e ".[dev]"     # adds pytest, pytest-cov, ruff
pytest -q                   # 218 passed, 2 deselected
```

The 2 deselected tests need Julia (`@pytest.mark.vle`) -- run them with
`pytest -m vle` after installing the `[vle]` extra below. See {doc}`reproduce_paper`.

```{admonition} Coverage
:class: note
`pytest -q` alone never measures coverage -- it stays fast and its output stays
unchanged for a plain local run. Measure it explicitly, scoped to the shipped
package (`examples/`, `paper/`, and `tests/` are repo-only and excluded):

    pytest --cov=bits_for_gaps --cov-report=term-missing

CI measures coverage this way on one Python version (3.12) per run and uploads it to
[Codecov](https://codecov.io/gh/dowlinglab/bits_for_gaps) (works without a token for
this public repo); `pyproject.toml`'s `[tool.coverage.report]` sets a `fail_under`
floor so coverage can't silently regress.
```

## `examples/` and `paper/` are repo-only

```{important}
The paper's worked example (`examples/vle_distillation/`) and figure-reproduction
scripts (`paper/`) are **not part of the `bits_for_gaps` PyPI package** -- the wheel
ships only `src/bits_for_gaps/`, to keep the installed package lightweight. To use
either, **clone the repository**; `pip install bits_for_gaps` alone does not give you
them.
```

Once cloned, `examples/` and `paper/` are importable as regular Python packages
without a separate install step -- `tests/conftest.py` puts the repo's `examples/`
directory and the repo root on `sys.path` for the test suite, and each entry-point
script (`examples/vle_distillation/run_case_study.py`, `paper/reproduce.py`) does the
equivalent `sys.path` insertion itself so it works when run directly too.

## Optional: the VLE/distillation example (Julia)

The paper's H2O-PrOH case study (`examples/vle_distillation/`) uses
[Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl) via
[`juliacall`](https://github.com/JuliaPy/PythonCall.jl) for the Wilson
activity-coefficient ground truth. This is entirely optional and lazily imported --
`import bits_for_gaps` and `import vle_distillation.<module>` never touch Julia;
only *calling* the activity model does.

```bash
pip install -e ".[vle]"     # adds juliacall, juliapkg
```

```{caution}
**macOS:** set this **before** anything imports `juliacall` in a given process (it
must be set before the Python interpreter starts, not from inside a script):

    export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

Without it, `juliacall` intermittently crashes with a bus error (SIGBUS) because
Python's and Julia's signal handlers collide.
```

The first call into `activity_model.py` auto-downloads a Julia toolchain and the
pinned Clapeyron.jl version (see `examples/vle_distillation/juliapkg.json`) into a
per-conda-env directory -- no manual Julia install needed. Full setup and a sanity
check: `examples/vle_distillation/README.md`.

## Docs (building this site locally)

```bash
pip install -e ".[docs]"
sphinx-build -W docs docs/_build/html
open docs/_build/html/index.html   # or your platform's equivalent
```

`docs/_build/` is gitignored -- nothing here gets committed.
