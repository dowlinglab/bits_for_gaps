# Installation

`bits_for_gaps`'s **core is pure Python** (NumPy, SciPy, GPflow, TensorFlow,
TensorFlow Probability) -- it has no Julia dependency. Julia is only needed for the
optional VLE/distillation example, and even then only when you actually call into it.

## Core library

```{admonition} Frozen dependency stack
:class: note
The core pins an exact, verified-working stack rather than floating version ranges:
Python 3.9, NumPy 1.26, SciPy 1.13, GPflow 2.9.2, TensorFlow 2.16.2, TensorFlow
Probability 0.24.0. This is a deliberate reproducibility choice (see
`REFACTOR_PLAN.md` §7 decision 6) -- GPflow's TensorFlow dependency makes casual
version bumps risky, so modernizing the stack is left as a separate, later effort.
```

Once published (see `HANDOFF.md` for current status -- this repo is pre-1.0):

```bash
pip install bits_for_gaps
```

**From source** (the current way to get it, and the way to get `examples/` and
`paper/` -- see below):

```bash
git clone https://github.com/dowlinglab/bits_for_gaps
cd bits_for_gaps
conda env create -f environment.yml   # Python 3.9 + the pinned stack
conda activate bits_for_gaps
pip install -e .
```

Verify it imports (no Julia touched):

```bash
python -c "import bits_for_gaps; print(bits_for_gaps.__version__)"
```

## Development install

```bash
pip install -e ".[dev]"     # adds pytest, pytest-cov
pytest -q                   # 193 passed, 2 deselected
```

The 2 deselected tests need the private archived published run and/or Julia -- see
{doc}`reproduce_paper` and `HANDOFF.md`.

## `examples/` and `paper/` are repo-only

```{important}
The paper's worked example (`examples/vle_distillation/`) and figure-reproduction
scripts (`paper/`) are **not part of the `bits_for_gaps` PyPI package** -- the wheel
ships only `src/bits_for_gaps/`, to keep the installed package lightweight
(`REFACTOR_PLAN.md` §7 decision 3). To use either, **clone the repository**; `pip
install bits_for_gaps` alone does not give you them.
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
