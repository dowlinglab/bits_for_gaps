# VLE / distillation case study

The paper's H2O-PrOH vapor-liquid-equilibrium case study, ported onto the public
`bits_for_gaps` API: a Wilson activity-coefficient model (via Julia/Clapeyron.jl),
Gibbs-Duhem correction, bubble-point/dew-point phase diagram, and a McCabe-Thiele
distillation column solver.

**This example is repo-only** -- it is *not* part of the `bits_for_gaps` package
distributed on PyPI (the pip wheel ships only `bits_for_gaps/*`; see
`REFACTOR_PLAN.md` §7.3). To run it you need a clone of this repository, not just
`pip install bits_for_gaps`.

## Setup

```bash
git clone <this-repo>
cd bits_for_gaps
conda env create -f environment.yml   # or: conda activate bits_for_gaps if it exists already
conda activate bits_for_gaps
pip install -e ".[dev,vle]"           # the [vle] extra adds juliacall/juliapkg
```

**macOS -- mandatory:**

```bash
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
```

Set this in every shell **before** anything imports `juliacall` (it has to be set
before the Python process starts -- setting it later, e.g. inside a script, is not
reliable). Without it, `juliacall` intermittently SIGBUSes on macOS because Python and
Julia's own signal handlers collide. `activity_model.py` also sets this defensively at
import time via `os.environ.setdefault`, but the shell-level `export` is the real fix.

## Julia + Clapeyron.jl bootstrap (one-time, automatic)

You do **not** need to install Julia yourself. The first time any code here imports
`juliacall` (e.g. the first call to `activity_model.activity_coefficients`), `juliapkg`
automatically downloads a Julia toolchain into a per-conda-env directory
(`$CONDA_PREFIX/julia_env/`) and resolves the package dependencies declared in
`juliapkg.json` (next to this README). This first run downloads/precompiles Julia
packages and can take a few minutes; subsequent runs are fast.

The Clapeyron.jl version is **pinned** in `juliapkg.json`:

```json
{"packages": {"Clapeyron": {"uuid": "7c7805af-...", "version": "=0.6.26"}}}
```

`juliapkg` discovers this file automatically once `examples/` is on `sys.path` (it
scans every `<sys.path entry>/<subdir>/juliapkg.json` -- see `tests/conftest.py`, which
does this for the test suite; `run_case_study.py` does the equivalent `sys.path` insert
itself when run as a script). No manual `Pkg.add` step is needed on a fresh machine.

## Run the case study

```bash
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
python examples/vle_distillation/run_case_study.py
```

This runs the full pipeline: a Latin-hypercube initial design over
`(z_PrOH, T) ∈ [10⁻⁶, 0.999] × [350, 367] K` → Clapeyron (Wilson) activity-coefficient
evaluation at each point → `BitsForGaps.run` (adaptive, entropy-driven design) → a
Wilson-vs-surrogate phase diagram → a Geankoplis Ex. 11.4-1 McCabe-Thiele distillation
column solve. It prints the resulting per-stage liquid/vapor PrOH mole fractions for
both the Wilson ground truth and the GP surrogate. Runtime is a couple of minutes
(dominated by the adaptive design's HMC sampling, a handful of iterations by default --
see `N_ITERS` in the script; the paper's published run used 15).

This is a *demonstration* of the ported pipeline, not a reproduction of the paper's
exact published figures (that full reproduction, matching the paper's real 15-iteration
adaptive run, is Phase 7's job -- see `HANDOFF.md`).

## Sanity check

A quick way to confirm Julia/Clapeyron are wired up correctly, independent of the full
case study:

```bash
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
PYTHONPATH=examples python -c "
from vle_distillation.activity_model import activity_coefficients
print(activity_coefficients(0.5, 350.0))   # -> (1.469..., 1.746...)
"
```

(`PYTHONPATH=examples` puts the example package on the import path -- `run_case_study.py`
does the equivalent `sys.path` insert itself, and the test suite gets it from
`tests/conftest.py`, so this is only needed for ad hoc one-liners like this.)

Expected: Wilson `gamma(PrOH/water, T=350 K, x_PrOH=0.5) ≈ (1.469, 1.746)`.

## Module map

| Module | What it does |
|---|---|
| `activity_model.py` | Wilson activity coefficients via Clapeyron.jl (lazy Julia import) |
| `gibbs_duhem.py` | Recovers gamma_H2O from a modeled gamma_PrOH curve (pure NumPy) |
| `phase_diagram.py` | Antoine vapor pressures + bubble-point/dew-point solve (pure NumPy + lazy Julia via `wilson_gamma`) |
| `equilibrium.py` | Wraps a VLE curve as an `x_liquid -> y_vapor` interpolant (pure NumPy) |
| `distillation.py` | McCabe-Thiele stage-by-stage column solver (pure NumPy/SciPy) |
| `run_case_study.py` | Ties it all together: LHS design → Clapeyron → `BitsForGaps.run` → phase diagram → column |

Everything except `activity_model.py` (and `phase_diagram.wilson_gamma`, which calls
it) is pure Python/NumPy/SciPy -- importable and unit-testable without Julia. See
`tests/unit/test_gibbs_duhem.py`, `test_phase_diagram.py`, `test_equilibrium.py`, and
`test_distillation.py` for the no-Julia tests, and
`tests/regression/test_mccabe_thiele.py` (`@pytest.mark.vle`) for the Julia-backed
end-to-end regression against `paper/golden/mccabe_thiele_stages.json`.
