# Reproducing the paper's figures

All 11 of the paper's figures (2 through 12) regenerate through `paper/reproduce.py` +
`paper/figures/` -- **repo-only** scripts (see {doc}`installation`), not part of the
pip package.

```{note}
This page is a pointer, not a copy -- the authoritative, up-to-date reference is
[`paper/REPRODUCTION.md`](https://github.com/dowlinglab/bits_for_gaps/blob/main/paper/REPRODUCTION.md)
in the repository, which has the full figure -> script -> data -> reference-diff table,
known discrepancies, and the simplifications taken from the original plotting code.
Nothing here is executed as part of the docs build.
```

## No private-archive access needed

As of Phase 9, this needs **no private-archive access by default**: the data comes
from a curated, committed subset at `paper/data/` (~16 MB -- exactly the plot-input
files the figures read; see `paper/data/README.md`), not the private archive of
record. Clone the repo and run:

```bash
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes    # macOS; only Fig 8/9 touch Julia
python paper/reproduce.py
```

Output goes to `results_remaked/` by default (gitignored -- nothing this produces is
committed). See `paper/REPRODUCTION.md` for regenerating a subset
(`--figures 5 8 9 10`) and for which figures are quantitatively pinned against
`paper/reference/*` (Figs 5, 8, 9, 10, 11) versus visually reproduced only (2, 3, 4, 6,
7, 12).

Point `--archive`/`$BFG_ARCHIVE_DIR` at the full private archive (see
[`paper/DATA.md`](https://github.com/dowlinglab/bits_for_gaps/blob/main/paper/DATA.md))
only if you want figures at iterations beyond the curated subset (e.g. Fig 6/7 at an
iteration other than 1/15) -- that access is author-only, not needed for the default
path above.

## The full stochastic loop, from scratch

`paper/reproduce.py` renders what the paper's 15-iteration adaptive HMC loop already
produced -- it does not re-run that loop (stochastic, hours-long, and not what
reproducing a *figure* requires). `paper/full_reproduction.py` does run it, from a
fresh initial design against the live Clapeyron/Wilson black box, as a separate,
one-time validation exercise (not part of the regression suite, not required to
reproduce a figure) -- see `paper/REPRODUCTION.md`'s "Phase 9" section for the result
and `paper/PHASE9B_INVESTIGATION.md` for a bug found and fixed while validating it.

## The regression tests

Regression tests that check figure/column reproduction against `paper/reference/*`
(`tests/regression/test_paper_figures.py`, `tests/regression/test_mccabe_thiele.py`)
mostly run in the **default** `pytest -q` suite now -- only the tests that recompute
something via live Clapeyron calls (needing Julia) stay behind
`@pytest.mark.vle` (run with `pytest -m vle`). You can also read `paper/figures/*.py`
to see exactly how each figure is built, and run `examples/vle_distillation/
run_case_study.py` ({doc}`vle_example`) for a small, fresh (non-archived)
demonstration of the same underlying pipeline.
