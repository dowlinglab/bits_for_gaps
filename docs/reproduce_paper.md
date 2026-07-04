# Reproducing the paper's figures

All 11 of the paper's figures (2 through 12) regenerate from the archived published
run through `paper/reproduce.py` + `paper/figures/` -- **repo-only** scripts (see
{doc}`installation`), not part of the pip package.

```{note}
This page is a pointer, not a copy -- the authoritative, up-to-date reference is
[`paper/REPRODUCTION.md`](https://github.com/dowlinglab/bits_for_gaps/blob/main/paper/REPRODUCTION.md)
in the repository, which has the full figure -> script -> archived-inputs -> golden-diff
table plus known discrepancies and the simplifications taken from the original
plotting code. Nothing here is executed as part of the docs build.
```

## You need the archive

The figures are built from the **archived published run** (iteration 15 of
`less_x_new_manuscript_revisions`), not by re-running the paper's stochastic
15-iteration adaptive HMC loop -- see `paper/reproduce.py`'s docstring and
`paper/REPRODUCTION.md` for why (re-running would give a qualitatively similar but
different stochastic realization, not a reproduction of the specific published
figures).

That archive lives in the **private old repository** (the archive of record --
see [`paper/DATA.md`](https://github.com/dowlinglab/bits_for_gaps/blob/main/paper/DATA.md)),
not in this repo and not on Zenodo. Reproducing the figures requires author access to
it. If you have that access:

```bash
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes    # macOS; only Fig 8/9 touch Julia
python paper/reproduce.py --archive /path/to/less_x_new_manuscript_revisions
```

Output goes to `results_remaked/` by default (gitignored -- nothing this produces is
committed). See `paper/REPRODUCTION.md` for regenerating a subset
(`--figures 5 8 9 10`) and for which figures are quantitatively pinned against
`paper/golden/*` (Figs 5, 8, 9, 10, 11) versus visually reproduced only (2, 3, 4, 6,
7, 12).

## Without the archive

The gated regression tests that check figure reproduction
(`tests/regression/test_paper_figures.py`, `tests/regression/test_mccabe_thiele.py`)
are marked `@pytest.mark.vle` and skip cleanly if the archive isn't present -- they
don't run as part of the default `pytest -q`. You can still read `paper/figures/*.py`
to see exactly how each figure is built, and run `examples/vle_distillation/
run_case_study.py` ({doc}`vle_example`) for a fresh (non-archived) demonstration of
the same underlying pipeline.
