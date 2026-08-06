# Paper data

**Everything needed to regenerate the paper's figures is committed in this repository.**
`python paper/reproduce.py` works from a fresh clone with no additional downloads and no
special access. If you only want the figures, you can stop reading here and go to
[`REPRODUCTION.md`](REPRODUCTION.md).

## What is committed

- **`paper/data/`** (~16 MB) — the run artifacts the figure scripts read: HMC traces and
  posterior samples, the entropy fields, the surrogate phase-diagram draws, the Wilson
  ground-truth curve, and the training/test activity-coefficient data. These are the
  published run's own outputs, copied verbatim. See
  [`data/README.md`](data/README.md) for the file-by-file manifest.
- **`paper/reference/`** — small scalar targets used by the regression tests (R̂ and ESS,
  train/test error metrics, the hyperparameter-posterior summary, the McCabe–Thiele stage
  table), regenerable with `paper/extract_reference.py`.

## What is not committed, and why that doesn't matter

The published run also produced roughly half a gigabyte of intermediate artifacts — full
posterior-prediction grids at all 60 iterations, per-iteration trace files, and rendered
figure images. Committing all of it would bloat the repository to no purpose: the figure
scripts read only the subset above, so that subset is what ships.

The scripts accept `--archive` / `$BFG_ARCHIVE_DIR` to point at a different directory of run
artifacts, laid out the same way as `paper/data/`. That is useful if you re-run the loop
yourself (`paper/full_reproduction.py` writes such a directory) and want to plot *your* run
instead of the published one. It is not a prerequisite for anything in
[`REPRODUCTION.md`](REPRODUCTION.md).

## The bundled paper

`paper/bits_for_gaps_paper.pdf` is the published article, included so the method and the code
can be read side by side. It is redistributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see
[`PAPER_LICENSE.md`](PAPER_LICENSE.md) for the citation and license statement.
