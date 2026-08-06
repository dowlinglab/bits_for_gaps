# Paper data & reproduction

The bulk archived results from the published run (`less_x_new_manuscript_revisions`,
iteration 15) are ~2.5 GB and are **not** tracked in this repo.

**Archive of record:** the authors' private repository containing the original research
code and its full history, under
`entropy_driven_hms/results/less_x_new_manuscript_revisions/`. There is intentionally
**no Zenodo deposit** — the private repository is the archive of record.

Two small, curated subsets of that bulk archive **are** committed here:

- `paper/reference/` — small scalar targets for regression (R̂/ESS, error metrics, posterior
  summary, stage table), regenerable via `paper/extract_reference.py`.
- `paper/data/` — the exact plot-input files `paper/figures/*.py` read (~16 MB; see
  `paper/data/README.md` for the file manifest and how it was determined), so
  `python paper/reproduce.py` regenerates every figure from a fresh clone with **no
  private-archive access**. Author access to the full private archive is needed only if
  you want figures at iterations beyond the curated subset (e.g. Fig 6/7 at an iteration
  other than 1/15) — pass `--archive`/`$BFG_ARCHIVE_DIR` to point at it.
