# Paper data & reproduction

The bulk archived results from the published run (`less_x_new_manuscript_revisions`,
iteration 15) are ~2.5 GB and are **not** tracked in this repo.

**Archive of record:** the private repository `dowlinglab/entropy_driven_hybrid_models_code`
(the original paper code, kept private with its full history) under
`entropy_driven_hms/results/less_x_new_manuscript_revisions/`. There is intentionally **no
Zenodo deposit** — the private old repo is the archive of record (REFACTOR_PLAN.md §7
decision 4). Phase 7 figure reproduction assumes author access to that repo.

Only the small **golden scalars** needed for regression are committed here, under
`paper/golden/` (regenerable via `paper/extract_golden.py`, which reads the old-repo archive).
