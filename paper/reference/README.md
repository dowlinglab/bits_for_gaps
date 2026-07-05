# Reference scalar targets

Small, version-controlled JSON snapshots of the paper's key scalar results, extracted
from the **archived published run** (`results/less_x_new_manuscript_revisions`,
**iteration 15** — the run whose R-hat/ESS match paper Fig 10 exactly). The bulk
`results/` archive is *not* committed (it goes to Zenodo, see `paper/DATA.md`); only
these scalars live in the repo.

They serve two purposes:
1. **Characterization / regression** — `tests/regression/` reads these files and pins
   them against the values reported in the paper (locks them against corruption).
2. **Reproduction diff** — Phase 7's `paper/reproduce.py` regenerates the figures
   through the clean API and diffs against these reference targets (with the stated
   tolerances).

| File | Paper ref | Contents | Source |
|---|---|---|---|
| `hmc_diagnostics.json` | Fig 10 | HMC R-hat & ESS for the 3 hyperparameters | `rhat_value_15.txt`, `ess_value_15.txt` |
| `fig5_error_metrics.json` | Fig 5 | Train/test RMSE & MAE distributions at iters 1 & 15 | `gp_predict_{train,test}_{1,15}` vs `activity_data_1`/`activity_test_points` |
| `hyperparameter_posterior.json` | Fig 10 | Posterior mean/median/std/quantiles of the kernel hyperparameters | `param_posterior_samples_15` |
| `mccabe_thiele_stages.json` | Fig 9c | Distillation stage liquid/vapor mole fractions (Wilson vs surrogate) | paper Fig 9c / `run_example.py` |

**Regenerating** (needs read access to the old-repo archive; pure NumPy):

```bash
python paper/extract_reference.py
```

produces all but the McCabe-Thiele table (which is transcribed from the paper — its
recompute needs the Julia VLE backend ported in Phase 6).
