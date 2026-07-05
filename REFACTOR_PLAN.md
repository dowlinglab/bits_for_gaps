# Refactoring Plan — `bits_for_gaps`

**Goal:** Extract the *already-conceptually-general* BITS for GAPS algorithm from the paper's VLE/distillation example code and ship it as a pip-installable, documented, tested Python library. Reproduce the paper figures from a clean public API.

**Source:** `entropy_driven_hybrid_models_code/entropy_driven_hms/` (branch `main`), with fixes and env from branch `codex-refactor`.
**Paper:** Jones & Dowling, "BITS for GAPS," *Computers & Chemical Engineering* 211 (2026) 109650. CC BY 4.0.

---

## Progress log

### Session 1 (2026-07-03, Opus) — audit + env + reproduce-from-archive ✅
- **Audit (Step 1) done** — algorithm/example split mapped (§2), Codex branch mined, git history analyzed (fresh repo confirmed).
- **Environment (Step 2) done & verified** — conda env `bits_for_gaps` created from `environment.yml`; ML stack confirmed (gpflow 2.9.2 / TF 2.16.2 / TFP 0.24.0 / Py 3.9.23). Julia 1.12.6 + Clapeyron auto-installed via juliapkg. **Fix: must set `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`** or juliacall SIGBUSes on macOS (exit 138). Wilson γ(PrOH/water, 350 K, x=0.5) = [1.469, 1.746]. Full recipe in the `environment-setup-recipe` memory.
- **Reproduce-from-archive (Step 3) substantial** — all done non-destructively into `results_remaked/`:
  - **Fig 10** (HMC traces): regenerated `trace_all_15.png` via `plot_res.py -m all_traces`; R̂/ESS already matched paper numerically.
  - **Fig 9** (McCabe-Thiele): both panels regenerated; column design matches paper exactly (xW=0.01, F=100, xF=0.10, R=1.0, xD=0.43, 4 stages).
  - **GP-predict path**: archived `gp_model_15.pkl` unpickles (GPR + `AnisotropicSE`, 24 pts) and `predict_f` works.
- **Bug fixed** — `equilibrium.water_proh_eqm_julia` pointed at the missing `results/less_x/phase_diagram_data_saftgammamie`; repointed to the archived `gt_Wilson_data` (see NOTE in `equilibrium.py`).
- **Artifacts written** — `environment.yml` (repo root), `results_remaked/` (regenerated figs), scratchpad repro scripts.
- **Not yet done** — Fig 5 parity/error full regen (path validated via GP smoke test); Figs 2/3/4/6/7/11/12 regen; fresh-repo bootstrap (Phase 0); test harness (Phase 2).

---

## 1. What the algorithm is (the thing we generalize)

A sequential experimental-design loop over a **hierarchical Gaussian-process surrogate**:

1. **Initialize** — space-filling design (Latin hypercube) over the input space; GP prior on the black-box output; priors on GP hyperparameters θ.
2. **Calibrate** — evaluate the black-box `f(x)` at design points; run **HMC** (TFP) to sample the hyperparameter posterior `p(θ|y)`; propagate a subset of samples through the GP → a **Gaussian-mixture predictive posterior**.
3. **Evaluate** — score the surrogate; pick the next `x` by **maximizing the predictive differential entropy** (Taylor approximation, or the closed-form lower bound). Repeat until a stopping criterion.

The entropy math (GMM density, 1st/2nd-order Taylor entropy, closed-form lower bound) is **pure numpy/scipy and already generic**. The rest of the loop is generic *in intent* but **hardcoded to 2 inputs and 3 hyperparameters in the code**.

---

## 2. Current code: algorithm vs. example (audit result)

| Layer | Files (in `entropy_driven_hms/`) | Disposition |
|---|---|---|
| **CORE — sequential design engine** | `driver_new.py` (class `adaptiveEntropy`) | Generalize & decompose → package |
| **CORE — entropy math** | `fxns/max_ent_design.py` | Move ~verbatim; already generic (numpy) |
| **CORE — GP kernel / mean** | `fxns/my_kermel_fxn.py` (`AnisotropicSE`), `fxns/my_mean_fxn.py` (`FixedInverseMean`) | Generalize to N-D → package |
| **CORE — small utils** | `fxns/util.py`, `fxns/plot_settings.py` | Move; tidy |
| **EXAMPLE — thermo / data gen** | `proh_water_class.py`, `fxns/calculate_activities.jl`, `equilibrium.py` | → `examples/vle_distillation/` |
| **EXAMPLE — physics** | `gibbs_duhem.py`, `new_phase_diagram.py`, `distillation_model.py`, `solve_distillation.py` | → `examples/vle_distillation/` |
| **EXAMPLE — validation plots** | `train_test_split_proh.py` (Fig 5) | → `paper/` figure scripts |
| **PLOTTING — paper figure library** | `fxns/mcmc_plotter.py` (847 lines), `fxns/plot_res.py` (CLI) | Split into `paper/figures/`; extract generic bits |
| **VALIDATION — entropy approx** | `huber_et_al.py` (5D-mixture `h_vs_c` test) | → a **unit test** for `entropy.py` |
| **LEGACY — drop** | `driver.py` (superseded), `phase_diagram.py` (superseded, iter-10/`less_x`), `old/**`, `osu_presentation/**`, `paper_writing/**`, `power_point/**`, `fxns/*.jl` plotters (still EtOH-labeled) | Do **not** migrate |
| **DEAD code** | `max_ent_design.entropy_lower_bound` (never called), commented `second_order_entropy` block, `gibbs_duhem.load_ground_truth` (undefined `iters`) | Delete or fix during migration |

**Critical coupling to sever:**
- `driver_new.py`, `proh_water_class.py` call `jl.include("fxns/calculate_activities.jl")` **at module import** → the core must never import Julia. The black-box `f(x)` is *injected* by the caller (this pattern already exists via `fwd_model`/`fwd_model_args`).
- `driver_new.py` starts with `from proh_water_class import PrOHwater` — core importing the example. Remove.
- Hardcoded 2-D everywhere: `gp_predict_2D`, `gen_entropy_surface_data_2D`, `optimize_2D`; `run_mcmc` indexes `trainable_parameters[2],[0],[1]` (exactly 3 hyperparameters); mixture/entropy code assigns kernel params **by name** (`std_dev`, `lengthscale_1`, `lengthscale_2`).
- `run_model` has `i += 50` — a resume offset from the manuscript-revision run. Remove; replace with real checkpoint/resume.
- State lives on disk (ad-hoc `np.savetxt`/`pickle` under `results/{exp}/`), not in memory. Make state in-memory objects with *optional* checkpointing.

---

## 3. Reproducibility status (verified this session)

- ✅ Archived `results/less_x_new_manuscript_revisions/` **iteration 15 = the published run**. Its `rhat_value_15.txt` / `ess_value_15.txt` match **Figure 10 exactly**: R̂ = 1.005 / 1.007 / 1.009, ESS = 1468.3 / 2428.1 / 653.1.
- ✅ `phase_diagram_15` (feeds Figs 8–9) and `gt_Wilson_data` are present on disk.
- ✅ Final results are archived — we can **regenerate figures from archived data without re-running MCMC**.
- ⚠️ **Gap:** `equilibrium.water_proh_eqm_julia` reads `results/less_x/phase_diagram_data_saftgammamie` — **absent from disk and git history**. The Fig 9 ground-truth ("Wilson") panel as-coded will fail. Fix: repoint to `gt_Wilson_data`, or regenerate via the Julia Wilson path. (The `_julia`/`saftgammamie` naming is a leftover from the earlier SAFT-γ-Mie system.)
- ⚠️ **Clapeyron.jl is unpinned** (no `juliapkg.json`) → activity-coefficient ground truth could drift. Pin it in the new repo.
- ⚠️ **Env is fragile & old:** Python 3.9 / TF 2.16.2 / GPflow 2.9.2 / TFP 0.24.0, macOS-arm64. **`juliacall` MUST import before TensorFlow/GPflow** or you get a bus-error segfault (fixed on `codex-refactor`). `codex-refactor` also has a working `environment.yml` — reuse it.
- ✅ **Fresh repo is correct:** 104 commits, ~99.8% of all file paths ever added are under `results/` (2.45 GB pack), no tags, no packaging config ever. Reference commits: `e17c818` (journal submission), `db3a2de` (pre-revision).

---

## 4. Proposed package architecture (`src` layout, fresh repo)

```
bits_for_gaps/
├── pyproject.toml              # hatchling; core deps only (no Julia)
├── README.md  LICENSE  CHANGELOG.md
├── environment.yml             # dev/repro env (from codex-refactor + fixes)
├── juliapkg.json               # pin Clapeyron.jl (examples extra only)
├── .github/workflows/ci.yml    # unit+integration on pure-Python core
├── docs/                       # Sphinx + MyST → ReadTheDocs
├── src/bits_for_gaps/
│   ├── __init__.py             # public API surface
│   ├── sampler.py              # BitsForGaps: the sequential-design loop (was adaptiveEntropy)
│   ├── gp.py                   # build GPR, hierarchical priors, run HMC, posterior samples
│   ├── mixture.py              # GMM predictive posterior from θ-samples (generic param assignment)
│   ├── entropy.py              # max_ent_design.py: GMM density, Taylor 1st/2nd, lower bound
│   ├── acquisition.py          # entropy objective + multistart optimize (N-D)
│   ├── kernels.py              # AnisotropicSE (N-D), extensible
│   ├── means.py                # mean functions
│   ├── design.py               # LHS / full-factorial space-filling (N-D)
│   ├── transforms.py           # per-dim fwd/bkwd input & output transforms (class)
│   ├── diagnostics.py          # R-hat, ESS helpers
│   ├── state.py                # in-memory state + optional checkpoint/resume
│   └── plotting.py             # generic, problem-agnostic plot helpers
├── examples/
│   ├── synthetic/              # pure-Python demos (NO Julia) — run in CI & docs
│   │   ├── branin_1d.py        # 1-D toy: exercises N-D generality + fast tests
│   │   └── synthetic_3d.py     # 3-D toy: exercises >2 inputs
│   └── vle_distillation/       # the paper case study (needs Julia/Clapeyron)
│       ├── activity_model.py   # was proh_water_class.py (generalized thermo wrapper)
│       ├── calculate_activities.jl
│       ├── gibbs_duhem.py  phase_diagram.py  distillation.py  equilibrium.py
│       └── run_case_study.py
├── paper/                      # reproduce published figures
│   ├── reproduce.py  Makefile
│   ├── figures/                # mcmc_plotter split into per-figure scripts (5,8,9,10, etc.)
│   ├── reference/                 # small scalar targets (R̂, ESS, MAP, stage table) + tolerances
│   └── DATA.md                 # pointer to the archived 2.5 GB results in the private old repo
└── tests/
    ├── unit/                   # entropy math, transforms, kernels, GD integral, Antoine, design
    ├── integration/            # tiny end-to-end seeded run (few samples, 1 iter)
    └── regression/             # reference-file checks vs paper metrics (Julia-gated, slow job)
```

**Key API idea (target):**
```python
from bits_for_gaps import BitsForGaps, AnisotropicSE, InputTransform

bfg = BitsForGaps(
    black_box=my_fx,                 # callable: x (n,d) -> y (n,)
    bounds=[(lo, hi), ...],          # any dimension d
    kernel=AnisotropicSE(ndim=d),    # priors carried by the kernel Parameters
    input_transforms=InputTransform([...]),  # per-dim fwd/bkwd, optional
    output_transform=...,            # optional
    likelihood_variance=0.1,
)
bfg.mcmc = MCMCConfig(n_samples=5000, n_chains=4, step_size=0.05, ...)
result = bfg.run(n_iterations=30, seed=10)   # returns history: designs, θ-traces, entropy field, diagnostics
```
Core has **zero Julia dependency**; `pip install bits_for_gaps` pulls GPflow/TF/numpy/scipy only. The VLE example's Julia backend is an optional extra: `pip install "bits_for_gaps[vle]"` + documented Julia/Clapeyron setup.

---

## 5. Execution phases (maps to your 8 steps, reordered so tests precede the refactor)

### Phase 0 — Decisions + fresh-repo bootstrap  *(this Opus session)*
- Lock the open decisions in §7.
- Create the new repo skeleton (src layout, `pyproject.toml`, `.gitignore`, `environment.yml`, `LICENSE`, empty `tests/`). No history from the old repo.
- Carry over from `codex-refactor`: `environment.yml`, `.gitignore`, the juliacall-before-TF import fix, and `code_plan.md`/`reactor_log.md` as reference.

### Phase 1 — Environment + reproduce-from-archive  *(Opus; this is "the tricky part")*
- Build the conda env; verify `gpflow/tf/tfp` and `juliacall` import (correct order). Install & **pin Clapeyron.jl** (`juliapkg.json`).
- Regenerate Figs **5, 8, 9, 10 from archived `less_x_new_manuscript_revisions` data** (no MCMC re-run) and diff against the paper PDF. Resolve the missing `phase_diagram_data_saftgammamie` gap (repoint to `gt_Wilson_data`).
- Deliverable: a working env + a short "repro report" confirming archived == published.

### Phase 2 — Characterization / regression harness  *(BEFORE any refactor — your stated priority)*
- **Reference scalars** from archived iter-15: R̂, ESS, MAP hyperparameters, Fig 9 stage-composition table, Fig 5 RMSE/MAE. Store in `paper/reference/` with tolerances.
- **Unit tests on current behavior:** port `huber_et_al.py`'s 5D-mixture entropy curve to a pytest with pinned expected values; add tests for Gibbs-Duhem integral, Antoine `pvap`, transforms, LHS design determinism (seeded).
- **Seeded integration test:** a minimal end-to-end run (e.g. 100 MCMC samples, 1 iteration) whose outputs are stable.
- Everything green against the *unrefactored* code = the safety net.

### Phase 3 — Package scaffold + verbatim move  *(Sonnet)*
- Move CORE files into `src/bits_for_gaps/` with minimal edits: fix imports, remove module-load Julia, remove `from proh_water_class import`. Keep 2-D behavior identical. Get `import bits_for_gaps` working and the Phase-2 tests passing through the package.
- Stand up CI (unit + integration on pure-Python core; Julia/regression as a separate gated job).

### Phase 4 — Incremental modular refactor of the core  *(Sonnet)*
- Decompose `adaptiveEntropy` into `sampler` / `gp` / `mixture` / `acquisition` / `entropy` / `transforms` / `design` / `diagnostics` / `state`.
- Remove `i += 50`; replace disk-as-state with in-memory state + optional checkpointing.
- Keep regression tests green at every step.

### Phase 5 — Generalize to N-D  *(Sonnet, distinct milestone)*
- Remove 2-D/3-hyperparameter hardcoding: generic kernel-parameter introspection in HMC + mixture; N-D acquisition grid/optimizer; N-D design.
- New tests: 1-D and 3-D **synthetic** problems (no Julia) proving generality. This is where the "already general" claim becomes true in code.

### Phase 6 — Port the VLE example onto the clean API  *(Sonnet)*
- Rewrite `examples/vle_distillation/` to consume the public API (inject the Julia activity `f(x)`), fix hardcoded paths/`iters`, fix the `equilibrium.py` cross-experiment reference.

### Phase 7 — Reproduce ALL paper figures via the new API  *(Sonnet + Opus check)*
- `paper/reproduce.py` regenerates Figs 5, 8, 9, 10 (and 2, 3, 4, 6, 7) through the package + examples. Diff against reference scalars and the PDF.

### Phase 8 — Documentation  *(Sonnet)*
- Sphinx + MyST: install/env (incl. the juliacall gotcha), quickstart on the 1-D synthetic, API autodoc, a "reproduce the paper" guide, theory notes linking to the paper. Wire ReadTheDocs (`.readthedocs.yaml`); build the pure-Python parts without Julia.

### Phase 9 — Reproduce ALL paper results, including the stochastic loop  ✅ DONE (+ Phase 9b)
- Ran the full adaptive loop from scratch (`paper/full_reproduction.py`, ~26 min): seeded parts (HMC posterior, R̂/ESS, hyperparam posterior, entropy decay) reproduce the paper to **7–8 sig figs**; only the non-seedable `predict_f_samples` path (test-RMSE, surrogate curve) drifts.
- **Phase 9b correction:** Phase 9 initially reported the fully-adaptive surrogate's McCabe-Thiele column as non-converging and attributed it to entropy-driven design — that was a **bug** (shared-mutable-state in `full_reproduction.py`: the test-RMSE loop mutated `GPmodel.kernel` in place before the phase diagram reused it), **not** a scientific finding. Fixed (example layer only); the adaptive surrogate's column now converges and tracks Wilson within 0.03. See `paper/PHASE9B_INVESTIGATION.md`; retraction in `paper/REPRODUCTION.md`.

### Phase 9c — Robustness hardening (make it a reliable package)  *(Sonnet + Opus check)*
- Make BITS for GAPS as robust as possible **without changing numerical behavior** — the pre-Phase-4 baseline (`synthetic_baseline.json`, atol 1e-10) + all reference regressions are the safety net and stay green. First sanctioned core change since Phase 4.
- Targets: (1) fix the `mixture.sample_gp_posterior_mixture` in-place kernel-mutation footgun (save/restore state) — the Phase-9b bug; (2) public-API input validation with clear errors (bounds vs kernel ndim, lo<hi, positive config, X/y shapes, black-box output); (3) guard fragile spots (distillation `fsolve` convergence check/message, entropy `assert`→explicit raise, GP/Cholesky conditioning); (4) optional TF seed so the one documented non-reproducibility (`predict_f_samples`) can be made deterministic on request.
- Rerun `paper/full_reproduction.py` to confirm the paper still reproduces (column converges, HMC to 7–8 sig figs, stage table tracks Wilson).
- Document all improvements/fixes over the original paper code in `docs/improvements_over_paper.md`.

### Phase 9d — Polish pass (hygiene + faithfulness), whole-codebase  *(Sonnet + Opus check)*
- Behavior-preserving (baseline 1e-10 + reference green at every commit). Hygiene: ruff lint+format whole repo (preserve load-bearing import order — lazy `__init__`, `HANDLE_SIGNALS`-before-juliacall), type hints across the codebase via `from __future__ import annotations` (keep `import bits_for_gaps` Julia-free + TF-lazy), `py.typed`, broaden CI to the full default suite + README badge, coverage pass.
- Faithfulness/features: wire `entropy_lower_bound` (the paper's SI closed-form bound) as a SELECTABLE acquisition objective (Taylor stays default); MC-validation test for the entropy approximation; a pure-Python synthetic example/tutorial (no Julia); GP Cholesky-conditioning guard; `CHANGELOG.md`.

### Phase 9e — Docs / tests / comments quality pass  *(Sonnet + Opus check)*
- Behavior-preserving (baseline 1e-10 + reference regressions green each commit). Three passes:
  1. **RTD docs**: sensible organization, working cross-links, math faithful to the paper (key equations in the paper's notation — Eq 1/2 entropy+acquisition, Eq 6 SE kernel, Eq 7 GMM predictive, Eq 9 entropy, the Huber 2nd-order Taylor approx, the Theorem lower bound + SI-2 cross-overlap, Eq 10 extended Raoult, Eq 11 Gibbs–Duhem, SI-4 distillation), clearly explain the 3-phase loop. **Rename "golden" → "reference" everywhere** (git mv `paper/golden/`→`paper/reference/`, `golden` fixture→`reference`, all usages + docs).
  2. **Tests**: unit tests on most functions; assert errors are raised; find + upgrade low-value/superficial tests.
  3. **Comments**: NumPy-style docstrings on all functions; inline comments on non-trivial code citing paper/SI equation numbers (map in the Phase-9e prompt / HANDOFF).

### Phase 10 — Publish  *(Opus + user)*
- Artifacts are already finalized (Phase 9 committed `paper/data/`; §7 decision 4). Release engineering: polish `pyproject.toml` → `0.1.0`, `CHANGELOG.md` + `RELEASE.md`, `python -m build` + `twine check`, clean-env install audit, GitHub Actions **trusted-publishing** workflow.
- User actions: create PyPI/TestPyPI projects + configure trusted publishing (OIDC); TestPyPI dry-run; `git tag v0.1.0`; activate RTD (steps in `HANDOFF.md`).

---

## 6. Cross-cutting concerns

- **Dependency stack:** freeze the exact old stack (Py3.9/TF2.16/GPflow2.9/TFP0.24) as the reproduction baseline first. Modernization (newer Python/TF) is a *later, separate* effort — GPflow ties us to TensorFlow, so it's non-trivial; note GPJax/alternatives as future work, don't block on it.
- **Determinism:** every stochastic path takes an explicit seed; document that HMC exact bitwise reproducibility can vary across TF/BLAS builds → regression uses tolerances, not equality.
- **Data:** `paper/data/` holds the committed plot-input subset (16 MB, §7 decision 4) — never the full 564 MB run (that stays in the private old repo; no Zenodo). Reference scalars stay in `paper/reference/`.
- **License:** add one (BSD-3-Clause or MIT recommended for scientific Python; paper text is CC BY 4.0).
- **Session handoff:** keep a `LOG.md` + this `REFACTOR_PLAN.md` in the new repo so Opus↔Sonnet sessions hand off cleanly (the `codex-refactor` `reactor_log.md` is a good template).

---

## 7. Decisions (locked — 2026-07-03)

1. **Package name = `bits_for_gaps`.** Import `import bits_for_gaps`.
2. **One repo, no separate paper repo.** The old `entropy_driven_hybrid_models_code` repo stays **private** (with its bloated history) as the archive of record; the paper-reproduction code migrates into this `bits_for_gaps` repo.
3. **Examples + paper-reproduction are repo-only — NOT shipped in the pip wheel** (keeps the package lightweight). Only `src/bits_for_gaps/` ships. Layout: `examples/vle_distillation/` = the reusable H2O–PrOH case study; `paper/` = thin reproduction scripts (import from `examples/`) + `reference/` + `DATA.md`. Both are importable in dev/CI via a `tests/conftest.py` `sys.path` insert, not via install.
4. **`paper/data/` = a committed ~16 MB verbatim subset of the archive; keep it AS-IS** (FINAL, 2026-07-04). The figures read only a subset of the published run; `paper/data/` holds exactly those files (verified byte-identical copies of the private archive — provenance in `paper/data/README.md`), so `paper/reproduce.py` runs archive-free from a fresh clone. User set a ≤20 MB ceiling and chose **simple over aggressively lean** (minor package, maybe 1–2 follow-on papers): do NOT trim — keep the 12 MB `gp_predict_{1,15}` grids (no pickle-recompute machinery) and keep `traces_chain_*` (Fig 10 stays reproducible). The FULL 564 MB run (365 MB `gp_predict`×60 + 88 MB PNGs + 86 MB traces) stays in the **private old repo**; **no Zenodo**. (Correction: the earlier "2.5 GB" was the old repo's entire git history; the single published-run directory is 564 MB.)
5. **2-D faithful first, then N-D.** Clean 2-D refactor + lock regression (Phases 3–4), then generalize to N-D as Phase 5 with new synthetic tests.
6. **Freeze the current dependency stack** (Py3.9 / TF 2.16.2 / GPflow 2.9.2 / TFP 0.24.0) as the reproduction baseline; modernization is a separate later effort.
7. **Core is pure Python** (GPflow/TF/numpy/scipy); Julia/Clapeyron only for the `[vle]` example extra, imported lazily.
