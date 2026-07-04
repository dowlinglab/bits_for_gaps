# The VLE/distillation example

The paper's motivating case study: designing experiments for a Wilson
activity-coefficient model of a propanol-water (H2O-PrOH) mixture, then using the
resulting GP surrogate in a McCabe-Thiele distillation-column design. It lives at
`examples/vle_distillation/` in the repository -- **repo-only**, not part of the pip
package (see {doc}`installation`) -- with its own
[README](https://github.com/dowlinglab/bits_for_gaps/blob/main/examples/vle_distillation/README.md)
covering setup (including the Julia/Clapeyron bootstrap) and how to run it.

```{note}
This page is a narrative walkthrough, not a runnable tutorial embedded in the docs
build -- the example needs Julia + Clapeyron.jl, which the docs build deliberately
never installs (see `docs/conf.py`'s autodoc notes). Run it locally by cloning the
repo and following the linked README.
```

## The physics

1. **Activity model** (`activity_model.py`) -- Clapeyron.jl's Wilson model gives the
   PrOH and water activity coefficients at a given liquid composition and
   temperature. This is the black box `bits_for_gaps` designs experiments for.
2. **Gibbs-Duhem correction** (`gibbs_duhem.py`) -- the GP surrogate only ever learns
   $\gamma_{\mathrm{PrOH}}(z, T)$ directly; the water coefficient is *derived* from it
   via the binary Gibbs-Duhem relation, not learned by a second GP output. This halves
   the surrogate-modeling burden for a binary system.
3. **Phase diagram** (`phase_diagram.py`) -- Antoine vapor pressures plus the
   activity coefficients (from either the Wilson ground truth or the GP surrogate +
   Gibbs-Duhem) give bubble-point temperatures and dew-point vapor compositions
   across the whole composition range.
4. **Distillation** (`distillation.py`) -- a McCabe-Thiele stage-by-stage column
   solver, given the phase diagram's liquid-vapor equilibrium curve and a column
   specification (feed composition, reflux ratio, product purities).

## Wiring it to `bits_for_gaps`

`run_case_study.py` ties the pieces together: a Latin-hypercube initial design over
`(z_PrOH, T)`, Clapeyron evaluations at each point, then
{class}`~bits_for_gaps.sampler.BitsForGaps` running the adaptive entropy-driven
design loop with the injected Clapeyron black box, on the paper's exact 2-D kernel
config ({meth}`bits_for_gaps.kernels.AnisotropicSE.paper_2d`). The resulting GP
surrogate feeds the same phase-diagram and distillation code the Wilson ground truth
does, so the two can be compared stage-by-stage.

This example is also where the "black box" calling convention
({doc}`quickstart`) shows up in a case that isn't just a toy: `activity_model.
black_box(z_proh, temperature)` returns `[gamma_proh]` -- one value, matching what a
single-output GP models -- not the pair of activity coefficients Clapeyron actually
computes.

## Reproducing the case study's numbers

The example's own `README.md` has the runnable steps. The paper's *published* figures
(built from a specific archived 15-iteration run, not a fresh one) are a separate
concern -- see {doc}`reproduce_paper`.
