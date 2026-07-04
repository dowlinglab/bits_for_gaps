# BITS for GAPS

**B**ayesian **I**nformation-**T**heoretic **S**ampling for hierarchical **GA**ussian
**P**rocess **S**urrogates.

`bits_for_gaps` is a small library for information-theoretic sequential experimental
design with Bayesian hierarchical Gaussian-process (GP) surrogates. Prior physical
knowledge is encoded through priors on the GP hyperparameters; new data is chosen by
maximizing the predictive differential entropy of the resulting hierarchical
posterior, so *hyperparameter uncertainty* -- not just predictive variance -- drives
data acquisition. That distinction is the paper's contribution and the reason the
library exists: a plain GP's predictive variance ignores uncertainty in the
hyperparameters themselves, which matters most exactly when data is scarce.

```{admonition} Reference
K. D. Jones and A. W. Dowling, "BITS for GAPS: Bayesian Information-Theoretic
Sampling for hierarchical GAussian Process Surrogates," *Computers & Chemical
Engineering* **211** (2026) 109650.
[doi:10.1016/j.compchemeng.2026.109650](https://doi.org/10.1016/j.compchemeng.2026.109650)
```

```{admonition} Status
:class: warning
Pre-1.0. The public API (`BitsForGaps`, `AnisotropicSE`, ...) is stable across the
refactor phases described in the project's `HANDOFF.md`, but has not yet had a
tagged release.
```

## Where to start

- **New to the library?** Start with {doc}`installation` then {doc}`quickstart` -- a
  pure-Python example that runs in seconds, no Julia required.
- **Want the math?** {doc}`theory` summarizes the method and links to the paper for
  the full derivations.
- **Want the paper's case study?** {doc}`vle_example` walks through the H2O-PrOH
  vapor-liquid-equilibrium example that motivated the method.
- **Want to reproduce the published figures?** {doc}`reproduce_paper` points at the
  reproduction scripts and what they need.
- **Looking for a specific class or function?** {doc}`api`.

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
theory
vle_example
reproduce_paper
api
```
