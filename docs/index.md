# BITS for GAPS

**B**ayesian **I**nformation-**T**heoretic **S**ampling for hierarchical **GA**ussian
**P**rocess **S**urrogates.

`bits_for_gaps` is a small library for information-theoretic sequential experimental
design with Bayesian hierarchical Gaussian-process (GP) surrogates. Prior physical
knowledge is encoded through priors on the GP hyperparameters; new data is chosen by
maximizing the predictive differential entropy of the resulting hierarchical
posterior, so *hyperparameter uncertainty* -- not just predictive variance -- drives
data acquisition. Many other adaptive sampling methods in literature use the GP's predictive variance 
and ignore uncertainty in the hyperparameters themselves, which (we argue) matters most exactly when data are scarce.
BITS for GAPS addresses this limitation.

```{figure} graphical_abstract.jpg
:alt: BITS for GAPS framework overview: a hierarchical Gaussian process surrogate (GAPS) places priors on the kernel hyperparameters; Bayesian information-theoretic sampling (BITS) repeatedly selects the next input by maximizing the predictive differential entropy and collects data there; the result is an information-optimal surrogate whose predictions average over the hyperparameter posterior.
:align: center
:width: 100%

The BITS for GAPS framework. A hierarchical GP surrogate (**GAPS**) carries priors on its
kernel hyperparameters; **BITS** repeatedly picks the next input by maximizing the
predictive differential entropy of the hierarchical posterior, collects data there, and
updates the surrogate; predictions then average over the hyperparameter posterior.
*Graphical abstract from the paper (© 2026 The Authors,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).*
```

```{admonition} Reference
K. D. Jones and A. W. Dowling, "BITS for GAPS: Bayesian Information-Theoretic
Sampling for hierarchical GAussian Process Surrogates," *Computers & Chemical
Engineering* **211** (2026) 109650.
[doi:10.1016/j.compchemeng.2026.109650](https://doi.org/10.1016/j.compchemeng.2026.109650)

The paper is also bundled in the repository —
[`paper/bits_for_gaps_paper.pdf`](https://github.com/dowlinglab/bits_for_gaps/blob/main/paper/bits_for_gaps_paper.pdf)
— redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), so you can
read the method alongside the code. Please cite the DOI.
```

```{admonition} Source code
:class: note
`bits_for_gaps` is developed in the open at
[github.com/dowlinglab/bits_for_gaps](https://github.com/dowlinglab/bits_for_gaps) —
source, the worked examples, the paper-reproduction scripts, and the issue tracker. Install
the library with `pip install bits_for_gaps`; the examples and reproduction scripts live in
the repository only (see {doc}`installation`).
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
- **Curious what changed from the original paper code?** {doc}`improvements_over_paper`
  -- bugs found and fixed, robustness hardening, and architecture wins, specifically.
- **Looking for a specific class or function?** {doc}`api`.

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
theory
vle_example
reproduce_paper
improvements_over_paper
api
```
