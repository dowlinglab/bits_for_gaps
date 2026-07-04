"""Per-figure reproduction scripts (paper Figs 2-12), one module per figure.

Ported from the paper code's ``fxns/mcmc_plotter.py`` (847-line figure library) and
``fxns/plot_res.py`` (CLI dispatch), split one figure per module and simplified --
this is reproduction code, not a library API, so it favors reading directly over
matching every original styling detail. Each module exposes a ``make(archive_dir,
out_dir)`` function that reads the archived published run (iteration 15) and writes
the figure into ``out_dir`` (see ``paper/reproduce.py``).
"""
