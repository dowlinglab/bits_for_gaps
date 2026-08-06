"""Per-figure reproduction scripts (paper Figs 2-12), one module per figure.

This is reproduction code, not a library API, so it favors reading directly over
matching every detail of the paper's original plotting code. Each module exposes a
``make(archive_dir, out_dir)`` function that reads the archived published run
(iteration 15) and writes the figure into ``out_dir`` (see ``paper/reproduce.py``).
"""
