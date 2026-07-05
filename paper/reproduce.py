"""Regenerate the paper's figures from the published run (iteration 15).

Reads plot-input artifacts (GP posterior draws, HMC traces, phase-diagram data,
activity data) and renders them through the refactored ``bits_for_gaps`` package +
``examples/vle_distillation``, proving the new code reproduces the published
figures. This does NOT re-run the paper's 15-iteration adaptive HMC loop (stochastic,
expensive, and not what "reproduce the figures" requires) -- it loads what that loop
already produced.

As of Phase 9, the default data source is the curated, committed ``paper/data/``
subset (~16 MB -- exactly the files the figures read, copied from the private
archive; see ``paper/data/README.md``), so this runs with **no private-archive
access** out of the box. Pass ``--archive`` (or set ``BFG_ARCHIVE_DIR``) to point at
the full private archive instead -- e.g. to regenerate Fig 3/4 with more than the 6
committed early-iteration panels, or Fig 6/7 at iterations other than 1/15.

Usage::

    export PYTHON_JULIACALL_HANDLE_SIGNALS=yes   # macOS, only needed for Fig 8/9
    python paper/reproduce.py                     # uses the committed paper/data/
    python paper/reproduce.py --archive /path/to/less_x_new_manuscript_revisions
    python paper/reproduce.py --figures 5 8 9 10  # only regenerate a subset

Output goes to ``--out-dir`` (default: ``results_remaked/``, already gitignored) --
nothing this script produces is committed.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)).rsplit(os.sep, 1)[0]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

# Phase 9: default to the curated, committed subset (no private-archive access
# needed); override with --archive/$BFG_ARCHIVE_DIR to use the full private archive.
DEFAULT_ARCHIVE = os.path.join(REPO_ROOT, "paper", "data")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results_remaked")

# Figure number -> (module name, needs_julia). "needs_julia" figures call into
# examples/vle_distillation's Clapeyron-backed physics (Fig 8/9); the rest only read
# archived text/pickle files.
FIGURE_MODULES = {
    "2": ("fig02_lhs_design", False),
    "3": ("fig03_entropy_field", False),
    "4": ("fig04_entropy_evolution", False),
    "5": ("fig05_parity", False),
    "6": ("fig06_gp_posterior_surface", False),
    "7": ("fig07_gp_posterior_isotherms", False),
    "8": ("fig08_phase_diagram", True),
    "9": ("fig09_mccabe_thiele", True),
    "10": ("fig10_traces", False),
    "11": ("fig11_marginals", False),
    "12": ("fig12_joint_marginals", False),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--archive",
        default=os.environ.get("BFG_ARCHIVE_DIR", DEFAULT_ARCHIVE),
        help="Path to the plot-input data (default: $BFG_ARCHIVE_DIR or the "
        f"committed paper/data/ subset, {DEFAULT_ARCHIVE}); point this at the "
        "full private archive for iterations/figures beyond the curated subset",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR, help=f"Output directory (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=sorted(FIGURE_MODULES, key=int),
        choices=sorted(FIGURE_MODULES, key=int),
        help="Which figure numbers to regenerate (default: all)",
    )
    args = parser.parse_args(argv)

    from paper.figures import _archive

    _archive.require_archive(args.archive)
    os.makedirs(args.out_dir, exist_ok=True)

    for fig_num in sorted(args.figures, key=int):
        module_name, needs_julia = FIGURE_MODULES[fig_num]
        if needs_julia:
            os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
        print(f"--- Fig {fig_num} ({module_name}) ---")
        module = __import__(f"paper.figures.{module_name}", fromlist=[module_name])
        result = module.make(args.archive, args.out_dir)
        path = result.get("path", result) if isinstance(result, dict) else result
        print(f"    wrote {path}")

    print(f"\nDone. Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
