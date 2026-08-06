"""Sphinx configuration for bits_for_gaps.

Docs are repo-only (not shipped in the pip wheel, mirroring examples/ and paper/'s
policy) and built with Sphinx + MyST + furo, all pulled in via the ``[docs]`` extra
(``pip install -e ".[docs]"``).
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from bits_for_gaps import __version__  # noqa: E402

project = "bits_for_gaps"
copyright = "2026, Alexander W. Dowling and Kyla D. Jones"
author = "Alexander W. Dowling, Kyla D. Jones"
release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "BITS for GAPS"
html_static_path = []

# --- autodoc ------------------------------------------------------------------
#
# RTD/CI robustness: autodoc imports bits_for_gaps, which imports
# gpflow/tensorflow/tensorflow_probability for the TF-backed modules
# (kernels, means, sampler, gp, mixture, acquisition, diagnostics). We install the
# REAL frozen stack rather than mocking it -- pip always installs a package's base
# `dependencies` alongside any extra, so `pip install -e ".[docs]"` (what
# .readthedocs.yaml runs) already pulls in the pinned TF 2.16.2 / GPflow 2.9.2 /
# TFP 0.24.0 stack. This gives autodoc a real, importable `AnisotropicSE` (a
# `gpflow.kernels.Kernel` subclass) to introspect, so its signature and inherited
# members render correctly -- verified locally with `sphinx-build -W`.
#
# Fallback, if a future RTD build times out or runs out of memory installing TF:
# uncomment the line below to mock the heavy deps out instead (autodoc will still
# render each function/class's own signature and docstring from source, just without
# resolving base classes or default values that depend on importing gpflow/tf).
#
# autodoc_mock_imports = ["tensorflow", "tensorflow_probability", "gpflow", "tf_keras"]
#
# `juliacall`/`juliapkg` are NEVER imported by the docs build: the VLE walkthrough
# page (vle_example.md) is narrative only, with no autodoc directives over
# examples/vle_distillation -- so no mock is needed for them.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autoclass_content = "both"
add_module_names = False
