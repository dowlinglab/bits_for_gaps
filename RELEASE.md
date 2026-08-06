# Release process — `bits_for_gaps`

The maintainer's checklist for cutting a PyPI release. Trusted publishing and
ReadTheDocs are already configured (see "One-time setup" below) -- most releases only
need the "Before every release" checks plus "Cutting a release"'s few steps.

## Already set up

- **Trusted publishing** (PyPI and TestPyPI): `.github/workflows/publish.yml` publishes
  via OIDC -- no API tokens stored anywhere in this repo. A `v*` tag push reaches PyPI;
  a manual `workflow_dispatch` reaches TestPyPI only, for a pre-release dry run.
- **ReadTheDocs**: imported and building from `.readthedocs.yaml` at
  https://bits-for-gaps.readthedocs.io -- rebuilds automatically on pushes to `main` and
  on new tags; no manual step needed per release.
- **Version**: single-sourced from `src/bits_for_gaps/__init__.py`'s `__version__`
  (`[tool.hatch.version]` in `pyproject.toml` reads it) -- that's the one place to edit
  to bump the version.

## Before every release

**1. Build artifact contents.** `python -m build` should produce:

- A **wheel** containing exactly `bits_for_gaps/*.py` + `py.typed` + dist-info -- no
  `examples/`, `paper/`, `docs/`, or `tests/` (`unzip -l dist/*.whl` or
  `zipfile.namelist()` to check).
- An **sdist** restricted via `[tool.hatch.build.targets.sdist]`'s `only-include` (see
  `pyproject.toml`) to the package source plus `LICENSE`/`README.md`/`CHANGELOG.md`.
  Without `only-include`, hatchling's default sdist file set is the *whole repository*
  (confirmed once by building without it: 193 entries, 6.9 MB, including all of
  `paper/data/`) -- `only-include` (not `include`, which *adds* to the default set
  rather than replacing it) is what actually restricts it.
- `twine check dist/*` passes for both artifacts.
- The wheel's `METADATA` `Version:` field matches `bits_for_gaps.__version__`.

**2. Clean-env install audit.** In a fresh, throwaway environment (no dev tooling, no
editable install):

```bash
conda create -n bfg-release-check python=3.9 pip && conda activate bfg-release-check
pip install dist/bits_for_gaps-*.whl
python -c "import bits_for_gaps, sys; assert 'juliacall' not in sys.modules and 'tensorflow' not in sys.modules; print(bits_for_gaps.__version__)"
# REQUIRED: also force-load a TensorFlow-backed module. The line above only touches
# the eagerly-imported pure modules, so the lazy __getattr__ never loads GPflow --
# exactly how 0.1.0 shipped with every TF-backed module broken on setuptools >= 81
# (see CHANGELOG 0.1.1). Do this with CURRENT setuptools, so a missing dependency
# bound shows up here rather than in a user's traceback.
python -c "import bits_for_gaps as b; k = b.AnisotropicSE(); print('TF-backed OK, ndim =', k.ndim)"
python -c "import setuptools; print('setuptools resolved to', setuptools.__version__)"
```

Also smoke-test the public API without Julia: `AnisotropicSE()` construction,
`latin_hypercube_design(...)`, `entropy.second_order_entropy(...)`, and
`BitsForGaps(...)` construction using the exact kwargs from `docs/quickstart.md`'s
snippet -- confirm they still match the installed API. Tear the environment down after.

## Cutting a release

1. Bump `__version__` in `src/bits_for_gaps/__init__.py`.
2. Move `CHANGELOG.md`'s `## [Unreleased]` content into a new `## [x.y.z] - YYYY-MM-DD`
   section (today's date); leave a fresh empty `[Unreleased]` above it.
3. Commit both, on `main`.
4. Run the "Before every release" checks above against a local `python -m build`.
5. Optional: dry-run on TestPyPI first (Actions tab -> "Publish" workflow -> "Run
   workflow" -> TestPyPI target), then in a fresh venv:
   ```bash
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bits_for_gaps
   ```
   (`--extra-index-url` is needed because TestPyPI doesn't mirror PyPI's dependencies.)
   Run the same smoke test as step 2 above.
6. Tag and push:
   ```bash
   git tag vx.y.z
   git push --tags
   ```
   This triggers `publish.yml`'s tag-push job, which builds fresh and publishes to PyPI
   via trusted publishing.
7. Verify: `https://pypi.org/project/bits_for_gaps/` shows the new version, then in
   another fresh env, `pip install bits_for_gaps` and re-run the smoke test. RTD picks
   up the new tag automatically.

## Optional hardening

`.github/workflows/publish.yml` references `pypa/gh-action-pypi-publish` via PyPA's own
recommended floating tag (`@release/v1`). For a fully immutable pin, look up the latest
release at https://github.com/pypa/gh-action-pypi-publish/releases and replace
`@release/v1` with `@<commit-sha> # v1.x.y` in both jobs.
