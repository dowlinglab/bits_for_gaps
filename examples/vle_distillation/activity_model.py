"""Julia/Clapeyron activity-coefficient model for the H2O-PrOH VLE case study.

Ported from the paper code's ``proh_water_class.py`` + ``fxns/calculate_activities.jl``
(Wilson activity-coefficient model via Clapeyron.jl). Requires
``pip install "bits_for_gaps[vle]"`` plus a working Julia installation --
``juliacall``/Clapeyron are imported LAZILY (only when a function here is actually
called), so ``import vle_distillation.activity_model`` succeeds without Julia; only
calling :func:`activity_coefficients` / :func:`black_box` requires it.

The Clapeyron.jl dependency is pinned via ``juliapkg.json`` (sibling to this file,
Clapeyron 0.6.26) -- juliapkg auto-discovers any ``<sys.path entry>/<subdir>/
juliapkg.json``, and ``tests/conftest.py`` puts ``examples/`` on ``sys.path``, so this
works without ``examples/`` being pip-installed. See ``README.md`` for a from-scratch
setup (a fresh machine without this pin already resolved).

IMPORTANT (macOS): ``PYTHON_JULIACALL_HANDLE_SIGNALS=yes`` must be set in the
environment BEFORE ``juliacall`` is imported for the first time in a process, or
juliacall SIGBUSes. Set defensively here, before the lazy import, in case the caller
forgot to set it in the shell.
"""
import os

COMPOUNDS = ("propanol", "water")   # Clapeyron component order: index 0 = PrOH, 1 = H2O
PRESSURE_PA = 101325.0              # 1 atm; Wilson's gamma doesn't depend on P anyway
THERMO_MODEL = "Wilson"

_JL_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calculate_activities.jl")

_activity_fn = None   # lazily-populated Julia closure (juliacall wraps it as callable)


def _require_activity_fn():
    """Import juliacall and load the Julia activity-coefficient function, once."""
    global _activity_fn
    if _activity_fn is not None:
        return _activity_fn

    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    try:
        from juliacall import Main as jl
    except ImportError as exc:
        raise ImportError(
            "The VLE example's activity model needs Julia + Clapeyron.jl. Install "
            "with `pip install \"bits_for_gaps[vle]\"` and a working Julia "
            "installation -- see examples/vle_distillation/README.md."
        ) from exc

    _activity_fn = jl.include(_JL_SOURCE)
    return _activity_fn


def activity_coefficients(z_proh, temperature, pressure=PRESSURE_PA,
                          compounds=COMPOUNDS, thermo_model=THERMO_MODEL):
    """Activity coefficients ``(gamma_proh, gamma_water)`` from Clapeyron.jl.

    Parameters
    ----------
    z_proh : float
        Liquid mole fraction of PrOH (component 0); water is ``1 - z_proh``.
    temperature : float
        Temperature [K].
    pressure : float
        Pressure [Pa]. Irrelevant for the Wilson model (a liquid-phase excess-Gibbs
        model, a function of composition and temperature only) but accepted for
        parity with Clapeyron's ``activity_coefficient`` signature and for thermo
        models that do depend on it.

    Returns
    -------
    (gamma_proh, gamma_water) : tuple of float
    """
    from juliacall import Main as jl   # cheap: already imported by _require_activity_fn

    activity_fn = _require_activity_fn()
    out = activity_fn(thermo_model, jl.seval("Vector{String}")(list(compounds)),
                      float(pressure), float(temperature), float(z_proh))
    return float(out[0]), float(out[1])


def black_box(z_proh, temperature):
    """BitsForGaps-compatible black box: called as ``FwdModel(*args, *xStar)``.

    ``xStar`` is in bounds order ``[z_PrOH, T]`` (see ``run_case_study.py``), so this
    accepts ``(z_proh, temperature)`` positionally, matching Phase 5's natural-
    dimension-order calling convention for the injected black box.

    Returns only the PrOH activity coefficient (component 0): the GP surrogate models
    ``gamma_PrOH`` as a function of ``(z_proh, T)`` directly; the water coefficient is
    recovered from it via the Gibbs-Duhem relation (see ``gibbs_duhem.py``), not
    learned by a second GP output.
    """
    gamma_proh, _gamma_water = activity_coefficients(z_proh, temperature)
    return [gamma_proh]
