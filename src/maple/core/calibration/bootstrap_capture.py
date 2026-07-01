"""Recover the across-patient population sample a CalibrationTarget's
``empirical_data.distribution_code`` declares via its ``samples`` return key.

``derive_distribution(inputs, ureg)`` returns ``median_obs`` / ``ci95_lower`` /
``ci95_upper`` and MAY additionally return ``samples`` — the 1-D population draw
it builds internally (a Pint Quantity array, one value per patient-equivalent).
That array is the *real* observation distribution, with its true skew/tails/
bounds, and its empirical spread is the population-variability signal for
hierarchical inference. We read it straight off the declared output.

Used by the SBI pipeline to feed empirical observation noise into
``add_observation_noise`` (instead of a parametric lognormal/Gaussian refit of
the CI endpoints, which only matches when the bootstrap is log-symmetric), and
by the hierarchical pipeline to read the observed population spread. Targets
whose code does not declare ``samples`` (older summary-only codes, or analytic/
closed-form derivations) return ``None`` → parametric fallback downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from maple.core.unit_registry import make_quantity, ureg

__all__ = ["capture_bootstrap_samples", "build_distribution_inputs"]


def build_distribution_inputs(empirical_data: dict) -> dict:
    """Build the ``{name: Pint Quantity}`` inputs dict for ``derive_distribution``
    from a target's ``empirical_data`` (literature inputs + modeling assumptions),
    mirroring CalibrationTarget validation (list values become Quantity arrays).
    """
    mock: dict[str, Any] = {}
    for inp in empirical_data.get("inputs") or []:
        v, u = inp["value"], inp["units"]
        mock[inp["name"]] = (np.array(v) * ureg(u)) if isinstance(v, list) else make_quantity(v, u)
    for a in empirical_data.get("assumptions") or []:
        v, u = a["value"], a["units"]
        mock[a["name"]] = (np.array(v) * ureg(u)) if isinstance(v, list) else (v * ureg(u))
    return mock


def _empirical_data_of(target: Any) -> dict | None:
    """Resolve the empirical_data dict from a YAML path, a parsed YAML/document
    dict, an empirical_data dict, or a CalibrationTarget-like object."""
    if isinstance(target, (str, Path)):
        doc = yaml.safe_load(Path(target).read_text()) or {}
        return doc.get("empirical_data") or {}
    if isinstance(target, dict):
        return target.get("empirical_data", target)
    ed = getattr(target, "empirical_data", None)
    if ed is None:
        return None
    if isinstance(ed, dict):
        return ed
    # pydantic model
    dump = getattr(ed, "model_dump", None)
    return dump() if callable(dump) else None


def capture_bootstrap_samples(target: Any, *, max_samples: int | None = None) -> np.ndarray | None:
    """Run a CalibrationTarget's ``distribution_code`` and return the population
    sample it declares via the ``samples`` return key, converted to
    ``empirical_data.units``.

    Args:
        target: a YAML path, a parsed YAML document dict, an ``empirical_data``
            dict, or a CalibrationTarget-like object with ``.empirical_data``.
        max_samples: if set and the captured array is larger, deterministically
            subsample to this size (evenly spaced over the sorted-by-index array).

    Returns:
        1-D float array of finite population samples in the target's units, or
        None when the code declares no ``samples`` key (summary-only / analytic
        derivations) or on any error — recovery is best-effort.
    """
    ed = _empirical_data_of(target)
    if not ed:
        return None
    code = ed.get("distribution_code")
    units = ed.get("units")
    if not code or not units:
        return None

    try:
        mock = build_distribution_inputs(ed)
        scope: dict[str, Any] = {"ureg": ureg, "np": np}
        exec(code, scope)
        fn = scope.get("derive_distribution")
        if fn is None:
            return None
        result = fn(mock, ureg)
    except Exception:
        return None

    if not isinstance(result, dict):
        return None
    samp = result.get("samples")
    if samp is None:
        return None

    try:
        if hasattr(samp, "to"):
            samp = samp.to(units)
        arr = np.asarray(samp.magnitude if hasattr(samp, "magnitude") else samp, dtype=float)
    except Exception:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return None
    if max_samples is not None and arr.size > max_samples:
        idx = np.linspace(0, arr.size - 1, max_samples).astype(int)
        arr = arr[idx]
    return arr
