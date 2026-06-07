"""Capture the internal bootstrap sample array a CalibrationTarget's
``empirical_data.distribution_code`` reduces to median/CI95.

The ``derive_distribution(inputs, ureg)`` function only returns ``median_obs`` /
``ci95_lower`` / ``ci95_upper`` — but most targets build an explicit ``samples``
(or ``boot_medians``) array and then call ``np.median`` / ``np.percentile`` /
``np.quantile`` on it. That array is the sampling distribution of the observed
statistic — the *real* observation noise, with its true skew/tails/bounds. We
recover it by intercepting those numpy reductions during execution.

Used by the SBI pipeline to feed empirical observation noise into
``add_observation_noise`` (instead of a parametric lognormal/Gaussian refit of
the CI endpoints, which only matches when the bootstrap is log-symmetric), and
by diagnostics that overlay the observed distribution.
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
    """Run a CalibrationTarget's ``distribution_code`` and return the internal
    bootstrap sample array, converted to ``empirical_data.units``.

    Args:
        target: a YAML path, a parsed YAML document dict, an ``empirical_data``
            dict, or a CalibrationTarget-like object with ``.empirical_data``.
        max_samples: if set and the captured array is larger, deterministically
            subsample to this size (evenly spaced over the sorted-by-index array).

    Returns:
        1-D float array of finite bootstrap samples in the target's units, or None
        for analytic/closed-form codes that never build a sample array (and on any
        error — capture is best-effort).
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
    except Exception:
        return None

    captured: list = []
    real_pct, real_med, real_quant = np.percentile, np.median, np.quantile

    def _rec(a):
        arr = a.magnitude if hasattr(a, "magnitude") else a
        try:
            arr = np.asarray(arr)
        except Exception:
            return
        if arr.ndim >= 1 and arr.size > 1:
            captured.append(a)

    def _p(a, *ar, **kw):
        _rec(a)
        return real_pct(a, *ar, **kw)

    def _m(a, *ar, **kw):
        _rec(a)
        return real_med(a, *ar, **kw)

    def _q(a, *ar, **kw):
        _rec(a)
        return real_quant(a, *ar, **kw)

    # The distribution_code may `import numpy as np` inside the function, which
    # would bypass an injected proxy — so patch the real module functions.
    try:
        np.percentile, np.median, np.quantile = _p, _m, _q
        scope: dict[str, Any] = {"ureg": ureg, "np": np}
        exec(code, scope)
        fn = scope.get("derive_distribution")
        if fn is None:
            return None
        fn(mock, ureg)
    except Exception:
        return None
    finally:
        np.percentile, np.median, np.quantile = real_pct, real_med, real_quant

    if not captured:
        return None

    def _size(a):
        m = a.magnitude if hasattr(a, "magnitude") else a
        return np.asarray(m).size

    samp = max(captured, key=_size)
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
