"""Unit tests for capture_bootstrap_samples (declared 'samples' output)."""

import numpy as np

from maple.core.calibration.bootstrap_capture import (
    capture_bootstrap_samples,
    build_distribution_inputs,
)

# A target whose distribution_code declares the across-patient POPULATION sample
# via the 'samples' return key (the canonical hierarchical-omega signal). Its
# spread is patient-to-patient variability (~sd_frac), not a sqrt-n-shrunk SEM.
POPULATION_DOC = {
    "empirical_data": {
        "units": "dimensionless",
        "inputs": [
            {"name": "mean_frac", "value": 0.30, "units": "dimensionless"},
            {"name": "sd_frac", "value": 0.17, "units": "dimensionless"},
        ],
        "distribution_code": (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(42)\n"
            "    m = inputs['mean_frac'].magnitude\n"
            "    s = inputs['sd_frac'].magnitude\n"
            "    dl = ureg('dimensionless')\n"
            "    samples = rng.normal(m, s, 10000) * dl\n"
            "    return {'median_obs': np.median(samples),\n"
            "            'ci95_lower': np.percentile(samples, 2.5),\n"
            "            'ci95_upper': np.percentile(samples, 97.5),\n"
            "            'samples': samples}\n"
        ),
    }
}

# Summary-only code: no 'samples' key -> capture returns None (the interception
# path is retired, so summary-only / analytic codes get parametric fallback).
SUMMARY_ONLY_DOC = {
    "empirical_data": {
        "units": "dimensionless",
        "inputs": [
            {"name": "mean_frac", "value": 0.02, "units": "dimensionless"},
            {"name": "sd_frac", "value": 0.03, "units": "dimensionless"},
        ],
        "distribution_code": (
            "def derive_distribution(inputs, ureg):\n"
            "    import math\n"
            "    m = inputs['mean_frac'].to('dimensionless').magnitude\n"
            "    s = inputs['sd_frac'].to('dimensionless').magnitude\n"
            "    cv = s / m\n"
            "    sigma = math.sqrt(math.log(1 + cv*cv))\n"
            "    mu = math.log(m) - 0.5*sigma*sigma\n"
            "    dl = ureg('dimensionless')\n"
            "    return {'median_obs': math.exp(mu) * dl,\n"
            "            'ci95_lower': math.exp(mu - 1.96*sigma) * dl,\n"
            "            'ci95_upper': math.exp(mu + 1.96*sigma) * dl}\n"
        ),
    }
}


def test_returns_declared_population_sample():
    s = capture_bootstrap_samples(POPULATION_DOC)
    assert s is not None
    assert s.shape == (10000,)
    # spread is the across-patient population spread (~sd_frac), not sqrt-n shrunk
    assert abs(np.std(s) - 0.17) < 0.02
    assert abs(np.median(s) - 0.30) < 0.02


def test_summary_only_returns_none():
    assert capture_bootstrap_samples(SUMMARY_ONLY_DOC) is None


def test_accepts_empirical_data_dict_directly():
    s = capture_bootstrap_samples(POPULATION_DOC["empirical_data"])
    assert s is not None and s.shape == (10000,)


def test_max_samples_subsamples():
    s = capture_bootstrap_samples(POPULATION_DOC, max_samples=1000)
    assert s is not None and s.shape == (1000,)


def test_units_conversion_applied():
    """Declared samples are returned in empirical_data.units (percent -> fraction
    via Pint), not the raw magnitude the code sampled in."""
    doc = {
        "empirical_data": {
            "units": "dimensionless",
            "inputs": [{"name": "m", "value": 5.0, "units": "percent"}],
            "distribution_code": (
                "def derive_distribution(inputs, ureg):\n"
                "    import numpy as np\n"
                "    m = inputs['m'].to('percent').magnitude\n"
                "    rng = np.random.default_rng(0)\n"
                "    samp = rng.normal(m, 0.5, size=5000) * ureg('percent')\n"
                "    return {'median_obs': np.median(samp).to('dimensionless'),\n"
                "            'ci95_lower': np.percentile(samp.magnitude, 2.5) * ureg('percent'),\n"
                "            'ci95_upper': np.percentile(samp.magnitude, 97.5) * ureg('percent'),\n"
                "            'samples': samp}\n"
            ),
        }
    }
    s = capture_bootstrap_samples(doc)
    assert s is not None
    # 5 percent -> 0.05 fraction after conversion to dimensionless units
    assert abs(np.median(s) - 0.05) < 0.01


def test_build_distribution_inputs_arrays_and_scalars():
    ed = {
        "inputs": [{"name": "v", "value": [1.0, 2.0], "units": "dimensionless"}],
        "assumptions": [{"name": "k", "value": 3.0, "units": "dimensionless"}],
    }
    d = build_distribution_inputs(ed)
    assert d["v"].magnitude.tolist() == [1.0, 2.0]
    assert float(d["k"].magnitude) == 3.0
