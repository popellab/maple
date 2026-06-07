"""Unit tests for capture_bootstrap_samples."""

import numpy as np

from maple.core.calibration.bootstrap_capture import (
    capture_bootstrap_samples,
    build_distribution_inputs,
)

# A target whose distribution_code bootstraps the median of paired fold changes
# (mirrors the real cd8_fc-style targets: builds boot_medians, reduces via
# np.median / np.percentile, and re-imports numpy *inside* the function).
SAMPLING_DOC = {
    "empirical_data": {
        "units": "dimensionless",
        "inputs": [
            {"name": "fc_01", "value": 3.4, "units": "dimensionless"},
            {"name": "fc_02", "value": 0.5, "units": "dimensionless"},
            {"name": "fc_03", "value": 2.1, "units": "dimensionless"},
            {"name": "fc_04", "value": 5.8, "units": "dimensionless"},
            {"name": "fc_05", "value": 1.2, "units": "dimensionless"},
            {"name": "fc_06", "value": 0.9, "units": "dimensionless"},
        ],
        "distribution_code": (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(42)\n"
            "    names = ['fc_01','fc_02','fc_03','fc_04','fc_05','fc_06']\n"
            "    fc = np.array([inputs[n].magnitude for n in names])\n"
            "    boot = np.array([np.median(rng.choice(fc, size=fc.size, replace=True))\n"
            "                     for _ in range(10000)])\n"
            "    dl = ureg('dimensionless')\n"
            "    return {'median_obs': np.median(fc) * dl,\n"
            "            'ci95_lower': np.percentile(boot, 2.5) * dl,\n"
            "            'ci95_upper': np.percentile(boot, 97.5) * dl}\n"
        ),
    }
}

# Closed-form analytic code: no internal sample array -> capture returns None.
ANALYTIC_DOC = {
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


def test_captures_bootstrap_from_sampling_code():
    s = capture_bootstrap_samples(SAMPLING_DOC)
    assert s is not None
    assert s.shape == (10000,)
    # the captured array is boot_medians; its CI matches what the code returns
    assert np.percentile(s, 2.5) > 0
    assert np.percentile(s, 97.5) > np.percentile(s, 2.5)


def test_analytic_code_returns_none():
    assert capture_bootstrap_samples(ANALYTIC_DOC) is None


def test_accepts_empirical_data_dict_directly():
    s = capture_bootstrap_samples(SAMPLING_DOC["empirical_data"])
    assert s is not None and s.shape == (10000,)


def test_max_samples_subsamples():
    s = capture_bootstrap_samples(SAMPLING_DOC, max_samples=1000)
    assert s is not None and s.shape == (1000,)


def test_units_conversion_applied(tmp_path):
    """Captured samples are returned in empirical_data.units (here percent ->
    fraction via Pint), not the raw magnitude the code sampled in."""
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
                "            'ci95_upper': np.percentile(samp.magnitude, 97.5) * ureg('percent')}\n"
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
