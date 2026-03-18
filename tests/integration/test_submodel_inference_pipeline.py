"""Integration tests for the submodel prior inference pipeline.

End-to-end: CSV priors + SubmodelTarget YAMLs -> MCMC -> parameterized priors YAML.

Test cases:
  1. Single-parameter identity model (EC50 — param IS the observable)
  2. Multi-parameter algebraic model (Hill function with d_crit, n_hill)
  3. Shared parameter across two targets (k_shared constrained by two YAMLs)
  4. Analytical posterior regression (lognormal conjugate)
"""

from unittest.mock import patch

import numpy as np
import pytest

try:
    import jax  # noqa: F401
    import numpyro  # noqa: F401

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX/NumPyro not installed")


# =============================================================================
# Shared source_relevance block (valid for all test targets)
# =============================================================================

_SOURCE_RELEVANCE = """\
source_relevance:
  indication_match: proxy
  indication_match_justification: >
    Test data used as proxy for model context. Measurement is from a related
    but not identical experimental system.
  species_source: human
  species_target: human
  source_quality: primary_human_in_vitro
  perturbation_type: physiological_baseline
  perturbation_relevance: >
    Baseline measurement under physiological conditions with no external
    perturbation applied. Directly applicable to model parameter.
  tme_compatibility: moderate
  tme_compatibility_notes: >
    In vitro system approximates the in vivo biology but lacks full tissue
    context including stromal interactions and immune components.
  measurement_directness: direct
  temporal_resolution: endpoint_pair
  experimental_system: in_vitro_primary
"""


# =============================================================================
# Test case 1: Single-parameter identity model (based on EC50_GMCSF)
# =============================================================================

EC50_YAML = f"""\
target_id: ec50_test

study_interpretation: >
  EC50 measurement from dose-response. Identity forward model where the
  parameter directly equals the observed value.

key_assumptions:
  - "EC50 directly constrains the model parameter"

experimental_context:
  species: human
  system: in_vitro

primary_data_source:
  doi: "10.1038/test.001"
  source_tag: Test2024a
  title: "Test paper for EC50"

secondary_data_sources: []

inputs:
  - name: ec50_val_1
    value: 0.0055
    units: nanomolarity
    input_type: direct_measurement
    source_ref: Test2024a
    source_location: "Figure 3"
    value_snippet: "EC50 was 5.5 pM (0.0055 nM)"
  - name: ec50_val_2
    value: 0.0016
    units: nanomolarity
    input_type: direct_measurement
    source_ref: Test2024a
    source_location: "Figure 4"
    value_snippet: "EC50 was 1.6 pM (0.0016 nM)"
  - name: n_exp
    value: 2
    units: dimensionless
    input_type: direct_measurement
    source_ref: Test2024a
    source_location: "Methods"
    value_snippet: "representative of at least two experiments"

calibration:
  parameters:
    - name: EC50_test
      units: nanomolarity
  forward_model:
    type: algebraic
    formula: "EC50 = observed"
    code: |
      def compute(params, inputs):
          return params['EC50_test']
    data_rationale: "Direct EC50 measurement"
    submodel_rationale: "Hill function EC50"
  error_model:
    - name: ec50_obs
      units: nanomolarity
      uses_inputs: [ec50_val_1, ec50_val_2]
      sample_size_input: n_exp
      observation_code: |
        def derive_observation(inputs, sample_size, rng, n_bootstrap):
            import numpy as np
            vals = [inputs['ec50_val_1'], inputs['ec50_val_2']]
            log_vals = np.log(vals)
            mu = np.mean(log_vals)
            sigma = np.std(log_vals, ddof=1)
            return rng.lognormal(mean=mu, sigma=sigma, size=n_bootstrap)
  identifiability_notes: "Single parameter, directly observed."

{_SOURCE_RELEVANCE}
"""


# =============================================================================
# Test case 2: Multi-parameter algebraic (based on d_crit_T_n_pore)
# Two parameters, Hill function, evaluated at a single pore diameter
# =============================================================================

HILL_YAML = f"""\
target_id: hill_2param_test

study_interpretation: >
  Transmigration assay through filters with known pore diameters. Hill function
  models the transition from free to arrested migration as pore size decreases.

key_assumptions:
  - "Transmigration follows a Hill function of pore diameter"

experimental_context:
  species: human
  system: in_vitro

primary_data_source:
  doi: "10.1083/test.003"
  source_tag: Test2024c
  title: "Test paper for transmigration"

secondary_data_sources: []

inputs:
  - name: pore_diameter
    value: 3.0
    units: micrometer
    input_type: direct_measurement
    source_ref: Test2024c
    source_location: "Methods"
    value_snippet: "polycarbonate filters with 3 um pores"
  - name: transmigration_mean
    value: 10.0
    units: percent
    input_type: direct_measurement
    source_ref: Test2024c
    source_location: "Figure 7A"
    value_snippet: "transmigration at 3 um was approximately 10%"
  - name: transmigration_sd
    value: 0.4
    units: dimensionless
    input_type: direct_measurement
    source_ref: Test2024c
    source_location: "Figure 7A"
    value_snippet: "log-scale SD from error bars approximately 0.4"
  - name: normalization
    value: 100.0
    units: percent
    input_type: unit_conversion
    source_ref: Test2024c
    source_location: "Figure 7A"
    value_snippet: "normalized to 5 um pores as 100%"
    rationale: "Transmigration normalized to 5 um pore as 100% reference"
  - name: n_exp_hill
    value: 3
    units: dimensionless
    input_type: direct_measurement
    source_ref: Test2024c
    source_location: "Figure legend"
    value_snippet: "n = 3 independent experiments"

calibration:
  parameters:
    - name: d_crit
      units: micrometer
    - name: n_hill
      units: dimensionless
  forward_model:
    type: algebraic
    formula: "T = norm * (d/d_crit)^n / (1 + (d/d_crit)^n)"
    code: |
      def compute(params, inputs):
          import numpy as np
          d_crit = params['d_crit']
          n = params['n_hill']
          d_pore = inputs['pore_diameter']
          norm = inputs['normalization']
          ratio = d_pore / d_crit
          return norm * ratio**n / (1.0 + ratio**n)
    data_rationale: "Transmigration through defined pore sizes"
    submodel_rationale: "Hill function for physical arrest"
  error_model:
    - name: transmig_3um
      units: percent
      uses_inputs: [transmigration_mean, transmigration_sd]
      sample_size_input: n_exp_hill
      observation_code: |
        def derive_observation(inputs, sample_size, rng, n_bootstrap):
            import numpy as np
            median = inputs['transmigration_mean']
            log_sd = inputs['transmigration_sd']
            n = int(sample_size)
            return np.exp(rng.normal(np.log(median), log_sd / np.sqrt(n), n_bootstrap))
  identifiability_notes: >
    Two parameters from single observation point. Partially degenerate:
    prior information helps resolve d_crit vs n_hill.

{_SOURCE_RELEVANCE}
"""


# =============================================================================
# Test case 3: Shared parameter across two targets
# k_shared appears in both targets with different forward models
# =============================================================================

SHARED_TARGET_A = f"""\
target_id: shared_target_a

study_interpretation: >
  Direct measurement of rate constant k_shared via half-life assay.

key_assumptions:
  - "First-order kinetics"

experimental_context:
  species: human
  system: in_vitro

primary_data_source:
  doi: "10.1234/test.004a"
  source_tag: Test2024d
  title: "Test paper A for shared parameter"

secondary_data_sources: []

inputs:
  - name: halflife_a
    value: 7.0
    units: day
    input_type: direct_measurement
    source_ref: Test2024d
    source_location: "Table 2"
    value_snippet: "half-life of 7.0 days"
  - name: halflife_a_sd
    value: 0.2
    units: dimensionless
    input_type: direct_measurement
    source_ref: Test2024d
    source_location: "Table 2"
    value_snippet: "log-scale SD of 0.2"
  - name: n_a
    value: 8
    units: dimensionless
    input_type: direct_measurement
    source_ref: Test2024d
    source_location: "Methods"
    value_snippet: "n = 8 replicates"

calibration:
  parameters:
    - name: k_shared
      units: 1/day
  forward_model:
    type: algebraic
    formula: "t_half = ln(2) / k_shared"
    code: |
      def compute(params, inputs):
          import numpy as np
          return np.log(2) / params['k_shared']
    data_rationale: "Half-life measurement inverted to rate"
    submodel_rationale: "First-order decay rate"
  error_model:
    - name: halflife_obs_a
      units: day
      uses_inputs: [halflife_a, halflife_a_sd]
      sample_size_input: n_a
      observation_code: |
        def derive_observation(inputs, sample_size, rng, n_bootstrap):
            import numpy as np
            median = inputs['halflife_a']
            log_sd = inputs['halflife_a_sd']
            n = int(sample_size)
            return np.exp(rng.normal(np.log(median), log_sd / np.sqrt(n), n_bootstrap))
  identifiability_notes: "Single parameter, monotonic transform."

{_SOURCE_RELEVANCE}
"""

SHARED_TARGET_B = """\
target_id: shared_target_b

study_interpretation: >
  Independent measurement of the same rate constant k_shared from a different
  assay (steady-state concentration measurement).

key_assumptions:
  - "Steady-state relationship holds"

experimental_context:
  species: human
  system: in_vitro

primary_data_source:
  doi: "10.1234/test.004b"
  source_tag: Test2024e
  title: "Test paper B for shared parameter"

secondary_data_sources: []

inputs:
  - name: conc_ss
    value: 50.0
    units: nanomolarity
    input_type: direct_measurement
    source_ref: Test2024e
    source_location: "Figure 2"
    value_snippet: "steady-state concentration of 50 nM"
  - name: conc_ss_sd
    value: 0.25
    units: dimensionless
    input_type: direct_measurement
    source_ref: Test2024e
    source_location: "Figure 2"
    value_snippet: "log-scale SD of 0.25"
  - name: production_rate
    value: 5.0
    units: nanomolarity/day
    input_type: direct_measurement
    source_ref: Test2024e
    source_location: "Table 1"
    value_snippet: "production rate of 5.0 nM/day"
  - name: n_b
    value: 6
    units: dimensionless
    input_type: direct_measurement
    source_ref: Test2024e
    source_location: "Methods"
    value_snippet: "n = 6 replicates"

calibration:
  parameters:
    - name: k_shared
      units: 1/day
  forward_model:
    type: algebraic
    formula: "C_ss = production_rate / k_shared"
    code: |
      def compute(params, inputs):
          return inputs['production_rate'] / params['k_shared']
    data_rationale: "Steady-state concentration constrains degradation rate"
    submodel_rationale: "At steady state, production = degradation"
  error_model:
    - name: conc_obs_b
      units: nanomolarity
      uses_inputs: [conc_ss, conc_ss_sd]
      sample_size_input: n_b
      observation_code: |
        def derive_observation(inputs, sample_size, rng, n_bootstrap):
            import numpy as np
            median = inputs['conc_ss']
            log_sd = inputs['conc_ss_sd']
            n = int(sample_size)
            return np.exp(rng.normal(np.log(median), log_sd / np.sqrt(n), n_bootstrap))
  identifiability_notes: "k_shared identifiable from steady-state concentration given known production rate."

source_relevance:
  indication_match: exact
  indication_match_justification: >
    Direct measurement from the same cell type and disease context as the model.
    No cross-indication translation needed.
  species_source: human
  species_target: human
  source_quality: primary_human_in_vitro
  perturbation_type: physiological_baseline
  perturbation_relevance: >
    Baseline steady-state measurement. No perturbation applied.
  tme_compatibility: high
  tme_compatibility_notes: >
    In vitro system closely matches the model context. Minimal translation
    uncertainty expected.
  measurement_directness: single_inversion
  temporal_resolution: snapshot_or_equilibrium
  experimental_system: in_vitro_primary
"""


# =============================================================================
# Priors CSV covering all test parameters
# =============================================================================

PRIORS_CSV = """\
name,median,units,distribution,dist_param1,dist_param2
EC50_test,0.01,nanomolarity,lognormal,-4.6,2.0
d_crit,3.5,micrometer,lognormal,1.25,0.5
n_hill,15.0,dimensionless,lognormal,2.7,0.5
k_shared,0.1,1/day,lognormal,-2.3,1.0
"""


# =============================================================================
# Mock DOI resolution
# =============================================================================


DOI_TITLES = {
    "10.1038/test.001": "Test paper for EC50",
    "10.1083/test.003": "Test paper for transmigration",
    "10.1234/test.004a": "Test paper A for shared parameter",
    "10.1234/test.004b": "Test paper B for shared parameter",
}


@pytest.fixture(autouse=True)
def mock_doi_resolution():
    def mock_resolve(doi):
        return {
            "title": DOI_TITLES.get(doi, "Test paper"),
            "year": 2024,
            "first_author": "Test",
        }

    with patch(
        "maple.core.calibration.validators.resolve_doi",
        side_effect=mock_resolve,
    ):
        yield


# =============================================================================
# Tests
# =============================================================================


class TestProcessYaml:
    """Test process_yaml (single-target MCMC, used by MCP validation)."""

    def test_single_param_returns_expected_keys(self, tmp_path):
        from maple.core.calibration.yaml_to_prior import process_yaml

        (tmp_path / "ec50.yaml").write_text(EC50_YAML)
        (tmp_path / "priors.csv").write_text(PRIORS_CSV)

        results = process_yaml(tmp_path / "ec50.yaml", priors_csv=tmp_path / "priors.csv")

        assert len(results) == 1
        result = results[0]
        assert "error" not in result
        expected_keys = {
            "name",
            "units",
            "target_id",
            "best_dist",
            "all_fits",
            "param_samples",
            "median_data",
            "sigma_data",
            "translation_sigma",
            "translation_breakdown",
            "median_prior",
            "sigma_prior",
            "mu_prior",
            "cv_data",
            "cv_prior",
            "mcmc_diagnostics",
        }
        assert expected_keys <= set(result.keys())
        assert result["name"] == "EC50_test"
        assert result["units"] == "nanomolarity"
        assert 0.0005 < result["median_prior"] < 0.05

    def test_format_report_works(self, tmp_path):
        from maple.core.calibration.yaml_to_prior import format_report, process_yaml

        (tmp_path / "ec50.yaml").write_text(EC50_YAML)
        (tmp_path / "priors.csv").write_text(PRIORS_CSV)

        results = process_yaml(tmp_path / "ec50.yaml", priors_csv=tmp_path / "priors.csv")
        report = format_report(results[0])

        assert "EC50_test" in report
        assert "lognormal" in report.lower() or "gamma" in report.lower()

    def test_missing_prior_returns_error(self, tmp_path):
        from maple.core.calibration.yaml_to_prior import process_yaml

        (tmp_path / "ec50.yaml").write_text(EC50_YAML)
        # CSV missing EC50_test
        (tmp_path / "priors.csv").write_text(
            "name,median,units,distribution,dist_param1,dist_param2\n"
            "other_param,1.0,dimensionless,lognormal,0.0,1.0\n"
        )

        results = process_yaml(tmp_path / "ec50.yaml", priors_csv=tmp_path / "priors.csv")
        assert len(results) == 1
        assert "error" in results[0]
        assert "not found in priors CSV" in results[0]["error"]

    def test_multi_param_hill(self, tmp_path):
        from maple.core.calibration.yaml_to_prior import process_yaml

        (tmp_path / "hill.yaml").write_text(HILL_YAML)
        (tmp_path / "priors.csv").write_text(PRIORS_CSV)

        # process_yaml reports on all parameters
        results = process_yaml(tmp_path / "hill.yaml", priors_csv=tmp_path / "priors.csv")
        assert len(results) == 2
        names = {r["name"] for r in results}
        assert names == {"d_crit", "n_hill"}
        for r in results:
            assert "error" not in r
            assert r["median_prior"] > 0


class TestSingleParameterIdentity:
    """Test case 1: EC50 with identity forward model."""

    def test_posterior_near_data(self, tmp_path):
        from maple.core.calibration.yaml_to_prior import process_targets

        (tmp_path / "ec50.yaml").write_text(EC50_YAML)
        (tmp_path / "priors.csv").write_text(PRIORS_CSV)

        result = process_targets(
            priors_csv=tmp_path / "priors.csv",
            yaml_paths=[tmp_path / "ec50.yaml"],
            output_dir=tmp_path / "output",
            num_warmup=200,
            num_samples=500,
            num_chains=1,
        )

        assert result["metadata"]["n_parameters"] == 1  # only target-referenced params
        ec50 = [p for p in result["parameters"] if p["name"] == "EC50_test"][0]
        median = ec50["marginal"]["median"]

        # Geometric mean of 0.0055 and 0.0016 ≈ 0.003
        # Posterior should be in that neighborhood
        assert 0.0005 < median < 0.05

        # YAML file written
        assert (tmp_path / "output" / "submodel_priors.yaml").exists()


class TestMultiParameterHill:
    """Test case 2: Two-parameter Hill function."""

    def test_both_params_estimated(self, tmp_path):
        from maple.core.calibration.yaml_to_prior import process_targets

        (tmp_path / "hill.yaml").write_text(HILL_YAML)
        (tmp_path / "priors.csv").write_text(PRIORS_CSV)

        result = process_targets(
            priors_csv=tmp_path / "priors.csv",
            yaml_paths=[tmp_path / "hill.yaml"],
            num_warmup=200,
            num_samples=500,
            num_chains=1,
        )

        param_names = {p["name"] for p in result["parameters"]}
        assert "d_crit" in param_names
        assert "n_hill" in param_names

        d_crit = [p for p in result["parameters"] if p["name"] == "d_crit"][0]
        n_hill = [p for p in result["parameters"] if p["name"] == "n_hill"][0]

        # d_crit should be in a biologically reasonable range (1-10 um)
        assert 0.5 < d_crit["marginal"]["median"] < 20.0
        # n_hill should be positive
        assert n_hill["marginal"]["median"] > 0


class TestSharedParameter:
    """Test case 3: Same parameter constrained by two targets."""

    def test_shared_param_sampled_once(self, tmp_path):
        from maple.core.calibration.yaml_to_prior import process_targets

        (tmp_path / "target_a.yaml").write_text(SHARED_TARGET_A)
        (tmp_path / "target_b.yaml").write_text(SHARED_TARGET_B)
        (tmp_path / "priors.csv").write_text(PRIORS_CSV)

        result = process_targets(
            priors_csv=tmp_path / "priors.csv",
            yaml_paths=[tmp_path / "target_a.yaml", tmp_path / "target_b.yaml"],
            output_dir=tmp_path / "output",
            num_warmup=200,
            num_samples=500,
            num_chains=1,
        )

        # k_shared should appear exactly once in parameters
        k_entries = [p for p in result["parameters"] if p["name"] == "k_shared"]
        assert len(k_entries) == 1

        k = k_entries[0]
        assert k["marginal"]["median"] > 0

        # Should have source_targets from both targets
        assert "source_targets" in k
        assert set(k["source_targets"]) == {"shared_target_a", "shared_target_b"}

    def test_shared_tighter_than_single(self, tmp_path):
        """Two targets constraining the same param should give tighter posterior."""
        from maple.core.calibration.yaml_to_prior import process_targets

        (tmp_path / "target_a.yaml").write_text(SHARED_TARGET_A)
        (tmp_path / "target_b.yaml").write_text(SHARED_TARGET_B)
        (tmp_path / "priors.csv").write_text(PRIORS_CSV)

        # Single target
        result_single = process_targets(
            priors_csv=tmp_path / "priors.csv",
            yaml_paths=[tmp_path / "target_a.yaml"],
            num_warmup=300,
            num_samples=2000,
            num_chains=1,
        )

        # Both targets
        result_both = process_targets(
            priors_csv=tmp_path / "priors.csv",
            yaml_paths=[tmp_path / "target_a.yaml", tmp_path / "target_b.yaml"],
            num_warmup=300,
            num_samples=2000,
            num_chains=1,
        )

        k_single = [p for p in result_single["parameters"] if p["name"] == "k_shared"][0]
        k_both = [p for p in result_both["parameters"] if p["name"] == "k_shared"][0]

        # More data should yield smaller CV (tighter posterior)
        assert k_both["marginal"]["cv"] < k_single["marginal"]["cv"]


class TestAnalyticalPosterior:
    """Lognormal prior + lognormal likelihood -> known analytical posterior."""

    def test_conjugate_posterior(self):
        import jax
        from numpyro.infer import MCMC, NUTS

        from maple.core.calibration.submodel_inference import (
            ErrorModelEntry,
            PriorSpec,
            TargetLikelihood,
            submodel_joint_model,
        )
        from maple.core.calibration.yaml_to_prior import DistFit

        mu_0, sig_0 = 0.0, 0.5
        obs_value = 2.0
        obs_sigma = 0.3
        sigma_trans = 0.15

        prior_specs = {
            "k": PriorSpec(
                name="k",
                distribution="lognormal",
                units="1/day",
                mu=mu_0,
                sigma=sig_0,
            )
        }

        entry = ErrorModelEntry(
            forward_fn=lambda params: params["k"],
            value=obs_value,
            sigma=obs_sigma,
            family="lognormal",
            fit=DistFit(
                name="lognormal",
                params={"mu": np.log(obs_value), "sigma": obs_sigma},
                aic=0,
                ad_stat=0,
                ad_crit_5pct=1,
                ad_pass=True,
                median=obs_value,
                cv=0.3,
            ),
        )

        tl = TargetLikelihood(
            target_id="test",
            sigma_trans=sigma_trans,
            sigma_breakdown={},
            entries=[entry],
        )

        kernel = NUTS(submodel_joint_model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=5000, num_chains=2)
        mcmc.run(
            jax.random.PRNGKey(0),
            prior_specs=prior_specs,
            target_likelihoods=[tl],
        )

        log_k = np.log(np.asarray(mcmc.get_samples()["k"]))

        # Analytical posterior for lognormal-lognormal conjugate
        sigma_total = np.sqrt(obs_sigma**2 + sigma_trans**2)
        prec_prior = 1 / sig_0**2
        prec_lik = 1 / sigma_total**2
        mu_post = (mu_0 * prec_prior + np.log(obs_value) * prec_lik) / (prec_prior + prec_lik)
        sig_post = np.sqrt(1 / (prec_prior + prec_lik))

        assert np.mean(log_k) == pytest.approx(mu_post, abs=0.05)
        assert np.std(log_k) == pytest.approx(sig_post, abs=0.05)


class TestOutputArtifacts:
    """Verify output files are complete and valid."""

    def test_yaml_roundtrip(self, tmp_path):
        from ruamel.yaml import YAML

        from maple.core.calibration.yaml_to_prior import process_targets

        (tmp_path / "ec50.yaml").write_text(EC50_YAML)
        (tmp_path / "priors.csv").write_text(PRIORS_CSV)

        process_targets(
            priors_csv=tmp_path / "priors.csv",
            yaml_paths=[tmp_path / "ec50.yaml"],
            output_dir=tmp_path / "output",
            num_warmup=100,
            num_samples=200,
            num_chains=1,
        )

        yaml = YAML()
        loaded = yaml.load(tmp_path / "output" / "submodel_priors.yaml")

        assert "metadata" in loaded
        assert "parameters" in loaded
        assert loaded["metadata"]["n_parameters"] == 1  # only target-referenced params

        for param in loaded["parameters"]:
            assert "name" in param
            assert "marginal" in param
            assert "distribution" in param["marginal"]
            assert "median" in param["marginal"]

    def test_csv_export(self, tmp_path):
        import pandas as pd

        from maple.core.calibration.yaml_to_prior import process_targets

        (tmp_path / "ec50.yaml").write_text(EC50_YAML)
        (tmp_path / "priors.csv").write_text(PRIORS_CSV)

        process_targets(
            priors_csv=tmp_path / "priors.csv",
            yaml_paths=[tmp_path / "ec50.yaml"],
            export_csv=tmp_path / "exported.csv",
            num_warmup=100,
            num_samples=200,
            num_chains=1,
        )

        df = pd.read_csv(tmp_path / "exported.csv")
        assert "EC50_test" in df["name"].values
        assert len(df) == 1  # only target-referenced params
