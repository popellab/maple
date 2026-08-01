#!/usr/bin/env python3
"""
Shared Pydantic models used across different workflows.

These models are defined here to avoid circular imports between
calibration_target_models.py and pydantic_models.py.
"""

import math
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from maple.core.calibration.enums import (
    ExperimentalSystem,
    ExtractionMethod,
    HeterogeneityTransfer,
    IndicationMatch,
    MeasurementDirectness,
    PerturbationType,
    SourceQuality,
    SourceType,
    TemporalResolution,
    TMECompatibility,
)


class InputType(str, Enum):
    """Classification of literature input data type for calibration targets."""

    DIRECT_PARAMETER = "direct_parameter"
    """Literature reports the model parameter value directly (e.g., 'k = 3/day')."""

    PROXY_MEASUREMENT = "proxy_measurement"
    """Literature reports a proxy that requires conversion (e.g., 'doubling time = 8h' → k = ln(2)/t)."""

    EXPERIMENTAL_CONDITION = "experimental_condition"
    """Protocol/experimental choice from paper (e.g., seeding density, E:T ratio)."""

    INFERRED_ESTIMATE = "inferred_estimate"
    """Value interpreted from qualitative text (e.g., 'maintained viability' → 0.95).

    Use when the numeric value does not appear literally in the paper but is
    a reasonable interpretation of qualitative statements. The value_snippet
    should contain the qualitative text that supports the interpretation,
    and the description should explain how the value was derived.

    Snippet validation is skipped for this input type since the value is
    not expected to appear literally in the text.
    """

    DERIVED_ARITHMETIC = "derived_arithmetic"
    """Deterministic arithmetic derivation from other inputs.

    Use when a value is calculated from extracted inputs via an explicit
    formula (e.g., fold change = post / pre, slope * time, ratio of
    measurements within the same patient). The formula is evaluated at
    schema-validation time and checked against the declared value.

    Requires ``formula`` and ``source_inputs`` fields on the Input.
    Snippet/excerpt validation is skipped since the derived value does
    not appear literally in the source text.
    """


class UncertaintyType(str, Enum):
    """Type of uncertainty measure reported in literature."""

    SD = "sd"  # Standard deviation
    SE = "se"  # Standard error
    CI95 = "ci95"  # 95% confidence interval
    RANGE = "range"  # Min-max range
    IQR = "iqr"  # Interquartile range


# =============================================================================
# POPULATION SPREAD PROVENANCE + QUANTILE-ANCHOR DISTRIBUTION
# =============================================================================
#
# Shared across CalibrationTarget (empirical_data) and SubmodelTarget (error_model).
# Motivation: a single reported "spread" silently stands in for several distinct
# quantities (uncertainty on the mean, biological SD across experimental units,
# in-vitro→in-vivo translation gap, and target-population variability). For
# hierarchical (virtual-patient) inference only two are needed, cleanly separated:
# the CENTER (feeds the population mean mu) and the POPULATION SPREAD (feeds omega).
# The tags below make that provenance explicit so the inference can route each
# quantity to the right hyperparameter instead of reusing an SEM-scale width as
# population variability. See the reparameterized-hierarchical methods writeup.


class SpreadSource(str, Enum):
    """Provenance of a reported spread — what kind of variability it measures.

    This drives whether a spread feeds the population-spread hyperparameter
    (omega) in hierarchical inference or only the center's error budget. It is
    the sole declaration of that, on both calibration and submodel targets.
    """

    ACROSS_PATIENT = "across_patient"
    """Genuine inter-individual spread in the TARGET population (e.g. PDAC
    patients). The gold standard for population spread; feeds omega directly."""

    BIOLOGICAL_EXPERIMENTAL = "biological_experimental"
    """Biological SD across experimental units (donors/animals) in an assay.
    A physical spread, but almost certainly a LOWER BOUND on patient-to-patient
    variability (controlled system, often healthy donors). Feeds omega, but may
    need translation widening (see ``translation``)."""

    TRANSLATION = "translation"
    """In-vitro/ex-vivo -> in-vivo PDAC transfer gap (species, missing TME,
    proxy cell types). An amplification/modifier on the spread, not a base
    magnitude; graded from source_relevance. May widen omega."""

    ASSUMED = "assumed"
    """No valid spread information; a deliberately wide, class-based mechanistic
    default. Tagged so it is never mistaken for a data-anchored constraint."""

    TECHNICAL = "technical"
    """Assay/technical-replicate noise. A spread, but NOT biological — it must
    NOT feed population spread."""

    CENTER_ONLY = "center_only"
    """The reported width is uncertainty on the MEAN (SEM / CI on the estimate),
    not a population spread. Feeds the center's error budget only."""


# Which provenance tags contribute a genuine population-spread MAGNITUDE (omega).
# ``translation`` and ``assumed`` are modifiers/defaults applied on top; they are
# handled by the inference layer, not counted as a base spread here.
POPULATION_SPREAD_SOURCES: frozenset = frozenset(
    {SpreadSource.ACROSS_PATIENT, SpreadSource.BIOLOGICAL_EXPERIMENTAL}
)


class ExperimentalUnitType(str, Enum):
    """What one replicate in the reported n actually is.

    Gates the ``SD = SEM * sqrt(n)`` round-trip: it is valid ONLY when n counts
    biological units. SEM over technical replicates yields a meaningless "SD".
    """

    BIOLOGICAL = "biological"  # distinct donors / animals / patients
    TECHNICAL = "technical"  # replicate wells / measurements of the same unit
    CLONAL = "clonal"  # clones / passages of one line — treat as technical for spread


class StatKind(str, Enum):
    """A statistic a source can print. What it is, not what a model does with it."""

    QUANTILE = "quantile"  # needs `p`; the median is p=0.5
    MEAN = "mean"
    GEOMETRIC_MEAN = "geometric_mean"
    SD = "sd"  # dispersion of the sample
    CV = "cv"  # sd / mean, dimensionless
    IQR = "iqr"  # width, q75 - q25
    RANGE = "range"  # width, max - min
    SE = "se"  # dispersion of the estimator, not the sample
    CI95_LO = "ci95_lo"
    CI95_HI = "ci95_hi"
    MIN = "min"
    MAX = "max"


#: Statistics that locate the distribution.
LOCATION_STATS: frozenset = frozenset({StatKind.MEAN, StatKind.GEOMETRIC_MEAN})
#: Statistics that describe a width, in the value's own units unless noted.
WIDTH_STATS: frozenset = frozenset({StatKind.SD, StatKind.CV, StatKind.IQR, StatKind.RANGE})
#: Widths of an estimator rather than of the sample. These carry a population
#: width too, but only alongside the n they were computed over: sd = se * sqrt(n).
#: Kept apart from WIDTH_STATS so nothing reads one as the other by accident.
SAMPLING_WIDTH_STATS: frozenset = frozenset({StatKind.SE})


class ReportedStatistic(BaseModel):
    """One number the source printed."""

    model_config = ConfigDict(extra="forbid")

    stat: StatKind = Field(description="Which statistic this is.")
    value: float = Field(
        description="The printed value, in the target's units (dimensionless for cv)."
    )
    p: Optional[float] = Field(
        default=None,
        description="Probability level in (0, 1). Required for stat='quantile', forbidden "
        "otherwise. The median is p=0.5.",
    )

    @model_validator(mode="after")
    def _p_matches_stat(self) -> "ReportedStatistic":
        if self.stat == StatKind.QUANTILE:
            if self.p is None:
                raise ValueError("stat='quantile' needs a probability level p.")
            if not (0.0 < self.p < 1.0):
                raise ValueError(f"quantile p must be in (0, 1), got {self.p}.")
        elif self.p is not None:
            raise ValueError(f"stat='{self.stat.value}' must not set p; only quantiles have one.")
        if self.stat in (WIDTH_STATS | SAMPLING_WIDTH_STATS) and self.value < 0:
            raise ValueError(f"stat='{self.stat.value}' is a width and cannot be negative.")
        return self


# Standard normal quantiles used to expand a scalar scale into quartiles.
_Z_Q = 0.6744897501960817  # Phi^-1(0.75)
_Z_95 = 1.959963984540054  # Phi^-1(0.975)


class DistributionShape(str, Enum):
    """Shape used to expand a center + scale into quantiles, when asked."""

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    LOGIT_NORMAL = "logit_normal"  # bounded to (0, 1): fractions, probabilities


def _quartiles_from_center_scale(
    center: float, center_is_mean: bool, sd: float, shape: DistributionShape
) -> tuple:
    """Expand a center and a normal-equivalent SD into (q25, q50, q75)."""
    if shape == DistributionShape.NORMAL:
        return (center - _Z_Q * sd, center, center + _Z_Q * sd)

    if shape == DistributionShape.LOGIT_NORMAL:
        if not (0.0 < center < 1.0):
            raise ValueError(
                f"shape='logit_normal' needs a center in (0, 1), got {center}. Express a "
                "percent as a fraction."
            )
        mu_l = math.log(center / (1.0 - center))
        sigma_l = sd / (center * (1.0 - center))

        def _expit(z: float) -> float:
            return 1.0 / (1.0 + math.exp(-z))

        return (_expit(mu_l - _Z_Q * sigma_l), center, _expit(mu_l + _Z_Q * sigma_l))

    # lognormal
    if center <= 0:
        raise ValueError(f"shape='lognormal' needs a positive center, got {center}.")
    cv = sd / abs(center)
    sigma_ln = math.sqrt(math.log(1.0 + cv**2))
    median = center / math.sqrt(1.0 + cv**2) if center_is_mean else center
    return (
        median * math.exp(-_Z_Q * sigma_ln),
        median,
        median * math.exp(_Z_Q * sigma_ln),
    )


class ObservedDistribution(BaseModel):
    """What a source printed about a distribution, as a flat list of statistics.

    One entry is one number the paper reported. ``spread_source`` says whether the
    width among them is genuine population variability. Nothing here says what a
    model should do with any entry; that is the consumer's decision.

    ``quantile`` / ``median`` / ``iqr`` derive values on demand, using explicit
    quantile entries where available and otherwise expanding a center and scale
    through ``shape``.
    """

    model_config = ConfigDict(extra="forbid")

    statistics: List[ReportedStatistic] = Field(
        description="Every statistic the source printed for this quantity. Record what was "
        "actually on the page: median plus quartiles, mean plus SD, an SE, a range. Do not "
        "convert between them."
    )
    spread_source: SpreadSource = Field(
        description="Provenance of the spread these statistics describe (see SpreadSource)."
    )
    shape: Optional[DistributionShape] = Field(
        default=None,
        description="Shape assumed when deriving quantiles from a center and a scale. Only "
        "needed when the source printed no quartiles and a consumer asks for them; recorded "
        "here so the assumption is explicit rather than applied silently.",
    )
    n_biological: Optional[int] = Field(
        default=None,
        ge=1,
        description="Biological units (donors/animals/patients) the summary is computed over. "
        "For submodel targets, where there is no cohort registry to carry it. Calibration "
        "targets name a cohort instead and must leave this unset.",
    )
    n_biological_is_floor: bool = Field(
        default=False,
        description="True when ``n_biological`` is a lower bound rather than an exact count.",
    )
    n_technical: Optional[int] = Field(
        default=None, ge=1, description="Technical replicates, when reported separately."
    )
    experimental_unit_type: Optional[ExperimentalUnitType] = Field(
        default=None,
        description="What one replicate is. Gates whether an SE can be read as a population "
        "SD. Submodel side; calibration targets carry it on the cohort.",
    )
    unit_group: Optional[str] = Field(
        default=None,
        description="Shared biological-unit group, for observables measured on the SAME units "
        "(one donor panel across the doses of a dose-response). Grouped observables share one "
        "random effect rather than counting as independent measurements. Submodel side; "
        "calibration targets name a cohort instead.",
    )

    @model_validator(mode="after")
    def _statistics_well_formed(self) -> "ObservedDistribution":
        if not self.statistics:
            raise ValueError("observed_distribution.statistics must have at least one entry.")

        keys = [(s.stat, s.p) for s in self.statistics]
        dupes = sorted(
            {
                f"{k[0].value}{'' if k[1] is None else f'@p={k[1]}'}"
                for k in keys
                if keys.count(k) > 1
            }
        )
        if dupes:
            raise ValueError(f"observed_distribution repeats statistic(s): {dupes}")

        quants = sorted(
            ((s.p, s.value) for s in self.statistics if s.stat == StatKind.QUANTILE),
            key=lambda t: t[0],
        )
        for (p_lo, v_lo), (p_hi, v_hi) in zip(quants[:-1], quants[1:]):
            if v_hi < v_lo:
                raise ValueError(
                    "quantiles must be non-decreasing in p: "
                    f"value {v_hi} at p={p_hi} is below value {v_lo} at p={p_lo}."
                )

        if self.spread_source in POPULATION_SPREAD_SOURCES and not self._has_width():
            raise ValueError(
                f"spread_source='{self.spread_source.value}' declares a population spread but "
                "no width statistic was reported. Provide quartiles, an sd/iqr/cv/range, or "
                "declare a center-only spread_source."
            )
        return self

    def _has_width(self) -> bool:
        """Whether any entry carries a width, explicit or recoverable from one.

        An SE counts: it is the sample width divided by sqrt(n), so it determines
        one given the n it was computed over. Recovering it needs that n, which
        the derivation accessors take as an argument rather than assume.
        """
        kinds = {s.stat for s in self.statistics}
        if kinds & (WIDTH_STATS | SAMPLING_WIDTH_STATS):
            return True
        if len({s.p for s in self.statistics if s.stat == StatKind.QUANTILE}) >= 2:
            return True
        # A min/max pair is the sample range stated as its two endpoints, which is
        # what a source prints when it gives an observed span rather than a width.
        return {StatKind.MIN, StatKind.MAX} <= kinds

    # ---- Accessors --------------------------------------------------------

    def get(self, stat: StatKind, p: Optional[float] = None) -> Optional[float]:
        """The reported value for ``stat`` (at ``p`` for a quantile), or None."""
        for s in self.statistics:
            if s.stat == stat and s.p == p:
                return s.value
        return None

    def center(self) -> Optional[float]:
        """The reported center: the median if given, else the mean."""
        med = self.get(StatKind.QUANTILE, 0.5)
        return med if med is not None else self.get(StatKind.MEAN)

    # ---- Derivations ------------------------------------------------------

    def _anchor_pairs(self, n: Optional[int] = None) -> List[tuple]:
        """Effective (p, value) quantile anchors, p-sorted.

        Explicit quantiles are used as reported. With fewer than two of them, a
        center and a scale are expanded through ``shape``. ``n`` is the count the
        statistics were computed over, needed only to widen an SE into a sample SD.
        """
        quants = sorted(
            ((s.p, s.value) for s in self.statistics if s.stat == StatKind.QUANTILE),
            key=lambda t: t[0],
        )
        if len(quants) >= 2:
            return quants

        sd = self._normal_equivalent_sd(n)
        center = self.center()
        if sd is None or center is None:
            missing = "no center" if center is None else "no width"
            if center is not None and n is None and self.get(StatKind.SE) is not None:
                missing = (
                    "only an SE, which is a width per sqrt(n); pass n (the cohort's n_c) "
                    "to widen it into a sample SD"
                )
            raise ValueError(
                "observed_distribution cannot produce quantiles: it reports "
                f"{[s.stat.value for s in self.statistics]}, which gives {missing}."
            )
        if self.shape is None:
            raise ValueError(
                "observed_distribution needs `shape` to expand a center and scale into "
                "quantiles, since the source reported fewer than two quantiles."
            )
        center_is_mean = self.get(StatKind.QUANTILE, 0.5) is None
        q25, q50, q75 = _quartiles_from_center_scale(center, center_is_mean, sd, self.shape)
        return [(0.25, q25), (0.5, q50), (0.75, q75)]

    def _normal_equivalent_sd(self, n: Optional[int] = None) -> Optional[float]:
        """A linear SD implied by whichever width statistic was reported.

        An SE is used last and only with ``n``: it is the sample SD over sqrt(n),
        so without n it is not a sample width at all. At n=10 the two differ by a
        factor of 3.2, which is why this never guesses.
        """
        sd = self.get(StatKind.SD)
        if sd is not None:
            return sd
        iqr = self.get(StatKind.IQR)
        if iqr is not None:
            return iqr / (2.0 * _Z_Q)
        cv = self.get(StatKind.CV)
        if cv is not None:
            center = self.center()
            return cv * abs(center) if center is not None else None
        lo, hi = self.get(StatKind.CI95_LO), self.get(StatKind.CI95_HI)
        if lo is not None and hi is not None:
            return (hi - lo) / (2.0 * _Z_95)
        se = self.get(StatKind.SE)
        if se is not None and n is not None:
            if n < 1:
                raise ValueError(f"widening an SE needs a positive n, got {n}.")
            return se * math.sqrt(n)
        return None

    def population_sd(self, n: Optional[int] = None) -> Optional[float]:
        """The sample SD the statistics imply, or None if none is recoverable.

        ``n`` is the count the statistics were computed over: a calibration
        target's cohort ``n_c``, or a submodel target's ``n_biological``. It is
        read only when the reported width is an SE.
        """
        return self._normal_equivalent_sd(n)

    def quantile(self, p: float, n: Optional[int] = None) -> float:
        """Value at ``p``: reported if the source printed it, else interpolated."""
        reported = self.get(StatKind.QUANTILE, p)
        if reported is not None:
            return reported
        anchors = self._anchor_pairs(n)
        if p <= anchors[0][0]:
            return anchors[0][1]
        if p >= anchors[-1][0]:
            return anchors[-1][1]
        for (p_lo, v_lo), (p_hi, v_hi) in zip(anchors[:-1], anchors[1:]):
            if p_lo <= p <= p_hi:
                if p_hi == p_lo:
                    return v_lo
                return v_lo + (p - p_lo) / (p_hi - p_lo) * (v_hi - v_lo)
        return anchors[-1][1]  # pragma: no cover

    def median(self, n: Optional[int] = None) -> float:
        """Value at p=0.5, reported if given and interpolated otherwise."""
        return self.quantile(0.5, n)

    def iqr(self, n: Optional[int] = None) -> Optional[float]:
        """Interquartile range, reported if given, else from the quantile anchors."""
        reported = self.get(StatKind.IQR)
        if reported is not None:
            return reported
        ps = [p for p, _ in self._anchor_pairs(n)]
        if min(ps) > 0.25 or max(ps) < 0.75:
            return None
        return self.quantile(0.75, n) - self.quantile(0.25, n)

    @property
    def feeds_population_spread(self) -> bool:
        """Whether these statistics contribute a base population-spread magnitude (omega)."""
        return self.spread_source in POPULATION_SPREAD_SOURCES

    def require_unit_provenance(self, label: str = "observed_distribution") -> None:
        """Raise unless a population-spread claim states its unit count and unit type.

        Called by owners that carry unit accounting on the distribution itself
        (submodel error models). Calibration targets name a cohort instead.
        """
        if self.spread_source not in POPULATION_SPREAD_SOURCES:
            return
        if self.experimental_unit_type in (
            ExperimentalUnitType.TECHNICAL,
            ExperimentalUnitType.CLONAL,
        ):
            raise ValueError(
                f"{label}: experimental_unit_type='{self.experimental_unit_type.value}' cannot "
                f"support spread_source='{self.spread_source.value}'. A spread over "
                "technical/clonal replicates is not population variability."
            )
        if self.n_biological is None:
            raise ValueError(
                f"{label}: spread_source='{self.spread_source.value}' declares a population "
                "spread but n_biological is not set."
            )
        if self.experimental_unit_type is None:
            raise ValueError(
                f"{label}: spread_source='{self.spread_source.value}' declares a population "
                "spread but experimental_unit_type is not set."
            )


class TableExcerpt(BaseModel):
    """
    Structured excerpt from a table in a paper.

    Use this instead of value_snippet when the value comes from a table,
    where PDF text extraction produces unreadable concatenated rows.
    External validators check that table_id, column, row, and value all
    appear somewhere in the extracted paper text.
    """

    model_config = ConfigDict(extra="forbid")

    table_id: str = Field(
        description="Table identifier (e.g., 'Table 2', 'Supplementary Table S1')"
    )
    column: str = Field(description="Column header the value falls under")
    row: str = Field(description="Row label/identifier")
    value: str = Field(
        description="Value as it appears in the table cell (e.g., '29 ± 10'). "
        "Used for validation against extracted paper text."
    )
    context: str = Field(
        description="Additional context (e.g., units in column header, table caption, "
        "or surrounding text that clarifies the value)",
    )


class FigureExcerpt(BaseModel):
    """
    Structured excerpt from a figure in a paper.

    Use this instead of value_snippet when the value is read from a figure
    (e.g., scatter plots, bar charts, dose-response curves). Figure-derived
    values cannot be validated by text matching, so inputs with figure_excerpt
    are flagged for manual review instead of failing snippet validation.
    """

    model_config = ConfigDict(extra="forbid")

    figure_id: str = Field(
        description="Figure identifier (e.g., 'Figure 1C', 'Supplementary Figure S2A')"
    )
    value: str = Field(
        description="Value as read from the figure (e.g., '~5', '2-5 range'). "
        "Used for documentation; not validated by text matching."
    )
    description: str = Field(
        description="What was read from the figure (e.g., 'highest data point in scatter plot at 16h')"
    )
    context: str = Field(
        description="Additional context (e.g., figure caption text, axis labels, "
        "or experimental conditions shown in the panel)",
    )


class EstimateInput(BaseModel):
    """
    A literature-extracted input value for distribution derivation.

    Used in CalibrationTargetEstimates.inputs for values that feed into
    distribution_code to derive calibration target estimates. All provenance
    fields are REQUIRED to ensure full traceability.

    Supports both scalar and vector-valued inputs:
    - Scalar: value=42.0 (single measurement, or constant applied to all index points)
    - Vector: value=[10.0, 20.0, 30.0] (measurements at each index point)

    For experimental conditions used in submodel/observable code, use SubmodelInput.
    For modeling assumptions (e.g., n_mc_samples), use ModelingAssumption.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Input name (used as key in inputs dict)")
    value: Union[float, List[float]] = Field(
        description=(
            "Input value(s). Use a list for vector-valued data (e.g., measurements at "
            "multiple time points or doses). Scalar values are broadcast across all index points."
        )
    )
    units: str = Field(
        description="Input units (must be Pint-parseable, e.g., 'pg/mL', 'cell/mm^2', 'dimensionless')"
    )
    description: str = Field(description="What this input represents and how it was extracted")
    source_ref: str = Field(
        description=(
            "Source reference tag. MUST match a source_tag in primary_data_source or "
            "secondary_data_sources."
        )
    )
    value_location: str = Field(
        description="Where the value appears in the source (e.g., 'Table 2', 'Figure 3A', 'Results p.5')"
    )
    value_snippet: Optional[str] = Field(
        default=None,
        description="Exact text snippet from the source containing or supporting the value(s). "
        "Use table_excerpt when the value comes from a table, or figure_excerpt when the value "
        "is digitized from a figure. At least one of value_snippet, table_excerpt, or "
        "figure_excerpt must be provided.",
    )
    table_excerpt: Optional[TableExcerpt] = Field(
        default=None,
        description="Structured table excerpt when value comes from a table. "
        "Preferred over value_snippet for table-sourced data because PDF text extraction "
        "often produces unreadable concatenated rows.",
    )
    figure_excerpt: Optional[FigureExcerpt] = Field(
        default=None,
        description="Structured figure excerpt when value is digitized from a figure. "
        "Figure-derived values are flagged for manual review instead of failing snippet "
        "validation (since figure pixels are not in the PDF text layer).",
    )

    input_type: InputType = Field(
        default=InputType.DIRECT_PARAMETER,
        description=(
            "Classification of this input:\n"
            "- direct_parameter: Literature reports the value directly (e.g., 'mean = 42.0')\n"
            "- proxy_measurement: Requires conversion (e.g., 'doubling time = 8h' → rate constant)\n"
            "- experimental_condition: Protocol choice from paper (e.g., seeding density, E:T ratio)\n"
            "- inferred_estimate: Value interpreted from qualitative text (e.g., 'maintained viability' → 0.95).\n"
            "  Use when numeric value doesn't appear literally but is a reasonable interpretation.\n"
            "  Snippet validation is skipped for this type."
        ),
    )

    # Derived-arithmetic fields
    formula: Optional[str] = Field(
        None,
        description=(
            "Arithmetic formula deriving this value from source_inputs. "
            "Required for derived_arithmetic inputs. Use input names directly in the "
            "expression (e.g., 'post_01 / pre_01', '3 * Gprime_stiff_kPa'). "
            "The validator evaluates this against the source_inputs values and checks "
            "it matches the declared value within 1%."
        ),
    )
    source_inputs: Optional[List[str]] = Field(
        None,
        description=(
            "Names of other inputs used in the formula. Required for derived_arithmetic "
            "inputs. All referenced names must exist as other inputs in the same target."
        ),
    )

    conversion_formula: Optional[str] = Field(
        None,
        description=(
            "For proxy_measurement type: formula showing how to convert to model parameter.\n\n"
            "Common conversions by parameter type:\n"
            "Rate constants (1/time):\n"
            "- Doubling time → k = ln(2) / t_double\n"
            "- Half-life → k = ln(2) / t_half\n"
            "- Fold change → k = ln(fold) / time\n"
            "- Mean residence time → k = 1 / MRT\n\n"
            "Binding parameters:\n"
            "- Ka (association) → Kd = 1 / Ka\n"
            "- koff from Kd, kon → koff = Kd * kon\n\n"
            "PK parameters:\n"
            "- CL + Vd → kel = CL / Vd\n"
            "- AUC → CL = Dose * F / AUC\n\n"
            "Production rates:\n"
            "- Steady-state → k_prod = k_decay * C_ss\n\n"
            "Not required for direct_parameter or experimental_condition types."
        ),
    )

    # Figure extraction fields
    source_type: SourceType = Field(
        default=SourceType.TEXT,
        description=(
            "Type of source from which the value was extracted:\n"
            "- text: Body text, results section, or abstract (default)\n"
            "- table: Table\n"
            "- figure: Figure (requires figure_id and extraction_method)"
        ),
    )

    figure_id: Optional[str] = Field(
        None,
        description=(
            "Figure identifier (e.g., 'Figure 2A', 'Fig. 3B'). "
            "Required when source_type='figure'."
        ),
    )

    extraction_method: Optional[ExtractionMethod] = Field(
        None,
        description=(
            "Method used to extract value from figure. Required when source_type='figure'.\n"
            "- manual: Manual reading from figure axes\n"
            "- digitizer: Generic digitizer software\n"
            "- webplotdigitizer: WebPlotDigitizer tool\n"
            "- other: Other method (specify in extraction_notes)"
        ),
    )

    extraction_notes: Optional[str] = Field(
        None,
        description=(
            "Additional context for figure extraction.\n"
            "Example: 'Read from y-axis at day 14 timepoint'\n"
            "Example: 'Digitized all points from survival curve'"
        ),
    )

    # Dispersion identification fields
    dispersion_type: Optional[UncertaintyType] = Field(
        None,
        description=(
            "Type of dispersion measure this input represents. REQUIRED when this input is a "
            "measure of spread/uncertainty (SD, SEM, CI bound, IQR bound, range).\n\n"
            "CRITICAL: Papers often report 'mean ± X' without specifying whether X is SD or SEM. "
            "Misidentifying SEM as SD underestimates uncertainty by a factor of sqrt(n), producing "
            "calibration targets that are far too constraining.\n\n"
            "How to distinguish SD vs SEM:\n"
            "1. Check methods section for explicit statement ('values are mean ± SD/SEM')\n"
            "2. Check table/figure legends for the error type\n"
            "3. Use sqrt(n) test: if '±X' scales as 1/sqrt(n) across subgroups with similar "
            "biology, it's SEM (SD = SEM × sqrt(n))\n"
            "4. Biological plausibility: SD should give CV typical for the measurement type "
            "(immune cell densities: CV 50-200%; tumor volumes: CV 50-100%). "
            "If treating ± as SD gives CV < 20%, it's almost certainly SEM.\n\n"
            "Set to None for inputs that are NOT dispersion measures (e.g., sample sizes, "
            "medians, means, quartile values)."
        ),
    )
    dispersion_type_rationale: Optional[str] = Field(
        None,
        description=(
            "REQUIRED when dispersion_type is set. Explain how you determined the dispersion type.\n\n"
            "Examples:\n"
            "- 'Methods section states: all values presented as mean ± standard deviation'\n"
            "- 'Table legend specifies SEM. SD = SEM × sqrt(n) = 15.0 × sqrt(368) = 287.7'\n"
            "- 'Paper does not specify. ±15.0 with n=368 gives CV=6.6% if SD (implausibly low) "
            "vs CV=126% if SEM (typical for immune cell densities). Treating as SEM.'\n"
            "- 'IQR explicitly stated in table header as median (Q1-Q3)'"
        ),
    )

    @field_validator("value")
    @classmethod
    def ensure_list_not_empty(cls, v: Union[float, List[float]]) -> Union[float, List[float]]:
        """Ensure vector-valued inputs are not empty lists."""
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Vector-valued input cannot be an empty list")
        return v

    @model_validator(mode="after")
    def require_one_snippet_form(self) -> "EstimateInput":
        """At least one of value_snippet, table_excerpt, or figure_excerpt must be provided.

        Skipped for derived_arithmetic inputs, which carry their provenance in the
        ``formula`` + ``source_inputs`` fields rather than a literal snippet/excerpt.
        """
        if self.input_type == InputType.DERIVED_ARITHMETIC:
            return self
        if not any([self.value_snippet, self.table_excerpt, self.figure_excerpt]):
            raise ValueError(
                f"Input '{self.name}': at least one of value_snippet, table_excerpt, or "
                f"figure_excerpt must be provided for traceability."
            )
        return self

    @field_validator("conversion_formula")
    @classmethod
    def require_conversion_for_proxy(cls, v: Optional[str], info) -> Optional[str]:
        """Warn if proxy_measurement type lacks conversion_formula."""
        import warnings

        # Access input_type from the data being validated
        input_type = info.data.get("input_type", InputType.DIRECT_PARAMETER)

        if input_type == InputType.PROXY_MEASUREMENT and v is None:
            warnings.warn(
                "EstimateInput with input_type='proxy_measurement' should have conversion_formula "
                "documenting how to convert to model parameter (e.g., 'k = ln(2) / t_half').",
                UserWarning,
            )
        return v

    @model_validator(mode="after")
    def validate_figure_fields(self) -> "EstimateInput":
        """Ensure figure sources have required figure_id and extraction_method."""
        if self.source_type == SourceType.FIGURE:
            missing = []
            if not self.figure_id:
                missing.append("figure_id")
            if not self.extraction_method:
                missing.append("extraction_method")
            if missing:
                raise ValueError(
                    f"When source_type='figure', the following fields are required: {', '.join(missing)}"
                )
        return self

    @model_validator(mode="after")
    def validate_dispersion_type_fields(self) -> "EstimateInput":
        """Require dispersion_type on dispersion-like inputs; require rationale when type is set."""
        # If dispersion_type is set, rationale is required
        if self.dispersion_type is not None and not self.dispersion_type_rationale:
            raise ValueError(
                f"Input '{self.name}': dispersion_type_rationale is required when "
                f"dispersion_type is set ('{self.dispersion_type.value}'). "
                f"Explain how you determined this is {self.dispersion_type.value.upper()} "
                f"(e.g., paper states explicitly, inferred from sqrt(n) scaling, "
                f"biological plausibility of implied CV)."
            )

        # If input name looks like a dispersion measure, dispersion_type is required
        if self.dispersion_type is None:
            _DISPERSION_KEYWORDS = [
                "sd",
                "std",
                "stdev",
                "se",
                "sem",
                "stderr",
                "sigma",
                "dispersion",
                "error_bar",
            ]
            name_lower = self.name.lower()
            # Check for keyword matches (as whole tokens separated by _ or at boundaries)
            name_tokens = set(name_lower.replace("-", "_").split("_"))
            if name_tokens & set(_DISPERSION_KEYWORDS):
                raise ValueError(
                    f"Input '{self.name}' appears to be a dispersion measure but "
                    f"dispersion_type is not set. Set dispersion_type to 'sd', 'se', "
                    f"'ci95', 'iqr', or 'range' and provide dispersion_type_rationale "
                    f"explaining how you determined the type.\n\n"
                    f"CRITICAL: Misidentifying SEM as SD underestimates uncertainty by "
                    f"sqrt(n). Use the sqrt(n) test or check biological plausibility of "
                    f"the implied CV to distinguish them."
                )

        return self


class ModelingAssumption(BaseModel):
    """
    An assumed value for computation that is not from literature.

    Used in CalibrationTargetEstimates.assumptions for values like:
    - n_mc_samples: Number of Monte Carlo samples for bootstrap
    - assumed_cv: Assumed coefficient of variation when not reported
    - scaling_factor: Scaling factor for unit conversion

    Unlike LiteratureInput, this requires a rationale field instead of
    value_location and value_snippet.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Assumption name (used as key in inputs dict)")
    value: Union[float, List[float]] = Field(
        description="Assumed value(s). Scalar for single values, list for per-index-point values."
    )
    units: str = Field(description="Units (must be Pint-parseable, e.g., 'dimensionless', '1/day')")
    description: str = Field(description="What this assumption represents")
    rationale: str = Field(
        description=(
            "Why this value was chosen. Required for all assumptions.\n"
            "Example: 'Standard sample size for stable percentile estimates'\n"
            "Example: 'Typical CV for biological measurements when not reported'"
        )
    )

    @field_validator("value")
    @classmethod
    def ensure_list_not_empty(cls, v: Union[float, List[float]]) -> Union[float, List[float]]:
        """Ensure vector-valued assumptions are not empty lists."""
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Vector-valued assumption cannot be an empty list")
        return v


class SubmodelInput(BaseModel):
    """
    An experimental input used in submodel or observable code.

    Used in Submodel.inputs and Observable.inputs for values that appear directly
    in ODE or observable calculations. Unlike EstimateInput (which feeds
    distribution_code), these are inputs to the model simulation itself.

    Examples:
    - E:T ratio for killing assay
    - Drug concentration for dose-response
    - Culture duration for growth experiments

    All provenance fields are REQUIRED for traceability.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Input name (used as key in inputs dict)")
    value: Union[float, List[float]] = Field(
        description=(
            "Input value(s). Use a list for vector-valued data (e.g., measurements at "
            "multiple time points or doses). Scalar values are broadcast across all index points."
        )
    )
    units: str = Field(
        description="Input units (must be Pint-parseable, e.g., 'pg/mL', 'cell/mm^2', 'dimensionless')"
    )
    description: str = Field(description="What this input represents and how it was extracted")
    source_ref: str = Field(
        description=(
            "Source reference tag. MUST match a source_tag in primary_data_source or "
            "secondary_data_sources."
        )
    )
    value_location: str = Field(
        description="Where the value appears in the source (e.g., 'Table 2', 'Figure 3A', 'Results p.5')"
    )
    value_snippet: str = Field(
        description="Exact text snippet from the source containing or supporting the value(s)"
    )

    # Figure extraction fields
    source_type: SourceType = Field(
        default=SourceType.TEXT,
        description=(
            "Type of source from which the value was extracted:\n"
            "- text: Body text, results section, or abstract (default)\n"
            "- table: Table\n"
            "- figure: Figure (requires figure_id and extraction_method)"
        ),
    )

    figure_id: Optional[str] = Field(
        None,
        description=(
            "Figure identifier (e.g., 'Figure 2A', 'Fig. 3B'). "
            "Required when source_type='figure'."
        ),
    )

    extraction_method: Optional[ExtractionMethod] = Field(
        None,
        description=(
            "Method used to extract value from figure. Required when source_type='figure'.\n"
            "- manual: Manual reading from figure axes\n"
            "- digitizer: Generic digitizer software\n"
            "- webplotdigitizer: WebPlotDigitizer tool\n"
            "- other: Other method (specify in extraction_notes)"
        ),
    )

    extraction_notes: Optional[str] = Field(
        None,
        description=(
            "Additional context for figure extraction.\n"
            "Example: 'Read from y-axis at day 14 timepoint'\n"
            "Example: 'Digitized all points from survival curve'"
        ),
    )

    @field_validator("value")
    @classmethod
    def ensure_list_not_empty(cls, v: Union[float, List[float]]) -> Union[float, List[float]]:
        """Ensure vector-valued inputs are not empty lists."""
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Vector-valued input cannot be an empty list")
        return v

    @model_validator(mode="after")
    def validate_figure_fields(self) -> "SubmodelInput":
        """Ensure figure sources have required figure_id and extraction_method."""
        if self.source_type == SourceType.FIGURE:
            missing = []
            if not self.figure_id:
                missing.append("figure_id")
            if not self.extraction_method:
                missing.append("extraction_method")
            if missing:
                raise ValueError(
                    f"When source_type='figure', the following fields are required: {', '.join(missing)}"
                )
        return self


class WeightScore(BaseModel):
    """A rubric-based weight score with justification."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(description="Rubric value (0-1)")
    justification: str = Field(description="Justification for this value")


class Source(BaseModel):
    """A bibliographic source (primary data)."""

    model_config = ConfigDict(extra="forbid")

    source_tag: str = Field(description="Unique tag for referencing")
    title: str = Field(description="Full title")
    first_author: str = Field(description="First author last name")
    year: int = Field(description="Publication year")
    doi: Optional[str] = Field(None, description="DOI (or null)")
    source_relevance: "SourceRelevanceAssessment" = Field(
        description=(
            "Structured assessment of how well this source's data translates to the target model. "
            "Captures indication match, source quality, perturbation context, and TME compatibility."
        ),
    )


class SecondarySource(BaseModel):
    """A secondary data source (reference values, textbooks)."""

    model_config = ConfigDict(extra="forbid")

    source_tag: str = Field(description="Unique tag for referencing")
    title: str = Field(description="Full title")
    first_author: str = Field(description="First author last name")
    year: int = Field(description="Publication year")
    doi_or_url: Optional[str] = Field(None, description="DOI or URL (or null)")
    source_relevance: "SourceRelevanceAssessment" = Field(
        description=(
            "Structured assessment of how well this source's data translates to the target model. "
            "Captures indication match, source quality, perturbation context, and TME compatibility."
        ),
    )


# ============================================================================
# In Vitro / Experimental System Models
# ============================================================================


class CellSpecies(str, Enum):
    """Species origin of cell line."""

    HUMAN = "human"
    MOUSE = "mouse"
    RAT = "rat"
    OTHER = "other"


class CellLine(BaseModel):
    """Specification of a cell line used in an experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Cell line name (e.g., 'Jurkat', 'HeLa', 'MCF-7', 'primary CD8 T cells')"
    )
    species: CellSpecies = Field(description="Species origin of cell line")
    cell_type: str = Field(description="Cell type (e.g., 'T cell', 'epithelial', 'fibroblast')")
    additional_info: Optional[str] = Field(
        None, description="Additional info (e.g., 'immortalized', 'primary', 'GFP-expressing')"
    )


class CultureConditions(BaseModel):
    """Culture conditions for in vitro experimental systems."""

    model_config = ConfigDict(extra="forbid")

    medium: Optional[str] = Field(None, description="Culture medium (e.g., 'RPMI-1640', 'DMEM')")
    duration_hours: Optional[float] = Field(None, description="Culture duration in hours")
    additional: Optional[dict] = Field(
        None,
        description="Additional conditions (serum, supplements, temperature, CO2, etc.)",
    )


# =============================================================================
# SOURCE RELEVANCE ASSESSMENT
# =============================================================================


class SourceRelevanceAssessment(BaseModel):
    """
    Structured assessment of source-to-target relevance for calibration targets.

    This model captures how well the source data translates to the target model,
    including indication match, source quality, perturbation context, and TME
    compatibility. Validators use this information to flag potential issues.
    """

    model_config = ConfigDict(extra="forbid")

    # Indication relevance
    indication_match: IndicationMatch = Field(
        description=(
            "How well does the source indication match the target indication?\n"
            "- exact: Same disease (e.g., PDAC data for PDAC model)\n"
            "- related: Same organ or disease class (e.g., other pancreatic diseases)\n"
            "- proxy: Different tissue used as mechanistic proxy (e.g., melanoma for PDAC)\n"
            "- unrelated: No clear biological connection"
        )
    )
    indication_match_justification: str = Field(
        description=(
            "Justify the indication match rating. For exact matches, a one-line statement "
            "is fine. If PROXY or UNRELATED, explain why this source is acceptable and "
            "what translation uncertainty is expected."
        ),
    )

    # Species
    species_source: str = Field(description="Species in the source study (human, mouse, rat, etc.)")
    species_target: str = Field(
        default="human", description="Target species for the model (usually 'human')"
    )

    # Source quality
    source_quality: SourceQuality = Field(
        description=(
            "Quality tier of the primary data source.\n"
            "IMPORTANT: 'non_peer_reviewed' includes Wikipedia, preprints, and "
            "unreviewed sources. Avoid if possible; if used, document rationale."
        )
    )

    # Perturbation context
    perturbation_type: PerturbationType = Field(
        description=(
            "Type of experimental perturbation in the source study.\n"
            "If 'pharmacological' or 'genetic_perturbation', explain in "
            "perturbation_relevance how this relates to physiological parameter values."
        )
    )
    perturbation_relevance: str = Field(
        description=(
            "Explain relevance of the experimental perturbation to the physiological "
            "parameter being estimated. For physiological_baseline, a brief note is fine. "
            "For pharmacological/genetic perturbations, explain whether this represents "
            "an upper bound, typical value, or requires scaling."
        ),
    )

    # TME compatibility
    tme_compatibility: TMECompatibility = Field(
        description=(
            "Tumor microenvironment compatibility assessment.\n"
            "- high: Source TME similar to target (e.g., both desmoplastic)\n"
            "- moderate: Some TME differences that may affect values\n"
            "- low: Major differences (e.g., T cell-permissive model for T cell-excluded tumor)"
        ),
    )
    tme_compatibility_notes: str = Field(
        description=(
            "Notes on TME differences and their expected impact on the parameter. "
            "Brief is fine when tme_compatibility=high. For low/moderate, "
            "describe the expected directional impact."
        ),
    )

    # Measurement and system characterization (for translation sigma rubric).
    # Optional: required for SubmodelTarget (in vitro / preclinical), omitted for
    # clinical CalibrationTargets since those are always direct patient measurements.
    measurement_directness: Optional[MeasurementDirectness] = Field(
        default=None,
        description=(
            "How many inferential steps between the raw measurement and the model parameter.\n"
            "- direct: trivial transform (unit conversion, ln2/x)\n"
            "- single_inversion: one kinetic/mechanistic model assumption\n"
            "- steady_state_inversion: inferred via balance equations with assumed rates\n"
            "- proxy_observable: surrogate measurement (RNA for protein, IHC score for concentration)\n"
            "Omit (null) for clinical CalibrationTarget sources."
        ),
    )
    temporal_resolution: Optional[TemporalResolution] = Field(
        default=None,
        description=(
            "How well the temporal or dose structure constrains the parameter.\n"
            "- timecourse: >=3 timepoints/doses spanning the dynamic range\n"
            "- endpoint_pair: 2 timepoints or conditions\n"
            "- snapshot_or_equilibrium: single timepoint or assumed steady state\n"
            "Omit (null) for clinical CalibrationTarget sources."
        ),
    )
    experimental_system: Optional[ExperimentalSystem] = Field(
        default=None,
        description=(
            "Biological fidelity of the experimental system. Distinct from source_quality "
            "(evidence reliability). Captures how well the experimental conditions "
            "recapitulate the in vivo tumor biology being modeled.\n"
            "- clinical_in_vivo: direct patient measurements\n"
            "- animal_in_vivo: animal model (intact system, species gap)\n"
            "- ex_vivo: freshly isolated tissue\n"
            "- in_vitro_coculture: multi-cell-type culture\n"
            "- in_vitro_primary: primary cells in standard culture\n"
            "- in_vitro_cell_line: immortalized cell lines\n"
            "Omit (null) for clinical CalibrationTarget sources."
        ),
    )

    # Spread transfer (for hierarchical / virtual-patient inference).
    # The SPREAD analog of the center-transfer fields above: how well the source's
    # biological variability transfers to target-population variability. Like
    # measurement_directness, required for SubmodelTarget (in vitro / preclinical),
    # omitted for clinical CalibrationTargets (which measure the target population's
    # spread directly).
    heterogeneity_transfer: Optional[HeterogeneityTransfer] = Field(
        default=None,
        description=(
            "How well the source's measured biological spread transfers to TARGET-"
            "population (patient-to-patient) spread. Sets translation widening of the "
            "population-spread hyperparameter (omega) on top of the measured biological SD, "
            "which is typically a lower bound on patient spread.\n"
            "- high: source spread spans most target heterogeneity (patient-derived panel, "
            "across-patient cohort) -> little widening\n"
            "- moderate: partial (healthy human donors, a few cell lines, outbred animals)\n"
            "- low: homogeneous system (single cell line, inbred strain, technical "
            "replicates) -> large widening or use a mechanistic-default spread\n"
            "Distinct from the center-transfer grades: a source can pin the mean well yet "
            "transfer heterogeneity poorly. Omit (null) for clinical CalibrationTarget "
            "sources, which measure the target-population spread directly."
        ),
    )
    heterogeneity_transfer_justification: Optional[str] = Field(
        default=None,
        description=(
            "Justify the heterogeneity_transfer grade: what units the source's spread is "
            "across, and which target-population heterogeneity axes (disease stage, TME, "
            "genetics, treatment) it does or does not capture. Required when "
            "heterogeneity_transfer is set."
        ),
    )

    # Computed (machine-populated, not user-facing)
    validation_warnings: Optional[List[str]] = Field(
        default=None,
        description="Validation warnings generated by automated checks (populated by validators)",
    )

    @model_validator(mode="after")
    def _require_heterogeneity_transfer_justification(self) -> "SourceRelevanceAssessment":
        """Require a justification whenever the heterogeneity_transfer grade is set."""
        if (
            self.heterogeneity_transfer is not None
            and not self.heterogeneity_transfer_justification
        ):
            raise ValueError(
                "heterogeneity_transfer_justification is required when "
                f"heterogeneity_transfer is set ('{self.heterogeneity_transfer.value}'). "
                "State what units the source's spread is across and which target-population "
                "heterogeneity axes it does/does not capture."
            )
        return self


# Rebuild models that use forward references to SourceRelevanceAssessment
Source.model_rebuild()
SecondarySource.model_rebuild()
