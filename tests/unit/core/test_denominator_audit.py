"""Cross-target denominator checks: mapping collisions, and code against declaration."""

import warnings

from maple.core.calibration.denominator_audit import (
    find_code_readout_mismatches,
    find_mapping_collisions,
    numerator_and_denominator,
    warn_code_readout_mismatches,
)

_FRACTION = (
    "def compute_observable(time, species_dict, constants):\n"
    "    return species_dict['V_T.CD8'] / species_dict['V_T.nucleated']\n"
)


def _target(numerator=("V_T.CD8",), denominator=("V_T.nucleated",), code=_FRACTION, **over):
    observable = {
        "code": code,
        "readout_time": 0.0,
        "readout": {
            "quantity_kind": "fraction",
            "assay_modality": "mihc",
            "numerator_species": list(numerator),
            "denominator_species": list(denominator),
        },
    }
    observable.update(over)
    return {"observable": observable}


class TestMappingCollisions:
    def test_same_declared_composition_collides(self):
        collisions = find_mapping_collisions({"a": _target(), "b": _target()})
        assert len(collisions) == 1
        assert collisions[0].members == ("a", "b")

    def test_different_numerator_does_not_collide(self):
        assert find_mapping_collisions({"a": _target(), "b": _target(numerator=["V_T.CD4"])}) == []

    def test_absolute_quantity_is_skipped(self):
        """No denominator, no shared model quantity to collide over."""
        assert (
            find_mapping_collisions({"a": _target(denominator=[]), "b": _target(denominator=[])})
            == []
        )

    def test_collision_keyed_on_declaration_not_code(self):
        """Two spellings of one row still collide."""
        other_spelling = (
            "def compute_observable(time, species_dict, constants):\n"
            "    total = species_dict['V_T.nucleated']\n"
            "    return species_dict['V_T.CD8'] / total\n"
        )
        collisions = find_mapping_collisions({"a": _target(), "b": _target(code=other_spelling)})
        assert len(collisions) == 1


class TestCodeReadoutAgreement:
    def test_agreement_is_silent(self):
        assert find_code_readout_mismatches({"a": _target()}) == []

    def test_disagreement_reported(self):
        aggregated = (
            "def compute_observable(time, species_dict, constants):\n"
            "    return species_dict['V_T.CD8'] / species_dict['CD8_total_T']\n"
        )
        found = find_code_readout_mismatches({"a": _target(code=aggregated)})
        assert len(found) == 1
        assert found[0].declared == ("V_T.nucleated",)
        assert found[0].in_code == ("CD8_total_T",)

    def test_non_dividing_code_is_not_compared(self):
        """An observable can reach its denominator without a division."""
        code = (
            "def compute_observable(time, species_dict, constants):\n"
            "    return species_dict['V_T.CD8'] * constants['area']\n"
        )
        assert find_code_readout_mismatches({"a": _target(code=code)}) == []

    def test_undeclared_denominator_is_not_compared(self):
        assert find_code_readout_mismatches({"a": _target(denominator=[])}) == []

    def test_mismatch_warns(self):
        aggregated = _FRACTION.replace("V_T.nucleated", "CD8_total_T")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = warn_code_readout_mismatches({"a": _target(code=aggregated)})
        assert len(found) == 1
        assert "divides by" in str(caught[0].message)


def test_parser_still_reads_a_division():
    num, den = numerator_and_denominator(_FRACTION)
    assert (num, den) == ({"V_T.CD8"}, {"V_T.nucleated"})
