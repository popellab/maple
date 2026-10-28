"""Closed measurement-context vocabulary carried by literature readouts."""

import pytest
from pydantic import ValidationError

from maple.core.calibration import MeasurementContext, Readout


CONTEXT = {
    "acquisition_method": "resection",
    "specimen_compartment": "primary_tumor",
    "specimen_material": "tissue_section",
    "preservation": "ffpe",
    "sampling_unit": "region_of_interest",
    "sampling_region": "non_lymphoid_tumor",
    "detection_basis": "marker_defined_cells",
    "analyte_pool": "cellular",
    "normalization_mode": "cell_fraction_of_all_nucleated_cells",
}


def test_complete_context_parses_as_closed_enums():
    context = MeasurementContext(**CONTEXT)
    assert context.specimen_material.value == "tissue_section"
    assert context.preservation.value == "ffpe"


@pytest.mark.parametrize("field", tuple(CONTEXT))
def test_every_context_field_is_required(field):
    data = dict(CONTEXT)
    data.pop(field)
    with pytest.raises(ValidationError, match=field):
        MeasurementContext(**data)


def test_context_rejects_unknown_enum_level():
    with pytest.raises(ValidationError, match="specimen_material"):
        MeasurementContext(**{**CONTEXT, "specimen_material": "paraffin_blob"})


def test_context_rejects_free_form_extra_fields():
    with pytest.raises(ValidationError, match="free_text_protocol|Extra inputs"):
        MeasurementContext(**CONTEXT, free_text_protocol="whatever the paper said")


def test_readout_carries_context_without_assigning_model_roles():
    readout = Readout(
        quantity_kind="fraction",
        assay_modality="mihc",
        measurement_context=CONTEXT,
        numerator_species=["V_T.CD8"],
        denominator_species=["V_T.nucleated"],
    )
    assert readout.measurement_context.normalization_mode.value == (
        "cell_fraction_of_all_nucleated_cells"
    )


def test_bare_readout_remains_available_to_mechanistic_compositions():
    readout = Readout(
        quantity_kind="density",
        assay_modality="mihc",
        numerator_species=["V_T.CD8"],
    )
    assert readout.measurement_context is None
