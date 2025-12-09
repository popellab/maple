"""
Tests for the shared Pint UnitRegistry.
"""

import numpy as np
import pint

from qsp_llm_workflows.core.unit_registry import create_unit_registry, ureg


class TestUnitRegistry:
    """Test the shared unit registry factory and instance."""

    def test_create_unit_registry_returns_registry(self):
        """Test that create_unit_registry returns a Pint UnitRegistry."""
        registry = create_unit_registry()
        assert isinstance(registry, pint.UnitRegistry)

    def test_shared_ureg_is_registry(self):
        """Test that the shared ureg instance is a UnitRegistry."""
        assert isinstance(ureg, pint.UnitRegistry)

    def test_cell_unit_defined(self):
        """Test that cell unit is defined."""
        registry = create_unit_registry()
        cell_count = 1000 * registry.cell
        assert cell_count.magnitude == 1000
        assert str(cell_count.units) == "cell"

    def test_cells_alias(self):
        """Test that cells is an alias for cell."""
        registry = create_unit_registry()
        cells = 500 * registry.cells
        cell = 500 * registry.cell
        assert cells == cell

    def test_nanomolarity_alias(self):
        """Test that nanomolarity is an alias for nanomolar."""
        registry = create_unit_registry()
        conc_molarity = 50 * registry.nanomolarity
        conc_molar = 50 * registry.nanomolar
        assert conc_molarity == conc_molar

    def test_micromolarity_alias(self):
        """Test that micromolarity is an alias for micromolar."""
        registry = create_unit_registry()
        conc = 1.5 * registry.micromolarity
        assert conc.dimensionality == (1 * registry.micromolar).dimensionality

    def test_millimolarity_alias(self):
        """Test that millimolarity is an alias for millimolar."""
        registry = create_unit_registry()
        conc = 2.0 * registry.millimolarity
        assert conc.dimensionality == (1 * registry.millimolar).dimensionality

    def test_molarity_alias(self):
        """Test that molarity is an alias for molar."""
        registry = create_unit_registry()
        conc = 0.1 * registry.molarity
        assert conc.dimensionality == (1 * registry.molar).dimensionality

    def test_unit_conversion_nanomolar_to_molar(self):
        """Test converting between nanomolar and molar."""
        registry = create_unit_registry()
        conc_nm = 1000 * registry.nanomolar
        conc_um = conc_nm.to(registry.micromolar)
        assert abs(conc_um.magnitude - 1.0) < 1e-10

    def test_dimensionless_ratio(self):
        """Test that dividing same units gives dimensionless."""
        registry = create_unit_registry()
        tregs = 100 * registry.cell
        cd8 = 50 * registry.cell
        ratio = tregs / cd8
        assert ratio.dimensionless
        assert ratio.magnitude == 2.0

    def test_rate_constant_units(self):
        """Test rate constant units (1/day, 1/hour)."""
        registry = create_unit_registry()
        k_day = 0.5 / registry.day
        k_hour = k_day.to(1 / registry.hour)
        assert abs(k_hour.magnitude - 0.5 / 24) < 1e-10

    def test_numpy_operations_preserve_units(self):
        """Test that numpy operations preserve Pint units."""
        registry = create_unit_registry()
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * registry.nanomolar

        median = np.median(values)
        assert isinstance(median, pint.Quantity)
        assert median.magnitude == 3.0
        assert str(median.units) == "nanomolar"

    def test_numpy_percentile_preserves_units(self):
        """Test that np.percentile preserves Pint units."""
        registry = create_unit_registry()
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) / registry.day

        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)

        assert isinstance(p25, pint.Quantity)
        assert isinstance(p75, pint.Quantity)
        assert p25.dimensionality == p75.dimensionality

    def test_each_call_creates_fresh_registry(self):
        """Test that each call to create_unit_registry creates a fresh instance."""
        reg1 = create_unit_registry()
        reg2 = create_unit_registry()
        assert reg1 is not reg2
