#!/usr/bin/env python3
"""
Shared Pint UnitRegistry for QSP workflows.

This module provides a single, consistent UnitRegistry with custom QSP units
defined. Use this registry throughout the codebase to ensure unit consistency.

Usage:
    from qsp_llm_workflows.core.unit_registry import ureg, create_unit_registry

    # Use the shared registry (recommended for most cases)
    concentration = 50 * ureg.nanomolar
    cell_count = 1000 * ureg.cell

    # Or create a fresh registry if needed (e.g., for isolated tests)
    fresh_ureg = create_unit_registry()
"""

import pint


def create_unit_registry() -> pint.UnitRegistry:
    """
    Create a Pint UnitRegistry with custom QSP units.

    Custom units defined:
    - cell/cells: Cell counts (custom dimension [cell_count])
    - nanomolarity: Alias for nanomolar (SimBiology convention)
    - micromolarity: Alias for micromolar
    - millimolarity: Alias for millimolar
    - molarity: Alias for molar

    Returns:
        Configured Pint UnitRegistry
    """
    ureg = pint.UnitRegistry()

    # Cell counts - custom dimension
    ureg.define("cells = [cell_count]")
    ureg.define("cell = cells")

    # SimBiology uses "*molarity" but Pint uses "*molar"
    ureg.define("nanomolarity = nanomolar")
    ureg.define("micromolarity = micromolar")
    ureg.define("millimolarity = millimolar")
    ureg.define("molarity = molar")

    return ureg


# Shared registry instance for use across the codebase
ureg = create_unit_registry()
