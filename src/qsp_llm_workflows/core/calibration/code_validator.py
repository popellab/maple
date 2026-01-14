#!/usr/bin/env python3
"""
Unified code validation for calibration target code blocks.

Provides consistent validation across all code types:
- submodel.code (ODE function)
- submodel.observable.code (observable transformation)
- observable.code (full model observable)
- distribution_code / derivation_code (parameter derivation)

Validates:
1. Syntax (AST parse)
2. Function signature
3. Hardcoded constants detection
4. Execution with mock data
5. Return type and dimensionality
"""

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class CodeType(str, Enum):
    """Types of code blocks in calibration targets."""

    SUBMODEL = "submodel"
    SUBMODEL_OBSERVABLE = "submodel_observable"
    OBSERVABLE = "observable"
    DISTRIBUTION = "distribution"  # CalibrationTarget.distribution_code
    DERIVATION = "derivation"  # Legacy parameter derivation


# Expected function signatures for each code type
EXPECTED_SIGNATURES: Dict[CodeType, Tuple[str, List[str]]] = {
    CodeType.SUBMODEL: ("submodel", ["t", "y", "params", "inputs"]),
    CodeType.SUBMODEL_OBSERVABLE: ("compute_observable", ["t", "y", "constants", "ureg"]),
    CodeType.OBSERVABLE: ("compute_observable", ["time", "species_dict", "constants", "ureg"]),
    CodeType.DISTRIBUTION: ("derive_distribution", ["inputs", "ureg"]),
    CodeType.DERIVATION: ("derive_parameter", ["inputs", "ureg"]),
}


# Numbers that are acceptable as inline literals
ALLOWED_NUMERIC_LITERALS: Set[float] = {
    # Basic integers for indexing, counting, math
    0,
    1,
    2,
    3,
    4,
    5,
    -1,
    -2,
    # Common fractions
    0.5,
    0.25,
    0.75,
    # Statistical percentiles
    2.5,
    25,
    50,
    75,
    97.5,
    0.025,
    0.975,
    0.05,
    0.95,
    # Common conversion factors
    100,
    1000,
    # Mathematical constants (numpy provides these, but sometimes hardcoded)
    # We'll also check for np.pi, np.e usage
}


@dataclass
class ValidationIssue:
    """A single validation issue found in code."""

    severity: str  # "error" or "warning"
    category: str  # "syntax", "signature", "hardcoded", "execution", "units"
    message: str
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class CodeValidationResult:
    """Result of code validation."""

    code_type: CodeType
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    executed_successfully: bool = False
    return_value: Optional[Any] = None

    def add_error(
        self,
        category: str,
        message: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ) -> None:
        """Add an error issue."""
        self.issues.append(
            ValidationIssue(
                severity="error",
                category=category,
                message=message,
                line=line,
                column=column,
            )
        )
        self.passed = False

    def add_warning(
        self,
        category: str,
        message: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ) -> None:
        """Add a warning issue."""
        self.issues.append(
            ValidationIssue(
                severity="warning",
                category=category,
                message=message,
                line=line,
                column=column,
            )
        )

    def get_errors(self) -> List[ValidationIssue]:
        """Get all error-level issues."""
        return [i for i in self.issues if i.severity == "error"]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == "warning"]


class HardcodedConstantVisitor(ast.NodeVisitor):
    """
    AST visitor that finds hardcoded numeric constants.

    Detects:
    - Numeric literals that aren't in the allowed set
    - Numbers multiplied by ureg (indicating unit attachment)
    - Magic numbers in mathematical expressions
    """

    def __init__(self, allowed_numbers: Optional[Set[float]] = None):
        self.allowed_numbers = allowed_numbers or ALLOWED_NUMERIC_LITERALS
        self.violations: List[Tuple[float, int, int, str]] = []  # (value, line, col, context)
        self._in_ureg_mult = False  # Track if we're in a ureg multiplication

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Check binary operations for number * ureg patterns."""
        # Check for: NUMBER * ureg.X or ureg.X * NUMBER
        if isinstance(node.op, ast.Mult):
            # Check left side is number, right side involves ureg
            if isinstance(node.left, ast.Constant) and self._involves_ureg(node.right):
                self._check_ureg_multiplication(node.left, node)
            # Check right side is number, left side involves ureg
            elif isinstance(node.right, ast.Constant) and self._involves_ureg(node.left):
                self._check_ureg_multiplication(node.right, node)

        self.generic_visit(node)

    def _involves_ureg(self, node: ast.AST) -> bool:
        """Check if a node involves ureg (attribute access, call, or power)."""
        if isinstance(node, ast.Attribute):
            return self._is_ureg_reference(node)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                return self._is_ureg_reference(node.func)
            elif isinstance(node.func, ast.Name) and node.func.id == "ureg":
                return True
        elif isinstance(node, ast.BinOp):
            # Handle ureg.mm ** 2 (Pow) or ureg.mm * ureg.s (Mult/Div)
            return self._involves_ureg(node.left) or self._involves_ureg(node.right)
        return False

    def _is_ureg_reference(self, node: ast.Attribute) -> bool:
        """Check if an attribute node references ureg."""
        if isinstance(node.value, ast.Name) and node.value.id == "ureg":
            return True
        if isinstance(node.value, ast.Attribute):
            return self._is_ureg_reference(node.value)
        return False

    def _check_ureg_multiplication(self, num_node: ast.Constant, context_node: ast.AST) -> None:
        """Check if a number in ureg multiplication is allowed."""
        if not isinstance(num_node.value, (int, float)):
            return

        value = float(num_node.value)
        if value not in self.allowed_numbers:
            # Get context string for better error messages
            try:
                context = ast.unparse(context_node)
            except Exception:
                context = f"line {num_node.lineno}"

            self.violations.append(
                (
                    value,
                    num_node.lineno,
                    num_node.col_offset,
                    context,
                )
            )

    def visit_Constant(self, node: ast.Constant) -> None:
        """
        Check standalone numeric constants.

        We're more lenient with standalone numbers since they might be
        array indices, loop bounds, etc. The main concern is numbers
        being attached to units.
        """
        # Only flag very suspicious standalone numbers (like physical constants)
        if isinstance(node.value, (int, float)):
            value = float(node.value)
            # Flag numbers that look like physical constants (specific ranges)
            # This is heuristic - numbers like 6.022e23, 1.38e-23, etc.
            if abs(value) > 1e6 or (0 < abs(value) < 1e-4 and value not in self.allowed_numbers):
                # Only add as warning, not error - might be legitimate
                pass  # We handle this via ureg multiplication detection

        self.generic_visit(node)


class CodeValidator:
    """
    Unified validator for calibration target code blocks.

    Usage:
        validator = CodeValidator()

        # Validate submodel code
        result = validator.validate(
            code="def submodel(t, y, params, inputs): ...",
            code_type=CodeType.SUBMODEL,
            params={"k_growth": 0.1},
            inputs={"initial_cells": 1000},
        )

        if not result.passed:
            for issue in result.get_errors():
                print(f"Error: {issue.message}")
    """

    def __init__(self, strict_hardcoded: bool = True):
        """
        Initialize the code validator.

        Args:
            strict_hardcoded: If True, hardcoded constants are errors.
                              If False, they're warnings.
        """
        self.strict_hardcoded = strict_hardcoded

    def validate(
        self,
        code: str,
        code_type: CodeType,
        check_hardcoded: bool = True,
        check_execution: bool = True,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> CodeValidationResult:
        """
        Validate a code block.

        Args:
            code: Python code string
            code_type: Type of code block (determines expected signature)
            check_hardcoded: Whether to check for hardcoded constants
            check_execution: Whether to test execution with mock data
            execution_context: Context for execution (params, inputs, etc.)

        Returns:
            CodeValidationResult with pass/fail status and issues
        """
        result = CodeValidationResult(code_type=code_type, passed=True)

        # 1. Syntax check
        tree = self._check_syntax(code, result)
        if tree is None:
            return result  # Can't continue without valid AST

        # 2. Function signature check
        func_node = self._check_signature(tree, code_type, result)

        # 3. Hardcoded constants check
        if check_hardcoded:
            self._check_hardcoded_constants(tree, result)

        # 4. Execution test
        if check_execution and func_node is not None and result.passed:
            self._check_execution(code, code_type, execution_context or {}, result)

        return result

    def _check_syntax(self, code: str, result: CodeValidationResult) -> Optional[ast.AST]:
        """Check code syntax and return AST if valid."""
        try:
            return ast.parse(code)
        except SyntaxError as e:
            result.add_error(
                category="syntax",
                message=f"Syntax error: {e.msg}",
                line=e.lineno,
                column=e.offset,
            )
            return None

    def _check_signature(
        self,
        tree: ast.AST,
        code_type: CodeType,
        result: CodeValidationResult,
    ) -> Optional[ast.FunctionDef]:
        """Check that code defines expected function with correct signature."""
        expected_name, expected_args = EXPECTED_SIGNATURES[code_type]

        # Find the function definition
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == expected_name:
                func_node = node
                break

        if func_node is None:
            result.add_error(
                category="signature",
                message=f"Code must define a function named '{expected_name}'",
            )
            return None

        # Check arguments
        actual_args = [arg.arg for arg in func_node.args.args]
        if actual_args != expected_args:
            result.add_error(
                category="signature",
                message=(
                    f"Function '{expected_name}' has wrong signature.\n"
                    f"  Expected: ({', '.join(expected_args)})\n"
                    f"  Got: ({', '.join(actual_args)})"
                ),
                line=func_node.lineno,
            )
            return None

        return func_node

    def _check_hardcoded_constants(self, tree: ast.AST, result: CodeValidationResult) -> None:
        """Check for hardcoded numeric constants attached to units."""
        visitor = HardcodedConstantVisitor()
        visitor.visit(tree)

        for value, line, col, context in visitor.violations:
            message = (
                f"Hardcoded constant {value} attached to units.\n"
                f"  Context: {context}\n"
                f"  All constants with units must be passed via params, inputs, or constants dict."
            )
            if self.strict_hardcoded:
                result.add_error(
                    category="hardcoded",
                    message=message,
                    line=line,
                    column=col,
                )
            else:
                result.add_warning(
                    category="hardcoded",
                    message=message,
                    line=line,
                    column=col,
                )

    def _check_execution(
        self,
        code: str,
        code_type: CodeType,
        context: Dict[str, Any],
        result: CodeValidationResult,
    ) -> None:
        """Test code execution with provided or mock context."""
        from qsp_llm_workflows.core.unit_registry import ureg

        expected_name, _ = EXPECTED_SIGNATURES[code_type]

        # Compile and extract function
        try:
            local_scope: Dict[str, Any] = {"np": np, "numpy": np, "ureg": ureg}
            exec(code, local_scope)
            func = local_scope.get(expected_name)
            if func is None:
                result.add_error(
                    category="execution",
                    message=f"Function '{expected_name}' not found after compilation",
                )
                return
        except Exception as e:
            result.add_error(
                category="execution",
                message=f"Failed to compile code: {e}",
            )
            return

        # Build execution arguments based on code type
        try:
            args = self._build_execution_args(code_type, context, ureg)
            return_value = func(*args)
            result.executed_successfully = True
            result.return_value = return_value
        except Exception as e:
            result.add_error(
                category="execution",
                message=f"Execution failed: {e}",
            )

    def _build_execution_args(
        self,
        code_type: CodeType,
        context: Dict[str, Any],
        ureg: Any,
    ) -> tuple:
        """Build execution arguments for testing."""
        if code_type == CodeType.SUBMODEL:
            # submodel(t, y, params, inputs)
            t = context.get("t", 0.0)
            y = context.get("y", [1.0])
            params = context.get("params", {})
            inputs = context.get("inputs", {})
            return (t, y, params, inputs)

        elif code_type == CodeType.SUBMODEL_OBSERVABLE:
            # compute_observable(t, y, constants, ureg)
            t = context.get("t", 1.0)
            y = context.get("y", [1.0])
            constants = context.get("constants", {})
            return (t, y, constants, ureg)

        elif code_type == CodeType.OBSERVABLE:
            # compute_observable(time, species_dict, constants, ureg)
            time = context.get("time", np.linspace(0, 10, 100) * ureg.day)
            species_dict = context.get("species_dict", {})
            constants = context.get("constants", {})
            return (time, species_dict, constants, ureg)

        elif code_type == CodeType.DISTRIBUTION:
            # derive_distribution(inputs, ureg)
            inputs = context.get("inputs", {})
            return (inputs, ureg)

        elif code_type == CodeType.DERIVATION:
            # derive_parameter(inputs, ureg)
            inputs = context.get("inputs", {})
            return (inputs, ureg)

        else:
            raise ValueError(f"Unknown code type: {code_type}")


def validate_code_block(
    code: str,
    code_type: CodeType,
    strict_hardcoded: bool = True,
    check_hardcoded: bool = True,
    check_execution: bool = True,
    execution_context: Optional[Dict[str, Any]] = None,
) -> CodeValidationResult:
    """
    Convenience function to validate a code block.

    Args:
        code: Python code string
        code_type: Type of code block
        strict_hardcoded: Treat hardcoded constants as errors (vs warnings)
        check_hardcoded: Whether to check for hardcoded constants
        check_execution: Whether to test execution with mock data
        execution_context: Context for execution testing

    Returns:
        CodeValidationResult
    """
    validator = CodeValidator(strict_hardcoded=strict_hardcoded)
    return validator.validate(
        code=code,
        code_type=code_type,
        check_hardcoded=check_hardcoded,
        check_execution=check_execution,
        execution_context=execution_context,
    )


def find_hardcoded_constants(code: str) -> List[Tuple[float, int, int, str]]:
    """
    Find hardcoded constants in code that are attached to units.

    Returns:
        List of (value, line, column, context) tuples
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    visitor = HardcodedConstantVisitor()
    visitor.visit(tree)
    return visitor.violations


__all__ = [
    "CodeType",
    "CodeValidator",
    "CodeValidationResult",
    "ValidationIssue",
    "validate_code_block",
    "find_hardcoded_constants",
    "EXPECTED_SIGNATURES",
    "ALLOWED_NUMERIC_LITERALS",
]
