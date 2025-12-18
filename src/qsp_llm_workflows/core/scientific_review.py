"""
Scientific review orchestration using Claude Code headless mode.

Provides a three-phase review workflow:
1. Scientific soundness review against a rubric
2. Automated validation with intelligent fix suggestions
3. Prompt improvement recommendations
"""

import json
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

from qsp_llm_workflows.core.resource_utils import read_prompt
from qsp_llm_workflows.core.pydantic_models import ParameterMetadata, TestStatistic
from qsp_llm_workflows.validate.orchestrator import ValidationOrchestrator
from qsp_llm_workflows.validate.run_all_validations import get_validators


class ScientificReviewer:
    """
    Orchestrates scientific review using Claude Code headless mode.

    Three-phase workflow:
    1. Scientific soundness review (Claude evaluates against rubric)
    2. Validation + intelligent fix (run validators, Claude fixes issues)
    3. Prompt improvement recommendations (Claude suggests prompt changes)
    """

    def __init__(
        self,
        workflow_type: str,
        species_units_file: Optional[str] = None,
        recommendations_file: str = "review_recommendations.md",
    ):
        """
        Initialize scientific reviewer.

        Args:
            workflow_type: Either "parameter_estimates" or "test_statistics"
            species_units_file: Path to species_units.json (for test statistics validation)
            recommendations_file: Path to save recommendations markdown
        """
        self.workflow_type = workflow_type
        self.species_units_file = species_units_file
        self.recommendations_file = Path(recommendations_file)
        self.rubric = read_prompt("scientific_review_rubric.md")

        # Determine model class
        if workflow_type == "test_statistics":
            self.model_class = TestStatistic
        else:
            self.model_class = ParameterMetadata

    def review_file(self, yaml_path: Path) -> bool:
        """
        Run full three-phase review on a single YAML file.

        Args:
            yaml_path: Path to the YAML file to review

        Returns:
            True if file passed review and validation, False otherwise
        """
        yaml_path = yaml_path.resolve()

        if not yaml_path.exists():
            print(f"Error: File not found: {yaml_path}")
            return False

        print(self._header("SCIENTIFIC SOUNDNESS REVIEW"))
        print(f"\nFile: {yaml_path.name}\n")

        # Phase 1: Scientific soundness review
        print("Reviewing against scientific soundness rubric...\n")
        review_result = self._run_scientific_review(yaml_path)

        if review_result is None:
            print("Error: Could not complete scientific review")
            return False

        self._display_review(review_result)

        # Check for FAILs
        if review_result.get("overall") == "FAIL":
            print("\nReview result: FAIL - Cannot proceed with validation.")
            self._generate_recommendations(review_result, None, yaml_path)
            return False

        # Check for CONCERNs
        if review_result.get("overall") == "CONCERN":
            concerns = sum(
                1
                for dim in review_result.get("dimensions", {}).values()
                if dim.get("score") == "CONCERN"
            )
            print(f"\n{concerns} CONCERN(s) found. Proceed despite concerns? [y/N] ", end="")
            response = input().strip().lower()
            if response != "y":
                print("Review stopped by user.")
                self._generate_recommendations(review_result, None, yaml_path)
                return False

        # Phase 2: Validation + fix
        print(self._header("AUTOMATED VALIDATION"))

        validation_result = self._run_validation(yaml_path)

        if validation_result and validation_result.has_failures:
            print(f"\n{self._count_failures(validation_result)} validation failure(s) found.")
            print("\nAttempt intelligent fix? [Y/n] ", end="")
            response = input().strip().lower()
            if response != "n":
                self._run_intelligent_fix(yaml_path, validation_result)
                # Re-run validation after fix
                print("\nRe-running validation after fix...")
                validation_result = self._run_validation(yaml_path)
        elif validation_result:
            print("\nAll validations passed!")

        # Phase 3: Prompt recommendations
        self._generate_recommendations(review_result, validation_result, yaml_path)

        return validation_result is not None and not validation_result.has_failures

    def _header(self, title: str) -> str:
        """Create a formatted header."""
        width = 60
        return f"\n{'=' * width}\n{title}\n{'=' * width}"

    def _run_scientific_review(self, yaml_path: Path) -> Optional[dict]:
        """
        Run Phase 1: Scientific soundness review using Claude Code.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Review result dict or None on error
        """
        prompt = f"""Review this QSP {self.workflow_type.replace('_', ' ')} extraction for scientific soundness.

File to review: {yaml_path}

First, read the file using the Read tool, then evaluate it against this rubric:

{self.rubric}

After your analysis, output ONLY a JSON object with your assessment (no markdown code blocks, just raw JSON)."""

        try:
            result = subprocess.run(
                [
                    "claude",
                    "-p",
                    prompt,
                    "--allowedTools",
                    "Read",
                    "--output-format",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                print(f"Claude Code error: {result.stderr}")
                return None

            # Parse the JSON output
            output = json.loads(result.stdout)

            # The response is in the "result" field
            response_text = output.get("result", "")

            # Try to extract JSON from the response
            return self._extract_json(response_text)

        except subprocess.TimeoutExpired:
            print("Error: Claude Code timed out")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing Claude Code output: {e}")
            return None
        except FileNotFoundError:
            print("Error: Claude Code CLI not found. Please install it first.")
            return None

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON object from response text."""
        # Try parsing directly first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        return None

    def _display_review(self, result: dict):
        """Display the review results."""
        dimensions = result.get("dimensions", {})

        for dim_name, dim_data in dimensions.items():
            score = dim_data.get("score", "N/A")
            reasoning = dim_data.get("reasoning", "")
            display_name = dim_name.replace("_", " ").title()

            # Color-code the score
            if score == "PASS":
                score_display = f"✓ {score}"
            elif score == "CONCERN":
                score_display = f"⚠ {score}"
            else:
                score_display = f"✗ {score}"

            print(f"\n{display_name}: {score_display}")
            if reasoning:
                # Wrap long reasoning text
                print(f"  {reasoning}")

        # Display overall
        overall = result.get("overall", "N/A")
        print(f"\n{'─' * 60}")
        print(f"OVERALL: {overall}")

        # Display critical issues
        issues = result.get("critical_issues", [])
        if issues:
            print("\nCritical Issues:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

    def _run_validation(self, yaml_path: Path):
        """
        Run validation suite on the file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            ValidationResult or None
        """
        data_dir = yaml_path.parent
        output_dir = Path("validation-outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        orchestrator = ValidationOrchestrator(str(data_dir), output_dir, self.model_class.__name__)

        validators = get_validators(str(data_dir), self.model_class, self.species_units_file)

        return orchestrator.run_all_validations(validators)

    def _count_failures(self, validation_result) -> int:
        """Count total failures across all validators."""
        return sum(len(report.failed) for report in validation_result.reports)

    def _run_intelligent_fix(self, yaml_path: Path, validation_result):
        """
        Run Phase 2: Intelligent fix using Claude Code.

        Args:
            yaml_path: Path to YAML file
            validation_result: ValidationResult from validation run
        """
        # Collect all failures
        failures = []
        for report in validation_result.reports:
            for failure in report.failed:
                failures.append(
                    {
                        "validator": report.name,
                        "item": failure.get("item"),
                        "reason": failure.get("reason"),
                    }
                )

        failures_json = json.dumps(failures, indent=2)

        prompt = f"""Fix validation errors in this QSP extraction.

File to fix: {yaml_path}

Validation errors:
{failures_json}

Instructions:
1. First, read the file using the Read tool
2. Understand WHY each error occurred
3. Make targeted fixes using the Edit tool - do not regenerate the entire file
4. Preserve everything that is correct
5. After fixing, explain what you changed and why

Focus on the root cause of each error. For example:
- If a text snippet doesn't contain the value, check if the value is correct or if the snippet needs updating
- If code doesn't execute, debug it carefully
- If DOIs are invalid, verify and fix them"""

        print("\nAnalyzing failures and attempting fixes...\n")

        try:
            result = subprocess.run(
                [
                    "claude",
                    "-p",
                    prompt,
                    "--allowedTools",
                    "Read,Edit",
                    "--output-format",
                    "stream-json",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                print(f"Claude Code error: {result.stderr}")
                return

            # Parse streaming JSON output
            for line in result.stdout.strip().split("\n"):
                if line:
                    try:
                        event = json.loads(line)
                        if event.get("type") == "assistant" and "message" in event:
                            content = event["message"].get("content", [])
                            for block in content:
                                if block.get("type") == "text":
                                    print(block.get("text", ""))
                    except json.JSONDecodeError:
                        continue

        except subprocess.TimeoutExpired:
            print("Error: Claude Code timed out during fix attempt")
        except FileNotFoundError:
            print("Error: Claude Code CLI not found")

    def _generate_recommendations(
        self, review_result: Optional[dict], validation_result, yaml_path: Path
    ):
        """
        Run Phase 3: Generate prompt improvement recommendations.

        Args:
            review_result: Scientific review result
            validation_result: Validation result (may be None)
            yaml_path: Path to YAML file
        """
        print(self._header("PROMPT IMPROVEMENT RECOMMENDATIONS"))

        # Collect issues
        issues = []

        if review_result:
            for issue in review_result.get("critical_issues", []):
                issues.append(f"Scientific review: {issue}")
            for dim_name, dim_data in review_result.get("dimensions", {}).items():
                if dim_data.get("score") in ["CONCERN", "FAIL"]:
                    issues.append(
                        f"{dim_name.replace('_', ' ').title()}: {dim_data.get('reasoning', '')}"
                    )

        if validation_result:
            for report in validation_result.reports:
                for failure in report.failed:
                    issues.append(f"Validation ({report.name}): {failure.get('reason', '')}")

        if not issues:
            print("\nNo issues found - no prompt improvements needed.")
            return

        issues_text = "\n".join(f"- {issue}" for issue in issues)

        prompt = f"""Based on reviewing this {self.workflow_type.replace('_', ' ')} extraction, suggest prompt improvements.

File reviewed: {yaml_path}

Issues found:
{issues_text}

Recommend changes that:
1. Would prevent similar issues in future extractions
2. Are generalizable (not just for this specific file)
3. Balance improvement vs. added complexity

For each recommendation, provide:
- The issue it addresses
- The specific prompt change (with example text)
- Priority (High/Medium/Low)
- Generalizability (High/Medium/Low)
- Complexity to implement (High/Medium/Low)

Format your response as a numbered list of recommendations."""

        try:
            result = subprocess.run(
                [
                    "claude",
                    "-p",
                    prompt,
                    "--allowedTools",
                    "Read",
                    "--output-format",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                print(f"\nCould not generate recommendations: {result.stderr}")
                return

            output = json.loads(result.stdout)
            recommendations = output.get("result", "No recommendations generated.")

            print(f"\n{recommendations}")

            # Save to file
            self._save_recommendations(yaml_path, issues, recommendations)

        except subprocess.TimeoutExpired:
            print("\nError: Timed out generating recommendations")
        except json.JSONDecodeError:
            print("\nError: Could not parse recommendations")
        except FileNotFoundError:
            print("\nError: Claude Code CLI not found")

    def _save_recommendations(self, yaml_path: Path, issues: list, recommendations: str):
        """Save recommendations to markdown file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content = f"""# Prompt Improvement Recommendations

Generated: {timestamp}
File reviewed: {yaml_path.name}
Workflow type: {self.workflow_type}

## Issues Found

{chr(10).join(f'- {issue}' for issue in issues)}

## Recommendations

{recommendations}

---
"""

        # Append to existing file or create new
        mode = "a" if self.recommendations_file.exists() else "w"
        with open(self.recommendations_file, mode) as f:
            f.write(content)

        print(f"\nRecommendations saved to: {self.recommendations_file}")
