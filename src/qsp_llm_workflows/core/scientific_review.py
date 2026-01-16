"""
Scientific review orchestration using Claude Code headless mode.

Provides a two-phase review workflow:
1. Scientific soundness review against a rubric
2. Prompt improvement recommendations (max 2, low complexity)
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from datetime import datetime

from qsp_llm_workflows.core.resource_utils import read_prompt


class ScientificReviewer:
    """
    Orchestrates scientific review using Claude Code headless mode.

    Two-phase workflow:
    1. Scientific soundness review (Claude evaluates against rubric)
    2. Prompt improvement recommendations (Claude suggests up to 2 low-complexity changes)
    """

    def __init__(
        self,
        workflow_type: str,
        recommendations_file: str = "review_recommendations.md",
    ):
        """
        Initialize scientific reviewer.

        Args:
            workflow_type: Either "parameter_estimates" or "test_statistics"
            recommendations_file: Path to save recommendations markdown
        """
        self.workflow_type = workflow_type
        self.recommendations_file = Path(recommendations_file)
        self.rubric = read_prompt("scientific_review_rubric.md")

        # Determine which prompt file to use
        if workflow_type == "test_statistics":
            self.prompt_file = "qsp_test_statistic_prompt.md"
        else:
            self.prompt_file = "qsp_parameter_extraction_prompt.md"

    def review_file(self, yaml_path: Path, interactive: bool = True) -> bool:
        """
        Run two-phase review on a single YAML file.

        Args:
            yaml_path: Path to the YAML file to review
            interactive: If True, prompt user for CONCERN decisions. If False, auto-reject CONCERNs.

        Returns:
            True if file passed review, False otherwise
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

        # Determine pass/fail
        overall = review_result.get("overall", "FAIL")
        passed = overall == "PASS"

        # Check for CONCERNs
        if overall == "CONCERN":
            concerns = sum(
                1
                for dim in review_result.get("dimensions", {}).values()
                if dim.get("score") == "CONCERN"
            )
            if interactive:
                print(f"\n{concerns} CONCERN(s) found. Accept despite concerns? [y/N] ", end="")
                response = input().strip().lower()
                if response == "y":
                    passed = True
                else:
                    print("Review rejected by user.")
            else:
                print(f"\n{concerns} CONCERN(s) found. (Non-interactive mode: not accepted)")

        # Phase 2: Prompt recommendations (regardless of pass/fail)
        self._generate_recommendations(review_result, yaml_path)

        return passed

    def review_directory(self, dir_path: Path, max_workers: int = 4) -> dict[str, bool]:
        """
        Review all YAML files in a directory in parallel.

        Args:
            dir_path: Directory containing YAML files
            max_workers: Maximum number of parallel workers

        Returns:
            Dict mapping filename to pass/fail status
        """
        dir_path = dir_path.resolve()
        yaml_files = list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml"))

        if not yaml_files:
            print(f"No YAML files found in {dir_path}")
            return {}

        # Clear recommendations file at start
        if self.recommendations_file.exists():
            self.recommendations_file.unlink()

        print(f"\nFound {len(yaml_files)} YAML files to review")
        print(f"Running with {max_workers} parallel workers\n")
        print("=" * 60)

        results: dict[str, bool] = {}
        all_review_results: dict[str, dict] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._review_file_quiet, f): f for f in yaml_files}

            for future in as_completed(future_to_file):
                yaml_file = future_to_file[future]
                try:
                    passed, review_result = future.result()
                    results[yaml_file.name] = passed
                    if review_result:
                        all_review_results[yaml_file.name] = review_result

                    # Print summary for this file
                    overall = review_result.get("overall", "ERROR") if review_result else "ERROR"
                    status = "✓ PASS" if passed else f"✗ {overall}"
                    print(f"{yaml_file.name}: {status}")

                except Exception as e:
                    results[yaml_file.name] = False
                    print(f"{yaml_file.name}: ✗ ERROR - {e}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        passed_count = sum(1 for v in results.values() if v)
        failed_count = len(results) - passed_count
        print(f"Passed: {passed_count}")
        print(f"Failed/Concerns: {failed_count}")

        # Generate consolidated recommendations based on all reviews
        if all_review_results:
            self._generate_consolidated_recommendations(all_review_results, len(yaml_files))

        return results

    def _review_file_quiet(self, yaml_path: Path) -> tuple[bool, Optional[dict]]:
        """
        Review a file without interactive prompts (for parallel execution).

        Only runs scientific soundness review - recommendations are generated
        once at the end based on all reviews.

        Returns:
            Tuple of (passed, review_result)
        """
        yaml_path = yaml_path.resolve()

        if not yaml_path.exists():
            return False, None

        # Scientific soundness review only (no per-file recommendations)
        review_result = self._run_scientific_review(yaml_path)

        if review_result is None:
            return False, None

        # Determine pass/fail (CONCERNs count as not passed)
        overall = review_result.get("overall", "FAIL")
        passed = overall == "PASS"

        return passed, review_result

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

    def _generate_recommendations(self, review_result: dict, yaml_path: Path):
        """
        Run Phase 2: Generate prompt improvement recommendations.

        Args:
            review_result: Scientific review result
            yaml_path: Path to YAML file
        """
        print(self._header("PROMPT IMPROVEMENT RECOMMENDATIONS"))

        # Collect issues from review
        issues = []
        if review_result:
            for issue in review_result.get("critical_issues", []):
                issues.append(issue)
            for dim_name, dim_data in review_result.get("dimensions", {}).items():
                if dim_data.get("score") in ["CONCERN", "FAIL"]:
                    issues.append(
                        f"{dim_name.replace('_', ' ').title()}: {dim_data.get('reasoning', '')}"
                    )

        if not issues:
            print("\nNo issues found - no prompt improvements needed.")
            return

        issues_text = "\n".join(f"- {issue}" for issue in issues)

        # Get the path to the current prompt file
        from qsp_llm_workflows.core.resource_utils import get_package_root

        prompt_path = get_package_root() / "prompts" / self.prompt_file

        prompt = f"""Based on reviewing a {self.workflow_type.replace('_', ' ')} extraction, suggest prompt improvements.

IMPORTANT CONSTRAINTS:
- Provide AT MOST 2 recommendations
- Each recommendation must be LOW COMPLEXITY (1-2 sentences to add, no structural changes)
- Do NOT suggest changes that add significant complexity to the prompt
- Only suggest changes that are highly generalizable across many extractions

First, read the current prompt to understand what guidance already exists:
{prompt_path}

Then read the extraction that was reviewed:
{yaml_path}

Issues found in the extraction:
{issues_text}

For each recommendation (max 2), provide:
1. The issue it addresses
2. The exact text to add to the prompt (keep it brief!)
3. Where in the prompt it should go

Skip any recommendation if:
- The prompt already covers this adequately
- It would only help this specific extraction (not generalizable)
- It requires significant prompt restructuring

If no low-complexity, generalizable improvements are warranted, say so."""

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

    def _generate_consolidated_recommendations(
        self, all_review_results: dict[str, dict], num_files: int
    ):
        """
        Generate consolidated recommendations based on all review results.

        Args:
            all_review_results: Dict mapping filename to review result
            num_files: Total number of files reviewed (max recommendations)
        """
        print("\n" + "=" * 60)
        print("PROMPT IMPROVEMENT RECOMMENDATIONS")
        print("=" * 60)

        # Collect all issues across files
        all_issues: list[dict] = []
        for filename, review_result in all_review_results.items():
            for issue in review_result.get("critical_issues", []):
                all_issues.append({"file": filename, "issue": issue})
            for dim_name, dim_data in review_result.get("dimensions", {}).items():
                if dim_data.get("score") in ["CONCERN", "FAIL"]:
                    all_issues.append(
                        {
                            "file": filename,
                            "dimension": dim_name.replace("_", " ").title(),
                            "score": dim_data.get("score"),
                            "reasoning": dim_data.get("reasoning", ""),
                        }
                    )

        if not all_issues:
            print("\nNo issues found across all files - no prompt improvements needed.")
            return

        # Format issues for the prompt
        issues_text = ""
        for item in all_issues:
            if "dimension" in item:
                issues_text += f"- [{item['file']}] {item['dimension']} ({item['score']}): {item['reasoning']}\n"
            else:
                issues_text += f"- [{item['file']}] {item['issue']}\n"

        # Get the path to the current prompt file
        from qsp_llm_workflows.core.resource_utils import get_package_root

        prompt_path = get_package_root() / "prompts" / self.prompt_file

        print(f"\nAnalyzing {len(all_issues)} issues across {len(all_review_results)} files...")

        prompt = f"""Based on reviewing {len(all_review_results)} {self.workflow_type.replace('_', ' ')} extractions, suggest prompt improvements.

IMPORTANT CONSTRAINTS:
- Provide AT MOST {num_files} recommendations (one per file reviewed)
- Each recommendation must be LOW COMPLEXITY (1-2 sentences to add, no structural changes)
- Do NOT suggest changes that add significant complexity to the prompt
- Only suggest changes that are highly generalizable
- Look for PATTERNS across multiple files - prioritize issues that appear repeatedly

First, read the current prompt to understand what guidance already exists:
{prompt_path}

Issues found across all reviewed files:
{issues_text}

For each recommendation, provide:
1. The pattern/issue it addresses (mention which files exhibited this)
2. The exact text to add to the prompt (keep it brief!)
3. Where in the prompt it should go

Skip any recommendation if:
- The prompt already covers this adequately
- It would only help one specific extraction (not generalizable)
- It requires significant prompt restructuring

Prioritize recommendations that would fix issues appearing in multiple files.
If no low-complexity, generalizable improvements are warranted, say so."""

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
            self._save_consolidated_recommendations(all_issues, recommendations, num_files)

        except subprocess.TimeoutExpired:
            print("\nError: Timed out generating recommendations")
        except json.JSONDecodeError:
            print("\nError: Could not parse recommendations")
        except FileNotFoundError:
            print("\nError: Claude Code CLI not found")

    def _save_consolidated_recommendations(
        self, all_issues: list[dict], recommendations: str, num_files: int
    ):
        """Save consolidated recommendations to markdown file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format issues for markdown
        issues_md = ""
        for item in all_issues:
            if "dimension" in item:
                issues_md += f"- **[{item['file']}]** {item['dimension']} ({item['score']}): {item['reasoning']}\n"
            else:
                issues_md += f"- **[{item['file']}]** {item['issue']}\n"

        content = f"""# Consolidated Prompt Improvement Recommendations

Generated: {timestamp}
Workflow type: {self.workflow_type}
Files reviewed: {num_files}
Total issues found: {len(all_issues)}

## Issues Found

{issues_md}

## Recommendations

{recommendations}
"""

        with open(self.recommendations_file, "w") as f:
            f.write(content)

        print(f"\nRecommendations saved to: {self.recommendations_file}")

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
