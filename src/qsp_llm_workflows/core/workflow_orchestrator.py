#!/usr/bin/env python3
"""
Workflow orchestrator for automated extraction pipeline.

Handles the complete extraction workflow from batch creation through git commit/push:
1. Create batch requests
2. Upload and monitor batch
3. Run validation (checklist)
4. Unpack results to review folder
5. Commit and push to review branch
"""

import json
import sys
import time
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from openai import OpenAI

from qsp_llm_workflows.core.batch_creator import (
    ParameterBatchCreator,
    TestStatisticBatchCreator,
)
from qsp_llm_workflows.core.immediate_processor import ImmediateRequestProcessor


class WorkflowOrchestrator:
    """Orchestrates complete extraction workflow with validation and git operations."""

    def __init__(self, base_dir: Path, storage_dir: Path, api_key: str):
        """
        Initialize workflow orchestrator.

        Args:
            base_dir: Base directory of qsp-llm-workflows
            storage_dir: Path to qsp-metadata-storage repository
            api_key: OpenAI API key
        """
        self.base_dir = Path(base_dir)
        self.storage_dir = Path(storage_dir)
        self.batch_jobs_dir = self.base_dir / "batch_jobs"
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

        # Ensure directories exist
        self.batch_jobs_dir.mkdir(exist_ok=True)
        self.to_review_dir = self.storage_dir / "to-review"
        self.to_review_dir.mkdir(exist_ok=True)

    def extract_prompt_to_file(self, batch_file: Path, index: int = 0) -> Path:
        """
        Extract a sample prompt from batch file to scratch directory.

        Uses scripts/debug/extract_prompt.py to extract prompt and save to file.

        Args:
            batch_file: Path to batch JSONL file
            index: Index of request to extract (default: 0 for first request)

        Returns:
            Path to extracted prompt file
        """
        # Create scratch directory
        scratch_dir = self.base_dir / "scratch"
        scratch_dir.mkdir(exist_ok=True)

        # Generate output filename
        output_file = scratch_dir / f"prompt_preview_{batch_file.stem}.txt"

        # Run extract_prompt.py script
        extract_script = self.base_dir / "scripts" / "debug" / "extract_prompt.py"

        result = subprocess.run(
            ["python3", str(extract_script), str(batch_file), str(index)],
            cwd=self.base_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        # Write output to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        return output_file

    def create_batch(
        self,
        input_csv: Path,
        workflow_type: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """
        Create batch requests using appropriate batch creator.

        Args:
            input_csv: Path to input CSV file
            workflow_type: Type of workflow (parameter/test_statistic)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to created batch requests JSONL file
        """
        if progress_callback:
            progress_callback(f"Creating {workflow_type} batch requests...")

        # Select appropriate batch creator
        if workflow_type == "parameter":
            creator = ParameterBatchCreator(self.base_dir)
        elif workflow_type == "test_statistic":
            creator = TestStatisticBatchCreator(self.base_dir)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        # Create batch
        output_file = creator.run(None, input_csv)

        if not output_file.exists():
            raise RuntimeError(f"Expected batch file not created: {output_file}")

        if progress_callback:
            progress_callback(f"✓ Batch requests created: {output_file.name}")

        return output_file

    def upload_batch(
        self, batch_file: Path, progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Upload batch to OpenAI API.

        Args:
            batch_file: Path to batch requests JSONL file
            progress_callback: Optional callback for progress updates

        Returns:
            Batch ID
        """
        if progress_callback:
            progress_callback(f"Uploading batch: {batch_file.name}...")

        # Upload file
        with open(batch_file, "rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")

        # Create batch
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id, endpoint="/v1/responses", completion_window="24h"
        )

        # Save batch metadata
        batch_id_file = batch_file.with_suffix(".batch_id")
        with open(batch_id_file, "w") as f:
            json.dump(
                {
                    "batch_id": batch.id,
                    "batch_type": batch_file.stem.replace("_requests", ""),
                    "source_csv": None,
                    "created_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        if progress_callback:
            progress_callback(f"✓ Batch uploaded: {batch.id}")

        return batch.id

    def monitor_batch(
        self,
        batch_id: str,
        timeout: int = 3600,
        poll_interval: int = 30,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """
        Monitor batch until completion and download results.

        Args:
            batch_id: Batch ID to monitor
            timeout: Maximum seconds to wait (default: 1 hour)
            poll_interval: Seconds between status checks (default: 30)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to downloaded results file

        Raises:
            TimeoutError: If batch doesn't complete within timeout
            RuntimeError: If batch fails
        """
        if progress_callback:
            progress_callback(f"Monitoring batch {batch_id}...")

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")

            batch = self.client.batches.retrieve(batch_id)

            # Progress update
            if batch.request_counts and progress_callback:
                completed = batch.request_counts.completed
                total = batch.request_counts.total
                progress_callback(f"  Status: {batch.status} ({completed}/{total} completed)")

            if batch.status == "completed":
                if not batch.output_file_id:
                    raise RuntimeError(f"Batch completed but no output file: {batch_id}")

                # Download results
                content = self.client.files.content(batch.output_file_id)
                output_file = self.batch_jobs_dir / f"{batch_id}_results.jsonl"

                with open(output_file, "wb") as f:
                    f.write(content.content)

                if progress_callback:
                    progress_callback(f"✓ Results downloaded: {output_file.name}")

                return output_file

            elif batch.status == "failed":
                raise RuntimeError(f"Batch {batch_id} failed")

            elif batch.status in ["expired", "cancelled"]:
                raise RuntimeError(f"Batch {batch_id} was {batch.status}")

            # Wait before next check
            time.sleep(poll_interval)

    def process_immediate_direct(
        self,
        input_csv: Path,
        workflow_type: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """
        Process requests directly via Responses API (no batch file creation).

        Args:
            input_csv: Path to input CSV file
            workflow_type: Type of workflow (parameter/test_statistic)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to results file (batch-compatible format for unpacker)
        """
        # Create immediate processor
        processor = ImmediateRequestProcessor(self.base_dir, self.api_key)

        # Process requests directly from CSV
        results = processor.run(input_csv, workflow_type, progress_callback)

        # Write results to file (for unpacker compatibility)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.batch_jobs_dir / f"immediate_{workflow_type}_{timestamp}_results.jsonl"

        with open(results_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        if progress_callback:
            progress_callback(f"✓ Results saved: {results_file.name}")

        return results_file

    def check_and_commit_existing_changes(
        self, progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Check for existing changes in storage repo and commit them on current branch.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            True if changes were committed, False if no changes found

        Raises:
            RuntimeError: If user declines to commit changes
        """
        # Check git status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.storage_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        # If no changes, return early
        if not result.stdout.strip():
            return False

        # Parse status to show what will be committed
        status_lines = result.stdout.strip().split("\n")
        tracked_changes = [line for line in status_lines if not line.startswith("??")]
        untracked_files = [line for line in status_lines if line.startswith("??")]

        if progress_callback:
            progress_callback("\n⚠  Detected existing changes in qsp-metadata-storage:")
            if tracked_changes:
                progress_callback(f"  - {len(tracked_changes)} tracked change(s)")
            if untracked_files:
                progress_callback(f"  - {len(untracked_files)} untracked file(s)")

        # Get current branch name
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self.storage_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        current_branch = branch_result.stdout.strip()

        if progress_callback:
            progress_callback(f"  - Current branch: {current_branch}")

        # Prompt user to commit existing changes
        response = input("\nCommit these changes on current branch before proceeding? [y/N]: ")
        if response.lower() != "y":
            raise RuntimeError(
                "Cannot proceed with uncommitted changes in qsp-metadata-storage. "
                "Please commit or stash changes manually."
            )

        # Stage all changes (tracked and untracked)
        subprocess.run(["git", "add", "-A"], cwd=self.storage_dir, check=True, capture_output=True)

        # Create commit message
        commit_msg = f"""Save work in progress before automated workflow

Branch: {current_branch}
Tracked changes: {len(tracked_changes)}
Untracked files: {len(untracked_files)}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This commit was created automatically to preserve existing work
before unpacking new results from LLM workflow."""

        # Commit
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=self.storage_dir,
            check=True,
            capture_output=True,
        )

        if progress_callback:
            progress_callback(f"✓ Committed existing changes on branch: {current_branch}\n")

        return True

    def unpack_to_review(
        self,
        validated_results: Path,
        input_csv: Path,
        workflow_type: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Path]:
        """
        Unpack validated results to to-review folder.

        Args:
            validated_results: Path to validation results JSONL
            input_csv: Path to original input CSV
            workflow_type: Type of workflow (parameter/test_statistic)
            progress_callback: Optional callback for progress updates

        Returns:
            List of unpacked YAML files
        """
        # Create subdirectory based on workflow type
        subdirectory_map = {"parameter": "parameter_estimates", "test_statistic": "test_statistics"}

        subdir_name = subdirectory_map.get(workflow_type, workflow_type)
        output_dir = self.to_review_dir / subdir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(f"Unpacking validated results to to-review/{subdir_name}/...")

        # Determine template based on workflow type
        template_map = {
            "parameter": "templates/parameter_metadata_template.yaml",
            "test_statistic": "templates/test_statistic_template.yaml",
        }

        template_path = template_map.get(workflow_type)

        # Run unpack script
        script_path = self.base_dir / "scripts" / "process" / "unpack_results.py"

        result = subprocess.run(
            [
                "python3",
                str(script_path),
                str(validated_results),
                str(output_dir),
                str(input_csv),
                "",  # No source directory for header extraction
                str(self.base_dir / template_path) if template_path else "",
            ],
            cwd=self.base_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Unpacking failed: {result.stderr}")

        # Find unpacked files
        unpacked_files = list(output_dir.glob("*.yaml"))

        if progress_callback:
            progress_callback(f"✓ Unpacked {len(unpacked_files)} files to to-review/{subdir_name}/")

        return unpacked_files

    def run_full_validation(
        self,
        unpacked_files: List[Path],
        workflow_type: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run full validation suite on unpacked YAML files.

        Args:
            unpacked_files: List of unpacked YAML files
            workflow_type: Type of workflow (parameter/test_statistic)
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with validation results
        """
        if progress_callback:
            progress_callback("Running full validation suite...")

        # Import validation runner
        from validation_runner import ValidationRunner

        # Determine template path
        template_map = {
            "parameter": "templates/parameter_metadata_template.yaml",
            "test_statistic": "templates/test_statistic_template.yaml",
        }

        template_path = self.base_dir / template_map.get(workflow_type, "")

        # Create validation output directory
        validation_output = self.batch_jobs_dir / "validation"
        validation_output.mkdir(exist_ok=True)

        # Determine subdirectory based on workflow type
        subdirectory_map = {"parameter": "parameter_estimates", "test_statistic": "test_statistics"}
        subdir_name = subdirectory_map.get(workflow_type, workflow_type)
        data_dir = self.to_review_dir / subdir_name

        # Run validation
        runner = ValidationRunner(self.base_dir)
        validation_results = runner.run_validation(
            workflow_type=workflow_type,
            data_dir=data_dir,
            template=template_path,
            output_dir=validation_output,
            timeout=600,
        )

        # Report results
        if progress_callback:
            if validation_results.get("status") == "completed":
                total = validation_results.get("total_validations", 0)
                passed = validation_results.get("passed", 0)
                failed = validation_results.get("failed", 0)
                progress_callback(f"✓ Validation complete: {passed}/{total} checks passed")
                if failed > 0:
                    progress_callback(f"  ⚠ {failed} validation(s) failed - review recommended")
            elif validation_results.get("status") == "skipped":
                progress_callback(
                    f"⚠ Validation skipped: {validation_results.get('message', 'N/A')}"
                )
            else:
                progress_callback(
                    f"⚠ Validation error: {validation_results.get('message', 'Unknown error')}"
                )

        return validation_results

    def commit_and_push(
        self,
        unpacked_files: List[Path],
        batch_id: str,
        workflow_type: str,
        validation_summary: Optional[Dict[str, Any]] = None,
        branch_prefix: str = "review/batch",
        push: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Commit unpacked files to review branch and push to remote.

        Args:
            unpacked_files: List of unpacked YAML files
            batch_id: Batch ID for naming
            workflow_type: Type of workflow
            validation_summary: Optional validation results to include in commit
            branch_prefix: Prefix for branch name
            push: Whether to push to remote (default: True)
            progress_callback: Optional callback for progress updates

        Returns:
            Name of created branch
        """
        if progress_callback:
            progress_callback("Creating review branch and committing files...")

        # Generate branch name
        date_str = datetime.now().strftime("%Y-%m-%d")
        batch_hash = hashlib.md5(batch_id.encode()).hexdigest()[:6]
        branch_name = f"{branch_prefix}-{workflow_type}-{date_str}-{batch_hash}"

        # Git operations in storage directory
        try:
            # Create new branch
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=self.storage_dir,
                check=True,
                capture_output=True,
            )

            # Determine subdirectory based on workflow type
            subdirectory_map = {
                "parameter": "parameter_estimates",
                "test_statistic": "test_statistics",
            }
            subdir_name = subdirectory_map.get(workflow_type, workflow_type)

            # Add only files from this workflow's subdirectory
            subprocess.run(
                ["git", "add", f"to-review/{subdir_name}/"],
                cwd=self.storage_dir,
                check=True,
                capture_output=True,
            )

            # Format validation summary for commit message (if provided)
            validation_text = ""
            if validation_summary:
                from validation_runner import ValidationRunner

                runner = ValidationRunner(self.base_dir)
                validation_text = runner.format_summary_for_commit(validation_summary)

            # Create commit message
            commit_msg = f"""Add {workflow_type} extractions for review

Batch ID: {batch_id}
Files: {len(unpacked_files)} extractions
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{validation_text}
Automated extraction. Ready for validation and manual review."""

            # Commit
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.storage_dir,
                check=True,
                capture_output=True,
            )

            # Push if requested
            if push:
                subprocess.run(
                    ["git", "push", "-u", "origin", branch_name],
                    cwd=self.storage_dir,
                    check=True,
                    capture_output=True,
                )
                if progress_callback:
                    progress_callback(f"✓ Pushed to origin/{branch_name}")
            else:
                if progress_callback:
                    progress_callback(f"✓ Created local branch: {branch_name}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git operation failed: {e.stderr.decode() if e.stderr else str(e)}")

        return branch_name

    def run_complete_workflow(
        self,
        input_csv: Path,
        workflow_type: str,
        timeout: int = 3600,
        push: bool = True,
        branch_prefix: str = "review/batch",
        immediate: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete extraction workflow from start to finish.

        Validation is NOT run during this workflow. After completion, run:
        python scripts/validate/run_all_validations.py <workflow_type>

        Args:
            input_csv: Path to input CSV file
            workflow_type: Type of workflow (parameter/test_statistic)
            timeout: Maximum seconds to wait for batch completion
            push: Whether to push to remote
            branch_prefix: Prefix for review branch name
            immediate: Use Responses API for immediate processing (faster, good for testing)
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with workflow results and metadata
        """
        start_time = time.time()
        results = {
            "workflow_type": workflow_type,
            "input_csv": str(input_csv),
            "started_at": datetime.now().isoformat(),
        }

        try:
            # Branch: immediate mode or batch mode
            if immediate:
                # Immediate mode: Direct processing via Responses API (no batch file)
                results_file = self.process_immediate_direct(
                    input_csv, workflow_type, progress_callback
                )
                results["results_file"] = str(results_file)
                results["immediate_mode"] = True
                batch_id = f"immediate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                # Batch mode: Create batch file, upload, monitor
                # Step 1: Create batch
                batch_file = self.create_batch(input_csv, workflow_type, progress_callback)
                results["batch_file"] = str(batch_file)

                # Step 1.5: Extract sample prompt and ask for confirmation
                if progress_callback:
                    progress_callback("\n" + "=" * 70)
                    progress_callback("SAMPLE PROMPT VERIFICATION")
                    progress_callback("=" * 70)
                    progress_callback("\nExtracting first prompt for review...")

                prompt_file = self.extract_prompt_to_file(batch_file, index=0)

                if progress_callback:
                    progress_callback(f"✓ Sample prompt saved to: {prompt_file}")
                    progress_callback("\nPlease review the prompt before proceeding.")
                    progress_callback("=" * 70)

                response = input("\nProceed with batch submission? [y/N]: ")
                if response.lower() != "y":
                    if progress_callback:
                        progress_callback("Aborted by user.")
                    results["status"] = "aborted"
                    results["error"] = "User aborted after prompt verification"
                    results["duration_seconds"] = time.time() - start_time
                    return results

                # Step 2: Upload batch
                batch_id = self.upload_batch(batch_file, progress_callback)
                results["batch_id"] = batch_id

                # Step 3: Monitor batch
                results_file = self.monitor_batch(
                    batch_id, timeout, progress_callback=progress_callback
                )
                results["results_file"] = str(results_file)
                results["immediate_mode"] = False

            # Step 4: Check for existing changes and commit if needed
            self.check_and_commit_existing_changes(progress_callback)

            # Step 5: Unpack to review
            unpacked_files = self.unpack_to_review(
                results_file, input_csv, workflow_type, progress_callback
            )
            results["unpacked_files"] = [str(f) for f in unpacked_files]
            results["file_count"] = len(unpacked_files)

            # Step 6: Commit and push
            branch_name = self.commit_and_push(
                unpacked_files,
                batch_id,
                workflow_type,
                validation_summary=None,  # Validation happens separately
                branch_prefix=branch_prefix,
                push=push,
                progress_callback=progress_callback,
            )
            results["branch_name"] = branch_name
            results["pushed"] = push

            # Success!
            results["status"] = "success"
            results["completed_at"] = datetime.now().isoformat()
            results["duration_seconds"] = time.time() - start_time

            # Add next steps info
            workflow_type_map = {
                "parameter": "parameter_estimates",
                "test_statistic": "test_statistics",
            }
            validation_type = workflow_type_map.get(workflow_type, workflow_type)
            results["next_step_validation_command"] = (
                f"python scripts/validate/run_all_validations.py {validation_type}"
            )

            return results

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            results["completed_at"] = datetime.now().isoformat()
            results["duration_seconds"] = time.time() - start_time
            raise
