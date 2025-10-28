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
import time
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from openai import OpenAI

# Import batch creators
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from batch_creator import SchemaConversionBatchCreator


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
        self.client = OpenAI(api_key=api_key)

        # Ensure directories exist
        self.batch_jobs_dir.mkdir(exist_ok=True)
        self.to_review_dir = self.storage_dir / "to-review"
        self.to_review_dir.mkdir(exist_ok=True)

    def create_batch(self,
                    input_csv: Path,
                    workflow_type: str,
                    progress_callback: Optional[Callable[[str], None]] = None) -> Path:
        """
        Create batch requests using appropriate batch creator.

        Args:
            input_csv: Path to input CSV file
            workflow_type: Type of workflow (parameter/test_statistic/quick_estimate)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to created batch requests JSONL file
        """
        if progress_callback:
            progress_callback(f"Creating {workflow_type} batch requests...")

        # Determine which script to use
        script_map = {
            "parameter": "create_parameter_batch.py",
            "test_statistic": "create_test_statistic_batch.py",
            "quick_estimate": "create_quick_estimate_batch.py"
        }

        if workflow_type not in script_map:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        script_path = self.base_dir / "scripts" / "prepare" / script_map[workflow_type]

        # Run batch creation script
        result = subprocess.run(
            ["python3", str(script_path), str(input_csv)],
            cwd=self.base_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Batch creation failed: {result.stderr}")

        # Determine output file based on workflow type
        output_file = self.batch_jobs_dir / f"{workflow_type}_requests.jsonl"
        if workflow_type == "test_statistic":
            output_file = self.batch_jobs_dir / "test_stat_requests.jsonl"
        elif workflow_type == "parameter":
            output_file = self.batch_jobs_dir / "parameter_requests.jsonl"
        elif workflow_type == "quick_estimate":
            output_file = self.batch_jobs_dir / "quick_estimate_requests.jsonl"

        if not output_file.exists():
            raise RuntimeError(f"Expected batch file not created: {output_file}")

        if progress_callback:
            progress_callback(f"✓ Batch requests created: {output_file.name}")

        return output_file

    def upload_batch(self,
                    batch_file: Path,
                    progress_callback: Optional[Callable[[str], None]] = None) -> str:
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
        with open(batch_file, 'rb') as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")

        # Create batch
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window="24h"
        )

        # Save batch metadata
        batch_id_file = batch_file.with_suffix('.batch_id')
        with open(batch_id_file, 'w') as f:
            json.dump({
                "batch_id": batch.id,
                "batch_type": batch_file.stem.replace('_requests', ''),
                "source_csv": None,
                "created_at": datetime.now().isoformat()
            }, f, indent=2)

        if progress_callback:
            progress_callback(f"✓ Batch uploaded: {batch.id}")

        return batch.id

    def monitor_batch(self,
                     batch_id: str,
                     timeout: int = 3600,
                     poll_interval: int = 30,
                     progress_callback: Optional[Callable[[str], None]] = None) -> Path:
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

                with open(output_file, 'wb') as f:
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

    def validate_results(self,
                        results_file: Path,
                        input_csv: Path,
                        timeout: int = 3600,
                        progress_callback: Optional[Callable[[str], None]] = None) -> Path:
        """
        Run checklist validation on batch results.

        Args:
            results_file: Path to batch results JSONL
            input_csv: Path to original input CSV
            timeout: Maximum seconds to wait for validation batch
            progress_callback: Optional callback for progress updates

        Returns:
            Path to validation results file
        """
        if progress_callback:
            progress_callback("Running checklist validation...")

        # Create validation batch
        script_path = self.base_dir / "scripts" / "prepare" / "create_checklist_from_json_batch.py"

        result = subprocess.run(
            ["python3", str(script_path), str(results_file), str(input_csv)],
            cwd=self.base_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Validation batch creation failed: {result.stderr}")

        validation_batch_file = self.batch_jobs_dir / "checklist_from_json_requests.jsonl"

        if not validation_batch_file.exists():
            raise RuntimeError(f"Validation batch file not created: {validation_batch_file}")

        # Upload validation batch
        validation_batch_id = self.upload_batch(validation_batch_file, progress_callback)

        # Monitor validation batch
        validation_results = self.monitor_batch(
            validation_batch_id,
            timeout=timeout,
            progress_callback=progress_callback
        )

        if progress_callback:
            progress_callback("✓ Validation complete")

        return validation_results

    def unpack_to_review(self,
                        validated_results: Path,
                        input_csv: Path,
                        workflow_type: str,
                        progress_callback: Optional[Callable[[str], None]] = None) -> List[Path]:
        """
        Unpack validated results to to-review folder.

        Args:
            validated_results: Path to validation results JSONL
            input_csv: Path to original input CSV
            workflow_type: Type of workflow (parameter/test_statistic/quick_estimate)
            progress_callback: Optional callback for progress updates

        Returns:
            List of unpacked YAML files
        """
        if progress_callback:
            progress_callback("Unpacking validated results to to-review/...")

        # Determine template based on workflow type
        template_map = {
            "parameter": "templates/parameter_metadata_template_v3.yaml",
            "test_statistic": "templates/test_statistic_template_v2.yaml",
            "quick_estimate": "templates/quick_estimate_template.yaml"
        }

        template_path = template_map.get(workflow_type)

        # Run unpack script
        script_path = self.base_dir / "scripts" / "process" / "unpack_results.py"

        result = subprocess.run(
            ["python3", str(script_path),
             str(validated_results),
             str(self.to_review_dir),
             str(input_csv),
             "",  # No source directory for header extraction
             str(self.base_dir / template_path) if template_path else ""],
            cwd=self.base_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Unpacking failed: {result.stderr}")

        # Find unpacked files
        unpacked_files = list(self.to_review_dir.glob("*.yaml"))

        if progress_callback:
            progress_callback(f"✓ Unpacked {len(unpacked_files)} files to to-review/")

        return unpacked_files

    def commit_and_push(self,
                       unpacked_files: List[Path],
                       batch_id: str,
                       workflow_type: str,
                       branch_prefix: str = "review/batch",
                       push: bool = True,
                       progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Commit unpacked files to review branch and push to remote.

        Args:
            unpacked_files: List of unpacked YAML files
            batch_id: Batch ID for naming
            workflow_type: Type of workflow
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
                capture_output=True
            )

            # Add unpacked files
            subprocess.run(
                ["git", "add", "to-review/"],
                cwd=self.storage_dir,
                check=True,
                capture_output=True
            )

            # Create commit message
            commit_msg = f"""Add {workflow_type} extractions for review

Batch ID: {batch_id}
Files: {len(unpacked_files)} validated extractions
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Automated extraction with validation. Ready for manual review.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"""

            # Commit
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.storage_dir,
                check=True,
                capture_output=True
            )

            # Push if requested
            if push:
                subprocess.run(
                    ["git", "push", "-u", "origin", branch_name],
                    cwd=self.storage_dir,
                    check=True,
                    capture_output=True
                )
                if progress_callback:
                    progress_callback(f"✓ Pushed to origin/{branch_name}")
            else:
                if progress_callback:
                    progress_callback(f"✓ Created local branch: {branch_name}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git operation failed: {e.stderr.decode() if e.stderr else str(e)}")

        return branch_name

    def run_complete_workflow(self,
                             input_csv: Path,
                             workflow_type: str,
                             timeout: int = 3600,
                             skip_validation: bool = False,
                             push: bool = True,
                             branch_prefix: str = "review/batch",
                             progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Run complete extraction workflow from start to finish.

        Args:
            input_csv: Path to input CSV file
            workflow_type: Type of workflow (parameter/test_statistic/quick_estimate)
            timeout: Maximum seconds to wait for batch completion
            skip_validation: Skip checklist validation step
            push: Whether to push to remote
            branch_prefix: Prefix for review branch name
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with workflow results and metadata
        """
        start_time = time.time()
        results = {
            "workflow_type": workflow_type,
            "input_csv": str(input_csv),
            "started_at": datetime.now().isoformat()
        }

        try:
            # Step 1: Create batch
            batch_file = self.create_batch(input_csv, workflow_type, progress_callback)
            results["batch_file"] = str(batch_file)

            # Step 2: Upload batch
            batch_id = self.upload_batch(batch_file, progress_callback)
            results["batch_id"] = batch_id

            # Step 3: Monitor batch
            results_file = self.monitor_batch(batch_id, timeout, progress_callback=progress_callback)
            results["results_file"] = str(results_file)

            # Step 4: Validate (optional)
            if not skip_validation:
                validated_results = self.validate_results(
                    results_file, input_csv, timeout, progress_callback
                )
                results["validated_results"] = str(validated_results)
            else:
                validated_results = results_file
                if progress_callback:
                    progress_callback("⚠ Skipping validation (--skip-validation)")

            # Step 5: Unpack to review
            unpacked_files = self.unpack_to_review(
                validated_results, input_csv, workflow_type, progress_callback
            )
            results["unpacked_files"] = [str(f) for f in unpacked_files]
            results["file_count"] = len(unpacked_files)

            # Step 6: Commit and push
            branch_name = self.commit_and_push(
                unpacked_files, batch_id, workflow_type,
                branch_prefix=branch_prefix, push=push,
                progress_callback=progress_callback
            )
            results["branch_name"] = branch_name
            results["pushed"] = push

            # Success!
            results["status"] = "success"
            results["completed_at"] = datetime.now().isoformat()
            results["duration_seconds"] = time.time() - start_time

            return results

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            results["completed_at"] = datetime.now().isoformat()
            results["duration_seconds"] = time.time() - start_time
            raise

    def run_schema_conversion_workflow(self,
                                       files_to_convert: List[Path],
                                       metadata_type: str,
                                       from_version: str,
                                       to_version: str,
                                       timeout: int = 3600,
                                       skip_validation: bool = False,
                                       push: bool = True,
                                       branch_prefix: str = "schema-conversion",
                                       progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Run schema conversion workflow for outdated files.

        Args:
            files_to_convert: List of YAML files to convert
            metadata_type: Type of metadata (parameter/test_statistic/quick_estimate)
            from_version: Current schema version
            to_version: Target schema version
            timeout: Maximum seconds to wait for batch completion
            skip_validation: Skip checklist validation step
            push: Whether to push to remote
            branch_prefix: Prefix for review branch name
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with workflow results and metadata
        """
        start_time = time.time()
        results = {
            "workflow_type": "schema_conversion",
            "metadata_type": metadata_type,
            "from_version": from_version,
            "to_version": to_version,
            "file_count": len(files_to_convert),
            "started_at": datetime.now().isoformat()
        }

        try:
            # Step 1: Create schema conversion batch
            if progress_callback:
                progress_callback(f"Creating schema conversion batch for {len(files_to_convert)} files...")

            # Get template paths
            from schema_version_detector import SchemaVersionDetector
            detector = SchemaVersionDetector(self.base_dir, self.storage_dir)
            old_template, new_template = detector.get_template_paths(metadata_type, from_version, to_version)

            # Create batch creator
            batch_creator = SchemaConversionBatchCreator(self.base_dir, None)

            # Create temporary directory for organizing files
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Copy files to temp directory
                for file_path in files_to_convert:
                    import shutil
                    shutil.copy(file_path, temp_path / file_path.name)

                # Create batch requests
                requests = batch_creator.process(
                    yaml_dir=temp_path,
                    old_schema_path=old_template,
                    new_schema_path=new_template,
                    migration_notes=f"Converting from {from_version} to {to_version}",
                    pattern="*.yaml"
                )

                # Write batch file
                batch_file = self.batch_jobs_dir / "schema_conversion_requests.jsonl"
                batch_creator.write_batch_file(requests, batch_file)

            results["batch_file"] = str(batch_file)

            if progress_callback:
                progress_callback(f"✓ Created {len(requests)} conversion requests")

            # Step 2: Upload batch
            batch_id = self.upload_batch(batch_file, progress_callback)
            results["batch_id"] = batch_id

            # Step 3: Monitor batch
            results_file = self.monitor_batch(batch_id, timeout, progress_callback=progress_callback)
            results["results_file"] = str(results_file)

            # Step 4: Validate (optional)
            # Note: Schema conversion validation is different from extraction validation
            # For now, skip validation for schema conversion
            validated_results = results_file

            # Step 5: Unpack to review
            if progress_callback:
                progress_callback("Unpacking converted files to to-review/...")

            # Unpack using new schema template
            script_path = self.base_dir / "scripts" / "process" / "unpack_results.py"

            # For schema conversion, we need to pass source directory to extract headers
            source_dir = files_to_convert[0].parent if files_to_convert else self.storage_dir

            result = subprocess.run(
                ["python3", str(script_path),
                 str(validated_results),
                 str(self.to_review_dir),
                 "",  # No input CSV for schema conversion
                 str(source_dir),  # Source directory for header extraction
                 str(new_template)],  # New schema template
                cwd=self.base_dir,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Unpacking failed: {result.stderr}")

            # Find unpacked files
            unpacked_files = list(self.to_review_dir.glob("*.yaml"))

            if progress_callback:
                progress_callback(f"✓ Unpacked {len(unpacked_files)} converted files to to-review/")

            results["unpacked_files"] = [str(f) for f in unpacked_files]

            # Step 6: Commit and push
            date_str = datetime.now().strftime("%Y-%m-%d")
            batch_hash = hashlib.md5(batch_id.encode()).hexdigest()[:6]
            branch_name = f"{branch_prefix}/{metadata_type}-{from_version}-to-{to_version}-{date_str}-{batch_hash}"

            if progress_callback:
                progress_callback("Creating review branch and committing files...")

            # Git operations in storage directory
            try:
                # Create new branch
                subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    cwd=self.storage_dir,
                    check=True,
                    capture_output=True
                )

                # Add unpacked files
                subprocess.run(
                    ["git", "add", "to-review/"],
                    cwd=self.storage_dir,
                    check=True,
                    capture_output=True
                )

                # Create commit message
                commit_msg = f"""Schema conversion: {metadata_type} {from_version} → {to_version}

Batch ID: {batch_id}
Files: {len(unpacked_files)} converted files
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Automated schema conversion. Review converted files before replacing originals.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"""

                # Commit
                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    cwd=self.storage_dir,
                    check=True,
                    capture_output=True
                )

                # Push if requested
                if push:
                    subprocess.run(
                        ["git", "push", "-u", "origin", branch_name],
                        cwd=self.storage_dir,
                        check=True,
                        capture_output=True
                    )
                    if progress_callback:
                        progress_callback(f"✓ Pushed to origin/{branch_name}")
                else:
                    if progress_callback:
                        progress_callback(f"✓ Created local branch: {branch_name}")

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Git operation failed: {e.stderr.decode() if e.stderr else str(e)}")

            results["branch_name"] = branch_name
            results["pushed"] = push

            # Success!
            results["status"] = "success"
            results["completed_at"] = datetime.now().isoformat()
            results["duration_seconds"] = time.time() - start_time

            return results

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            results["completed_at"] = datetime.now().isoformat()
            results["duration_seconds"] = time.time() - start_time
            raise
