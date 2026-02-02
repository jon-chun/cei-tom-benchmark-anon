"""
REPAIR_DATA stage: Repair invalid emotion labels via semantic mapping.

This stage:
1. Loads error_manifest.csv from data_qa
2. Filters for invalid_emotion errors
3. Applies EMOTION_MAPPING to fix mappable emotions
4. Rotates original files with timestamp backup
5. Updates response files in outputs/main_inference/
6. Generates comprehensive report_repair.md with error rates by model and scenario type

Outputs:
- outputs/data_repair/report_repair.md (combined repair report)
- outputs/data_repair/report_repair_{timestamp}.md (timestamped backup)
- Updated response files in outputs/main_inference/
- Backup files with timestamps
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.emotion_mapping import (
    EMOTION_MAPPING,
    MappingAction,
    map_emotion,
)
from pipeline.execution_context import ExecutionContext
from pipeline.stage_base import PipelineStage, StageResult, StageStatus
from pipeline.stage_registry import register_stage

logger = logging.getLogger(__name__)


@dataclass
class RepairAction:
    """A single repair action taken."""

    file_path: str
    scenario_id: str
    model_id: str
    original_emotion: str
    repaired_emotion: str | None
    action: str  # "mapped", "retry", "unknown"
    rationale: str | None = None


@dataclass
class RepairSummary:
    """Summary of repair operations."""

    total_invalid_emotions: int = 0
    mappable_count: int = 0
    retry_count: int = 0
    unknown_count: int = 0
    files_modified: int = 0
    files_backed_up: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Detailed tracking
    emotions_by_original: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    emotions_by_mapped: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    actions_by_model: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_invalid_emotions": self.total_invalid_emotions,
            "mappable_count": self.mappable_count,
            "retry_count": self.retry_count,
            "unknown_count": self.unknown_count,
            "files_modified": self.files_modified,
            "files_backed_up": self.files_backed_up,
            "emotions_by_original": dict(self.emotions_by_original),
            "emotions_by_mapped": dict(self.emotions_by_mapped),
            "actions_by_model": {k: dict(v) for k, v in self.actions_by_model.items()},
        }


def rotate_file(filepath: Path, backup_dir: Path | None = None) -> Path:
    """Copy file to {name}_{timestamp}.{ext} as backup.

    Args:
        filepath: Path to the file to backup
        backup_dir: Optional directory for backups (default: same directory)

    Returns:
        Path to the backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = filepath.stem
    ext = filepath.suffix

    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"{stem}_{timestamp}{ext}"
    else:
        backup = filepath.parent / f"{stem}_{timestamp}{ext}"

    shutil.copy2(filepath, backup)
    return backup


@register_stage("repair_data")
class RepairDataStage(PipelineStage):
    """
    Repair invalid emotion labels via semantic mapping.

    This stage repairs emotion labels that don't match the valid
    Plutchik emotion set by applying semantic mappings.
    """

    name = "repair_data"
    description = "Repair invalid emotion labels via semantic mapping"
    required_inputs: list[str] = []  # Can run standalone
    produces_outputs = [
        "repair_data.summary",
        "repair_data.actions",
        "repair_data.report_path",
    ]

    def run(self, context: ExecutionContext) -> StageResult:
        """Execute data repair stage."""
        start_time = datetime.now()
        errors: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}

        # Get stage config
        stage_config = context.get_stage_config("repair_data")
        create_backups = stage_config.get("create_backups", True)
        generate_report = stage_config.get("generate_report", True)

        # Get paths
        llm_output_dir = context.get_llm_output_dir()
        qa_dir = llm_output_dir / "data_qa"
        inference_dir = llm_output_dir / "main_inference"
        repair_dir = llm_output_dir / "data_repair"
        repair_dir.mkdir(parents=True, exist_ok=True)

        # Initialize summary
        summary = RepairSummary()
        repair_actions: list[RepairAction] = []

        # Load error manifest
        error_manifest_path = qa_dir / "error_manifest.csv"
        if not error_manifest_path.exists():
            warnings.append("No error_manifest.csv found, running data_qa first recommended")
            return StageResult(
                status=StageStatus.SKIPPED,
                stage_name=self.name,
                start_time=start_time,
                end_time=datetime.now(),
                skip_reason="No error manifest found",
                warnings=warnings,
            )

        # Load QA report for pre-repair stats
        qa_report_path = qa_dir / "qa_report.json"
        pre_repair_stats = self._load_qa_report(qa_report_path)

        # Backup QA files
        if create_backups:
            backup_dir = repair_dir / "backups"
            for qa_file in qa_dir.glob("*.json"):
                rotate_file(qa_file, backup_dir)
                summary.files_backed_up += 1
            for qa_file in qa_dir.glob("*.csv"):
                rotate_file(qa_file, backup_dir)
                summary.files_backed_up += 1
            logger.info(f"Backed up {summary.files_backed_up} QA files")

        # Parse error manifest for invalid_emotion errors
        invalid_emotion_errors = self._parse_error_manifest(error_manifest_path)
        summary.total_invalid_emotions = len(invalid_emotion_errors)

        logger.info(f"Found {len(invalid_emotion_errors)} invalid emotion errors to repair")

        # Group errors by file for efficient processing
        errors_by_file: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for error in invalid_emotion_errors:
            key = (error["model_id"], error["scenario_id"])
            errors_by_file[key].append(error)

        # Process each file
        modified_files: set[Path] = set()
        for (model_id, scenario_id), _file_errors in errors_by_file.items():
            # Find the response file
            model_dir_name = model_id.replace("/", "_")
            response_file = inference_dir / model_dir_name / f"{scenario_id}.json"

            if not response_file.exists():
                warnings.append(f"Response file not found: {response_file}")
                continue

            # Load response
            try:
                with open(response_file) as f:
                    response_data = json.load(f)
            except json.JSONDecodeError as e:
                errors.append(f"Failed to parse {response_file}: {e}")
                continue

            # Get prediction data
            prediction = response_data.get("prediction", {})
            original_emotion = prediction.get("emotion", "")

            if not original_emotion:
                continue

            # Apply mapping
            mapping_result = map_emotion(original_emotion)
            summary.emotions_by_original[original_emotion] += 1

            action = RepairAction(
                file_path=str(response_file),
                scenario_id=scenario_id,
                model_id=model_id,
                original_emotion=original_emotion,
                repaired_emotion=mapping_result.mapped,
                action=mapping_result.action.value,
                rationale=mapping_result.rationale,
            )
            repair_actions.append(action)
            summary.actions_by_model[model_id][mapping_result.action.value] += 1

            if mapping_result.action == MappingAction.MAPPED:
                summary.mappable_count += 1
                summary.emotions_by_mapped[mapping_result.mapped] += 1

                # Update the response file
                if create_backups:
                    rotate_file(response_file, backup_dir)

                prediction["emotion"] = mapping_result.mapped
                prediction["_original_emotion"] = original_emotion
                prediction["_repair_rationale"] = mapping_result.rationale
                response_data["prediction"] = prediction
                response_data["_repaired"] = True
                response_data["_repair_timestamp"] = datetime.now().isoformat()

                with open(response_file, "w") as f:
                    json.dump(response_data, f, indent=2)

                modified_files.add(response_file)
                logger.debug(
                    f"Repaired {model_id}/{scenario_id}: "
                    f"{original_emotion} -> {mapping_result.mapped}"
                )

            elif mapping_result.action == MappingAction.RETRY:
                summary.retry_count += 1
            else:
                summary.unknown_count += 1

        summary.files_modified = len(modified_files)

        # Generate repair report with datetime stamp
        report_path = None
        if generate_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = repair_dir / f"report_repair_{timestamp}.md"
            self._generate_repair_report(
                report_path,
                summary,
                repair_actions,
                pre_repair_stats,
                context,
            )
            # Also create combined report as "report_repair.md" for easy access
            latest_path = repair_dir / "report_repair.md"
            if latest_path.exists():
                latest_path.unlink()
            shutil.copy2(report_path, latest_path)
            logger.info(f"Generated repair report: {report_path}")

        # Set context outputs
        context.set_output("repair_data.summary", summary.to_dict())
        context.set_output(
            "repair_data.actions",
            [
                {
                    "file_path": a.file_path,
                    "scenario_id": a.scenario_id,
                    "model_id": a.model_id,
                    "original_emotion": a.original_emotion,
                    "repaired_emotion": a.repaired_emotion,
                    "action": a.action,
                    "rationale": a.rationale,
                }
                for a in repair_actions
            ],
        )
        if report_path:
            context.set_output("repair_data.report_path", str(report_path))

        # Calculate metrics
        metrics["total_invalid_emotions"] = summary.total_invalid_emotions
        metrics["mappable_repairs"] = summary.mappable_count
        metrics["retry_required"] = summary.retry_count
        metrics["unknown_emotions"] = summary.unknown_count
        metrics["files_modified"] = summary.files_modified
        metrics["files_backed_up"] = summary.files_backed_up
        metrics["repair_rate"] = (
            summary.mappable_count / summary.total_invalid_emotions * 100
            if summary.total_invalid_emotions > 0
            else 0
        )

        # Determine status
        status = StageStatus.COMPLETED
        if errors:
            status = StageStatus.FAILED

        return StageResult(
            status=status,
            stage_name=self.name,
            start_time=start_time,
            end_time=datetime.now(),
            outputs={
                "repair_report": str(report_path) if report_path else None,
                "files_modified": summary.files_modified,
            },
            metrics=metrics,
            errors=errors,
            warnings=warnings,
        )

    def _load_qa_report(self, qa_report_path: Path) -> dict[str, Any]:
        """Load pre-repair QA statistics."""
        if not qa_report_path.exists():
            return {}
        try:
            with open(qa_report_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _parse_error_manifest(self, error_manifest_path: Path) -> list[dict[str, Any]]:
        """Parse error manifest CSV and filter for invalid_emotion errors."""
        invalid_emotions: list[dict[str, Any]] = []

        with open(error_manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("error_type") == "invalid_emotion":
                    # Extract actual emotion value from message
                    message = row.get("message", "")
                    actual_emotion = ""
                    if "Invalid emotion: " in message:
                        actual_emotion = message.replace("Invalid emotion: ", "")

                    invalid_emotions.append(
                        {
                            "scenario_id": row.get("scenario_id", ""),
                            "model_id": row.get("model_id", ""),
                            "error_type": row.get("error_type", ""),
                            "severity": row.get("severity", ""),
                            "field": row.get("field", ""),
                            "message": message,
                            "actual_emotion": actual_emotion,
                        }
                    )

        return invalid_emotions

    def _generate_repair_report(
        self,
        report_path: Path,
        summary: RepairSummary,
        repair_actions: list[RepairAction],
        pre_repair_stats: dict[str, Any],
        context: ExecutionContext,
    ) -> None:
        """Generate comprehensive repair report in Markdown format."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pre_summary = pre_repair_stats.get("summary", {})

        # Count emotions by type for the table
        emotion_counts: dict[str, int] = defaultdict(int)
        for action in repair_actions:
            emotion_counts[action.original_emotion] += 1

        # Sort by count descending
        sorted_emotions = sorted(
            emotion_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Build the report
        report_lines = [
            "# CEI-ToM Data Repair Report",
            "",
            f"Generated: {timestamp}",
            f"Pipeline: {context.pipeline_type}",
            f"Error Threshold: {context.get_stage_config('data_qa').get('error_threshold_percent', 5.0)}%",
            "",
            "---",
            "",
            "## 1. Pre-Repair QA Summary",
            "",
            "### Overall Statistics",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Responses | {pre_summary.get('total_responses', 'N/A')} |",
            f"| Valid Responses | {pre_summary.get('valid_responses', 'N/A')} |",
            f"| Invalid Responses | {pre_summary.get('error_count', 'N/A')} |",
            f"| Error Rate | {pre_summary.get('error_rate_percent', 0):.2f}% |",
            f"| **QA Status** | **{'FAILED' if pre_summary.get('error_rate_percent', 0) > 5.0 else 'PASSED'}** |",
            "",
            "### Pre-Repair Error Rates by Model",
            "| Model | Errors | Error Rate |",
            "|-------|--------|------------|",
        ]

        # Add model error rates
        errors_by_model = pre_summary.get("errors_by_model", {})
        total_responses = pre_summary.get("total_responses", 0)
        num_models = len(errors_by_model) if errors_by_model else 1

        for model_id, error_count in sorted(
            errors_by_model.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            # Estimate per-model total (assume equal distribution)
            model_total = total_responses // num_models if num_models > 0 else 300
            rate = error_count / model_total * 100 if model_total > 0 else 0
            report_lines.append(f"| {model_id} | {error_count} | {rate:.1f}% |")

        report_lines.extend(
            [
                "",
                "### Error Types Distribution",
                "| Error Type | Count | Severity | Repair Action |",
                "|------------|-------|----------|---------------|",
            ]
        )

        errors_by_type = pre_summary.get("errors_by_type", {})
        for error_type, count in sorted(errors_by_type.items(), key=lambda x: x[1], reverse=True):
            action = "Map or Retry" if error_type == "invalid_emotion" else "Retry"
            report_lines.append(f"| {error_type} | {count} | MEDIUM | {action} |")

        # Add errors by scenario type
        report_lines.extend(
            [
                "",
                "### Error Rates by Scenario Type",
                "| Scenario Type | Errors | % of Total |",
                "|---------------|--------|------------|",
            ]
        )

        # Count errors by scenario type from repair actions
        scenario_type_counts: dict[str, int] = defaultdict(int)
        for action in repair_actions:
            if "_" in action.scenario_id:
                scenario_type = action.scenario_id.rsplit("_", 1)[0]
                scenario_type_counts[scenario_type] += 1
            else:
                scenario_type_counts["unknown"] += 1

        total_errors = sum(scenario_type_counts.values()) if scenario_type_counts else 1
        for scenario_type, count in sorted(
            scenario_type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            pct = count / total_errors * 100 if total_errors > 0 else 0
            report_lines.append(f"| {scenario_type} | {count} | {pct:.1f}% |")

        report_lines.extend(
            [
                "",
                "---",
                "",
                "## 2. Invalid Emotion Analysis",
                "",
                f"### 2.1 Invalid Emotions Found (Total: {summary.total_invalid_emotions})",
                "",
                "| Invalid Word | Count | % of Errors | Proposed Mapping |",
                "|--------------|-------|-------------|------------------|",
            ]
        )

        for emotion, count in sorted_emotions:
            pct = (
                count / summary.total_invalid_emotions * 100
                if summary.total_invalid_emotions > 0
                else 0
            )
            mapping_result = map_emotion(emotion)
            if mapping_result.action == MappingAction.MAPPED:
                mapping_str = f"-> {mapping_result.mapped}"
            elif mapping_result.action == MappingAction.RETRY:
                mapping_str = "-> RETRY (indeterminate)"
            else:
                mapping_str = "-> UNKNOWN"
            report_lines.append(f"| {emotion} | {count} | {pct:.1f}% | {mapping_str} |")

        report_lines.extend(
            [
                "",
                "### 2.2 Mapping Rules (Plutchik Emotion Wheel)",
                "",
                "| Invalid Value | Maps To | Rationale |",
                "|---------------|---------|-----------|",
            ]
        )

        for invalid_emotion, (valid_emotion, rationale) in sorted(EMOTION_MAPPING.items()):
            report_lines.append(f"| {invalid_emotion} | {valid_emotion} | {rationale} |")

        report_lines.extend(
            [
                "",
                "### 2.3 Mappable vs Retry Classification",
                "| Category | Count | Action |",
                "|----------|-------|--------|",
                f"| **Mappable** | {summary.mappable_count} | Auto-repair via mapping |",
                f"| **Requires Retry** | {summary.retry_count} | API retry needed |",
                f"| **Unknown** | {summary.unknown_count} | No mapping defined |",
                f"| **Total** | {summary.total_invalid_emotions} | |",
                "",
                "---",
                "",
                "## 3. Repair Actions Taken",
                "",
                "### 3.1 Emotion Mappings Applied",
                "| File | Model | Original | Repaired To |",
                "|------|-------|----------|-------------|",
            ]
        )

        # Show up to 20 mapping examples
        mapped_actions = [a for a in repair_actions if a.action == "mapped"]
        for action in mapped_actions[:20]:
            scenario = action.scenario_id
            model = action.model_id.split("/")[-1] if "/" in action.model_id else action.model_id
            report_lines.append(
                f"| {scenario}.json | {model} | {action.original_emotion} | {action.repaired_emotion} |"
            )
        if len(mapped_actions) > 20:
            report_lines.append(f"| ... | ... | ... | ({len(mapped_actions) - 20} more) |")

        report_lines.extend(
            [
                "",
                "### 3.2 Files Marked for Retry",
                "| Scenario ID | Model | Reason |",
                "|-------------|-------|--------|",
            ]
        )

        retry_actions = [a for a in repair_actions if a.action == "retry"]
        for action in retry_actions[:20]:
            model = action.model_id.split("/")[-1] if "/" in action.model_id else action.model_id
            report_lines.append(
                f"| {action.scenario_id} | {model} | emotion={action.original_emotion} |"
            )
        if len(retry_actions) > 20:
            report_lines.append(f"| ... | ... | ({len(retry_actions) - 20} more) |")

        # Post-repair stats
        post_valid = pre_summary.get("valid_responses", 0) + summary.mappable_count
        post_total = pre_summary.get("total_responses", 3000)
        post_rate = (post_total - post_valid) / post_total * 100 if post_total > 0 else 0
        post_status = "PASSED" if post_rate <= 5.0 else "FAILED"
        pre_error_rate = pre_summary.get("error_rate_percent", 0)
        error_rate_change = pre_error_rate - post_rate

        report_lines.extend(
            [
                "",
                "---",
                "",
                "## 4. Post-Repair QA Summary",
                "",
                "### Overall Statistics (After Mapping Only)",
                "| Metric | Before | After | Change |",
                "|--------|--------|-------|----------|",
                f"| Valid Responses | {pre_summary.get('valid_responses', 'N/A')} | {post_valid} | +{summary.mappable_count} |",
                f"| Error Rate | {pre_error_rate:.2f}% | {post_rate:.2f}% | -{error_rate_change:.2f}% |",
                f"| Mappable Errors Fixed | - | {summary.mappable_count} | - |",
                f"| Remaining (need retry) | - | {summary.retry_count} | - |",
                f"| **QA Status** | FAILED | **{post_status}** | - |",
                "",
                "---",
                "",
                "## 5. File Operations",
                "",
                "### 5.1 Backup Files Created (Rotated)",
                f"Total backups: {summary.files_backed_up}",
                "",
                "### 5.2 Response Files Modified",
                f"Total modified: {summary.files_modified}",
                "",
                "### 5.3 Actions by Model",
                "| Model | Mapped | Retry | Unknown |",
                "|-------|--------|-------|---------|",
            ]
        )

        for model_id, actions in sorted(summary.actions_by_model.items()):
            model_short = model_id.split("/")[-1] if "/" in model_id else model_id
            report_lines.append(
                f"| {model_short} | {actions.get('mapped', 0)} | "
                f"{actions.get('retry', 0)} | {actions.get('unknown', 0)} |"
            )

        # Build recommendation text (avoid f-string escape issues with Python 3.10)
        if summary.retry_count > 0:
            retry_recommendation = (
                "1. **Run --retry** to re-query "
                + str(summary.retry_count)
                + ' "unknown" responses'
            )
        else:
            retry_recommendation = "1. **No retry needed**"

        report_lines.extend(
            [
                "",
                "---",
                "",
                "## 6. Recommendations",
                "",
                retry_recommendation,
                "2. **Review** models with highest error rates for potential prompt improvements",
                "3. **Re-run data_qa** after retry to verify improvements",
                "",
                "---",
                "",
                "*Report generated by CEI-ToM Pipeline*",
            ]
        )

        # Write report
        report_path.write_text("\n".join(report_lines))
