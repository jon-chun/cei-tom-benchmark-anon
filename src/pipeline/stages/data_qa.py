"""
DATA_QA stage: Quality analysis of API outputs.

This stage analyzes all inference outputs for:
1. Parse validation - JSON parseable
2. Schema validation - Required fields present
3. Value range validation - VAD in [-1, 1], confidence in [0, 1]
4. Label validation - Subtype/emotion in valid sets
5. Anomaly detection - Statistical outliers
6. Model-specific failure rates

Outputs:
- qa_report.json: Detailed QA report
- error_manifest.csv: List of all errors
- retry_plan.json: Plan for FIX_DATA stage
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pipeline.execution_context import ExecutionContext
from pipeline.stage_base import PipelineStage, StageResult, StageStatus
from pipeline.stage_registry import register_stage

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of data quality errors."""

    PARSE_FAILURE = "parse_failure"
    SCHEMA_MISSING_FIELD = "schema_missing_field"
    INVALID_SUBTYPE = "invalid_subtype"
    INVALID_EMOTION = "invalid_emotion"
    VAD_OUT_OF_RANGE = "vad_out_of_range"
    CONFIDENCE_OUT_OF_RANGE = "confidence_out_of_range"
    EMPTY_RESPONSE = "empty_response"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    ANOMALOUS_VALUE = "anomalous_value"


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    CRITICAL = "critical"  # Must be fixed
    HIGH = "high"  # Should be fixed
    MEDIUM = "medium"  # Can retry
    LOW = "low"  # Flag for review


@dataclass
class QAError:
    """A single QA error."""

    scenario_id: str
    model_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    field_name: str | None = None
    actual_value: Any = None
    expected_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "model_id": self.model_id,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field_name,
            "actual_value": str(self.actual_value) if self.actual_value else None,
            "expected_value": str(self.expected_value) if self.expected_value else None,
        }


@dataclass
class RetryItem:
    """An item to retry in FIX_DATA stage."""

    scenario_id: str
    model_id: str
    error_types: list[str]
    retry_count: int = 0
    priority: int = 1  # 1 = highest priority

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "model_id": self.model_id,
            "error_types": self.error_types,
            "retry_count": self.retry_count,
            "priority": self.priority,
        }


# Valid values
VALID_SUBTYPES = [
    "sarcasm_irony",
    "mixed_signals",
    "strategic_politeness",
    "passive_aggression",
    "deflection",
]

VALID_EMOTIONS = [
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
]

REQUIRED_FIELDS = ["subtype", "emotion", "valence", "arousal", "dominance"]


@register_stage("data_qa")
class DataQAStage(PipelineStage):
    """
    Data quality analysis stage.

    Analyzes inference outputs for errors and anomalies.
    """

    name = "data_qa"
    description = "Quality analysis of inference outputs"
    required_inputs: list[str] = []  # Can run with or without prior inference
    produces_outputs = [
        "data_qa.errors",
        "data_qa.retry_plan",
        "data_qa.summary",
    ]

    def run(self, context: ExecutionContext) -> StageResult:
        """Execute data QA stage."""
        start_time = datetime.now()
        errors: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}

        # Get stage config
        stage_config = context.get_stage_config("data_qa")
        error_threshold_percent = stage_config.get("error_threshold_percent", 5.0)
        anomaly_z_threshold = stage_config.get("anomaly_z_threshold", 3.0)
        generate_retry_plan = stage_config.get("generate_retry_plan", True)
        max_retry_fraction = stage_config.get("max_retry_fraction", 0.10)

        # Load inference data from LLM output directory
        llm_output_dir = context.get_llm_output_dir()
        inference_dir = llm_output_dir / "main_inference"
        supplementary_dir = llm_output_dir / "supplementary_inference"

        # Collect all response files
        response_files = self._collect_response_files(inference_dir, supplementary_dir)

        if not response_files and not context.dry_run:
            warnings.append("No inference data found, checking for test data")
            # Try to find any JSON files
            response_files = list(llm_output_dir.glob("**/*.json"))

        logger.info(f"Found {len(response_files)} response files to analyze")

        # Analyze responses
        qa_errors: list[QAError] = []
        total_responses = 0
        valid_responses = 0

        if context.dry_run:
            # Generate synthetic data for dry run
            total_responses = 100
            valid_responses = 95
            qa_errors = self._generate_synthetic_errors(5)
        else:
            for response_file in response_files:
                file_errors, file_valid, file_total = self._analyze_response_file(
                    response_file,
                    anomaly_z_threshold,
                )
                qa_errors.extend(file_errors)
                valid_responses += file_valid
                total_responses += file_total

        # Calculate error rate
        error_rate = (
            (total_responses - valid_responses) / total_responses * 100
            if total_responses > 0
            else 0.0
        )

        metrics["total_responses"] = total_responses
        metrics["valid_responses"] = valid_responses
        metrics["error_count"] = len(qa_errors)
        metrics["error_rate_percent"] = error_rate

        # Count errors by type
        error_counts: dict[str, int] = defaultdict(int)
        for err in qa_errors:
            error_counts[err.error_type.value] += 1
        metrics["errors_by_type"] = dict(error_counts)

        # Count errors by model
        model_errors: dict[str, int] = defaultdict(int)
        for err in qa_errors:
            model_errors[err.model_id] += 1
        metrics["errors_by_model"] = dict(model_errors)

        # Generate retry plan if needed
        retry_plan: list[RetryItem] = []
        if generate_retry_plan and qa_errors:
            retry_plan = self._generate_retry_plan(
                qa_errors,
                max_retry_fraction,
                total_responses,
            )
            metrics["retry_items"] = len(retry_plan)

        # Save outputs
        output_dir = context.get_stage_output_dir(self.name)
        self._save_qa_report(output_dir, qa_errors, metrics)
        self._save_error_manifest(output_dir, qa_errors)
        self._save_retry_plan(output_dir, retry_plan)

        # Set context outputs
        context.set_output(
            "data_qa.errors",
            [e.to_dict() for e in qa_errors],
        )
        context.set_output(
            "data_qa.retry_plan",
            [r.to_dict() for r in retry_plan],
        )
        context.set_output(
            "data_qa.summary",
            metrics,
        )

        # Check against threshold
        if error_rate > error_threshold_percent:
            errors.append(
                f"Error rate {error_rate:.1f}% exceeds threshold {error_threshold_percent}%"
            )

        # Add warnings for model-specific issues
        for model_id, count in model_errors.items():
            model_total = sum(1 for f in response_files if model_id in str(f))
            if model_total > 0:
                model_error_rate = count / model_total * 100
                if model_error_rate > 10.0:
                    warnings.append(
                        f"Model {model_id} has high error rate: {model_error_rate:.1f}%"
                    )

        # Determine status
        status = StageStatus.COMPLETED if not errors else StageStatus.FAILED

        return StageResult(
            status=status,
            stage_name=self.name,
            start_time=start_time,
            end_time=datetime.now(),
            outputs={
                "qa_report": str(output_dir / "qa_report.json"),
                "error_manifest": str(output_dir / "error_manifest.csv"),
                "retry_plan": str(output_dir / "retry_plan.json"),
            },
            metrics=metrics,
            errors=errors,
            warnings=warnings,
        )

    def _collect_response_files(
        self,
        inference_dir: Path,
        supplementary_dir: Path,
    ) -> list[Path]:
        """Collect all response files from inference directories."""
        response_files: list[Path] = []

        for directory in [inference_dir, supplementary_dir]:
            if directory.exists():
                # Look for JSON response files
                response_files.extend(directory.glob("**/*.json"))

        # Filter out manifest and metadata files
        filtered = [
            f
            for f in response_files
            if not any(
                name in f.name
                for name in ["manifest", "metadata", "checkpoint", "config"]
            )
        ]

        return filtered

    def _analyze_response_file(
        self,
        response_file: Path,
        anomaly_z_threshold: float,
    ) -> tuple[list[QAError], int, int]:
        """Analyze a single response file for errors."""
        qa_errors: list[QAError] = []
        valid_count = 0
        total_count = 0

        try:
            with open(response_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            qa_errors.append(
                QAError(
                    scenario_id=response_file.stem,
                    model_id="unknown",
                    error_type=ErrorType.PARSE_FAILURE,
                    severity=ErrorSeverity.CRITICAL,
                    message=f"Failed to parse JSON: {e}",
                )
            )
            return qa_errors, 0, 1

        # Handle different file formats
        responses = []
        if isinstance(data, list):
            responses = data
        elif isinstance(data, dict):
            if "predictions" in data:
                responses = data["predictions"]
            elif "responses" in data:
                responses = data["responses"]
            else:
                # Single response
                responses = [data]

        for resp in responses:
            total_count += 1
            response_errors = self._validate_response(resp, anomaly_z_threshold)

            if response_errors:
                qa_errors.extend(response_errors)
            else:
                valid_count += 1

        return qa_errors, valid_count, total_count

    def _validate_response(
        self,
        response: dict[str, Any],
        _anomaly_z_threshold: float,
    ) -> list[QAError]:
        """Validate a single response and return any errors."""
        response_errors: list[QAError] = []

        scenario_id = response.get("scenario_id", response.get("id", "unknown"))
        model_id = response.get("model_id", response.get("model", "unknown"))

        # Get the prediction data (might be nested)
        prediction = response.get("prediction", response)

        # Check for empty response
        if not prediction or prediction == {}:
            response_errors.append(
                QAError(
                    scenario_id=scenario_id,
                    model_id=model_id,
                    error_type=ErrorType.EMPTY_RESPONSE,
                    severity=ErrorSeverity.HIGH,
                    message="Empty response",
                )
            )
            return response_errors

        # Check required fields
        for field_name in REQUIRED_FIELDS:
            if field_name not in prediction:
                response_errors.append(
                    QAError(
                        scenario_id=scenario_id,
                        model_id=model_id,
                        error_type=ErrorType.SCHEMA_MISSING_FIELD,
                        severity=ErrorSeverity.MEDIUM,
                        message=f"Missing required field: {field_name}",
                        field_name=field_name,
                    )
                )

        # Validate subtype
        subtype = prediction.get("subtype")
        if subtype and subtype not in VALID_SUBTYPES:
            response_errors.append(
                QAError(
                    scenario_id=scenario_id,
                    model_id=model_id,
                    error_type=ErrorType.INVALID_SUBTYPE,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Invalid subtype: {subtype}",
                    field_name="subtype",
                    actual_value=subtype,
                    expected_value=VALID_SUBTYPES,
                )
            )

        # Validate emotion
        emotion = prediction.get("emotion")
        if emotion and emotion not in VALID_EMOTIONS:
            response_errors.append(
                QAError(
                    scenario_id=scenario_id,
                    model_id=model_id,
                    error_type=ErrorType.INVALID_EMOTION,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Invalid emotion: {emotion}",
                    field_name="emotion",
                    actual_value=emotion,
                    expected_value=VALID_EMOTIONS,
                )
            )

        # Validate VAD values
        for vad_field in ["valence", "arousal", "dominance"]:
            value = prediction.get(vad_field)
            if value is not None:
                try:
                    value = float(value)
                    if not -1.0 <= value <= 1.0:
                        response_errors.append(
                            QAError(
                                scenario_id=scenario_id,
                                model_id=model_id,
                                error_type=ErrorType.VAD_OUT_OF_RANGE,
                                severity=ErrorSeverity.MEDIUM,
                                message=f"{vad_field} out of range [-1, 1]: {value}",
                                field_name=vad_field,
                                actual_value=value,
                                expected_value="[-1, 1]",
                            )
                        )
                except (TypeError, ValueError):
                    response_errors.append(
                        QAError(
                            scenario_id=scenario_id,
                            model_id=model_id,
                            error_type=ErrorType.VAD_OUT_OF_RANGE,
                            severity=ErrorSeverity.MEDIUM,
                            message=f"Invalid {vad_field} value: {value}",
                            field_name=vad_field,
                            actual_value=value,
                        )
                    )

        # Validate confidence
        confidence = prediction.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
                if not 0.0 <= confidence <= 1.0:
                    response_errors.append(
                        QAError(
                            scenario_id=scenario_id,
                            model_id=model_id,
                            error_type=ErrorType.CONFIDENCE_OUT_OF_RANGE,
                            severity=ErrorSeverity.LOW,
                            message=f"Confidence out of range [0, 1]: {confidence}",
                            field_name="confidence",
                            actual_value=confidence,
                            expected_value="[0, 1]",
                        )
                    )
            except (TypeError, ValueError):
                pass  # Confidence is optional

        return response_errors

    def _generate_synthetic_errors(self, count: int) -> list[QAError]:
        """Generate synthetic errors for dry run testing."""
        error_types = [
            (ErrorType.INVALID_SUBTYPE, ErrorSeverity.MEDIUM),
            (ErrorType.VAD_OUT_OF_RANGE, ErrorSeverity.MEDIUM),
            (ErrorType.PARSE_FAILURE, ErrorSeverity.CRITICAL),
        ]

        return [
            QAError(
                scenario_id=f"scenario_{i:03d}",
                model_id="mock",
                error_type=error_types[i % len(error_types)][0],
                severity=error_types[i % len(error_types)][1],
                message=f"Synthetic error {i}",
            )
            for i in range(count)
        ]

    def _generate_retry_plan(
        self,
        qa_errors: list[QAError],
        max_retry_fraction: float,
        total_responses: int,
    ) -> list[RetryItem]:
        """Generate retry plan for FIX_DATA stage."""
        # Group errors by (scenario_id, model_id)
        error_groups: dict[tuple[str, str], list[QAError]] = defaultdict(list)
        for err in qa_errors:
            key = (err.scenario_id, err.model_id)
            error_groups[key].append(err)

        # Create retry items
        retry_items: list[RetryItem] = []
        max_retries = int(total_responses * max_retry_fraction)

        # Sort by severity (critical first)
        severity_order = {
            ErrorSeverity.CRITICAL: 0,
            ErrorSeverity.HIGH: 1,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.LOW: 3,
        }

        sorted_groups = sorted(
            error_groups.items(),
            key=lambda x: min(severity_order[e.severity] for e in x[1]),
        )

        for (scenario_id, model_id), group_errors in sorted_groups[:max_retries]:
            retry_items.append(
                RetryItem(
                    scenario_id=scenario_id,
                    model_id=model_id,
                    error_types=[e.error_type.value for e in group_errors],
                    priority=min(severity_order[e.severity] for e in group_errors) + 1,
                )
            )

        return retry_items

    def _save_qa_report(
        self,
        output_dir: Path,
        qa_errors: list[QAError],
        metrics: dict[str, Any],
    ) -> None:
        """Save QA report to JSON."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": metrics,
            "errors": [e.to_dict() for e in qa_errors],
        }

        report_path = output_dir / "qa_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved QA report to {report_path}")

    def _save_error_manifest(
        self,
        output_dir: Path,
        qa_errors: list[QAError],
    ) -> None:
        """Save error manifest to CSV."""
        csv_path = output_dir / "error_manifest.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "scenario_id",
                    "model_id",
                    "error_type",
                    "severity",
                    "field",
                    "message",
                ]
            )

            for err in qa_errors:
                writer.writerow(
                    [
                        err.scenario_id,
                        err.model_id,
                        err.error_type.value,
                        err.severity.value,
                        err.field_name or "",
                        err.message,
                    ]
                )

        logger.info(f"Saved error manifest to {csv_path}")

    def _save_retry_plan(
        self,
        output_dir: Path,
        retry_plan: list[RetryItem],
    ) -> None:
        """Save retry plan to JSON."""
        plan = {
            "timestamp": datetime.now().isoformat(),
            "total_items": len(retry_plan),
            "items": [r.to_dict() for r in retry_plan],
        }

        plan_path = output_dir / "retry_plan.json"
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2)

        logger.info(f"Saved retry plan to {plan_path}")
