"""
RETRY_INFERENCE stage: Retry failed API calls for specified models.

This stage:
1. Loads retry_plan.json or error_manifest.csv from data_qa
2. Filters by specified models (or all models above threshold)
3. Gets failed scenario IDs per model
4. Retries API calls with exponential backoff + jitter
5. Updates response files on success
6. Rotates and regenerates QA files

Outputs:
- Updated response files in outputs/main_inference/
- outputs/data_repair/retry_report.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.execution_context import ExecutionContext
from pipeline.stage_base import PipelineStage, StageResult, StageStatus
from pipeline.stage_registry import register_stage
from pipeline.stages.repair_data import rotate_file

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry with exponential backoff + jitter."""

    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    jitter_factor: float = 0.25

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff + jitter."""
        delay = min(self.base_delay * (2**attempt), self.max_delay)
        jitter = delay * self.jitter_factor * random.uniform(-1, 1)
        return max(0.1, delay + jitter)


@dataclass
class RetryResult:
    """Result of a retry attempt."""

    scenario_id: str
    model_id: str
    success: bool
    attempts: int
    original_error: str
    new_response: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class RetryProgress:
    """Track retry progress."""

    total_retries: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    skipped_retries: int = 0
    results: list[RetryResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_retries": self.total_retries,
            "successful_retries": self.successful_retries,
            "failed_retries": self.failed_retries,
            "skipped_retries": self.skipped_retries,
            "success_rate": (
                self.successful_retries / self.total_retries * 100 if self.total_retries > 0 else 0
            ),
            "results": [
                {
                    "scenario_id": r.scenario_id,
                    "model_id": r.model_id,
                    "success": r.success,
                    "attempts": r.attempts,
                    "original_error": r.original_error,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


@register_stage("retry_inference")
class RetryInferenceStage(PipelineStage):
    """
    Retry failed API calls for specified models.

    This stage retries inference calls that failed or produced
    invalid responses, targeting specific models or all models
    above an error threshold.
    """

    name = "retry_inference"
    description = "Retry failed API calls for specified models"
    required_inputs: list[str] = []  # Can run standalone
    produces_outputs = [
        "retry_inference.progress",
        "retry_inference.results",
    ]

    def run(self, context: ExecutionContext) -> StageResult:
        """Execute retry inference stage using asyncio."""
        return asyncio.run(self._run_async(context))

    async def _run_async(self, context: ExecutionContext) -> StageResult:
        """Async implementation of retry inference."""
        start_time = datetime.now()
        errors: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}

        # Get stage config
        stage_config = context.get_stage_config("retry_inference")
        retry_config = RetryConfig(
            max_retries=stage_config.get("max_retries", 3),
            base_delay=stage_config.get("base_delay", 2.0),
            max_delay=stage_config.get("max_delay", 60.0),
            jitter_factor=stage_config.get("jitter_factor", 0.25),
        )
        error_threshold = stage_config.get("error_threshold_percent", 5.0)
        create_backups = stage_config.get("create_backups", True)

        # Get retry configuration from context (set by CLI)
        retry_models: list[str] = context.get_output("retry_inference.target_models", [])
        retry_all_failed: bool = context.get_output("retry_inference.retry_all_failed", False)
        retry_all_invalid: bool = context.get_output("retry_inference.retry_all_invalid", False)
        exclude_models: list[str] = context.get_output("retry_inference.exclude_models", [])

        # Get paths
        llm_output_dir = context.get_llm_output_dir()
        qa_dir = llm_output_dir / "data_qa"
        inference_dir = llm_output_dir / "main_inference"
        repair_dir = llm_output_dir / "data_repair"
        repair_dir.mkdir(parents=True, exist_ok=True)

        # Load retry plan
        retry_plan = self._load_retry_plan(qa_dir, inference_dir)

        if not retry_plan:
            warnings.append("No retry items found")
            return StageResult(
                status=StageStatus.SKIPPED,
                stage_name=self.name,
                start_time=start_time,
                end_time=datetime.now(),
                skip_reason="No items to retry",
                warnings=warnings,
            )

        # Apply model exclusions first (always applies)
        if exclude_models:
            original_count = len(retry_plan)
            retry_plan = [item for item in retry_plan if item["model_id"] not in exclude_models]
            excluded_count = original_count - len(retry_plan)
            logger.info(f"Excluded {excluded_count} items from models: {exclude_models}")

        # Filter by specified models
        if retry_models:
            retry_plan = [item for item in retry_plan if item["model_id"] in retry_models]
            logger.info(f"Filtered to {len(retry_plan)} items for models: {retry_models}")

        elif retry_all_invalid:
            # Retry ALL items in the retry plan (no filtering by error rate)
            # Items have already been filtered by exclusions above
            logger.info(f"Retrying all {len(retry_plan)} invalid items (retry_all_invalid=True)")

        elif retry_all_failed:
            # Load model error rates and filter by threshold
            model_error_rates = self._get_model_error_rates(qa_dir)
            high_error_models = [
                model_id for model_id, rate in model_error_rates.items() if rate > error_threshold
            ]
            retry_plan = [item for item in retry_plan if item["model_id"] in high_error_models]
            logger.info(
                f"Filtered to {len(retry_plan)} items for models above "
                f"{error_threshold}% threshold: {high_error_models}"
            )

        if not retry_plan:
            warnings.append("No items match retry criteria")
            return StageResult(
                status=StageStatus.SKIPPED,
                stage_name=self.name,
                start_time=start_time,
                end_time=datetime.now(),
                skip_reason="No items match criteria",
                warnings=warnings,
            )

        logger.info(f"Starting retry for {len(retry_plan)} items")

        # Initialize progress tracking
        progress = RetryProgress(total_retries=len(retry_plan))

        # Load scenarios for prompt generation
        scenarios = self._load_scenarios(context)

        # Process retries
        for item in retry_plan:
            scenario_id = item["scenario_id"]
            model_id = item["model_id"]
            error_types = item.get("error_types", [])

            # Find scenario data
            scenario = scenarios.get(scenario_id)
            if not scenario:
                warnings.append(f"Scenario not found: {scenario_id}")
                progress.skipped_retries += 1
                continue

            # Find response file
            model_dir_name = model_id.replace("/", "_")
            response_file = inference_dir / model_dir_name / f"{scenario_id}.json"

            if context.dry_run:
                # Simulate successful retry
                result = RetryResult(
                    scenario_id=scenario_id,
                    model_id=model_id,
                    success=True,
                    attempts=1,
                    original_error=", ".join(error_types),
                    new_response=self._simulate_prediction(model_id, scenario_id),
                )
            else:
                # Perform actual retry
                result = await self._retry_single_call(
                    scenario=scenario,
                    scenario_id=scenario_id,
                    model_id=model_id,
                    retry_config=retry_config,
                    original_error=", ".join(error_types),
                )

            progress.results.append(result)

            if result.success and result.new_response:
                progress.successful_retries += 1

                # Backup and update response file
                if create_backups and response_file.exists():
                    backup_dir = repair_dir / "backups"
                    rotate_file(response_file, backup_dir)

                # Write new response
                response_file.parent.mkdir(parents=True, exist_ok=True)
                with open(response_file, "w") as f:
                    json.dump(result.new_response, f, indent=2)

                logger.debug(f"Retry succeeded: {model_id}/{scenario_id}")
            else:
                progress.failed_retries += 1
                if result.error:
                    errors.append(f"{model_id}/{scenario_id}: {result.error}")

        # Save retry report with datetime stamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = repair_dir / f"retry_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        "retry_models": retry_models,
                        "retry_all_failed": retry_all_failed,
                        "error_threshold": error_threshold,
                    },
                    "progress": progress.to_dict(),
                },
                f,
                indent=2,
            )
        # Also create a "latest" copy for easy access
        latest_path = repair_dir / "retry_report_latest.json"
        shutil.copy2(report_path, latest_path)

        # Set context outputs
        context.set_output("retry_inference.progress", progress.to_dict())
        context.set_output(
            "retry_inference.results",
            [
                {
                    "scenario_id": r.scenario_id,
                    "model_id": r.model_id,
                    "success": r.success,
                    "attempts": r.attempts,
                }
                for r in progress.results
            ],
        )

        # Calculate metrics
        metrics["total_retries"] = progress.total_retries
        metrics["successful_retries"] = progress.successful_retries
        metrics["failed_retries"] = progress.failed_retries
        metrics["skipped_retries"] = progress.skipped_retries
        metrics["success_rate"] = (
            progress.successful_retries / progress.total_retries * 100
            if progress.total_retries > 0
            else 0
        )

        # Determine status
        if progress.failed_retries == progress.total_retries:
            status = StageStatus.FAILED
        elif progress.failed_retries > 0:
            status = StageStatus.COMPLETED
            warnings.append(f"{progress.failed_retries} retries still failed")
        else:
            status = StageStatus.COMPLETED

        return StageResult(
            status=status,
            stage_name=self.name,
            start_time=start_time,
            end_time=datetime.now(),
            outputs={
                "retry_report": str(report_path),
                "successful_retries": progress.successful_retries,
            },
            metrics=metrics,
            errors=errors[:10],  # Limit error list
            warnings=warnings,
        )

    def _load_retry_plan(
        self,
        qa_dir: Path,
        inference_dir: Path,
    ) -> list[dict[str, Any]]:
        """Load retry plan from QA outputs."""
        retry_plan: list[dict[str, Any]] = []

        # Try retry_plan.json first
        retry_plan_path = qa_dir / "retry_plan.json"
        if retry_plan_path.exists():
            try:
                with open(retry_plan_path) as f:
                    data = json.load(f)
                retry_plan = data.get("items", [])
                logger.info(f"Loaded {len(retry_plan)} items from retry_plan.json")
            except json.JSONDecodeError:
                pass

        # Fall back to error_manifest.csv
        if not retry_plan:
            error_manifest_path = qa_dir / "error_manifest.csv"
            if error_manifest_path.exists():
                import csv

                with open(error_manifest_path, newline="") as f:
                    reader = csv.DictReader(f)
                    # Group by (scenario_id, model_id)
                    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
                    for row in reader:
                        key = (row.get("scenario_id", ""), row.get("model_id", ""))
                        groups[key].append(row.get("error_type", ""))

                    retry_plan = [
                        {
                            "scenario_id": scenario_id,
                            "model_id": model_id,
                            "error_types": error_types,
                        }
                        for (scenario_id, model_id), error_types in groups.items()
                        if scenario_id and model_id and scenario_id != "unknown"
                    ]
                logger.info(f"Loaded {len(retry_plan)} items from error_manifest.csv")

        return retry_plan

    def _get_model_error_rates(self, qa_dir: Path) -> dict[str, float]:
        """Get error rates per model from QA report."""
        model_rates: dict[str, float] = {}

        qa_report_path = qa_dir / "qa_report.json"
        if qa_report_path.exists():
            try:
                with open(qa_report_path) as f:
                    data = json.load(f)
                summary = data.get("summary", {})
                errors_by_model = summary.get("errors_by_model", {})
                total_responses = summary.get("total_responses", 0)
                num_models = len(errors_by_model) if errors_by_model else 1
                per_model_total = total_responses // num_models if num_models > 0 else 300

                for model_id, error_count in errors_by_model.items():
                    rate = error_count / per_model_total * 100 if per_model_total > 0 else 0
                    model_rates[model_id] = rate
            except (json.JSONDecodeError, KeyError):
                pass

        return model_rates

    def _load_scenarios(self, context: ExecutionContext) -> dict[str, dict[str, Any]]:
        """Load scenarios for prompt generation."""
        try:
            from core.data_loader import load_gold_scenarios_as_dicts

            gold_data_path = Path(
                context.config.get("paths", {}).get("gold_data", "data/human-gold-aggregate")
            )
            scenarios_list = load_gold_scenarios_as_dicts(gold_data_path)
            return {s.get("id", s.get("scenario_id", "")): s for s in scenarios_list}
        except ImportError:
            return {}

    async def _retry_single_call(
        self,
        scenario: dict[str, Any],
        scenario_id: str,
        model_id: str,
        retry_config: RetryConfig,
        original_error: str,
    ) -> RetryResult:
        """Retry a single API call with exponential backoff."""
        prompt = self._generate_prompt(scenario)

        for attempt in range(retry_config.max_retries):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    self._run_prediction_sync,
                    model_id,
                    scenario,
                    prompt,
                )

                if response.get("success"):
                    return RetryResult(
                        scenario_id=scenario_id,
                        model_id=model_id,
                        success=True,
                        attempts=attempt + 1,
                        original_error=original_error,
                        new_response=response,
                    )

            except Exception as e:
                logger.warning(
                    f"[{model_id}/{scenario_id}] Retry attempt {attempt + 1} failed: {e}"
                )

                if attempt == retry_config.max_retries - 1:
                    return RetryResult(
                        scenario_id=scenario_id,
                        model_id=model_id,
                        success=False,
                        attempts=attempt + 1,
                        original_error=original_error,
                        error=str(e),
                    )

            # Wait before next retry
            delay = retry_config.get_delay(attempt)
            await asyncio.sleep(delay)

        return RetryResult(
            scenario_id=scenario_id,
            model_id=model_id,
            success=False,
            attempts=retry_config.max_retries,
            original_error=original_error,
            error="Exhausted retries",
        )

    def _run_prediction_sync(
        self,
        model_id: str,
        scenario: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Synchronous prediction wrapper."""
        scenario_id = scenario.get("id", scenario.get("scenario_id", "unknown"))

        try:
            from core.model_interface import get_model

            model = get_model(model_id)
            start = time.time()
            output = model.predict(prompt, return_confidence=True)
            latency_ms = (time.time() - start) * 1000

            return {
                "scenario_id": scenario_id,
                "model_id": model_id,
                "success": True,
                "prediction": {
                    "subtype": output.subtype_prediction,
                    "emotion": output.emotion_prediction,
                    "valence": output.vad_prediction[0],
                    "arousal": output.vad_prediction[1],
                    "dominance": output.vad_prediction[2],
                    "confidence": output.confidence,
                },
                "latency_ms": latency_ms,
                "timestamp": datetime.now().isoformat(),
                "_retried": True,
            }

        except Exception as e:
            return {
                "scenario_id": scenario_id,
                "model_id": model_id,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _generate_prompt(self, scenario: dict[str, Any]) -> str:
        """Generate prompt for a scenario."""
        try:
            from core.prompt_templates import format_cei_prompt

            return format_cei_prompt(
                situation=scenario.get("situation", ""),
                utterance=scenario.get("utterance", ""),
                speaker_role=scenario.get("speaker_role", "speaker"),
                listener_role=scenario.get("listener_role", "listener"),
            )
        except ImportError:
            return (
                f"Situation: {scenario.get('situation', '')}\n"
                f"Utterance: {scenario.get('utterance', '')}"
            )

    def _simulate_prediction(
        self,
        model_id: str,
        scenario_id: str,
    ) -> dict[str, Any]:
        """Simulate a prediction for dry run."""
        return {
            "scenario_id": scenario_id,
            "model_id": model_id,
            "success": True,
            "prediction": {
                "subtype": random.choice(
                    ["sarcasm_irony", "mixed_signals", "strategic_politeness"]
                ),
                "emotion": random.choice(["fear", "trust", "joy", "sadness", "anger"]),
                "valence": random.uniform(-1, 1),
                "arousal": random.uniform(-1, 1),
                "dominance": random.uniform(-1, 1),
                "confidence": random.uniform(0.5, 1.0),
            },
            "latency_ms": random.uniform(100, 500),
            "timestamp": datetime.now().isoformat(),
            "_retried": True,
            "_simulated": True,
        }
