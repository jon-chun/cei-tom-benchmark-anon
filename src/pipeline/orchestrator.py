"""
Pipeline orchestrator for executing multi-stage pipelines.

This module provides:
- PipelineOrchestrator: Main controller for pipeline execution
- Execution mode parsing (full, stage, from, to, range)
- Progress tracking and logging
- Checkpoint/resume support
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.execution_context import ExecutionContext, PipelineType
from pipeline.stage_base import PipelineStage, StageResult, StageStatus
from pipeline.stage_registry import StageRegistry

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Plan for pipeline execution.

    Attributes:
        stages: List of (stage_name, stage_class) tuples to execute
        mode: Original execution mode string
        resume_from: Stage to resume from (if resuming)
    """

    stages: list[tuple[str, type[PipelineStage]]]
    mode: str
    resume_from: str | None = None


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    Attributes:
        success: Whether pipeline completed successfully
        stages_executed: Number of stages executed
        stages_failed: Number of stages that failed
        stages_skipped: Number of stages skipped
        total_duration_seconds: Total execution time
        stage_results: Dictionary of stage name to result
        errors: List of error messages
    """

    success: bool
    stages_executed: int
    stages_failed: int
    stages_skipped: int
    total_duration_seconds: float
    stage_results: dict[str, StageResult]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "stages_executed": self.stages_executed,
            "stages_failed": self.stages_failed,
            "stages_skipped": self.stages_skipped,
            "total_duration_seconds": self.total_duration_seconds,
            "stage_results": {
                name: result.to_dict() for name, result in self.stage_results.items()
            },
            "errors": self.errors,
        }


class PipelineOrchestrator:
    """Main controller for pipeline execution.

    The orchestrator manages:
    - Execution plan creation from mode strings
    - Stage execution in order
    - Progress tracking and logging
    - Checkpoint/resume functionality
    - Error handling and recovery

    Usage:
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run(
            pipeline_type="facct26",
            mode="full",
            output_dir=Path("outputs"),
        )
    """

    def __init__(
        self,
        config: dict[str, Any],
        verbose: bool = True,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config: Full pipeline configuration dictionary
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        self._context: ExecutionContext | None = None

    def parse_execution_mode(
        self,
        mode: str,
        pipeline_type: PipelineType,
    ) -> ExecutionPlan:
        """Parse execution mode string into execution plan.

        Supported modes:
        - "full": Run all stages
        - "stage:<name>": Run single stage only
        - "from:<name>": Run from stage to end
        - "to:<name>": Run from start to stage
        - "range:<start>:<end>": Run stage range

        Args:
            mode: Mode string
            pipeline_type: Pipeline type for stage lookup

        Returns:
            ExecutionPlan with stages to execute
        """
        mode = mode.strip().lower()

        if mode == "full":
            stages = StageRegistry.get_stages_for_pipeline(pipeline_type)
            return ExecutionPlan(stages=stages, mode=mode)

        # Parse mode patterns
        stage_match = re.match(r"^stage:(\w+)$", mode)
        if stage_match:
            stage_name = stage_match.group(1)
            stage_class = StageRegistry.get_stage(stage_name, pipeline_type)
            if stage_class is None:
                raise ValueError(f"Stage not found: {stage_name}")
            return ExecutionPlan(stages=[(stage_name, stage_class)], mode=mode)

        from_match = re.match(r"^from:(\w+)$", mode)
        if from_match:
            start_stage = from_match.group(1)
            stages = StageRegistry.get_stages_in_range(pipeline_type, start=start_stage, end=None)
            if not stages:
                raise ValueError(f"No stages found from: {start_stage}")
            return ExecutionPlan(stages=stages, mode=mode)

        to_match = re.match(r"^to:(\w+)$", mode)
        if to_match:
            end_stage = to_match.group(1)
            stages = StageRegistry.get_stages_in_range(pipeline_type, start=None, end=end_stage)
            if not stages:
                raise ValueError(f"No stages found to: {end_stage}")
            return ExecutionPlan(stages=stages, mode=mode)

        range_match = re.match(r"^range:(\w+):(\w+)$", mode)
        if range_match:
            start_stage = range_match.group(1)
            end_stage = range_match.group(2)
            stages = StageRegistry.get_stages_in_range(
                pipeline_type, start=start_stage, end=end_stage
            )
            if not stages:
                raise ValueError(f"No stages found in range: {start_stage}:{end_stage}")
            return ExecutionPlan(stages=stages, mode=mode)

        raise ValueError(
            f"Invalid execution mode: {mode}. "
            "Valid formats: full, stage:<name>, from:<name>, to:<name>, range:<start>:<end>"
        )

    def run(
        self,
        pipeline_type: PipelineType,
        mode: str = "full",
        output_dir: Path | str = Path("outputs"),
        dry_run: bool = False,
        resume: bool = False,
        stop_on_failure: bool = True,
        repair_mode: bool = False,
        retry_models: list[str] | None = None,
        retry_all_failed: bool = False,
        retry_all_invalid: bool = False,
        exclude_models: list[str] | None = None,
    ) -> PipelineResult:
        """Execute the pipeline.

        Args:
            pipeline_type: Type of pipeline (facct26 or icml26)
            mode: Execution mode string
            output_dir: Base output directory
            dry_run: If True, don't make actual API calls
            resume: If True, attempt to resume from checkpoint
            stop_on_failure: If True, stop on first stage failure
            repair_mode: If True, run repair data stage
            retry_models: List of specific model IDs to retry
            retry_all_failed: If True, retry all models above error threshold
            retry_all_invalid: If True, retry all individual invalid responses
            exclude_models: List of model IDs to exclude from retry

        Returns:
            PipelineResult with execution summary
        """
        start_time = datetime.now()
        output_dir = Path(output_dir)

        # Create execution context
        self._context = ExecutionContext(
            pipeline_type=pipeline_type,
            config=self.config,
            output_dir=output_dir,
            execution_mode=mode,
            dry_run=dry_run,
            verbose=self.verbose,
        )

        # Set repair/retry configuration in context outputs for stages to access
        if repair_mode:
            self._context.set_output("repair_data.enabled", True)
        if retry_models:
            self._context.set_output("retry_inference.target_models", retry_models)
        if retry_all_failed:
            self._context.set_output("retry_inference.retry_all_failed", True)
        if retry_all_invalid:
            self._context.set_output("retry_inference.retry_all_invalid", True)
        if exclude_models:
            self._context.set_output("retry_inference.exclude_models", exclude_models)

        # Handle resume
        if resume and self._context.load_checkpoint():
            logger.info("Resuming from checkpoint")
        else:
            self._context.clear_checkpoint()

        # Parse execution mode
        try:
            plan = self.parse_execution_mode(mode, pipeline_type)
        except ValueError as e:
            return PipelineResult(
                success=False,
                stages_executed=0,
                stages_failed=1,
                stages_skipped=0,
                total_duration_seconds=0.0,
                stage_results={},
                errors=[str(e)],
            )

        logger.info(f"Starting {pipeline_type} pipeline (mode={mode}, stages={len(plan.stages)})")

        # Execute stages
        stage_results: dict[str, StageResult] = {}
        errors: list[str] = []
        stages_executed = 0
        stages_failed = 0
        stages_skipped = 0

        for stage_name, stage_class in plan.stages:
            # Skip already completed stages (when resuming)
            if stage_name in self._context.completed_stages:
                logger.info(f"Skipping completed stage: {stage_name}")
                result = self._context.get_stage_result(stage_name)
                if result:
                    stage_results[stage_name] = result
                stages_skipped += 1
                continue

            # Create and execute stage
            stage_config = self._context.get_stage_config(stage_name)
            stage = stage_class(config=stage_config)

            self._log_stage_start(stage_name, stage)
            self._context.mark_stage_started(stage_name)

            result = stage.execute(self._context)

            self._context.mark_stage_completed(stage_name, result)
            stage_results[stage_name] = result

            # Save checkpoint after each stage
            self._context.save_checkpoint()

            self._log_stage_result(stage_name, result)

            # Track execution stats
            if result.status == StageStatus.COMPLETED:
                stages_executed += 1
            elif result.status == StageStatus.FAILED:
                stages_failed += 1
                errors.extend(result.errors)
                if stop_on_failure:
                    logger.error(f"Pipeline stopped due to failure in {stage_name}")
                    break
            elif result.status == StageStatus.SKIPPED:
                stages_skipped += 1

        # Calculate total duration
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Determine success
        success = stages_failed == 0

        if success:
            logger.info(
                f"Pipeline completed successfully "
                f"({stages_executed} stages in {total_duration:.1f}s)"
            )
            self._context.clear_checkpoint()
        else:
            logger.error(f"Pipeline failed ({stages_failed} failures, {stages_executed} completed)")

        return PipelineResult(
            success=success,
            stages_executed=stages_executed,
            stages_failed=stages_failed,
            stages_skipped=stages_skipped,
            total_duration_seconds=total_duration,
            stage_results=stage_results,
            errors=errors,
        )

    def _log_stage_start(self, stage_name: str, stage: PipelineStage) -> None:
        """Log stage start."""
        if self.verbose:
            logger.info(f"{'=' * 60}")
            logger.info(f"Stage: {stage_name}")
            logger.info(f"Description: {stage.description}")
            logger.info(f"{'=' * 60}")

    def _log_stage_result(self, stage_name: str, result: StageResult) -> None:
        """Log stage result."""
        status_emoji = {
            StageStatus.COMPLETED: "✓",
            StageStatus.FAILED: "✗",
            StageStatus.SKIPPED: "○",
            StageStatus.PENDING: "?",
            StageStatus.RUNNING: "→",
        }
        emoji = status_emoji.get(result.status, "?")

        if self.verbose:
            logger.info(
                f"{emoji} {stage_name}: {result.status.value} ({result.duration_seconds:.2f}s)"
            )

            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"  ⚠ {warning}")

            if result.errors:
                for error in result.errors:
                    logger.error(f"  ✗ {error}")

    @property
    def context(self) -> ExecutionContext | None:
        """Get the current execution context."""
        return self._context

    def get_available_stages(
        self,
        pipeline_type: PipelineType,
    ) -> list[dict[str, Any]]:
        """Get list of available stages for a pipeline.

        Args:
            pipeline_type: Pipeline type

        Returns:
            List of stage info dictionaries
        """
        stages = StageRegistry.get_stages_for_pipeline(pipeline_type)
        return [
            {
                "name": name,
                "class": cls.__name__,
                "description": getattr(cls, "description", ""),
            }
            for name, cls in stages
        ]


def run_pipeline(
    pipeline_type: PipelineType,
    config: dict[str, Any],
    mode: str = "full",
    output_dir: Path | str = Path("outputs"),
    dry_run: bool = False,
    resume: bool = False,
    verbose: bool = True,
    repair_mode: bool = False,
    retry_models: list[str] | None = None,
    retry_all_failed: bool = False,
    retry_all_invalid: bool = False,
    exclude_models: list[str] | None = None,
) -> PipelineResult:
    """Convenience function to run a pipeline.

    Args:
        pipeline_type: Type of pipeline (facct26 or icml26)
        config: Pipeline configuration dictionary
        mode: Execution mode string
        output_dir: Base output directory
        dry_run: If True, don't make actual API calls
        resume: If True, attempt to resume from checkpoint
        verbose: Enable verbose logging
        repair_mode: If True, run repair data stage
        retry_models: List of specific model IDs to retry
        retry_all_failed: If True, retry all models above error threshold
        retry_all_invalid: If True, retry all individual invalid responses
        exclude_models: List of model IDs to exclude from retry

    Returns:
        PipelineResult with execution summary
    """
    orchestrator = PipelineOrchestrator(config=config, verbose=verbose)
    return orchestrator.run(
        pipeline_type=pipeline_type,
        mode=mode,
        output_dir=output_dir,
        dry_run=dry_run,
        resume=resume,
        repair_mode=repair_mode,
        retry_models=retry_models,
        retry_all_failed=retry_all_failed,
        retry_all_invalid=retry_all_invalid,
        exclude_models=exclude_models,
    )
