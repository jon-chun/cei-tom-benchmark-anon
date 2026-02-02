"""
Pipeline stage base classes and data models.

This module provides:
- StageResult: Dataclass for stage execution results
- StageStatus: Enum for stage execution status
- PipelineStage: Abstract base class for all pipeline stages
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.execution_context import ExecutionContext


class StageStatus(Enum):
    """Execution status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of a pipeline stage execution.

    Attributes:
        status: The final status of the stage
        stage_name: Name of the stage that produced this result
        start_time: When the stage started executing
        end_time: When the stage finished executing
        duration_seconds: Total execution time in seconds
        outputs: Dictionary of output artifacts produced
        metrics: Dictionary of metrics collected during execution
        errors: List of error messages if any
        warnings: List of warning messages
        skip_reason: Reason for skipping if status is SKIPPED
    """

    status: StageStatus
    stage_name: str
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skip_reason: str | None = None

    def is_success(self) -> bool:
        """Check if the stage completed successfully."""
        return self.status == StageStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "status": self.status.value,
            "stage_name": self.stage_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings,
            "skip_reason": self.skip_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageResult:
        """Create result from dictionary."""
        return cls(
            status=StageStatus(data["status"]),
            stage_name=data["stage_name"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=(
                datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None
            ),
            duration_seconds=data.get("duration_seconds", 0.0),
            outputs=data.get("outputs", {}),
            metrics=data.get("metrics", {}),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            skip_reason=data.get("skip_reason"),
        )


class PipelineStage(ABC):
    """Abstract base class for pipeline stages.

    All pipeline stages must inherit from this class and implement
    the abstract methods.

    Lifecycle:
        1. __init__: Stage initialization with config
        2. validate_inputs(): Check prerequisites are met
        3. pre_run(): Setup before main execution
        4. run(): Main stage execution (abstract)
        5. post_run(): Cleanup after execution
        6. on_error(): Handle any errors

    Attributes:
        name: Unique identifier for the stage
        description: Human-readable description
        required_inputs: List of required input keys from context
        produces_outputs: List of output keys this stage produces
    """

    name: str = "base_stage"
    description: str = "Base pipeline stage"
    required_inputs: list[str] = []
    produces_outputs: list[str] = []

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the stage with configuration.

        Args:
            config: Stage-specific configuration dictionary
        """
        self.config = config or {}
        self._start_time: float | None = None
        self._result: StageResult | None = None

    @property
    def is_enabled(self) -> bool:
        """Check if stage is enabled in config."""
        return self.config.get("enabled", True)

    def validate_inputs(self, context: ExecutionContext) -> list[str]:
        """Validate that all required inputs are available.

        Args:
            context: The execution context

        Returns:
            List of missing input keys (empty if all present)
        """
        missing = []
        for input_key in self.required_inputs:
            if not context.has_output(input_key):
                missing.append(input_key)
        return missing

    def pre_run(self, context: ExecutionContext) -> None:  # noqa: B027
        """Setup before main execution.

        Override this method to perform any setup needed before run().
        Default implementation does nothing.

        Args:
            context: The execution context
        """

    @abstractmethod
    def run(self, context: ExecutionContext) -> StageResult:
        """Execute the main stage logic.

        This method must be implemented by all stage subclasses.

        Args:
            context: The execution context with shared state

        Returns:
            StageResult with execution status and outputs
        """
        pass

    def post_run(self, context: ExecutionContext, result: StageResult) -> None:  # noqa: B027
        """Cleanup after execution.

        Override this method to perform any cleanup after run().
        Called regardless of success or failure.

        Args:
            context: The execution context
            result: The result from run()
        """

    def on_error(self, _context: ExecutionContext, error: Exception) -> StageResult:
        """Handle errors during stage execution.

        Override this method to customize error handling.
        Default implementation creates a FAILED result.

        Args:
            _context: The execution context (unused in default implementation)
            error: The exception that occurred

        Returns:
            StageResult with FAILED status
        """
        return StageResult(
            status=StageStatus.FAILED,
            stage_name=self.name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            errors=[str(error)],
        )

    def execute(self, context: ExecutionContext) -> StageResult:
        """Execute the full stage lifecycle.

        This method orchestrates the stage lifecycle:
        1. Check if enabled
        2. Validate inputs
        3. Run pre_run hook
        4. Run main logic
        5. Run post_run hook

        Args:
            context: The execution context

        Returns:
            StageResult from execution
        """
        start_time = datetime.now()
        self._start_time = time.time()

        # Check if stage is enabled
        if not self.is_enabled:
            return StageResult(
                status=StageStatus.SKIPPED,
                stage_name=self.name,
                start_time=start_time,
                end_time=datetime.now(),
                skip_reason="Stage disabled in configuration",
            )

        # Validate inputs
        missing = self.validate_inputs(context)
        if missing:
            return StageResult(
                status=StageStatus.FAILED,
                stage_name=self.name,
                start_time=start_time,
                end_time=datetime.now(),
                errors=[f"Missing required inputs: {', '.join(missing)}"],
            )

        try:
            # Pre-run hook
            self.pre_run(context)

            # Main execution
            result = self.run(context)

            # Post-run hook
            self.post_run(context, result)

            # Update timing
            result.end_time = datetime.now()
            result.duration_seconds = time.time() - self._start_time

            self._result = result
            return result

        except Exception as e:
            result = self.on_error(context, e)
            result.duration_seconds = time.time() - self._start_time
            self._result = result
            return result

    def get_output_path(self, context: ExecutionContext, filename: str) -> Path:
        """Get the output path for a file within this stage's output directory.

        Args:
            context: The execution context
            filename: Name of the output file

        Returns:
            Full path to the output file
        """
        return context.get_stage_output_dir(self.name) / filename

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, enabled={self.is_enabled})"
