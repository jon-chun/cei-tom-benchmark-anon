"""
Execution context for pipeline state management.

This module provides:
- ExecutionContext: Shared state container for pipeline execution
- Checkpoint/resume functionality
- Inter-stage data passing
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pipeline.stage_base import StageResult

logger = logging.getLogger(__name__)


PipelineType = Literal["facct26", "icml26", "venue26"]
ExecutionMode = Literal["full", "stage", "from", "to", "range"]


@dataclass
class CheckpointData:
    """Data structure for checkpoint persistence.

    Attributes:
        pipeline_type: The type of pipeline (facct26 or icml26)
        execution_mode: The execution mode
        current_stage: Name of the current/last completed stage
        completed_stages: List of completed stage names
        stage_results: Results from each completed stage
        outputs: Accumulated outputs from all stages
        timestamp: When checkpoint was created
    """

    pipeline_type: PipelineType
    execution_mode: str
    current_stage: str | None
    completed_stages: list[str]
    stage_results: dict[str, dict[str, Any]]
    outputs: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pipeline_type": self.pipeline_type,
            "execution_mode": self.execution_mode,
            "current_stage": self.current_stage,
            "completed_stages": self.completed_stages,
            "stage_results": self.stage_results,
            "outputs": self.outputs,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointData:
        """Create from dictionary."""
        return cls(
            pipeline_type=data["pipeline_type"],
            execution_mode=data["execution_mode"],
            current_stage=data.get("current_stage"),
            completed_stages=data.get("completed_stages", []),
            stage_results=data.get("stage_results", {}),
            outputs=data.get("outputs", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


class ExecutionContext:
    """Shared execution context for pipeline stages.

    The ExecutionContext manages:
    - Pipeline configuration
    - Inter-stage data passing via outputs dictionary
    - Checkpoint/resume functionality
    - Stage result tracking
    - Output directory management

    Attributes:
        pipeline_type: Type of pipeline (facct26 or icml26)
        config: Full pipeline configuration
        outputs: Dictionary of outputs from stages
        stage_results: Results from completed stages
    """

    def __init__(
        self,
        pipeline_type: PipelineType,
        config: dict[str, Any],
        output_dir: Path,
        execution_mode: str = "full",
        dry_run: bool = False,
        verbose: bool = True,
    ) -> None:
        """Initialize execution context.

        Args:
            pipeline_type: Type of pipeline (facct26 or icml26)
            config: Full pipeline configuration dictionary
            output_dir: Base output directory for pipeline
            execution_mode: Execution mode string
            dry_run: If True, don't make API calls
            verbose: If True, enable verbose logging
        """
        self.pipeline_type = pipeline_type
        self.config = config
        self.output_dir = Path(output_dir)
        self.execution_mode = execution_mode
        self.dry_run = dry_run
        self.verbose = verbose

        # State management
        self._outputs: dict[str, Any] = {}
        self._stage_results: dict[str, StageResult] = {}
        self._completed_stages: list[str] = []
        self._current_stage: str | None = None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def outputs(self) -> dict[str, Any]:
        """Get all outputs from completed stages."""
        return self._outputs.copy()

    @property
    def completed_stages(self) -> list[str]:
        """Get list of completed stage names."""
        return self._completed_stages.copy()

    @property
    def current_stage(self) -> str | None:
        """Get current stage name."""
        return self._current_stage

    def get_output(self, key: str, default: Any = None) -> Any:
        """Get an output value by key.

        Args:
            key: The output key (can be dotted for nested access)
            default: Default value if key not found

        Returns:
            The output value or default
        """
        parts = key.split(".")
        value = self._outputs

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set_output(self, key: str, value: Any) -> None:
        """Set an output value.

        Args:
            key: The output key (can be dotted for nested access)
            value: The value to store
        """
        parts = key.split(".")
        target = self._outputs

        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        target[parts[-1]] = value

    def has_output(self, key: str) -> bool:
        """Check if an output key exists.

        Args:
            key: The output key to check

        Returns:
            True if key exists, False otherwise
        """
        return self.get_output(key) is not None

    def get_stage_config(self, stage_name: str) -> dict[str, Any]:
        """Get configuration for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Stage configuration dictionary
        """
        stages_config = self.config.get("pipeline", {}).get("stages", {})
        # Convert stage name to config key format (lowercase, underscores)
        config_key = stage_name.lower().replace("-", "_")
        return stages_config.get(config_key, {})

    def get_paper_config(self) -> dict[str, Any]:
        """Get paper-specific configuration.

        Returns:
            Configuration for the current pipeline type (facct26 or icml26)
        """
        return self.config.get(self.pipeline_type, {})

    def get_stage_output_dir(self, stage_name: str) -> Path:
        """Get output directory for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Path to stage output directory
        """
        # Use the LLM output directory based on pipeline type
        llm_output_dir = self.get_llm_output_dir()
        stage_dir = llm_output_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir

    def get_human_data_dir(self) -> Path:
        """Get the human annotation data directory (gold set).

        Returns:
            Path to human annotation data (data-human/gold)
        """
        paths = self.config.get("paths", {})
        human_data_dir = paths.get("human_data_dir", "data-human/gold")
        return Path(human_data_dir)

    def get_llm_output_dir(self) -> Path:
        """Get the LLM output directory for the current pipeline type.

        Returns:
            Path to LLM output directory
        """
        paths = self.config.get("paths", {})

        if self.pipeline_type == "facct26":
            llm_dir = paths.get("facct_llm_dir", "data-facct-llm")
        elif self.pipeline_type == "icml26":
            llm_dir = paths.get("icml_llm_dir", "data-icml-llm")
        elif self.pipeline_type == "venue26":
            # For venue26, use the output_dir directly
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return self.output_dir
        else:
            # Fallback to output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return self.output_dir

        path = Path(llm_dir)
        # If relative path, resolve against output_dir for test isolation
        if not path.is_absolute():
            path = self.output_dir / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_llm_analysis_dir(self) -> Path:
        """Get the LLM analysis output directory for the current pipeline type.

        Returns:
            Path to LLM analysis directory
        """
        paths = self.config.get("paths", {})

        if self.pipeline_type == "facct26":
            analysis_dir = paths.get("facct_llm_analysis_dir", "data-facct-llm-analysis")
        elif self.pipeline_type == "icml26":
            analysis_dir = paths.get("icml_llm_analysis_dir", "data-icml-llm-analysis")
        elif self.pipeline_type == "venue26":
            # For venue26, use output_dir/analysis
            path = self.output_dir / "analysis"
            path.mkdir(parents=True, exist_ok=True)
            return path
        else:
            # Fallback to output_dir/analysis
            path = self.output_dir / "analysis"
            path.mkdir(parents=True, exist_ok=True)
            return path

        path = Path(analysis_dir)
        # If relative path, resolve against output_dir for test isolation
        if not path.is_absolute():
            path = self.output_dir / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_cache_dir(self) -> Path:
        """Get the response cache directory.

        Returns:
            Path to response cache directory
        """
        paths = self.config.get("paths", {})
        cache_dir = paths.get("cache_dir", "data/response_cache")
        path = Path(cache_dir)
        # If relative path, resolve against output_dir for test isolation
        if not path.is_absolute():
            path = self.output_dir / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def mark_stage_started(self, stage_name: str) -> None:
        """Mark a stage as started.

        Args:
            stage_name: Name of the stage
        """
        self._current_stage = stage_name
        logger.info(f"Starting stage: {stage_name}")

    def mark_stage_completed(self, stage_name: str, result: StageResult) -> None:
        """Mark a stage as completed and store its result.

        Args:
            stage_name: Name of the stage
            result: The stage result
        """
        self._stage_results[stage_name] = result
        if stage_name not in self._completed_stages:
            self._completed_stages.append(stage_name)
        self._current_stage = None
        logger.info(
            f"Completed stage: {stage_name} "
            f"(status={result.status.value}, duration={result.duration_seconds:.2f}s)"
        )

    def get_stage_result(self, stage_name: str) -> StageResult | None:
        """Get result for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            StageResult if available, None otherwise
        """
        return self._stage_results.get(stage_name)

    def get_checkpoint_path(self) -> Path:
        """Get path to checkpoint file.

        Returns:
            Path to checkpoint JSON file
        """
        return self.get_llm_output_dir() / "checkpoint.json"

    def save_checkpoint(self) -> None:
        """Save current state to checkpoint file."""
        checkpoint = CheckpointData(
            pipeline_type=self.pipeline_type,
            execution_mode=self.execution_mode,
            current_stage=self._current_stage,
            completed_stages=self._completed_stages,
            stage_results={
                name: result.to_dict() for name, result in self._stage_results.items()
            },
            outputs=self._serialize_outputs(self._outputs),
            timestamp=datetime.now().isoformat(),
        )

        checkpoint_path = self.get_checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write with temporary file
        temp_path = checkpoint_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)
        temp_path.rename(checkpoint_path)

        logger.debug(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self) -> bool:
        """Load state from checkpoint file.

        Returns:
            True if checkpoint was loaded, False if not found
        """
        checkpoint_path = self.get_checkpoint_path()
        if not checkpoint_path.exists():
            return False

        with open(checkpoint_path) as f:
            data = json.load(f)

        checkpoint = CheckpointData.from_dict(data)

        # Verify checkpoint matches current pipeline
        if checkpoint.pipeline_type != self.pipeline_type:
            logger.warning(
                f"Checkpoint pipeline type mismatch: "
                f"{checkpoint.pipeline_type} != {self.pipeline_type}"
            )
            return False

        # Restore state
        self._completed_stages = checkpoint.completed_stages
        self._current_stage = checkpoint.current_stage
        self._outputs = checkpoint.outputs

        # Restore stage results
        from pipeline.stage_base import StageResult

        self._stage_results = {
            name: StageResult.from_dict(result_data)
            for name, result_data in checkpoint.stage_results.items()
        }

        logger.info(
            f"Loaded checkpoint from {checkpoint.timestamp}, "
            f"completed stages: {', '.join(self._completed_stages)}"
        )
        return True

    def clear_checkpoint(self) -> None:
        """Remove checkpoint file."""
        checkpoint_path = self.get_checkpoint_path()
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.debug(f"Removed checkpoint: {checkpoint_path}")

    def _serialize_outputs(self, obj: Any) -> Any:
        """Serialize outputs for JSON storage.

        Handles Path objects and other non-serializable types.
        """
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_outputs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_outputs(v) for v in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    def get_summary(self) -> dict[str, Any]:
        """Get summary of execution context state.

        Returns:
            Dictionary with context summary
        """
        return {
            "pipeline_type": self.pipeline_type,
            "execution_mode": self.execution_mode,
            "output_dir": str(self.output_dir),
            "dry_run": self.dry_run,
            "completed_stages": self._completed_stages,
            "current_stage": self._current_stage,
            "output_count": len(self._outputs),
        }

    def __repr__(self) -> str:
        return (
            f"ExecutionContext(pipeline_type={self.pipeline_type!r}, "
            f"completed={len(self._completed_stages)} stages)"
        )
