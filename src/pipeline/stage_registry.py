"""
Pipeline stage registry for stage discovery and management.

This module provides:
- StageRegistry: Central registry for pipeline stages
- @register_stage decorator for stage registration
- Stage discovery by name and pipeline type
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from pipeline.execution_context import PipelineType
    from pipeline.stage_base import PipelineStage

logger = logging.getLogger(__name__)

# Type variable for stage classes
T = TypeVar("T", bound="PipelineStage")


class StageRegistry:
    """Central registry for pipeline stages.

    The registry maintains:
    - Universal stages (shared by all pipelines)
    - Paper-specific stages (facct26, icml26)
    - Stage ordering information

    Usage:
        # Register a universal stage
        @StageRegistry.register("config_test")
        class ConfigTestStage(PipelineStage):
            ...

        # Register a paper-specific stage
        @StageRegistry.register("main_inference", pipeline_type="facct26")
        class FAccTMainInference(PipelineStage):
            ...

        # Get stages for a pipeline
        stages = StageRegistry.get_stages_for_pipeline("facct26")
    """

    # Universal stages (shared across pipelines)
    _universal_stages: dict[str, type[PipelineStage]] = {}

    # Paper-specific stage overrides
    _pipeline_stages: dict[str, dict[str, type[PipelineStage]]] = {
        "facct26": {},
        "icml26": {},
        "venue26": {},
    }

    # Stage execution order (canonical ordering)
    STAGE_ORDER: list[str] = [
        "config_test",
        "pilot_test",
        "main_inference",
        "supplementary_inference",
        "data_qa",
        "repair_data",  # Repair invalid emotion labels via semantic mapping
        "retry_inference",  # Retry failed API calls for specified models
        "fix_data",
        "transform_data",
        "analyze_data",
        "cogsci_analysis",  # CogSci 2026-specific statistical analyses
        "visualize_data",
        "cogsci_visualize",  # CogSci 2026-specific publication figures
        "cot_ablation",  # Chain-of-thought intervention ablation study
    ]

    @classmethod
    def register(
        cls,
        stage_name: str,
        pipeline_type: PipelineType | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register a stage class.

        Args:
            stage_name: Unique name for the stage (must be in STAGE_ORDER)
            pipeline_type: If set, registers as paper-specific override

        Returns:
            Decorator function

        Example:
            @StageRegistry.register("config_test")
            class ConfigTestStage(PipelineStage):
                name = "config_test"
                ...
        """

        def decorator(stage_class: type[T]) -> type[T]:
            # Validate stage name
            if stage_name not in cls.STAGE_ORDER:
                logger.warning(
                    f"Stage '{stage_name}' not in STAGE_ORDER. Valid stages: {cls.STAGE_ORDER}"
                )

            # Register the stage
            if pipeline_type is None:
                cls._universal_stages[stage_name] = stage_class
                logger.debug(f"Registered universal stage: {stage_name}")
            else:
                if pipeline_type not in cls._pipeline_stages:
                    cls._pipeline_stages[pipeline_type] = {}
                cls._pipeline_stages[pipeline_type][stage_name] = stage_class
                logger.debug(f"Registered {pipeline_type} stage: {stage_name}")

            return stage_class

        return decorator

    @classmethod
    def get_stage(
        cls,
        stage_name: str,
        pipeline_type: PipelineType,
    ) -> type[PipelineStage] | None:
        """Get stage class by name, with paper-specific override support.

        Lookup order:
        1. Paper-specific stage (facct26 or icml26)
        2. Universal stage

        Args:
            stage_name: Name of the stage
            pipeline_type: Pipeline type for override lookup

        Returns:
            Stage class or None if not found
        """
        # Check for paper-specific override first
        pipeline_stages = cls._pipeline_stages.get(pipeline_type, {})
        if stage_name in pipeline_stages:
            return pipeline_stages[stage_name]

        # Fall back to universal stage
        return cls._universal_stages.get(stage_name)

    @classmethod
    def get_stages_for_pipeline(
        cls,
        pipeline_type: PipelineType,
    ) -> list[tuple[str, type[PipelineStage]]]:
        """Get all stages for a pipeline in execution order.

        Args:
            pipeline_type: Pipeline type (facct26 or icml26)

        Returns:
            List of (stage_name, stage_class) tuples in order
        """
        stages = []

        for stage_name in cls.STAGE_ORDER:
            stage_class = cls.get_stage(stage_name, pipeline_type)
            if stage_class is not None:
                stages.append((stage_name, stage_class))

        return stages

    @classmethod
    def get_stage_index(cls, stage_name: str) -> int:
        """Get the index of a stage in the execution order.

        Args:
            stage_name: Name of the stage

        Returns:
            Index in STAGE_ORDER, or -1 if not found
        """
        try:
            return cls.STAGE_ORDER.index(stage_name)
        except ValueError:
            return -1

    @classmethod
    def get_stages_in_range(
        cls,
        pipeline_type: PipelineType,
        start: str | None = None,
        end: str | None = None,
    ) -> list[tuple[str, type[PipelineStage]]]:
        """Get stages within a specified range.

        Args:
            pipeline_type: Pipeline type
            start: Starting stage name (inclusive), None for beginning
            end: Ending stage name (inclusive), None for end

        Returns:
            List of (stage_name, stage_class) tuples in range
        """
        all_stages = cls.get_stages_for_pipeline(pipeline_type)
        stage_names = [name for name, _ in all_stages]

        # Determine start index
        if start is None:
            start_idx = 0
        else:
            try:
                start_idx = stage_names.index(start)
            except ValueError:
                logger.error(f"Start stage '{start}' not found")
                return []

        # Determine end index
        if end is None:
            end_idx = len(all_stages)
        else:
            try:
                end_idx = stage_names.index(end) + 1
            except ValueError:
                logger.error(f"End stage '{end}' not found")
                return []

        return all_stages[start_idx:end_idx]

    @classmethod
    def list_registered_stages(cls) -> dict[str, list[str]]:
        """List all registered stages.

        Returns:
            Dictionary with 'universal' and pipeline-specific stage lists
        """
        result = {
            "universal": list(cls._universal_stages.keys()),
            "facct26": list(cls._pipeline_stages.get("facct26", {}).keys()),
            "icml26": list(cls._pipeline_stages.get("icml26", {}).keys()),
            "venue26": list(cls._pipeline_stages.get("venue26", {}).keys()),
        }
        return result

    @classmethod
    def clear(cls) -> None:
        """Clear all registered stages (useful for testing)."""
        cls._universal_stages.clear()
        cls._pipeline_stages = {"facct26": {}, "icml26": {}, "venue26": {}}

    @classmethod
    def is_registered(cls, stage_name: str) -> bool:
        """Check if a stage name is registered.

        Args:
            stage_name: Name of the stage

        Returns:
            True if registered (universal or any pipeline-specific)
        """
        if stage_name in cls._universal_stages:
            return True

        for pipeline_stages in cls._pipeline_stages.values():
            if stage_name in pipeline_stages:
                return True

        return False


# Convenience decorator (module-level alias)
register_stage = StageRegistry.register
