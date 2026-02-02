"""
CEI-ToM Multi-Stage Pipeline Framework.

This package provides a unified pipeline architecture for running
FAccT 2026 and ICML 2026 research paper experiments.

Main Components:
- PipelineStage: Abstract base class for pipeline stages
- StageResult: Result dataclass from stage execution
- ExecutionContext: Shared state container for pipeline
- StageRegistry: Central registry for stage discovery
- PipelineOrchestrator: Main controller for pipeline execution

Usage:
    from pipeline import PipelineOrchestrator, PipelineStage, register_stage

    # Register a custom stage
    @register_stage("custom_stage")
    class CustomStage(PipelineStage):
        name = "custom_stage"
        description = "My custom stage"

        def run(self, context):
            # Stage implementation
            ...

    # Run the pipeline
    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run(
        pipeline_type="facct26",
        mode="full",
        output_dir="outputs",
    )
"""

from pipeline.execution_context import (
    CheckpointData,
    ExecutionContext,
    ExecutionMode,
    PipelineType,
)
from pipeline.orchestrator import (
    ExecutionPlan,
    PipelineOrchestrator,
    PipelineResult,
    run_pipeline,
)
from pipeline.stage_base import (
    PipelineStage,
    StageResult,
    StageStatus,
)
from pipeline.stage_registry import (
    StageRegistry,
    register_stage,
)

# Import stages to trigger registration decorators
import pipeline.stages  # noqa: F401

__all__ = [
    # Stage base
    "PipelineStage",
    "StageResult",
    "StageStatus",
    # Execution context
    "ExecutionContext",
    "CheckpointData",
    "PipelineType",
    "ExecutionMode",
    # Registry
    "StageRegistry",
    "register_stage",
    # Orchestrator
    "PipelineOrchestrator",
    "PipelineResult",
    "ExecutionPlan",
    "run_pipeline",
]
