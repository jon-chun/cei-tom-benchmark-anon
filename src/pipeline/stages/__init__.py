"""
Pipeline stage implementations.

This package contains the implementation of all pipeline stages:
- config_test: Configuration and API connectivity validation
- pilot_test: Small-scale pilot execution
- main_inference: Primary RQ API calls
- supplementary_inference: Ablation experiments
- data_qa: Output quality analysis
- repair_data: Repair invalid emotion labels via semantic mapping
- retry_inference: Retry failed API calls for specified models
- fix_data: Error retry and repair
- transform_data: Data normalization
- analyze_data: Statistical analysis
- cogsci_analysis: CogSci 2026 statistical analyses (anger bias, correlation, kappa)
- visualize_data: Figure and table generation
- cogsci_visualize: CogSci 2026 publication figures
- cot_ablation: Chain-of-thought intervention ablation study (optional)

Stages are automatically registered when imported.
"""

from pipeline.stages.analyze_data import AnalyzeDataStage
from pipeline.stages.cogsci_analysis import CogSciAnalysisStage
from pipeline.stages.cogsci_visualize import CogSciVisualizeStage
from pipeline.stages.config_test import ConfigTestStage
from pipeline.stages.cot_ablation import CoTAblationStage
from pipeline.stages.data_qa import DataQAStage
from pipeline.stages.fix_data import FixDataStage
from pipeline.stages.main_inference import MainInferenceStage
from pipeline.stages.pilot_test import PilotTestStage
from pipeline.stages.repair_data import RepairDataStage
from pipeline.stages.retry_inference import RetryInferenceStage
from pipeline.stages.supplementary_inference import SupplementaryInferenceStage
from pipeline.stages.transform_data import TransformDataStage
from pipeline.stages.visualize_data import VisualizeDataStage

__all__ = [
    "ConfigTestStage",
    "PilotTestStage",
    "MainInferenceStage",
    "SupplementaryInferenceStage",
    "DataQAStage",
    "RepairDataStage",
    "RetryInferenceStage",
    "FixDataStage",
    "TransformDataStage",
    "AnalyzeDataStage",
    "CogSciAnalysisStage",
    "VisualizeDataStage",
    "CogSciVisualizeStage",
    "CoTAblationStage",
]
