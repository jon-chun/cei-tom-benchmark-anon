"""
Core infrastructure for CEI benchmark evaluation.

This module provides shared infrastructure for both ICML 2026 and FAccT 2026 papers:
- Model interface for unified LLM evaluation
- Output caching for API call optimization
- Statistical testing utilities
- Prompt templates for CEI evaluation
"""

from core.model_interface import (
    ModelInterface,
    ModelOutput,
    ModelConfig,
    ModelType,
    get_model,
)
from core.output_cache import OutputCache, CacheKey, compute_prompt_hash, atomic_write_json
from core.statistical_tests import (
    paired_bootstrap_test,
    chi_squared_test,
    bonferroni_correction,
)
from core.prompt_templates import CEI_EVALUATION_PROMPT, format_cei_prompt

__all__ = [
    "ModelInterface",
    "ModelOutput",
    "ModelConfig",
    "ModelType",
    "get_model",
    "OutputCache",
    "CacheKey",
    "compute_prompt_hash",
    "atomic_write_json",
    "paired_bootstrap_test",
    "chi_squared_test",
    "bonferroni_correction",
    "CEI_EVALUATION_PROMPT",
    "format_cei_prompt",
]
