"""
Venue 2026 Paper: Pragmatic Inefficiency in Language Models

This module contains the experimental code for the Venue 2026 paper:
"Pragmatic Inefficiency in Language Models: Compositional Integration
Failures Across Social Hierarchies"

Key contributions:
1. Human vs LLM efficiency comparison on pragmatic inference
2. Compositional integration failure diagnosis (utterance anchoring)
3. Power-asymmetric inefficiency across social hierarchies
4. RSA L0/L1 listener comparison
5. Intervention study (prompting strategies)

Modules:
- rsa_comparison: Rational Speech Act L0/L1 alignment analysis
- efficiency_metrics: Pragmatic efficiency computation
- intervention_prompts: Prompting interventions (INT-2, INT-3, INT-5)
- compositional_tests: Cross-subtype recombination tests
- visualization: Publication-ready figures
"""

from venue.rsa_comparison import (
    compute_l0_prediction,
    compute_l1_prediction,
    compute_rsa_alignment,
    RSAComparisonResult,
)
from venue.efficiency_metrics import (
    compute_pragmatic_efficiency,
    compute_utterance_anchoring,
    compute_power_asymmetry,
    expected_calibration_error,
    chi_square_test,
    power_stratified_analysis,
    compute_all_efficiency_metrics,
    cohens_d_with_ci,
    EfficiencyMetric,
    EfficiencyResults,
)
from venue.intervention_prompts import (
    get_intervention_prompt,
    get_all_intervention_prompts,
    analyze_intervention_effects,
    generate_intervention_report,
    INTERVENTION_PROMPTS,
    InterventionResult,
)
from venue.compositional_tests import (
    create_recombined_scenarios,
    create_stratified_recombinations,
    analyze_compositional_generalization,
    compute_compositionality_score,
    generate_compositional_report,
    RecombinedScenario,
    CompositionalResult,
    # Role swap analysis
    create_role_swapped_scenarios,
    analyze_role_swap,
    generate_role_swap_report,
    RoleSwappedScenario,
    RoleSwapResult,
)

__all__ = [
    # RSA comparison
    "compute_l0_prediction",
    "compute_l1_prediction",
    "compute_rsa_alignment",
    "RSAComparisonResult",
    # Efficiency metrics
    "compute_pragmatic_efficiency",
    "compute_utterance_anchoring",
    "compute_power_asymmetry",
    "expected_calibration_error",
    "chi_square_test",
    "power_stratified_analysis",
    "compute_all_efficiency_metrics",
    "cohens_d_with_ci",
    "EfficiencyMetric",
    "EfficiencyResults",
    # Interventions
    "get_intervention_prompt",
    "get_all_intervention_prompts",
    "analyze_intervention_effects",
    "generate_intervention_report",
    "INTERVENTION_PROMPTS",
    "InterventionResult",
    # Compositional tests
    "create_recombined_scenarios",
    "create_stratified_recombinations",
    "analyze_compositional_generalization",
    "compute_compositionality_score",
    "generate_compositional_report",
    "RecombinedScenario",
    "CompositionalResult",
    # Role swap analysis
    "create_role_swapped_scenarios",
    "analyze_role_swap",
    "generate_role_swap_report",
    "RoleSwappedScenario",
    "RoleSwapResult",
]
