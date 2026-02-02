"""
Compositional Generalization Tests for Venue 2026.

This module implements tests for compositional generalization:
whether LLMs can correctly interpret novel combinations of
contexts and utterances.

Key tests (Section 4.3 of paper):
1. Cross-subtype recombination: Pair utterances from one pragmatic
   subtype with contexts from another
2. Role swap analysis: Swap speaker/listener roles to test
   sensitivity to power dynamics

Hypothesis: LLMs show "utterance anchoring" - they over-rely on
surface patterns in utterances and fail to integrate contextual cues.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RecombinedScenario:
    """A scenario created by cross-subtype recombination."""

    # Original source IDs
    utterance_source_id: str
    context_source_id: str

    # Recombined content
    situation: str
    speaker_role: str
    listener_role: str
    utterance: str

    # Subtype information
    utterance_original_subtype: str
    context_original_subtype: str

    # Expected interpretation after recombination
    # (depends on whether context or utterance dominates)
    expected_subtype: str | None = None

    # Metadata
    recombination_type: str = "cross_subtype"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "utterance_source_id": self.utterance_source_id,
            "context_source_id": self.context_source_id,
            "situation": self.situation,
            "speaker_role": self.speaker_role,
            "listener_role": self.listener_role,
            "utterance": self.utterance,
            "utterance_original_subtype": self.utterance_original_subtype,
            "context_original_subtype": self.context_original_subtype,
            "expected_subtype": self.expected_subtype,
            "recombination_type": self.recombination_type,
        }


@dataclass
class CompositionalResult:
    """Results from compositional generalization test."""

    # Core metrics
    original_accuracy: float = 0.0
    recombined_accuracy: float = 0.0
    compositional_drop: float = 0.0

    # Detailed breakdowns
    by_recombination: dict[str, dict[str, float]] = field(default_factory=dict)
    utterance_anchoring_rate: float = 0.0
    context_sensitivity_rate: float = 0.0

    # Sample sizes
    n_original: int = 0
    n_recombined: int = 0

    # Confidence intervals
    original_ci: tuple[float, float] = (0.0, 0.0)
    recombined_ci: tuple[float, float] = (0.0, 0.0)
    drop_ci: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_accuracy": self.original_accuracy,
            "recombined_accuracy": self.recombined_accuracy,
            "compositional_drop": self.compositional_drop,
            "by_recombination": self.by_recombination,
            "utterance_anchoring_rate": self.utterance_anchoring_rate,
            "context_sensitivity_rate": self.context_sensitivity_rate,
            "n_original": self.n_original,
            "n_recombined": self.n_recombined,
            "confidence_intervals": {
                "original": self.original_ci,
                "recombined": self.recombined_ci,
                "drop": self.drop_ci,
            },
        }


@dataclass
class RoleSwappedScenario:
    """A scenario with speaker and listener roles swapped."""

    # Original scenario ID
    original_id: str

    # Swapped content
    situation: str
    speaker_role: str  # Was listener_role in original
    listener_role: str  # Was speaker_role in original
    utterance: str

    # Original roles for reference
    original_speaker_role: str
    original_listener_role: str

    # Power relation changes
    original_power_relation: str
    swapped_power_relation: str

    # Ground truth (unchanged - same scenario)
    subtype: str
    emotion: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_id": self.original_id,
            "situation": self.situation,
            "speaker_role": self.speaker_role,
            "listener_role": self.listener_role,
            "utterance": self.utterance,
            "original_speaker_role": self.original_speaker_role,
            "original_listener_role": self.original_listener_role,
            "original_power_relation": self.original_power_relation,
            "swapped_power_relation": self.swapped_power_relation,
            "subtype": self.subtype,
            "emotion": self.emotion,
        }


@dataclass
class RoleSwapResult:
    """Results from role swap analysis."""

    # Core metrics
    original_accuracy: float = 0.0
    swapped_accuracy: float = 0.0
    accuracy_drop: float = 0.0

    # Effect size
    cohens_d: float = 0.0
    cohens_d_ci: tuple[float, float] = (0.0, 0.0)

    # Prediction stability
    prediction_changed_rate: float = 0.0
    appropriate_change_rate: float = 0.0

    # By power transition
    by_power_transition: dict[str, dict[str, float]] = field(default_factory=dict)

    # Sample sizes
    n_scenarios: int = 0

    # Confidence intervals
    original_ci: tuple[float, float] = (0.0, 0.0)
    swapped_ci: tuple[float, float] = (0.0, 0.0)
    drop_ci: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_accuracy": self.original_accuracy,
            "swapped_accuracy": self.swapped_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "cohens_d": self.cohens_d,
            "cohens_d_ci": self.cohens_d_ci,
            "prediction_changed_rate": self.prediction_changed_rate,
            "appropriate_change_rate": self.appropriate_change_rate,
            "by_power_transition": self.by_power_transition,
            "n_scenarios": self.n_scenarios,
            "confidence_intervals": {
                "original": self.original_ci,
                "swapped": self.swapped_ci,
                "drop": self.drop_ci,
            },
        }


# Pragmatic subtypes for recombination
SUBTYPES = [
    "sarcasm-irony",
    "mixed-signals",
    "passive-aggression",
    "deflection-misdirection",
    "strategic-politeness",
]


def create_recombined_scenarios(
    scenarios: list[dict[str, Any]],
    n_recombinations: int | None = None,
    seed: int = 42,
) -> list[RecombinedScenario]:
    """
    Create recombined scenarios by cross-subtype pairing.

    For each scenario, pairs its utterance with a context from a
    different subtype to test compositional integration.

    Args:
        scenarios: Original scenarios with subtype labels
        n_recombinations: Number of recombined scenarios to generate
                          (default: same as original)
        seed: Random seed for reproducibility

    Returns:
        List of RecombinedScenario objects
    """
    random.seed(seed)
    np.random.seed(seed)

    if n_recombinations is None:
        n_recombinations = len(scenarios)

    # Group scenarios by subtype
    by_subtype: dict[str, list[dict[str, Any]]] = {st: [] for st in SUBTYPES}
    for s in scenarios:
        subtype = s.get("subtype", s.get("ground_truth", {}).get("subtype", ""))
        if subtype in by_subtype:
            by_subtype[subtype].append(s)

    recombined = []

    # Generate recombinations
    for _ in range(n_recombinations):
        # Pick two different subtypes
        st1, st2 = random.sample([st for st in SUBTYPES if by_subtype[st]], 2)

        # Get utterance from st1, context from st2
        utt_scenario = random.choice(by_subtype[st1])
        ctx_scenario = random.choice(by_subtype[st2])

        # Extract fields
        utterance = utt_scenario.get("utterance", utt_scenario.get("sd_utterance", ""))
        situation = ctx_scenario.get("situation", ctx_scenario.get("sd_situation", ""))
        speaker_role = ctx_scenario.get("speaker_role", ctx_scenario.get("sd_speaker_role", ""))
        listener_role = ctx_scenario.get("listener_role", ctx_scenario.get("sd_listener_role", ""))

        # Create recombined scenario
        recombined.append(RecombinedScenario(
            utterance_source_id=utt_scenario.get("id", utt_scenario.get("scenario_id", "")),
            context_source_id=ctx_scenario.get("id", ctx_scenario.get("scenario_id", "")),
            situation=situation,
            speaker_role=speaker_role,
            listener_role=listener_role,
            utterance=utterance,
            utterance_original_subtype=st1,
            context_original_subtype=st2,
            # Expected subtype is ambiguous - could be either
            # We'll assess based on which the model chooses
            expected_subtype=None,
        ))

    return recombined


def create_stratified_recombinations(
    scenarios: list[dict[str, Any]],
    per_pair: int = 10,
    seed: int = 42,
) -> list[RecombinedScenario]:
    """
    Create stratified recombinations with equal representation.

    Generates recombinations for each pair of subtypes to ensure
    balanced coverage.

    Args:
        scenarios: Original scenarios
        per_pair: Number of recombinations per subtype pair
        seed: Random seed

    Returns:
        List of RecombinedScenario objects
    """
    random.seed(seed)

    # Group by subtype
    by_subtype: dict[str, list[dict[str, Any]]] = {st: [] for st in SUBTYPES}
    for s in scenarios:
        subtype = s.get("subtype", s.get("ground_truth", {}).get("subtype", ""))
        if subtype in by_subtype:
            by_subtype[subtype].append(s)

    recombined = []

    # Generate for each pair
    for i, st1 in enumerate(SUBTYPES):
        for st2 in SUBTYPES[i + 1:]:
            if not by_subtype[st1] or not by_subtype[st2]:
                continue

            for _ in range(per_pair):
                # st1 utterance + st2 context
                utt = random.choice(by_subtype[st1])
                ctx = random.choice(by_subtype[st2])

                recombined.append(RecombinedScenario(
                    utterance_source_id=utt.get("id", ""),
                    context_source_id=ctx.get("id", ""),
                    situation=ctx.get("situation", ctx.get("sd_situation", "")),
                    speaker_role=ctx.get("speaker_role", ctx.get("sd_speaker_role", "")),
                    listener_role=ctx.get("listener_role", ctx.get("sd_listener_role", "")),
                    utterance=utt.get("utterance", utt.get("sd_utterance", "")),
                    utterance_original_subtype=st1,
                    context_original_subtype=st2,
                ))

                # st2 utterance + st1 context
                utt = random.choice(by_subtype[st2])
                ctx = random.choice(by_subtype[st1])

                recombined.append(RecombinedScenario(
                    utterance_source_id=utt.get("id", ""),
                    context_source_id=ctx.get("id", ""),
                    situation=ctx.get("situation", ctx.get("sd_situation", "")),
                    speaker_role=ctx.get("speaker_role", ctx.get("sd_speaker_role", "")),
                    listener_role=ctx.get("listener_role", ctx.get("sd_listener_role", "")),
                    utterance=utt.get("utterance", utt.get("sd_utterance", "")),
                    utterance_original_subtype=st2,
                    context_original_subtype=st1,
                ))

    return recombined


def analyze_compositional_generalization(
    original_predictions: list[dict[str, Any]],
    original_ground_truth: list[dict[str, Any]],
    recombined_predictions: list[dict[str, Any]],
    recombined_metadata: list[RecombinedScenario],
    n_bootstrap: int = 10000,
) -> CompositionalResult:
    """
    Analyze compositional generalization performance.

    Computes:
    - Original vs recombined accuracy
    - Compositional drop (decrease in accuracy)
    - Utterance anchoring rate (predictions following utterance subtype)
    - Context sensitivity (predictions following context subtype)

    Args:
        original_predictions: Predictions on original scenarios
        original_ground_truth: Ground truth for original scenarios
        recombined_predictions: Predictions on recombined scenarios
        recombined_metadata: Metadata about recombinations
        n_bootstrap: Bootstrap resamples for CI

    Returns:
        CompositionalResult with all metrics
    """
    result = CompositionalResult()

    # Compute original accuracy
    orig_correct = []
    for pred, truth in zip(original_predictions, original_ground_truth):
        pred_label = pred.get("subtype", pred.get("prediction", {}).get("subtype", ""))
        true_label = truth.get("subtype", truth.get("ground_truth", {}).get("subtype", ""))
        orig_correct.append(1 if pred_label == true_label else 0)

    result.original_accuracy = np.mean(orig_correct) if orig_correct else 0
    result.n_original = len(orig_correct)
    result.original_ci = bootstrap_ci(orig_correct, n_bootstrap) if orig_correct else (0, 0)

    # Analyze recombined predictions
    utterance_matches = []
    context_matches = []

    by_pair: dict[str, dict[str, list[int]]] = {}

    for pred, meta in zip(recombined_predictions, recombined_metadata):
        pred_label = pred.get("subtype", pred.get("prediction", {}).get("subtype", ""))

        # Does prediction match utterance's original subtype?
        utt_match = 1 if pred_label == meta.utterance_original_subtype else 0
        utterance_matches.append(utt_match)

        # Does prediction match context's subtype?
        ctx_match = 1 if pred_label == meta.context_original_subtype else 0
        context_matches.append(ctx_match)

        # Track by recombination pair
        pair_key = f"{meta.utterance_original_subtype}→{meta.context_original_subtype}"
        if pair_key not in by_pair:
            by_pair[pair_key] = {"utterance_match": [], "context_match": []}
        by_pair[pair_key]["utterance_match"].append(utt_match)
        by_pair[pair_key]["context_match"].append(ctx_match)

    # Utterance anchoring = fraction that follow utterance subtype
    result.utterance_anchoring_rate = np.mean(utterance_matches) if utterance_matches else 0
    result.context_sensitivity_rate = np.mean(context_matches) if context_matches else 0

    # For recombined, we consider correct if it matches either source
    # (since interpretation is legitimately ambiguous)
    recomb_correct = [u or c for u, c in zip(utterance_matches, context_matches)]
    result.recombined_accuracy = np.mean(recomb_correct) if recomb_correct else 0
    result.n_recombined = len(recomb_correct)
    result.recombined_ci = bootstrap_ci(recomb_correct, n_bootstrap) if recomb_correct else (0, 0)

    # Compositional drop
    result.compositional_drop = result.original_accuracy - result.recombined_accuracy

    # Bootstrap CI for drop
    if orig_correct and recomb_correct:
        drops = []
        for _ in range(n_bootstrap):
            orig_sample = np.random.choice(orig_correct, size=len(orig_correct), replace=True)
            recomb_sample = np.random.choice(recomb_correct, size=len(recomb_correct), replace=True)
            drops.append(np.mean(orig_sample) - np.mean(recomb_sample))
        drops = np.sort(drops)
        result.drop_ci = (drops[int(0.025 * n_bootstrap)], drops[int(0.975 * n_bootstrap)])

    # Breakdown by recombination pair
    for pair_key, data in by_pair.items():
        result.by_recombination[pair_key] = {
            "utterance_anchoring": np.mean(data["utterance_match"]),
            "context_sensitivity": np.mean(data["context_match"]),
            "n": len(data["utterance_match"]),
        }

    return result


def compute_compositionality_score(
    result: CompositionalResult,
) -> float:
    """
    Compute a single compositionality score.

    Score = 1.0 means perfect compositionality (no drop)
    Score < 1.0 indicates compositional failure
    Score = 0.0 means recombined accuracy is 0

    Args:
        result: CompositionalResult from analysis

    Returns:
        Compositionality score in [0, 1]
    """
    if result.original_accuracy == 0:
        return 0.0

    return result.recombined_accuracy / result.original_accuracy


def generate_compositional_report(
    result: CompositionalResult,
) -> str:
    """
    Generate a text report of compositional generalization results.

    Args:
        result: CompositionalResult from analysis

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "COMPOSITIONAL GENERALIZATION RESULTS",
        "=" * 60,
        "",
        "OVERALL METRICS",
        "-" * 40,
        f"Original Accuracy:    {result.original_accuracy:.1%} "
        f"[{result.original_ci[0]:.1%}, {result.original_ci[1]:.1%}]",
        f"Recombined Accuracy:  {result.recombined_accuracy:.1%} "
        f"[{result.recombined_ci[0]:.1%}, {result.recombined_ci[1]:.1%}]",
        f"Compositional Drop:   {result.compositional_drop:.1%} "
        f"[{result.drop_ci[0]:.1%}, {result.drop_ci[1]:.1%}]",
        "",
        f"Compositionality Score: {compute_compositionality_score(result):.2f}",
        "",
        "ANCHORING ANALYSIS",
        "-" * 40,
        f"Utterance Anchoring Rate: {result.utterance_anchoring_rate:.1%}",
        f"Context Sensitivity Rate: {result.context_sensitivity_rate:.1%}",
        "",
    ]

    # Interpretation
    if result.utterance_anchoring_rate > result.context_sensitivity_rate + 0.1:
        lines.append("⚠ Model exhibits UTTERANCE ANCHORING")
        lines.append("  (over-relies on surface patterns in utterances)")
    elif result.context_sensitivity_rate > result.utterance_anchoring_rate + 0.1:
        lines.append("✓ Model shows CONTEXT SENSITIVITY")
        lines.append("  (appropriately integrates contextual cues)")
    else:
        lines.append("○ Balanced utterance/context integration")

    lines.extend([
        "",
        "BY RECOMBINATION PAIR",
        "-" * 40,
        f"{'Pair':<35} {'Utt%':<8} {'Ctx%':<8} {'n':<6}",
    ])

    for pair, data in sorted(result.by_recombination.items()):
        lines.append(
            f"{pair:<35} "
            f"{data['utterance_anchoring']:.1%}    "
            f"{data['context_sensitivity']:.1%}    "
            f"{data['n']}"
        )

    return "\n".join(lines)


def bootstrap_ci(
    data: list[int | float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for mean."""
    if len(data) < 2:
        return (0.0, 1.0)

    means = []
    data_array = np.array(data)
    n = len(data)

    for _ in range(n_resamples):
        sample = np.random.choice(data_array, size=n, replace=True)
        means.append(np.mean(sample))

    means = np.sort(means)
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_resamples)
    upper_idx = int((1 - alpha / 2) * n_resamples)

    return (means[lower_idx], means[min(upper_idx, len(means) - 1)])


# =============================================================================
# ROLE SWAP ANALYSIS
# =============================================================================

def infer_power_relation(speaker_role: str, listener_role: str) -> str:
    """Infer power relation from speaker and listener roles."""
    higher_terms = ["manager", "supervisor", "director", "ceo", "boss",
                    "senior", "lead", "head", "chief", "partner", "executive"]
    lower_terms = ["junior", "intern", "assistant", "associate", "employee",
                   "subordinate", "trainee", "new", "entry"]

    speaker_lower = speaker_role.lower()
    listener_lower = listener_role.lower()

    speaker_is_higher = any(t in speaker_lower for t in higher_terms)
    speaker_is_lower = any(t in speaker_lower for t in lower_terms)
    listener_is_higher = any(t in listener_lower for t in higher_terms)
    listener_is_lower = any(t in listener_lower for t in lower_terms)

    if speaker_is_lower and listener_is_higher:
        return "lower_to_higher"
    elif speaker_is_higher and listener_is_lower:
        return "higher_to_lower"
    else:
        return "peer"


def create_role_swapped_scenarios(
    scenarios: list[dict[str, Any]],
) -> list[RoleSwappedScenario]:
    """
    Create role-swapped versions of scenarios.

    Swaps speaker_role and listener_role to test whether models
    appropriately adjust interpretations based on power dynamics.

    Args:
        scenarios: Original scenarios with role information

    Returns:
        List of RoleSwappedScenario objects
    """
    swapped = []

    for s in scenarios:
        # Extract original roles
        orig_speaker = s.get("speaker_role", s.get("sd_speaker_role", "speaker"))
        orig_listener = s.get("listener_role", s.get("sd_listener_role", "listener"))

        # Compute power relations
        orig_power = infer_power_relation(orig_speaker, orig_listener)
        swapped_power = infer_power_relation(orig_listener, orig_speaker)

        # Extract other fields
        situation = s.get("situation", s.get("sd_situation", ""))
        utterance = s.get("utterance", s.get("sd_utterance", ""))
        subtype = s.get("subtype", s.get("ground_truth", {}).get("subtype", ""))
        emotion = s.get("emotion", s.get("gold_standard", s.get("ground_truth", {}).get("emotion", "")))
        scenario_id = s.get("id", s.get("scenario_id", ""))

        swapped.append(RoleSwappedScenario(
            original_id=scenario_id,
            situation=situation,
            speaker_role=orig_listener,  # Swapped
            listener_role=orig_speaker,  # Swapped
            utterance=utterance,
            original_speaker_role=orig_speaker,
            original_listener_role=orig_listener,
            original_power_relation=orig_power,
            swapped_power_relation=swapped_power,
            subtype=subtype,
            emotion=emotion,
        ))

    return swapped


def analyze_role_swap(
    original_predictions: list[dict[str, Any]],
    swapped_predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    swapped_metadata: list[RoleSwappedScenario],
    n_bootstrap: int = 10000,
) -> RoleSwapResult:
    """
    Analyze the effect of role swapping on model predictions.

    Tests whether models appropriately adjust interpretations when
    speaker and listener roles are exchanged.

    Args:
        original_predictions: Predictions on original scenarios
        swapped_predictions: Predictions on role-swapped scenarios
        ground_truth: Ground truth labels
        swapped_metadata: Metadata about role swaps
        n_bootstrap: Bootstrap resamples for CI

    Returns:
        RoleSwapResult with all metrics
    """
    result = RoleSwapResult()
    result.n_scenarios = len(original_predictions)

    # Compute accuracy on original and swapped
    orig_correct = []
    swap_correct = []
    prediction_changed = []

    by_transition: dict[str, dict[str, list[int]]] = {}

    for orig_pred, swap_pred, truth, meta in zip(
        original_predictions, swapped_predictions, ground_truth, swapped_metadata
    ):
        # Extract labels
        orig_label = orig_pred.get("subtype", orig_pred.get("prediction", {}).get("subtype", ""))
        swap_label = swap_pred.get("subtype", swap_pred.get("prediction", {}).get("subtype", ""))
        true_label = truth.get("subtype", truth.get("ground_truth", {}).get("subtype", ""))

        # Correctness
        orig_correct.append(1 if orig_label == true_label else 0)
        swap_correct.append(1 if swap_label == true_label else 0)

        # Did prediction change?
        changed = 1 if orig_label != swap_label else 0
        prediction_changed.append(changed)

        # Track by power transition
        transition = f"{meta.original_power_relation}→{meta.swapped_power_relation}"
        if transition not in by_transition:
            by_transition[transition] = {"orig_correct": [], "swap_correct": [], "changed": []}
        by_transition[transition]["orig_correct"].append(orig_correct[-1])
        by_transition[transition]["swap_correct"].append(swap_correct[-1])
        by_transition[transition]["changed"].append(changed)

    # Core metrics
    result.original_accuracy = np.mean(orig_correct) if orig_correct else 0
    result.swapped_accuracy = np.mean(swap_correct) if swap_correct else 0
    result.accuracy_drop = result.original_accuracy - result.swapped_accuracy
    result.prediction_changed_rate = np.mean(prediction_changed) if prediction_changed else 0

    # Confidence intervals
    result.original_ci = bootstrap_ci(orig_correct, n_bootstrap)
    result.swapped_ci = bootstrap_ci(swap_correct, n_bootstrap)

    # Bootstrap CI for drop
    if orig_correct and swap_correct:
        drops = []
        for _ in range(n_bootstrap):
            orig_sample = np.random.choice(orig_correct, size=len(orig_correct), replace=True)
            swap_sample = np.random.choice(swap_correct, size=len(swap_correct), replace=True)
            drops.append(np.mean(orig_sample) - np.mean(swap_sample))
        drops = np.sort(drops)
        result.drop_ci = (drops[int(0.025 * n_bootstrap)], drops[int(0.975 * n_bootstrap)])

    # Cohen's d
    if orig_correct and swap_correct:
        result.cohens_d, result.cohens_d_ci = _compute_cohens_d_paired(
            orig_correct, swap_correct, n_bootstrap
        )

    # By power transition
    for transition, data in by_transition.items():
        result.by_power_transition[transition] = {
            "original_accuracy": np.mean(data["orig_correct"]),
            "swapped_accuracy": np.mean(data["swap_correct"]),
            "drop": np.mean(data["orig_correct"]) - np.mean(data["swap_correct"]),
            "change_rate": np.mean(data["changed"]),
            "n": len(data["orig_correct"]),
        }

    return result


def _compute_cohens_d_paired(
    group1: list[int],
    group2: list[int],
    n_bootstrap: int = 10000,
) -> tuple[float, tuple[float, float]]:
    """Compute Cohen's d for paired samples with bootstrap CI."""
    diff = [g1 - g2 for g1, g2 in zip(group1, group2)]
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    d = mean_diff / std_diff if std_diff > 0 else 0

    # Bootstrap CI
    ds = []
    diff_arr = np.array(diff)
    for _ in range(n_bootstrap):
        sample = np.random.choice(diff_arr, size=len(diff_arr), replace=True)
        std_s = np.std(sample, ddof=1)
        ds.append(np.mean(sample) / std_s if std_s > 0 else 0)

    ds = np.sort(ds)
    ci = (ds[int(0.025 * n_bootstrap)], ds[int(0.975 * n_bootstrap)])

    return d, ci


def generate_role_swap_report(result: RoleSwapResult) -> str:
    """
    Generate a text report of role swap analysis results.

    Args:
        result: RoleSwapResult from analysis

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "ROLE SWAP ANALYSIS RESULTS",
        "=" * 60,
        "",
        "OVERALL METRICS",
        "-" * 40,
        f"Original Accuracy:  {result.original_accuracy:.1%} "
        f"[{result.original_ci[0]:.1%}, {result.original_ci[1]:.1%}]",
        f"Swapped Accuracy:   {result.swapped_accuracy:.1%} "
        f"[{result.swapped_ci[0]:.1%}, {result.swapped_ci[1]:.1%}]",
        f"Accuracy Drop:      {result.accuracy_drop:.1%} "
        f"[{result.drop_ci[0]:.1%}, {result.drop_ci[1]:.1%}]",
        "",
        f"Cohen's d:          {result.cohens_d:.2f} "
        f"[{result.cohens_d_ci[0]:.2f}, {result.cohens_d_ci[1]:.2f}]",
        f"Prediction Changed: {result.prediction_changed_rate:.1%}",
        "",
    ]

    # Effect size interpretation
    d = abs(result.cohens_d)
    if d < 0.2:
        effect_interp = "negligible"
    elif d < 0.5:
        effect_interp = "small"
    elif d < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    lines.append(f"Effect Size: {effect_interp}")

    # By power transition
    lines.extend([
        "",
        "BY POWER TRANSITION",
        "-" * 40,
        f"{'Transition':<25} {'Orig%':<8} {'Swap%':<8} {'Drop':<8} {'n':<6}",
    ])

    for transition, data in sorted(result.by_power_transition.items()):
        lines.append(
            f"{transition:<25} "
            f"{data['original_accuracy']:.1%}    "
            f"{data['swapped_accuracy']:.1%}    "
            f"{data['drop']:+.1%}    "
            f"{data['n']}"
        )

    # Key finding
    lines.extend([
        "",
        "-" * 40,
    ])

    if result.accuracy_drop > 0.10:
        lines.append(f"Finding: Role swap causes significant accuracy drop ({result.accuracy_drop:.1%})")
        lines.append("         Models are sensitive to speaker/listener role assignments")
    elif result.accuracy_drop > 0.05:
        lines.append(f"Finding: Moderate sensitivity to role swap ({result.accuracy_drop:.1%})")
    else:
        lines.append(f"Finding: Models are robust to role swap ({result.accuracy_drop:.1%})")

    return "\n".join(lines)
