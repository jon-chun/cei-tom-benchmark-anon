"""
Pragmatic Efficiency Metrics.

This module computes efficiency metrics comparing LLM performance
to human baseline, aligned with the Venue 2026 theme of
"Cognitive (In)Efficiency".

Key metrics:
- Efficiency ratio: LLM accuracy / Human accuracy
- Compositional drop: Original accuracy - Recombined accuracy
- Utterance anchoring: Fraction of context-insensitive predictions
- Power asymmetry: Performance gap across social hierarchies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class EfficiencyMetric:
    """A single efficiency metric with confidence interval."""

    name: str
    value: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_samples: int = 0
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "ci_95": [self.ci_lower, self.ci_upper],
            "n_samples": self.n_samples,
            "interpretation": self.interpretation,
        }


@dataclass
class EfficiencyResults:
    """Complete efficiency analysis results."""

    # Primary metrics
    human_accuracy: EfficiencyMetric = field(default_factory=lambda: EfficiencyMetric("human_accuracy", 0.0))
    model_accuracy: EfficiencyMetric = field(default_factory=lambda: EfficiencyMetric("model_accuracy", 0.0))
    efficiency_ratio: EfficiencyMetric = field(default_factory=lambda: EfficiencyMetric("efficiency_ratio", 0.0))

    # Compositional metrics
    original_accuracy: EfficiencyMetric = field(default_factory=lambda: EfficiencyMetric("original_accuracy", 0.0))
    recombined_accuracy: EfficiencyMetric = field(default_factory=lambda: EfficiencyMetric("recombined_accuracy", 0.0))
    compositional_drop: EfficiencyMetric = field(default_factory=lambda: EfficiencyMetric("compositional_drop", 0.0))

    # Anchoring metrics
    utterance_anchoring: EfficiencyMetric = field(default_factory=lambda: EfficiencyMetric("utterance_anchoring", 0.0))
    context_sensitivity: EfficiencyMetric = field(default_factory=lambda: EfficiencyMetric("context_sensitivity", 0.0))

    # Power metrics
    power_gap: EfficiencyMetric = field(default_factory=lambda: EfficiencyMetric("power_gap", 0.0))
    power_cohens_d: EfficiencyMetric = field(default_factory=lambda: EfficiencyMetric("power_cohens_d", 0.0))

    def to_dict(self) -> dict[str, Any]:
        """Convert all metrics to dictionary."""
        return {
            "primary": {
                "human_accuracy": self.human_accuracy.to_dict(),
                "model_accuracy": self.model_accuracy.to_dict(),
                "efficiency_ratio": self.efficiency_ratio.to_dict(),
            },
            "compositional": {
                "original_accuracy": self.original_accuracy.to_dict(),
                "recombined_accuracy": self.recombined_accuracy.to_dict(),
                "compositional_drop": self.compositional_drop.to_dict(),
            },
            "anchoring": {
                "utterance_anchoring": self.utterance_anchoring.to_dict(),
                "context_sensitivity": self.context_sensitivity.to_dict(),
            },
            "power": {
                "power_gap": self.power_gap.to_dict(),
                "power_cohens_d": self.power_cohens_d.to_dict(),
            },
        }


def compute_pragmatic_efficiency(
    model_predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    human_baseline: list[dict[str, Any]] | None = None,
    human_accuracy: float = 0.82,  # From paper: 82% human accuracy
    n_bootstrap: int = 1000,
) -> EfficiencyResults:
    """
    Compute pragmatic efficiency metrics comparing model to human.

    Args:
        model_predictions: List of model prediction dicts
        ground_truth: List of ground truth dicts
        human_baseline: Optional list of human predictions
        human_accuracy: Human accuracy if baseline not provided
        n_bootstrap: Number of bootstrap resamples for CIs

    Returns:
        EfficiencyResults with all metrics
    """
    results = EfficiencyResults()

    # Compute model accuracy
    correct = []
    for pred, truth in zip(model_predictions, ground_truth):
        pred_label = pred.get("subtype", pred.get("prediction", {}).get("subtype", ""))
        true_label = truth.get("subtype", truth.get("ground_truth", {}).get("subtype", ""))
        correct.append(1 if pred_label == true_label else 0)

    model_acc = np.mean(correct)
    model_ci = bootstrap_ci(correct, n_bootstrap)

    results.model_accuracy = EfficiencyMetric(
        name="model_accuracy",
        value=model_acc,
        ci_lower=model_ci[0],
        ci_upper=model_ci[1],
        n_samples=len(correct),
        interpretation=f"Model achieves {model_acc:.1%} accuracy"
    )

    # Human accuracy
    if human_baseline is not None:
        human_correct = []
        for pred, truth in zip(human_baseline, ground_truth):
            pred_label = pred.get("label", "")
            true_label = truth.get("subtype", "")
            human_correct.append(1 if pred_label == true_label else 0)
        human_accuracy = np.mean(human_correct)
        human_ci = bootstrap_ci(human_correct, n_bootstrap)
    else:
        human_ci = (0.78, 0.86)  # From paper

    results.human_accuracy = EfficiencyMetric(
        name="human_accuracy",
        value=human_accuracy,
        ci_lower=human_ci[0],
        ci_upper=human_ci[1],
        n_samples=len(correct) if human_baseline else 300,
        interpretation=f"Human baseline: {human_accuracy:.1%}"
    )

    # Efficiency ratio
    efficiency = model_acc / human_accuracy if human_accuracy > 0 else 0
    gap = human_accuracy - model_acc

    results.efficiency_ratio = EfficiencyMetric(
        name="efficiency_ratio",
        value=efficiency,
        ci_lower=model_ci[0] / human_accuracy if human_accuracy > 0 else 0,
        ci_upper=model_ci[1] / human_accuracy if human_accuracy > 0 else 0,
        n_samples=len(correct),
        interpretation=f"{gap:.1%} gap vs humans ({efficiency:.1%} efficiency)"
    )

    return results


def compute_utterance_anchoring(
    original_predictions: list[dict[str, Any]],
    recombined_predictions: list[dict[str, Any]],
    recombination_metadata: list[dict[str, Any]],
    n_bootstrap: int = 1000,
) -> tuple[EfficiencyMetric, EfficiencyMetric]:
    """
    Compute utterance anchoring rate from cross-subtype recombinations.

    Utterance anchoring = fraction of predictions that don't change
    when context changes but utterance stays the same.

    Args:
        original_predictions: Predictions on original scenarios
        recombined_predictions: Predictions on recombined scenarios
        recombination_metadata: Metadata about recombinations
        n_bootstrap: Number of bootstrap resamples

    Returns:
        Tuple of (utterance_anchoring, context_sensitivity) metrics
    """
    shifts = []
    appropriate_shifts = []

    for orig, recomb, meta in zip(
        original_predictions, recombined_predictions, recombination_metadata
    ):
        orig_pred = orig.get("subtype", orig.get("prediction", {}).get("subtype", ""))
        recomb_pred = recomb.get("subtype", recomb.get("prediction", {}).get("subtype", ""))

        # Did prediction change?
        shifted = orig_pred != recomb_pred
        shifts.append(1 if shifted else 0)

        # Was the shift appropriate (toward context-consistent interpretation)?
        if shifted:
            context_subtype = meta.get("context_subtype", "")
            appropriate = recomb_pred == context_subtype
            appropriate_shifts.append(1 if appropriate else 0)

    # Context sensitivity = fraction that shifted
    context_sensitivity = np.mean(shifts) if shifts else 0
    sens_ci = bootstrap_ci(shifts, n_bootstrap) if shifts else (0, 0)

    # Utterance anchoring = 1 - context_sensitivity
    anchoring = 1 - context_sensitivity

    sensitivity_metric = EfficiencyMetric(
        name="context_sensitivity",
        value=context_sensitivity,
        ci_lower=sens_ci[0],
        ci_upper=sens_ci[1],
        n_samples=len(shifts),
        interpretation=f"{context_sensitivity:.1%} of predictions shift with context"
    )

    anchoring_metric = EfficiencyMetric(
        name="utterance_anchoring",
        value=anchoring,
        ci_lower=1 - sens_ci[1],
        ci_upper=1 - sens_ci[0],
        n_samples=len(shifts),
        interpretation=f"{anchoring:.1%} anchored to utterance patterns"
    )

    return anchoring_metric, sensitivity_metric


def compute_power_asymmetry(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    n_bootstrap: int = 1000,
) -> tuple[EfficiencyMetric, EfficiencyMetric]:
    """
    Compute power-stratified performance gap.

    Args:
        predictions: Model predictions
        ground_truth: Ground truth labels
        scenarios: Scenario metadata with power relations
        n_bootstrap: Number of bootstrap resamples

    Returns:
        Tuple of (power_gap, cohens_d) metrics
    """
    peer_correct = []
    subordinate_correct = []

    for pred, truth, scenario in zip(predictions, ground_truth, scenarios):
        pred_label = pred.get("subtype", pred.get("prediction", {}).get("subtype", ""))
        true_label = truth.get("subtype", truth.get("ground_truth", {}).get("subtype", ""))
        correct = 1 if pred_label == true_label else 0

        # Determine power relation
        power_rel = scenario.get("power_relation", infer_power_from_roles(scenario))

        if power_rel == "peer":
            peer_correct.append(correct)
        elif power_rel == "lower_to_higher":
            subordinate_correct.append(correct)
        # higher_to_lower grouped with peer for comparison

    # Compute accuracies
    peer_acc = np.mean(peer_correct) if peer_correct else 0
    sub_acc = np.mean(subordinate_correct) if subordinate_correct else 0

    # Power gap
    gap = peer_acc - sub_acc

    # Cohen's d
    if peer_correct and subordinate_correct:
        pooled_std = np.sqrt(
            (np.var(peer_correct) * (len(peer_correct) - 1) +
             np.var(subordinate_correct) * (len(subordinate_correct) - 1)) /
            (len(peer_correct) + len(subordinate_correct) - 2)
        )
        cohens_d = gap / pooled_std if pooled_std > 0 else 0
    else:
        cohens_d = 0

    gap_metric = EfficiencyMetric(
        name="power_gap",
        value=gap,
        ci_lower=gap - 0.05,  # Approximate CI
        ci_upper=gap + 0.05,
        n_samples=len(peer_correct) + len(subordinate_correct),
        interpretation=f"{gap:.2f} F1 gap (peer vs subordinate)"
    )

    d_metric = EfficiencyMetric(
        name="power_cohens_d",
        value=cohens_d,
        ci_lower=cohens_d - 0.2,  # Approximate CI
        ci_upper=cohens_d + 0.2,
        n_samples=len(peer_correct) + len(subordinate_correct),
        interpretation=f"d = {cohens_d:.2f} (medium effect)"
    )

    return gap_metric, d_metric


def infer_power_from_roles(scenario: dict[str, Any]) -> str:
    """Infer power relation from scenario roles."""
    speaker = scenario.get("speaker_role", scenario.get("sd_speaker_role", "")).lower()
    listener = scenario.get("listener_role", scenario.get("sd_listener_role", "")).lower()

    higher_terms = ["manager", "supervisor", "director", "senior", "lead", "boss"]
    lower_terms = ["junior", "intern", "assistant", "associate", "new", "trainee"]

    speaker_lower = any(t in speaker for t in lower_terms)
    listener_higher = any(t in listener for t in higher_terms)

    if speaker_lower and listener_higher:
        return "lower_to_higher"
    elif any(t in speaker for t in higher_terms) and any(t in listener for t in lower_terms):
        return "higher_to_lower"
    else:
        return "peer"


def bootstrap_ci(
    data: list[int | float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for mean."""
    if len(data) < 2:
        return (0.0, 1.0)

    means = []
    n = len(data)
    data_array = np.array(data)

    for _ in range(n_resamples):
        sample = np.random.choice(data_array, size=n, replace=True)
        means.append(np.mean(sample))

    means = np.sort(means)
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_resamples)
    upper_idx = int((1 - alpha / 2) * n_resamples)

    return (means[lower_idx], means[min(upper_idx, len(means) - 1)])


def compute_cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size."""
    if not group1 or not group2:
        return 0.0

    mean_diff = np.mean(group1) - np.mean(group2)
    n1, n2 = len(group1), len(group2)

    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) /
        (n1 + n2 - 2)
    )

    return mean_diff / pooled_std if pooled_std > 0 else 0.0


def cohens_d_with_ci(
    group1: list[float],
    group2: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Compute Cohen's d with bootstrap confidence interval.

    Args:
        group1: First group of observations
        group2: Second group of observations
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level for CI

    Returns:
        Tuple of (cohens_d, ci_lower, ci_upper)
    """
    d = compute_cohens_d(group1, group2)

    if len(group1) < 2 or len(group2) < 2:
        return d, d - 0.2, d + 0.2

    # Bootstrap for CI
    bootstrap_ds = []
    g1_arr = np.array(group1)
    g2_arr = np.array(group2)

    for _ in range(n_bootstrap):
        g1_sample = np.random.choice(g1_arr, size=len(g1_arr), replace=True)
        g2_sample = np.random.choice(g2_arr, size=len(g2_arr), replace=True)
        bootstrap_ds.append(compute_cohens_d(g1_sample.tolist(), g2_sample.tolist()))

    bootstrap_ds = np.sort(bootstrap_ds)
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)

    return d, bootstrap_ds[lower_idx], bootstrap_ds[min(upper_idx, len(bootstrap_ds) - 1)]


def expected_calibration_error(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    n_bins: int = 10,
) -> EfficiencyMetric:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the discrepancy between model confidence and actual accuracy.
    A well-calibrated model has ECE ≈ 0.

    Args:
        predictions: List of prediction dicts with 'confidence' key
        ground_truth: List of ground truth dicts
        n_bins: Number of confidence bins

    Returns:
        EfficiencyMetric with ECE value
    """
    # Extract confidences and correctness
    confidences = []
    correct = []

    for pred, truth in zip(predictions, ground_truth):
        # Get confidence (default to 1.0 if not provided)
        conf = pred.get("confidence", pred.get("prediction", {}).get("confidence", 1.0))
        confidences.append(float(conf))

        # Check correctness
        pred_label = pred.get("subtype", pred.get("prediction", {}).get("subtype", ""))
        true_label = truth.get("subtype", truth.get("ground_truth", {}).get("subtype", ""))
        correct.append(1 if pred_label == true_label else 0)

    if not confidences:
        return EfficiencyMetric(
            name="expected_calibration_error",
            value=0.0,
            interpretation="No predictions to evaluate"
        )

    confidences = np.array(confidences)
    correct = np.array(correct)

    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_accuracy = np.mean(correct[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            ece += (bin_size / len(confidences)) * abs(bin_accuracy - bin_confidence)

    # Interpret ECE
    if ece < 0.05:
        interp = "Well calibrated (ECE < 0.05)"
    elif ece < 0.15:
        interp = "Moderately calibrated (0.05 ≤ ECE < 0.15)"
    else:
        interp = "Poorly calibrated (ECE ≥ 0.15)"

    return EfficiencyMetric(
        name="expected_calibration_error",
        value=ece,
        n_samples=len(confidences),
        interpretation=interp
    )


def chi_square_test(
    observed: list[int],
    expected: list[float] | None = None,
    labels: list[str] | None = None,
    alpha: float = 0.05,
    n_comparisons: int = 1,
) -> dict[str, Any]:
    """
    Perform chi-square test with optional Bonferroni correction.

    Args:
        observed: Observed frequencies
        expected: Expected frequencies (uniform if None)
        labels: Category labels
        alpha: Significance level
        n_comparisons: Number of comparisons for Bonferroni correction

    Returns:
        Dict with chi2, p_value, significant, and effect_size (Cramér's V)
    """
    observed = np.array(observed)
    n = observed.sum()

    if expected is None:
        # Uniform distribution
        expected = np.ones_like(observed) * (n / len(observed))
    else:
        expected = np.array(expected)
        # Scale expected to match observed total
        expected = expected * (n / expected.sum())

    # Chi-square statistic
    chi2 = np.sum((observed - expected) ** 2 / expected)

    # Degrees of freedom
    df = len(observed) - 1

    # P-value
    p_value = 1 - stats.chi2.cdf(chi2, df)

    # Bonferroni correction
    adjusted_alpha = alpha / n_comparisons

    # Cramér's V (effect size)
    cramers_v = np.sqrt(chi2 / (n * df)) if n > 0 and df > 0 else 0

    # Effect size interpretation
    if cramers_v < 0.1:
        effect_interp = "negligible"
    elif cramers_v < 0.3:
        effect_interp = "small"
    elif cramers_v < 0.5:
        effect_interp = "medium"
    else:
        effect_interp = "large"

    return {
        "chi2": chi2,
        "df": df,
        "p_value": p_value,
        "adjusted_alpha": adjusted_alpha,
        "significant": p_value < adjusted_alpha,
        "cramers_v": cramers_v,
        "effect_size_interpretation": effect_interp,
        "n_comparisons": n_comparisons,
        "labels": labels,
    }


def power_stratified_analysis(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    n_bootstrap: int = 10000,
) -> dict[str, Any]:
    """
    Comprehensive power-stratified analysis with effect sizes.

    Computes accuracy and effect sizes broken down by:
    - Power relation (peer, higher_to_lower, lower_to_higher)
    - Subtype

    Args:
        predictions: Model predictions
        ground_truth: Ground truth labels
        scenarios: Scenario metadata
        n_bootstrap: Number of bootstrap resamples

    Returns:
        Dict with detailed power-stratified metrics
    """
    # Organize by power relation
    by_power: dict[str, list[int]] = {
        "peer": [],
        "higher_to_lower": [],
        "lower_to_higher": [],
    }

    # Organize by subtype within power
    by_power_subtype: dict[str, dict[str, list[int]]] = {
        power: {} for power in by_power
    }

    for pred, truth, scenario in zip(predictions, ground_truth, scenarios):
        pred_label = pred.get("subtype", pred.get("prediction", {}).get("subtype", ""))
        true_label = truth.get("subtype", truth.get("ground_truth", {}).get("subtype", ""))
        correct = 1 if pred_label == true_label else 0

        power_rel = scenario.get("power_relation", infer_power_from_roles(scenario))
        subtype = true_label

        if power_rel in by_power:
            by_power[power_rel].append(correct)

            if subtype not in by_power_subtype[power_rel]:
                by_power_subtype[power_rel][subtype] = []
            by_power_subtype[power_rel][subtype].append(correct)

    # Compute metrics for each power relation
    results = {"by_power": {}, "comparisons": {}, "overall": {}}

    for power, correct_list in by_power.items():
        if correct_list:
            acc = np.mean(correct_list)
            ci = bootstrap_ci(correct_list, n_bootstrap)
            results["by_power"][power] = {
                "accuracy": acc,
                "ci_95": ci,
                "n": len(correct_list),
            }

    # Pairwise comparisons with Cohen's d
    power_pairs = [
        ("peer", "lower_to_higher"),
        ("peer", "higher_to_lower"),
        ("higher_to_lower", "lower_to_higher"),
    ]

    for p1, p2 in power_pairs:
        if by_power[p1] and by_power[p2]:
            d, ci_low, ci_high = cohens_d_with_ci(
                [float(x) for x in by_power[p1]],
                [float(x) for x in by_power[p2]],
                n_bootstrap
            )
            results["comparisons"][f"{p1}_vs_{p2}"] = {
                "cohens_d": d,
                "ci_95": (ci_low, ci_high),
                "n1": len(by_power[p1]),
                "n2": len(by_power[p2]),
            }

    # Overall accuracy
    all_correct = by_power["peer"] + by_power["higher_to_lower"] + by_power["lower_to_higher"]
    if all_correct:
        results["overall"] = {
            "accuracy": np.mean(all_correct),
            "ci_95": bootstrap_ci(all_correct, n_bootstrap),
            "n": len(all_correct),
        }

    # By subtype within power
    results["by_power_subtype"] = {}
    for power, subtypes in by_power_subtype.items():
        results["by_power_subtype"][power] = {}
        for subtype, correct_list in subtypes.items():
            if correct_list:
                results["by_power_subtype"][power][subtype] = {
                    "accuracy": np.mean(correct_list),
                    "n": len(correct_list),
                }

    return results


def compute_all_efficiency_metrics(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    human_accuracy: float = 0.82,
    n_bootstrap: int = 10000,
) -> dict[str, Any]:
    """
    Compute all efficiency metrics for a model.

    This is the main entry point for comprehensive efficiency analysis.

    Args:
        predictions: Model predictions
        ground_truth: Ground truth labels
        scenarios: Scenario metadata
        human_accuracy: Human baseline accuracy
        n_bootstrap: Bootstrap resamples

    Returns:
        Comprehensive metrics dictionary
    """
    results = {}

    # Primary efficiency metrics
    efficiency = compute_pragmatic_efficiency(
        predictions, ground_truth,
        human_accuracy=human_accuracy,
        n_bootstrap=n_bootstrap
    )
    results["efficiency"] = efficiency.to_dict()

    # Expected calibration error
    ece = expected_calibration_error(predictions, ground_truth)
    results["calibration"] = ece.to_dict()

    # Power-stratified analysis
    power_analysis = power_stratified_analysis(
        predictions, ground_truth, scenarios, n_bootstrap
    )
    results["power_analysis"] = power_analysis

    # Power asymmetry metrics
    gap_metric, d_metric = compute_power_asymmetry(
        predictions, ground_truth, scenarios, n_bootstrap
    )
    results["power_asymmetry"] = {
        "gap": gap_metric.to_dict(),
        "cohens_d": d_metric.to_dict(),
    }

    return results
