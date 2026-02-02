"""Tests for efficiency metrics module."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from venue.efficiency_metrics import (
    compute_pragmatic_efficiency,
    compute_utterance_anchoring,
    compute_power_asymmetry,
    bootstrap_ci,
    compute_cohens_d,
    EfficiencyMetric,
    EfficiencyResults,
)


class TestEfficiencyMetric:
    """Tests for EfficiencyMetric dataclass."""

    def test_creation(self):
        """Should create metric with all fields."""
        metric = EfficiencyMetric(
            name="test_metric",
            value=0.75,
            ci_lower=0.70,
            ci_upper=0.80,
            n_samples=100,
            interpretation="Test interpretation",
        )

        assert metric.name == "test_metric"
        assert metric.value == 0.75
        assert metric.ci_lower == 0.70
        assert metric.ci_upper == 0.80
        assert metric.n_samples == 100

    def test_to_dict(self):
        """Should serialize to dictionary."""
        metric = EfficiencyMetric(
            name="accuracy",
            value=0.82,
            ci_lower=0.78,
            ci_upper=0.86,
            n_samples=300,
        )

        result = metric.to_dict()

        assert result["name"] == "accuracy"
        assert result["value"] == 0.82
        assert result["ci_95"] == [0.78, 0.86]
        assert result["n_samples"] == 300


class TestEfficiencyResults:
    """Tests for EfficiencyResults dataclass."""

    def test_default_creation(self):
        """Should create with default metrics."""
        results = EfficiencyResults()

        assert results.human_accuracy.name == "human_accuracy"
        assert results.model_accuracy.name == "model_accuracy"
        assert results.efficiency_ratio.name == "efficiency_ratio"

    def test_to_dict_structure(self):
        """Should serialize with correct structure."""
        results = EfficiencyResults()
        result_dict = results.to_dict()

        assert "primary" in result_dict
        assert "compositional" in result_dict
        assert "anchoring" in result_dict
        assert "power" in result_dict


class TestPragmaticEfficiency:
    """Tests for pragmatic efficiency computation."""

    def test_basic_computation(self):
        """Should compute efficiency metrics correctly."""
        predictions = [
            {"subtype": "sarcasm-irony"},
            {"subtype": "mixed-signals"},
            {"subtype": "sarcasm-irony"},  # Correct
            {"subtype": "sarcasm-irony"},  # Correct
        ]
        ground_truth = [
            {"subtype": "mixed-signals"},   # Wrong
            {"subtype": "mixed-signals"},   # Correct
            {"subtype": "sarcasm-irony"},   # Correct
            {"subtype": "sarcasm-irony"},   # Correct
        ]

        results = compute_pragmatic_efficiency(
            predictions,
            ground_truth,
            human_accuracy=0.82,
            n_bootstrap=100,
        )

        # 3/4 correct = 0.75
        assert results.model_accuracy.value == 0.75
        assert results.human_accuracy.value == 0.82

        # Efficiency = 0.75 / 0.82 ≈ 0.915
        assert abs(results.efficiency_ratio.value - 0.75/0.82) < 0.01

    def test_empty_predictions(self):
        """Should handle empty predictions gracefully."""
        results = compute_pragmatic_efficiency([], [], human_accuracy=0.82)

        # Should return results, possibly with nan or 0
        assert isinstance(results, EfficiencyResults)


class TestUtteranceAnchoring:
    """Tests for utterance anchoring computation."""

    def test_no_shifts(self):
        """When predictions don't shift, anchoring should be high."""
        original = [
            {"subtype": "sarcasm-irony"},
            {"subtype": "sarcasm-irony"},
        ]
        recombined = [
            {"subtype": "sarcasm-irony"},  # Same
            {"subtype": "sarcasm-irony"},  # Same
        ]
        metadata = [
            {"context_subtype": "mixed-signals"},
            {"context_subtype": "passive-aggression"},
        ]

        anchoring, sensitivity = compute_utterance_anchoring(
            original, recombined, metadata, n_bootstrap=100
        )

        # No shifts = 100% anchoring
        assert anchoring.value == 1.0
        assert sensitivity.value == 0.0

    def test_all_shifts(self):
        """When all predictions shift, sensitivity should be high."""
        original = [
            {"subtype": "sarcasm-irony"},
            {"subtype": "mixed-signals"},
        ]
        recombined = [
            {"subtype": "mixed-signals"},     # Different
            {"subtype": "passive-aggression"},  # Different
        ]
        metadata = [
            {"context_subtype": "mixed-signals"},
            {"context_subtype": "passive-aggression"},
        ]

        anchoring, sensitivity = compute_utterance_anchoring(
            original, recombined, metadata, n_bootstrap=100
        )

        # All shifts = 0% anchoring, 100% sensitivity
        assert anchoring.value == 0.0
        assert sensitivity.value == 1.0


class TestPowerAsymmetry:
    """Tests for power asymmetry computation."""

    def test_gap_computation(self):
        """Should compute gap between peer and subordinate performance."""
        predictions = [
            {"subtype": "sarcasm-irony"},
            {"subtype": "sarcasm-irony"},
            {"subtype": "mixed-signals"},
            {"subtype": "mixed-signals"},
        ]
        ground_truth = [
            {"subtype": "sarcasm-irony"},  # Correct (peer)
            {"subtype": "sarcasm-irony"},  # Correct (peer)
            {"subtype": "sarcasm-irony"},  # Wrong (subordinate)
            {"subtype": "sarcasm-irony"},  # Wrong (subordinate)
        ]
        scenarios = [
            {"power_relation": "peer"},
            {"power_relation": "peer"},
            {"power_relation": "lower_to_higher"},
            {"power_relation": "lower_to_higher"},
        ]

        gap, cohens_d = compute_power_asymmetry(
            predictions, ground_truth, scenarios
        )

        # Peer: 2/2 = 100%, Subordinate: 0/2 = 0%
        # Gap should be 1.0
        assert gap.value == 1.0


class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_ci_contains_mean(self):
        """CI should contain the sample mean."""
        data = [0.8, 0.85, 0.82, 0.78, 0.81]
        mean = sum(data) / len(data)

        ci_lower, ci_upper = bootstrap_ci(data, n_resamples=1000)

        assert ci_lower <= mean <= ci_upper

    def test_ci_bounds(self):
        """CI should be within [0, 1] for proportion data."""
        data = [1, 1, 1, 0, 1, 0, 1, 1]  # Binary data

        ci_lower, ci_upper = bootstrap_ci(data, n_resamples=1000)

        assert 0 <= ci_lower <= ci_upper <= 1

    def test_single_value(self):
        """Should handle single value gracefully."""
        ci_lower, ci_upper = bootstrap_ci([0.5], n_resamples=100)

        # Should return some bounds
        assert ci_lower <= ci_upper


class TestCohensD:
    """Tests for Cohen's d effect size computation."""

    def test_no_difference(self):
        """Same groups should have d = 0."""
        group = [0.8, 0.82, 0.78, 0.81]
        d = compute_cohens_d(group, group)
        assert abs(d) < 0.01

    def test_large_difference(self):
        """Very different groups should have large d."""
        group1 = [0.9, 0.92, 0.88, 0.91]
        group2 = [0.1, 0.12, 0.08, 0.11]

        d = compute_cohens_d(group1, group2)

        # Should be large positive
        assert d > 2.0

    def test_empty_groups(self):
        """Empty groups should return 0."""
        assert compute_cohens_d([], [1, 2, 3]) == 0.0
        assert compute_cohens_d([1, 2, 3], []) == 0.0
