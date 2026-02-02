"""Tests for RSA listener comparison module."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from venue.rsa_comparison import (
    compute_l0_prediction,
    compute_l1_prediction,
    compute_rsa_alignment,
    classify_utterance_valence,
    infer_power_relation,
    compute_cohens_kappa,
    EMOTIONS,
)


class TestUtteranceValence:
    """Tests for utterance valence classification."""

    def test_positive_utterance(self):
        """Positive words should classify as positive."""
        result = classify_utterance_valence("That's wonderful, thank you!")
        assert result == "positive"

    def test_negative_utterance(self):
        """Negative words should classify as negative."""
        result = classify_utterance_valence("I hate this terrible problem.")
        assert result == "negative"

    def test_neutral_utterance(self):
        """Neutral utterances should classify as neutral."""
        result = classify_utterance_valence("The meeting is at 3pm.")
        assert result == "neutral"


class TestL0Prediction:
    """Tests for L0 (literal) listener predictions."""

    def test_returns_probability_distribution(self):
        """L0 should return valid probability distribution."""
        probs = compute_l0_prediction("That's great!")

        # Should have all emotions
        assert set(probs.keys()) == set(EMOTIONS)

        # Should sum to 1
        assert abs(sum(probs.values()) - 1.0) < 1e-6

        # All values should be non-negative
        assert all(p >= 0 for p in probs.values())

    def test_positive_utterance_favors_positive_emotions(self):
        """Positive utterances should favor joy/trust."""
        probs = compute_l0_prediction("I love this, it's wonderful!")

        # Joy should be among top emotions for positive utterance
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_emotions = [e for e, _ in sorted_probs[:3]]
        assert "joy" in top_emotions or "trust" in top_emotions


class TestL1Prediction:
    """Tests for L1 (pragmatic) listener predictions."""

    def test_returns_probability_distribution(self):
        """L1 should return valid probability distribution."""
        probs = compute_l1_prediction(
            utterance="That's just great.",
            context="Employee receives negative review",
            speaker_role="employee",
            listener_role="manager",
        )

        assert set(probs.keys()) == set(EMOTIONS)
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert all(p >= 0 for p in probs.values())

    def test_context_modifies_prediction(self):
        """L1 should differ from L0 when context matters."""
        utterance = "That's just great."

        l0_probs = compute_l0_prediction(utterance)
        l1_probs = compute_l1_prediction(
            utterance=utterance,
            context="Employee receives a terrible performance review",
            speaker_role="junior employee",
            listener_role="senior manager",
        )

        # Predictions should differ (context should matter)
        # Not testing exact values, just that they're different
        l0_top = max(l0_probs, key=l0_probs.get)
        l1_top = max(l1_probs, key=l1_probs.get)

        # The distributions should be different
        diff = sum(abs(l0_probs[e] - l1_probs[e]) for e in EMOTIONS)
        assert diff > 0  # Some difference expected


class TestPowerRelation:
    """Tests for power relation inference."""

    def test_lower_to_higher(self):
        """Junior to senior should be lower_to_higher."""
        result = infer_power_relation("junior employee", "senior manager")
        assert result == "lower_to_higher"

    def test_higher_to_lower(self):
        """Manager to intern should be higher_to_lower."""
        result = infer_power_relation("department manager", "intern")
        assert result == "higher_to_lower"

    def test_peer(self):
        """Same level should be peer."""
        result = infer_power_relation("colleague", "colleague")
        assert result == "peer"


class TestCohensKappa:
    """Tests for Cohen's kappa computation."""

    def test_perfect_agreement(self):
        """Perfect agreement should give kappa = 1."""
        labels = ["joy", "anger", "sadness", "fear"]
        kappa = compute_cohens_kappa(labels, labels)
        assert abs(kappa - 1.0) < 1e-6

    def test_no_agreement(self):
        """Random disagreement should give low kappa."""
        labels1 = ["joy", "joy", "joy", "joy"]
        labels2 = ["anger", "anger", "anger", "anger"]
        kappa = compute_cohens_kappa(labels1, labels2)
        assert kappa < 0.5

    def test_empty_lists(self):
        """Empty lists should return 0."""
        kappa = compute_cohens_kappa([], [])
        assert kappa == 0.0


class TestRSAAlignment:
    """Tests for RSA alignment computation."""

    def test_alignment_result_structure(self):
        """Alignment results should have correct structure."""
        predictions = [
            {"emotion": "joy"},
            {"emotion": "anger"},
        ]
        scenarios = [
            {
                "utterance": "That's wonderful!",
                "situation": "Good news",
                "speaker_role": "friend",
                "listener_role": "friend",
            },
            {
                "utterance": "How dare you!",
                "situation": "Conflict",
                "speaker_role": "employee",
                "listener_role": "manager",
            },
        ]

        result = compute_rsa_alignment(predictions, scenarios)

        # Check structure
        assert hasattr(result, "l0_alignment_kappa")
        assert hasattr(result, "l1_alignment_kappa")
        assert hasattr(result, "interpretation")
        assert len(result.l0_predictions) == 2
        assert len(result.l1_predictions) == 2

    def test_to_dict_serialization(self):
        """Result should serialize to dict."""
        predictions = [{"emotion": "joy"}]
        scenarios = [{
            "utterance": "Great!",
            "situation": "Happy news",
            "speaker_role": "friend",
            "listener_role": "friend",
        }]

        result = compute_rsa_alignment(predictions, scenarios)
        result_dict = result.to_dict()

        assert "l0_alignment" in result_dict
        assert "l1_alignment" in result_dict
        assert "interpretation" in result_dict
