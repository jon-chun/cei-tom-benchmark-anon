"""
Rational Speech Act (RSA) L0/L1 Listener Comparison.

This module implements RSA listener models to compare LLM behavior
with literal (L0) and pragmatic (L1) listeners.

Reference: Goodman & Frank (2016). Pragmatic language interpretation
as probabilistic inference. Trends in Cognitive Sciences.

Key insight from paper:
- LLMs behave like L0 (literal) listeners: they interpret utterances
  without appropriately integrating contextual signals
- L1 alignment is low, indicating failure to model speaker intent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


# Plutchik emotion categories
EMOTIONS = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation"
]

# Emotion-utterance semantic similarity matrix (simplified)
# Based on typical linguistic associations
LITERAL_SEMANTICS = {
    "joy": {"positive": 0.9, "negative": 0.1, "neutral": 0.3},
    "trust": {"positive": 0.7, "negative": 0.2, "neutral": 0.5},
    "fear": {"positive": 0.1, "negative": 0.8, "neutral": 0.3},
    "surprise": {"positive": 0.5, "negative": 0.5, "neutral": 0.4},
    "sadness": {"positive": 0.1, "negative": 0.9, "neutral": 0.3},
    "disgust": {"positive": 0.05, "negative": 0.9, "neutral": 0.2},
    "anger": {"positive": 0.1, "negative": 0.85, "neutral": 0.2},
    "anticipation": {"positive": 0.6, "negative": 0.3, "neutral": 0.5},
}


@dataclass
class RSAComparisonResult:
    """Results of RSA listener comparison."""

    l0_predictions: list[dict[str, float]] = field(default_factory=list)
    l1_predictions: list[dict[str, float]] = field(default_factory=list)
    llm_predictions: list[dict[str, float]] = field(default_factory=list)

    l0_alignment_kappa: float = 0.0
    l1_alignment_kappa: float = 0.0
    l0_alignment_corr: float = 0.0
    l1_alignment_corr: float = 0.0

    interpretation: str = ""

    by_subtype: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "l0_alignment": {
                "kappa": self.l0_alignment_kappa,
                "correlation": self.l0_alignment_corr,
            },
            "l1_alignment": {
                "kappa": self.l1_alignment_kappa,
                "correlation": self.l1_alignment_corr,
            },
            "interpretation": self.interpretation,
            "by_subtype": self.by_subtype,
            "n_scenarios": len(self.llm_predictions),
        }


def classify_utterance_valence(utterance: str) -> str:
    """
    Classify utterance valence based on surface features.

    This is a simplified classifier - production version would use
    a proper sentiment model.
    """
    positive_markers = [
        "great", "happy", "love", "wonderful", "excellent",
        "thank", "appreciate", "glad", "sure", "yes"
    ]
    negative_markers = [
        "no", "not", "never", "hate", "terrible", "bad",
        "sorry", "unfortunately", "problem", "issue"
    ]

    utterance_lower = utterance.lower()
    pos_count = sum(1 for m in positive_markers if m in utterance_lower)
    neg_count = sum(1 for m in negative_markers if m in utterance_lower)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


def compute_l0_prediction(
    utterance: str,
    emotions: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute L0 (literal) listener prediction.

    L0 interprets utterances based on literal meaning only,
    ignoring contextual factors like speaker role or situation.

    P(emotion | utterance) ∝ P(utterance | emotion)

    Args:
        utterance: The speaker's utterance
        emotions: Possible emotion labels (default: Plutchik wheel)

    Returns:
        Probability distribution over emotions
    """
    if emotions is None:
        emotions = EMOTIONS

    # Classify utterance valence
    valence = classify_utterance_valence(utterance)

    # Compute unnormalized probabilities based on literal semantics
    unnorm_probs = {}
    for emotion in emotions:
        if emotion in LITERAL_SEMANTICS:
            unnorm_probs[emotion] = LITERAL_SEMANTICS[emotion].get(valence, 0.3)
        else:
            unnorm_probs[emotion] = 0.125  # Uniform prior

    # Normalize to probability distribution
    total = sum(unnorm_probs.values())
    probs = {e: p / total for e, p in unnorm_probs.items()}

    return probs


def compute_l1_prediction(
    utterance: str,
    context: str,
    speaker_role: str,
    listener_role: str,
    emotions: list[str] | None = None,
    alpha: float = 1.0,
) -> dict[str, float]:
    """
    Compute L1 (pragmatic) listener prediction.

    L1 recursively reasons about what a rational speaker
    would say given their intended meaning.

    P(emotion | utterance, context) ∝
        P(utterance | emotion, context)^α × P(emotion | context)

    Args:
        utterance: The speaker's utterance
        context: Situational context
        speaker_role: Role of speaker (e.g., "employee", "manager")
        listener_role: Role of listener
        emotions: Possible emotion labels
        alpha: Rationality parameter (higher = more rational speaker)

    Returns:
        Probability distribution over emotions
    """
    if emotions is None:
        emotions = EMOTIONS

    # Start with L0 prediction
    l0_probs = compute_l0_prediction(utterance, emotions)

    # Compute context prior P(emotion | context)
    context_prior = compute_context_prior(context, speaker_role, listener_role, emotions)

    # Combine with speaker rationality
    unnorm_probs = {}
    for emotion in emotions:
        # L1 = L0^alpha * prior
        unnorm_probs[emotion] = (l0_probs[emotion] ** alpha) * context_prior[emotion]

    # Normalize
    total = sum(unnorm_probs.values())
    if total > 0:
        probs = {e: p / total for e, p in unnorm_probs.items()}
    else:
        probs = {e: 1.0 / len(emotions) for e in emotions}

    return probs


def compute_context_prior(
    context: str,
    speaker_role: str,
    listener_role: str,
    emotions: list[str],
) -> dict[str, float]:
    """
    Compute context-based emotion prior P(emotion | context).

    Incorporates:
    - Power dynamics (subordinate speakers may mask negative emotions)
    - Situational factors (workplace contexts affect expression)
    """
    # Detect power relation
    power_relation = infer_power_relation(speaker_role, listener_role)

    # Base prior (uniform)
    prior = {e: 1.0 / len(emotions) for e in emotions}

    # Adjust based on power dynamics
    if power_relation == "lower_to_higher":
        # Subordinates may express constrained emotions
        # Increase probability of masked negative emotions
        prior["sadness"] *= 1.5
        prior["fear"] *= 1.5
        prior["anger"] *= 0.7  # Less likely to express openly
        prior["trust"] *= 1.3  # Strategic compliance

    elif power_relation == "higher_to_lower":
        # Superiors may be more direct
        prior["anger"] *= 1.2
        prior["disgust"] *= 1.2

    # Normalize
    total = sum(prior.values())
    prior = {e: p / total for e, p in prior.items()}

    return prior


def infer_power_relation(speaker_role: str, listener_role: str) -> str:
    """Infer power relation from roles."""
    higher_power_terms = [
        "manager", "supervisor", "director", "ceo", "boss",
        "senior", "lead", "head", "chief", "partner"
    ]
    lower_power_terms = [
        "junior", "intern", "assistant", "associate", "employee",
        "subordinate", "trainee", "new"
    ]

    speaker_lower = speaker_role.lower()
    listener_lower = listener_role.lower()

    speaker_is_higher = any(t in speaker_lower for t in higher_power_terms)
    speaker_is_lower = any(t in speaker_lower for t in lower_power_terms)
    listener_is_higher = any(t in listener_lower for t in higher_power_terms)
    listener_is_lower = any(t in listener_lower for t in lower_power_terms)

    if speaker_is_lower and listener_is_higher:
        return "lower_to_higher"
    elif speaker_is_higher and listener_is_lower:
        return "higher_to_lower"
    else:
        return "peer"


def compute_rsa_alignment(
    llm_predictions: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
) -> RSAComparisonResult:
    """
    Compute alignment between LLM predictions and RSA listeners.

    Args:
        llm_predictions: List of LLM prediction dicts with 'emotion' key
        scenarios: List of scenario dicts with context, utterance, roles

    Returns:
        RSAComparisonResult with alignment metrics
    """
    result = RSAComparisonResult()

    l0_labels = []
    l1_labels = []
    llm_labels = []

    for pred, scenario in zip(llm_predictions, scenarios):
        utterance = scenario.get("utterance", scenario.get("sd_utterance", ""))
        context = scenario.get("situation", scenario.get("sd_situation", ""))
        speaker_role = scenario.get("speaker_role", scenario.get("sd_speaker_role", "speaker"))
        listener_role = scenario.get("listener_role", scenario.get("sd_listener_role", "listener"))

        # Compute RSA predictions
        l0_probs = compute_l0_prediction(utterance)
        l1_probs = compute_l1_prediction(
            utterance, context, speaker_role, listener_role
        )

        # Get predicted labels
        l0_label = max(l0_probs, key=l0_probs.get)
        l1_label = max(l1_probs, key=l1_probs.get)
        llm_label = pred.get("emotion", pred.get("prediction", {}).get("emotion", "unknown"))

        l0_labels.append(l0_label)
        l1_labels.append(l1_label)
        llm_labels.append(llm_label)

        result.l0_predictions.append(l0_probs)
        result.l1_predictions.append(l1_probs)
        result.llm_predictions.append({"label": llm_label})

    # Compute Cohen's kappa
    result.l0_alignment_kappa = compute_cohens_kappa(llm_labels, l0_labels)
    result.l1_alignment_kappa = compute_cohens_kappa(llm_labels, l1_labels)

    # Compute correlation of agreement patterns
    l0_agreement = [1 if llm == l0 else 0 for llm, l0 in zip(llm_labels, l0_labels)]
    l1_agreement = [1 if llm == l1 else 0 for llm, l1 in zip(llm_labels, l1_labels)]

    result.l0_alignment_corr = np.mean(l0_agreement)
    result.l1_alignment_corr = np.mean(l1_agreement)

    # Interpretation
    if result.l0_alignment_kappa > result.l1_alignment_kappa + 0.1:
        result.interpretation = "LLMs behave like L0 (literal) listeners"
    elif result.l1_alignment_kappa > result.l0_alignment_kappa + 0.1:
        result.interpretation = "LLMs behave like L1 (pragmatic) listeners"
    else:
        result.interpretation = "Mixed L0/L1 behavior"

    return result


def compute_cohens_kappa(labels1: list[str], labels2: list[str]) -> float:
    """Compute Cohen's kappa between two label lists."""
    if len(labels1) != len(labels2) or len(labels1) == 0:
        return 0.0

    # Get unique labels
    all_labels = list(set(labels1) | set(labels2))
    n = len(labels1)

    # Observed agreement
    observed = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2) / n

    # Expected agreement (by chance)
    expected = 0.0
    for label in all_labels:
        p1 = labels1.count(label) / n
        p2 = labels2.count(label) / n
        expected += p1 * p2

    # Kappa
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0

    kappa = (observed - expected) / (1 - expected)
    return kappa
