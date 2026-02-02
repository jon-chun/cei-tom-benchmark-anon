"""
Emotion mapping system for repairing invalid emotion labels.

This module provides:
- VALID_PLUTCHIK_EMOTIONS: Set of valid Plutchik wheel emotions
- EMOTION_MAPPING: Dictionary mapping invalid emotions to valid Plutchik emotions
- RETRY_EMOTIONS: Set of emotions that require retry (not mappable)
- map_emotion(): Function to map invalid emotions to valid ones
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# Valid Plutchik wheel emotions (8 primary emotions)
VALID_PLUTCHIK_EMOTIONS: frozenset[str] = frozenset(
    {
        "joy",
        "trust",
        "fear",
        "surprise",
        "sadness",
        "disgust",
        "anger",
        "anticipation",
    }
)


class MappingAction(Enum):
    """Action to take for an invalid emotion."""

    VALID = "valid"  # Already valid, no mapping needed
    MAPPED = "mapped"  # Successfully mapped to a valid emotion
    RETRY = "retry"  # Requires retry (e.g., "unknown")
    UNKNOWN = "unknown"  # Unknown emotion, not in mapping


@dataclass
class EmotionMappingResult:
    """Result of attempting to map an emotion."""

    original: str
    mapped: str | None
    action: MappingAction
    rationale: str | None = None

    @property
    def is_valid(self) -> bool:
        """Check if the result is valid (either already valid or successfully mapped)."""
        return self.action in (MappingAction.VALID, MappingAction.MAPPED)


# Confirmed mappings (unambiguous synonyms based on Plutchik emotion wheel)
# Keys are lowercase, normalized versions of invalid emotions
EMOTION_MAPPING: dict[str, tuple[str, str]] = {
    # Sadness-related
    "guilt": ("sadness", "Self-directed negative emotion, Plutchik secondary"),
    "disappointment": ("sadness", "Unmet expectation, sadness variant"),
    # Fear-related
    "anxiety": ("fear", "Fear of future threat"),
    "embarrassment": ("fear", "Social threat response, anxiety-related"),
    "concern": ("fear", "Worry about negative outcome"),
    "discomfort": ("fear", "Unease, mild fear response"),
    "distrust": ("fear", "Opposite of trust, fear-based caution"),
    "uncertainty": ("fear", "Unknown threat response"),
    "stress": ("fear", "Physiological fear response"),
    "defensive": ("fear", "Self-protection from threat"),
    # Anger-related
    "frustration": ("anger", "Blocked goal, anger precursor"),
    "annoyance": ("anger", "Mild anger, irritation"),
    # Joy-related
    "relief": ("joy", "Positive resolution of tension"),
    "amusement": ("joy", "Positive affect, humor response"),
    "gratitude": ("joy", "Positive social emotion"),
}

# Values that require retry (not mappable)
RETRY_EMOTIONS: frozenset[str] = frozenset(
    {
        "unknown",
    }
)


def normalize_emotion(emotion: str) -> str:
    """Normalize an emotion string for lookup.

    Args:
        emotion: The emotion string to normalize

    Returns:
        Lowercase, stripped emotion string
    """
    return emotion.lower().strip() if emotion else ""


def map_emotion(emotion: str) -> EmotionMappingResult:
    """Map an emotion to a valid Plutchik emotion.

    Args:
        emotion: The emotion to map

    Returns:
        EmotionMappingResult with mapping information

    Examples:
        >>> map_emotion("anger")
        EmotionMappingResult(original='anger', mapped='anger', action=VALID)

        >>> map_emotion("guilt")
        EmotionMappingResult(original='guilt', mapped='sadness', action=MAPPED)

        >>> map_emotion("unknown")
        EmotionMappingResult(original='unknown', mapped=None, action=RETRY)
    """
    if not emotion:
        return EmotionMappingResult(
            original=emotion,
            mapped=None,
            action=MappingAction.UNKNOWN,
            rationale="Empty emotion string",
        )

    normalized = normalize_emotion(emotion)

    # Check if already valid
    if normalized in VALID_PLUTCHIK_EMOTIONS:
        return EmotionMappingResult(
            original=emotion,
            mapped=normalized,
            action=MappingAction.VALID,
            rationale="Already a valid Plutchik emotion",
        )

    # Check if requires retry
    if normalized in RETRY_EMOTIONS:
        return EmotionMappingResult(
            original=emotion,
            mapped=None,
            action=MappingAction.RETRY,
            rationale="Indeterminate emotion, requires retry",
        )

    # Check mapping table
    if normalized in EMOTION_MAPPING:
        mapped_emotion, rationale = EMOTION_MAPPING[normalized]
        return EmotionMappingResult(
            original=emotion,
            mapped=mapped_emotion,
            action=MappingAction.MAPPED,
            rationale=rationale,
        )

    # Unknown emotion
    return EmotionMappingResult(
        original=emotion,
        mapped=None,
        action=MappingAction.UNKNOWN,
        rationale=f"No mapping defined for '{emotion}'",
    )


def is_valid_emotion(emotion: str) -> bool:
    """Check if an emotion is a valid Plutchik emotion.

    Args:
        emotion: The emotion to check

    Returns:
        True if valid, False otherwise
    """
    return normalize_emotion(emotion) in VALID_PLUTCHIK_EMOTIONS


def get_mapping_rationale(emotion: str) -> str | None:
    """Get the rationale for an emotion mapping.

    Args:
        emotion: The invalid emotion

    Returns:
        Rationale string if mapping exists, None otherwise
    """
    normalized = normalize_emotion(emotion)
    if normalized in EMOTION_MAPPING:
        return EMOTION_MAPPING[normalized][1]
    return None


def get_all_mappable_emotions() -> list[str]:
    """Get list of all emotions that can be mapped.

    Returns:
        List of mappable emotion names
    """
    return sorted(EMOTION_MAPPING.keys())


def get_mapping_stats() -> dict[str, list[str]]:
    """Get statistics about emotion mappings grouped by target emotion.

    Returns:
        Dictionary mapping target emotions to list of source emotions
    """
    stats: dict[str, list[str]] = {}
    for source, (target, _) in EMOTION_MAPPING.items():
        if target not in stats:
            stats[target] = []
        stats[target].append(source)
    return {k: sorted(v) for k, v in sorted(stats.items())}
