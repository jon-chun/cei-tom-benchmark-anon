# CEI-ToM Data Dictionary

This document describes the data format for the CEI (Contextual Emotional Inference) benchmark.

## Human Annotations (`data/human-gold-aggregate/*.csv`)

### Scenario Columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique scenario identifier within subtype |
| `sd_situation` | string | Background context establishing the scenario setting |
| `sd_utterance` | string | Ambiguous statement requiring pragmatic interpretation |
| `sd_speaker_role` | string | Speaker's social/organizational role (e.g., "employee", "professor") |
| `sd_listener_role` | string | Listener's social/organizational role |
| `gold_standard` | string | Majority-vote emotion label from Plutchik's 8 basic emotions |

### Annotator Labels

Each scenario was labeled by 3 trained annotators. Column naming follows the pattern `sl_{measure}_{annotator}`:

| Column Pattern | Description |
|----------------|-------------|
| `sl_plutchik_primary_{name}` | Primary emotion judgment (Plutchik's 8) |
| `sl_v_{name}` | Valence rating (very unpleasant → very pleasant) |
| `sl_a_{name}` | Arousal rating (very calm → very excited) |
| `sl_d_{name}` | Dominance rating (very controlled → very in control) |
| `sl_confidence_{name}` | Annotator confidence (unsure → very confident) |

### Plutchik's 8 Basic Emotions

| Emotion | Description |
|---------|-------------|
| `joy` | Happiness, contentment, positive affect |
| `trust` | Confidence, acceptance, openness |
| `fear` | Anxiety, apprehension, threat response |
| `surprise` | Unexpected, startled, astonishment |
| `sadness` | Grief, disappointment, sorrow |
| `disgust` | Revulsion, distaste, rejection |
| `anger` | Frustration, irritation, hostility |
| `anticipation` | Expectation, interest, vigilance |

### Power Relations

Derived from speaker/listener roles:

| Relation | Description |
|----------|-------------|
| `peer` | Equal social standing |
| `higher_to_lower` | Superior speaking to subordinate |
| `lower_to_higher` | Subordinate speaking to superior |

## LLM Predictions (`outputs/main_inference/{model}/*.json`)

| Field | Type | Description |
|-------|------|-------------|
| `scenario_id` | string | Format: `{subtype}_{id}` (e.g., `sarcasm-irony_42`) |
| `model_id` | string | Model identifier |
| `success` | boolean | Whether prediction was successful |
| `prediction.emotion` | string | Predicted Plutchik emotion |
| `prediction.confidence` | float | Model confidence (0-1) |
| `prediction.valence` | float | Predicted valence (-1 to 1) |
| `prediction.arousal` | float | Predicted arousal (-1 to 1) |
| `prediction.dominance` | float | Predicted dominance (-1 to 1) |
| `latency_ms` | float | API response time in milliseconds |
| `timestamp` | string | ISO 8601 timestamp |

## Pragmatic Subtypes

| Subtype | Description | N |
|---------|-------------|---|
| `sarcasm-irony` | Surface meaning contradicts intended meaning | 60 |
| `mixed-signals` | Verbal/non-verbal cues conflict | 60 |
| `strategic-politeness` | Face-saving language masks negative affect | 60 |
| `passive-aggression` | Indirect hostility through neutral behavior | 60 |
| `deflection-misdirection` | Topic change to avoid addressing issues | 60 |

**Total: 300 scenarios** (60 per subtype × 5 subtypes)
