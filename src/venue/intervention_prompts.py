"""
Intervention Prompts for Venue 2026 Study.

This module defines prompting interventions to test if explicit guidance
can restore pragmatic efficiency in LLMs.

Interventions (Section 4.5 of paper):
1. INT-2: Perspective-taking prompt (consider speaker's viewpoint)
2. INT-3: Subtype awareness prompt (explicitly list pragmatic categories)
3. INT-5: Counterfactual prompting (contrast with alternative interpretations)

These interventions test whether pragmatic failures are due to:
- Lack of perspective-taking capacity
- Missing knowledge of pragmatic categories
- Insufficient contrastive reasoning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class InterventionResult:
    """Result of applying an intervention."""

    intervention_type: str
    original_accuracy: float
    intervention_accuracy: float
    improvement: float
    n_scenarios: int
    significant: bool = False
    p_value: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intervention_type": self.intervention_type,
            "original_accuracy": self.original_accuracy,
            "intervention_accuracy": self.intervention_accuracy,
            "improvement": self.improvement,
            "n_scenarios": self.n_scenarios,
            "significant": self.significant,
            "p_value": self.p_value,
        }


# Base prompt for CEI-ToM classification
BASE_PROMPT = """You are analyzing a workplace conversation to identify the underlying emotion and pragmatic strategy.

**Scenario:**
{situation}

**Speaker:** {speaker_role}
**Listener:** {listener_role}

**Utterance:** "{utterance}"

**Task:** Based on the full context, identify:
1. The speaker's underlying emotion (from: joy, trust, fear, surprise, sadness, disgust, anger, anticipation)
2. The pragmatic subtype (from: sarcasm-irony, mixed-signals, passive-aggression, deflection-misdirection, strategic-politeness)

Respond in JSON format:
```json
{{
  "emotion": "<emotion>",
  "subtype": "<subtype>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}
```"""


# INT-2: Perspective-taking intervention
PERSPECTIVE_TAKING_PROMPT = """You are analyzing a workplace conversation to identify the underlying emotion and pragmatic strategy.

**IMPORTANT: Before answering, take the speaker's perspective.**
- Consider: What might the speaker be feeling but not directly expressing?
- Consider: What social constraints might affect how they communicate?
- Consider: What would a reasonable person in their position actually mean?

**Scenario:**
{situation}

**Speaker:** {speaker_role}
**Listener:** {listener_role}

**Utterance:** "{utterance}"

**Perspective-taking questions to consider:**
1. If you were the {speaker_role}, what would you really be feeling in this situation?
2. Why might the {speaker_role} choose these particular words instead of expressing themselves directly?
3. What social or professional pressures might influence how the {speaker_role} communicates?

**Task:** Based on the full context AND your perspective-taking analysis, identify:
1. The speaker's underlying emotion (from: joy, trust, fear, surprise, sadness, disgust, anger, anticipation)
2. The pragmatic subtype (from: sarcasm-irony, mixed-signals, passive-aggression, deflection-misdirection, strategic-politeness)

Respond in JSON format:
```json
{{
  "emotion": "<emotion>",
  "subtype": "<subtype>",
  "confidence": <0.0-1.0>,
  "perspective_analysis": "<what you understood from taking their perspective>",
  "reasoning": "<brief explanation>"
}}
```"""


# INT-3: Subtype awareness intervention
SUBTYPE_AWARENESS_PROMPT = """You are analyzing a workplace conversation to identify the underlying emotion and pragmatic strategy.

**IMPORTANT: Review these pragmatic strategy definitions before answering:**

1. **Sarcasm-irony**: Speaker says the opposite of what they mean, often with exaggerated tone or obvious contradiction to context. The literal meaning directly contradicts the intended meaning.

2. **Mixed-signals**: Speaker's verbal and non-verbal cues conflict, or their words suggest multiple possible interpretations. There's genuine ambiguity in what they mean.

3. **Passive-aggression**: Speaker expresses negative feelings indirectly through subtle actions, omissions, or veiled comments rather than direct confrontation.

4. **Deflection-misdirection**: Speaker avoids a topic or question by redirecting attention elsewhere, changing the subject, or giving an unrelated response.

5. **Strategic-politeness**: Speaker uses polite language strategically to soften criticism, make requests more palatable, or maintain face while conveying potentially negative information.

**Scenario:**
{situation}

**Speaker:** {speaker_role}
**Listener:** {listener_role}

**Utterance:** "{utterance}"

**Task:** Match the speaker's communication to ONE of the above categories based on:
- What the literal words say vs. what they likely mean
- The relationship between utterance and context
- The social function being served

Also identify the underlying emotion (from: joy, trust, fear, surprise, sadness, disgust, anger, anticipation)

Respond in JSON format:
```json
{{
  "emotion": "<emotion>",
  "subtype": "<subtype>",
  "confidence": <0.0-1.0>,
  "subtype_match_reasoning": "<why this subtype matches>",
  "eliminated_subtypes": ["<subtypes that don't fit and why>"]
}}
```"""


# INT-5: Counterfactual prompting intervention
COUNTERFACTUAL_PROMPT = """You are analyzing a workplace conversation to identify the underlying emotion and pragmatic strategy.

**IMPORTANT: Use counterfactual reasoning before answering.**

**Scenario:**
{situation}

**Speaker:** {speaker_role}
**Listener:** {listener_role}

**Utterance:** "{utterance}"

**Counterfactual Analysis:**
Consider these alternative interpretations and rule them out:

1. **If this were sincere/literal**: What would the speaker need to believe for the utterance to be taken at face value? Does the context support this?

2. **If this were sarcasm**: Is the speaker saying the opposite of what they mean? Is there an obvious contradiction between words and context?

3. **If this were passive-aggressive**: Is the speaker indirectly expressing negativity while maintaining plausible deniability?

4. **If this were deflection**: Is the speaker avoiding something by changing the topic or giving an unrelated response?

5. **If this were strategic politeness**: Is the speaker using politeness to soften something negative or make a difficult request?

For each alternative, briefly note why it does or doesn't fit.

**Task:** Based on your counterfactual analysis, identify:
1. The speaker's underlying emotion (from: joy, trust, fear, surprise, sadness, disgust, anger, anticipation)
2. The pragmatic subtype (from: sarcasm-irony, mixed-signals, passive-aggression, deflection-misdirection, strategic-politeness)

Respond in JSON format:
```json
{{
  "emotion": "<emotion>",
  "subtype": "<subtype>",
  "confidence": <0.0-1.0>,
  "counterfactual_analysis": {{
    "if_sincere": "<ruling in or out>",
    "if_sarcasm": "<ruling in or out>",
    "if_passive_aggressive": "<ruling in or out>",
    "if_deflection": "<ruling in or out>",
    "if_strategic_politeness": "<ruling in or out>"
  }},
  "reasoning": "<why chosen interpretation is best>"
}}
```"""


# Mapping of intervention types to prompts
INTERVENTION_PROMPTS = {
    "baseline": BASE_PROMPT,
    "INT-2": PERSPECTIVE_TAKING_PROMPT,
    "INT-3": SUBTYPE_AWARENESS_PROMPT,
    "INT-5": COUNTERFACTUAL_PROMPT,
}

# Short names for reporting
INTERVENTION_NAMES = {
    "baseline": "Baseline",
    "INT-2": "Perspective-Taking",
    "INT-3": "Subtype Awareness",
    "INT-5": "Counterfactual",
}


def get_intervention_prompt(
    intervention_type: str,
    scenario: dict[str, Any],
) -> str:
    """
    Get the formatted prompt for a given intervention.

    Args:
        intervention_type: One of 'baseline', 'INT-2', 'INT-3', 'INT-5'
        scenario: Scenario dict with situation, roles, utterance

    Returns:
        Formatted prompt string
    """
    if intervention_type not in INTERVENTION_PROMPTS:
        raise ValueError(f"Unknown intervention type: {intervention_type}")

    template = INTERVENTION_PROMPTS[intervention_type]

    # Extract scenario fields
    situation = scenario.get("situation", scenario.get("sd_situation", ""))
    speaker_role = scenario.get("speaker_role", scenario.get("sd_speaker_role", "speaker"))
    listener_role = scenario.get("listener_role", scenario.get("sd_listener_role", "listener"))
    utterance = scenario.get("utterance", scenario.get("sd_utterance", ""))

    return template.format(
        situation=situation,
        speaker_role=speaker_role,
        listener_role=listener_role,
        utterance=utterance,
    )


def get_all_intervention_prompts(
    scenario: dict[str, Any],
) -> dict[str, str]:
    """
    Get all intervention prompts for a scenario.

    Args:
        scenario: Scenario dict

    Returns:
        Dict mapping intervention type to formatted prompt
    """
    return {
        itype: get_intervention_prompt(itype, scenario)
        for itype in INTERVENTION_PROMPTS
    }


def analyze_intervention_effects(
    baseline_predictions: list[dict[str, Any]],
    intervention_predictions: dict[str, list[dict[str, Any]]],
    ground_truth: list[dict[str, Any]],
) -> dict[str, InterventionResult]:
    """
    Analyze the effect of each intervention compared to baseline.

    Args:
        baseline_predictions: Predictions with baseline prompt
        intervention_predictions: Dict mapping intervention type to predictions
        ground_truth: Ground truth labels

    Returns:
        Dict mapping intervention type to InterventionResult
    """
    import numpy as np
    from scipy import stats

    # Compute baseline accuracy
    baseline_correct = []
    for pred, truth in zip(baseline_predictions, ground_truth):
        pred_label = pred.get("subtype", pred.get("prediction", {}).get("subtype", ""))
        true_label = truth.get("subtype", truth.get("ground_truth", {}).get("subtype", ""))
        baseline_correct.append(1 if pred_label == true_label else 0)

    baseline_acc = np.mean(baseline_correct) if baseline_correct else 0

    results = {}

    for itype, predictions in intervention_predictions.items():
        if itype == "baseline":
            continue

        # Compute intervention accuracy
        int_correct = []
        for pred, truth in zip(predictions, ground_truth):
            pred_label = pred.get("subtype", pred.get("prediction", {}).get("subtype", ""))
            true_label = truth.get("subtype", truth.get("ground_truth", {}).get("subtype", ""))
            int_correct.append(1 if pred_label == true_label else 0)

        int_acc = np.mean(int_correct) if int_correct else 0
        improvement = int_acc - baseline_acc

        # McNemar's test for significance
        # Count discordant pairs
        b = sum(1 for bc, ic in zip(baseline_correct, int_correct) if bc == 1 and ic == 0)
        c = sum(1 for bc, ic in zip(baseline_correct, int_correct) if bc == 0 and ic == 1)

        if b + c > 0:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2, 1)
        else:
            p_value = 1.0

        results[itype] = InterventionResult(
            intervention_type=itype,
            original_accuracy=baseline_acc,
            intervention_accuracy=int_acc,
            improvement=improvement,
            n_scenarios=len(int_correct),
            significant=p_value < 0.05,
            p_value=p_value,
        )

    return results


def generate_intervention_report(
    results: dict[str, InterventionResult],
) -> str:
    """
    Generate a text report summarizing intervention effects.

    Args:
        results: Dict of intervention results

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "INTERVENTION STUDY RESULTS",
        "=" * 60,
        "",
        f"Baseline Accuracy: {results[list(results.keys())[0]].original_accuracy:.1%}",
        "",
        "-" * 60,
        f"{'Intervention':<25} {'Accuracy':<12} {'Δ':<10} {'p-value':<10} {'Sig?':<5}",
        "-" * 60,
    ]

    for itype, result in results.items():
        sig_marker = "*" if result.significant else ""
        lines.append(
            f"{INTERVENTION_NAMES.get(itype, itype):<25} "
            f"{result.intervention_accuracy:.1%}        "
            f"{result.improvement:+.1%}      "
            f"{result.p_value:.4f}     "
            f"{sig_marker}"
        )

    lines.extend([
        "-" * 60,
        "",
        "* Significant at p < 0.05 (McNemar's test)",
        "",
    ])

    # Key findings
    best_intervention = max(results.items(), key=lambda x: x[1].improvement)
    if best_intervention[1].improvement > 0:
        lines.append(f"Best intervention: {INTERVENTION_NAMES.get(best_intervention[0], best_intervention[0])} "
                     f"(+{best_intervention[1].improvement:.1%})")
    else:
        lines.append("No intervention improved over baseline.")

    return "\n".join(lines)
