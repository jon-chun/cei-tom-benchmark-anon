"""
Prompt templates for CEI benchmark evaluation.

Provides standardized prompts for evaluating scenarios across different models,
ensuring consistent input format for fair comparison.

Example:
    >>> from core.prompt_templates import format_cei_prompt
    >>> prompt = format_cei_prompt(
    ...     situation="In the office kitchen, a colleague asks about the project.",
    ...     utterance="Sure, I'll handle it, no problem at all.",
    ...     speaker_role="junior developer",
    ...     listener_role="manager"
    ... )
"""

from __future__ import annotations

from typing import Optional


# Main CEI evaluation prompt
CEI_EVALUATION_PROMPT = """You are evaluating a communication scenario. Based on the context, determine:
1. The pragmatic subtype (sarcasm_irony, mixed_signals, strategic_politeness, passive_aggression, deflection)
2. The listener's primary emotion (joy, trust, fear, surprise, sadness, disgust, anger, anticipation)
3. The listener's emotional state on VAD dimensions (valence, arousal, dominance) each from -1 to 1

IMPORTANT: You are predicting the LISTENER's emotional response to hearing this utterance, NOT the speaker's emotion.

SCENARIO:
Situation: {situation}
Speaker ({speaker_role}) says to Listener ({listener_role}):
"{utterance}"

Provide your response in this exact JSON format:
{{
    "subtype": "<one of: sarcasm_irony, mixed_signals, strategic_politeness, passive_aggression, deflection>",
    "emotion": "<one of: joy, trust, fear, surprise, sadness, disgust, anger, anticipation>",
    "valence": <float from -1 (very unpleasant) to 1 (very pleasant)>,
    "arousal": <float from -1 (very calm) to 1 (very excited)>,
    "dominance": <float from -1 (very controlled) to 1 (very in control)>,
    "confidence": <float from 0 to 1 indicating your confidence>
}}"""


# Intervention prompt templates (from ICML paper Section 3.5)
INTERVENTION_PROMPTS = {
    "INT-1_role_modeling": """Before interpreting this message, consider: The speaker is a {speaker_role} communicating with a {listener_role}. What power dynamics might influence how they express themselves?

{base_prompt}""",
    "INT-2_perspective_taking": """Put yourself in the listener's position. Given this situation, how would a typical person in this role interpret this message? What might the speaker really be feeling or trying to communicate?

{base_prompt}""",
    "INT-3_subtype_awareness": """This message may involve one of these communication patterns:
- Sarcasm/Irony: Saying the opposite of what you mean
- Strategic Politeness: Using polite language to mask criticism
- Passive Aggression: Indirect expression of hostility
- Mixed Signals: Conflicting emotional cues
- Deflection: Avoiding the topic through redirection
Consider which pattern, if any, applies here.

{base_prompt}""",
    "INT-4_chain_of_thought": """Think step by step:
1. What is the literal meaning of this utterance?
2. What is the situational context?
3. What is the relationship between speaker and listener?
4. Is there a mismatch between literal meaning and context?
5. What is the likely intended meaning?

Then provide your final answer.

{base_prompt}""",
    "INT-5_counterfactual": """Consider: If this message were meant sincerely, would it fit the situation? If not, what alternative interpretation makes more sense given the context?

{base_prompt}""",
}


def format_cei_prompt(
    situation: str,
    utterance: str,
    speaker_role: str,
    listener_role: str,
    intervention: Optional[str] = None,
) -> str:
    """
    Format a CEI scenario into an evaluation prompt.

    Args:
        situation: The situational context
        utterance: The speaker's utterance
        speaker_role: Role/relationship of the speaker
        listener_role: Role/relationship of the listener
        intervention: Optional intervention prompt key (e.g., "INT-1_role_modeling")

    Returns:
        Formatted prompt string ready for model evaluation
    """
    base_prompt = CEI_EVALUATION_PROMPT.format(
        situation=situation,
        utterance=utterance,
        speaker_role=speaker_role,
        listener_role=listener_role,
    )

    if intervention is None:
        return base_prompt

    if intervention not in INTERVENTION_PROMPTS:
        raise ValueError(
            f"Unknown intervention: {intervention}. "
            f"Available: {list(INTERVENTION_PROMPTS.keys())}"
        )

    return INTERVENTION_PROMPTS[intervention].format(
        base_prompt=base_prompt,
        speaker_role=speaker_role,
        listener_role=listener_role,
    )


def format_probing_prompt(
    situation: str,
    utterance: str,
    speaker_role: str,
    listener_role: str,
) -> str:
    """
    Format a minimal prompt for hidden state extraction.

    Uses a simpler format to minimize prompt influence on representations.

    Args:
        situation: The situational context
        utterance: The speaker's utterance
        speaker_role: Role/relationship of the speaker
        listener_role: Role/relationship of the listener

    Returns:
        Minimal prompt for probing
    """
    return f"""Context: {situation}
Speaker ({speaker_role}) to Listener ({listener_role}): "{utterance}"
"""


def get_intervention_names() -> list[str]:
    """Return list of available intervention names."""
    return list(INTERVENTION_PROMPTS.keys())


def get_intervention_description(intervention: str) -> str:
    """Get description of an intervention strategy."""
    descriptions = {
        "INT-1_role_modeling": "Explicit role and power dynamics consideration",
        "INT-2_perspective_taking": "Listener perspective emphasis",
        "INT-3_subtype_awareness": "Pragmatic subtype pattern awareness",
        "INT-4_chain_of_thought": "Step-by-step reasoning scaffold",
        "INT-5_counterfactual": "Sincerity counterfactual analysis",
    }
    return descriptions.get(intervention, "Unknown intervention")
