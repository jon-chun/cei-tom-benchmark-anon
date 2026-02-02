"""
Data loading utilities for CEI-ToM gold standard data.

This module handles loading the human-annotated aggregate CSV files
from the gold data directory.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GoldScenario:
    """A human-annotated gold standard scenario."""

    id: str
    original_id: str
    subtype: str
    situation: str
    utterance: str
    speaker_role: str
    listener_role: str
    power_relation: str
    gold_emotion: str
    annotator_labels: dict[str, str | None] = field(default_factory=dict)
    annotator_vad: dict[str, dict[str, str | None]] = field(default_factory=dict)
    annotator_confidence: dict[str, str | None] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "original_id": self.original_id,
            "subtype": self.subtype,
            "situation": self.situation,
            "utterance": self.utterance,
            "speaker_role": self.speaker_role,
            "listener_role": self.listener_role,
            "power_relation": self.power_relation,
            "gold_emotion": self.gold_emotion,
            "annotator_labels": self.annotator_labels,
            "annotator_vad": self.annotator_vad,
            "annotator_confidence": self.annotator_confidence,
            "notes": self.notes,
        }


def infer_power_relation(speaker_role: str, listener_role: str) -> str:
    """Infer power relation from speaker and listener roles.

    Returns one of: 'peer', 'higher_to_lower', 'lower_to_higher'
    """
    speaker = speaker_role.lower().strip()
    listener = listener_role.lower().strip()

    # Keywords indicating higher power
    higher_power = {
        'senior', 'manager', 'director', 'supervisor', 'lead', 'chief',
        'executive', 'boss', 'owner', 'president', 'vp', 'head', 'ceo',
        'cto', 'cfo', 'partner', 'principal', 'dean', 'professor'
    }
    # Keywords indicating lower power
    lower_power = {
        'junior', 'intern', 'trainee', 'assistant', 'new', 'entry',
        'apprentice', 'student', 'rookie', 'associate', 'aide', 'clerk'
    }

    def has_higher_power(role: str) -> bool:
        return any(kw in role for kw in higher_power)

    def has_lower_power(role: str) -> bool:
        return any(kw in role for kw in lower_power)

    speaker_high = has_higher_power(speaker)
    speaker_low = has_lower_power(speaker)
    listener_high = has_higher_power(listener)
    listener_low = has_lower_power(listener)

    if speaker_high and listener_low:
        return 'higher_to_lower'
    elif speaker_low and listener_high:
        return 'lower_to_higher'
    elif speaker_high and not listener_high:
        return 'higher_to_lower'
    elif listener_high and not speaker_high:
        return 'lower_to_higher'
    else:
        return 'peer'


def load_gold_scenarios(data_dir: Path | str) -> list[GoldScenario]:
    """Load human-annotated gold standard scenarios from aggregate CSVs.

    Args:
        data_dir: Path to directory containing aggregate_*.csv files

    Returns:
        List of GoldScenario objects

    Expected CSV format:
    - Files named 'aggregate_<subtype>.csv'
    - Columns: id, sd_listener_role, sd_situation, sd_speaker_role, sd_utterance
    - Annotator columns: sl_plutchik_primary_<Name>, sl_v_<Name>, sl_a_<Name>,
      sl_d_<Name>, sl_confidence_<Name>
    - gold_standard: final QA'd emotion label
    """
    data_dir = Path(data_dir)
    scenarios: list[GoldScenario] = []

    if not data_dir.exists():
        logger.warning(f"Gold data directory does not exist: {data_dir}")
        return scenarios

    csv_files = sorted(data_dir.glob("aggregate_*.csv"))
    if not csv_files:
        logger.warning(f"No aggregate_*.csv files found in {data_dir}")
        return scenarios

    for csv_file in csv_files:
        # Extract subtype from filename
        match = re.match(r'aggregate_(.+)\.csv', csv_file.name)
        if not match:
            logger.debug(f"Skipping file with unexpected name format: {csv_file.name}")
            continue

        subtype = match.group(1)

        # Skip non-data files
        if 'key' in subtype.lower() or 'flagged' in subtype.lower():
            logger.debug(f"Skipping metadata file: {csv_file.name}")
            continue

        file_count = 0
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Skip rows with missing essential data
                if not row.get('id') or not row.get('sd_utterance'):
                    continue

                # Extract individual annotator data
                annotator_labels: dict[str, str | None] = {}
                annotator_vad: dict[str, dict[str, str | None]] = {}
                annotator_confidence: dict[str, str | None] = {}

                for col, val in row.items():
                    val_clean = val.strip() if val else None
                    if col.startswith('sl_plutchik_primary_'):
                        annotator = col.replace('sl_plutchik_primary_', '')
                        annotator_labels[annotator] = val_clean
                    elif col.startswith('sl_v_'):
                        annotator = col.replace('sl_v_', '')
                        annotator_vad.setdefault(annotator, {})['valence'] = val_clean
                    elif col.startswith('sl_a_'):
                        annotator = col.replace('sl_a_', '')
                        annotator_vad.setdefault(annotator, {})['arousal'] = val_clean
                    elif col.startswith('sl_d_'):
                        annotator = col.replace('sl_d_', '')
                        annotator_vad.setdefault(annotator, {})['dominance'] = val_clean
                    elif col.startswith('sl_confidence_'):
                        annotator = col.replace('sl_confidence_', '')
                        annotator_confidence[annotator] = val_clean

                speaker_role = row.get('sd_speaker_role', '').strip()
                listener_role = row.get('sd_listener_role', '').strip()

                scenario = GoldScenario(
                    id=f"{subtype}_{row['id']}",
                    original_id=row['id'],
                    subtype=subtype,
                    situation=row.get('sd_situation', '').strip(),
                    utterance=row.get('sd_utterance', '').strip(),
                    speaker_role=speaker_role,
                    listener_role=listener_role,
                    power_relation=infer_power_relation(speaker_role, listener_role),
                    gold_emotion=row.get('gold_standard', '').strip(),
                    annotator_labels=annotator_labels,
                    annotator_vad=annotator_vad,
                    annotator_confidence=annotator_confidence,
                    notes=row.get('Notes', '').strip(),
                )

                scenarios.append(scenario)
                file_count += 1

        logger.info(f"Loaded {file_count} scenarios from {csv_file.name}")

    logger.info(f"Total: {len(scenarios)} gold scenarios loaded from {data_dir}")
    return scenarios


def load_gold_scenarios_as_dicts(data_dir: Path | str) -> list[dict[str, Any]]:
    """Load gold scenarios and return as list of dictionaries.

    Convenience wrapper for code expecting dict format.
    """
    scenarios = load_gold_scenarios(data_dir)
    return [s.to_dict() for s in scenarios]
