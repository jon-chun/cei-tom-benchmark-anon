# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

CEI-ToM (Contextual Emotional Inference for Theory of Mind) is a research benchmark for evaluating pragmatic reasoning in LLMs. It accompanies the CogSci 2026 paper "Pragmatic Inefficiency in Language Models: Compositional Integration Failures Across Social Hierarchies".

The benchmark tests whether LLMs can integrate contextual features with utterance meaning to infer emotional states from indirect communication, comparing LLM performance against human baselines (82% accuracy).

## Build & Development Commands

```bash
# Installation
pip install -e .             # Core dependencies
pip install -e ".[dev]"      # With dev tools (pytest, ruff, mypy)
pip install -e ".[llm]"      # With LLM API support

# Run tests
pytest tests/ -v
pytest -m "not llm" tests/   # Skip API tests

# Code quality
ruff check src tests
ruff format src tests
mypy src

# Pipeline execution (analysis only - no API keys needed)
python scripts/run_venue.py --mode analyze_only
python scripts/run_venue.py --mode figures_only
python scripts/run_venue.py --dry-run

# Full pipeline (requires API keys)
python scripts/run_venue.py --mode full
```

## Architecture

```
src/
├── core/                    # Shared infrastructure
│   ├── data_loader.py       # GoldScenario loading from CSVs
│   ├── statistical_tests.py # Bootstrap CIs, chi-square, Fleiss' κ
│   ├── model_interface.py   # Multi-provider LLM API wrapper
│   └── output_cache.py      # API response caching
├── venue/                   # Paper-specific analysis
│   ├── rsa_comparison.py    # RSA L0/L1 listener alignment
│   ├── efficiency_metrics.py # Pragmatic efficiency computation
│   ├── compositional_tests.py # Cross-subtype recombination
│   ├── intervention_prompts.py # Perspective-taking prompts
│   └── visualization/       # Publication figures
└── pipeline/                # Experiment execution
    ├── cli.py               # Command-line interface
    ├── orchestrator.py      # Pipeline coordination
    └── stages/              # Individual pipeline stages
```

## Key Data Types

- **GoldScenario** (`core/data_loader.py`): Input scenario with situation, utterance, speaker/listener roles, power relation, gold emotion
- **EfficiencyMetric** (`venue/efficiency_metrics.py`): Analysis results with CIs
- **RSAComparisonResult** (`venue/rsa_comparison.py`): L0/L1 alignment metrics

## Data Schema

- **300 scenarios** across 5 pragmatic subtypes
- **3 power relations**: peer, higher_to_lower, lower_to_higher
- **8 Plutchik emotions**: joy, trust, fear, surprise, sadness, disgust, anger, anticipation
- **3 annotators per scenario**

## Configuration

Primary config: `config/config.yml`

Key settings:
- `pipeline.mode`: `test` (1 model) or `complete` (11 models)
- `venue26.human_baseline`: 0.82
- `reproducibility.seed`: 42

## Key Outputs

- `outputs/main_inference/{model_id}/`: LLM predictions (pre-computed for 11 models)
- `outputs/cogsci_analysis/`: Statistical analyses
- `outputs/cogsci_visualize/figures/`: Publication figures (PDF)
- `data/human-gold-aggregate/`: Human annotations (300 scenarios × 3 annotators)
