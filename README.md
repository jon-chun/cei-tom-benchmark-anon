# CEI-ToM: Contextual Emotional Inference for Theory of Mind

**Anonymous Repository for CogSci 2026 Submission**

*Pragmatic Inefficiency in Language Models: Compositional Integration Failures Across Social Hierarchies*

---

## Overview

This repository contains the CEI benchmark, analysis code, and pre-computed results for evaluating pragmatic reasoning in Large Language Models. The benchmark tests whether LLMs can integrate contextual features (situation, speaker roles, power dynamics) with utterance meaning to infer emotional states from indirect communication.

### Key Findings

| Finding | Result |
|---------|--------|
| **Human baseline accuracy** | 82% (Fleiss' κ = 0.31) |
| **Best LLM accuracy** | 68% (GPT-5-mini) |
| **Utterance anchoring rate** | 69% of predictions ignore context |
| **Power asymmetry gap** | 10-11 point F1 drop for subordinate speakers |
| **Anger bias** | χ²(1) = 1461.78, p < .001, Cramér's V = 0.83 |
| **RSA alignment** | L0: κ = 0.72, L1: κ = 0.31 (literal > pragmatic) |

---

## Quick Start for Reviewers

All analyses can be reproduced from cached data **without API keys**.

```bash
# 1. Clone and setup
git clone [repository-url]
cd cei-tom
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Reproduce paper statistics (Table 2, Figures 1-3)
python scripts/run_venue.py --mode analyze_only

# 3. Regenerate publication figures
python scripts/run_venue.py --mode figures_only

# 4. Run tests
pytest tests/ -v
```

### Verify Key Statistics

```bash
# Anger bias chi-square test (Section 4, Figure 3)
cat outputs/cogsci_analysis/anger_bias_analysis.json | python -m json.tool

# Inter-annotator agreement (Table 1)
cat outputs/cogsci_analysis/fleiss_kappa_by_subtype.json | python -m json.tool

# Human-LLM correlation (Section 4, RQ3)
cat outputs/cogsci_analysis/human_llm_correlation.json | python -m json.tool
```

---

## Repository Structure

```
cei-tom/
├── data/
│   └── human-gold-aggregate/     # 300 scenarios with human annotations
│       ├── aggregate_sarcasm-irony.csv
│       ├── aggregate_mixed-signals.csv
│       ├── aggregate_strategic-politeness.csv
│       ├── aggregate_passive-aggression.csv
│       └── aggregate_deflection-misdirection.csv
│
├── outputs/
│   ├── main_inference/           # LLM predictions (11 models × 300 scenarios)
│   │   ├── gpt-5-mini/
│   │   ├── claude-haiku-4-5/
│   │   ├── grok-4-1-fast/
│   │   ├── gemini-3-flash-preview/
│   │   ├── mistralai_Mistral-Small-24B-Instruct-2501/
│   │   ├── meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo/
│   │   ├── google_gemma-3n-E4B-it/
│   │   ├── accounts_fireworks_models_qwen3-235b-a22b-instruct-2507/
│   │   ├── accounts_fireworks_models_deepseek-v3p2/
│   │   ├── accounts_fireworks_models_kimi-k2-instruct-0905/
│   │   └── accounts_fireworks_models_minimax-m2p1/
│   │
│   ├── cogsci_analysis/          # Statistical analysis results
│   │   ├── anger_bias_analysis.json
│   │   ├── confusion_matrix.json
│   │   ├── fleiss_kappa_by_subtype.json
│   │   ├── human_llm_correlation.json
│   │   └── valence_control_analysis.json
│   │
│   └── cogsci_visualize/figures/ # Publication figures (PDF)
│       ├── fig1_quadrant.pdf     # Human agreement × LLM accuracy
│       ├── fig2_power_gap.pdf    # Performance by power relation
│       ├── fig3_confusion.pdf    # Emotion confusion matrix
│       └── fig4_correlation.pdf  # Human-LLM difficulty correlation
│
├── src/
│   ├── core/                     # Shared infrastructure
│   │   ├── data_loader.py        # GoldScenario loading
│   │   ├── statistical_tests.py  # Bootstrap CIs, χ², Fleiss' κ
│   │   ├── model_interface.py    # Multi-provider LLM API wrapper
│   │   └── output_cache.py       # Response caching
│   │
│   ├── venue/                    # Paper-specific analysis
│   │   ├── rsa_comparison.py     # RSA L0/L1 listener models
│   │   ├── efficiency_metrics.py # Pragmatic efficiency computation
│   │   ├── compositional_tests.py # Cross-subtype recombination tests
│   │   ├── intervention_prompts.py # Perspective-taking prompts
│   │   └── visualization/        # Figure generation
│   │
│   └── pipeline/                 # Experiment execution
│       ├── cli.py                # Command-line interface
│       ├── orchestrator.py       # Pipeline coordination
│       └── stages/               # Individual pipeline stages
│
├── prompts/                      # LLM prompt templates
├── tests/                        # Unit tests
├── config/config.yml             # Configuration
└── scripts/run_venue.py          # Main entry point
```

---

## Data Format

### Scenario Format (CSV)

| Column | Description |
|--------|-------------|
| `id` | Unique identifier (e.g., `1`, `42`) |
| `sd_situation` | Background context establishing setting |
| `sd_utterance` | Ambiguous statement requiring pragmatic interpretation |
| `sd_speaker_role` | Speaker's organizational role |
| `sd_listener_role` | Listener's organizational role |
| `gold_standard` | Majority-vote emotion label (Plutchik's 8) |
| `sl_plutchik_primary_*` | Individual annotator emotion labels |
| `sl_v_*`, `sl_a_*`, `sl_d_*` | Valence, arousal, dominance ratings |
| `sl_confidence_*` | Annotator confidence ratings |

### Prediction Format (JSON)

```json
{
  "scenario_id": "sarcasm-irony_42",
  "model_id": "gpt-5-mini",
  "prediction": {
    "emotion": "anger",
    "confidence": 0.82
  },
  "latency_ms": 1250
}
```

---

## Reproducing Paper Results

### RQ1: Compositional Integration (Section 4)

```python
from src.venue.compositional_tests import compute_compositionality_score

# Cross-subtype recombination test
results = compute_compositionality_score(predictions, recombined_scenarios)
print(f"Accuracy drop: {results.original_acc - results.recombined_acc:.1%}")
# Expected: 13-14 percentage point drop
```

### RQ2: Power Asymmetry (Section 4, Figure 2)

```python
from src.venue.efficiency_metrics import compute_power_asymmetry

gap, cohens_d = compute_power_asymmetry(predictions, gold_data, scenarios)
print(f"Power gap: {gap.value:.2f}, d = {cohens_d.value:.2f}")
# Expected: 10-11 point F1 gap, d = 0.54-0.55
```

### RQ3: Difficulty Dissociation (Section 4, Figure 1)

```python
from src.core.statistical_tests import compute_human_llm_correlation

r, p = compute_human_llm_correlation(human_agreement, llm_accuracy)
print(f"Correlation: r = {r:.2f}, p = {p:.3f}")
# Expected: r = -0.32 (weakly negative)
```

### Anger Bias Analysis (Section 4, Figure 3)

```python
from src.core.statistical_tests import chi_square_goodness_of_fit

result = chi_square_goodness_of_fit(observed_anger=918, expected_anger=302)
print(f"χ²({result.df}) = {result.chi2:.2f}, p < {result.p_value:.3f}")
# Expected: χ²(1) = 1461.78, p < .001
```

---

## Running the Full Pipeline

### With API Keys (Full Reproduction)

```bash
# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Run full pipeline
python scripts/run_venue.py --mode full

# Dry run (validate without API calls)
python scripts/run_venue.py --dry-run
```

### Analysis Only (No API Keys Required)

```bash
# Reproduce all statistics from cached predictions
python scripts/run_venue.py --mode analyze_only

# Generate figures only
python scripts/run_venue.py --mode figures_only
```

---

## Configuration

Edit `config/config.yml` to customize:

```yaml
pipeline:
  mode: complete           # 'test' (1 model) or 'complete' (11 models)

venue26:
  human_baseline: 0.82     # Human accuracy from Table 2
  n_bootstrap: 1000        # Bootstrap resamples for CIs

reproducibility:
  seed: 42                 # Random seed for all operations
```

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/

# Skip tests requiring API access
pytest -m "not llm" tests/
```

---

## Paper-to-Code Mapping

| Paper Section | Code Location | Output Files |
|--------------|---------------|--------------|
| Table 1: Inter-Annotator Agreement | `src/core/statistical_tests.py:fleiss_kappa` | `fleiss_kappa_by_subtype.json` |
| Table 2: Model Performance | `src/pipeline/stages/cogsci_analysis.py` | `cogsci_stats.json` |
| Figure 1: Quadrant Plot | `src/pipeline/stages/cogsci_visualize.py` | `fig1_quadrant.pdf` |
| Figure 2: Power Gap | `src/pipeline/stages/cogsci_visualize.py` | `fig2_power_gap.pdf` |
| Figure 3: Confusion Matrix | `src/pipeline/stages/cogsci_visualize.py` | `fig3_confusion.pdf` |
| §3.4: Compositional Tests | `src/venue/compositional_tests.py` | — |
| §3.5: RSA Baselines | `src/venue/rsa_comparison.py` | — |
| §3.5: Interventions | `src/venue/intervention_prompts.py` | — |
| §4: Anger Bias χ² Test | `src/core/statistical_tests.py:chi_square_goodness_of_fit` | `anger_bias_analysis.json` |

---

## Requirements

- Python 3.10+
- Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyyaml`, `click`, `rich`, `pydantic`
- Optional (for LLM inference): `openai`, `anthropic`, `google-genai`

```bash
pip install -e .           # Core dependencies
pip install -e ".[llm]"    # With LLM API support
pip install -e ".[dev]"    # With dev tools (pytest, ruff, mypy)
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Citation

```bibtex
@inproceedings{anonymous2026pragmatic,
  title={Pragmatic Inefficiency in Language Models:
         Compositional Integration Failures Across Social Hierarchies},
  author={Anonymous},
  booktitle={Proceedings of the 48th Annual Conference of the
             Cognitive Science Society},
  year={2026},
  note={Under review}
}
```
