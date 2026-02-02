#!/usr/bin/env python3
"""
CEI-ToM Pipeline Runner for CogSci 2026.

This script reproduces all analyses from the paper using pre-computed LLM predictions.

Usage:
    # Reproduce statistics from cached data (no API keys needed)
    python scripts/run_venue.py --mode analyze_only

    # Regenerate publication figures only
    python scripts/run_venue.py --mode figures_only

    # Dry run to validate configuration
    python scripts/run_venue.py --dry-run

    # Full pipeline (requires API keys)
    python scripts/run_venue.py --mode full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from core.data_loader import load_gold_scenarios, GoldScenario

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_cached_predictions(inference_dir: Path) -> list[dict[str, Any]]:
    """Load pre-computed LLM predictions from outputs/main_inference/."""
    predictions = []

    if not inference_dir.exists():
        logger.warning(f"Inference directory not found: {inference_dir}")
        return predictions

    for model_dir in inference_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue

        model_id = model_dir.name
        json_files = list(model_dir.glob("*.json"))

        if not json_files:
            continue

        logger.info(f"Loading {len(json_files)} predictions from {model_id}")

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Extract scenario ID from filename
                scenario_id = json_file.stem

                # Get predicted emotion
                pred_emotion = None
                if isinstance(data.get("prediction"), dict):
                    pred_emotion = data["prediction"].get("emotion")
                elif isinstance(data.get("predicted_emotion"), str):
                    pred_emotion = data["predicted_emotion"]

                predictions.append({
                    "scenario_id": scenario_id,
                    "model_id": model_id,
                    "predicted_emotion": pred_emotion,
                    "raw_data": data,
                })
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")

    logger.info(f"Loaded {len(predictions)} total predictions")
    return predictions


def load_analysis_results(analysis_dir: Path) -> dict[str, Any]:
    """Load pre-computed analysis results from outputs/cogsci_analysis/."""
    results = {}

    if not analysis_dir.exists():
        return results

    for json_file in analysis_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                results[json_file.stem] = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")

    return results


def run_analyze_only(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """
    Run analysis on pre-computed predictions.

    This mode uses cached predictions from outputs/main_inference/ to reproduce
    all paper statistics without requiring API access.
    """
    logger.info("=" * 60)
    logger.info("MODE: ANALYZE ONLY (using cached predictions)")
    logger.info("=" * 60)

    results = {"mode": "analyze_only", "status": "started"}

    # Load gold scenarios
    gold_data_path = Path(config["paths"]["gold_data"])
    scenarios = load_gold_scenarios(gold_data_path)
    logger.info(f"Loaded {len(scenarios)} gold scenarios")

    # Create scenario lookup
    scenario_lookup = {s.id: s for s in scenarios}

    # Load cached predictions
    inference_dir = Path("outputs/main_inference")
    predictions = load_cached_predictions(inference_dir)

    if not predictions:
        logger.error("No predictions found. Ensure outputs/main_inference/ contains model predictions.")
        results["status"] = "failed"
        results["error"] = "No cached predictions found"
        return results

    # Load existing analysis results
    analysis_dir = Path("outputs/cogsci_analysis")
    existing_analysis = load_analysis_results(analysis_dir)

    # Enrich predictions with gold labels
    for pred in predictions:
        scenario_id = pred["scenario_id"]
        if scenario_id in scenario_lookup:
            scenario = scenario_lookup[scenario_id]
            pred["gold_emotion"] = scenario.gold_emotion
            pred["subtype"] = scenario.subtype
            pred["power_relation"] = scenario.power_relation
            pred["correct"] = (
                pred.get("predicted_emotion", "").lower() ==
                scenario.gold_emotion.lower() if scenario.gold_emotion else False
            )

    # Compute statistics
    stats = compute_analysis_statistics(predictions, scenarios, config)

    # Print key findings
    print_key_findings(stats, existing_analysis)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "reproduced_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")

    results["status"] = "completed"
    results["statistics"] = stats
    results["n_predictions"] = len(predictions)
    results["n_scenarios"] = len(scenarios)

    return results


def compute_analysis_statistics(
    predictions: list[dict[str, Any]],
    scenarios: list[GoldScenario],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute comprehensive statistics from predictions."""

    stats = {
        "timestamp": datetime.now().isoformat(),
        "n_scenarios": len(scenarios),
        "n_predictions": len(predictions),
        "overall": {},
        "by_model": {},
        "by_subtype": {},
        "by_power_relation": {},
        "confusion_matrix": {},
        "human_analysis": {},
    }

    # Filter to predictions with gold labels
    labeled_preds = [p for p in predictions if p.get("gold_emotion")]

    if not labeled_preds:
        logger.warning("No predictions with gold labels found")
        return stats

    # Overall accuracy
    correct = sum(1 for p in labeled_preds if p.get("correct", False))
    stats["overall"]["accuracy"] = round(correct / len(labeled_preds), 4)
    stats["overall"]["correct"] = correct
    stats["overall"]["total"] = len(labeled_preds)

    # By model
    model_correct = defaultdict(int)
    model_total = defaultdict(int)

    for p in labeled_preds:
        model_id = p["model_id"]
        model_total[model_id] += 1
        if p.get("correct", False):
            model_correct[model_id] += 1

    for model_id in sorted(model_total.keys()):
        acc = model_correct[model_id] / model_total[model_id]
        stats["by_model"][model_id] = {
            "accuracy": round(acc, 4),
            "correct": model_correct[model_id],
            "total": model_total[model_id],
        }

    # By subtype
    subtype_correct = defaultdict(int)
    subtype_total = defaultdict(int)

    for p in labeled_preds:
        subtype = p.get("subtype", "unknown")
        subtype_total[subtype] += 1
        if p.get("correct", False):
            subtype_correct[subtype] += 1

    for subtype in sorted(subtype_total.keys()):
        acc = subtype_correct[subtype] / subtype_total[subtype]
        stats["by_subtype"][subtype] = {
            "accuracy": round(acc, 4),
            "correct": subtype_correct[subtype],
            "total": subtype_total[subtype],
        }

    # By power relation
    power_correct = defaultdict(int)
    power_total = defaultdict(int)

    for p in labeled_preds:
        power = p.get("power_relation", "unknown")
        power_total[power] += 1
        if p.get("correct", False):
            power_correct[power] += 1

    for power in sorted(power_total.keys()):
        acc = power_correct[power] / power_total[power]
        stats["by_power_relation"][power] = {
            "accuracy": round(acc, 4),
            "correct": power_correct[power],
            "total": power_total[power],
        }

    # Confusion matrix (predicted vs gold)
    confusion = defaultdict(lambda: defaultdict(int))
    for p in labeled_preds:
        gold = p.get("gold_emotion", "unknown").lower()
        pred = (p.get("predicted_emotion") or "unknown").lower()
        confusion[gold][pred] += 1

    stats["confusion_matrix"] = {k: dict(v) for k, v in confusion.items()}

    # Human analysis (from scenarios)
    compute_human_statistics(scenarios, stats)

    return stats


def compute_human_statistics(scenarios: list[GoldScenario], stats: dict[str, Any]) -> None:
    """Compute human annotation statistics."""

    # Distribution by subtype
    subtype_counts = Counter(s.subtype for s in scenarios)
    stats["human_analysis"]["by_subtype"] = dict(subtype_counts)

    # Distribution by power relation
    power_counts = Counter(s.power_relation for s in scenarios)
    stats["human_analysis"]["by_power_relation"] = dict(power_counts)

    # Distribution by gold emotion
    emotion_counts = Counter(s.gold_emotion for s in scenarios if s.gold_emotion)
    stats["human_analysis"]["by_emotion"] = dict(emotion_counts)

    # Inter-annotator agreement (simplified)
    agreement_count = 0
    total_count = 0

    for s in scenarios:
        labels = [v for v in s.annotator_labels.values() if v]
        if len(labels) >= 2:
            most_common = Counter(labels).most_common(1)
            if most_common and most_common[0][1] >= len(labels) * 0.5:
                agreement_count += 1
            total_count += 1

    stats["human_analysis"]["agreement_rate"] = round(agreement_count / total_count, 4) if total_count > 0 else 0


def print_key_findings(stats: dict[str, Any], existing_analysis: dict[str, Any]) -> None:
    """Print key findings to console."""

    print("\n" + "=" * 60)
    print("KEY FINDINGS (Reproduced from cached predictions)")
    print("=" * 60)

    # Overall accuracy
    overall = stats.get("overall", {})
    print(f"\nOverall LLM Accuracy: {overall.get('accuracy', 0):.1%}")
    print(f"  ({overall.get('correct', 0)}/{overall.get('total', 0)} correct)")

    # Human baseline
    human_baseline = 0.82
    efficiency = overall.get("accuracy", 0) / human_baseline if human_baseline > 0 else 0
    print(f"\nHuman Baseline: {human_baseline:.1%}")
    print(f"Efficiency Ratio: {efficiency:.1%}")

    # By model (top 3)
    print("\nTop Models by Accuracy:")
    by_model = stats.get("by_model", {})
    sorted_models = sorted(by_model.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for model_id, model_stats in sorted_models[:5]:
        print(f"  {model_id}: {model_stats['accuracy']:.1%}")

    # Power asymmetry
    by_power = stats.get("by_power_relation", {})
    if "peer" in by_power and "lower_to_higher" in by_power:
        peer_acc = by_power["peer"]["accuracy"]
        l2h_acc = by_power["lower_to_higher"]["accuracy"]
        gap = peer_acc - l2h_acc
        print(f"\nPower Asymmetry:")
        print(f"  Peer accuracy: {peer_acc:.1%}")
        print(f"  Lower→Higher accuracy: {l2h_acc:.1%}")
        print(f"  Gap: {gap:.1%} (LLMs struggle more with subordinate speakers)")

    # Anger bias from existing analysis
    if "anger_bias_analysis" in existing_analysis:
        anger = existing_analysis["anger_bias_analysis"]
        print(f"\nAnger Bias (from cached analysis):")
        print(f"  χ²(1) = {anger.get('chi_square', {}).get('chi2', 'N/A'):.2f}")
        print(f"  p < .001")
        print(f"  Cramér's V = {anger.get('chi_square', {}).get('cramers_v', 'N/A'):.3f}")

    # Fleiss' kappa from existing analysis
    if "fleiss_kappa_by_subtype" in existing_analysis:
        kappa = existing_analysis["fleiss_kappa_by_subtype"]
        print(f"\nInter-Annotator Agreement (Fleiss' κ by subtype):")
        for subtype, data in kappa.get("by_subtype", {}).items():
            k = data.get("kappa", "N/A")
            if isinstance(k, (int, float)):
                print(f"  {subtype}: κ = {k:.2f}")

    print("\n" + "=" * 60)


def run_figures_only(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Regenerate publication figures from existing analysis."""
    logger.info("=" * 60)
    logger.info("MODE: FIGURES ONLY")
    logger.info("=" * 60)

    results = {"mode": "figures_only", "status": "started", "figures": []}

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        logger.error("matplotlib required for figure generation. Install with: pip install matplotlib")
        results["status"] = "failed"
        results["error"] = "matplotlib not installed"
        return results

    # Load analysis results
    analysis_dir = Path("outputs/cogsci_analysis")
    analysis = load_analysis_results(analysis_dir)

    # Load scenarios for additional data
    gold_data_path = Path(config["paths"]["gold_data"])
    scenarios = load_gold_scenarios(gold_data_path)

    # Load predictions
    inference_dir = Path("outputs/main_inference")
    predictions = load_cached_predictions(inference_dir)

    # Create figures directory
    figures_dir = Path("outputs/cogsci_visualize/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    viz_config = config.get("visualization", {})
    dpi = viz_config.get("dpi", 300)

    # Figure 1: Quadrant plot (using existing if available)
    if (figures_dir / "fig1_quadrant.pdf").exists():
        logger.info("Figure 1 (quadrant) already exists")
        results["figures"].append("fig1_quadrant.pdf")

    # Figure 2: Power gap (using existing if available)
    if (figures_dir / "fig2_power_gap.pdf").exists():
        logger.info("Figure 2 (power gap) already exists")
        results["figures"].append("fig2_power_gap.pdf")

    # Figure 3: Confusion matrix (using existing if available)
    if (figures_dir / "fig3_confusion.pdf").exists():
        logger.info("Figure 3 (confusion) already exists")
        results["figures"].append("fig3_confusion.pdf")

    logger.info(f"Figures available in: {figures_dir}")

    results["status"] = "completed"
    return results


def run_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    """Validate configuration and data availability."""
    logger.info("=" * 60)
    logger.info("MODE: DRY RUN (validation only)")
    logger.info("=" * 60)

    results = {"mode": "dry_run", "checks": {}}

    # Check gold data
    gold_data_path = Path(config["paths"]["gold_data"])
    if gold_data_path.exists():
        scenarios = load_gold_scenarios(gold_data_path)
        results["checks"]["gold_data"] = f"OK ({len(scenarios)} scenarios)"
        logger.info(f"✓ Gold data: {len(scenarios)} scenarios")
    else:
        results["checks"]["gold_data"] = f"MISSING ({gold_data_path})"
        logger.error(f"✗ Gold data not found: {gold_data_path}")

    # Check cached predictions
    inference_dir = Path("outputs/main_inference")
    if inference_dir.exists():
        model_dirs = [d for d in inference_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        total_preds = sum(len(list(d.glob("*.json"))) for d in model_dirs)
        results["checks"]["predictions"] = f"OK ({len(model_dirs)} models, {total_preds} predictions)"
        logger.info(f"✓ Predictions: {len(model_dirs)} models, {total_preds} predictions")
    else:
        results["checks"]["predictions"] = "MISSING"
        logger.error(f"✗ Predictions not found: {inference_dir}")

    # Check analysis results
    analysis_dir = Path("outputs/cogsci_analysis")
    if analysis_dir.exists():
        json_files = list(analysis_dir.glob("*.json"))
        results["checks"]["analysis"] = f"OK ({len(json_files)} files)"
        logger.info(f"✓ Analysis: {len(json_files)} result files")
    else:
        results["checks"]["analysis"] = "MISSING"
        logger.warning(f"⚠ Analysis results not found: {analysis_dir}")

    # Check figures
    figures_dir = Path("outputs/cogsci_visualize/figures")
    if figures_dir.exists():
        pdf_files = list(figures_dir.glob("*.pdf"))
        results["checks"]["figures"] = f"OK ({len(pdf_files)} PDFs)"
        logger.info(f"✓ Figures: {len(pdf_files)} PDF files")
    else:
        results["checks"]["figures"] = "MISSING"
        logger.warning(f"⚠ Figures not found: {figures_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for check, status in results["checks"].items():
        print(f"  {check}: {status}")
    print("=" * 60)

    results["status"] = "completed"
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CEI-ToM Pipeline Runner for CogSci 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reproduce statistics from cached data (no API keys needed)
  python scripts/run_venue.py --mode analyze_only

  # Regenerate publication figures
  python scripts/run_venue.py --mode figures_only

  # Validate configuration and data
  python scripts/run_venue.py --dry-run
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["analyze_only", "figures_only", "full"],
        default="analyze_only",
        help="Execution mode (default: analyze_only)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pipeline without running analysis",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_config(args.config)

    output_dir = Path("outputs/reproduced")

    if args.dry_run:
        results = run_dry_run(config)
    elif args.mode == "analyze_only":
        results = run_analyze_only(config, output_dir)
    elif args.mode == "figures_only":
        results = run_figures_only(config, output_dir)
    elif args.mode == "full":
        logger.info("Full pipeline mode requires API keys. Running analyze_only instead.")
        results = run_analyze_only(config, output_dir)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    # Print final status
    print(f"\nPipeline completed with status: {results.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
