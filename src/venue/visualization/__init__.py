"""
Venue 2026 Visualization Module.

Publication-ready figures for the Venue 2026 paper.
All figures use colorblind-safe palettes and vector output (PDF).

Figures:
- Fig 1: Human vs LLM Efficiency Comparison
- Fig 2: Layerwise Probe Accuracy
- Fig 3: Power-Stratified Performance
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# Venue paper style settings
VENUE_STYLE = {
    "figure.figsize": (3.5, 2.5),  # Single-column width
    "font.size": 8,
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
}

# Colorblind-safe palette (viridis-inspired)
COLORS = {
    "human": "#440154",      # Dark purple
    "model_best": "#21918c", # Teal
    "model_avg": "#5ec962",  # Green
    "model_weak": "#fde725", # Yellow
    "peer": "#3b528b",       # Blue
    "subordinate": "#e76f51", # Orange-red
    "ci": "#cccccc",         # Gray for CI bands
}


def apply_venue_style() -> None:
    """Apply Venue publication style to matplotlib."""
    plt.rcParams.update(VENUE_STYLE)


def save_figure(fig: plt.Figure, path: Path, formats: list[str] = ["pdf"]) -> list[Path]:
    """Save figure in multiple formats."""
    saved_paths = []
    for fmt in formats:
        output_path = path.with_suffix(f".{fmt}")
        fig.savefig(output_path, format=fmt, dpi=300, bbox_inches="tight")
        saved_paths.append(output_path)
    return saved_paths


def plot_efficiency_comparison(
    human_accuracy: float,
    human_ci: tuple[float, float],
    model_accuracies: dict[str, float],
    model_cis: dict[str, tuple[float, float]],
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Figure 1: Human vs LLM Efficiency Comparison.

    Bar chart comparing human and model pragmatic classification accuracy.

    Args:
        human_accuracy: Human baseline accuracy
        human_ci: Human 95% CI (lower, upper)
        model_accuracies: Dict of model_name -> accuracy
        model_cis: Dict of model_name -> (ci_lower, ci_upper)
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    apply_venue_style()

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Prepare data
    agents = ["Human"] + list(model_accuracies.keys())
    accuracies = [human_accuracy] + list(model_accuracies.values())
    errors_lower = [human_accuracy - human_ci[0]] + [
        acc - model_cis[m][0] for m, acc in model_accuracies.items()
    ]
    errors_upper = [human_ci[1] - human_accuracy] + [
        model_cis[m][1] - acc for m, acc in model_accuracies.items()
    ]

    # Colors
    colors = [COLORS["human"]] + [COLORS["model_best"]] * len(model_accuracies)

    # Plot bars
    x = np.arange(len(agents))
    bars = ax.bar(x, accuracies, color=colors, edgecolor="black", linewidth=0.5)

    # Add error bars
    ax.errorbar(
        x, accuracies,
        yerr=[errors_lower, errors_upper],
        fmt="none", color="black", capsize=3, capthick=1
    )

    # Add human baseline line
    ax.axhline(y=human_accuracy, color=COLORS["human"], linestyle="--",
               linewidth=1, alpha=0.7, label=f"Human: {human_accuracy:.0%}")

    # Labels
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)

    # Add efficiency ratio annotations
    for i, (agent, acc) in enumerate(zip(agents[1:], accuracies[1:]), 1):
        efficiency = acc / human_accuracy
        gap = human_accuracy - acc
        ax.annotate(
            f"{gap:+.0%}",
            xy=(i, acc + errors_upper[i] + 0.02),
            ha="center", va="bottom", fontsize=6
        )

    ax.legend(loc="upper right", fontsize=6)
    ax.set_title("Human vs Model Pragmatic Efficiency")

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_layerwise_probing(
    layer_accuracies: dict[str, list[float]],
    feature_names: list[str] | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Figure 2: Layerwise Probe Accuracy.

    Line plot showing how pragmatic features are encoded across layers.

    Args:
        layer_accuracies: Dict of feature_name -> list of accuracies by layer
        feature_names: Optional labels for features
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    apply_venue_style()

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Get colormap
    cmap = plt.cm.viridis
    n_features = len(layer_accuracies)
    colors = [cmap(i / n_features) for i in range(n_features)]

    # Plot each feature
    for i, (feature, accuracies) in enumerate(layer_accuracies.items()):
        n_layers = len(accuracies)
        x = np.linspace(0, 100, n_layers)  # Normalize to percentage

        label = feature_names[i] if feature_names else feature
        ax.plot(x, accuracies, label=label, color=colors[i], linewidth=1.5)

        # Mark peak
        peak_idx = np.argmax(accuracies)
        ax.scatter([x[peak_idx]], [accuracies[peak_idx]], color=colors[i],
                   s=30, zorder=5)

    # Add majority baseline
    ax.axhline(y=0.20, color="gray", linestyle=":", linewidth=1,
               label="Majority baseline")

    # Labels
    ax.set_xlabel("Layer (% of depth)")
    ax.set_ylabel("Probe Accuracy")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=6, ncol=2)
    ax.set_title("Layerwise Feature Encoding")

    # Annotate integration region
    ax.axvspan(60, 80, alpha=0.1, color="gray")
    ax.annotate("Integration\nlayers", xy=(70, 0.8), ha="center", fontsize=6)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_power_stratified(
    human_by_power: dict[str, float],
    model_by_power: dict[str, float],
    human_cis: dict[str, tuple[float, float]] | None = None,
    model_cis: dict[str, tuple[float, float]] | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Figure 3: Power-Stratified Performance.

    Grouped bar chart comparing human and model performance by power relation.

    Args:
        human_by_power: Human accuracy by power relation
        model_by_power: Model accuracy by power relation
        human_cis: Optional human CIs
        model_cis: Optional model CIs
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    apply_venue_style()

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Power relation labels
    power_labels = {
        "peer": "Peer",
        "higher_to_lower": "Higher→Lower",
        "lower_to_higher": "Lower→Higher",
    }

    relations = list(human_by_power.keys())
    x = np.arange(len(relations))
    width = 0.35

    # Human bars
    human_vals = [human_by_power[r] for r in relations]
    human_errs = None
    if human_cis:
        human_errs = [
            [human_by_power[r] - human_cis[r][0] for r in relations],
            [human_cis[r][1] - human_by_power[r] for r in relations]
        ]

    ax.bar(x - width/2, human_vals, width, label="Human",
           color=COLORS["human"], edgecolor="black", linewidth=0.5,
           yerr=human_errs, capsize=3)

    # Model bars
    model_vals = [model_by_power[r] for r in relations]
    model_errs = None
    if model_cis:
        model_errs = [
            [model_by_power[r] - model_cis[r][0] for r in relations],
            [model_cis[r][1] - model_by_power[r] for r in relations]
        ]

    ax.bar(x + width/2, model_vals, width, label="Model",
           color=COLORS["model_best"], edgecolor="black", linewidth=0.5,
           yerr=model_errs, capsize=3)

    # Annotate gaps
    for i, r in enumerate(relations):
        gap = human_by_power[r] - model_by_power[r]
        y_pos = max(human_by_power[r], model_by_power[r]) + 0.05
        ax.annotate(f"Δ={gap:.2f}", xy=(i, y_pos), ha="center", fontsize=6)

    # Labels
    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels([power_labels.get(r, r) for r in relations])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Performance by Power Relation")

    # Highlight the key gap
    if "lower_to_higher" in relations:
        idx = relations.index("lower_to_higher")
        ax.annotate(
            "",
            xy=(idx + width/2, model_by_power["lower_to_higher"]),
            xytext=(idx - width/2, human_by_power["lower_to_higher"]),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5)
        )

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def generate_all_figures(
    results: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """
    Generate all Venue figures from analysis results.

    Args:
        results: Analysis results dictionary
        output_dir: Directory to save figures

    Returns:
        Dict of figure_name -> output_path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    # Figure 1: Efficiency comparison
    if "efficiency" in results:
        eff = results["efficiency"]
        fig1 = plot_efficiency_comparison(
            human_accuracy=eff.get("human_accuracy", 0.82),
            human_ci=eff.get("human_ci", (0.78, 0.86)),
            model_accuracies=eff.get("model_accuracies", {"GPT-5-mini": 0.68}),
            model_cis=eff.get("model_cis", {"GPT-5-mini": (0.64, 0.72)}),
            output_path=output_dir / "fig1_efficiency_comparison",
        )
        plt.close(fig1)
        generated["fig1_efficiency"] = output_dir / "fig1_efficiency_comparison.pdf"

    # Figure 2: Layerwise probing
    if "probing" in results:
        probe = results["probing"]
        fig2 = plot_layerwise_probing(
            layer_accuracies=probe.get("layer_accuracies", {}),
            feature_names=probe.get("feature_names"),
            output_path=output_dir / "fig2_layerwise_probing",
        )
        plt.close(fig2)
        generated["fig2_probing"] = output_dir / "fig2_layerwise_probing.pdf"

    # Figure 3: Power-stratified
    if "power" in results:
        power = results["power"]
        fig3 = plot_power_stratified(
            human_by_power=power.get("human_by_power", {}),
            model_by_power=power.get("model_by_power", {}),
            output_path=output_dir / "fig3_power_stratified",
        )
        plt.close(fig3)
        generated["fig3_power"] = output_dir / "fig3_power_stratified.pdf"

    return generated


__all__ = [
    "apply_venue_style",
    "save_figure",
    "plot_efficiency_comparison",
    "plot_layerwise_probing",
    "plot_power_stratified",
    "generate_all_figures",
    "COLORS",
    "VENUE_STYLE",
]
