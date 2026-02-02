"""
Pipeline CLI commands.

Command-line interface for running the multi-stage pipeline.

Usage:
    cei-pipeline run --type facct26 --mode full
    cei-pipeline run --type icml26 --mode stage:main_inference
    cei-pipeline stages --type facct26
    cei-pipeline status --output-dir outputs
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pipeline import (
    StageRegistry,
    run_pipeline,
)

console = Console()


def setup_logging(level: str, rich_console: bool = True) -> None:
    """Configure logging with optional rich formatting."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    if rich_console:
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def load_pipeline_config(config_path: Path | None) -> dict:
    """Load pipeline configuration from YAML file."""
    if config_path is None:
        # Look for default locations
        candidates = [
            Path("config/config.yml"),
            Path("config/config.yaml"),
            Path("config.yml"),
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not config_path.exists():
        console.print("[yellow]Warning: No config file found, using defaults[/yellow]")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


@click.group()
@click.version_option(version="2.0.0", prog_name="cei-pipeline")
def main() -> None:
    """
    CEI-ToM Multi-Stage Pipeline - Research experiment orchestration.

    Run FAccT 2026 or ICML 2026 research pipelines with configurable
    stage execution, checkpoint/resume support, and progress tracking.

    \b
    Examples:
        cei-pipeline run --type facct26 --mode full
        cei-pipeline run --type icml26 --mode stage:main_inference
        cei-pipeline stages --type facct26
    """
    pass


@main.command()
@click.option(
    "--type",
    "-t",
    "pipeline_type",
    type=click.Choice(["facct26", "icml26", "venue26"], case_sensitive=False),
    default=None,
    help="Pipeline type (default: from config.yml).",
)
@click.option(
    "--mode",
    "-m",
    default=None,
    help="Execution mode: full, stage:<name>, from:<name>, to:<name>, range:<start>:<end>.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to configuration YAML file.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("outputs"),
    help="Output directory for results.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Simulate execution without making API calls.",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume from checkpoint if available.",
)
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging verbosity level.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress console output.",
)
@click.option(
    "--repair",
    is_flag=True,
    default=False,
    help="Repair invalid emotion labels using semantic mapping.",
)
@click.option(
    "--retry",
    "retry_models",
    multiple=True,
    default=None,
    help="Retry failed calls for specific model(s). Use multiple times for multiple models.",
)
@click.option(
    "--retry-all-failed",
    is_flag=True,
    default=False,
    help="Retry all models with error rates above threshold.",
)
@click.option(
    "--retry-all-invalid",
    is_flag=True,
    default=False,
    help="Retry ALL individual invalid responses (regardless of model error rate).",
)
@click.option(
    "--exclude-model",
    "exclude_models",
    multiple=True,
    default=None,
    help="Exclude specific model(s) from retry. Use multiple times for multiple models.",
)
@click.option(
    "--ablation",
    is_flag=True,
    default=False,
    help="Run chain-of-thought ablation study (optional side study).",
)
@click.option(
    "--ablation-scenarios",
    type=int,
    default=100,
    help="Number of scenarios for ablation study (default: 100).",
)
@click.option(
    "--ablation-models",
    "ablation_models",
    multiple=True,
    default=None,
    help="Models to test in ablation. Use multiple times for multiple models.",
)
def run(
    pipeline_type: str | None,
    mode: str | None,
    config: Path | None,
    output_dir: Path,
    dry_run: bool,
    resume: bool,
    log_level: str,
    quiet: bool,
    repair: bool,
    retry_models: tuple[str, ...],
    retry_all_failed: bool,
    retry_all_invalid: bool,
    exclude_models: tuple[str, ...],
    ablation: bool,
    ablation_scenarios: int,
    ablation_models: tuple[str, ...],
) -> None:
    """
    Run the pipeline.

    Execute all or selected stages of the FAccT 2026 or ICML 2026 pipeline.

    \b
    Examples:
        # Run full FAccT pipeline
        cei-pipeline run --type facct26 --mode full

        # Run single stage
        cei-pipeline run --type icml26 --mode stage:config_test

        # Resume from checkpoint
        cei-pipeline run --type facct26 --resume

        # Dry run (no API calls)
        cei-pipeline run --type facct26 --dry-run
    """
    setup_logging(log_level, rich_console=not quiet)
    logger = logging.getLogger("pipeline.cli")

    # Load configuration
    cfg = load_pipeline_config(config)

    # Get pipeline type from config or CLI
    if pipeline_type is None:
        pipeline_type = cfg.get("pipeline", {}).get("type", "facct26")

    # Get mode from config or CLI
    if mode is None:
        mode = cfg.get("pipeline", {}).get("mode", "full")

    # Validate pipeline type
    if pipeline_type not in ["facct26", "icml26", "venue26"]:
        console.print(f"[red]Invalid pipeline type:[/red] {pipeline_type}")
        console.print("Valid types: facct26, icml26, venue26")
        sys.exit(1)

    # Handle --repair and --retry flags (override mode)
    has_retry = retry_models or retry_all_failed or retry_all_invalid
    if repair or has_retry:
        # Build custom mode for repair/retry workflow
        # Note: Stage order is data_qa -> repair_data -> retry_inference
        # We run repair first (if requested), then retry (if requested)
        # After these operations, data_qa should be re-run to validate
        if repair and has_retry:
            # Repair -> Retry (data_qa will need separate run)
            mode = "range:repair_data:retry_inference"
        elif repair:
            # Just repair stage
            mode = "stage:repair_data"
        elif has_retry:
            # Just retry stage
            mode = "stage:retry_inference"

        # Store retry configuration in config for stages to access
        if "pipeline" not in cfg:
            cfg["pipeline"] = {}
        if "stages" not in cfg["pipeline"]:
            cfg["pipeline"]["stages"] = {}
        if "retry_inference" not in cfg["pipeline"]["stages"]:
            cfg["pipeline"]["stages"]["retry_inference"] = {}

        cfg["pipeline"]["stages"]["retry_inference"]["target_models"] = list(retry_models)
        cfg["pipeline"]["stages"]["retry_inference"]["retry_all_failed"] = retry_all_failed
        cfg["pipeline"]["stages"]["retry_inference"]["retry_all_invalid"] = retry_all_invalid
        cfg["pipeline"]["stages"]["retry_inference"]["exclude_models"] = list(exclude_models)

    # Handle --ablation flag
    if ablation:
        # If no mode specified, run ablation stage only
        if mode is None or mode == "full":
            mode = "stage:cot_ablation"

        # Store ablation configuration
        if "pipeline" not in cfg:
            cfg["pipeline"] = {}
        if "stages" not in cfg["pipeline"]:
            cfg["pipeline"]["stages"] = {}
        if "cot_ablation" not in cfg["pipeline"]["stages"]:
            cfg["pipeline"]["stages"]["cot_ablation"] = {}

        cfg["pipeline"]["stages"]["cot_ablation"]["enabled"] = True
        cfg["pipeline"]["stages"]["cot_ablation"]["n_scenarios"] = ablation_scenarios
        if ablation_models:
            cfg["pipeline"]["stages"]["cot_ablation"]["models"] = list(ablation_models)

    if not quiet:
        console.print("[bold blue]CEI-ToM Pipeline v2.0.0[/bold blue]")
        console.print()
        console.print(f"  Pipeline: [cyan]{pipeline_type}[/cyan]")
        console.print(f"  Mode:     [cyan]{mode}[/cyan]")
        console.print(f"  Output:   [cyan]{output_dir}[/cyan]")
        if dry_run:
            console.print("  [yellow]DRY RUN - No API calls will be made[/yellow]")
        if resume:
            console.print("  [yellow]Resuming from checkpoint if available[/yellow]")
        if repair:
            console.print("  [green]REPAIR MODE - Fixing invalid emotion labels[/green]")
        if retry_models:
            console.print(
                f"  [green]RETRY MODE - Retrying models: {', '.join(retry_models)}[/green]"
            )
        if retry_all_failed:
            console.print(
                "  [green]RETRY ALL FAILED - Retrying models above error threshold[/green]"
            )
        if retry_all_invalid:
            console.print(
                "  [green]RETRY ALL INVALID - Retrying all individual invalid responses[/green]"
            )
        if exclude_models:
            console.print(
                f"  [yellow]EXCLUDING models: {', '.join(exclude_models)}[/yellow]"
            )
        if ablation:
            console.print(
                f"  [magenta]ABLATION MODE - CoT study with {ablation_scenarios} scenarios[/magenta]"
            )
            if ablation_models:
                console.print(
                    f"  [magenta]Ablation models: {', '.join(ablation_models)}[/magenta]"
                )
        console.print()

    # Run pipeline
    try:
        result = run_pipeline(
            pipeline_type=pipeline_type,  # type: ignore[arg-type]
            config=cfg,
            mode=mode,
            output_dir=output_dir,
            dry_run=dry_run,
            resume=resume,
            verbose=not quiet,
            repair_mode=repair,
            retry_models=list(retry_models) if retry_models else None,
            retry_all_failed=retry_all_failed,
            retry_all_invalid=retry_all_invalid,
            exclude_models=list(exclude_models) if exclude_models else None,
        )

        if not quiet:
            console.print()
            if result.success:
                console.print("[bold green]Pipeline completed successfully![/bold green]")
            else:
                console.print("[bold red]Pipeline failed![/bold red]")

            console.print()
            console.print(f"  Stages executed: {result.stages_executed}")
            console.print(f"  Stages failed:   {result.stages_failed}")
            console.print(f"  Stages skipped:  {result.stages_skipped}")
            console.print(f"  Total duration:  {result.total_duration_seconds:.1f}s")

            if result.errors:
                console.print()
                console.print("[red]Errors:[/red]")
                for error in result.errors[:5]:
                    console.print(f"  • {error}")

        if not result.success:
            sys.exit(1)

    except Exception as e:
        logger.exception("Pipeline execution failed")
        console.print(f"[red]Pipeline error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--type",
    "-t",
    "pipeline_type",
    type=click.Choice(["facct26", "icml26", "venue26"], case_sensitive=False),
    default="facct26",
    help="Pipeline type.",
)
def stages(pipeline_type: str) -> None:
    """
    List available pipeline stages.

    Shows all stages registered for the specified pipeline type,
    including universal and paper-specific stages.

    \b
    Example:
        cei-pipeline stages --type facct26
    """
    # Import stages to trigger registration
    # (stages will be registered when we add them)

    console.print(f"[bold]Stages for {pipeline_type} pipeline:[/bold]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Order", style="dim", width=6)
    table.add_column("Stage Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Registered", style="yellow")

    for i, stage_name in enumerate(StageRegistry.STAGE_ORDER, 1):
        stage_class = StageRegistry.get_stage(stage_name, pipeline_type)  # type: ignore[arg-type]

        # Determine if universal or paper-specific
        if stage_name in StageRegistry._universal_stages:
            stage_type = "universal"
        elif stage_name in StageRegistry._pipeline_stages.get(pipeline_type, {}):
            stage_type = pipeline_type
        else:
            stage_type = "-"

        registered = "✓" if stage_class else "○"

        table.add_row(str(i), stage_name, stage_type, registered)

    console.print(table)

    # Show registered counts
    registered = StageRegistry.list_registered_stages()
    console.print()
    console.print(f"Universal stages: {len(registered['universal'])}")
    console.print(f"FAccT stages:     {len(registered['facct26'])}")
    console.print(f"ICML stages:      {len(registered['icml26'])}")
    console.print(f"Venue stages:     {len(registered.get('venue26', []))}")


@main.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, path_type=Path),
    default=Path("outputs"),
    help="Output directory to check.",
)
@click.option(
    "--type",
    "-t",
    "pipeline_type",
    type=click.Choice(["facct26", "icml26", "venue26"], case_sensitive=False),
    default="facct26",
    help="Pipeline type.",
)
def status(output_dir: Path, pipeline_type: str) -> None:
    """
    Show pipeline execution status.

    Displays checkpoint information and completed stages.

    \b
    Example:
        cei-pipeline status --output-dir outputs --type facct26
    """
    checkpoint_path = output_dir / pipeline_type / "checkpoint.json"

    if not checkpoint_path.exists():
        console.print(f"[yellow]No checkpoint found at:[/yellow] {checkpoint_path}")
        console.print("Pipeline has not been started or checkpoint was cleared.")
        return

    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    console.print(f"[bold]Pipeline Status: {pipeline_type}[/bold]")
    console.print()
    console.print(f"  Checkpoint:    {checkpoint_path}")
    console.print(f"  Timestamp:     {checkpoint.get('timestamp', 'N/A')}")
    console.print(f"  Mode:          {checkpoint.get('execution_mode', 'N/A')}")
    console.print(f"  Current stage: {checkpoint.get('current_stage', 'None')}")
    console.print()

    completed = checkpoint.get("completed_stages", [])
    console.print(f"[bold]Completed stages ({len(completed)}):[/bold]")

    if completed:
        for stage_name in completed:
            result = checkpoint.get("stage_results", {}).get(stage_name, {})
            status_str = result.get("status", "unknown")
            duration = result.get("duration_seconds", 0)
            console.print(f"  ✓ {stage_name} ({status_str}, {duration:.1f}s)")
    else:
        console.print("  No stages completed yet")

    # Show remaining stages
    remaining = [s for s in StageRegistry.STAGE_ORDER if s not in completed]
    if remaining:
        console.print()
        console.print(f"[bold]Remaining stages ({len(remaining)}):[/bold]")
        for stage_name in remaining:
            console.print(f"  ○ {stage_name}")


@main.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, path_type=Path),
    default=Path("outputs"),
    help="Output directory.",
)
@click.option(
    "--type",
    "-t",
    "pipeline_type",
    type=click.Choice(["facct26", "icml26", "venue26"], case_sensitive=False),
    default="facct26",
    help="Pipeline type.",
)
@click.confirmation_option(prompt="Clear checkpoint and allow fresh start?")
def clear(output_dir: Path, pipeline_type: str) -> None:
    """
    Clear pipeline checkpoint.

    Removes the checkpoint file to allow a fresh pipeline run.

    \b
    Example:
        cei-pipeline clear --type facct26
    """
    checkpoint_path = output_dir / pipeline_type / "checkpoint.json"

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        console.print(f"[green]Cleared checkpoint:[/green] {checkpoint_path}")
    else:
        console.print(f"[yellow]No checkpoint found at:[/yellow] {checkpoint_path}")


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to configuration YAML file.",
)
@click.option(
    "--type",
    "-t",
    "pipeline_type",
    type=click.Choice(["facct26", "icml26", "venue26"], case_sensitive=False),
    default="facct26",
    help="Pipeline type for cost estimate.",
)
def estimate(config: Path | None, pipeline_type: str) -> None:
    """
    Estimate pipeline cost.

    Calculates estimated API costs based on configuration.

    \b
    Example:
        cei-pipeline estimate --type facct26
    """
    cfg = load_pipeline_config(config)

    console.print(f"[bold]Cost Estimate: {pipeline_type}[/bold]")
    console.print()

    # Get paper-specific config
    paper_config = cfg.get(pipeline_type, {})

    if pipeline_type == "facct26":
        # FAccT cost estimate
        models = paper_config.get("models", {})
        total_models = (
            len(models.get("us_commercial", []))
            + len(models.get("cn_commercial", []))
            + len(models.get("oss_local", []))
        )
        demographics = len(paper_config.get("demographics", []))

        scenarios = 300  # v2 default
        rq1_calls = scenarios * total_models
        rq2_calls = scenarios * demographics * total_models

        console.print(f"  Models:        {total_models}")
        console.print(f"  Scenarios:     {scenarios}")
        console.print(f"  Demographics:  {demographics}")
        console.print()
        console.print("  [bold]API Calls:[/bold]")
        console.print(f"    RQ1 (baseline):     {rq1_calls:,}")
        console.print(f"    RQ2 (demographic):  {rq2_calls:,}")
        console.print(f"    Total:              {rq1_calls + rq2_calls:,}")

        # Rough cost estimate ($0.0008/call average)
        total_calls = rq1_calls + rq2_calls
        estimated_cost = total_calls * 0.0008
        console.print()
        console.print(f"  [bold]Estimated cost:[/bold] ${estimated_cost:.2f}")

    elif pipeline_type == "icml26":
        # ICML cost estimate
        api_models = paper_config.get("api_models", {})
        total_api_models = len(api_models.get("paid", [])) + len(api_models.get("oss", []))
        interventions = len(paper_config.get("interventions", []))

        scenarios = 300
        baseline_calls = scenarios * total_api_models
        recombined_calls = 600 * total_api_models  # 600 new recombinations
        intervention_calls = scenarios * interventions * total_api_models

        console.print(f"  API Models:      {total_api_models}")
        console.print(f"  Scenarios:       {scenarios}")
        console.print(f"  Interventions:   {interventions}")
        console.print()
        console.print("  [bold]API Calls:[/bold]")
        console.print(f"    Baseline (shared):    {baseline_calls:,}")
        console.print(f"    Recombined (RQ2):     {recombined_calls:,}")
        console.print(f"    Interventions (RQ3):  {intervention_calls:,}")
        total_calls = baseline_calls + recombined_calls + intervention_calls
        console.print(f"    Total:                {total_calls:,}")

        estimated_cost = total_calls * 0.0008
        console.print()
        console.print(f"  [bold]Estimated cost:[/bold] ${estimated_cost:.2f}")

    console.print()
    console.print("[dim]Note: Actual costs depend on model pricing and token usage[/dim]")


if __name__ == "__main__":
    main()
