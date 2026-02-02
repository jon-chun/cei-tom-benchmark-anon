"""
MAIN_INFERENCE stage: Execute primary RQ API calls across all models and scenarios.

This stage:
1. Loads full scenario set
2. Checks ResponseCache before each API call (reuses pilot data)
3. Runs inference for all model-scenario combinations IN PARALLEL
4. Handles rate limiting, retries with exponential backoff + jitter
5. Tracks progress and manages checkpoints
6. Produces raw inference outputs

Parallelization Strategy:
- All models run concurrently (11 models in parallel)
- Within each model, multiple scenarios run concurrently (controlled by semaphore)
- Per-provider rate limits prevent API throttling
- Backoff + jitter on retries to avoid thundering herd

Features:
- Automatic reuse of pilot study responses (saves API costs)
- Stop/save/resume with checkpoint persistence
- Cache-first approach: check cache before any API call
- Atomic writes to prevent data corruption on interruption
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from core.data_loader import load_gold_scenarios_as_dicts
from pipeline.execution_context import ExecutionContext
from pipeline.response_cache import (
    InferenceResponse,
    ResponseCache,
    create_response_cache_from_config,
)
from pipeline.stage_base import PipelineStage, StageResult, StageStatus
from pipeline.stage_registry import register_stage

logger = logging.getLogger(__name__)


# Default rate limits per provider (max concurrent requests)
DEFAULT_PROVIDER_LIMITS = {
    "openai": 50,       # OpenAI has high rate limits
    "anthropic": 40,    # Anthropic fairly generous
    "google": 60,       # Google/Gemini high limits
    "xai": 20,          # xAI/Grok conservative
    "fireworks": 10,    # Fireworks more restrictive
    "together": 10,     # Together AI conservative
    "ollama": 5,        # Local, limited by hardware
    "default": 5,       # Fallback for unknown providers
}


@dataclass
class ModelStats:
    """Per-model statistics for tracking parse success and failures."""
    total_calls: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    consecutive_failures: int = 0
    dropped: bool = False
    drop_reason: str = ""


@dataclass
class InferenceProgress:
    """Track inference progress with thread-safe counters."""

    total_calls: int
    completed_calls: int = 0
    failed_calls: int = 0
    skipped_calls: int = 0
    cache_hits: int = 0
    api_calls_made: int = 0
    successful_parses: int = 0  # Successfully parsed API responses
    failed_parses: int = 0  # Failed to parse API responses
    current_model: str = ""
    current_scenario: str = ""
    start_time: datetime | None = None
    last_checkpoint: datetime | None = None
    last_progress_pct: int = 0  # Last reported progress percentage
    last_milestone_time: datetime | None = None  # Last time a milestone was printed
    model_stats: dict[str, ModelStats] = field(default_factory=dict)
    dropped_models: set[str] = field(default_factory=set)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def increment_completed(self, model_id: str = "", parsed_ok: bool = True) -> None:
        async with self._lock:
            self.completed_calls += 1
            if parsed_ok:
                self.successful_parses += 1
            else:
                self.failed_parses += 1
            # Update per-model stats
            if model_id:
                if model_id not in self.model_stats:
                    self.model_stats[model_id] = ModelStats()
                self.model_stats[model_id].total_calls += 1
                if parsed_ok:
                    self.model_stats[model_id].successful_parses += 1
                    self.model_stats[model_id].consecutive_failures = 0
                else:
                    self.model_stats[model_id].failed_parses += 1
                    self.model_stats[model_id].consecutive_failures += 1

    async def increment_failed(self, model_id: str = "") -> None:
        async with self._lock:
            self.failed_calls += 1
            self.failed_parses += 1
            if model_id:
                if model_id not in self.model_stats:
                    self.model_stats[model_id] = ModelStats()
                self.model_stats[model_id].total_calls += 1
                self.model_stats[model_id].failed_parses += 1
                self.model_stats[model_id].consecutive_failures += 1

    async def increment_cache_hit(self) -> None:
        async with self._lock:
            self.cache_hits += 1

    async def increment_api_call(self) -> None:
        async with self._lock:
            self.api_calls_made += 1

    async def check_and_drop_model(self, model_id: str, threshold: int) -> bool:
        """Check if model should be dropped due to consecutive failures."""
        async with self._lock:
            if model_id in self.dropped_models:
                return True
            stats = self.model_stats.get(model_id)
            if stats and stats.consecutive_failures >= threshold:
                self.dropped_models.add(model_id)
                stats.dropped = True
                stats.drop_reason = f"Exceeded {threshold} consecutive failures"
                return True
            return False

    def is_model_dropped(self, model_id: str) -> bool:
        """Check if model has been dropped (non-async for quick checks)."""
        return model_id in self.dropped_models

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        total_responses = self.successful_parses + self.failed_parses
        parse_success_rate = (
            self.successful_parses / total_responses * 100
            if total_responses > 0 else 0
        )
        return {
            "total_calls": self.total_calls,
            "completed_calls": self.completed_calls,
            "failed_calls": self.failed_calls,
            "skipped_calls": self.skipped_calls,
            "cache_hits": self.cache_hits,
            "api_calls_made": self.api_calls_made,
            "successful_parses": self.successful_parses,
            "failed_parses": self.failed_parses,
            "parse_success_rate_pct": round(parse_success_rate, 2),
            "dropped_models": list(self.dropped_models),
            "current_model": self.current_model,
            "current_scenario": self.current_scenario,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_checkpoint": (
                self.last_checkpoint.isoformat() if self.last_checkpoint else None
            ),
            "completion_percent": (
                (self.completed_calls + self.failed_calls + self.skipped_calls)
                / self.total_calls
                * 100
                if self.total_calls > 0
                else 0
            ),
            "model_stats": {
                model_id: {
                    "total_calls": stats.total_calls,
                    "successful_parses": stats.successful_parses,
                    "failed_parses": stats.failed_parses,
                    "parse_success_rate_pct": round(
                        stats.successful_parses / stats.total_calls * 100
                        if stats.total_calls > 0 else 0, 2
                    ),
                    "dropped": stats.dropped,
                    "drop_reason": stats.drop_reason,
                }
                for model_id, stats in self.model_stats.items()
            },
        }

    def get_progress_stats(self) -> dict[str, Any]:
        """Get real-time progress statistics for display."""
        processed = self.completed_calls + self.failed_calls + self.skipped_calls
        remaining = self.total_calls - processed
        percent = (processed / self.total_calls * 100) if self.total_calls > 0 else 0

        # Calculate ETA
        eta_seconds = None
        calls_per_sec = 0.0
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed > 0 and processed > 0:
                calls_per_sec = processed / elapsed
                if calls_per_sec > 0:
                    eta_seconds = remaining / calls_per_sec

        # Calculate parse success rate
        total_responses = self.successful_parses + self.failed_parses
        parse_success_rate = (
            self.successful_parses / total_responses * 100
            if total_responses > 0 else 100.0
        )

        return {
            "processed": processed,
            "remaining": remaining,
            "percent": percent,
            "completed": self.completed_calls,
            "failed": self.failed_calls,
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls_made,
            "calls_per_sec": calls_per_sec,
            "eta_seconds": eta_seconds,
            "parse_success_rate": parse_success_rate,
            "dropped_models": len(self.dropped_models),
            "active_models": len(self.model_stats) - len(self.dropped_models),
        }


@dataclass
class RetryConfig:
    """Configuration for retry with exponential backoff + jitter."""

    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    jitter_factor: float = 0.25

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff + jitter."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = delay * self.jitter_factor * random.uniform(-1, 1)
        return max(0.1, delay + jitter)


@register_stage("main_inference")
class MainInferenceStage(PipelineStage):
    """
    Main inference stage with full parallelization.

    Executes primary RQ API calls across all models and scenarios.
    - All models run in parallel
    - Within each model, scenarios run concurrently (rate-limited)
    """

    name = "main_inference"
    description = "Execute primary RQ API calls (parallelized)"
    required_inputs = ["pilot_test.decision"]
    produces_outputs = [
        "main_inference.responses",
        "main_inference.manifest",
        "main_inference.progress",
    ]

    def run(self, context: ExecutionContext) -> StageResult:
        """Execute main inference stage using asyncio for parallelization."""
        # Run the async main function
        return asyncio.run(self._run_async(context))

    async def _run_async(self, context: ExecutionContext) -> StageResult:
        """Async implementation of main inference."""
        start_time = datetime.now()
        errors: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}

        # Get stage config
        stage_config = context.get_stage_config("main_inference")
        api_limits = context.config.get("pipeline", {}).get("api_limits", {})
        retry_config = RetryConfig(
            max_retries=stage_config.get("retry_count", 3),
            base_delay=api_limits.get("backoff_base_seconds", 2.0),
            max_delay=api_limits.get("backoff_max_seconds", 60.0),
            jitter_factor=api_limits.get("jitter_factor", 0.25),
        )
        checkpoint_interval = stage_config.get("checkpoint_interval", 100)
        resume_from_checkpoint = stage_config.get("resume_from_checkpoint", True)
        consecutive_fail_threshold = api_limits.get("consecutive_fail_threshold", 100)
        progress_report_interval_pct = api_limits.get("progress_report_interval_pct", 5)
        progress_report_interval_seconds = api_limits.get("progress_report_interval_seconds", 300)

        # Get provider rate limits from config or use defaults
        provider_limits = stage_config.get("provider_limits", DEFAULT_PROVIDER_LIMITS)

        # Initialize response cache for cross-stage data sharing
        cache_enabled = stage_config.get("use_cache", True)
        self._response_cache: ResponseCache | None = (
            create_response_cache_from_config(context.config) if cache_enabled else None
        )
        if self._response_cache:
            self._response_cache.reset_stats()
            logger.info("Response cache enabled - will reuse pilot study data if available")

        # Check pilot decision
        pilot_decision = context.get_output("pilot_test.decision", {})
        if pilot_decision and not pilot_decision.get("proceed", True):
            warnings.append("Pilot test did not proceed, running anyway in dry run mode")
            if not context.dry_run:
                return StageResult(
                    status=StageStatus.SKIPPED,
                    stage_name=self.name,
                    start_time=start_time,
                    end_time=datetime.now(),
                    skip_reason="Pilot test failed",
                )

        # Get models and scenarios
        models = self._get_models(context)
        scenarios = self._get_scenarios(context)

        if not models:
            models = [{"id": "mock", "provider": "mock"}]
            warnings.append("No models found, using mock")

        if not scenarios:
            scenarios = self._create_synthetic_scenarios(100)
            warnings.append("No scenarios found, using synthetic scenarios")

        total_calls = len(models) * len(scenarios)
        logger.info(
            f"Starting PARALLEL inference: {len(models)} models × {len(scenarios)} scenarios = {total_calls} calls"
        )
        logger.info(f"Models will run in parallel, with per-provider rate limiting")

        # Initialize progress tracking
        progress = InferenceProgress(
            total_calls=total_calls,
            start_time=datetime.now(),
        )

        # Check for existing checkpoint
        output_dir = context.get_stage_output_dir(self.name)
        completed_calls: set[tuple[str, str]] = set()
        if resume_from_checkpoint:
            completed_calls = self._load_checkpoint(output_dir)
            if completed_calls:
                progress.skipped_calls = len(completed_calls)
                logger.info(f"Resuming from checkpoint, skipping {len(completed_calls)} completed calls")

        # Create semaphores for each provider to enforce rate limits
        provider_semaphores: dict[str, asyncio.Semaphore] = {}
        for model in models:
            provider = model.get("provider", "default") if isinstance(model, dict) else "default"
            if provider not in provider_semaphores:
                limit = provider_limits.get(provider, provider_limits.get("default", 5))
                provider_semaphores[provider] = asyncio.Semaphore(limit)
                logger.info(f"Provider '{provider}' rate limit: {limit} concurrent requests")

        # Thread-safe collections for results
        manifest: list[dict[str, Any]] = []
        responses: list[dict[str, Any]] = []
        manifest_lock = asyncio.Lock()
        checkpoint_counter = {"count": 0}
        checkpoint_lock = asyncio.Lock()

        # Create tasks for ALL models to run in parallel
        model_tasks = []
        for model in models:
            model_id = model.get("id", model) if isinstance(model, dict) else model
            provider = model.get("provider", "default") if isinstance(model, dict) else "default"
            semaphore = provider_semaphores.get(provider, provider_semaphores.get("default", asyncio.Semaphore(5)))

            task = self._process_model_scenarios(
                model_id=model_id,
                provider=provider,
                scenarios=scenarios,
                context=context,
                completed_calls=completed_calls,
                progress=progress,
                retry_config=retry_config,
                semaphore=semaphore,
                output_dir=output_dir,
                manifest=manifest,
                responses=responses,
                manifest_lock=manifest_lock,
                checkpoint_counter=checkpoint_counter,
                checkpoint_lock=checkpoint_lock,
                checkpoint_interval=checkpoint_interval,
                consecutive_fail_threshold=consecutive_fail_threshold,
                errors=errors,
            )
            model_tasks.append(task)

        # Start progress reporter task
        progress_reporter_task = asyncio.create_task(
            self._progress_reporter(
                progress,
                update_interval=5.0,
                report_interval_pct=progress_report_interval_pct,
                report_interval_seconds=progress_report_interval_seconds,
            )
        )

        # Run all models in parallel
        logger.info(f"Launching {len(model_tasks)} model tasks in parallel...")
        try:
            await asyncio.gather(*model_tasks, return_exceptions=True)
        finally:
            # Stop progress reporter
            progress_reporter_task.cancel()
            try:
                await progress_reporter_task
            except asyncio.CancelledError:
                pass
            # Print final progress
            self._print_progress(progress, final=True)

        # Save final outputs
        self._save_manifest(output_dir, manifest)
        self._save_progress(output_dir, progress)

        # Calculate metrics
        metrics["total_calls"] = total_calls
        metrics["completed_calls"] = progress.completed_calls
        metrics["failed_calls"] = progress.failed_calls
        metrics["skipped_calls"] = progress.skipped_calls
        metrics["success_rate"] = (
            progress.completed_calls / (total_calls - progress.skipped_calls)
            if (total_calls - progress.skipped_calls) > 0
            else 0
        )
        metrics["cache_hits"] = progress.cache_hits
        metrics["api_calls_made"] = progress.api_calls_made
        metrics["api_calls_saved"] = progress.cache_hits
        metrics["parallelization"] = {
            "models_parallel": len(models),
            "provider_limits": {k: v for k, v in provider_limits.items() if k in provider_semaphores},
        }

        if self._response_cache:
            metrics["cache_stats"] = self._response_cache.get_cache_stats()

        # Log cache savings
        if progress.cache_hits > 0:
            total_requests = progress.cache_hits + progress.api_calls_made
            logger.info(
                f"Cache saved {progress.cache_hits} API calls "
                f"({progress.cache_hits / total_requests * 100:.1f}% hit rate)"
            )

        # Log parallelization summary
        elapsed = (datetime.now() - start_time).total_seconds()
        calls_per_second = (progress.completed_calls + progress.failed_calls) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Completed {progress.completed_calls} calls in {elapsed:.1f}s "
            f"({calls_per_second:.1f} calls/sec)"
        )

        # Set context outputs
        context.set_output("main_inference.responses", responses)
        context.set_output("main_inference.manifest", str(output_dir / "batch_manifest.json"))
        context.set_output("main_inference.progress", progress.to_dict())

        # Determine status
        status = (
            StageStatus.COMPLETED
            if progress.failed_calls == 0
            else StageStatus.FAILED
            if progress.completed_calls == 0
            else StageStatus.COMPLETED  # Partial success
        )

        if progress.failed_calls > 0 and status == StageStatus.COMPLETED:
            warnings.append(f"{progress.failed_calls} calls failed but stage completed")

        return StageResult(
            status=status,
            stage_name=self.name,
            start_time=start_time,
            end_time=datetime.now(),
            outputs={
                "batch_manifest": str(output_dir / "batch_manifest.json"),
                "progress": str(output_dir / "progress.json"),
            },
            metrics=metrics,
            errors=errors[:10],  # Limit error list
            warnings=warnings,
        )

    async def _process_model_scenarios(
        self,
        model_id: str,
        provider: str,
        scenarios: list[dict[str, Any]],
        context: ExecutionContext,
        completed_calls: set[tuple[str, str]],
        progress: InferenceProgress,
        retry_config: RetryConfig,
        semaphore: asyncio.Semaphore,
        output_dir: Path,
        manifest: list[dict[str, Any]],
        responses: list[dict[str, Any]],
        manifest_lock: asyncio.Lock,
        checkpoint_counter: dict[str, int],
        checkpoint_lock: asyncio.Lock,
        checkpoint_interval: int,
        consecutive_fail_threshold: int,
        errors: list[str],
    ) -> None:
        """Process all scenarios for a single model SERIALLY to avoid rate limits.

        Parallelization happens across models (11 models run in parallel),
        but within each model, scenarios are processed one at a time.
        """
        model_dir = output_dir / model_id.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Count scenarios to process (excluding already completed)
        scenarios_to_process = [
            s for s in scenarios
            if (model_id, s.get("id", s.get("scenario_id", "unknown"))) not in completed_calls
        ]
        logger.info(f"[{model_id}] Starting {len(scenarios_to_process)} scenarios (serial)...")

        # Process scenarios SERIALLY to avoid rate limiting
        for scenario in scenarios_to_process:
            # Check if model was dropped
            if progress.is_model_dropped(model_id):
                skipped = len(scenarios_to_process) - scenarios_to_process.index(scenario)
                logger.info(f"[{model_id}] Skipping remaining {skipped} scenarios (model dropped)")
                break

            await self._process_single_scenario(
                model_id=model_id,
                provider=provider,
                scenario=scenario,
                context=context,
                progress=progress,
                retry_config=retry_config,
                semaphore=semaphore,
                model_dir=model_dir,
                manifest=manifest,
                responses=responses,
                manifest_lock=manifest_lock,
                checkpoint_counter=checkpoint_counter,
                checkpoint_lock=checkpoint_lock,
                checkpoint_interval=checkpoint_interval,
                output_dir=output_dir,
                consecutive_fail_threshold=consecutive_fail_threshold,
                errors=errors,
            )

        logger.info(f"[{model_id}] Completed all scenarios")

    async def _process_single_scenario(
        self,
        model_id: str,
        provider: str,
        scenario: dict[str, Any],
        context: ExecutionContext,
        progress: InferenceProgress,
        retry_config: RetryConfig,
        semaphore: asyncio.Semaphore,
        model_dir: Path,
        manifest: list[dict[str, Any]],
        responses: list[dict[str, Any]],
        manifest_lock: asyncio.Lock,
        checkpoint_counter: dict[str, int],
        checkpoint_lock: asyncio.Lock,
        checkpoint_interval: int,
        output_dir: Path,
        consecutive_fail_threshold: int,
        errors: list[str],
    ) -> None:
        """Process a single scenario with rate limiting and retries."""
        scenario_id = scenario.get("id", scenario.get("scenario_id", "unknown"))

        # Check if model has been dropped due to consecutive failures
        if progress.is_model_dropped(model_id):
            progress.skipped_calls += 1
            return

        prompt = self._generate_prompt(scenario)

        # Check cache first (before acquiring semaphore)
        cached_response = None
        if self._response_cache and not context.dry_run:
            cached_response = self._response_cache.get_response(
                model_id=model_id,
                scenario_id=scenario_id,
                prompt=prompt,
            )

        if cached_response is not None:
            # Use cached response (may be from pilot study)
            response = cached_response.to_dict()
            response["from_cache"] = True
            response["source_stage"] = cached_response.source_stage
            await progress.increment_cache_hit()
            logger.debug(f"Cache hit for {model_id}/{scenario_id}")
        elif context.dry_run:
            # Simulate successful prediction
            response = self._simulate_prediction(model_id, scenario_id, scenario)
        else:
            # Acquire semaphore to respect rate limits
            async with semaphore:
                response = await self._run_prediction_with_retry(
                    model_id=model_id,
                    provider=provider,
                    scenario=scenario,
                    prompt=prompt,
                    retry_config=retry_config,
                )
                await progress.increment_api_call()

            # Store in cache for future use
            if self._response_cache and response.get("success"):
                inference_response = InferenceResponse(
                    scenario_id=str(scenario_id),
                    model_id=model_id,
                    success=response.get("success", False),
                    error=response.get("error"),
                    prediction=response.get("prediction", {}),
                    latency_ms=response.get("latency_ms", 0.0),
                    source_stage="main_inference",
                )
                self._response_cache.store_response(inference_response, prompt)

        # Check if response was successfully parsed
        parsed_ok = response.get("success", False) and response.get("prediction") is not None

        # Update progress and store results
        if response.get("success"):
            await progress.increment_completed(model_id=model_id, parsed_ok=parsed_ok)

            # Save response (atomic write for crash safety)
            response_file = model_dir / f"{scenario_id}.json"
            self._atomic_write_json(response_file, response)

            async with manifest_lock:
                responses.append(response)
        else:
            await progress.increment_failed(model_id=model_id)
            if response.get("error"):
                async with manifest_lock:
                    errors.append(f"{model_id}/{scenario_id}: {response['error']}")

        # Check if model should be dropped due to consecutive failures
        if await progress.check_and_drop_model(model_id, consecutive_fail_threshold):
            logger.warning(
                f"Model {model_id} dropped after {consecutive_fail_threshold} consecutive failures"
            )

        # Add to manifest
        async with manifest_lock:
            manifest.append({
                "model_id": model_id,
                "scenario_id": scenario_id,
                "success": response.get("success", False),
                "from_cache": response.get("from_cache", False),
                "timestamp": datetime.now().isoformat(),
            })

        # Checkpoint periodically
        async with checkpoint_lock:
            checkpoint_counter["count"] += 1
            if checkpoint_counter["count"] % checkpoint_interval == 0:
                self._save_checkpoint(output_dir, manifest)
                progress.last_checkpoint = datetime.now()
                logger.info(
                    f"Checkpoint saved: {progress.completed_calls}/{progress.total_calls} completed "
                    f"(cache hits: {progress.cache_hits}, API calls: {progress.api_calls_made})"
                )

    async def _run_prediction_with_retry(
        self,
        model_id: str,
        provider: str,
        scenario: dict[str, Any],
        prompt: str,
        retry_config: RetryConfig,
    ) -> dict[str, Any]:
        """Run prediction with exponential backoff + jitter retry strategy."""
        scenario_id = scenario.get("id", scenario.get("scenario_id", "unknown"))

        for attempt in range(retry_config.max_retries):
            try:
                # Run the blocking API call in a thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,  # Use default executor
                    self._run_prediction_sync,
                    model_id,
                    provider,
                    scenario,
                    prompt,
                )

                if response.get("success"):
                    return response

                # If failed but not due to rate limit, don't retry
                error = response.get("error", "")
                if "rate" not in error.lower() and "limit" not in error.lower():
                    return response

            except Exception as e:
                error_str = str(e)
                logger.warning(f"[{model_id}/{scenario_id}] Attempt {attempt + 1} failed: {error_str}")

                if attempt == retry_config.max_retries - 1:
                    return {
                        "scenario_id": scenario_id,
                        "model_id": model_id,
                        "success": False,
                        "error": f"All {retry_config.max_retries} retries failed: {error_str}",
                        "timestamp": datetime.now().isoformat(),
                    }

            # Wait with exponential backoff + jitter before retry
            delay = retry_config.get_delay(attempt)
            logger.debug(f"[{model_id}/{scenario_id}] Retrying in {delay:.2f}s...")
            await asyncio.sleep(delay)

        return {
            "scenario_id": scenario_id,
            "model_id": model_id,
            "success": False,
            "error": "Exhausted retries",
            "timestamp": datetime.now().isoformat(),
        }

    def _run_prediction_sync(
        self,
        model_id: str,
        provider: str,
        scenario: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Synchronous prediction wrapper for thread pool execution."""
        scenario_id = scenario.get("id", scenario.get("scenario_id", "unknown"))

        try:
            from core.model_interface import get_model

            model = get_model(model_id)

            start = time.time()
            output = model.predict(prompt, return_confidence=True)
            latency_ms = (time.time() - start) * 1000

            return {
                "scenario_id": scenario_id,
                "model_id": model_id,
                "success": True,
                "prediction": {
                    "subtype": output.subtype_prediction,
                    "emotion": output.emotion_prediction,
                    "valence": output.vad_prediction[0],
                    "arousal": output.vad_prediction[1],
                    "dominance": output.vad_prediction[2],
                    "confidence": output.confidence,
                },
                "latency_ms": latency_ms,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "scenario_id": scenario_id,
                "model_id": model_id,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _get_models(self, context: ExecutionContext) -> list[dict[str, Any]]:
        """Get list of models with provider info from config."""
        config = context.config
        execution_mode = config.get("pipeline", {}).get("execution_mode", "test")

        llm_config = config.get("llm_inference", {})
        models_config = llm_config.get("models", {})

        # Get models based on execution mode
        models = models_config.get(execution_mode, models_config.get("test", []))

        if not models:
            # Fallback to paper config
            paper_config = context.get_paper_config()
            if context.pipeline_type == "facct26":
                model_groups = paper_config.get("models", {})
                if isinstance(model_groups, dict):
                    for group_models in model_groups.values():
                        if isinstance(group_models, list):
                            models.extend([{"id": m, "provider": "unknown"} for m in group_models])
                elif isinstance(model_groups, list):
                    models = [{"id": m, "provider": "unknown"} for m in model_groups]
            elif context.pipeline_type == "icml26":
                models = [{"id": m, "provider": "unknown"} for m in paper_config.get("api_models", [])]
            else:
                models = [{"id": m, "provider": "unknown"} for m in paper_config.get("models", [])]

        logger.info(f"Loaded {len(models)} models for execution mode '{execution_mode}'")
        return models

    def _get_scenarios(self, context: ExecutionContext) -> list[dict[str, Any]]:
        """Load scenarios for inference from gold data."""
        gold_data_path = Path(
            context.config.get("paths", {}).get("gold_data", "data/human-gold-aggregate")
        )

        scenarios = load_gold_scenarios_as_dicts(gold_data_path)

        if scenarios:
            logger.info(f"Loaded {len(scenarios)} scenarios from {gold_data_path}")
        else:
            logger.warning(f"No scenarios found in {gold_data_path}")

        return scenarios

    def _create_synthetic_scenarios(self, count: int) -> list[dict[str, Any]]:
        """Create synthetic scenarios for testing."""
        scenarios = []
        subtypes = ["sarcasm_irony", "mixed_signals", "strategic_politeness"]

        for i in range(count):
            scenarios.append({
                "id": f"scenario_{i:03d}",
                "situation": f"Test situation {i}",
                "utterance": f"Test utterance {i}",
                "speaker_role": "speaker",
                "listener_role": "listener",
                "ground_truth": {"subtype": subtypes[i % len(subtypes)]},
            })

        return scenarios

    def _generate_prompt(self, scenario: dict[str, Any]) -> str:
        """Generate prompt for a scenario (used for cache key hashing)."""
        try:
            from core.prompt_templates import format_cei_prompt

            return format_cei_prompt(
                situation=scenario.get("situation", ""),
                utterance=scenario.get("utterance", ""),
                speaker_role=scenario.get("speaker_role", "speaker"),
                listener_role=scenario.get("listener_role", "listener"),
            )
        except ImportError:
            return (
                f"Situation: {scenario.get('situation', '')}\n"
                f"Utterance: {scenario.get('utterance', '')}"
            )

    def _atomic_write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON file atomically using temp file + rename pattern."""
        try:
            from core.output_cache import atomic_write_json
            atomic_write_json(path, data)
        except ImportError:
            # Fallback to simple write
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def _simulate_prediction(
        self,
        model_id: str,
        scenario_id: str,
        _scenario: dict[str, Any],
    ) -> dict[str, Any]:
        """Simulate a prediction for dry run."""
        import random

        return {
            "scenario_id": scenario_id,
            "model_id": model_id,
            "success": True,
            "prediction": {
                "subtype": random.choice(
                    ["sarcasm_irony", "mixed_signals", "strategic_politeness"]
                ),
                "emotion": random.choice(["fear", "trust", "joy", "sadness"]),
                "valence": random.uniform(-1, 1),
                "arousal": random.uniform(-1, 1),
                "dominance": random.uniform(-1, 1),
                "confidence": random.uniform(0.5, 1.0),
            },
            "latency_ms": random.uniform(100, 500),
            "timestamp": datetime.now().isoformat(),
        }

    def _load_checkpoint(self, output_dir: Path) -> set[tuple[str, str]]:
        """Load completed calls from checkpoint."""
        completed: set[tuple[str, str]] = set()

        manifest_file = output_dir / "batch_manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file) as f:
                    manifest = json.load(f)
                for entry in manifest.get("calls", []):
                    if entry.get("success"):
                        completed.add((entry["model_id"], entry["scenario_id"]))
            except (json.JSONDecodeError, KeyError):
                pass

        return completed

    def _save_checkpoint(self, output_dir: Path, manifest: list[dict[str, Any]]) -> None:
        """Save checkpoint."""
        checkpoint_file = output_dir / "batch_manifest.json"
        with open(checkpoint_file, "w") as f:
            json.dump({"calls": list(manifest)}, f, indent=2)

    def _save_manifest(self, output_dir: Path, manifest: list[dict[str, Any]]) -> None:
        """Save final manifest."""
        manifest_file = output_dir / "batch_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_calls": len(manifest),
                "successful": sum(1 for m in manifest if m.get("success")),
                "failed": sum(1 for m in manifest if not m.get("success")),
                "calls": list(manifest),
            }, f, indent=2)

    def _save_progress(self, output_dir: Path, progress: InferenceProgress) -> None:
        """Save progress."""
        progress_file = output_dir / "progress.json"
        with open(progress_file, "w") as f:
            json.dump(progress.to_dict(), f, indent=2)

    async def _progress_reporter(
        self,
        progress: InferenceProgress,
        update_interval: float = 5.0,
        report_interval_pct: int = 5,
        report_interval_seconds: int = 300,  # 5 minutes default
    ) -> None:
        """Background task that periodically reports progress to terminal.

        Prints a milestone update when EITHER:
        - Progress reaches next percentage threshold (e.g., every 5%)
        - Time since last milestone exceeds threshold (e.g., every 5 minutes)
        Whichever comes first.
        """
        import sys

        # Initialize milestone time tracking
        if progress.last_milestone_time is None:
            progress.last_milestone_time = datetime.now()

        while True:
            await asyncio.sleep(update_interval)
            stats = progress.get_progress_stats()
            current_pct = int(stats["percent"])
            now = datetime.now()

            # Check if we should print a milestone
            pct_milestone = current_pct >= progress.last_progress_pct + report_interval_pct
            time_since_last = (now - progress.last_milestone_time).total_seconds()
            time_milestone = time_since_last >= report_interval_seconds

            if pct_milestone or time_milestone:
                # Update tracking
                progress.last_progress_pct = (current_pct // report_interval_pct) * report_interval_pct
                progress.last_milestone_time = now
                self._print_progress(progress, final=False, milestone=True)
            else:
                self._print_progress(progress, final=False, milestone=False)

    def _print_progress(
        self, progress: InferenceProgress, final: bool = False, milestone: bool = False
    ) -> None:
        """Print real-time progress update to terminal."""
        import sys

        stats = progress.get_progress_stats()

        # Format ETA
        if stats["eta_seconds"] is not None:
            eta_mins = int(stats["eta_seconds"] // 60)
            eta_secs = int(stats["eta_seconds"] % 60)
            if eta_mins > 60:
                eta_hrs = eta_mins // 60
                eta_mins = eta_mins % 60
                eta_str = f"{eta_hrs}h {eta_mins}m"
            elif eta_mins > 0:
                eta_str = f"{eta_mins}m {eta_secs}s"
            else:
                eta_str = f"{eta_secs}s"
        else:
            eta_str = "calculating..."

        # Build progress bar
        bar_width = 30
        filled = int(bar_width * stats["percent"] / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Format status line
        if final:
            # Final summary with parse success rates
            print()  # New line before final
            status = (
                f"✓ Complete: {stats['completed']}/{progress.total_calls} "
                f"({stats['percent']:.1f}%) | "
                f"Parse OK: {stats['parse_success_rate']:.1f}% | "
                f"Failed: {stats['failed']} | "
                f"Dropped: {stats['dropped_models']} models | "
                f"Rate: {stats['calls_per_sec']:.1f}/s"
            )
            print(status)
            # Print per-model stats
            if progress.model_stats:
                print("\nPer-Model Parse Success Rates:")
                for model_id, model_stats in sorted(progress.model_stats.items()):
                    rate = (
                        model_stats.successful_parses / model_stats.total_calls * 100
                        if model_stats.total_calls > 0 else 0
                    )
                    dropped_str = " [DROPPED]" if model_stats.dropped else ""
                    print(f"  {model_id}: {rate:.1f}% ({model_stats.successful_parses}/{model_stats.total_calls}){dropped_str}")
            sys.stdout.flush()
        elif milestone:
            # Milestone report (every N%)
            print()  # New line for milestone
            status = (
                f"[{bar}] {stats['percent']:.0f}% | "
                f"{stats['processed']}/{progress.total_calls} | "
                f"ETA: {eta_str} | "
                f"Parse: {stats['parse_success_rate']:.1f}% | "
                f"Models: {stats['active_models']} active | "
                f"{stats['calls_per_sec']:.1f}/s"
            )
            print(status)
            sys.stdout.flush()
        else:
            # Regular update (same line)
            status = (
                f"\r[{bar}] {stats['percent']:.1f}% | "
                f"{stats['processed']}/{progress.total_calls} | "
                f"ETA: {eta_str} | "
                f"Parse: {stats['parse_success_rate']:.1f}% | "
                f"{stats['calls_per_sec']:.1f}/s"
            )
            print(status, end="", flush=True)
