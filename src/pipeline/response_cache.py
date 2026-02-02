"""
Unified response cache for pipeline inference stages.

Provides:
- InferenceResponse: Unified dataclass for pilot and main inference responses
- ResponseCache: Pipeline-aware wrapper around OutputCache for cross-stage sharing

This enables:
1. Pilot study responses to be reused in main inference (cost savings)
2. Stop/save/resume for long-running inference
3. Automatic cache lookup before any API call
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.output_cache import CacheKey, OutputCache, compute_prompt_hash

logger = logging.getLogger(__name__)

# Default cache location (central, shared across runs)
DEFAULT_CACHE_DIR = Path("data/response_cache")


@dataclass
class InferenceResponse:
    """
    Unified response format for all inference stages (pilot, main, supplementary).

    This ensures data compatibility between pilot and main inference,
    enabling pilot responses to be reused in main inference runs.
    """

    # Required identifiers
    scenario_id: str
    model_id: str

    # Response status
    success: bool
    error: str | None = None

    # Prediction payload (when success=True)
    prediction: dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    latency_ms: float = 0.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    prompt_hash: str | None = None
    source_stage: str | None = None  # "pilot_test", "main_inference", etc.
    intervention: str | None = None

    # Cache metadata (populated when loaded from cache)
    from_cache: bool = False
    cached_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_id": self.scenario_id,
            "model_id": self.model_id,
            "success": self.success,
            "error": self.error,
            "prediction": self.prediction,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "prompt_hash": self.prompt_hash,
            "source_stage": self.source_stage,
            "intervention": self.intervention,
            "from_cache": self.from_cache,
            "cached_at": self.cached_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceResponse:
        """Create from dictionary."""
        return cls(
            scenario_id=str(data.get("scenario_id", "")),
            model_id=str(data.get("model_id", "")),
            success=data.get("success", False),
            error=data.get("error"),
            prediction=data.get("prediction", {}),
            latency_ms=data.get("latency_ms", 0.0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            prompt_hash=data.get("prompt_hash"),
            source_stage=data.get("source_stage"),
            intervention=data.get("intervention"),
            from_cache=data.get("from_cache", False),
            cached_at=data.get("cached_at"),
        )

    def to_cache_key(self) -> CacheKey:
        """Generate cache key for this response."""
        return CacheKey(
            model=self.model_id,
            scenario_id=self.scenario_id,
            intervention=self.intervention,
            prompt_hash=self.prompt_hash,
        )


@dataclass
class CacheStats:
    """Statistics about cache operations during an inference run."""

    cache_hits: int = 0
    cache_misses: int = 0
    api_calls_made: int = 0
    api_calls_saved: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def cost_savings_percent(self) -> float:
        """Estimate cost savings from cache hits."""
        total_potential = self.api_calls_made + self.api_calls_saved
        return (self.api_calls_saved / total_potential * 100) if total_potential > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "api_calls_made": self.api_calls_made,
            "api_calls_saved": self.api_calls_saved,
            "hit_rate": self.hit_rate,
            "cost_savings_percent": self.cost_savings_percent,
        }


class ResponseCache:
    """
    Pipeline-aware response cache for inference stages.

    Wraps OutputCache with pipeline-specific functionality:
    - Automatic prompt hashing for cache key generation
    - Cross-stage response sharing (pilot -> main inference)
    - Statistics tracking for cache performance
    - Unified InferenceResponse format

    Example usage:
        cache = ResponseCache()

        # Check cache before API call
        response = cache.get_response(model_id, scenario_id, prompt)
        if response is None:
            # Make API call
            response = make_api_call(model_id, prompt)
            # Store in cache
            cache.store_response(response)
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        """
        Initialize response cache.

        Args:
            cache_dir: Directory for cache storage (default: data/response_cache)
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.enabled = enabled
        self._stats = CacheStats()

        if enabled:
            self._cache = OutputCache(self.cache_dir)
            logger.info(f"Response cache initialized at {self.cache_dir}")
        else:
            self._cache = None
            logger.info("Response cache disabled")

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = CacheStats()

    def get_response(
        self,
        model_id: str,
        scenario_id: str | int,
        prompt: str,
        intervention: str | None = None,
    ) -> InferenceResponse | None:
        """
        Check cache for existing response.

        Args:
            model_id: Model identifier
            scenario_id: Scenario identifier
            prompt: The prompt that would be sent to the API
            intervention: Optional intervention identifier

        Returns:
            Cached InferenceResponse if found, None otherwise
        """
        if not self.enabled or self._cache is None:
            self._stats.cache_misses += 1
            return None

        # Generate cache key with prompt hash
        prompt_hash = compute_prompt_hash(prompt)
        key = CacheKey(
            model=model_id,
            scenario_id=str(scenario_id),
            intervention=intervention,
            prompt_hash=prompt_hash,
        )

        # Check cache
        cached_data = self._cache.get(key)
        if cached_data is not None:
            self._stats.cache_hits += 1
            self._stats.api_calls_saved += 1

            # Convert to InferenceResponse
            response = InferenceResponse.from_dict(cached_data)
            response.from_cache = True
            response.cached_at = cached_data.get("_cached_at")

            logger.debug(f"Cache hit for {key}")
            return response

        self._stats.cache_misses += 1
        logger.debug(f"Cache miss for {key}")
        return None

    def has_response(
        self,
        model_id: str,
        scenario_id: str | int,
        prompt: str,
        intervention: str | None = None,
    ) -> bool:
        """
        Check if cache contains a response without loading it.

        Args:
            model_id: Model identifier
            scenario_id: Scenario identifier
            prompt: The prompt that would be sent to the API
            intervention: Optional intervention identifier

        Returns:
            True if response is cached, False otherwise
        """
        if not self.enabled or self._cache is None:
            return False

        prompt_hash = compute_prompt_hash(prompt)
        key = CacheKey(
            model=model_id,
            scenario_id=str(scenario_id),
            intervention=intervention,
            prompt_hash=prompt_hash,
        )
        return self._cache.has(key)

    def store_response(
        self,
        response: InferenceResponse,
        prompt: str,
    ) -> None:
        """
        Store response in cache.

        Args:
            response: The inference response to cache
            prompt: The prompt used (for hash generation)
        """
        if not self.enabled or self._cache is None:
            return

        # Generate cache key
        prompt_hash = compute_prompt_hash(prompt)
        response.prompt_hash = prompt_hash

        key = CacheKey(
            model=response.model_id,
            scenario_id=response.scenario_id,
            intervention=response.intervention,
            prompt_hash=prompt_hash,
        )

        # Store with atomic write
        self._cache.set(key, response.to_dict(), atomic=True)
        self._stats.api_calls_made += 1

        logger.debug(f"Stored response for {key}")

    def get_or_call(
        self,
        model_id: str,
        scenario_id: str | int,
        prompt: str,
        api_call_fn: callable,
        intervention: str | None = None,
        source_stage: str | None = None,
    ) -> InferenceResponse:
        """
        Get cached response or make API call if not cached.

        This is the primary interface for inference stages. It handles:
        1. Cache lookup
        2. API call if not cached
        3. Storing new response in cache

        Args:
            model_id: Model identifier
            scenario_id: Scenario identifier
            prompt: The prompt to send
            api_call_fn: Function to call if not cached. Should return InferenceResponse.
            intervention: Optional intervention identifier
            source_stage: Stage name for tracking (e.g., "pilot_test")

        Returns:
            InferenceResponse (from cache or fresh API call)
        """
        # Check cache first
        cached = self.get_response(model_id, scenario_id, prompt, intervention)
        if cached is not None:
            return cached

        # Make API call
        response = api_call_fn()

        # Set metadata
        response.source_stage = source_stage
        response.intervention = intervention

        # Store in cache
        self.store_response(response, prompt)

        return response

    def get_cached_model_scenarios(
        self,
        model_id: str,
        intervention: str | None = None,
    ) -> list[str | int]:
        """
        Get list of scenario IDs that are cached for a model.

        Args:
            model_id: Model identifier
            intervention: Optional intervention filter

        Returns:
            List of cached scenario IDs
        """
        if not self.enabled or self._cache is None:
            return []

        return self._cache.get_cached_scenarios(model_id, intervention)

    def get_all_cached_responses(
        self,
        model_id: str | None = None,
    ) -> list[InferenceResponse]:
        """
        Load all cached responses, optionally filtered by model.

        Args:
            model_id: Optional model filter

        Returns:
            List of cached InferenceResponse objects
        """
        if not self.enabled or self._cache is None:
            return []

        responses = []
        keys = self._cache.get_cached_keys(model_id)

        for key in keys:
            data = self._cache.get(key)
            if data is not None:
                response = InferenceResponse.from_dict(data)
                response.from_cache = True
                response.cached_at = data.get("_cached_at")
                responses.append(response)

        return responses

    def clear_model(self, model_id: str) -> int:
        """
        Clear all cached responses for a model.

        Args:
            model_id: Model identifier

        Returns:
            Number of entries cleared
        """
        if not self.enabled or self._cache is None:
            return 0

        return self._cache.clear_model(model_id)

    def clear_all(self) -> int:
        """
        Clear entire cache.

        Returns:
            Number of entries cleared
        """
        if not self.enabled or self._cache is None:
            return 0

        return self._cache.clear_all()

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache stats
        """
        stats = self._stats.to_dict()

        if self._cache is not None:
            cache_stats = self._cache.get_stats()
            stats["storage"] = cache_stats

        return stats


def create_response_cache_from_config(config: dict[str, Any]) -> ResponseCache:
    """
    Create ResponseCache from pipeline configuration.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Configured ResponseCache instance
    """
    cache_config = config.get("pipeline", {}).get("cache", {})

    cache_dir = cache_config.get("dir", str(DEFAULT_CACHE_DIR))
    enabled = cache_config.get("enabled", True)

    return ResponseCache(cache_dir=cache_dir, enabled=enabled)
