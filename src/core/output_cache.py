"""
Output caching system for CEI benchmark evaluation.

Provides file-based caching of model predictions to avoid redundant API calls.
Supports versioning, config-aware cache keys, and selective invalidation.

Example:
    >>> cache = OutputCache("./data/cached_outputs")
    >>> key = CacheKey(model="gpt-4", scenario_id=1, intervention=None)
    >>> if cache.has(key):
    ...     output = cache.get(key)
    ... else:
    ...     output = model.predict(prompt)
    ...     cache.set(key, output)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheKey:
    """
    Unique key for cached model outputs.

    Attributes:
        model: Model identifier
        scenario_id: Scenario ID within CEI benchmark (string or int)
        intervention: Optional intervention applied (None for base)
        prompt_hash: Optional hash of the prompt (for custom prompts)
    """

    model: str
    scenario_id: str | int
    intervention: str | None = None
    prompt_hash: str | None = None

    def to_path_components(self) -> tuple[str, str]:
        """Return (directory, filename) for this key."""
        # Sanitize scenario_id for filesystem (replace problematic chars)
        safe_scenario_id = str(self.scenario_id).replace("/", "_").replace("\\", "_")
        intervention_suffix = f"_{self.intervention}" if self.intervention else ""
        hash_suffix = f"_{self.prompt_hash[:8]}" if self.prompt_hash else ""
        filename = f"scenario_{safe_scenario_id}{intervention_suffix}{hash_suffix}.json"
        return self.model, filename

    def __str__(self) -> str:
        """String representation for logging."""
        parts = [f"model={self.model}", f"scenario={self.scenario_id}"]
        if self.intervention:
            parts.append(f"intervention={self.intervention}")
        if self.prompt_hash:
            parts.append(f"prompt_hash={self.prompt_hash[:8]}")
        return f"CacheKey({', '.join(parts)})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "scenario_id": self.scenario_id,
            "intervention": self.intervention,
            "prompt_hash": self.prompt_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheKey:
        """Create from dictionary."""
        return cls(
            model=data["model"],
            scenario_id=data["scenario_id"],
            intervention=data.get("intervention"),
            prompt_hash=data.get("prompt_hash"),
        )


@dataclass
class CacheMetadata:
    """Metadata about cached outputs."""

    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    config_hash: str | None = None
    model_config: dict[str, Any] | None = None
    total_entries: int = 0
    models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "config_hash": self.config_hash,
            "model_config": self.model_config,
            "total_entries": self.total_entries,
            "models": self.models,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheMetadata:
        """Create from dictionary."""
        return cls(
            version=data.get("version", "1.0.0"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            config_hash=data.get("config_hash"),
            model_config=data.get("model_config"),
            total_entries=data.get("total_entries", 0),
            models=data.get("models", []),
        )


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__numpy__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return super().default(obj)


def numpy_decoder(dct):
    """Decode numpy arrays from JSON."""
    if "__numpy__" in dct:
        return np.array(dct["data"], dtype=dct["dtype"])
    return dct


class OutputCache:
    """
    File-based cache for model outputs.

    Stores outputs in a hierarchical directory structure:
    cache_dir/
    ├── manifest.json          # Cache metadata
    ├── gpt-4/
    │   ├── scenario_1.json
    │   ├── scenario_2.json
    │   └── scenario_1_INT-1_role_modeling.json
    ├── llama-3-8b/
    │   └── ...
    └── hidden_states/         # Separate storage for large arrays
        └── llama-3-8b/
            └── scenario_1_layer_0.npy
    """

    MANIFEST_FILE = "manifest.json"
    HIDDEN_STATES_DIR = "hidden_states"

    def __init__(
        self,
        cache_dir: str | Path,
        config_hash: str | None = None,
    ) -> None:
        """
        Initialize output cache.

        Args:
            cache_dir: Directory for cache storage
            config_hash: Hash of configuration for version checking
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash

        self._metadata = self._load_or_create_metadata()

    def _load_or_create_metadata(self) -> CacheMetadata:
        """Load existing metadata or create new."""
        manifest_path = self.cache_dir / self.MANIFEST_FILE
        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
                return CacheMetadata.from_dict(data)
        return CacheMetadata(config_hash=self.config_hash)

    def _save_metadata(self) -> None:
        """Save metadata to manifest file."""
        manifest_path = self.cache_dir / self.MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(self._metadata.to_dict(), f, indent=2)

    def _get_path(self, key: CacheKey) -> Path:
        """Get file path for a cache key."""
        model_dir, filename = key.to_path_components()
        path = self.cache_dir / model_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _get_hidden_state_path(self, key: CacheKey, layer: int) -> Path:
        """Get path for hidden state numpy file."""
        model_dir, base_filename = key.to_path_components()
        base_name = base_filename.replace(".json", "")
        path = (
            self.cache_dir
            / self.HIDDEN_STATES_DIR
            / model_dir
            / f"{base_name}_layer_{layer}.npy"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def has(self, key: CacheKey) -> bool:
        """Check if cache contains an entry for the key."""
        return self._get_path(key).exists()

    def get(self, key: CacheKey, include_hidden_states: bool = False) -> dict[str, Any] | None:
        """
        Retrieve cached output for a key.

        Args:
            key: Cache key to look up
            include_hidden_states: Whether to load hidden states from npy files

        Returns:
            Cached output dictionary or None if not found
        """
        path = self._get_path(key)
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f, object_hook=numpy_decoder)

        # Load hidden states if requested and available
        if include_hidden_states and data.get("has_hidden_states"):
            hidden_states = {}
            for layer in data.get("hidden_state_layers", []):
                hs_path = self._get_hidden_state_path(key, layer)
                if hs_path.exists():
                    hidden_states[layer] = np.load(hs_path)
            data["hidden_states"] = hidden_states

        return data

    def set(
        self,
        key: CacheKey,
        output: dict[str, Any],
        hidden_states: dict[int, np.ndarray] | None = None,
        atomic: bool = True,
    ) -> None:
        """
        Store output in cache.

        Args:
            key: Cache key
            output: Output dictionary to cache
            hidden_states: Optional hidden states to store separately
            atomic: Use atomic write (temp file + rename) for crash safety
        """
        path = self._get_path(key)

        # Handle hidden states separately (large numpy arrays)
        output = output.copy()
        if hidden_states:
            output["has_hidden_states"] = True
            output["hidden_state_layers"] = list(hidden_states.keys())
            for layer, states in hidden_states.items():
                hs_path = self._get_hidden_state_path(key, layer)
                np.save(hs_path, states)

        # Store cache key in output for later retrieval
        output["_cache_key"] = key.to_dict()
        output["_cached_at"] = datetime.now().isoformat()

        if atomic:
            atomic_write_json(path, output, NumpyEncoder)
        else:
            with open(path, "w") as f:
                json.dump(output, f, cls=NumpyEncoder, indent=2)

        # Update metadata
        if key.model not in self._metadata.models:
            self._metadata.models.append(key.model)
        self._metadata.total_entries += 1
        self._save_metadata()

        logger.debug(f"Cached output for {key}")

    def delete(self, key: CacheKey) -> bool:
        """
        Delete a cached entry.

        Args:
            key: Cache key to delete

        Returns:
            True if entry was deleted, False if not found
        """
        path = self._get_path(key)
        if not path.exists():
            return False

        # Delete hidden states if present
        data = self.get(key, include_hidden_states=False)
        if data and data.get("has_hidden_states"):
            for layer in data.get("hidden_state_layers", []):
                hs_path = self._get_hidden_state_path(key, layer)
                if hs_path.exists():
                    hs_path.unlink()

        path.unlink()
        self._metadata.total_entries -= 1
        self._save_metadata()
        return True

    def clear_model(self, model: str) -> int:
        """
        Clear all cached outputs for a specific model.

        Args:
            model: Model identifier to clear

        Returns:
            Number of entries deleted
        """
        model_dir = self.cache_dir / model
        if not model_dir.exists():
            return 0

        count = 0
        for path in model_dir.glob("*.json"):
            path.unlink()
            count += 1

        # Clear hidden states
        hs_dir = self.cache_dir / self.HIDDEN_STATES_DIR / model
        if hs_dir.exists():
            for path in hs_dir.glob("*.npy"):
                path.unlink()

        if model in self._metadata.models:
            self._metadata.models.remove(model)
        self._metadata.total_entries -= count
        self._save_metadata()

        logger.info(f"Cleared {count} cached entries for model {model}")
        return count

    def clear_all(self) -> int:
        """
        Clear entire cache.

        Returns:
            Number of entries deleted
        """
        total = 0
        for model in list(self._metadata.models):
            total += self.clear_model(model)
        return total

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "cache_dir": str(self.cache_dir),
            "total_entries": self._metadata.total_entries,
            "models": self._metadata.models,
            "version": self._metadata.version,
            "created_at": self._metadata.created_at,
        }

        # Count entries per model
        entries_by_model = {}
        for model in self._metadata.models:
            model_dir = self.cache_dir / model
            if model_dir.exists():
                entries_by_model[model] = len(list(model_dir.glob("*.json")))

        stats["entries_by_model"] = entries_by_model
        return stats

    def get_cached_scenarios(
        self, model: str, intervention: str | None = None
    ) -> list[str | int]:
        """
        Get list of scenario IDs that are cached for a model.

        Args:
            model: Model identifier
            intervention: Optional intervention filter

        Returns:
            List of cached scenario IDs (may be strings or ints)
        """
        model_dir = self.cache_dir / model
        if not model_dir.exists():
            return []

        scenario_ids: list[str | int] = []
        pattern = "scenario_*.json"
        for path in model_dir.glob(pattern):
            # Parse scenario ID from filename
            name = path.stem
            if intervention and f"_{intervention}" not in name:
                continue
            # Extract scenario ID (after "scenario_" prefix)
            if name.startswith("scenario_"):
                remainder = name[9:]  # Remove "scenario_" prefix
                # Split on underscore to handle intervention/hash suffixes
                parts = remainder.split("_")
                scenario_id_str = parts[0]
                # Try to convert to int if it looks like a number
                if scenario_id_str.isdigit():
                    scenario_ids.append(int(scenario_id_str))
                else:
                    scenario_ids.append(scenario_id_str)

        unique_ids = list(set(scenario_ids))
        # Use numeric sorting if all are ints, otherwise fall back to string sorting
        if all(isinstance(x, int) for x in unique_ids):
            return sorted(unique_ids)
        return sorted(unique_ids, key=lambda x: (isinstance(x, str), str(x)))

    def get_cached_keys(self, model: str | None = None) -> list[CacheKey]:
        """
        Get all cached keys, optionally filtered by model.

        Args:
            model: Optional model filter

        Returns:
            List of CacheKey objects for cached entries
        """
        keys: list[CacheKey] = []
        models_to_check = [model] if model else self._metadata.models

        for m in models_to_check:
            model_dir = self.cache_dir / m
            if not model_dir.exists():
                continue

            for path in model_dir.glob("scenario_*.json"):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    # Extract key info from cached data
                    if "_cache_key" in data:
                        keys.append(CacheKey.from_dict(data["_cache_key"]))
                    else:
                        # Fallback: parse from filename
                        name = path.stem
                        if name.startswith("scenario_"):
                            scenario_id = name[9:].split("_")[0]
                            keys.append(CacheKey(model=m, scenario_id=scenario_id))
                except (json.JSONDecodeError, KeyError):
                    continue

        return keys


def compute_prompt_hash(prompt: str) -> str:
    """Compute hash of a prompt for cache key."""
    return hashlib.sha256(prompt.encode()).hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any], encoder: type = NumpyEncoder) -> None:
    """
    Write JSON file atomically using temp file + rename pattern.

    This ensures that interrupted writes don't corrupt the cache file.

    Args:
        path: Target file path
        data: Data to write
        encoder: JSON encoder class to use
    """
    import os
    import tempfile

    # Write to temp file in same directory (ensures same filesystem for atomic rename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, cls=encoder, indent=2)
        # Atomic rename (on POSIX systems)
        os.replace(temp_path, path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
