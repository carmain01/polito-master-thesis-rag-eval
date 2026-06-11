"""Disk-based LLM response cache — avoid redundant API calls during development."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from rag_eval.core.config import CacheConfig
from rag_eval.utils.provider import LLMResponse


class ResponseCache:
    """JSON file-based cache for LLM responses.

    Each cached response is stored as a JSON file, keyed by a SHA-256 hash
    of the prompt, system message, and model. This avoids redundant (and
    expensive) LLM calls when re-running evaluations during development.

    Directory structure::

        .cache/rag_eval/
        ├── a1b2c3d4...json   # hash of (prompt, system, model)
        ├── e5f6a7b8...json
        └── ...
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        config = config or CacheConfig()
        self._cache_dir = Path(config.directory)
        self._ttl = config.ttl_seconds
        self._hits = 0
        self._misses = 0

        # Create cache directory if it doesn't exist
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, prompt: str, system: str, model: str) -> str:
        """Generate a cache key from the request parameters."""
        content = f"{model}::{system}::{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Return the file path for a cache key."""
        return self._cache_dir / f"{key}.json"

    def get(self, prompt: str, system: str, model: str) -> LLMResponse | None:
        """Look up a cached response.

        Args:
            prompt: The user prompt.
            system: The system prompt.
            model: The model identifier.

        Returns:
            The cached LLMResponse, or None if not found / expired.
        """
        key = self._make_key(prompt, system, model)
        path = self._cache_path(key)

        if not path.exists():
            self._misses += 1
            return None

        try:
            data = json.loads(path.read_text())

            # Check TTL expiration
            if self._ttl is not None:
                cached_at = data.get("_cached_at", 0)
                if time.time() - cached_at > self._ttl:
                    path.unlink(missing_ok=True)
                    self._misses += 1
                    return None

            # Remove our internal metadata before reconstructing
            data.pop("_cached_at", None)
            self._hits += 1
            return LLMResponse(**data)
        except (json.JSONDecodeError, KeyError, ValueError):
            # Corrupted cache entry — remove it
            path.unlink(missing_ok=True)
            self._misses += 1
            return None

    def put(self, prompt: str, system: str, model: str, response: LLMResponse) -> None:
        """Store a response in the cache.

        Args:
            prompt: The user prompt.
            system: The system prompt.
            model: The model identifier.
            response: The LLM response to cache.
        """
        key = self._make_key(prompt, system, model)
        path = self._cache_path(key)

        data = response.model_dump()
        data["_cached_at"] = time.time()
        path.write_text(json.dumps(data, indent=2))

    def clear(self) -> None:
        """Remove all cached responses."""
        for path in self._cache_dir.glob("*.json"):
            path.unlink(missing_ok=True)
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dictionary with hits, misses, hit rate, and number of cached entries.
        """
        total = self._hits + self._misses
        entries = len(list(self._cache_dir.glob("*.json")))
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "entries": entries,
            "directory": str(self._cache_dir),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return f"ResponseCache(entries={s['entries']}, hit_rate={s['hit_rate']:.1%})"
