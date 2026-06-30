"""Metric Registry — auto-discovery and instantiation of metrics."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Any

import rag_eval.metrics
from rag_eval.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


class MetricRegistry:
    """Registry for evaluating metrics."""

    _registry: dict[str, type[BaseMetric]] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, metric_class: type[BaseMetric]) -> None:
        """Register a metric class."""
        # We need an instance to get the name property, or we can look for a class attribute.
        # Since BaseMetric defines 'name' as a property on instances, we can instantiate it
        # temporarily if it takes no required args, but that's unsafe.
        # Alternatively, we just use the class name formatted, but ideally they'd have a class-level name.
        # However, to be safe, we'll map the snake_case of the class name or inspect it.
        # Let's try to instantiate it with empty kwargs, but some require LLMClients.
        # Since name is an @property, we can't easily get it without instantiation.
        # Workaround: we index by class name (lowercased) and also try to extract __name__.

        name = metric_class.__name__.lower()
        cls._registry[name] = metric_class
        logger.debug(f"Registered metric class: {metric_class.__name__} as '{name}'")

    @classmethod
    def discover_metrics(cls) -> None:
        """Scan rag_eval.metrics module to discover and register all BaseMetric subclasses."""
        if cls._discovered:
            return

        package = rag_eval.metrics
        prefix = package.__name__ + "."

        for _, modname, _ in pkgutil.iter_modules(package.__path__, prefix):
            try:
                module = importlib.import_module(modname)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseMetric) and obj is not BaseMetric:
                        cls.register(obj)
            except Exception as e:
                logger.warning(f"Failed to load module {modname} during metric discovery: {e}")

        cls._discovered = True

    @classmethod
    def create_metric(cls, name: str, **kwargs: Any) -> BaseMetric:
        """Create a metric instance by name."""
        if not cls._discovered:
            cls.discover_metrics()

        # Try finding exactly or case-insensitive matching
        target = name.lower().replace("-", "").replace("_", "")

        for key, metric_cls in cls._registry.items():
            norm_key = key.lower().replace("-", "").replace("_", "")
            if norm_key == target:
                return metric_cls(**kwargs)

        # If not found, perhaps they passed the exact class name
        if name.lower() in cls._registry:
            return cls._registry[name.lower()](**kwargs)

        raise ValueError(
            f"Metric '{name}' not found in registry. Available: {list(cls._registry.keys())}"
        )
