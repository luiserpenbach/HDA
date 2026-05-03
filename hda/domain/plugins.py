"""v3 plugin protocol.

Cleaned up vs. the legacy ``core.plugins``:

  - ``name`` and ``version`` are class-level attributes, so a registry can
    introspect plugins without instantiating them — fixes the
    "metadata only set in __init__" trap from the audit.
  - Plugins receive an explicit ``AnalysisContext`` (immutable) instead of
    free-floating dicts. The orchestrator builds it; plugins consume it.
  - Plugins return ``Mapping[str, MeasurementWithUncertainty]`` only.
    The orchestrator owns traceability, persistence, derived-measurement
    chaining, and state-machine progression. Plugins do one thing.
  - ``required_channels`` lets the orchestrator validate the steady_df
    has what the plugin needs *before* invoking it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd

from hda.domain.errors import ConfigError
from hda.domain.types import MeasurementWithUncertainty, SteadyWindow, TestMetadata


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    df: pd.DataFrame
    steady_df: pd.DataFrame
    steady_window: SteadyWindow
    metadata: TestMetadata
    sensor_uncertainties: Mapping[str, float]
    geometry: Mapping[str, float]
    timestamp_column: str = "timestamp"


@runtime_checkable
class AnalysisPlugin(Protocol):
    name: str
    version: str

    def required_channels(self) -> Sequence[str]: ...

    def compute(
        self, ctx: AnalysisContext
    ) -> Mapping[str, MeasurementWithUncertainty]: ...


class PluginRegistry:
    """Explicit registry. No filesystem auto-discovery (the legacy app's
    auto-discover hid registration bugs); plugins are registered at app
    startup so wiring failures surface immediately and visibly.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, AnalysisPlugin] = {}

    def register(self, plugin: AnalysisPlugin) -> None:
        if not isinstance(plugin, AnalysisPlugin):
            raise ConfigError(
                f"{type(plugin).__name__} does not satisfy AnalysisPlugin protocol "
                "(missing name/version/required_channels/compute)"
            )
        if not plugin.name:
            raise ConfigError("Plugin.name must be non-empty")
        if not plugin.version:
            raise ConfigError(f"Plugin '{plugin.name}'.version must be non-empty")
        if plugin.name in self._plugins:
            raise ConfigError(f"Plugin '{plugin.name}' already registered")
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> AnalysisPlugin:
        if name not in self._plugins:
            raise ConfigError(f"No plugin registered under name '{name}'")
        return self._plugins[name]

    def names(self) -> Sequence[str]:
        return tuple(self._plugins.keys())
