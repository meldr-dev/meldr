"""Entry-point plugin discovery.

Groups: meldr.connectors, meldr.sinks, meldr.ops, meldr.providers,
meldr.parsers, meldr.matchers. Plugins come from installed packages only. Sink entry-point
names can start with `table.` or `vector.`. A name without a prefix keeps the table default.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, cast

from meldr.observability import safe_log

GROUPS = (
    "meldr.connectors",
    "meldr.sinks",
    "meldr.ops",
    "meldr.providers",
    "meldr.parsers",
    "meldr.matchers",
)

_loaded: dict[str, dict[str, Any]] = {}


def discover(group: str) -> dict[str, Any]:
    """Load every entry point in `group` (cached). Log and skip broken plugins."""
    if group in _loaded:
        return _loaded[group]
    found: dict[str, Any] = {}
    for ep in entry_points(group=group):
        try:
            found[ep.name] = ep.load()
        except Exception as e:
            safe_log("plugin.load_failed", plugin_name=ep.name, kind=group, reason=type(e).__name__)
    _loaded[group] = found
    return found


def load_all(ops_registry: Any = None) -> dict[str, dict[str, Any]]:
    """Register discovered connectors, parsers, providers, sinks, and ops, and return all by group.

    This function returns matchers but does not register them. `Meldr` gives them to its
    mapping engine.
    """
    from meldr.connectors.base import register_connector
    from meldr.extraction.parsers import register_parser
    from meldr.indexing.sinks import SinkKind, register_sink
    from meldr.llm.providers import register_provider

    def register_sink_plugin(name: str, factory: Any) -> None:
        kind = str(getattr(factory, "sink_kind", "table"))
        for prefix, prefix_kind in (("table.", "table"), ("table:", "table")):
            if name.startswith(prefix):
                name, kind = name.removeprefix(prefix), prefix_kind
                break
        for prefix, prefix_kind in (("vector.", "vector"), ("vector:", "vector")):
            if name.startswith(prefix):
                name, kind = name.removeprefix(prefix), prefix_kind
                break
        if kind not in ("table", "vector"):
            raise ValueError(f"Sink plugin '{name}' has an unsupported kind.")
        register_sink(name, factory, kind=cast(SinkKind, kind))

    found = {g: discover(g) for g in GROUPS}
    registrars: dict[str, Callable[[str, Any], None]] = {
        "meldr.connectors": register_connector,
        "meldr.parsers": register_parser,
        "meldr.providers": register_provider,
        "meldr.sinks": register_sink_plugin,
    }
    for group, register in registrars.items():
        for name, obj in found[group].items():
            register(name, obj)
    if ops_registry is not None:
        for obj in found["meldr.ops"].values():
            ops = obj() if callable(obj) else obj  # an OpDef, a list of them, or a factory
            for op in ops if isinstance(ops, list | tuple) else [ops]:
                ops_registry.register(op)
    return found
