"""Sink construction from config, plus registries for table and vector plugins."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from meldr.config import SinksConfig
from meldr.core.errors import ConfigError
from meldr.indexing.sinks.base import TableSink, VectorSink
from meldr.secrets import interpolate_str

SinkFactory = Callable[[Any, Path], Any]
SinkKind = Literal["table", "vector"]
TABLE_SINKS: dict[str, SinkFactory] = {}
VECTOR_SINKS: dict[str, SinkFactory] = {}


def register_sink(
    type_name: str,
    factory: SinkFactory,
    *,
    kind: SinkKind = "table",
    replace: bool = False,
) -> None:
    """Register a table or vector sink factory.

    The factory gets the matching config section and the storage root. The default kind keeps
    the old two-argument registration form in use for table sink plugins.
    """
    if kind not in ("table", "vector"):
        raise ConfigError("Sink kind must be 'table' or 'vector'.")
    registry = TABLE_SINKS if kind == "table" else VECTOR_SINKS
    if type_name in registry and not replace:
        raise ConfigError(
            f"{kind} sink '{type_name}' is already registered. "
            "Choose a different name or pass replace=True."
        )
    registry[type_name] = factory


def _unknown(kind: SinkKind, name: str, known: Mapping[str, SinkFactory]) -> ConfigError:
    return ConfigError(
        f"Unknown {kind} sink '{name}'. Known: {', '.join(sorted(known)) or 'none'}. "
        "Register one with register_sink()."
    )


def _resolved(options: Mapping[str, Any]) -> dict[str, Any]:
    """Option values with `${env:...}` references resolved, only at construction time."""
    return {k: interpolate_str(v) for k, v in options.items()}


def _duckdb(config: Any, root: Path) -> TableSink:
    from meldr.indexing.sinks.duckdb import DuckDBSink

    return DuckDBSink(config.path or root / "tables" / "meldr.duckdb")


def _parquet(config: Any, root: Path) -> TableSink:
    from meldr.indexing.sinks.parquet import ParquetSink

    return ParquetSink(config.path or root / "tables")


def _sql(config: Any, root: Path) -> TableSink:
    from meldr.indexing.sinks.sql import SQLTableSink

    return SQLTableSink(config.url, schema=config.namespace)


def _delta(config: Any, root: Path) -> TableSink:
    from meldr.indexing.sinks.delta import DeltaSink

    return DeltaSink(config.path or root / "tables", storage_options=_resolved(config.options))


def _iceberg(config: Any, root: Path) -> TableSink:
    from meldr.indexing.sinks.iceberg import IcebergSink

    options = _resolved(config.options)
    catalog = str(options.pop("catalog", "default"))
    return IcebergSink(catalog, namespace=config.namespace or "meldr", properties=options)


def _memory(config: Any, root: Path) -> VectorSink:
    from meldr.indexing.sinks.memory import MemoryVectorSink

    return MemoryVectorSink()


def _lancedb(config: Any, root: Path) -> VectorSink:
    from meldr.indexing.sinks.lancedb import LanceDBSink

    return LanceDBSink(config.path or root / "vectors")


def _qdrant(config: Any, root: Path) -> VectorSink:
    from meldr.indexing.sinks.qdrant import QdrantSink

    return QdrantSink(config.url.reveal() if config.url else None)


def _pgvector(config: Any, root: Path) -> VectorSink:
    from meldr.indexing.sinks.pgvector import PgVectorSink

    return PgVectorSink(config.url.reveal() if config.url else None)


def _chroma(config: Any, root: Path) -> VectorSink:
    from meldr.indexing.sinks.chroma import ChromaSink

    return ChromaSink(config.path or root / "vectors")


def _register_builtins() -> None:
    builtins: tuple[tuple[str, SinkFactory, SinkKind], ...] = (
        ("duckdb", _duckdb, "table"),
        ("parquet", _parquet, "table"),
        ("sql", _sql, "table"),
        ("delta", _delta, "table"),
        ("iceberg", _iceberg, "table"),
        ("memory", _memory, "vector"),
        ("lancedb", _lancedb, "vector"),
        ("qdrant", _qdrant, "vector"),
        ("pgvector", _pgvector, "vector"),
        ("chroma", _chroma, "vector"),
    )
    for name, factory, kind in builtins:
        register_sink(name, factory, kind=kind, replace=True)


def build_table_sink(cfg: SinksConfig, root: Path) -> TableSink:
    config = cfg.tables
    factory = TABLE_SINKS.get(config.type)
    if factory is None:
        raise _unknown("table", config.type, TABLE_SINKS)
    sink: TableSink = factory(config, root)
    return sink


def build_vector_sink(cfg: SinksConfig, root: Path) -> VectorSink:
    config = cfg.vectors
    factory = VECTOR_SINKS.get(config.type)
    if factory is None:
        raise _unknown("vector", config.type, VECTOR_SINKS)
    sink: VectorSink = factory(config, root)
    return sink


_register_builtins()
