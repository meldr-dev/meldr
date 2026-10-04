"""Connector protocol and the connector registry."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from meldr.core.errors import ConfigError
from meldr.core.types import Modality, RawBatch

if TYPE_CHECKING:
    from meldr.config import SourceConfig


@runtime_checkable
class Connector(Protocol):
    """Anything that yields RawBatch chunks."""

    source_id: str
    modality: Modality

    def pull(self, **params: Any) -> Iterator[RawBatch]: ...


ConnectorFactory = Callable[[str, "SourceConfig", int], Connector]
_REGISTRY: dict[str, ConnectorFactory] = {}


def register_connector(name: str, factory: ConnectorFactory) -> None:
    """Register a connector type usable as `sources.<id>.connector: <name>`."""
    _REGISTRY[name] = factory


def build_connector(source_id: str, cfg: SourceConfig, batch_size: int) -> Connector:
    """Instantiate the connector for one configured source."""
    _ensure_builtins()
    factory = _REGISTRY.get(cfg.connector)
    if factory is None:
        raise ConfigError(
            f"sources.{source_id}.connector: unknown connector '{cfg.connector}'. "
            f"Known: {sorted(_REGISTRY)}. Register custom ones with register_connector()."
        )
    return factory(source_id, cfg, batch_size)


def _ensure_builtins() -> None:
    if _REGISTRY.keys() >= {"files", "sql", "rest", "documents", "media", "python", "dlt"}:
        return
    from meldr.connectors import dlt, files, python, rest, sql

    register_connector("files", files.from_config)
    register_connector("documents", files.documents_from_config)
    register_connector("media", files.documents_from_config)
    register_connector("sql", sql.from_config)
    register_connector("rest", rest.from_config)
    register_connector("python", python.from_config)
    register_connector("dlt", dlt.from_config)
