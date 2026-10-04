"""Optional dlt bridge connector."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from meldr.connectors.python import PythonConnector
from meldr.core.errors import ConnectorError, MissingExtraError
from meldr.core.types import Modality, RawBatch

if TYPE_CHECKING:
    from meldr.config import SourceConfig


class DltConnector(PythonConnector):
    """Read a dlt source or resource without a dlt pipeline."""

    def __init__(
        self,
        source_id: str,
        source: Any,
        *,
        resource: str | None = None,
        batch_size: int = 50_000,
        modality: Modality = Modality.TABULAR,
    ) -> None:
        try:
            import dlt
        except ImportError as e:
            raise MissingExtraError("dlt", "the dlt connector") from e
        if not hasattr(dlt, "source"):
            raise ConnectorError("The installed dlt package does not include dlt.source.")
        super().__init__(source_id, source, batch_size=batch_size, modality=modality)
        self.resource = resource

    def pull(self, **params: Any) -> Iterator[RawBatch]:
        """Read the selected dlt source and yield normalized batches."""
        source = self.source(**params) if callable(self.source) else self.source
        if self.resource:
            if hasattr(source, "with_resources"):
                source = source.with_resources(self.resource)
            elif hasattr(source, "resources"):
                try:
                    source = source.resources[self.resource]
                except KeyError as e:
                    raise ConnectorError(
                        f"The dlt source has no resource named '{self.resource}'."
                    ) from e
            else:
                raise ConnectorError(
                    "The dlt source does not support resource selection. "
                    "Remove options.resource or pass a DltSource."
                )
        yield from self._batches(source)


def from_config(source_id: str, cfg: SourceConfig, batch_size: int) -> DltConnector:
    source = cfg.options.get("source")
    if source is None:
        source = cfg.options.get("factory")
    if source is None:
        raise ConnectorError(
            f"sources.{source_id}: the dlt connector must have options.source or options.factory."
        )
    try:
        modality = Modality(str(cfg.options.get("modality", Modality.TABULAR.value)))
    except ValueError as e:
        raise ConnectorError(
            f"sources.{source_id}: options.modality must be a correct Meldr modality."
        ) from e
    resource = cfg.options.get("resource")
    if resource is not None and not isinstance(resource, str):
        raise ConnectorError(f"sources.{source_id}: options.resource must be a string.")
    return DltConnector(
        source_id,
        source,
        resource=resource,
        batch_size=batch_size,
        modality=modality,
    )
