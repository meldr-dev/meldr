"""Connector for Python iterables and callables."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

import polars as pl
import pyarrow as pa

from meldr.core.errors import ConnectorError
from meldr.core.types import Modality, RawBatch

if TYPE_CHECKING:
    from meldr.config import SourceConfig


class PythonConnector:
    """Read batches from a Python callable or iterable."""

    modality = Modality.TABULAR

    def __init__(
        self,
        source_id: str,
        source: Any,
        *,
        batch_size: int = 50_000,
        modality: Modality = Modality.TABULAR,
    ) -> None:
        self.source_id = source_id
        self.source = source
        self.batch_size = max(batch_size, 1)
        self.modality = modality

    def pull(self, **params: Any) -> Iterator[RawBatch]:
        """Call the source and yield normalized batches."""
        value = self.source(**params) if callable(self.source) else self.source
        yield from self._batches(value)

    def _batches(self, value: Any) -> Iterator[RawBatch]:
        if isinstance(value, RawBatch):
            yield value
            return
        if isinstance(value, pl.DataFrame):
            yield from self._frame_batches(value)
            return
        if isinstance(value, pa.Table):
            yield from self._frame_batches(cast(pl.DataFrame, pl.from_arrow(value)))
            return
        if _is_pandas_frame(value):
            try:
                frame = pl.from_pandas(value)
                yield from self._frame_batches(frame)
            except (ImportError, ValueError, TypeError) as e:
                raise ConnectorError(
                    "The python connector could not convert the pandas DataFrame. "
                    "Install pandas and pass a correct DataFrame."
                ) from e
            return
        if isinstance(value, Mapping):
            yield from self._row_batches([value])
            return
        if isinstance(value, (str, bytes, bytearray)):
            raise ConnectorError(
                "The python connector source must yield rows, frames, Arrow tables, or RawBatch."
            )
        if isinstance(value, Iterable):
            yield from self._iterable_batches(iter(value))
            return
        raise ConnectorError(
            "The python connector source must return an iterable, a frame, an Arrow table, "
            "or a RawBatch."
        )

    def _iterable_batches(self, values: Iterator[Any]) -> Iterator[RawBatch]:
        try:
            first = next(values)
        except StopIteration:
            return
        if isinstance(first, Mapping):
            rows = [first]
            for value in values:
                if not isinstance(value, Mapping):
                    raise ConnectorError("The python connector iterable must contain only rows.")
                rows.append(value)
                if len(rows) == self.batch_size:
                    yield from self._row_batches(rows)
                    rows = []
            if rows:
                yield from self._row_batches(rows)
            return
        yield from self._batches(first)
        for value in values:
            yield from self._batches(value)

    def _frame_batches(self, frame: pl.DataFrame) -> Iterator[RawBatch]:
        if frame.height == 0:
            yield RawBatch.from_polars(frame, self.source_id)
            return
        for chunk in frame.iter_slices(n_rows=self.batch_size):
            yield RawBatch.from_polars(chunk, self.source_id)

    def _row_batches(self, rows: list[Mapping[str, Any]]) -> Iterator[RawBatch]:
        if not rows:
            return
        if self.modality == Modality.TABULAR:
            yield from self._frame_batches(pl.DataFrame(rows))
        else:
            yield RawBatch.from_dicts(rows, self.source_id)


def _is_pandas_frame(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "pandas" and hasattr(value, "columns")


def from_config(source_id: str, cfg: SourceConfig, batch_size: int) -> PythonConnector:
    source = cfg.options.get("source")
    if source is None:
        source = cfg.options.get("factory")
    if source is None:
        raise ConnectorError(
            f"sources.{source_id}: the python connector must have "
            "options.source or options.factory."
        )
    try:
        modality = Modality(str(cfg.options.get("modality", Modality.TABULAR.value)))
    except ValueError as e:
        raise ConnectorError(
            f"sources.{source_id}: options.modality must be a correct Meldr modality."
        ) from e
    return PythonConnector(source_id, source, batch_size=batch_size, modality=modality)
