"""File connector: CSV, Parquet, JSON, JSONL, XLSX tables and document/media files."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from datetime import datetime
from fnmatch import fnmatch
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import polars as pl

from meldr._compat import UTC
from meldr._fsspec import StoragePath, storage_path
from meldr.core.errors import ConnectorError, MissingExtraError
from meldr.core.types import Modality, RawBatch, RawRecord, mime_for_path, modality_for_path
from meldr.observability import safe_log

if TYPE_CHECKING:
    from meldr.config import SourceConfig

TABULAR = {".csv", ".tsv", ".parquet", ".pq", ".xlsx"}
JSONS = {".json", ".jsonl", ".ndjson"}


class FileConnector:
    """A file or directory, read in batches.

    Tables come in batches of `batch_size` rows. Documents come in batches of `batch_size` files.
    """

    def __init__(
        self,
        source_id: str,
        path: str | Path,
        *,
        batch_size: int = 50_000,
        format: str | None = None,
        records_mode: bool = False,
        pattern: str = "*",
    ) -> None:
        self.source_id = source_id
        self.path = path
        self.batch_size = batch_size
        self.format = format
        self.records_mode = records_mode
        self.pattern = pattern
        self.modality = Modality.DOCUMENT if records_mode else Modality.TABULAR

    def _files(self, path: StoragePath, pattern: str) -> list[StoragePath]:
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(
                (
                    p
                    for p in path.find()
                    if not p.name.startswith(".")
                    and (
                        pattern == "*"
                        or fnmatch(p.name, pattern)
                        or fnmatch(p.relative_to(path), pattern)
                    )
                ),
                key=lambda p: p.uri,
            )
        raise ConnectorError(f"Cannot find path {path.uri}. Fix sources.{self.source_id}.path.")

    def pull(self, **params: Any) -> Iterator[RawBatch]:
        """Yield batches. Params: `path` (override), `pattern` (glob in directories)."""
        path = storage_path(params.get("path", self.path), feature="the files connector")
        files = self._files(path, str(params.get("pattern", self.pattern)))
        documents: list[StoragePath] = []
        for f in files:
            ext = (self.format or f.suffix).lower()
            ext = ext if ext.startswith(".") else f".{ext}"
            if self.records_mode or ext not in TABULAR | JSONS:
                documents.append(f)
                continue
            yield from self._tabular(f, ext)
        root = path if path.is_dir() else None
        for i in range(0, len(documents), max(self.batch_size, 1)):
            yield self._document_batch(documents[i : i + self.batch_size], root)

    def _meta(self, f: StoragePath) -> dict[str, Any]:
        return {"entity_hint": Path(f.name).stem, "file": f.name}

    def _tabular(self, f: StoragePath, ext: str) -> Iterator[RawBatch]:
        try:
            if ext in (".csv", ".tsv"):
                # Read all columns as strings. This keeps zeros at the start of ZIP codes and IDs.
                # Every batch then has the same types. The profiler infers semantic types instead.
                separator = "\t" if ext == ".tsv" else ","
                if f.is_local:
                    lazy = pl.scan_csv(f._local, separator=separator, infer_schema=False)
                    empty = True
                    for chunk in lazy.collect_batches(chunk_size=self.batch_size):
                        empty = False
                        yield RawBatch.from_polars(chunk, self.source_id, **self._meta(f))
                    if empty:
                        yield RawBatch.from_polars(
                            lazy.head(0).collect(), self.source_id, **self._meta(f)
                        )
                else:
                    with f.open("rb") as handle:
                        yield from self._slices(
                            pl.read_csv(handle, separator=separator, infer_schema=False), f
                        )
            elif ext in (".parquet", ".pq"):
                import pyarrow.parquet as pq

                if f.is_local:
                    pf = pq.ParquetFile(f._local)
                    for rb in pf.iter_batches(batch_size=self.batch_size):
                        frame = cast(pl.DataFrame, pl.from_arrow(rb))
                        yield RawBatch.from_polars(frame, self.source_id, **self._meta(f))
                else:
                    with f.open("rb") as handle:
                        pf = pq.ParquetFile(handle)
                        for rb in pf.iter_batches(batch_size=self.batch_size):
                            frame = cast(pl.DataFrame, pl.from_arrow(rb))
                            yield RawBatch.from_polars(frame, self.source_id, **self._meta(f))
            elif ext == ".xlsx":
                try:
                    import openpyxl  # noqa: F401
                except ImportError as e:
                    raise MissingExtraError("xlsx", "The XLSX reader") from e
                source: Any = f._local if f.is_local else f.open("rb")
                try:
                    sheet = pl.read_excel(source, engine="openpyxl")
                finally:
                    if not f.is_local:
                        source.close()
                yield from self._slices(sheet, f)
            elif ext == ".json":
                data = json.loads(f.read_text())
                yield from self._json_batches(data if isinstance(data, list) else [data], f)
            else:
                with f.open("rt", encoding="utf-8") as lines:
                    yield from self._json_batches((json.loads(x) for x in lines if x.strip()), f)
        except (OSError, json.JSONDecodeError, pl.exceptions.PolarsError) as e:
            raise ConnectorError(f"Cannot read {f.uri}: {e}") from e

    def _slices(self, df: pl.DataFrame, f: StoragePath) -> Iterator[RawBatch]:
        if df.height == 0:
            yield RawBatch.from_polars(df, self.source_id, **self._meta(f))
            return
        for chunk in df.iter_slices(n_rows=self.batch_size):
            yield RawBatch.from_polars(chunk, self.source_id, **self._meta(f))

    def _json_batches(self, rows: Iterable[Any], f: StoragePath) -> Iterator[RawBatch]:
        """Batches of the JSON objects in `rows`, read lazily so JSONL files stream."""
        objs = (r for r in rows if isinstance(r, dict))
        empty = True
        while chunk := list(islice(objs, self.batch_size)):
            empty = False
            yield RawBatch.from_dicts(chunk, self.source_id, **self._meta(f))
        if empty:  # skip, like an empty CSV, so one empty file does not abort a directory
            safe_log("files.no_json_objects", level=logging.WARNING, file=f.name)

    def _document_batch(self, files: list[StoragePath], root: StoragePath | None) -> RawBatch:
        if files and all(f.is_local for f in files):
            local_root = root._local if root is not None and root.is_local else None
            return RawBatch.from_files([f._local for f in files], self.source_id, root=local_root)
        now = datetime.now(UTC)
        records = [
            RawRecord(
                record_id=f.relative_to(root) if root is not None else f.name,
                source_id=self.source_id,
                modality=modality_for_path(f.name),
                uri=f.uri,
                mime_type=mime_for_path(f.name),
                fetched_at=now,
                content_hash=f.digest(),
                metadata={"filename": f.name, "size": f.size()},
            )
            for f in files
        ]
        return RawBatch(
            source_id=self.source_id,
            modality=records[0].modality,
            records=records,
        )


def from_config(source_id: str, cfg: SourceConfig, batch_size: int) -> FileConnector:
    if not cfg.path:
        raise ConnectorError(f"sources.{source_id}: the files connector must have `path`.")
    return FileConnector(
        source_id,
        cfg.path,
        batch_size=batch_size,
        format=cfg.format,
        pattern=str(cfg.options.get("pattern", "*")),
    )


def documents_from_config(source_id: str, cfg: SourceConfig, batch_size: int) -> FileConnector:
    if not cfg.path:
        raise ConnectorError(f"sources.{source_id}: the documents connector must have `path`.")
    return FileConnector(
        source_id,
        cfg.path,
        batch_size=int(cfg.options.get("files_per_batch", 50)),
        records_mode=True,
        pattern=str(cfg.options.get("pattern", "*")),
    )
