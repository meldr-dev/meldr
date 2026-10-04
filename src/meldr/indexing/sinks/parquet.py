"""Parquet table sink: an append-only log of part files for each entity.

The sink numbers the parts in write order (one writer for each entity at a time). `read()` keeps
the latest row for each key (highest `_meldr_sequence`, then the newest part). It hides a key if
its latest entry is a tombstone. If you write the same batch twice in a row, the sink adds nothing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from meldr._fsspec import StoragePath, storage_path
from meldr.indexing.sinks.base import SEQUENCE, WriteResult, keep_latest, safe_identifier

_VOLATILE = {"_meldr_run_id", "_meldr_ingested_at"}
_DELETED = "_meldr_deleted"
_KEY_FILE = "_key.json"


class ParquetSink:
    """Append-only part files `{root}/{entity}/part-<n>-<hash>.parquet` for upserts and deletes."""

    def __init__(self, root: str | Path) -> None:
        self.root = storage_path(root, feature="the Parquet sink")

    def _dir(self, entity: str) -> StoragePath:
        return self.root.child(safe_identifier(entity))

    def _parts(self, entity: str) -> list[StoragePath]:
        return sorted(self._dir(entity).glob("part-*.parquet"), key=lambda path: path.name)

    def _append(self, entity: str, df: pl.DataFrame, digest: str) -> StoragePath:
        parts = self._parts(entity)
        n = int(parts[-1].name.split("-")[1]) + 1 if parts else 1
        path = self._dir(entity).child(f"part-{n:012d}-{digest}.parquet")
        with path.open("wb") as handle:
            df.write_parquet(handle)
        return path

    def write(self, entity: str, df: pl.DataFrame, *, key: Sequence[str]) -> WriteResult:
        d = self._dir(entity)
        if df.height == 0:
            return WriteResult(rows_written=0, target=f"parquet:{d}")
        d.mkdir()
        d.child(_KEY_FILE).write_text(json.dumps(list(key)))
        stable = df.drop([c for c in df.columns if c in _VOLATILE])
        digest = hashlib.sha256(stable.write_ipc(None).getvalue()).hexdigest()[:24]
        parts = self._parts(entity)
        if parts and parts[-1].name.endswith(f"-{digest}.parquet"):
            return WriteResult(rows_written=df.height, target=f"parquet:{parts[-1]}")
        return WriteResult(
            rows_written=df.height, target=f"parquet:{self._append(entity, df, digest)}"
        )

    def delete(self, entity: str, keys: pl.DataFrame) -> None:
        """Append tombstones for the keys that one or more parts hold (usually none)."""
        parts = self._parts(entity)
        if not keys.height or not parts:
            return
        key = [c for c in keys.columns if c != SEQUENCE]
        stored = pl.concat([self._read_part(p).select(key) for p in parts]).unique()
        hits = keys.join(stored, on=key, how="semi")
        if hits.height:
            self._append(entity, hits.with_columns(pl.lit(True).alias(_DELETED)), "tombstone")

    def read(self, entity: str) -> pl.DataFrame:
        parts = self._parts(entity)
        if not parts:
            return pl.DataFrame()
        key = json.loads(self._dir(entity).child(_KEY_FILE).read_text())
        df = keep_latest(
            pl.concat([self._read_part(p) for p in parts], how="diagonal_relaxed"), key
        )
        if _DELETED in df.columns:
            df = df.filter(pl.col(_DELETED).is_null()).drop(_DELETED)
        return df

    @staticmethod
    def _read_part(path: StoragePath) -> pl.DataFrame:
        with path.open("rb") as handle:
            return pl.read_parquet(handle)
