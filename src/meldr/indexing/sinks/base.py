"""Sink protocols and shared item types."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import polars as pl
from pydantic import BaseModel

from meldr.core.errors import SinkError

if TYPE_CHECKING:
    from meldr.core.canonical import CanonicalEntity

MetaValue = str | int | float | bool | None

# This pair identifies every harmonized row (see execution/record_id.py).
RECORD_KEY = ("_meldr_source_id", "_meldr_record_id")
# Optional order column: when present, a sink keeps the row with the highest value for each key
# (nulls lowest) and ignores older writes and deletes, so replays do not regress data.
SEQUENCE = "_meldr_sequence"
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
LINEAGE_DOCS = {
    "_meldr_source_id": "Source system the row came from.",
    "_meldr_record_id": (
        "Record ID within a source: UUIDv5 of the primary key, a content hash for keyless rows, "
        "or the document ID for extracted rows."
    ),
    "_meldr_spec_id": "Mapping spec that produced the row.",
    "_meldr_spec_version": "Version of that mapping spec.",
    "_meldr_run_id": "Run that last wrote the row.",
    "_meldr_ingested_at": "When the row was last written (UTC).",
    SEQUENCE: "Source ordering value; the row with the highest value wins.",
}
_MAX_LISTED_VALUES = 30


class WriteResult(BaseModel):
    rows_written: int
    target: str


class IndexItem(BaseModel):
    id: str
    vector: list[float]
    text: str
    metadata: dict[str, MetaValue]


class SearchHit(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


@runtime_checkable
class TableSink(Protocol):
    """A sink that upserts by key, so replays converge and a newer `_meldr_sequence` always wins."""

    def write(self, entity: str, df: pl.DataFrame, *, key: Sequence[str]) -> WriteResult: ...

    def read(self, entity: str) -> pl.DataFrame: ...

    def delete(self, entity: str, keys: pl.DataFrame) -> None:
        """Remove stored rows matching `keys` (key columns, plus `_meldr_sequence` as a guard)."""
        ...


@runtime_checkable
class Describable(Protocol):
    """Optional TableSink capability: store descriptions where SQL users and agents read them."""

    def describe(self, entity: str, *, table: str, columns: Mapping[str, str]) -> None: ...


class TenantTableSink:
    """Add a stable tenant namespace to all table sink names."""

    def __init__(self, sink: TableSink, tenant: str) -> None:
        self.sink = sink
        self.tenant = tenant
        self.namespace = "tenant_" + hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:24]

    def _name(self, entity: str) -> str:
        return safe_identifier(f"{self.namespace}__{entity}")

    def write(self, entity: str, df: pl.DataFrame, *, key: Sequence[str]) -> WriteResult:
        return self.sink.write(self._name(entity), df, key=key)

    def read(self, entity: str) -> pl.DataFrame:
        return self.sink.read(self._name(entity))

    def delete(self, entity: str, keys: pl.DataFrame) -> None:
        self.sink.delete(self._name(entity), keys)

    def describe(self, entity: str, *, table: str, columns: Mapping[str, str]) -> None:
        method = getattr(self.sink, "describe", None)
        if callable(method):
            method(self._name(entity), table=table, columns=columns)

    def close(self) -> None:
        method = getattr(self.sink, "close", None)
        if callable(method):
            method()

    async def aclose(self) -> None:
        method = getattr(self.sink, "aclose", None)
        if callable(method):
            result = method()
            if hasattr(result, "__await__"):
                await result
            return
        self.close()


class TenantVectorSink:
    """Add a stable tenant namespace to all vector collection names."""

    def __init__(self, sink: VectorSink, tenant: str) -> None:
        self.sink = sink
        self.tenant = tenant
        self.namespace = "tenant_" + hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:24]

    def _name(self, collection: str) -> str:
        return safe_identifier(f"{self.namespace}__{collection}")

    def upsert(self, collection: str, items: Sequence[IndexItem]) -> WriteResult:
        return self.sink.upsert(self._name(collection), items)

    def search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        return self.sink.search(self._name(collection), vector, k=k, filters=filters)

    def close(self) -> None:
        method = getattr(self.sink, "close", None)
        if callable(method):
            method()

    async def aclose(self) -> None:
        method = getattr(self.sink, "aclose", None)
        if callable(method):
            result = method()
            if hasattr(result, "__await__"):
                await result
            return
        self.close()


def entity_docs(entity: CanonicalEntity) -> tuple[str, dict[str, str]]:
    """Table and column descriptions for a harmonized table, written like docstrings."""
    columns = dict(LINEAGE_DOCS)
    for f in entity.fields:
        parts = [f.description.strip()] if f.description.strip() else []
        if f.unit:
            parts.append(f"Unit: {f.unit}.")
        if f.vocabulary and len(f.vocabulary) <= _MAX_LISTED_VALUES:
            parts.append("One of: " + ", ".join(f.vocabulary) + ".")
        elif f.vocabulary:
            parts.append(f"One of {len(f.vocabulary)} allowed values.")
        if f.pii:
            parts.append("Personal data.")
        if parts:
            columns[f.name] = " ".join(parts)
    table = f"{entity.description.strip() or entity.name} (Meldr entity {entity.name} "
    table += f"v{entity.version}; one row per source record, lineage in _meldr_* columns.)"
    return table, columns


@runtime_checkable
class VectorSink(Protocol):
    def upsert(self, collection: str, items: Sequence[IndexItem]) -> WriteResult: ...

    def search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]: ...


def keep_latest(df: pl.DataFrame, key: Sequence[str], *, order: Sequence[str] = ()) -> pl.DataFrame:
    """One row for each key, with the highest (SEQUENCE, *order).

    Nulls are lowest. If rows tie, the last row wins.
    """
    by = [c for c in (SEQUENCE, *order) if c in df.columns]
    if by:
        df = df.sort(by, nulls_last=False, maintain_order=True)
    return df.unique(subset=list(key), keep="last", maintain_order=True)


_CURRENT = "_meldr_current_sequence"


def _stored(current: pl.DataFrame, like: pl.DataFrame, key: Sequence[str]) -> pl.DataFrame:
    """The stored (key, sequence) rows, cast to the dtypes of `like` for the join."""
    cols = [pl.col(k).cast(like.schema[k], strict=False) for k in key]
    if SEQUENCE in current.columns and SEQUENCE in like.columns:
        cols.append(pl.col(SEQUENCE).cast(like.schema[SEQUENCE], strict=False).alias(_CURRENT))
    return current.select(cols)


def fresh_rows(incoming: pl.DataFrame, current: pl.DataFrame, key: Sequence[str]) -> pl.DataFrame:
    """Drop rows of `incoming` that are older than the stored row with the same key (nulls lowest).

    `current` holds the stored key columns and, when the table has one, `SEQUENCE`.
    """
    stored = _stored(current, incoming, key)
    if _CURRENT not in stored.columns or stored.height == 0:
        return incoming
    keep = pl.col(_CURRENT).is_null() | (pl.col(SEQUENCE) >= pl.col(_CURRENT))
    return incoming.join(stored, on=list(key), how="left").filter(keep).drop(_CURRENT)


def removable(keys: pl.DataFrame, current: pl.DataFrame, key: Sequence[str]) -> pl.DataFrame:
    """Keys with a stored row that a delete can remove: present, and not newer than the delete."""
    if current.height == 0:
        return keys.select(key).head(0)
    stored = _stored(current, keys, key)
    hits = keys.join(stored, on=list(key), how="inner")
    if _CURRENT in hits.columns:
        hits = hits.filter(pl.col(_CURRENT).is_null() | (pl.col(_CURRENT) <= pl.col(SEQUENCE)))
    return hits.select(key).unique()


def unsuperseded(failed: pl.DataFrame, valid: pl.DataFrame, key: Sequence[str]) -> pl.DataFrame:
    """Failed rows with no valid row of the same key in this batch that is at least as new.

    With `_meldr_sequence`, the function keeps a failure that is newer than every valid version.
    Without it, a valid version in the batch always supersedes the failure.
    """
    if failed.height == 0 or valid.height == 0:
        return failed
    sequenced = SEQUENCE in failed.columns and SEQUENCE in valid.columns
    newest = (
        valid.group_by(key).agg(pl.col(SEQUENCE).max().alias(_CURRENT))
        if sequenced
        else valid.select(key).unique().with_columns(pl.lit(None).alias(_CURRENT))
    )
    present = "_meldr_present"
    joined = failed.join(newest.with_columns(pl.lit(True).alias(present)), on=list(key), how="left")
    keep = pl.col(present).is_null()
    if sequenced:
        newer = pl.col(_CURRENT).is_null() | (pl.col(SEQUENCE) > pl.col(_CURRENT))
        keep = keep | (pl.col(SEQUENCE).is_not_null() & newer)
    return joined.filter(keep).drop(_CURRENT, present)


def key_groups(
    keys: pl.DataFrame, key: Sequence[str]
) -> Iterator[tuple[dict[str, Any], list[Any]]]:
    """Split keys into groups for IN filters.

    Each group is (values of the key columns before the last, values of the last).
    """
    *prefix, last = key
    if not prefix:
        yield {}, keys[last].unique(maintain_order=True).to_list()
        return
    for values, group in keys.group_by(prefix, maintain_order=True):
        yield (
            dict(zip(prefix, values, strict=True)),
            group[last].unique(maintain_order=True).to_list(),
        )


def sink_failure(action: str, e: Exception) -> SinkError:
    """A SinkError with the error type only: driver messages can quote row values."""
    return SinkError(f"{action} failed ({type(e).__name__}). Check that the column types match.")


def safe_identifier(name: str) -> str:
    """Table/collection names: letters, digits, and underscores, and not a digit first."""
    if not _IDENT.match(name):
        raise SinkError(f"'{name}' is not a correct table/collection name. Use only [A-Za-z0-9_].")
    return name


OVERFETCH = 10  # candidates for each wanted hit when some filters run after the vector search


def sql_literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def scalar_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """Equality filters that a store can push down.

    The list values ('one of') stay for `matches` to apply.
    """
    return {k: v for k, v in (filters or {}).items() if not isinstance(v, list | tuple | set)}


def matches(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Apply equality filters, where a list value means 'one of'."""
    if not filters:
        return True
    for k, want in filters.items():
        have = metadata.get(k)
        if isinstance(want, list | tuple | set):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


def dumps_meta(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True, default=str)
